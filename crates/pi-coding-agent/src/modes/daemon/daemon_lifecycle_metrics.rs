//! A15: daemon lifecycle telemetry for the supervisor and its workers.
//!
//! A 15-day /monitor review counted 767 stale-daemon client errors in ONE
//! minute (09-20 03:38, schema-30/schema-29 flap) from an unthrottled launch
//! loop that the performance-metrics sidecar could not see at all, plus one
//! worker OOM and recovery-journal entries unresolved for 19 days. The
//! supervisor therefore reports its own lifecycle into the existing
//! performance-metrics recorder with the same content-free contract:
//! a fixed `daemon_lifecycle` operation, a bounded stage token
//! (`start` / `stop` / `crash` / `stale_detected` / `relaunch`), an outcome,
//! an optional worker correlation id, and - for failures - the bounded A3
//! failure class. No paths, no error text beyond the A3 opt-in, no payloads.
//!
//! Everything here is disposable telemetry: a recorder failure or a disabled
//! opt-in (`PRIME_AGENT_PERFORMANCE_METRICS`, default off) makes every emit a
//! no-op and never changes supervision behavior.

use std::sync::{Arc, Mutex, Weak};

use std::sync::Arc as StdArc;

use pi_agent_core::performance_metrics::{
    safe_record_performance_metric, sanitize_performance_metric_stage, PerformanceMetricComponent,
    PerformanceMetricCorrelation, PerformanceMetricEvent, PerformanceMetricFailure,
    PerformanceMetricOperation, PerformanceMetricOutcome, PerformanceMetricRecorder,
};

use crate::core::performance_metrics::{
    create_local_performance_metric_recorder_from_environment,
    EnvironmentPerformanceMetricRecorderOptions, LocalPerformanceMetricRecorder,
};

use super::daemon_supervisor::MAX_SUPERVISOR_PERFORMANCE_RECORDERS;

/// Stage token: a supervisor or worker started.
pub const DAEMON_LIFECYCLE_STAGE_START: &str = "start";
/// Stage token: a supervisor or worker stopped.
pub const DAEMON_LIFECYCLE_STAGE_STOP: &str = "stop";
/// Stage token: a worker connection was lost unexpectedly.
pub const DAEMON_LIFECYCLE_STAGE_CRASH: &str = "crash";
/// Stage token: a stale worker registration or roster entry was detected.
pub const DAEMON_LIFECYCLE_STAGE_STALE_DETECTED: &str = "stale_detected";
/// Stage token: a worker was relaunched or reconnected by the recovery ladder.
pub const DAEMON_LIFECYCLE_STAGE_RELAUNCH: &str = "relaunch";

/// Live lifecycle recorders retained per process. A supervisor that restarts
/// repeatedly (for example in tests) must not accumulate open telemetry files
/// without bound; the oldest recorder is closed when the bound is exceeded.
fn live_lifecycle_recorders() -> &'static Mutex<Vec<Weak<LocalPerformanceMetricRecorder>>> {
    static RECORDERS: std::sync::OnceLock<Mutex<Vec<Weak<LocalPerformanceMetricRecorder>>>> =
        std::sync::OnceLock::new();
    RECORDERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn track_lifecycle_recorder(recorder: LocalPerformanceMetricRecorder) -> Arc<LocalPerformanceMetricRecorder> {
    let recorder = Arc::new(recorder);
    let mut recorders = live_lifecycle_recorders().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    recorders.retain(|weak| weak.upgrade().is_some());
    while recorders.len() >= MAX_SUPERVISOR_PERFORMANCE_RECORDERS {
        match recorders.first().and_then(Weak::upgrade) {
            Some(oldest) => {
                // Telemetry only: closing the oldest recorder bounds open
                // files; its buffered lines get one best-effort drain on the
                // ambient runtime (or are dropped without one).
                PerformanceMetricRecorder::close(oldest.as_ref());
            }
            None => break,
        }
        recorders.remove(0);
    }
    recorders.push(Arc::downgrade(&recorder));
    recorder
}

/// Emits `daemon_lifecycle` events for one supervisor instance.
#[derive(Clone)]
pub struct DaemonLifecycleEmitter {
    recorder: Option<Arc<LocalPerformanceMetricRecorder>>,
}

impl DaemonLifecycleEmitter {
    /// Creates the emitter for one supervisor run. The recorder is the same
    /// opt-in sidecar the sessions use; when the opt-in is off every emit is a
    /// no-op.
    pub fn new(agent_dir: &str, socket_path: &str, supervisor_generation: &str) -> Self {
        Self::new_with_env(agent_dir, socket_path, supervisor_generation, None)
    }

    /// Same as [`DaemonLifecycleEmitter::new`] with an explicit environment
    /// source, so tests (and embedded hosts) can opt in without mutating the
    /// process environment.
    pub fn new_with_env(
        agent_dir: &str,
        socket_path: &str,
        supervisor_generation: &str,
        env: Option<std::collections::HashMap<String, String>>,
    ) -> Self {
        let _ = socket_path;
        let recorder = create_local_performance_metric_recorder_from_environment(
            EnvironmentPerformanceMetricRecorderOptions {
                agent_dir: agent_dir.to_string(),
                // The session id keeps supervisor files distinct from session
                // files and bounded like every other recorder instance.
                session_id: format!("supervisor-{}", short_generation(supervisor_generation)),
                env,
                max_buffered_records: None,
                max_buffered_bytes: None,
                max_record_bytes: None,
                max_file_bytes: None,
                max_files: None,
                flush_interval_ms: None,
                close_timeout_ms: None,
                monotonic_now: None,
                wall_now: None,
                random_id: None,
                file_io: None,
                error_text_enabled: None,
            },
        )
        .map(track_lifecycle_recorder);
        Self { recorder }
    }

    /// The JSONL path this emitter writes to, when the opt-in is on.
    pub fn log_path(&self) -> Option<String> {
        self.recorder.as_ref().map(|recorder| recorder.log_path().to_string())
    }

    /// A disabled emitter for tests and paths without a supervisor identity.
    pub fn disabled() -> Self {
        Self { recorder: None }
    }

    /// Records one lifecycle event. `worker_id` correlates worker-scoped
    /// stages; supervisor-scoped stages pass `None`.
    pub fn emit(
        &self,
        stage: &str,
        outcome: PerformanceMetricOutcome,
        worker_id: Option<&str>,
        failure: Option<&PerformanceMetricFailure>,
    ) {
        let Some(recorder) = self.recorder.as_ref() else {
            return;
        };
        let Some(stage) = sanitize_performance_metric_stage(Some(stage)) else {
            return;
        };
        let event = PerformanceMetricEvent {
            operation: PerformanceMetricOperation::DaemonLifecycle,
            correlation: Some(PerformanceMetricCorrelation {
                // The worker id is an opaque identifier like a tool-call id.
                action_id: worker_id.map(str::to_string),
                logical_request_id: None,
                provider_attempt_id: None,
                tool_call_id: None,
            }),
            identity: Some(pi_agent_core::performance_metrics::PerformanceMetricIdentity {
                provider: None,
                model: None,
                api: None,
                component: Some(PerformanceMetricComponent::Recorder),
                tool: None,
            }),
            outcome: Some(outcome),
            measurements: None,
            usage: None,
            stage: Some(stage),
            error_class: failure.map(|failure| failure.class),
            http_status: failure.and_then(|failure| failure.http_status),
            error_message: failure.and_then(|failure| failure.message.clone()),
        };
        // Telemetry is disposable and must never change supervision behavior.
        let recorder: StdArc<dyn PerformanceMetricRecorder> = recorder.clone();
        safe_record_performance_metric(Some(&recorder), event);
    }

    /// Classifies a supervision error string with the shared A3 heuristics.
    pub fn classify_failure(message: &str) -> PerformanceMetricFailure {
        PerformanceMetricFailure::classify_message(message)
    }

    /// Best-effort bounded close at supervisor shutdown: flushes buffered
    /// lifecycle records within the recorder's close timeout.
    pub async fn close(&self) {
        if let Some(recorder) = self.recorder.as_ref() {
            recorder.close().await;
        }
    }
}

/// Keeps the session id filename-safe and short; a full uuid is not needed to
/// distinguish supervisor generations within one agent dir.
fn short_generation(generation: &str) -> String {
    let cleaned: String = generation
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || *character == '-')
        .take(12)
        .collect();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_tokens_are_the_documented_vocabulary() {
        for stage in [
            DAEMON_LIFECYCLE_STAGE_START,
            DAEMON_LIFECYCLE_STAGE_STOP,
            DAEMON_LIFECYCLE_STAGE_CRASH,
            DAEMON_LIFECYCLE_STAGE_STALE_DETECTED,
            DAEMON_LIFECYCLE_STAGE_RELAUNCH,
        ] {
            assert_eq!(
                sanitize_performance_metric_stage(Some(stage)).as_deref(),
                Some(stage),
                "every documented stage must satisfy the bounded token contract"
            );
        }
    }

    #[test]
    fn short_generation_is_filename_safe() {
        assert_eq!(short_generation("1b67f4a2-9c3d-4e5f-8a90-1234567890ab"), "1b67f4a2-9c3");
        assert_eq!(short_generation(""), "unknown");
        assert_eq!(short_generation("///"), "unknown");
    }

    #[tokio::test]
    async fn disabled_emitter_is_a_no_op() {
        let emitter = DaemonLifecycleEmitter::disabled();
        emitter.emit(
            DAEMON_LIFECYCLE_STAGE_START,
            PerformanceMetricOutcome::Success,
            None,
            None,
        );
        emitter.close().await;
    }

    #[tokio::test]
    async fn opt_in_emitter_records_lifecycle_events() {
        let root = std::env::temp_dir().join(format!(
            "daemon-lifecycle-metrics-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("PRIME_AGENT_PERFORMANCE_METRICS".to_string(), "1".to_string());
        env.insert(
            "PRIME_AGENT_PERFORMANCE_METRICS_DIR".to_string(),
            root.to_string_lossy().to_string(),
        );
        let emitter = DaemonLifecycleEmitter::new_with_env(
            &root.to_string_lossy(),
            r"\\.\pipe\test-socket",
            "1b67f4a2-9c3d-4e5f-8a90-1234567890ab",
            Some(env),
        );
        let log_path = emitter.log_path().expect("opt-in recorder");
        emitter.emit(DAEMON_LIFECYCLE_STAGE_START, PerformanceMetricOutcome::Success, None, None);
        emitter.emit(
            DAEMON_LIFECYCLE_STAGE_CRASH,
            PerformanceMetricOutcome::Failure,
            Some("worker-7"),
            Some(&DaemonLifecycleEmitter::classify_failure("connection reset by peer")),
        );
        emitter.emit(
            DAEMON_LIFECYCLE_STAGE_RELAUNCH,
            PerformanceMetricOutcome::Success,
            Some("worker-7"),
            None,
        );
        emitter.close().await;
        let content = std::fs::read_to_string(&log_path).expect("records written");
        let records: Vec<serde_json::Value> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("valid JSON record"))
            .collect();
        assert_eq!(records.len(), 3);
        for record in &records {
            assert_eq!(record["operation"], serde_json::json!("daemon_lifecycle"));
            assert!(record["correlation"]["sessionId"]
                .as_str()
                .unwrap()
                .starts_with("supervisor-"));
        }
        assert_eq!(records[0]["stage"], serde_json::json!("start"));
        assert_eq!(records[0]["outcome"], serde_json::json!("success"));
        assert_eq!(records[1]["stage"], serde_json::json!("crash"));
        assert_eq!(records[1]["outcome"], serde_json::json!("failure"));
        assert_eq!(records[1]["error_class"], serde_json::json!("network"));
        assert_eq!(records[1]["correlation"]["actionId"], serde_json::json!("worker-7"));
        assert_eq!(records[2]["stage"], serde_json::json!("relaunch"));
        assert_eq!(records[2]["correlation"]["actionId"], serde_json::json!("worker-7"));
        // Error text stays opt-in and off by default.
        assert!(records[1].get("error_message").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
