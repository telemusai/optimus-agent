//! Routes the kernel's content-free measurements through the session's live monitor.

use std::sync::Arc;

use pi_agent_core::performance_metrics as session_metrics;
use serde_json::Value;

use super::shared::{PerformanceMetricEvent, PerformanceMetricRecorder};

pub struct KernelPerformanceMetricAdapter {
    recorder: Arc<dyn session_metrics::PerformanceMetricRecorder>,
}

impl KernelPerformanceMetricAdapter {
    pub fn new(recorder: Arc<dyn session_metrics::PerformanceMetricRecorder>) -> Self {
        Self { recorder }
    }
}

impl PerformanceMetricRecorder for KernelPerformanceMetricAdapter {
    fn session_id(&self) -> &str {
        self.recorder.session_id()
    }

    fn monotonic_now(&self) -> f64 {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.recorder.monotonic_now()))
            .unwrap_or(f64::NAN)
    }

    fn record(&self, event: PerformanceMetricEvent) {
        let Ok(operation) = serde_json::from_value(Value::String(event.operation)) else {
            return;
        };
        let component = event.component.and_then(|component| {
            serde_json::from_value(Value::String(component)).ok()
        });
        let outcome = event.outcome.and_then(|outcome| {
            serde_json::from_value(Value::String(outcome.as_str().to_owned())).ok()
        });
        let measurements = event.measurements.into_iter().filter_map(|(name, value)| {
            let measurement = serde_json::from_value(Value::String(name.to_owned())).ok()?;
            Some((measurement, value.filter(|value| value.is_finite() && *value >= 0.0)))
        }).collect();
        session_metrics::safe_record_performance_metric(Some(&self.recorder), session_metrics::PerformanceMetricEvent {
            operation,
            identity: Some(session_metrics::PerformanceMetricIdentity { component, ..Default::default() }),
            outcome,
            measurements: Some(measurements),
            correlation: None,
            usage: None,
            // The kernel event carries no failure detail or stage today; the
            // fields stay explicit so a future emitter cannot inherit stale
            // struct defaults by accident.
            stage: None,
            error_class: None,
            http_status: None,
            error_message: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::core::kernel::shared::PerformanceMetricOutcome;
    use crate::core::performance_monitor::PerformanceMonitor;

    #[derive(Default)]
    struct Capture(Mutex<Vec<session_metrics::PerformanceMetricEvent>>);

    impl session_metrics::PerformanceMetricRecorder for Capture {
        fn session_id(&self) -> &str { "kernel-test" }
        fn monotonic_now(&self) -> f64 { 42.0 }
        fn next_id(&self, _: session_metrics::PerformanceMetricIdScope) -> String { "unused".into() }
        fn record(&self, event: session_metrics::PerformanceMetricEvent) { self.0.lock().unwrap().push(event); }
        fn flush(&self) {}
        fn close(&self) {}
    }

    fn snapshot() -> PerformanceMetricEvent {
        PerformanceMetricEvent {
            operation: "snapshot".into(), component: Some("snapshot".into()),
            outcome: Some(PerformanceMetricOutcome::Success),
            measurements: vec![
                ("total_ms", Some(40.0)), ("queue_ms", Some(5.0)),
                ("serialization_ms", Some(20.0)), ("serialization_cpu_ms", Some(15.0)),
                ("write_ms", Some(10.0)), ("serialized_bytes", Some(1024.0)),
                ("written_bytes", Some(512.0)), ("next_cell_delay_ms", None),
            ],
        }
    }

    #[test]
    fn kernel_metrics_preserve_snapshot_measurements_without_content_or_token_usage() {
        let capture = Arc::new(Capture::default());
        let adapter = KernelPerformanceMetricAdapter::new(capture.clone());
        assert_eq!(adapter.session_id(), "kernel-test");
        assert_eq!(adapter.monotonic_now(), 42.0);
        adapter.record(snapshot());
        let events = capture.0.lock().unwrap();
        let wire = serde_json::to_value(&events[0]).unwrap();
        assert_eq!(wire["operation"], "snapshot");
        assert_eq!(wire["identity"]["component"], "snapshot");
        assert_eq!(wire["outcome"], "success");
        assert_eq!(wire["measurements"].as_object().unwrap().len(), 8);
        assert_eq!(wire["measurements"]["serialized_bytes"], 1024.0);
        assert!(wire["measurements"]["next_cell_delay_ms"].is_null());
        assert!(wire.get("usage").is_none());
        assert!(wire.get("correlation").is_none());
    }

    #[test]
    fn kernel_metrics_preserve_snapshot_detail_numbers_and_sanitize_unavailable() {
        let keys = [
            "serialization_native_values",
            "serialization_dill_values",
            "serialization_native_ms",
            "serialization_dill_ms",
            "serialization_native_probe_ms",
            "serialization_native_probe_bytes",
            "serialization_native_probe_attempts",
            "serialization_native_probe_rejected",
            "serialization_fragment_prepare_ms",
            "serialization_fragment_write_ms",
            "serialization_fragment_bytes",
            "serialization_fragment_segments",
            "serialization_buffer_reset_ms",
            "serialization_blob_extract_ms",
            "serialization_native_probe_saved_ms",
            "serialization_native_probe_skipped_ms",
            "serialization_envelope_count_ms",
            "serialization_envelope_count_calls",
            "serialization_envelope_write_ms",
            "snapshot_cas_captures",
            "snapshot_legacy_captures",
        ];
        let capture = Arc::new(Capture::default());
        let adapter = KernelPerformanceMetricAdapter::new(capture.clone());
        let mut event = snapshot();
        event.measurements = keys.iter().enumerate().map(|(index, name)|
            (*name, Some(index as f64 + 0.5))).collect();
        event.measurements.push(("private_variable_name", Some(1.0)));
        adapter.record(event);
        let mut event = snapshot();
        event.measurements = keys.iter().enumerate().map(|(index, name)| {
            let value = match index % 4 {
                0 => None, 1 => Some(-1.0), 2 => Some(f64::NAN), _ => Some(f64::INFINITY),
            };
            (*name, value)
        }).collect();
        adapter.record(event);
        let events = capture.0.lock().unwrap();
        assert_eq!(events.len(), 2);
        let wire = serde_json::to_value(&events[0]).unwrap();
        assert_eq!(wire["measurements"].as_object().unwrap().len(), keys.len());
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(wire["measurements"][*key], index as f64 + 0.5);
        }
        assert!(events[1].measurements.as_ref().unwrap().values().all(Option::is_none));
        assert_eq!(events[1].measurements.as_ref().unwrap().len(), keys.len());
        let wire = wire.to_string();
        assert!(!wire.contains("private_variable_name"));
        assert!(!wire.contains("synthetic-payload"));
    }

    #[test]
    fn kernel_metrics_reject_unknown_fields_and_invalid_numbers() {
        let capture = Arc::new(Capture::default());
        let adapter = KernelPerformanceMetricAdapter::new(capture.clone());
        let mut event = snapshot();
        event.measurements = vec![("total_ms", Some(f64::NAN)), ("queue_ms", Some(-1.0)), ("private_variable", Some(1.0))];
        adapter.record(event);
        adapter.record(PerformanceMetricEvent { operation: "private_code".into(), ..Default::default() });
        let events = capture.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        let measurements = events[0].measurements.as_ref().unwrap();
        assert_eq!(measurements.len(), 2);
        assert!(measurements.values().all(Option::is_none));
    }

    #[tokio::test]
    async fn kernel_metrics_obey_live_monitor_on_off_for_an_existing_adapter() {
        let root = tempfile::tempdir().unwrap();
        let monitor = PerformanceMonitor::new(root.path(), false);
        let recorder = Arc::new(monitor.recorder("kernel-live-toggle".into()));
        let adapter = KernelPerformanceMetricAdapter::new(recorder.clone());
        adapter.record(snapshot());
        recorder.flush_pending().await;
        assert!(!monitor.directory().exists());
        monitor.set_enabled(true).unwrap();
        adapter.record(snapshot());
        recorder.flush_pending().await;
        let files: Vec<_> = std::fs::read_dir(monitor.directory()).unwrap().map(|entry| entry.unwrap().path()).collect();
        assert_eq!(files.len(), 1);
        let before = std::fs::read(&files[0]).unwrap();
        let contents = String::from_utf8(before.clone()).unwrap();
        assert!(contents.contains("\"operation\":\"snapshot\""));
        assert!(contents.contains("kernel-live-toggle"));
        monitor.set_enabled(false).unwrap();
        adapter.record(snapshot());
        recorder.flush_pending().await;
        assert_eq!(std::fs::read(&files[0]).unwrap(), before);
    }
}
