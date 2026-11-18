//! Live, profile-scoped control of the bounded local performance recorder.

pub mod throughput;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pi_agent_core::performance_metrics::{
    PerformanceMetricEvent, PerformanceMetricIdScope, PerformanceMetricRecorder,
};
use serde::{Deserialize, Serialize};

use super::performance_metrics::{
    DefaultPerformanceMetricFileIo, LocalPerformanceMetricRecorder,
    LocalPerformanceMetricRecorderOptions, PerformanceMetricFileIo, PerformanceMetricIoError,
};

#[derive(Clone)]
pub struct PerformanceMonitor {
    agent_dir: PathBuf,
    environment_enabled: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preference {
    enabled: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MonitorCommand {
    Select,
    Status,
    Set(bool),
}

pub fn parse_monitor_command(args: &str) -> Result<MonitorCommand, String> {
    match args.trim().to_ascii_lowercase().as_str() {
        "" => Ok(MonitorCommand::Select),
        "status" => Ok(MonitorCommand::Status),
        "on" => Ok(MonitorCommand::Set(true)),
        "off" => Ok(MonitorCommand::Set(false)),
        _ => Err("Usage: /monitor [status|on|off]".into()),
    }
}

impl PerformanceMonitor {
    pub fn from_environment(agent_dir: impl Into<PathBuf>) -> Self {
        let enabled = std::env::var("PRIME_AGENT_PERFORMANCE_METRICS")
            .ok()
            .is_some_and(|value| {
                matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
            });
        Self::new(agent_dir, enabled)
    }

    pub fn new(agent_dir: impl Into<PathBuf>, environment_enabled: bool) -> Self {
        Self { agent_dir: agent_dir.into(), environment_enabled }
    }

    pub fn preference_path(&self) -> PathBuf {
        self.agent_dir.join("performance-monitor.json")
    }

    pub fn directory(&self) -> PathBuf {
        self.agent_dir.join("performance-metrics")
    }

    pub fn enabled(&self) -> Result<bool, String> {
        let file = match std::fs::File::open(self.preference_path()) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(self.environment_enabled);
            }
            Err(error) => return Err(format!("Cannot read monitoring preference: {error}")),
        };
        let mut contents = Vec::new();
        file.take(1025).read_to_end(&mut contents)
            .map_err(|error| format!("Cannot read monitoring preference: {error}"))?;
        if contents.len() > 1024 {
            return Err("Monitoring preference is too large; monitoring is OFF.".into());
        }
        serde_json::from_slice::<Preference>(&contents)
            .map(|preference| preference.enabled)
            .map_err(|_| "Invalid monitoring preference; monitoring is OFF.".into())
    }

    pub fn set_enabled(&self, enabled: bool) -> Result<(), String> {
        std::fs::create_dir_all(&self.agent_dir)
            .map_err(|error| format!("Cannot save monitoring preference: {error}"))?;
        let mut temporary = tempfile::NamedTempFile::new_in(&self.agent_dir)
            .map_err(|error| format!("Cannot save monitoring preference: {error}"))?;
        serde_json::to_writer(&mut temporary, &Preference { enabled })
            .map_err(|error| format!("Cannot save monitoring preference: {error}"))?;
        temporary.write_all(b"\n").and_then(|_| temporary.as_file().sync_all())
            .map_err(|error| format!("Cannot save monitoring preference: {error}"))?;
        temporary.persist(self.preference_path())
            .map_err(|error| format!("Cannot save monitoring preference: {error}"))?;
        if self.enabled()? != enabled {
            return Err("Monitoring preference changed concurrently; run /monitor status.".into());
        }
        Ok(())
    }

    pub fn status_text(&self) -> Result<String, String> {
        Ok(format!(
            "Performance monitoring: {}. Local timing and token metrics. Saved for this profile; applies to existing and future sessions.\nLogs: {}\nCommands: /monitor on, /monitor off, /monitor status",
            if self.enabled()? { "ON" } else { "OFF" },
            self.directory().display(),
        ))
    }

    pub fn recorder(&self, session_id: String) -> LivePerformanceMetricRecorder {
        LivePerformanceMetricRecorder {
            throughput: throughput::register(&session_id),
            monitor: self.clone(), session_id, started: Instant::now(),
            recorder: Mutex::new(None),
        }
    }
}

pub struct LivePerformanceMetricRecorder {
    monitor: PerformanceMonitor,
    session_id: String,
    started: Instant,
    recorder: Mutex<Option<LocalPerformanceMetricRecorder>>,
    throughput: Arc<Mutex<throughput::Counter>>,
}

impl LivePerformanceMetricRecorder {
    pub async fn flush_pending(&self) {
        let flush = self.recorder.lock().unwrap_or_else(|e| e.into_inner())
            .as_ref().map(LocalPerformanceMetricRecorder::flush);
        if let Some(flush) = flush {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(1), flush).await;
        }
    }
}

impl PerformanceMetricRecorder for LivePerformanceMetricRecorder {
    fn session_id(&self) -> &str { &self.session_id }
    fn monotonic_now(&self) -> f64 { self.started.elapsed().as_secs_f64() * 1000.0 }
    fn next_id(&self, scope: PerformanceMetricIdScope) -> String {
        format!("{scope:?}-{}", uuid::Uuid::new_v4())
    }
    fn generation_progress(&self, attempt_id: &str, output_tokens: f64, elapsed_ms: f64) {
        self.throughput.lock().unwrap_or_else(|e| e.into_inner()).progress(attempt_id, output_tokens, elapsed_ms);
    }
    fn record(&self, event: PerformanceMetricEvent) {
        self.throughput.lock().unwrap_or_else(|e| e.into_inner()).record(&event);
        if !self.monitor.enabled().unwrap_or(false) { return; }
        let mut recorder = self.recorder.lock().unwrap_or_else(|e| e.into_inner());
        let recorder = recorder.get_or_insert_with(|| LocalPerformanceMetricRecorder::new(
            LocalPerformanceMetricRecorderOptions {
                directory: self.monitor.directory().to_string_lossy().into_owned(),
                session_id: self.session_id.clone(),
                file_io: Some(Arc::new(MonitorFileIo(self.monitor.clone()))),
                ..Default::default()
            },
        ));
        recorder.record(event);
    }
    fn flush(&self) {
        if let Some(recorder) = self.recorder.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            PerformanceMetricRecorder::flush(recorder);
        }
    }
    fn close(&self) {
        if let Some(recorder) = self.recorder.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            PerformanceMetricRecorder::close(recorder);
        }
    }
}

impl Drop for LivePerformanceMetricRecorder {
    fn drop(&mut self) { self.close(); }
}

// Recheck the shared preference when the timer drains previously buffered records.
// An append already in progress can finish after OFF; subsequent appends are suppressed.
struct MonitorFileIo(PerformanceMonitor);
impl PerformanceMetricFileIo for MonitorFileIo {
    fn mkdir(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        if !self.0.enabled().unwrap_or(false) { return Box::pin(async { Ok(()) }); }
        DefaultPerformanceMetricFileIo.mkdir(path)
    }
    fn append(&self, path: &str, data: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        let monitor = self.0.clone();
        let path = path.to_string();
        let data = data.to_string();
        Box::pin(async move {
            if !monitor.enabled().unwrap_or(false) { return Ok(()); }
            DefaultPerformanceMetricFileIo.append(&path, &data).await
        })
    }
    fn size(&self, path: &str) -> pi_ai::types::BoxFuture<Result<Option<u64>, PerformanceMetricIoError>> {
        DefaultPerformanceMetricFileIo.size(path)
    }
    fn rename(&self, source: &str, destination: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        if !self.0.enabled().unwrap_or(false) { return Box::pin(async { Ok(()) }); }
        DefaultPerformanceMetricFileIo.rename(source, destination)
    }
    fn remove(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        if !self.0.enabled().unwrap_or(false) { return Box::pin(async { Ok(()) }); }
        DefaultPerformanceMetricFileIo.remove(path)
    }
}

pub fn attach_performance_monitor(
    agent: &dyn super::agent_session::AgentHandle,
    agent_dir: &Path,
    session_id: String,
) {
    if agent.performance_metrics().is_none() {
        let recorder = PerformanceMonitor::from_environment(agent_dir).recorder(session_id);
        agent.set_performance_metrics(Some(
            pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics::new(Arc::new(recorder)),
        ));
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::performance_metrics::PerformanceMetricOperation;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TimeoutFileIo {
        files: Arc<Mutex<HashMap<String, String>>>,
        append_calls: Arc<AtomicUsize>,
        first_append: Arc<tokio::sync::Semaphore>,
    }

    impl TimeoutFileIo {
        fn new() -> (Arc<Self>, Arc<Mutex<HashMap<String, String>>>, Arc<AtomicUsize>) {
            let io = Arc::new(Self {
                files: Arc::new(Mutex::new(HashMap::new())),
                append_calls: Arc::new(AtomicUsize::new(0)),
                first_append: Arc::new(tokio::sync::Semaphore::new(0)),
            });
            let files = Arc::clone(&io.files);
            let calls = Arc::clone(&io.append_calls);
            (io, files, calls)
        }
    }

    impl PerformanceMetricFileIo for TimeoutFileIo {
        fn mkdir(&self, _path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
            Box::pin(async { Ok(()) })
        }

        fn append(
            &self,
            path: &str,
            data: &str,
        ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
            let path = path.to_string();
            let data = data.to_string();
            let files = Arc::clone(&self.files);
            let calls = Arc::clone(&self.append_calls);
            let first_append = Arc::clone(&self.first_append);
            Box::pin(async move {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    first_append.acquire().await.unwrap().forget();
                }
                files.lock().unwrap().entry(path).or_default().push_str(&data);
                Ok(())
            })
        }

        fn size(
            &self,
            path: &str,
        ) -> pi_ai::types::BoxFuture<Result<Option<u64>, PerformanceMetricIoError>> {
            let path = path.to_string();
            let files = Arc::clone(&self.files);
            Box::pin(async move {
                Ok(files.lock().unwrap().get(&path).map(|data| data.len() as u64))
            })
        }

        fn rename(
            &self,
            source: &str,
            destination: &str,
        ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
            let source = source.to_string();
            let destination = destination.to_string();
            let files = Arc::clone(&self.files);
            Box::pin(async move {
                let mut files = files.lock().unwrap();
                let Some(data) = files.remove(&source) else {
                    return Err(PerformanceMetricIoError::with_code("ENOENT", "missing"));
                };
                files.insert(destination, data);
                Ok(())
            })
        }

        fn remove(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
            let path = path.to_string();
            let files = Arc::clone(&self.files);
            Box::pin(async move {
                files.lock().unwrap().remove(&path);
                Ok(())
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn flush_pending_timeout_accounts_for_the_cancelled_batch() {
        let temp = tempfile::tempdir().unwrap();
        let live = PerformanceMonitor::new(temp.path(), true).recorder("flush-timeout".into());
        let (io, files, append_calls) = TimeoutFileIo::new();
        *live.recorder.lock().unwrap() = Some(LocalPerformanceMetricRecorder::new(
            LocalPerformanceMetricRecorderOptions {
                directory: temp.path().to_string_lossy().into_owned(),
                session_id: "flush-timeout".into(),
                flush_interval_ms: Some(60_000),
                file_io: Some(io),
                ..Default::default()
            },
        ));
        PerformanceMetricRecorder::record(
            &live,
            PerformanceMetricEvent::new(PerformanceMetricOperation::UiTick),
        );

        live.flush_pending().await;
        assert_eq!(append_calls.load(Ordering::SeqCst), 1);
        assert!(files.lock().unwrap().is_empty());

        let flush = live
            .recorder
            .lock()
            .unwrap()
            .as_ref()
            .map(LocalPerformanceMetricRecorder::flush)
            .expect("injected recorder");
        flush.await;
        let records: Vec<serde_json::Value> = files
            .lock()
            .unwrap()
            .values()
            .flat_map(|data| data.lines())
            .map(|line| serde_json::from_str(line).expect("valid JSONL"))
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["operation"], "recorder");
        assert_eq!(records[0]["measurements"]["dropped_count"], 1.0);

        live.flush_pending().await;
        let line_count: usize = files
            .lock()
            .unwrap()
            .values()
            .map(|data| data.lines().count())
            .sum();
        assert_eq!(line_count, 1, "retry does not duplicate records or drop reports");
    }
}
