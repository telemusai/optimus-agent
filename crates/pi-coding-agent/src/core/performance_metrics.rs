//! Port of packages/coding-agent/src/core/performance-metrics.ts

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_agent_core::performance_metrics::{
    PerformanceMetricComponent, PerformanceMetricCorrelation, PerformanceMetricEvent,
    PerformanceMetricIdScope, PerformanceMetricIdentity, PerformanceMetricMeasurement,
    PerformanceMetricOperation, PerformanceMetricOutcome, PerformanceMetricRecorder,
    PerformanceMetricRecordCorrelation, PerformanceMetricRecordV1, PerformanceMetricUsageV1,
    PERFORMANCE_METRICS_SCHEMA_VERSION,
};
use serde_json::Value;

const DEFAULT_MAX_BUFFERED_RECORDS: usize = 512;
const DEFAULT_MAX_BUFFERED_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_RECORD_BYTES: usize = 8 * 1024;
const DEFAULT_MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_FILES: usize = 4;
const DEFAULT_FLUSH_INTERVAL_MS: usize = 1_000;
const DEFAULT_CLOSE_TIMEOUT_MS: usize = 1_000;

/// `OPERATIONS`.
const OPERATIONS: [PerformanceMetricOperation; 20] = [
    PerformanceMetricOperation::LogicalRequest,
    PerformanceMetricOperation::ProviderAttempt,
    PerformanceMetricOperation::Tool,
    PerformanceMetricOperation::Snapshot,
    PerformanceMetricOperation::Compaction,
    PerformanceMetricOperation::CompactionPrepare,
    PerformanceMetricOperation::CompactionHistory,
    PerformanceMetricOperation::CompactionPrefix,
    PerformanceMetricOperation::CompactionNative,
    PerformanceMetricOperation::CompactionPersist,
    PerformanceMetricOperation::CompactionRestore,
    PerformanceMetricOperation::FileRetry,
    PerformanceMetricOperation::SessionReopen,
    PerformanceMetricOperation::SessionInput,
    PerformanceMetricOperation::UiInput,
    PerformanceMetricOperation::UiInputAck,
    PerformanceMetricOperation::UiRender,
    PerformanceMetricOperation::UiMenuOpen,
    PerformanceMetricOperation::UiSessionOpen,
    PerformanceMetricOperation::Recorder,
];

/// `OUTCOMES`.
const OUTCOMES: [PerformanceMetricOutcome; 5] = [
    PerformanceMetricOutcome::Started,
    PerformanceMetricOutcome::Success,
    PerformanceMetricOutcome::Failure,
    PerformanceMetricOutcome::Cancelled,
    PerformanceMetricOutcome::Unavailable,
];

/// `COMPONENTS`.
const COMPONENTS: [PerformanceMetricComponent; 8] = [
    PerformanceMetricComponent::Agent,
    PerformanceMetricComponent::Provider,
    PerformanceMetricComponent::Tool,
    PerformanceMetricComponent::Snapshot,
    PerformanceMetricComponent::Compaction,
    PerformanceMetricComponent::Persistence,
    PerformanceMetricComponent::Session,
    PerformanceMetricComponent::Recorder,
];

/// `MEASUREMENTS`.
const MEASUREMENTS: [PerformanceMetricMeasurement; 35] = [
    PerformanceMetricMeasurement::TotalMs,
    PerformanceMetricMeasurement::WaitMs,
    PerformanceMetricMeasurement::DispatchToResponseHeadersMs,
    PerformanceMetricMeasurement::TransportOpenAckMs,
    PerformanceMetricMeasurement::DispatchToFirstEventMs,
    PerformanceMetricMeasurement::DispatchToFirstVisibleMs,
    PerformanceMetricMeasurement::DispatchToFirstRawMs,
    PerformanceMetricMeasurement::DispatchToFirstThinkingMs,
    PerformanceMetricMeasurement::DispatchToFirstToolMs,
    PerformanceMetricMeasurement::DispatchToFirstTextMs,
    PerformanceMetricMeasurement::DispatchToNetworkTerminalMs,
    PerformanceMetricMeasurement::LocalDrainMs,
    PerformanceMetricMeasurement::TransportWebsocket,
    PerformanceMetricMeasurement::LocalGatewayWaitMs,
    PerformanceMetricMeasurement::UpstreamWaitMs,
    PerformanceMetricMeasurement::SerializationMs,
    PerformanceMetricMeasurement::SerializationCpuMs,
    PerformanceMetricMeasurement::SerializationMaxVariableMs,
    PerformanceMetricMeasurement::SerializationSlowVariables,
    PerformanceMetricMeasurement::SerializationSavedMs,
    PerformanceMetricMeasurement::SerializationSkippedMs,
    PerformanceMetricMeasurement::WriteMs,
    PerformanceMetricMeasurement::QueueMs,
    PerformanceMetricMeasurement::InputAgentMessage,
    PerformanceMetricMeasurement::NextCellDelayMs,
    PerformanceMetricMeasurement::ReopenMs,
    PerformanceMetricMeasurement::SerializedBytes,
    PerformanceMetricMeasurement::WrittenBytes,
    PerformanceMetricMeasurement::ReadBytes,
    PerformanceMetricMeasurement::RetryCount,
    PerformanceMetricMeasurement::AttemptCount,
    PerformanceMetricMeasurement::AttemptOrdinal,
    PerformanceMetricMeasurement::DroppedCount,
    PerformanceMetricMeasurement::FrameCount,
    PerformanceMetricMeasurement::MaxMs,
];

/// A failed file operation.
///
/// `isErrorCode(error, "ENOENT")` is observable in `rotate()`, so the port keeps
/// Node's error `code` instead of collapsing every failure into a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceMetricIoError {
    pub code: Option<String>,
    pub message: String,
}

impl PerformanceMetricIoError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            code: None,
            message: message.into(),
        }
    }

    pub fn with_code(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: Some(code.to_string()),
            message: message.into(),
        }
    }

    fn from_io(error: std::io::Error) -> Self {
        let code = match error.kind() {
            std::io::ErrorKind::NotFound => Some("ENOENT".to_string()),
            std::io::ErrorKind::AlreadyExists => Some("EEXIST".to_string()),
            _ => None,
        };
        Self {
            code,
            message: error.to_string(),
        }
    }
}

/// `isErrorCode(error, code)`.
fn is_error_code(error: &PerformanceMetricIoError, code: &str) -> bool {
    error.code.as_deref() == Some(code)
}

/// `interface PerformanceMetricFileIO`.
///
/// Every method may fail, so each returns a `Result`; the recorder contains the
/// failure exactly where the TypeScript `try`/`catch` does.
pub trait PerformanceMetricFileIo: Send + Sync {
    fn mkdir(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>>;
    fn append(
        &self,
        path: &str,
        data: &str,
    ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>>;
    /// `size(path): Promise<number | null>` - `Ok(None)` is a missing file.
    fn size(&self, path: &str) -> pi_ai::types::BoxFuture<Result<Option<u64>, PerformanceMetricIoError>>;
    fn rename(
        &self,
        source: &str,
        destination: &str,
    ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>>;
    fn remove(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>>;
}

/// `interface LocalPerformanceMetricRecorderOptions`.
#[derive(Default)]
pub struct LocalPerformanceMetricRecorderOptions {
    pub directory: String,
    pub session_id: String,
    pub max_buffered_records: Option<usize>,
    pub max_buffered_bytes: Option<usize>,
    pub max_record_bytes: Option<usize>,
    pub max_file_bytes: Option<usize>,
    pub max_files: Option<usize>,
    pub flush_interval_ms: Option<u64>,
    /// Best-effort shutdown wait. Telemetry never owns process/session liveness.
    pub close_timeout_ms: Option<u64>,
    pub monotonic_now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub wall_now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub random_id: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub file_io: Option<Arc<dyn PerformanceMetricFileIo>>,
}

/// `interface EnvironmentPerformanceMetricRecorderOptions`.
pub struct EnvironmentPerformanceMetricRecorderOptions {
    pub agent_dir: String,
    pub session_id: String,
    /// `env?: Readonly<Record<string, string | undefined>>`.
    pub env: Option<HashMap<String, String>>,
    pub max_buffered_records: Option<usize>,
    pub max_buffered_bytes: Option<usize>,
    pub max_record_bytes: Option<usize>,
    pub max_file_bytes: Option<usize>,
    pub max_files: Option<usize>,
    pub flush_interval_ms: Option<u64>,
    pub close_timeout_ms: Option<u64>,
    pub monotonic_now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub wall_now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub random_id: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub file_io: Option<Arc<dyn PerformanceMetricFileIo>>,
}

/// `defaultFileIO`.
pub struct DefaultPerformanceMetricFileIo;

impl PerformanceMetricFileIo for DefaultPerformanceMetricFileIo {
    fn mkdir(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        let path = path.to_string();
        Box::pin(async move {
            tokio::fs::create_dir_all(&path)
                .await
                .map_err(PerformanceMetricIoError::from_io)
        })
    }

    fn append(
        &self,
        path: &str,
        data: &str,
    ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        let path = path.to_string();
        let data = data.to_string();
        Box::pin(async move {
            use tokio::io::AsyncWriteExt;
            let mut options = tokio::fs::OpenOptions::new();
            options.create(true).append(true);
            #[cfg(unix)]
            {
                options.mode(0o600);
            }
            let mut file = options
                .open(&path)
                .await
                .map_err(PerformanceMetricIoError::from_io)?;
            file.write_all(data.as_bytes())
                .await
                .map_err(PerformanceMetricIoError::from_io)?;
            // Tokio may still have a blocking write pending after write_all.
            // Complete it before the recorder reports a successful drain.
            file.flush()
                .await
                .map_err(PerformanceMetricIoError::from_io)
        })
    }

    fn size(
        &self,
        path: &str,
    ) -> pi_ai::types::BoxFuture<Result<Option<u64>, PerformanceMetricIoError>> {
        let path = path.to_string();
        Box::pin(async move {
            match tokio::fs::metadata(&path).await {
                Ok(metadata) => Ok(Some(metadata.len())),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(PerformanceMetricIoError::from_io(error)),
            }
        })
    }

    fn rename(
        &self,
        source: &str,
        destination: &str,
    ) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        let source = source.to_string();
        let destination = destination.to_string();
        Box::pin(async move {
            tokio::fs::rename(&source, &destination)
                .await
                .map_err(PerformanceMetricIoError::from_io)
        })
    }

    fn remove(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
        let path = path.to_string();
        Box::pin(async move {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(PerformanceMetricIoError::from_io(error)),
            }
        })
    }
}


/// `isErrorCode(error, code)` - kept next to the predicate name used in `rotate()`.
fn is_enoent(error: &PerformanceMetricIoError) -> bool {
    is_error_code(error, "ENOENT")
}

/// `boundedInteger(value, fallback, minimum, maximum)`.
fn bounded_integer(value: Option<f64>, fallback: usize, minimum: usize, maximum: usize) -> usize {
    let Some(value) = value else {
        return fallback;
    };
    if !value.is_finite() {
        return fallback;
    }
    let clamped = value.floor().min(maximum as f64).max(minimum as f64);
    clamped as usize
}

/// `sanitizeString(value, maxLength)`.
///
/// `value.slice(0, maxLength)` counts UTF-16 code units; the port truncates by
/// characters, the closest stable equivalent.
fn sanitize_string(value: Option<&Value>, max_length: usize) -> Option<String> {
    let text = value?.as_str()?;
    let sanitized: String = text
        .chars()
        .map(|character| {
            let code = character as u32;
            // `/[ -]/g` -> "?"
            if code <= 0x1f || code == 0x7f {
                '?'
            } else {
                character
            }
        })
        .take(max_length)
        .collect();
    if sanitized.is_empty() {
        return None;
    }
    Some(sanitized)
}

/// `sanitizeNullableString(value, maxLength)`.
fn sanitize_nullable_string(value: Option<&Value>, max_length: usize) -> Option<Option<String>> {
    let value = value?;
    if value.is_null() {
        return Some(None);
    }
    Some(sanitize_string(Some(value), max_length))
}

/// `sanitizeMeasurement(value)`.
fn sanitize_measurement(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    let number = value.as_f64()?;
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    Some(number.min(9_007_199_254_740_991.0))
}

/// `sanitizeTokenCount(value)`.
fn sanitize_token_count(value: Option<&Value>) -> Option<f64> {
    sanitize_measurement(value).map(|number| number.floor())
}

/// `sanitizeCorrelation(value)`.
fn sanitize_correlation(value: Option<&Value>) -> PerformanceMetricCorrelation {
    let Some(Value::Object(map)) = value else {
        return PerformanceMetricCorrelation::default();
    };
    PerformanceMetricCorrelation {
        action_id: sanitize_string(map.get("actionId"), 128),
        logical_request_id: sanitize_string(map.get("logicalRequestId"), 128),
        provider_attempt_id: sanitize_string(map.get("providerAttemptId"), 128),
        tool_call_id: sanitize_string(map.get("toolCallId"), 128),
    }
}

/// `sanitizeIdentity(value)`.
fn sanitize_identity(value: Option<&Value>) -> Option<PerformanceMetricIdentity> {
    let Some(Value::Object(map)) = value else {
        return None;
    };
    let component = map.get("component").and_then(Value::as_str).and_then(|name| {
        COMPONENTS
            .iter()
            .find(|component| component.as_str() == name)
            .copied()
    });
    let sanitized = PerformanceMetricIdentity {
        provider: sanitize_nullable_string(map.get("provider"), 96),
        model: sanitize_nullable_string(map.get("model"), 160),
        api: sanitize_nullable_string(map.get("api"), 96),
        component,
    };
    // `Object.values(sanitized).some((item) => item !== undefined)`.
    if sanitized.provider.is_some()
        || sanitized.model.is_some()
        || sanitized.api.is_some()
        || sanitized.component.is_some()
    {
        Some(sanitized)
    } else {
        None
    }
}

/// `sanitizeMeasurements(value)`.
fn sanitize_measurements(
    value: Option<&Value>,
) -> Option<pi_agent_core::performance_metrics::PerformanceMetricMeasurements> {
    let Some(Value::Object(map)) = value else {
        return None;
    };
    let mut sanitized: pi_agent_core::performance_metrics::PerformanceMetricMeasurements =
        indexmap::IndexMap::new();
    for (key, measurement) in map {
        if let Some(known) = MEASUREMENTS.iter().find(|known| known.as_str() == key) {
            sanitized.insert(*known, sanitize_measurement(Some(measurement)));
        }
    }
    if sanitized.is_empty() {
        return None;
    }
    Some(sanitized)
}

/// `sanitizeUsage(value)`.
fn sanitize_usage(value: Option<&Value>) -> Option<PerformanceMetricUsageV1> {
    let Some(Value::Object(map)) = value else {
        return None;
    };
    let source = map.get("source").and_then(Value::as_str)?;
    if source != "provider" && source != "local_estimate" {
        return None;
    }
    let bool_or_null = |item: Option<&Value>| item.and_then(Value::as_bool);
    let mut usage = PerformanceMetricUsageV1 {
        source: source.to_string(),
        input_tokens: sanitize_token_count(map.get("inputTokens")),
        cached_input_tokens: sanitize_token_count(map.get("cachedInputTokens")),
        output_tokens: sanitize_token_count(map.get("outputTokens")),
        reasoning_tokens: sanitize_token_count(map.get("reasoningTokens")),
        total_tokens: sanitize_token_count(map.get("totalTokens")),
        cached_input_included_in_input: bool_or_null(map.get("cachedInputIncludedInInput")),
        reasoning_included_in_output: bool_or_null(map.get("reasoningIncludedInOutput")),
        estimator: None,
    };
    if source == "local_estimate" {
        usage.estimator = Some(
            sanitize_string(map.get("estimator"), 64).unwrap_or_else(|| "unspecified".to_string()),
        );
    }
    Some(usage)
}

/// `optInEnabled(value)`.
fn opt_in_enabled(value: Option<&str>) -> bool {
    match value {
        Some(value) => ["1", "true", "yes", "on"].contains(&value.trim().to_lowercase().as_str()),
        None => false,
    }
}

/// `safeFileSegment(value, fallback)`.
fn safe_file_segment(value: &str, fallback: &str) -> String {
    let mut sanitized = String::new();
    let mut last_was_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            sanitized.push(character);
            last_was_dash = character == '-';
        } else if !last_was_dash {
            sanitized.push('-');
            last_was_dash = true;
        }
    }
    let truncated: String = sanitized.chars().take(48).collect();
    if truncated.is_empty() {
        fallback.to_string()
    } else {
        truncated
    }
}


/// `LocalPerformanceMetricRecorder implements PerformanceMetricRecorder`.
///
/// The TypeScript instance owns its buffers and file sinks. Rust callers cannot
/// observe promise identity, so the port splits the recorder into a thin wrapper
/// (`sessionId` and `logPath` are the observable fields) and an `Arc`-shared inner
/// value so the timer-owned and explicit flushes can share one state without
/// borrowing the public wrapper.
pub struct LocalPerformanceMetricRecorder {
    inner: Arc<LocalPerformanceMetricRecorderInner>,
}

pub struct LocalPerformanceMetricRecorderInner {
    session_id: String,
    log_path: String,
    directory: String,
    max_buffered_records: usize,
    max_buffered_bytes: usize,
    max_record_bytes: usize,
    max_file_bytes: usize,
    max_files: usize,
    close_timeout_ms: u64,
    flush_interval_ms: u64,
    now: Arc<dyn Fn() -> f64 + Send + Sync>,
    wall_now: Arc<dyn Fn() -> f64 + Send + Sync>,
    random_id: Arc<dyn Fn() -> String + Send + Sync>,
    file_io: Arc<dyn PerformanceMetricFileIo>,
    state: Mutex<LocalRecorderState>,
    closed: AtomicBool,
    /// `flushRequested` - at most one follow-up drain is represented by a flag,
    /// so stalled file I/O cannot grow a waiter queue.
    flush_requested: AtomicBool,
    /// `flushInFlight` - only one drain runs at a time.
    flush_lock: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct LocalRecorderState {
    buffered_lines: Vec<String>,
    buffered_bytes: usize,
    pending_dropped_records: f64,
    sequence: i64,
}

impl LocalPerformanceMetricRecorder {
    /// `constructor(options)`.
    pub fn new(options: LocalPerformanceMetricRecorderOptions) -> Self {
        let directory = resolve_path(&options.directory);
        let session_id = sanitize_string(Some(&Value::String(options.session_id.clone())), 128)
            .unwrap_or_else(|| "unavailable".to_string());
        let max_file_bytes = bounded_integer(
            options.max_file_bytes.map(|value| value as f64),
            DEFAULT_MAX_FILE_BYTES,
            4 * 1024,
            64 * 1024 * 1024,
        );
        let max_record_bytes = (max_file_bytes / 2).min(bounded_integer(
            options.max_record_bytes.map(|value| value as f64),
            DEFAULT_MAX_RECORD_BYTES,
            512,
            16 * 1024,
        ));
        let max_buffered_bytes = max_file_bytes.saturating_sub(max_record_bytes).min(bounded_integer(
            options.max_buffered_bytes.map(|value| value as f64),
            DEFAULT_MAX_BUFFERED_BYTES,
            1024,
            4 * 1024 * 1024,
        ));
        let max_buffered_records = bounded_integer(
            options.max_buffered_records.map(|value| value as f64),
            DEFAULT_MAX_BUFFERED_RECORDS,
            1,
            4096,
        );
        let max_files = bounded_integer(
            options.max_files.map(|value| value as f64),
            DEFAULT_MAX_FILES,
            1,
            16,
        );
        let close_timeout_ms = bounded_integer(
            options.close_timeout_ms.map(|value| value as f64),
            DEFAULT_CLOSE_TIMEOUT_MS,
            1,
            10_000,
        ) as u64;
        let now = options
            .monotonic_now
            .clone()
            .unwrap_or_else(|| Arc::new(monotonic_now_ms));
        let wall_now = options.wall_now.clone().unwrap_or_else(|| Arc::new(utc_now_ms));
        let random_id = options
            .random_id
            .clone()
            .unwrap_or_else(|| Arc::new(|| uuid::Uuid::new_v4().to_string()));
        let file_io = options
            .file_io
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPerformanceMetricFileIo));
        let instance_id = safe_file_segment(&random_id(), "instance");
        let session_segment = safe_file_segment(&session_id, "session");
        let log_path = join_path(
            &directory,
            &format!("performance-v1-{session_segment}-{instance_id}.jsonl"),
        );
        let flush_interval_ms = bounded_integer(
            options.flush_interval_ms.map(|value| value as f64),
            DEFAULT_FLUSH_INTERVAL_MS,
            100,
            60_000,
        ) as u64;
        let inner = Arc::new(LocalPerformanceMetricRecorderInner {
            session_id,
            log_path,
            directory,
            max_buffered_records,
            max_buffered_bytes,
            max_record_bytes,
            max_file_bytes,
            max_files,
            close_timeout_ms,
            flush_interval_ms,
            now,
            wall_now,
            random_id,
            file_io,
            state: Mutex::new(LocalRecorderState::default()),
            closed: AtomicBool::new(false),
            flush_requested: AtomicBool::new(false),
            flush_lock: tokio::sync::Mutex::new(()),
        });

        // `setInterval(() => void this.flush(), flushIntervalMs).unref()`: the
        // first tick lands one interval from now and the timer never keeps the
        // process alive. Without a runtime the queue is drained by `flush()`.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let weak = Arc::downgrade(&inner);
            let period = std::time::Duration::from_millis(flush_interval_ms);
            handle.spawn(async move {
                let mut ticker =
                    tokio::time::interval_at(tokio::time::Instant::now() + period, period);
                loop {
                    ticker.tick().await;
                    let Some(inner) = weak.upgrade() else {
                        return;
                    };
                    if inner.closed.load(Ordering::SeqCst) {
                        return;
                    }
                    inner.flush().await;
                }
            });
        }

        Self { inner }
    }

    /// `readonly sessionId`.
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// `readonly logPath`.
    pub fn log_path(&self) -> &str {
        &self.inner.log_path
    }

    /// `flushIntervalMs` - the interval the timer is armed with.
    pub fn flush_interval_ms(&self) -> u64 {
        self.inner.flush_interval_ms
    }

    /// `monotonicNow()`.
    pub fn monotonic_now(&self) -> f64 {
        (self.inner.now)()
    }

    /// `nextId(scope)`.
    pub fn next_id(&self, scope: &str) -> String {
        format!("{scope}-{}", (self.inner.random_id)())
    }

    /// `record(event)`.
    pub fn record(&self, event: PerformanceMetricEvent) {
        self.inner.record(event);
    }

    /// `flush()`.
    pub fn flush(&self) -> pi_ai::types::BoxFuture<()> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move { inner.flush().await })
    }

    /// `close()`.
    ///
    /// A stuck filesystem must not block session disposal; the wait is bounded by
    /// `closeTimeoutMs` exactly like `settleWithin`.
    pub fn close(&self) -> pi_ai::types::BoxFuture<()> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move { inner.close().await })
    }
}

impl LocalPerformanceMetricRecorderInner {
    /// `record(event)`.
    fn record(&self, event: PerformanceMetricEvent) {
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        let line = self.serialize_event(&event);
        let Some(line) = line else {
            self.note_dropped_record(1.0);
            return;
        };
        let bytes = line.len();
        let rejected = {
            let mut state = self.state.lock().unwrap();
            let rejected = bytes > self.max_record_bytes
                || state.buffered_lines.len() >= self.max_buffered_records
                || state.buffered_bytes + bytes > self.max_buffered_bytes;
            if !rejected {
                state.buffered_lines.push(line);
                state.buffered_bytes += bytes;
            }
            rejected
        };
        if rejected {
            self.note_dropped_record(1.0);
        }
    }

    /// `noteDroppedRecord(count = 1)`.
    fn note_dropped_record(&self, count: f64) {
        let mut state = self.state.lock().unwrap();
        state.pending_dropped_records =
            (state.pending_dropped_records + count).min(9_007_199_254_740_991.0);
    }

    /// `private serializeEvent(event)`.
    fn serialize_event(&self, event: &PerformanceMetricEvent) -> Option<String> {
        if !OPERATIONS.contains(&event.operation) {
            return None;
        }
        let recorded_at_ms = (self.wall_now)();
        if !recorded_at_ms.is_finite() {
            return None;
        }
        let correlation_value = event
            .correlation
            .as_ref()
            .map(|value| serde_json::to_value(value).unwrap_or(Value::Null));
        let correlation = sanitize_correlation(correlation_value.as_ref());
        let sequence = {
            let mut state = self.state.lock().unwrap();
            state.sequence += 1;
            state.sequence
        };
        let mut record = PerformanceMetricRecordV1 {
            operation: event.operation,
            schema_version: PERFORMANCE_METRICS_SCHEMA_VERSION,
            sequence,
            recorded_at: to_iso_string(recorded_at_ms),
            correlation: PerformanceMetricRecordCorrelation {
                action_id: correlation.action_id,
                session_id: self.session_id.clone(),
                logical_request_id: correlation.logical_request_id,
                provider_attempt_id: correlation.provider_attempt_id,
                tool_call_id: correlation.tool_call_id,
            },
            identity: None,
            outcome: None,
            measurements: None,
            usage: None,
        };
        let identity_value = event
            .identity
            .as_ref()
            .map(|value| serde_json::to_value(value).unwrap_or(Value::Null));
        if let Some(identity) = sanitize_identity(identity_value.as_ref()) {
            record.identity = Some(identity);
        }
        if let Some(outcome) = event.outcome {
            if OUTCOMES.contains(&outcome) {
                record.outcome = Some(outcome);
            }
        }
        let measurements_value = event.measurements.as_ref().map(|measurements| {
            let mut map: std::collections::BTreeMap<String, Option<f64>> =
                std::collections::BTreeMap::new();
            for (key, value) in measurements {
                map.insert(key.as_str().to_string(), *value);
            }
            serde_json::to_value(map).unwrap_or(Value::Null)
        });
        record.measurements = sanitize_measurements(measurements_value.as_ref());
        let usage_value = event
            .usage
            .as_ref()
            .map(|usage| serde_json::to_value(usage).unwrap_or(Value::Null));
        record.usage = sanitize_usage(usage_value.as_ref());
        let json = serde_json::to_string(&record).ok()?;
        Some(format!("{json}\n"))
    }

    /// `private async flushOnce()`.
    async fn flush_once(&self) {
        let (lines, buffered_record_count, previously_dropped) = {
            let mut state = self.state.lock().unwrap();
            if state.buffered_lines.is_empty() && state.pending_dropped_records == 0.0 {
                return;
            }
            let lines = std::mem::take(&mut state.buffered_lines);
            let buffered_record_count = lines.len();
            let previously_dropped = state.pending_dropped_records;
            state.buffered_bytes = 0;
            state.pending_dropped_records = 0.0;
            (lines, buffered_record_count, previously_dropped)
        };
        let mut lines = lines;
        if previously_dropped > 0.0 {
            let drop_line = self.serialize_event(&PerformanceMetricEvent {
                operation: PerformanceMetricOperation::Recorder,
                correlation: None,
                identity: Some(PerformanceMetricIdentity {
                    provider: None,
                    model: None,
                    api: None,
                    component: Some(PerformanceMetricComponent::Recorder),
                }),
                outcome: Some(PerformanceMetricOutcome::Unavailable),
                measurements: Some(
                    [(
                        PerformanceMetricMeasurement::DroppedCount,
                        Some(previously_dropped),
                    )]
                    .into_iter()
                    .collect(),
                ),
                usage: None,
            });
            // The TypeScript throws when the drop metric cannot be encoded; the
            // enclosing catch then re-accounts every lost record.
            let Some(drop_line) = drop_line else {
                self.note_dropped_record(previously_dropped + buffered_record_count as f64);
                return;
            };
            lines.insert(0, drop_line);
        }
        if self.file_io.mkdir(&self.directory).await.is_err() {
            self.note_dropped_record(previously_dropped + buffered_record_count as f64);
            return;
        }
        if self.write_lines(lines).await.is_err() {
            self.note_dropped_record(previously_dropped + buffered_record_count as f64);
        }
    }

    /// `flush()`.
    ///
    /// The TypeScript coalesces all callers onto one flight plus at most one
    /// follow-up drain, represented by `flushRequested`. Rust callers cannot
    /// observe promise identity, so the port keeps the same two facts: a mutex
    /// serializes the drains and the flag decides the follow-up. A failing sink
    /// therefore cannot spin this loop.
    async fn flush(&self) {
        self.flush_requested.store(true, Ordering::SeqCst);
        let _flight = self.flush_lock.lock().await;
        loop {
            self.flush_requested.store(false, Ordering::SeqCst);
            // `flushOnce` normally accounts for lost records, but no unexpected
            // sink error may escape to agent work.
            self.flush_once().await;
            if !self.flush_requested.load(Ordering::SeqCst) {
                return;
            }
        }
    }

    /// `private async writeLines(lines)`.
    async fn write_lines(&self, lines: Vec<String>) -> Result<(), ()> {
        let mut chunk: Vec<String> = Vec::new();
        let mut chunk_bytes = 0usize;
        for line in lines {
            let bytes = line.len();
            if bytes > self.max_file_bytes {
                self.note_dropped_record(1.0);
                continue;
            }
            if chunk_bytes > 0 && chunk_bytes + bytes > self.max_file_bytes {
                if self.append_chunk(&chunk.join(""), chunk_bytes).await.is_err() {
                    return Err(());
                }
                chunk = Vec::new();
                chunk_bytes = 0;
            }
            chunk.push(line);
            chunk_bytes += bytes;
        }
        if chunk_bytes > 0 {
            if self.append_chunk(&chunk.join(""), chunk_bytes).await.is_err() {
                return Err(());
            }
        }
        Ok(())
    }

    /// `private async appendChunk(data, bytes)`.
    async fn append_chunk(&self, data: &str, bytes: usize) -> Result<(), ()> {
        let current_size = self
            .file_io
            .size(&self.log_path)
            .await
            .ok()
            .flatten()
            .unwrap_or(0) as usize;
        if current_size + bytes > self.max_file_bytes && self.rotate().await.is_err() {
            return Err(());
        }
        self.file_io
            .append(&self.log_path, data)
            .await
            .map_err(|_| ())
    }

    /// `private async rotate()`.
    async fn rotate(&self) -> Result<(), ()> {
        if self.max_files == 1 {
            return self.file_io.remove(&self.log_path).await.map_err(|_| ());
        }
        for index in (1..self.max_files).rev() {
            let source = if index == 1 {
                self.log_path.clone()
            } else {
                format!("{}.{}", self.log_path, index - 1)
            };
            let destination = format!("{}.{}", self.log_path, index);
            if self.file_io.remove(&destination).await.is_err() {
                return Err(());
            }
            if let Err(error) = self.file_io.rename(&source, &destination).await {
                if !is_enoent(&error) {
                    return Err(());
                }
            }
        }
        Ok(())
    }

    /// `close()` bounded by `closeTimeoutMs`.
    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_millis(self.close_timeout_ms);
        let flush = self.flush();
        let _ = tokio::time::timeout_at(deadline, flush).await;
    }
}

impl PerformanceMetricRecorder for LocalPerformanceMetricRecorder {
    fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    fn monotonic_now(&self) -> f64 {
        (self.inner.now)()
    }

    fn next_id(&self, scope: PerformanceMetricIdScope) -> String {
        let scope = match scope {
            PerformanceMetricIdScope::LogicalRequest => "logical_request",
            PerformanceMetricIdScope::ProviderAttempt => "provider_attempt",
        };
        LocalPerformanceMetricRecorder::next_id(self, scope)
    }

    fn record(&self, event: PerformanceMetricEvent) {
        LocalPerformanceMetricRecorder::record(self, event);
    }

    fn flush(&self) {
        // The trait method is synchronous. The TypeScript timer calls `void
        // this.flush()`; the port spawns the same detached drain so no caller
        // blocks, and `close()` is the bounded wait used by disposal paths.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let inner = Arc::clone(&self.inner);
        handle.spawn(async move { inner.flush().await });
    }

    fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let inner = Arc::clone(&self.inner);
        handle.spawn(async move { inner.flush().await });
    }
}

/// `createLocalPerformanceMetricRecorder(options)`.
///
/// The TypeScript returns `undefined` when the constructor throws; the Rust
/// constructor is infallible, so the factory mirrors the optional return by
/// containing a panicking constructor.
pub fn create_local_performance_metric_recorder(
    options: LocalPerformanceMetricRecorderOptions,
) -> Option<LocalPerformanceMetricRecorder> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LocalPerformanceMetricRecorder::new(options)
    })) {
        Ok(recorder) => Some(recorder),
        Err(_) => None,
    }
}

/// `createLocalPerformanceMetricRecorderFromEnvironment(options)`.
pub fn create_local_performance_metric_recorder_from_environment(
    options: EnvironmentPerformanceMetricRecorderOptions,
) -> Option<LocalPerformanceMetricRecorder> {
    let env: HashMap<String, String> = match options.env.clone() {
        Some(env) => env,
        None => std::env::vars().collect(),
    };
    if !opt_in_enabled(env.get("PRIME_AGENT_PERFORMANCE_METRICS").map(String::as_str)) {
        return None;
    }
    let configured_directory = sanitize_string(
        env.get("PRIME_AGENT_PERFORMANCE_METRICS_DIR")
            .map(|value| Value::String(value.clone()))
            .as_ref(),
        2048,
    );
    create_local_performance_metric_recorder(LocalPerformanceMetricRecorderOptions {
        directory: configured_directory
            .unwrap_or_else(|| join_path(&options.agent_dir, "performance-metrics")),
        session_id: options.session_id,
        max_buffered_records: options.max_buffered_records,
        max_buffered_bytes: options.max_buffered_bytes,
        max_record_bytes: options.max_record_bytes,
        max_file_bytes: options.max_file_bytes,
        max_files: options.max_files,
        flush_interval_ms: options.flush_interval_ms,
        close_timeout_ms: options.close_timeout_ms,
        monotonic_now: options.monotonic_now,
        wall_now: options.wall_now,
        random_id: options.random_id,
        file_io: options.file_io,
    })
}

/// `resolve(path)` - `path.resolve` for the recorder directory.
fn resolve_path(path: &str) -> String {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(candidate).to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string())
}

/// `path.join`.
fn join_path(base: &str, leaf: &str) -> String {
    Path::new(base).join(leaf).to_string_lossy().to_string()
}

/// `Date.now()` in milliseconds.
fn utc_now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

/// `globalThis.performance.now()`; `Instant` is the monotonic clock.
fn monotonic_now_ms() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

/// Node's `Date#toISOString()` (millisecond precision, `Z` suffix).
fn to_iso_string(millis: f64) -> String {
    match chrono::DateTime::from_timestamp_millis(millis as i64) {
        Some(value) => value.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        None => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::performance_metrics::PerformanceMetricMeasurements;

    /// `MemoryFileIO` from the reference test file.
    ///
    /// The shared state sits behind `Arc` so each returned future owns its data
    /// and stays `'static`, the requirement the `PerformanceMetricFileIO` trait
    /// signature imposes.
    #[derive(Clone)]
    struct MemoryFileIo {
        files: Arc<Mutex<HashMap<String, String>>>,
        fail_append: Arc<AtomicBool>,
    }

    impl MemoryFileIo {
        fn new() -> Self {
            Self {
                files: Arc::new(Mutex::new(HashMap::new())),
                fail_append: Arc::new(AtomicBool::new(false)),
            }
        }

        fn read(&self, path: &str) -> Option<String> {
            self.files.lock().unwrap().get(path).cloned()
        }
    }

    impl PerformanceMetricFileIo for MemoryFileIo {
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
            let fail_append = Arc::clone(&self.fail_append);
            Box::pin(async move {
                if fail_append.load(Ordering::SeqCst) {
                    return Err(PerformanceMetricIoError::with_code("ENOSPC", "synthetic disk full"));
                }
                let mut files = files.lock().unwrap();
                let entry = files.entry(path).or_default();
                entry.push_str(&data);
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
                let files = files.lock().unwrap();
                Ok(files.get(&path).map(|data| data.len() as u64))
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
                    return Err(PerformanceMetricIoError::with_code("ENOENT", "synthetic missing file"));
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

    fn sample_event(id: &str) -> PerformanceMetricEvent {
        let mut measurements = PerformanceMetricMeasurements::new();
        measurements.insert(PerformanceMetricMeasurement::TotalMs, Some(12.5));
        measurements.insert(PerformanceMetricMeasurement::WaitMs, Some(1.5));
        measurements.insert(PerformanceMetricMeasurement::DispatchToFirstVisibleMs, Some(5.0));
        measurements.insert(PerformanceMetricMeasurement::LocalGatewayWaitMs, None);
        measurements.insert(PerformanceMetricMeasurement::UpstreamWaitMs, None);
        measurements.insert(PerformanceMetricMeasurement::SerializationCpuMs, Some(0.75));
        PerformanceMetricEvent {
            operation: PerformanceMetricOperation::LogicalRequest,
            correlation: Some(PerformanceMetricCorrelation {
                logical_request_id: Some(id.to_string()),
                action_id: None,
                provider_attempt_id: None,
                tool_call_id: None,
            }),
            identity: Some(PerformanceMetricIdentity {
                provider: Some(Some("openai".to_string())),
                model: Some(Some("gpt-test".to_string())),
                api: Some(Some("openai-responses".to_string())),
                component: Some(PerformanceMetricComponent::Agent),
            }),
            outcome: Some(PerformanceMetricOutcome::Success),
            measurements: Some(measurements),
            usage: Some(PerformanceMetricUsageV1 {
                source: "provider".to_string(),
                input_tokens: Some(100.0),
                cached_input_tokens: Some(20.0),
                output_tokens: Some(10.0),
                reasoning_tokens: None,
                total_tokens: Some(130.0),
                cached_input_included_in_input: Some(false),
                reasoning_included_in_output: None,
                estimator: None,
            }),
        }
    }

    fn create_recorder(
        io: Arc<MemoryFileIo>,
        overrides: LocalPerformanceMetricRecorderOptions,
    ) -> LocalPerformanceMetricRecorder {
        let wall = Arc::new(std::sync::atomic::AtomicU64::new(1_800_000_000_000));
        let wall_clock = Arc::clone(&wall);
        LocalPerformanceMetricRecorder::new(LocalPerformanceMetricRecorderOptions {
            directory: "C:/isolated/performance-metrics".to_string(),
            session_id: "session-1".to_string(),
            monotonic_now: Some(Arc::new(|| 1.0)),
            wall_now: Some(Arc::new(move || {
                wall_clock.fetch_add(1, Ordering::SeqCst) as f64
            })),
            random_id: Some(Arc::new(|| "instance-1".to_string())),
            flush_interval_ms: Some(60_000),
            file_io: Some(io),
            ..overrides
        })
    }

    fn parse_records(io: &MemoryFileIo) -> Vec<Value> {
        let files = io.files.lock().unwrap();
        let mut records: Vec<Value> = Vec::new();
        for data in files.values() {
            for line in data.trim().split('\n') {
                if line.is_empty() {
                    continue;
                }
                records.push(serde_json::from_str(line).unwrap());
            }
        }
        records
    }

    #[tokio::test]
    async fn backlog_real_metric_append_is_visible_when_await_completes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("visible.jsonl");
        let io = DefaultPerformanceMetricFileIo;
        let mut expected = String::new();
        for sequence in 0..16 {
            let line = format!("{{\"sequence\":{sequence},\"value\":\"{}\"}}\n", "metrics-λ".repeat(32));
            expected.push_str(&line);
            io.append(path.to_str().unwrap(), &line).await.unwrap();
            // A synchronous reader immediately after await must see every byte;
            // no sleep or subsequent async IO may finish a detached file write.
            assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn transport_metrics_preserve_queue_correlation_and_raw_phase_availability() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = LocalPerformanceMetricRecorder::new(LocalPerformanceMetricRecorderOptions {
            directory: "C:/isolated/transport-metrics".into(), session_id: "transport-metrics".into(),
            file_io: Some(io.clone()), ..Default::default()
        });
        let mut event = sample_event("logical-one");
        event.operation = PerformanceMetricOperation::SessionInput;
        event.correlation.as_mut().unwrap().action_id = Some("action-one".into());
        let measurements = event.measurements.as_mut().unwrap();
        measurements.insert(PerformanceMetricMeasurement::QueueMs, Some(17.0));
        measurements.insert(PerformanceMetricMeasurement::DispatchToFirstRawMs, Some(3.0));
        measurements.insert(PerformanceMetricMeasurement::DispatchToFirstThinkingMs, None);
        measurements.insert(PerformanceMetricMeasurement::DispatchToFirstToolMs, Some(4.0));
        measurements.insert(PerformanceMetricMeasurement::DispatchToFirstTextMs, Some(5.0));
        measurements.insert(PerformanceMetricMeasurement::DispatchToNetworkTerminalMs, Some(7.0));
        measurements.insert(PerformanceMetricMeasurement::LocalDrainMs, Some(2.0));
        let line = recorder.inner.serialize_event(&event).unwrap();
        let record: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(record["operation"], "session_input");
        assert_eq!(record["correlation"]["actionId"], "action-one");
        assert_eq!(record["measurements"]["queue_ms"], 17.0);
        assert_eq!(record["measurements"]["dispatch_to_first_raw_ms"], 3.0);
        assert!(record["measurements"]["dispatch_to_first_thinking_ms"].is_null());
        assert_eq!(record["measurements"]["local_drain_ms"], 2.0);
        assert!(!line.contains("prompt"));
        recorder.close().await;
    }

    #[tokio::test]
    async fn backlog_public_metric_flush_joins_the_in_flight_drain() {
        struct GatedFileIo {
            memory: MemoryFileIo,
            release: Arc<tokio::sync::Semaphore>,
            append_calls: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl PerformanceMetricFileIo for GatedFileIo {
            fn mkdir(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> { self.memory.mkdir(path) }
            fn size(&self, path: &str) -> pi_ai::types::BoxFuture<Result<Option<u64>, PerformanceMetricIoError>> { self.memory.size(path) }
            fn rename(&self, source: &str, destination: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> { self.memory.rename(source, destination) }
            fn remove(&self, path: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> { self.memory.remove(path) }
            fn append(&self, path: &str, data: &str) -> pi_ai::types::BoxFuture<Result<(), PerformanceMetricIoError>> {
                let write = self.memory.append(path, data);
                let release = self.release.clone();
                let calls = self.append_calls.clone();
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    release.acquire().await.unwrap().forget();
                    write.await
                })
            }
        }
        let io = Arc::new(GatedFileIo {
            memory: MemoryFileIo::new(),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
            append_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let recorder = LocalPerformanceMetricRecorder::new(LocalPerformanceMetricRecorderOptions {
            directory: "C:/isolated/performance-metrics".into(), session_id: "flush-join".into(),
            flush_interval_ms: Some(60_000), file_io: Some(io.clone()), ..Default::default()
        });
        recorder.record(sample_event("first"));
        let mut first = recorder.flush();
        assert!(futures::poll!(first.as_mut()).is_pending());
        assert_eq!(io.append_calls.load(Ordering::SeqCst), 1);

        // The first flush removed its records from the buffer but has not written
        // them. A public flush must join that flight, not return on an empty queue.
        let mut joined = recorder.flush();
        assert!(futures::poll!(joined.as_mut()).is_pending());
        recorder.record(sample_event("second"));
        let mut later = recorder.flush();
        assert!(futures::poll!(later.as_mut()).is_pending());
        assert_eq!(io.append_calls.load(Ordering::SeqCst), 1, "no overlapping drain");
        assert!(parse_records(&io.memory).is_empty());
        io.release.add_permits(2);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            futures::join!(first, joined, later);
        }).await.expect("all public flush waiters must settle");
        let records = parse_records(&io.memory);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["correlation"]["logicalRequestId"], "first");
        assert_eq!(records[1]["correlation"]["logicalRequestId"], "second");
        assert_eq!(io.append_calls.load(Ordering::SeqCst), 2);
        recorder.close().await;
    }

    #[tokio::test]
    async fn buffers_records_and_writes_a_versioned_content_free_schema() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = create_recorder(Arc::clone(&io), LocalPerformanceMetricRecorderOptions::default());
        recorder.record(sample_event("request-1"));
        assert!(io.files.lock().unwrap().is_empty());

        recorder.flush().await;
        let records = parse_records(&io);
        assert_eq!(records[0]["schemaVersion"], serde_json::json!(1));
        assert_eq!(records[0]["sequence"], serde_json::json!(1));
        assert_eq!(records[0]["operation"], serde_json::json!("logical_request"));
        assert_eq!(records[0]["correlation"]["sessionId"], serde_json::json!("session-1"));
        assert_eq!(records[0]["correlation"]["logicalRequestId"], serde_json::json!("request-1"));
        assert_eq!(records[0]["usage"]["cachedInputIncludedInInput"], serde_json::json!(false));
        assert_eq!(records[0]["measurements"]["serialization_cpu_ms"], serde_json::json!(0.75));
        recorder.close().await;
    }

    #[tokio::test]
    async fn drops_unknown_measurements_and_sanitizes_invalid_values() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = create_recorder(Arc::clone(&io), LocalPerformanceMetricRecorderOptions::default());
        let mut event = sample_event("request-privacy");
        let json = serde_json::json!({
            "total_ms": -5,
            "secret_measurement": 7,
        });
        event.measurements = sanitize_measurements(Some(&json));
        recorder.record(event);
        recorder.flush().await;

        let records = parse_records(&io);
        assert_eq!(records[0]["measurements"]["total_ms"], Value::Null);
        let serialized = serde_json::to_string(&records).unwrap();
        assert!(!serialized.contains("secret_measurement"));
        recorder.close().await;
    }

    #[tokio::test]
    async fn reports_records_dropped_by_the_strict_memory_bound() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = create_recorder(
            Arc::clone(&io),
            LocalPerformanceMetricRecorderOptions {
                max_buffered_records: Some(1),
                ..Default::default()
            },
        );
        recorder.record(sample_event("kept"));
        recorder.record(sample_event("dropped-1"));
        recorder.record(sample_event("dropped-2"));
        recorder.flush().await;

        let records = parse_records(&io);
        assert_eq!(records.len(), 2);
        let dropped = records
            .iter()
            .find(|record| record["operation"] == serde_json::json!("recorder"))
            .expect("drop report");
        assert_eq!(dropped["measurements"]["dropped_count"], serde_json::json!(2.0));
        recorder.close().await;
    }

    #[tokio::test]
    async fn contains_disk_full_failures_and_reports_the_lost_batch_after_recovery() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = create_recorder(Arc::clone(&io), LocalPerformanceMetricRecorderOptions::default());
        recorder.record(sample_event("lost-on-disk-full"));
        io.fail_append.store(true, Ordering::SeqCst);
        recorder.flush().await;
        assert!(io.files.lock().unwrap().is_empty());

        io.fail_append.store(false, Ordering::SeqCst);
        recorder.flush().await;
        let records = parse_records(&io);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["operation"], serde_json::json!("recorder"));
        assert_eq!(records[0]["outcome"], serde_json::json!("unavailable"));
        assert_eq!(records[0]["measurements"]["dropped_count"], serde_json::json!(1.0));
        recorder.close().await;
    }

    #[tokio::test]
    async fn rotates_bounded_files_instead_of_growing_one_log_indefinitely() {
        let io = Arc::new(MemoryFileIo::new());
        let recorder = create_recorder(
            Arc::clone(&io),
            LocalPerformanceMetricRecorderOptions {
                max_file_bytes: Some(4096),
                max_record_bytes: Some(1024),
                max_buffered_bytes: Some(3000),
                max_buffered_records: Some(100),
                max_files: Some(2),
                ..Default::default()
            },
        );
        for index in 0..4 {
            recorder.record(sample_event(&format!("first-{index}-{}", "x".repeat(100))));
        }
        recorder.flush().await;
        for index in 0..4 {
            recorder.record(sample_event(&format!("second-{index}-{}", "x".repeat(100))));
        }
        recorder.flush().await;

        let log_path = recorder.log_path().to_string();
        assert!(io.read(&log_path).is_some());
        assert!(io.read(&format!("{log_path}.1")).is_some());
        for data in io.files.lock().unwrap().values() {
            assert!(data.len() <= 4096);
        }
        recorder.close().await;
    }

    #[test]
    fn is_disabled_unless_the_coding_agent_opt_in_is_explicit() {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().join("agent");
        let metrics_dir = temp.path().join("metrics");
        let io: Arc<dyn PerformanceMetricFileIo> = Arc::new(MemoryFileIo::new());
        let disabled = create_local_performance_metric_recorder_from_environment(
            EnvironmentPerformanceMetricRecorderOptions {
                agent_dir: agent_dir.to_string_lossy().to_string(),
                session_id: "session-disabled".to_string(),
                env: Some(HashMap::new()),
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
                file_io: Some(io),
            },
        );
        assert!(disabled.is_none());

        let mut env: HashMap<String, String> = HashMap::new();
        env.insert("PRIME_AGENT_PERFORMANCE_METRICS".to_string(), "true".to_string());
        let enabled = create_local_performance_metric_recorder_from_environment(
            EnvironmentPerformanceMetricRecorderOptions {
                agent_dir: agent_dir.to_string_lossy().to_string(),
                session_id: "session-enabled".to_string(),
                env: Some(env.clone()),
                max_buffered_records: None,
                max_buffered_bytes: None,
                max_record_bytes: None,
                max_file_bytes: None,
                max_files: None,
                flush_interval_ms: Some(60_000),
                close_timeout_ms: None,
                monotonic_now: None,
                wall_now: None,
                random_id: Some(Arc::new(|| "instance-enabled".to_string())),
                file_io: None,
            },
        );
        let enabled = enabled.expect("opt-in recorder");
        // `join(agentDir, "performance-metrics")` when no directory is configured.
        let expected = agent_dir
            .join("performance-metrics")
            .to_string_lossy()
            .to_string();
        assert!(enabled.log_path().starts_with(&expected));

        env.insert(
            "PRIME_AGENT_PERFORMANCE_METRICS_DIR".to_string(),
            metrics_dir.to_string_lossy().to_string(),
        );
        let configured = create_local_performance_metric_recorder_from_environment(
            EnvironmentPerformanceMetricRecorderOptions {
                agent_dir: agent_dir.to_string_lossy().to_string(),
                session_id: "session-configured".to_string(),
                env: Some(env),
                max_buffered_records: None,
                max_buffered_bytes: None,
                max_record_bytes: None,
                max_file_bytes: None,
                max_files: None,
                flush_interval_ms: Some(60_000),
                close_timeout_ms: None,
                monotonic_now: None,
                wall_now: None,
                random_id: None,
                file_io: None,
            },
        )
        .expect("configured recorder");
        assert!(configured.log_path().starts_with(metrics_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn sanitizers_match_the_typescript_bounds_and_fallbacks() {
        assert_eq!(bounded_integer(None, 512, 1, 4096), 512);
        assert_eq!(bounded_integer(Some(0.0), 512, 1, 4096), 1);
        assert_eq!(bounded_integer(Some(9999.0), 512, 1, 4096), 4096);
        assert_eq!(bounded_integer(Some(f64::NAN), 512, 1, 4096), 512);
        assert_eq!(bounded_integer(Some(5.9), 512, 1, 4096), 5);

        assert_eq!(
            sanitize_string(Some(&serde_json::json!("a\u{0}b")), 8),
            Some("a?b".to_string())
        );
        assert_eq!(sanitize_string(Some(&serde_json::json!("")), 8), None);
        assert_eq!(sanitize_string(Some(&serde_json::json!(3)), 8), None);
        assert_eq!(sanitize_nullable_string(Some(&Value::Null), 8), Some(None));
        assert_eq!(sanitize_nullable_string(None, 8), None);
        assert_eq!(sanitize_measurement(Some(&Value::Null)), None);
        assert_eq!(sanitize_measurement(Some(&serde_json::json!(-1))), None);
        assert_eq!(sanitize_measurement(Some(&serde_json::json!(2.5))), Some(2.5));
        assert_eq!(sanitize_token_count(Some(&serde_json::json!(7.9))), Some(7.0));

        assert_eq!(safe_file_segment("a b/c", "fallback"), "a-b-c");
        // `value.replace(/[^A-Za-z0-9_-]/g, "-")` keeps a lone dash, so only an
        // empty segment falls back.
        assert_eq!(safe_file_segment("", "fallback"), "fallback");
        assert_eq!(safe_file_segment("///", "fallback"), "-");
        assert!(opt_in_enabled(Some(" ON ")));
        assert!(!opt_in_enabled(Some("2")));
        assert!(!opt_in_enabled(None));
    }

    #[test]
    fn usage_sanitizer_requires_a_known_source() {
        let provider = serde_json::json!({
            "source": "provider",
            "inputTokens": 100,
            "cachedInputIncludedInInput": false,
        });
        let usage = sanitize_usage(Some(&provider)).expect("provider usage");
        assert_eq!(usage.source, "provider");
        assert_eq!(usage.input_tokens, Some(100.0));
        assert_eq!(usage.estimator, None);

        let estimate = serde_json::json!({ "source": "local_estimate" });
        let usage = sanitize_usage(Some(&estimate)).expect("estimate usage");
        assert_eq!(usage.estimator.as_deref(), Some("unspecified"));

        assert!(sanitize_usage(Some(&serde_json::json!({ "source": "unknown" }))).is_none());
        assert!(sanitize_usage(Some(&serde_json::json!(3))).is_none());
    }
}
