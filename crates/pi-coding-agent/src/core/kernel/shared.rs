//! Port of packages/coding-agent/src/core/kernel/shared.ts.
//!
//! Local stand-ins for slices that have not landed yet are marked `TODO(slice)`:
//! they are minimal copies of the TypeScript they replace and are listed in the
//! slice status file under `blocked_on`.
#[path = "execution_report.rs"]
mod execution_report;
pub use execution_report::{parse_execution_reports, ScriptExecutionReport};

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::{oneshot, Notify};

use super::state_snapshot::{
    cas_snapshot_root_for_legacy_path, KernelRestoreSource, KernelSnapshotFormat, RestoreResult,
    SnapshotLegacyExportResult, SnapshotResult,
};

pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 65536;
pub const HOST_REQUEST_SHUTDOWN_TIMEOUT_MS: u64 = 5000;
pub const KERNEL_SHUTDOWN_TIMEOUT_MS: u64 = 5000;
pub const DEFAULT_SNAPSHOT_DEBOUNCE_MS: u64 = 1500;
pub const SNAPSHOT_EXECUTION_TIMEOUT_MS: u64 = 60_000;
pub const KERNEL_ABORT_GRACE_MS: u64 = 1000;
pub const KERNEL_BUSY_REUSE_WAIT_MS: u64 = 15_000;
pub const KERNEL_BUSY_INTERRUPT_INTERVAL_MS: u64 = 500;
pub const MAX_LATE_SENT_AGENT_MESSAGE_HANDLERS: usize = 256;
pub const KERNEL_BUSY_AFTER_INTERRUPT_MESSAGE: &str =
    "The Python kernel is still busy after Prime requested an interrupt. Wait to preserve state, or explicitly kill and restart the kernel.";

/// Boxed future used by the [`KernelClient`] trait so it stays object-safe without
/// pulling `async-trait` into this crate.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Local stand-in for the plain JS `Error` the TypeScript throws and stringifies.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KernelError {
    #[error("{}", KERNEL_BUSY_AFTER_INTERRUPT_MESSAGE)]
    BusyAfterInterrupt,
    #[error("Kernel startup aborted")]
    StartupAborted,
    #[error("{0}")]
    Message(String),
}

impl KernelError {
    pub fn new(message: impl Into<String>) -> Self {
        KernelError::Message(message.into())
    }
}

/// `KernelBusyAfterInterruptError`.
pub fn kernel_busy_after_interrupt_error() -> KernelError {
    KernelError::BusyAfterInterrupt
}

/// `errorMessage(error)`.
pub fn error_message(error: &KernelError) -> String {
    error.to_string()
}

/// `isRecord(value)`.
pub fn is_record(value: &Value) -> bool {
    value.is_object()
}

/// `createKernelStartupAbortError()`.
pub fn create_kernel_startup_abort_error() -> KernelError {
    KernelError::StartupAborted
}

// ---------------------------------------------------------------------------
// AbortSignal
// ---------------------------------------------------------------------------

struct AbortListenerEntry {
    id: u64,
    callback: Box<dyn Fn() + Send + Sync>,
}

struct AbortSignalInner {
    aborted: AtomicBool,
    reason: Mutex<Option<KernelError>>,
    listeners: Mutex<Vec<AbortListenerEntry>>,
    notify: Notify,
}

/// Equivalent of `AbortSignal`: abort flag, abort reason, `{ once: true }`
/// listeners and an awaitable abort event.
#[derive(Clone)]
pub struct AbortSignal {
    inner: Arc<AbortSignalInner>,
}

impl Default for AbortSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AbortSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AbortSignal({})", self.is_aborted())
    }
}

impl AbortSignal {
    pub fn new() -> Self {
        AbortSignal {
            inner: Arc::new(AbortSignalInner {
                aborted: AtomicBool::new(false),
                reason: Mutex::new(None),
                listeners: Mutex::new(Vec::new()),
                notify: Notify::new(),
            }),
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.inner.aborted.load(Ordering::SeqCst)
    }

    pub fn reason(&self) -> Option<KernelError> {
        self.inner.reason.lock().unwrap().clone()
    }

    /// `controller.abort()` / `controller.abort(reason)`.
    pub fn abort(&self, reason: Option<KernelError>) {
        if self.inner.aborted.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(reason) = reason {
            *self.inner.reason.lock().unwrap() = Some(reason);
        }
        let listeners: Vec<AbortListenerEntry> = {
            let mut listeners = self.inner.listeners.lock().unwrap();
            std::mem::take(&mut *listeners)
        };
        for listener in listeners {
            (listener.callback)();
        }
        self.inner.notify.notify_waiters();
    }

    /// `signal.addEventListener("abort", fn, { once: true })`.
    pub fn add_listener(&self, callback: impl Fn() + Send + Sync + 'static) -> AbortListener {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        self.inner.listeners.lock().unwrap().push(AbortListenerEntry {
            id,
            callback: Box::new(callback),
        });
        AbortListener {
            signal: self.clone(),
            id,
        }
    }

    fn remove_listener(&self, id: u64) {
        self.inner.listeners.lock().unwrap().retain(|entry| entry.id != id);
    }

    /// Await the abort event (resolves immediately when already aborted).
    pub async fn wait(&self) {
        loop {
            if self.is_aborted() {
                return;
            }
            let notified = self.inner.notify.notified();
            if self.is_aborted() {
                return;
            }
            notified.await;
        }
    }

    /// `AbortSignal.any([...])`.
    pub fn any(signals: Vec<AbortSignal>) -> AbortSignal {
        let combined = AbortSignal::new();
        for signal in signals {
            if signal.is_aborted() {
                combined.abort(signal.reason());
                continue;
            }
            let target = combined.clone();
            let source = signal.clone();
            signal.add_listener(move || target.abort(source.reason()));
        }
        combined
    }
}

/// Handle for `signal.removeEventListener("abort", fn)`.
pub struct AbortListener {
    signal: AbortSignal,
    id: u64,
}

impl AbortListener {
    pub fn remove(&self) {
        self.signal.remove_listener(self.id);
    }
}

/// `raceStartupWithAbort(promise, signal)`.
pub async fn race_startup_with_abort<F, T>(promise: F, signal: Option<AbortSignal>) -> Result<T, KernelError>
where
    F: Future<Output = Result<T, KernelError>>,
{
    let Some(signal) = signal else {
        return promise.await;
    };
    if signal.is_aborted() {
        return Err(create_kernel_startup_abort_error());
    }
    tokio::select! {
        biased;
        value = promise => value,
        _ = signal.wait() => Err(create_kernel_startup_abort_error()),
    }
}

// ---------------------------------------------------------------------------
// Deferred
// ---------------------------------------------------------------------------

/// `createDeferred<T>()`: the resolver half.
pub struct DeferredResolver<T> {
    sender: Option<oneshot::Sender<Result<T, KernelError>>>,
}

impl<T> DeferredResolver<T> {
    pub fn resolve(&mut self, value: T) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Ok(value));
        }
    }

    pub fn reject(&mut self, error: KernelError) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err(error));
        }
    }

    pub fn is_settled(&self) -> bool {
        self.sender.is_none()
    }
}

/// `createDeferred<T>()`: the promise half.
pub struct Deferred<T> {
    receiver: oneshot::Receiver<Result<T, KernelError>>,
}

impl<T> Deferred<T> {
    pub async fn await_promise(self) -> Result<T, KernelError> {
        match self.receiver.await {
            Ok(result) => result,
            Err(_) => Err(KernelError::new("Deferred was dropped without settling")),
        }
    }

    pub fn receiver(&mut self) -> &mut oneshot::Receiver<Result<T, KernelError>> {
        &mut self.receiver
    }
}

pub fn create_deferred<T>() -> (DeferredResolver<T>, Deferred<T>) {
    let (sender, receiver) = oneshot::channel();
    (DeferredResolver { sender: Some(sender) }, Deferred { receiver })
}

// ---------------------------------------------------------------------------
// Ordered map helpers (the TypeScript uses insertion-ordered `Map`s)
// ---------------------------------------------------------------------------

/// `map.set(key, value)`: keeps the original insertion position for a live key.
pub fn ordered_set<V>(entries: &mut Vec<(String, V)>, key: String, value: V) {
    if let Some(entry) = entries.iter_mut().find(|(existing, _)| *existing == key) {
        entry.1 = value;
        return;
    }
    entries.push((key, value));
}

/// `map.delete(key)`.
pub fn ordered_delete<V>(entries: &mut Vec<(String, V)>, key: &str) -> Option<V> {
    let index = entries.iter().position(|(existing, _)| existing == key)?;
    Some(entries.remove(index).1)
}

/// `map.get(key)`.
pub fn ordered_get<'a, V>(entries: &'a [(String, V)], key: &str) -> Option<&'a V> {
    entries
        .iter()
        .find(|(existing, _)| existing == key)
        .map(|(_, value)| value)
}

// ---------------------------------------------------------------------------
// Host request surface
// ---------------------------------------------------------------------------

/// Handles one typed request from Python code running in the kernel.
/// The returned record is delivered verbatim to the Python caller.
pub type HostRequestHandler =
    Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, KernelError>> + Send + Sync>;

/// Host request handlers keyed by request type (e.g. "rlm.run", "goal.complete").
pub type HostRequestHandlers = HashMap<String, HostRequestHandler>;

/// Where and how to persist the kernel's user namespace so it survives resume.
#[derive(Debug, Clone, Default)]
pub struct KernelSnapshotConfig {
    /// Absolute path for the legacy dill payload.
    pub path: String,
    /// Absolute path for the legacy JSON manifest written alongside the payload.
    pub manifest_path: String,
    /// Session-scoped CAS root. Derived from `path` when omitted.
    pub cas_root_path: Option<String>,
    /// Explicitly initialize CAS v2. Absence defaults fresh roots to legacy.
    pub format: Option<KernelSnapshotFormat>,
    /// Maximum aggregate legacy-equivalent payload size. Default 256 MiB.
    pub max_bytes: Option<u64>,
    /// Maximum serialized size of one variable. Default 16 MiB.
    pub max_variable_bytes: Option<u64>,
    /// Debounce window for the auto-snapshot after a successful execution. Default 1500 ms.
    pub debounce_ms: Option<u64>,
}

/// `KernelManagerOptions`.
#[derive(Clone, Default)]
pub struct KernelManagerOptions {
    /// Python interpreter with the kernel runtime available. Defaults to the auto-bootstrapped kernel.
    pub python: Option<String>,
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub session_id: Option<String>,
    pub host_handlers: Option<HostRequestHandlers>,
    pub python_skills: Option<Vec<KernelPythonSkill>>,
    /// Persist/revive the user namespace across kernel restarts and session resume.
    pub snapshot: Option<KernelSnapshotConfig>,
    /// Disposable content-free performance recorder; never an execution dependency.
    pub performance_metrics: Option<Arc<dyn PerformanceMetricRecorder>>,
    /// Runtime bootstrap re-run on a protocol-repaired kernel so live handles (rlm, bash, skills) exist again.
    pub bootstrap_code: Option<String>,
    /// File receiving the kernel process's stderr, rotated once at each spawn.
    pub stderr_log_path: Option<String>,
    /// Called after the last tracked background handle settles or is torn down.
    pub on_background_work_settled: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// `PythonSkillRuntimeInfo` from ../skills.ts (TODO(slice): needs core::skills).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelPythonSkill {
    pub import_name: String,
    pub package_path: String,
    pub pyproject_path: String,
    pub name: String,
}

/// `KernelBootstrapProgressHandler`.
pub type KernelBootstrapProgressHandler = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Clone, Default)]
pub struct KernelStartOptions {
    pub on_bootstrap_progress: Option<KernelBootstrapProgressHandler>,
    pub signal: Option<AbortSignal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamName {
    Stdout,
    Stderr,
}

impl StreamName {
    pub fn as_str(self) -> &'static str {
        match self {
            StreamName::Stdout => "stdout",
            StreamName::Stderr => "stderr",
        }
    }
}

/// `ExecuteOptions`.
#[derive(Clone, Default)]
pub struct ExecuteOptions {
    /// Aborting interrupts the kernel out-of-band.
    pub signal: Option<AbortSignal>,
    pub on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
    pub on_late_sent_agent_message: Option<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>>,
    /// Cap stdout / stderr / result at this many characters. Default 65536.
    pub max_output_chars: Option<usize>,
    /// Synthetic host cell (snapshot/restore/list); excluded from lastCellCode attribution.
    pub internal: bool,
    /// The protocol repair's own restore; exempt from waiting on the repair it belongs to.
    pub protocol_repair: bool,
}

impl std::fmt::Debug for ExecuteOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecuteOptions")
            .field("max_output_chars", &self.max_output_chars)
            .field("internal", &self.internal)
            .field("protocol_repair", &self.protocol_repair)
            .finish()
    }
}

/// MIME tag the `edit` skill emits diff payloads under.
pub const DIFF_DISPLAY_MIME: &str = "application/vnd.prime-agent.diff+json";

/// MIME tag the `attach-image` skill emits media payloads under.
pub const ATTACHMENT_DISPLAY_MIME: &str = "application/vnd.prime-agent.attachment+json";

/// MIME tag the `agent-message` skill emits after sending a message.
pub const AGENT_MESSAGE_DISPLAY_MIME: &str = "application/vnd.prime-agent.agent-message+json";

/// Internal lifetime notices, consumed before user display rendering.
pub const BASH_ACTIVITY_DISPLAY_MIME: &str = "application/vnd.prime-agent.bash-activity+json";

/// Hard ceiling on a single attachment's base64 payload, a defensive guard
/// against a runaway direct display emit. The `attach-image` skill caps
/// its own images well under this (see `_MAX_IMAGE_BYTES`), so a skill-produced
/// attachment is never dropped here - only a non-skill emit can hit this.
pub const MAX_ATTACHMENT_DATA_CHARS: usize = 10_000_000;

/// One file edit, captured from a [`DIFF_DISPLAY_MIME`] display payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelDiffDisplay {
    pub path: String,
    pub old_str: String,
    pub new_str: String,
    /// 1-based line where `oldStr` begins in the file, for absolute line numbers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<f64>,
}

/// One media attachment, captured from an [`ATTACHMENT_DISPLAY_MIME`] display payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelAttachment {
    pub mime_type: String,
    /// base64-encoded bytes.
    pub data: String,
    /// Source path, surfaced to the TUI renderer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSentAgentMessageTarget {
    pub active_session_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSentAgentMessage {
    pub id: String,
    pub message: String,
    pub delivery_status: KernelDeliveryStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receiver_role: Option<KernelReceiverRole>,
    pub target: KernelSentAgentMessageTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelDeliveryStatus {
    Delivered,
    Queued,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelReceiverRole {
    Parent,
    Sibling,
    Child,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecError {
    pub ename: String,
    pub evalue: String,
    pub traceback: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecuteStatus {
    Ok,
    Error,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecuteResult {
    pub stdout: String,
    pub stderr: String,
    /// Text of the cell's trailing expression value, if the cell produced one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Diffs emitted via display events, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffs: Option<Vec<KernelDiffDisplay>>,
    /// Media attachments emitted via display events, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<KernelAttachment>>,
    /// Agent messages sent from this cell, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sent_agent_messages: Option<Vec<KernelSentAgentMessage>>,
    /// Output that arrived without this cell's id (user threads, other cells' leftovers, raw fd writes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_output: Option<String>,
    pub status: ExecuteStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ExecError>,
    pub duration_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_reports: Option<Vec<ScriptExecutionReport>>,
}

impl ExecuteResult {
    pub fn aborted(duration_ms: f64) -> Self {
        ExecuteResult {
            stdout: String::new(),
            stderr: String::new(),
            result: None,
            diffs: None,
            attachments: None,
            sent_agent_messages: None,
            background_output: None,
            status: ExecuteStatus::Aborted,
            error: None,
            execution_reports: None,
            duration_ms,
        }
    }
}

/// `ExecuteResult` plus the raw fields of the request's `done` event (state ops).
#[derive(Debug, Clone, PartialEq)]
pub struct InternalExecuteResult {
    pub result: ExecuteResult,
    pub done_fields: Option<Map<String, Value>>,
}

impl InternalExecuteResult {
    pub fn aborted(duration_ms: f64) -> Self {
        InternalExecuteResult {
            result: ExecuteResult::aborted(duration_ms),
            done_fields: None,
        }
    }
}

/// Parse a [`DIFF_DISPLAY_MIME`] payload, tolerating malformed input.
pub fn parse_diff_display(payload: &Value) -> Option<KernelDiffDisplay> {
    if !is_record(payload) {
        return None;
    }
    let object = payload.as_object()?;
    let path = object.get("path")?;
    let old_str = object.get("old_str")?;
    let new_str = object.get("new_str")?;
    if !path.is_string() || !old_str.is_string() || !new_str.is_string() {
        return None;
    }
    let start_line = object.get("start_line").and_then(|value| value.as_f64());
    Some(KernelDiffDisplay {
        path: path.as_str()?.to_string(),
        old_str: old_str.as_str()?.to_string(),
        new_str: new_str.as_str()?.to_string(),
        start_line,
    })
}

/// Result of parsing an [`ATTACHMENT_DISPLAY_MIME`] payload.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedAttachment {
    Attachment(KernelAttachment),
    /// A well-formed payload exceeding [`MAX_ATTACHMENT_DATA_CHARS`].
    Oversized,
}

/// Parse an [`ATTACHMENT_DISPLAY_MIME`] payload. Malformed payloads are
/// tolerantly ignored (`None`); a well-formed payload exceeding
/// [`MAX_ATTACHMENT_DATA_CHARS`] is reported as [`ParsedAttachment::Oversized`]
/// so the caller can fail the cell loudly rather than silently dropping the image.
pub fn parse_attachment_display(payload: &Value) -> Option<ParsedAttachment> {
    if !is_record(payload) {
        return None;
    }
    let object = payload.as_object()?;
    let mime_type = object.get("mime_type")?;
    let data = object.get("data")?;
    if !mime_type.is_string() || !data.is_string() {
        return None;
    }
    let data = data.as_str()?;
    // TS `parseAttachmentDisplay` guards with `data.length`, which counts UTF-16
    // code units; encode_utf16().count() matches that for astral-plane input.
    if data.encode_utf16().count() > MAX_ATTACHMENT_DATA_CHARS {
        return Some(ParsedAttachment::Oversized);
    }
    Some(ParsedAttachment::Attachment(KernelAttachment {
        mime_type: mime_type.as_str()?.to_string(),
        data: data.to_string(),
        path: object
            .get("path")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
    }))
}

pub fn parse_sent_agent_message(payload: &Value) -> Option<KernelSentAgentMessage> {
    if !is_record(payload) {
        return None;
    }
    let object = payload.as_object()?;
    let target = object.get("target")?;
    if !is_record(target) {
        return None;
    }
    let target = target.as_object()?;
    let id = object.get("id")?;
    let message = object.get("message")?;
    let delivery_status = object.get("deliveryStatus")?;
    let active_session_id = target.get("activeSessionId")?;
    let session_id = target.get("sessionId")?;
    let delivery_status = match delivery_status.as_str() {
        Some("delivered") => KernelDeliveryStatus::Delivered,
        Some("queued") => KernelDeliveryStatus::Queued,
        _ => return None,
    };
    if !id.is_string() || !message.is_string() || !active_session_id.is_string() || !session_id.is_string() {
        return None;
    }
    let receiver_role = match object.get("receiverRole").and_then(|value| value.as_str()) {
        Some("parent") => Some(KernelReceiverRole::Parent),
        Some("sibling") => Some(KernelReceiverRole::Sibling),
        Some("child") => Some(KernelReceiverRole::Child),
        _ => None,
    };
    Some(KernelSentAgentMessage {
        id: id.as_str()?.to_string(),
        message: message.as_str()?.to_string(),
        delivery_status,
        receiver_role,
        target: KernelSentAgentMessageTarget {
            active_session_id: active_session_id.as_str()?.to_string(),
            session_id: session_id.as_str()?.to_string(),
            session_name: target
                .get("sessionName")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string()),
        },
    })
}

/// `asStringArray(value)`.
pub fn as_string_array(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(|text| text.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

/// `asReasonArray(value)`.
pub fn as_reason_array(value: Option<&Value>) -> Vec<super::state_snapshot::SkippedVariable> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in items {
        if !is_record(entry) {
            continue;
        }
        let Some(object) = entry.as_object() else { continue };
        let Some(name) = object.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        out.push(super::state_snapshot::SkippedVariable {
            name: name.to_string(),
            reason: object
                .get("reason")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    out
}

/// `asMetricNumber(value)`.
pub fn as_metric_number(value: Option<&Value>) -> Option<f64> {
    let value = value?.as_f64()?;
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

/// A process/descendant receipt, not admission of a kill request or protocol EOF.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct KernelSettlement {
    pub ownership_scope: String,
    pub supported: bool,
    pub settled: bool,
    pub kernel_exited: bool,
    pub descendants_exited: bool,
    pub local_tasks_settled: bool,
    pub errors: Vec<String>,
}

impl KernelSettlement {
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            ownership_scope: "unproven".into(),
            supported: false,
            settled: false,
            kernel_exited: false,
            descendants_exited: false,
            local_tasks_settled: false,
            errors: vec![reason.into()],
        }
    }
}

#[derive(Clone, Default)]
pub struct KernelShutdownOptions {
    pub snapshot: bool,
    pub drain_host_requests: bool,
}

#[derive(Clone, Default)]
pub struct KernelRestoreOptions {
    /// Auto prefers any v2 root. Previous and legacy are explicit recovery actions.
    pub source: Option<KernelRestoreSource>,
}

/// Public surface every kernel client exposes to the provisioner and session layer.
pub trait KernelClient: Send + Sync {
    fn owner_session_id(&self) -> Option<String>;
    fn is_running(&self) -> bool;
    fn has_background_work(&self) -> bool;
    /// Terminal: the kernel died or was torn down; only a fresh manager can serve again.
    fn is_defunct(&self) -> bool;
    fn start<'a>(&'a self, options: KernelStartOptions) -> BoxFuture<'a, Result<(), KernelError>>;
    fn execute<'a>(
        &'a self,
        code: String,
        opts: ExecuteOptions,
    ) -> BoxFuture<'a, Result<ExecuteResult, KernelError>>;
    fn shutdown<'a>(&'a self, opts: KernelShutdownOptions) -> BoxFuture<'a, Result<bool, KernelError>>;
    /// Terminal stop, without executing a snapshot. Only this receipt can prove
    /// scoped descendant settlement; legacy shutdown/kill/dispose cannot.
    fn shutdown_and_settle<'a>(
        &'a self,
        owner_session_id: &'a str,
        timeout_ms: u64,
    ) -> BoxFuture<'a, Result<KernelSettlement, KernelError>> {
        let owner = self.owner_session_id();
        Box::pin(async move {
            if owner.as_deref() != Some(owner_session_id) || owner_session_id.is_empty() {
                return Err(KernelError::new("Kernel settlement owner mismatch"));
            }
            let _ = timeout_ms;
            Ok(KernelSettlement::unsupported("Kernel backend has no scoped process settlement"))
        })
    }
    fn restart<'a>(&'a self) -> BoxFuture<'a, Result<(), KernelError>>;
    fn kill<'a>(&'a self) -> BoxFuture<'a, ()>;
    fn dispose_sync(&self);
    fn snapshot_state<'a>(&'a self) -> BoxFuture<'a, Option<SnapshotResult>>;
    fn prune_oversized_variables<'a>(&'a self) -> BoxFuture<'a, Option<SnapshotResult>>;
    fn restore_state<'a>(&'a self, options: KernelRestoreOptions)
        -> BoxFuture<'a, Option<RestoreResult>>;
    /// Explicit compatibility export; callers must gate older-runtime launch on success.
    fn export_state_for_legacy_runtime<'a>(
        &'a self,
        source: super::state_snapshot::LegacyExportSource,
    ) -> BoxFuture<'a, Option<SnapshotLegacyExportResult>> {
        let _ = source;
        Box::pin(async { None })
    }
    fn list_namespace_names<'a>(&'a self, signal: Option<AbortSignal>) -> BoxFuture<'a, Option<Vec<String>>>;
}

/// Local stand-in for the memoized in-flight `Promise` the TypeScript shares
/// between concurrent callers (`inFlightEnsureKernelPython`). Settling once,
/// awaitable many times.
pub struct SharedPromise<T: Clone> {
    state: Mutex<Option<Result<T, KernelError>>>,
    notify: Notify,
}

impl<T: Clone> Default for SharedPromise<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> SharedPromise<T> {
    pub fn new() -> Self {
        SharedPromise {
            state: Mutex::new(None),
            notify: Notify::new(),
        }
    }

    pub fn settle(&self, outcome: Result<T, KernelError>) {
        let mut state = self.state.lock().unwrap();
        if state.is_none() {
            *state = Some(outcome);
        }
        drop(state);
        self.notify.notify_waiters();
    }

    pub async fn wait(&self) -> Result<T, KernelError> {
        loop {
            if let Some(outcome) = self.state.lock().unwrap().clone() {
                return outcome;
            }
            let notified = self.notify.notified();
            if let Some(outcome) = self.state.lock().unwrap().clone() {
                return outcome;
            }
            notified.await;
        }
    }
}

// ---------------------------------------------------------------------------
// Process-wide kernel registry
// ---------------------------------------------------------------------------

/// One registry serves every client kind; two parallel registries would
/// double-install process signal handlers.
static LIVE_KERNELS: OnceLock<Mutex<Vec<Arc<dyn KernelClient>>>> = OnceLock::new();
static SIGNAL_HANDLERS_INSTALLED: AtomicBool = AtomicBool::new(false);
static SESSION_CLEANUPS: OnceLock<Mutex<Vec<Arc<dyn Fn(Option<&str>) + Send + Sync>>>> = OnceLock::new();

pub fn live_kernels() -> &'static Mutex<Vec<Arc<dyn KernelClient>>> {
    LIVE_KERNELS.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn live_kernels_add(client: Arc<dyn KernelClient>) {
    ensure_session_cleanup_registered();
    let mut kernels = live_kernels().lock().unwrap();
    if !kernels.iter().any(|existing| Arc::ptr_eq(existing, &client)) {
        kernels.push(client);
    }
}

pub fn live_kernels_delete(client: &Arc<dyn KernelClient>) {
    live_kernels().lock().unwrap().retain(|existing| !Arc::ptr_eq(existing, client));
}

/// Local stand-in for `registerSessionResourceCleanup` (TODO(slice): needs
/// pi-ai::session_resources). Runs the registered callbacks; a host calls this
/// when a session ends.
pub fn register_session_resource_cleanup(callback: Arc<dyn Fn(Option<&str>) + Send + Sync>) {
    SESSION_CLEANUPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(callback);
}

pub fn run_session_resource_cleanup(session_id: Option<&str>) {
    let callbacks: Vec<Arc<dyn Fn(Option<&str>) + Send + Sync>> = SESSION_CLEANUPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clone();
    for callback in callbacks {
        callback(session_id);
    }
}

fn ensure_session_cleanup_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        // Module-load equivalent of the top-level registration in shared.ts.
        register_session_resource_cleanup(Arc::new(|session_id: Option<&str>| {
            let kernels: Vec<Arc<dyn KernelClient>> = live_kernels().lock().unwrap().clone();
            for kernel in kernels {
                let owned = kernel.owner_session_id();
                if session_id.is_none() || owned.as_deref() == session_id {
                    let kernel = kernel.clone();
                    if let Ok(handle) = tokio::runtime::Handle::try_current() {
                        handle.spawn(async move {
                            let _ = kernel
                                .shutdown(KernelShutdownOptions {
                                    snapshot: true,
                                    drain_host_requests: true,
                                })
                                .await;
                        });
                    }
                }
            }
        }));
    });
}

/// `installSignalHandlersOnce()`.
///
/// Node also hooks `beforeExit` and `exit`; Rust has no equivalent, so the async
/// path is exposed as [`run_before_exit_shutdown`] and the synchronous path as
/// [`dispose_all_kernels_sync`] for the CLI entry point to call.
pub fn install_signal_handlers_once() {
    if SIGNAL_HANDLERS_INSTALLED.load(Ordering::SeqCst) {
        return;
    }
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    if SIGNAL_HANDLERS_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }

    handle.spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            async_shutdown().await;
            std::process::exit(130);
        }
    });

    #[cfg(unix)]
    handle.spawn(async {
        if let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            if term.recv().await.is_some() {
                async_shutdown().await;
                std::process::exit(143);
            }
        }
    });
}

/// These paths can await, so flush the namespace snapshot before tearing down.
pub async fn async_shutdown() {
    let kernels: Vec<Arc<dyn KernelClient>> = live_kernels().lock().unwrap().clone();
    let tasks = kernels.into_iter().map(|kernel| {
        tokio::spawn(async move {
            let _ = kernel
                .shutdown(KernelShutdownOptions {
                    snapshot: true,
                    drain_host_requests: false,
                })
                .await;
        })
    });
    for task in tasks {
        let _ = task.await;
    }
}

/// `process.on("exit", ...)`: synchronous best-effort teardown.
pub fn dispose_all_kernels_sync() {
    let kernels: Vec<Arc<dyn KernelClient>> = live_kernels().lock().unwrap().clone();
    for kernel in kernels {
        kernel.dispose_sync();
    }
}

/// Derive the sibling CAS root for a snapshot config, as repl-manager does.
pub fn cas_root_path_for(config: &KernelSnapshotConfig) -> String {
    config
        .cas_root_path
        .clone()
        .unwrap_or_else(|| cas_snapshot_root_for_legacy_path(&config.path))
}

// ---------------------------------------------------------------------------
// Performance metrics (TODO(slice): needs pi-agent-core::performance_metrics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMetricOutcome {
    Success,
    Failure,
    Cancelled,
    Unavailable,
}

impl PerformanceMetricOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricOutcome::Success => "success",
            PerformanceMetricOutcome::Failure => "failure",
            PerformanceMetricOutcome::Cancelled => "cancelled",
            PerformanceMetricOutcome::Unavailable => "unavailable",
        }
    }
}

/// Minimal local shape of `PerformanceMetricEvent` for the snapshot metric.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetricEvent {
    pub operation: String,
    pub component: Option<String>,
    pub outcome: Option<PerformanceMetricOutcome>,
    pub measurements: Vec<(&'static str, Option<f64>)>,
}

pub trait PerformanceMetricRecorder: Send + Sync {
    fn session_id(&self) -> &str;
    fn monotonic_now(&self) -> f64;
    fn record(&self, event: PerformanceMetricEvent);
}

/// `elapsedMetricMs(start, end)`.
pub fn elapsed_metric_ms(start: Option<f64>, end: Option<f64>) -> Option<f64> {
    let (start, end) = (start?, end?);
    if !start.is_finite() || !end.is_finite() || end < start {
        return None;
    }
    Some(end - start)
}

/// Defensively contains third-party recorder failures at every call site.
pub fn safe_record_performance_metric(
    recorder: Option<&Arc<dyn PerformanceMetricRecorder>>,
    event: PerformanceMetricEvent,
) {
    let Some(recorder) = recorder else { return };
    // `recorder.record` is infallible in Rust; a panic inside it is still
    // contained the same way the TypeScript catch contains a throwing recorder.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recorder.record(event)));
    let _ = result;
}

/// `safeMetricNow(recorder)`.
pub fn safe_metric_now(recorder: Option<&Arc<dyn PerformanceMetricRecorder>>) -> Option<f64> {
    let recorder = recorder?;
    let value = recorder.monotonic_now();
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// `Date.now()` in milliseconds.
pub fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as f64,
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_diff_display_accepts_valid_payloads_and_ignores_junk() {
        let parsed = parse_diff_display(&json!({
            "path": "a.ts", "old_str": "x", "new_str": "y", "start_line": 3
        }))
        .expect("valid payload");
        assert_eq!(parsed.path, "a.ts");
        assert_eq!(parsed.start_line, Some(3.0));
        assert!(parse_diff_display(&json!({"path": "a.ts"})).is_none());
        assert!(parse_diff_display(&json!([1, 2])).is_none());
        assert!(parse_diff_display(&json!({"path": 1, "old_str": "a", "new_str": "b"})).is_none());
    }

    #[test]
    fn parse_attachment_display_reports_oversized() {
        let ok = parse_attachment_display(&json!({"mime_type": "image/png", "data": "AAAA"}));
        match ok {
            Some(ParsedAttachment::Attachment(attachment)) => {
                assert_eq!(attachment.mime_type, "image/png");
                assert_eq!(attachment.path, None);
            }
            other => panic!("unexpected {other:?}"),
        }
        let big = "a".repeat(MAX_ATTACHMENT_DATA_CHARS + 1);
        assert_eq!(
            parse_attachment_display(&json!({"mime_type": "image/png", "data": big})),
            Some(ParsedAttachment::Oversized)
        );
        assert_eq!(parse_attachment_display(&json!({"data": "x"})), None);
    }

    #[test]
    fn parse_sent_agent_message_requires_the_documented_fields() {
        let parsed = parse_sent_agent_message(&json!({
            "id": "1",
            "message": "hi",
            "deliveryStatus": "queued",
            "receiverRole": "sibling",
            "target": {"activeSessionId": "a", "sessionId": "b", "sessionName": "n"}
        }))
        .expect("valid payload");
        assert_eq!(parsed.delivery_status, KernelDeliveryStatus::Queued);
        assert_eq!(parsed.receiver_role, Some(KernelReceiverRole::Sibling));
        assert_eq!(parsed.target.session_name.as_deref(), Some("n"));

        assert!(parse_sent_agent_message(&json!({
            "id": "1", "message": "hi", "deliveryStatus": "sent",
            "target": {"activeSessionId": "a", "sessionId": "b"}
        }))
        .is_none());
        assert!(parse_sent_agent_message(&json!({"id": "1"})).is_none());
    }

    #[test]
    fn sent_agent_message_omits_absent_optionals() {
        let message = KernelSentAgentMessage {
            id: "1".into(),
            message: "hi".into(),
            delivery_status: KernelDeliveryStatus::Delivered,
            receiver_role: None,
            target: KernelSentAgentMessageTarget {
                active_session_id: "a".into(),
                session_id: "b".into(),
                session_name: None,
            },
        };
        let value = serde_json::to_value(&message).unwrap();
        assert!(value.get("receiverRole").is_none());
        assert!(value["target"].get("sessionName").is_none());
        assert_eq!(value["deliveryStatus"], json!("delivered"));
    }

    #[test]
    fn reason_and_string_arrays_filter_bad_entries() {
        let reasons = as_reason_array(Some(&json!([{"name": "a", "reason": "r"}, {"name": 1}, "x"])));
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0].name, "a");
        assert_eq!(as_string_array(Some(&json!(["a", 2, "b"]))), vec!["a", "b"]);
        assert!(as_string_array(None).is_empty());
    }

    #[test]
    fn metric_numbers_reject_negative_and_non_numbers() {
        assert_eq!(as_metric_number(Some(&json!(3))), Some(3.0));
        assert_eq!(as_metric_number(Some(&json!(-1))), None);
        assert_eq!(as_metric_number(Some(&json!("3"))), None);
        assert_eq!(as_metric_number(None), None);
    }

    #[test]
    fn elapsed_metric_ms_matches_the_typescript_bounds() {
        assert_eq!(elapsed_metric_ms(Some(1.0), Some(4.0)), Some(3.0));
        assert_eq!(elapsed_metric_ms(Some(4.0), Some(1.0)), None);
        assert_eq!(elapsed_metric_ms(None, Some(1.0)), None);
    }

    #[test]
    fn ordered_map_helpers_keep_insertion_position() {
        let mut entries: Vec<(String, i32)> = Vec::new();
        ordered_set(&mut entries, "a".into(), 1);
        ordered_set(&mut entries, "b".into(), 2);
        ordered_set(&mut entries, "a".into(), 3);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], ("a".to_string(), 3));
        assert_eq!(ordered_get(&entries, "b"), Some(&2));
        assert_eq!(ordered_delete(&mut entries, "a"), Some(3));
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn deferred_settles_once() {
        let (mut resolver, deferred) = create_deferred::<i32>();
        resolver.resolve(7);
        resolver.resolve(9);
        assert_eq!(deferred.await_promise().await.unwrap(), 7);
    }

    #[tokio::test]
    async fn race_startup_with_abort_rejects_on_abort() {
        let signal = AbortSignal::new();
        let aborted = signal.clone();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            aborted.abort(None);
        });
        let pending = std::future::pending::<Result<(), KernelError>>();
        let result = race_startup_with_abort(pending, Some(signal)).await;
        assert_eq!(result, Err(KernelError::StartupAborted));
        assert_eq!(
            error_message(&create_kernel_startup_abort_error()),
            "Kernel startup aborted"
        );
    }

    #[tokio::test]
    async fn abort_signal_any_forwards_the_first_abort() {
        let first = AbortSignal::new();
        let second = AbortSignal::new();
        let combined = AbortSignal::any(vec![first.clone(), second.clone()]);
        assert!(!combined.is_aborted());
        second.abort(Some(KernelError::new("boom")));
        assert!(combined.is_aborted());
        assert_eq!(combined.reason(), Some(KernelError::new("boom")));
    }
}


#[cfg(test)]
mod t17_controls_tests {
    //! T17 owner 'controls': G-10 - the TS `data.length > MAX` guard counts
    //! UTF-16 code units (packages/coding-agent/src/core/kernel/shared.ts,
    //! `parseAttachmentDisplay`), while the Rust port counted `char`s, so
    //! astral-plane payloads could pass the Rust limit at half the TS size.

    use super::*;
    use serde_json::json;

    /// Boundary control (must stay green in baseline AND candidate): exactly the
    /// limit in UTF-16 units must still parse; BMP characters count identically
    /// under both units.
    #[test]
    fn t17_g10_bmp_and_exact_limit_stay_within() {
        let exact_ascii = "a".repeat(MAX_ATTACHMENT_DATA_CHARS);
        let parsed = parse_attachment_display(&json!({"mime_type": "image/png", "data": exact_ascii}));
        assert!(matches!(parsed, Some(ParsedAttachment::Attachment(_))), "exactly the limit must parse; got {parsed:?}");
    }

    /// G-10 reproduction: 5_000_001 astral emoji encode to 10_000_002 UTF-16
    /// units (over the TS limit) but only 5_000_001 chars, so the Rust port
    /// returned a valid attachment. Baseline expectation: FAIL
    /// (`Some(Attachment)` instead of `Oversized`).
    #[test]
    fn t17_g10_astral_payload_counts_utf16_units_like_typescript() {
        let astral = "\u{1F648}".repeat(5_000_001);
        assert_eq!(
            astral.encode_utf16().count(),
            MAX_ATTACHMENT_DATA_CHARS + 2,
            "fixture must encode above the limit in UTF-16 units"
        );
        assert_eq!(
            parse_attachment_display(&json!({"mime_type": "image/png", "data": astral})),
            Some(ParsedAttachment::Oversized),
            "TS data.length counts UTF-16 units; an astral payload over the limit must be Oversized"
        );
    }

    /// Just under the limit in UTF-16 units must still parse (fix must not
    /// overreach past the TS boundary).
    #[test]
    fn t17_g10_astral_payload_just_under_the_utf16_limit_parses() {
        // 4_999_999 astral emoji = 9_999_998 UTF-16 units <= 10_000_000.
        let data = "\u{1F648}".repeat(4_999_999);
        let parsed = parse_attachment_display(&json!({"mime_type": "image/png", "data": data}));
        assert!(matches!(parsed, Some(ParsedAttachment::Attachment(_))), "under-limit astral payload must parse; got {parsed:?}");
    }
}
