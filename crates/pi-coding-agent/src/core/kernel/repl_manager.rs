//! Port of packages/coding-agent/src/core/kernel/repl-manager.ts.
//!
//! Kernel client for the REPL runtime: the kernel is a JSON-lines subprocess
//! (`python -m rlm.repl`) - requests on stdin, events on stdout, stderr kept as
//! a diagnostics tail. The protocol is documented in prime-agent-runtime/src/rlm/repl.md.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use tokio::io::AsyncReadExt;

use crate::core::kernel::bootstrap::{ensure_kernel_python, EnsureKernelPythonOptions};
use crate::core::kernel::shared::{
    as_metric_number, as_reason_array, as_string_array, create_kernel_startup_abort_error,
    error_message, install_signal_handlers_once, is_record, kernel_busy_after_interrupt_error,
    live_kernels_add, live_kernels_delete, now_ms, ordered_delete, ordered_set,
    parse_attachment_display, parse_diff_display, parse_sent_agent_message,
    race_startup_with_abort, safe_metric_now, safe_record_performance_metric, AbortSignal,
    ExecError, ExecuteOptions, ExecuteResult, ExecuteStatus, HostRequestHandlers,
    InternalExecuteResult, KernelAttachment, KernelClient, KernelDiffDisplay, KernelError,
    KernelManagerOptions, KernelRestoreOptions, KernelSentAgentMessage, KernelShutdownOptions,
    KernelSnapshotConfig, KernelStartOptions, ParsedAttachment, PerformanceMetricEvent,
    PerformanceMetricOutcome, PerformanceMetricRecorder, SharedPromise, StreamName,
    AGENT_MESSAGE_DISPLAY_MIME, ATTACHMENT_DISPLAY_MIME, BASH_ACTIVITY_DISPLAY_MIME,
    DEFAULT_MAX_OUTPUT_CHARS, DEFAULT_SNAPSHOT_DEBOUNCE_MS, DIFF_DISPLAY_MIME,
    HOST_REQUEST_SHUTDOWN_TIMEOUT_MS, KERNEL_ABORT_GRACE_MS, KERNEL_BUSY_REUSE_WAIT_MS,
    KERNEL_SHUTDOWN_TIMEOUT_MS, MAX_ATTACHMENT_DATA_CHARS, MAX_LATE_SENT_AGENT_MESSAGE_HANDLERS,
    SNAPSHOT_EXECUTION_TIMEOUT_MS,
};
use crate::core::kernel::state_snapshot::{
    cas_snapshot_root_for_legacy_path, cas_snapshot_state_exists, KernelRestoreSource,
    KernelSnapshotFormat, LegacyExportSource, RestoreResult, SkippedVariable,
    SnapshotLegacyExportResult, SnapshotPerformanceMetadata, SnapshotResult,
    DEFAULT_SNAPSHOT_MAX_BYTES, DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES,
};
use crate::core::orphan_process_journal::{
    reap_kernel_orphan_processes, record_orphan_process_state,
};
use crate::utils::child_process::{spawn_hidden, spawn_sync_hidden_with_timeout, Signal, SpawnOptions};

const REPL_PROTOCOL_VERSION: f64 = 3.0;
const READY_TIMEOUT_MS: u64 = 30_000;
const REPAIR_STEP_TIMEOUT_MS: u64 = 30_000;
// Runtime-minted host-request ids never repeat; the bound only guards a
// misbehaving runtime from growing the dedup set forever.
const MAX_HANDLED_HOST_REQUEST_IDS: usize = 1024;
// Cap for unattributed background output buffered between and during cells.
const MAX_BACKGROUND_OUTPUT_CHARS: usize = 64 * 1024;

const MAX_KERNEL_STDERR_CHARS: usize = 8 * 1024;
const MAX_KERNEL_STDERR_LOG_BYTES: u64 = 5 * 1024 * 1024;
const KERNEL_STDERR_LOG_BUDGET_MARKER: &str = "[stderr log budget exhausted]\n";
/// Sentinel for `stderrLogWritable = false`: the budget is spent.
const STDERR_LOG_SPENT: u64 = u64::MAX;
/// Cap on the stderr tail embedded in a startup failure message.
const STARTUP_STDERR_TAIL_CHARS: usize = 1024;
/// `spawnSyncHidden(taskkill, ..., { timeout: 5000 })` in `cleanupResources`.
const TASKKILL_TIMEOUT_MS: u64 = 5000;
/// Read chunk size for the child's stdout / stderr pumps.
const STREAM_CHUNK_BYTES: usize = 16 * 1024;
// Wire bytes, including JSON escaping. Matches the Python sender's smaller caps.
const MAX_PROTOCOL_FRAME_BYTES: usize = 32 * 1024 * 1024;
const MAX_CELL_SOURCE_CHARS: usize = 2048;

fn cap_cell_source(code: &str) -> String {
    let mut chars = code.chars();
    let prefix: String = chars.by_ref().take(MAX_CELL_SOURCE_CHARS).collect();
    if chars.next().is_some() {
        format!("{prefix}\n[... cell source truncated ...]")
    } else {
        prefix
    }
}

fn first_protocol_frame_too_large(buffer: &str) -> bool {
    buffer.find('\n').unwrap_or(buffer.len()) > MAX_PROTOCOL_FRAME_BYTES
}

#[cfg(unix)]
fn tighten_log_permissions(path: &std::path::Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() || (mode == 0o700 && metadata.is_dir()) =>
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Complete event vocabulary of protocol version 2 (see prime-agent-runtime/src/rlm/repl.md).
/// The version handshake is exact, so an unknown kind is corruption, not a newer runtime.
const PROTOCOL_EVENT_KINDS: [&str; 8] = [
    "ready",
    "stdout",
    "stderr",
    "result",
    "display",
    "host_request",
    "error",
    "done",
];

/// `this.state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Starting,
    Running,
    Shutdown,
}

/// `fs.writeSync` may write fewer bytes than asked (partial ENOSPC, signals); loop until done.
fn write_fully_sync(file: &mut std::fs::File, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut offset = 0;
    while offset < data.len() {
        offset += file.write(&data[offset..])?;
    }
    Ok(())
}

/// `Array.isArray(value) ? value.filter(...) : []` for a raw event field.
fn as_string_array_field(value: Option<&Value>) -> Vec<String> {
    as_string_array(value)
}

/// `truncate`: `text.slice(0, max)` measured in characters.
fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// The last `max` characters, `text.slice(-max)`.
fn tail_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    text.chars().skip(count - max).collect()
}

/// `signal?.aborted === true`.
fn is_aborted(signal: &Option<AbortSignal>) -> bool {
    signal
        .as_ref()
        .map(|signal| signal.is_aborted())
        .unwrap_or(false)
}

/// Await `signal`, or never resolve when there is none (the `undefined` case).
async fn wait_for_abort(signal: Option<AbortSignal>) {
    match signal {
        Some(signal) => signal.wait().await,
        None => std::future::pending::<()>().await,
    }
}

/// `new StringDecoder("utf8")`: keeps an incomplete trailing sequence for the next chunk.
///
/// `pending` holds the bytes of a partial multi-byte character; a chunk that ends
/// mid-character must not emit replacement characters for bytes that the next
/// chunk completes.
fn decode_utf8_chunk(pending: &mut Vec<u8>, chunk: &[u8]) -> String {
    pending.extend_from_slice(chunk);
    match std::str::from_utf8(pending) {
        Ok(text) => {
            let text = text.to_string();
            pending.clear();
            text
        }
        Err(error) => {
            let valid = error.valid_up_to();
            let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            // A hopeless sequence (4+ invalid bytes) has to be flushed as replacement
            // characters instead of being held forever.
            if pending.len() > 3 {
                let text = format!("{text}{}", String::from_utf8_lossy(pending));
                pending.clear();
                text
            } else {
                text
            }
        }
    }
}

/// `decoder.end()`: flush whatever bytes are left, replacing an incomplete tail.
fn flush_utf8_pending(pending: &mut Vec<u8>) -> String {
    if pending.is_empty() {
        return String::new();
    }
    let text = String::from_utf8_lossy(pending).into_owned();
    pending.clear();
    text
}

/// `uuid()` from the `uuid` package: a random v4 uuid.
fn uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `^[a-f0-9]{32}$` for bash activity ids.
fn is_bash_activity_id(id: &str) -> bool {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let regex = REGEX.get_or_init(|| regex::Regex::new(r"^[a-f0-9]{32}$").expect("valid regex"));
    regex.is_match(id)
}

/**
 * Reason a JSON object still isn't a valid protocol frame, or None.
 * `done` and `host_request` route strictly by non-empty string id (the runtime
 * mints uuid hex ids and echoes the host's uuids); silently dropping an id-less
 * one would leave the awaiting request unsettled forever.
 */
fn invalid_protocol_frame_reason(event: &Map<String, Value>) -> Option<String> {
    let kind = match event.get("event").and_then(|value| value.as_str()) {
        Some(kind) if PROTOCOL_EVENT_KINDS.contains(&kind) => kind,
        _ => return Some("unknown protocol event".to_string()),
    };
    if kind == "done" || kind == "host_request" {
        match event.get("id").and_then(|value| value.as_str()) {
            Some(id) if !id.is_empty() => {}
            _ => return Some(format!("{kind} frame without id")),
        }
    }
    None
}

/// `asMetricsNumber`: only a finite, non-negative number counts.
fn as_metric_number_field(value: Option<&Value>) -> Option<f64> {
    as_metric_number(value)
}

/// `asSnapshotPerformanceMetadata(value)`.
fn as_snapshot_performance_metadata(value: Option<&Value>) -> Option<SnapshotPerformanceMetadata> {
    let value = value?;
    if !is_record(value) {
        return None;
    }
    Some(SnapshotPerformanceMetadata {
        serialization_wall_ms: as_metric_number_field(value.get("serialization_wall_ms")),
        serialization_cpu_ms: as_metric_number_field(value.get("serialization_cpu_ms")),
        serialization_max_variable_ms: as_metric_number_field(value.get("serialization_max_variable_ms")),
        serialization_slow_variables: as_metric_number_field(value.get("serialization_slow_variables")),
        serialization_saved_ms: as_metric_number_field(value.get("serialization_saved_ms")),
        serialization_skipped_ms: as_metric_number_field(value.get("serialization_skipped_ms")),
        serialized_bytes: as_metric_number_field(value.get("serialized_bytes")),
        write_ms: as_metric_number_field(value.get("write_ms")),
        written_bytes: as_metric_number_field(value.get("written_bytes")),
        total_wall_ms: as_metric_number_field(value.get("total_wall_ms")),
    })
}

/// `snapshotMetric` bookkeeping for one queued request.
#[derive(Clone)]
struct SnapshotMetricState {
    recorder: Arc<dyn PerformanceMetricRecorder>,
    started_at: Option<f64>,
    timing: Arc<Mutex<SnapshotQueueTiming>>,
}

#[derive(Default)]
struct SnapshotQueueTiming {
    dequeued_at: Option<f64>,
    following_cell_queued_at: Option<f64>,
}

/// One open append handle plus its remaining write budget. The stderr pump owns
/// it: Node keeps the fd in the stream's close handler closure.
struct StderrLog {
    file: std::fs::File,
    budget: u64,
}

/// `ActiveExecution`: the in-flight request and everything parsed from its events.
struct ActiveExecution {
    request_id: String,
    /// Source of the cell currently executing; surfaced to rlm.run spawns.
    code: String,
    started: f64,
    max_chars: usize,
    opts: ExecuteOptions,
    stdout: String,
    stderr: String,
    stdout_truncated: bool,
    stderr_truncated: bool,
    result: Option<String>,
    diffs: Vec<KernelDiffDisplay>,
    attachments: Vec<KernelAttachment>,
    sent_agent_messages: Vec<KernelSentAgentMessage>,
    /// Stream text without this execution's id: user threads, other cells' leftovers, raw fd writes.
    background_output: String,
    background_output_truncated: bool,
    error: Option<ExecError>,
    status: ExecuteStatus,
    done_fields: Option<Map<String, Value>>,
    settled: bool,
    interrupt_requested_at: Option<f64>,
    resolver: Option<tokio::sync::oneshot::Sender<Result<InternalExecuteResult, KernelError>>>,
}

impl ActiveExecution {
    fn resolve(&mut self, result: InternalExecuteResult) {
        if let Some(resolver) = self.resolver.take() {
            let _ = resolver.send(Ok(result));
        }
    }

    fn reject(&mut self, error: KernelError) {
        if let Some(resolver) = self.resolver.take() {
            let _ = resolver.send(Err(error));
        }
    }

    /// `this.activeExecution === execution` for a mutex-guarded slot.
    fn same_as(
        slot: &Option<Arc<Mutex<ActiveExecution>>>,
        execution: &Arc<Mutex<ActiveExecution>>,
    ) -> bool {
        match slot {
            Some(current) => Arc::ptr_eq(current, execution),
            None => false,
        }
    }
}

/// The `Pick<KernelManagerOptions, ...>` subset the manager keeps.
struct ManagerOptions {
    python: Mutex<Option<String>>,
    cwd: Option<String>,
    env: Option<HashMap<String, String>>,
    session_id: Option<String>,
    host_handlers: Option<HostRequestHandlers>,
    python_skills: Option<Vec<crate::core::kernel::shared::KernelPythonSkill>>,
    snapshot: Option<KernelSnapshotConfig>,
    performance_metrics: Option<Arc<dyn PerformanceMetricRecorder>>,
    bootstrap_code: Option<String>,
    stderr_log_path: Option<String>,
    on_background_work_settled: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// `{ superseded: boolean }` shared with an in-flight protocol repair.
struct ProtocolRepairOwner {
    superseded: AtomicBool,
}

impl ProtocolRepairOwner {
    fn new() -> Arc<Self> {
        Arc::new(ProtocolRepairOwner {
            superseded: AtomicBool::new(false),
        })
    }

    fn is_superseded(&self) -> bool {
        self.superseded.load(Ordering::SeqCst)
    }
}

/// `clearTimeout`-able debounce handle.
struct SnapshotTimerHandle {
    handle: tokio::task::JoinHandle<()>,
}

impl SnapshotTimerHandle {
    fn clear(self) {
        self.handle.abort();
    }
}

/// Exit state of one spawned child: `child.exitCode` / `child.signalCode` plus an
/// awaitable `exit` event (Node allows many `once("exit", ...)` listeners).
struct ExitState {
    status: Mutex<Option<(Option<i32>, Option<String>)>>,
    notify: tokio::sync::Notify,
}

impl ExitState {
    fn new() -> Arc<Self> {
        Arc::new(ExitState {
            status: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
        })
    }

    fn settle(&self, code: Option<i32>, signal: Option<String>) {
        {
            let mut status = self.status.lock().unwrap();
            if status.is_none() {
                *status = Some((code, signal));
            }
        }
        self.notify.notify_waiters();
    }

    fn exit_code(&self) -> Option<i32> {
        self.status
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|(code, _)| *code)
    }

    fn signal_code(&self) -> Option<String> {
        self.status
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|(_, signal)| signal.clone())
    }

    /// `child.exitCode !== null || child.signalCode !== null`.
    fn has_exited(&self) -> bool {
        self.status.lock().unwrap().is_some()
    }

    async fn wait(&self) {
        loop {
            if self.has_exited() {
                return;
            }
            let notified = self.notify.notified();
            if self.has_exited() {
                return;
            }
            notified.await;
        }
    }
}

/// One spawned kernel child: streams, exit state and the destroy hooks that
/// stand in for `stream.destroy()`.
struct ChildState {
    /// Monotonic spawn identity, replacing `this.child !== child` reference checks.
    id: u64,
    pid: Option<u32>,
    stdin: Arc<tokio::sync::Mutex<Option<tokio::process::ChildStdin>>>,
    /// `stdin.destroy()` from `cleanupResources`.
    stdin_destroyed: AtomicBool,
    exit: Arc<ExitState>,
    // `stdout.destroy()` / `stderr.destroy()`: break the pump loops.
    destroy_stdout: Arc<Latch>,
    destroy_stderr: Arc<Latch>,
    stderr_closed: Arc<Latch>,
}

impl ChildState {
    async fn wait_for_stderr_close(&self) {
        self.stderr_closed.wait().await;
    }
}

/// `ReplKernelManager`.
pub struct ReplKernelManager {
    state: Arc<KernelState>,
}

/// The client-side half of the shared state, kept so `liveKernels` sees this
/// manager as the `dyn KernelClient` Node pushes.
type ClientWeak = std::sync::Weak<ReplKernelManager>;

impl ReplKernelManager {
    /// Shared handle to the internal state: the stream pumps call back into the
    /// manager the way Node's event listeners capture `this`.
    pub fn state(&self) -> Arc<KernelState> {
        self.state.clone()
    }
}

/// `new ReplKernelManager(options)`. `new_cyclic` gives the state a weak handle
/// back to the client so `liveKernels` and the signal handlers can reach it.
pub fn new_repl_kernel_manager(options: KernelManagerOptions) -> Arc<ReplKernelManager> {
    Arc::new_cyclic(|weak| ReplKernelManager {
        state: Arc::new(KernelState::new(options, weak.clone())),
    })
}

pub struct KernelState {
    /// Weak handle to the owning client, injected by `new_repl_kernel_manager`.
    client_weak: Mutex<ClientWeak>,
    options: ManagerOptions,
    handled_host_request_ids: Mutex<Vec<String>>,
    child: Mutex<Option<Arc<ChildState>>>,
    next_child_id: std::sync::atomic::AtomicU64,
    ready_deferred: Mutex<Option<Arc<SharedPromise<f64>>>>,
    kernel_stderr: Mutex<String>,
    /// Serializes execute() calls - the runtime runs one request at a time.
    execution_queue: Mutex<Arc<SharedPromise<()>>>,
    /// Snapshot at the queue tail, used only to measure an observed following-cell block.
    execution_queue_tail_snapshot_metric: Mutex<Option<SnapshotMetricState>>,
    runtime_snapshot_formats: Mutex<Vec<String>>,
    active_execution: Mutex<Option<Arc<Mutex<ActiveExecution>>>>,
    /// Notification plus a monotonic epoch: `notifyActiveExecutionIdle` resolves
    /// every waiter registered before the bump, so a lost wakeup cannot park one.
    active_execution_idle: tokio::sync::Notify,
    active_execution_idle_epoch: std::sync::atomic::AtomicU64,
    active_execution_reconciliation: Mutex<Option<Arc<SharedPromise<bool>>>>,
    late_sent_agent_message_handlers: Mutex<Vec<LateSentAgentMessageHandler>>,
    /// Resolvers for done events outside the active execution (the shutdown reply).
    pending_done_waiters: Mutex<Vec<(String, Arc<Latch>)>>,
    /// Source of the most recently started cell, retained after it finishes so
    /// rlm.run spawns from detached asyncio tasks (cell already idle) can still
    /// attribute their spawning program.
    last_cell_code: Mutex<Option<String>>,
    /// Unattributed stream text that arrived between cells; surfaced on the next execution.
    pending_background_output: Mutex<String>,
    pending_background_output_truncated: AtomicBool,
    in_flight_host_requests: Mutex<Vec<Arc<SharedPromise<()>>>>,
    background_bash_handles: Mutex<Vec<(String, f64)>>,
    state: Mutex<State>,
    /// Bumped by every teardown so a stale in-flight doStart can never touch a newer kernel.
    start_generation: std::sync::atomic::AtomicU64,
    /// Generation whose graceful shutdown() owns the teardown, so the exit handler must not run it.
    graceful_shutdown_generation: Mutex<Option<u64>>,
    graceful_shutdown_promise: Mutex<Option<Arc<SharedPromise<bool>>>>,
    /// Memoized so concurrent callers all await the same in-flight startup.
    start_promise: Mutex<Option<Arc<SharedPromise<()>>>>,
    /// Pending debounced auto-snapshot, if one has been scheduled.
    snapshot_timer: Mutex<Option<SnapshotTimerHandle>>,
    /// While the final dispose snapshot is flushing, new external executions are rejected.
    flushing_snapshot_for_dispose: AtomicBool,
    /// In-flight final snapshot flush; concurrent teardowns join it instead of re-flushing.
    snapshot_flush_for_dispose: Mutex<Option<Arc<SharedPromise<()>>>>,
    /// Repairs a child whose dedicated protocol stream emitted an invalid frame.
    protocol_repair_promise: Mutex<Option<Arc<SharedPromise<()>>>>,
    protocol_repair_owner: Mutex<Option<Arc<ProtocolRepairOwner>>>,
    /// Corruption seen while still "starting" (e.g. ready and garbage in one chunk) fails that start.
    startup_protocol_error: Mutex<Option<KernelError>>,
    /// A repair discarded its kernel: the next fresh start must re-run the runtime bootstrap.
    pending_rebootstrap: AtomicBool,
    /// Restore the saved namespace on that fresh start too.
    pending_restore: AtomicBool,
    rebootstrap_promise: Mutex<Option<Arc<SharedPromise<bool>>>>,
    teardown_in_flight: std::sync::atomic::AtomicUsize,
}

type LateSentAgentMessageHandler = (String, Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>);

impl KernelState {
    fn new(options: KernelManagerOptions, client_weak: ClientWeak) -> KernelState {
        let execution_queue = Arc::new(SharedPromise::new());
        execution_queue.settle(Ok(()));
        KernelState {
            client_weak: Mutex::new(client_weak),
            options: ManagerOptions {
                python: Mutex::new(options.python),
                cwd: options.cwd,
                env: options.env,
                session_id: options.session_id,
                host_handlers: options.host_handlers,
                python_skills: options.python_skills,
                snapshot: options.snapshot,
                performance_metrics: options.performance_metrics,
                bootstrap_code: options.bootstrap_code,
                stderr_log_path: options.stderr_log_path,
                on_background_work_settled: options.on_background_work_settled,
            },
            handled_host_request_ids: Mutex::new(Vec::new()),
            child: Mutex::new(None),
            next_child_id: std::sync::atomic::AtomicU64::new(1),
            ready_deferred: Mutex::new(None),
            kernel_stderr: Mutex::new(String::new()),
            execution_queue: Mutex::new(execution_queue),
            execution_queue_tail_snapshot_metric: Mutex::new(None),
            runtime_snapshot_formats: Mutex::new(Vec::new()),
            active_execution: Mutex::new(None),
            active_execution_idle: tokio::sync::Notify::new(),
            active_execution_idle_epoch: std::sync::atomic::AtomicU64::new(0),
            active_execution_reconciliation: Mutex::new(None),
            late_sent_agent_message_handlers: Mutex::new(Vec::new()),
            pending_done_waiters: Mutex::new(Vec::new()),
            last_cell_code: Mutex::new(None),
            pending_background_output: Mutex::new(String::new()),
            pending_background_output_truncated: AtomicBool::new(false),
            in_flight_host_requests: Mutex::new(Vec::new()),
            background_bash_handles: Mutex::new(Vec::new()),
            state: Mutex::new(State::Idle),
            start_generation: std::sync::atomic::AtomicU64::new(0),
            graceful_shutdown_generation: Mutex::new(None),
            graceful_shutdown_promise: Mutex::new(None),
            start_promise: Mutex::new(None),
            snapshot_timer: Mutex::new(None),
            flushing_snapshot_for_dispose: AtomicBool::new(false),
            snapshot_flush_for_dispose: Mutex::new(None),
            protocol_repair_promise: Mutex::new(None),
            protocol_repair_owner: Mutex::new(None),
            startup_protocol_error: Mutex::new(None),
            pending_rebootstrap: AtomicBool::new(false),
            pending_restore: AtomicBool::new(false),
            rebootstrap_promise: Mutex::new(None),
            teardown_in_flight: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn state(&self) -> State {
        *self.state.lock().unwrap()
    }

    fn set_state(&self, state: State) {
        *self.state.lock().unwrap() = state;
    }

    fn owner_session_id(&self) -> Option<String> {
        self.options.session_id.clone()
    }

    fn has_background_work(&self) -> bool {
        !self.background_bash_handles.lock().unwrap().is_empty()
    }

    fn notify_background_work_settled(&self) {
        if let Some(callback) = &self.options.on_background_work_settled {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback()));
        }
    }

    fn append_kernel_diagnostic(&self, message: &str) {
        let suffix = if message.ends_with('\n') { "" } else { "\n" };
        self.append_kernel_stderr_text(&format!("[kernel] {message}{suffix}"));
    }

    fn append_kernel_stderr_text(&self, text: &str) {
        let mut guard = self.kernel_stderr.lock().unwrap();
        let combined = format!("{}{}", guard, text);
        *guard = tail_chars(&combined, MAX_KERNEL_STDERR_CHARS);
    }

    /**
     * The write budget is the file's remaining capacity, not a fresh allowance,
     * so current file and `.old` each stay near MAX_KERNEL_STDERR_LOG_BYTES and
     * per-session disk near 2x - even when rotation fails and the file is kept.
     */
    fn open_stderr_log(&self) -> Option<StderrLog> {
        let path = self.options.stderr_log_path.clone()?;
        let path_buf = std::path::PathBuf::from(&path);
        let result = (|| -> std::io::Result<(std::fs::File, u64)> {
            if let Some(parent) = path_buf.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                let mut builder = std::fs::DirBuilder::new();
                builder.recursive(true);
                #[cfg(unix)] {
                    use std::os::unix::fs::DirBuilderExt;
                    builder.mode(0o700);
                }
                builder.create(parent)?;
                #[cfg(unix)]
                tighten_log_permissions(parent, 0o700)?;
            }
            #[cfg(unix)] {
                tighten_log_permissions(&path_buf, 0o600)?;
                tighten_log_permissions(std::path::Path::new(&format!("{path}.old")), 0o600)?;
            }
            let mut size = match std::fs::metadata(&path_buf) {
                Ok(metadata) => metadata.len(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                Err(error) => return Err(error),
            };
            if size > MAX_KERNEL_STDERR_LOG_BYTES {
                // Drop any prior .old first: rename fails on Windows if it exists.
                let _ = std::fs::remove_file(format!("{path}.old"));
                if let Err(error) = std::fs::rename(&path_buf, format!("{path}.old")) {
                    // A failed rotation must not cost the log: keep appending instead
                    // (repl-manager.ts:308-311). The budget is the file's remaining
                    // capacity, so the oversized file writes the budget marker once.
                    self.append_kernel_diagnostic(&format!(
                        "cannot rotate kernel stderr log: {}",
                        error_message(&KernelError::new(error.to_string()))
                    ));
                } else {
                    size = 0;
                }
            }
            let mut options = std::fs::OpenOptions::new();
            options.append(true).create(true);
            #[cfg(unix)] {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            // Windows logs inherit the owning profile directory's ACL.
            let file = options.open(&path_buf)?;
            Ok((file, size))
        })();
        match result {
            Ok((file, size)) => Some(StderrLog {
                file,
                budget: MAX_KERNEL_STDERR_LOG_BYTES.saturating_sub(size),
            }),
            Err(error) => {
                self.append_kernel_diagnostic(&format!(
                    "cannot open kernel stderr log: {}",
                    error_message(&KernelError::new(error.to_string()))
                ));
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

impl KernelState {
    fn client(&self) -> Option<Arc<dyn KernelClient>> {
        self.client_weak
            .lock()
            .unwrap()
            .upgrade()
            .map(|manager| manager as Arc<dyn KernelClient>)
    }

    fn add_live_kernels(&self) {
        if let Some(client) = self.client() {
            live_kernels_add(client);
        }
    }

    fn delete_live_kernels(&self) {
        if let Some(client) = self.client() {
            live_kernels_delete(&client);
        }
    }

    fn is_current_child(&self, child: &Arc<ChildState>) -> bool {
        match self.child.lock().unwrap().as_ref() {
            Some(current) => Arc::ptr_eq(current, child),
            None => false,
        }
    }

    fn start_stale(&self, generation: u64) -> bool {
        generation != self.start_generation.load(Ordering::SeqCst)
    }

    fn active_execution(&self) -> Option<Arc<Mutex<ActiveExecution>>> {
        self.active_execution.lock().unwrap().clone()
    }

    /// `start(options)`.
    async fn start_with_options(
        self: &Arc<Self>,
        options: KernelStartOptions,
    ) -> Result<(), KernelError> {
        if is_aborted(&options.signal) {
            return Err(create_kernel_startup_abort_error());
        }
        let existing = self.start_promise.lock().unwrap().clone();
        let start_promise = match existing {
            Some(promise) => promise,
            None => {
                let promise = Arc::new(SharedPromise::<()>::new());
                *self.start_promise.lock().unwrap() = Some(promise.clone());
                let this = self.clone();
                let task_promise = promise.clone();
                let on_progress = options.on_bootstrap_progress.clone();
                tokio::spawn(async move {
                    let outcome = this
                        .do_start(KernelStartOptions {
                            on_bootstrap_progress: on_progress,
                            signal: None,
                        })
                        .await;
                    task_promise.settle(outcome.clone());
                    if outcome.is_err() {
                        // Only clear our own memoization: a stale start must not evict a newer one.
                        let mut guard = this.start_promise.lock().unwrap();
                        let ours = guard
                            .as_ref()
                            .map(|current| Arc::ptr_eq(current, &task_promise))
                            .unwrap_or(false);
                        if ours {
                            *guard = None;
                        }
                    }
                });
                promise
            }
        };
        race_startup_with_abort(start_promise.wait(), options.signal).await?;
        Ok(())
    }

    /// `start()` with default options.
    async fn start_default(self: &Arc<Self>) -> Result<(), KernelError> {
        self.start_with_options(KernelStartOptions::default()).await
    }

    async fn do_start(
        self: &Arc<Self>,
        start_options: KernelStartOptions,
    ) -> Result<(), KernelError> {
        if self.state() != State::Idle {
            return Ok(());
        }
        let generation = self.start_generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.set_state(State::Starting);
        install_signal_handlers_once();
        // Tracked from the moment startup begins so session cleanup and signal
        // handlers can dispose a kernel that is still booting.
        self.add_live_kernels();

        let configured_python = self.options.python.lock().unwrap().clone();
        let python = match configured_python {
            Some(python) => python,
            None => {
                let ensure = ensure_kernel_python(EnsureKernelPythonOptions {
                    python_skills: self.options.python_skills.clone(),
                    on_progress: start_options.on_bootstrap_progress.clone(),
                })
                .await;
                match ensure {
                    Ok(python) => {
                        if self.start_stale(generation) {
                            return Err(KernelError::new("Kernel start superseded"));
                        }
                        *self.options.python.lock().unwrap() = Some(python.clone());
                        python
                    }
                    Err(error) => {
                        // Never touch a newer start's state.
                        if self.start_stale(generation) {
                            return Err(error);
                        }
                        self.delete_live_kernels();
                        if self.state() != State::Shutdown {
                            self.set_state(State::Idle);
                        }
                        return Err(error);
                    }
                }
            }
        };

        if self.state() == State::Shutdown {
            return Err(KernelError::new("Kernel was disposed during startup"));
        }

        // bash.py journals its process groups under this pid so the host can
        // reap them if the runtime dies without running its shutdown hook.
        let mut env: Vec<(String, String)> = std::env::vars().collect();
        if let Some(extra) = &self.options.env {
            for (key, value) in extra {
                set_env_var(&mut env, key, value);
            }
        }
        if cfg!(windows) {
            set_env_var(&mut env, "PYTHONUTF8", "1");
        }
        set_env_var(
            &mut env,
            "PRIME_AGENT_KERNEL_OWNER_PID",
            &std::process::id().to_string(),
        );

        let spawned = spawn_hidden(
            &python,
            &["-m".to_string(), "rlm.repl".to_string()],
            SpawnOptions {
                cwd: self.options.cwd.clone(),
                env: Some(env),
                // stdio: ["pipe", "pipe", "pipe"]
                stdin_piped: true,
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
        );
        let mut child = match spawned {
            Ok(child) => child.child,
            Err(error) => {
                let error = KernelError::new(error.to_string());
                self.handle_child_error(&error);
                return Err(self.fail_start(error, generation).await);
            }
        };

        let id = self.next_child_id.fetch_add(1, Ordering::SeqCst);
        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdin = child.stdin.take();
        let child_state = Arc::new(ChildState {
            id,
            pid,
            stdin: Arc::new(tokio::sync::Mutex::new(stdin)),
            stdin_destroyed: AtomicBool::new(false),
            exit: ExitState::new(),
            destroy_stdout: Latch::new(),
            destroy_stderr: Latch::new(),
            stderr_closed: Latch::new(),
        });
        *self.child.lock().unwrap() = Some(child_state.clone());
        if let Some(pid) = pid {
            record_orphan_process_state(pid as i64, true);
        }
        *self.ready_deferred.lock().unwrap() = Some(Arc::new(SharedPromise::<f64>::new()));
        *self.startup_protocol_error.lock().unwrap() = None;
        self.wire_child(child_state.clone(), child, stdout, stderr);

        let ready = self.wait_for_ready(&child_state).await;
        let protocol = match ready {
            Ok(protocol) => {
                if self.start_stale(generation) {
                    return Err(KernelError::new("Kernel start superseded"));
                }
                // Ready and a corrupt frame can share one stdout chunk: ready resolved the
                // deferred synchronously before the corruption was parsed, so the rejection
                // in failProtocolFrame was a no-op. Never mark such a child running.
                // The guard must be dropped before the await below, so clone first.
                let startup_protocol_error = self.startup_protocol_error.lock().unwrap().clone();
                if let Some(error) = startup_protocol_error {
                    return Err(self.fail_start(error, generation).await);
                }
                if protocol != REPL_PROTOCOL_VERSION {
                    let error = KernelError::new(format!(
                        "Kernel runtime speaks protocol {protocol}, expected {REPL_PROTOCOL_VERSION}. \
                         Update prime-agent-runtime in the kernel Python (PRIME_AGENT_KERNEL_PYTHON) to match this prime-agent."
                    ));
                    return Err(self.fail_start(error, generation).await);
                }
                protocol
            }
            Err(error) => return Err(self.fail_start(error, generation).await),
        };
        let _ = protocol;

        self.set_state(State::Running);
        Ok(())
    }

    /// The `catch` around startup: only the call that performed the cleanup may
    /// resurrect to idle; a concurrent kill()/teardown owns the state otherwise.
    async fn fail_start(self: &Arc<Self>, error: KernelError, generation: u64) -> KernelError {
        if self.start_stale(generation) {
            return error; // never tear down a newer start's kernel
        }
        let can_retry_startup = self.state() != State::Shutdown;
        let performed = self
            .shutdown(KernelShutdownOptions::default())
            .await
            .unwrap_or(false);
        if performed && can_retry_startup {
            self.set_state(State::Idle);
        }
        error
    }

    /// `child.on("error", ...)`.
    fn handle_child_error(self: &Arc<Self>, error: &KernelError) {
        self.append_kernel_diagnostic(&format!("spawn error: {}", error_message(error)));
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        // Fail a pending start() promptly instead of letting it ride out the
        // ready timeout. cleanupResources clears readyDeferred, so reject first;
        // a late error after ready resolved is a no-op on the settled promise.
        let ready = self.ready_deferred.lock().unwrap().clone();
        if let Some(ready) = ready {
            ready.settle(Err(error.clone()));
        }
        self.cleanup_resources(Some(Signal::Term));
    }

    /// `child.on("exit", ...)`.
    async fn handle_child_exit(self: &Arc<Self>, child_state: &Arc<ChildState>) {
        if !self.is_current_child(child_state) {
            return;
        }
        if self.state() != State::Shutdown {
            let code = node_option_string(&child_state.exit.exit_code());
            let signal = child_state
                .exit
                .signal_code()
                .unwrap_or_else(|| "null".to_string());
            self.append_kernel_diagnostic(&format!("unexpected exit code={code} signal={signal}"));
        }
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        // This exit is part of an in-flight graceful shutdown(): that call owns the
        // teardown and runs cleanupResources itself. Cleaning up here would bump the
        // generation and misread the owning shutdown as superseded.
        let graceful = *self.graceful_shutdown_generation.lock().unwrap();
        if graceful == Some(self.start_generation.load(Ordering::SeqCst)) {
            return;
        }
        self.cleanup_resources(Some(Signal::Term));
    }
}

// ---------------------------------------------------------------------------
// Protocol repair
// ---------------------------------------------------------------------------

impl KernelState {
    fn fail_protocol_frame(self: &Arc<Self>, child: &Arc<ChildState>, diagnostic: &str) {
        if !self.is_current_child(child) {
            return;
        }
        self.append_kernel_diagnostic(diagnostic);
        let error = KernelError::new(format!("Kernel protocol error: {diagnostic}"));
        if self.state() == State::Starting {
            *self.startup_protocol_error.lock().unwrap() = Some(error.clone());
        }
        let ready = self.ready_deferred.lock().unwrap().clone();
        if let Some(ready) = ready {
            ready.settle(Err(error.clone()));
        }
        self.reject_active_execution(error);
        if self.teardown_in_flight.load(Ordering::SeqCst) > 0 || self.state() != State::Running {
            return;
        }

        if self.protocol_repair_owner.lock().unwrap().is_some() {
            // A repair's own replacement child corrupted: discard it instead of respawn-looping.
            self.append_kernel_diagnostic(
                "replacement kernel corrupted during protocol repair; giving up",
            );
            if let Some(owner) = self.protocol_repair_owner.lock().unwrap().clone() {
                owner.superseded.store(true, Ordering::SeqCst);
            }
            // performRestore clears pendingRestore, so it still being set means the
            // corruption struck at or before the restore phase: the snapshot stays
            // the prime suspect (ambiguous attribution, loop-safe - retrying it
            // would re-trigger the corruption). Corruption strictly after a
            // successful restore never implicates the snapshot; keeping the flag
            // costs at most one bounded restore per later attempt.
            let snapshot_suspect = self.pending_restore.load(Ordering::SeqCst);
            self.kill_child_to_idle();
            if snapshot_suspect {
                self.pending_restore.store(false, Ordering::SeqCst);
            }
            return;
        }
        let owner = ProtocolRepairOwner::new();
        *self.protocol_repair_owner.lock().unwrap() = Some(owner.clone());
        let repair = Arc::new(SharedPromise::<()>::new());
        *self.protocol_repair_promise.lock().unwrap() = Some(repair.clone());
        let this = self.clone();
        let child = child.clone();
        tokio::spawn(async move {
            let outcome = this.repair_protocol_child(&child, &owner).await;
            if let Err(error) = outcome {
                this.append_kernel_diagnostic(&format!(
                    "protocol repair failed: {}",
                    error_message(&error)
                ));
            }
            repair.settle(Ok(()));
            let mut guard = this.protocol_repair_promise.lock().unwrap();
            let ours = guard
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &repair))
                .unwrap_or(false);
            if ours {
                *guard = None;
            }
            let mut owner_guard = this.protocol_repair_owner.lock().unwrap();
            let ours = owner_guard
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &owner))
                .unwrap_or(false);
            if ours {
                *owner_guard = None;
            }
        });
    }

    async fn repair_protocol_child(
        self: &Arc<Self>,
        child: &Arc<ChildState>,
        owner: &Arc<ProtocolRepairOwner>,
    ) -> Result<(), KernelError> {
        if !self.is_current_child(child) || self.state() == State::Shutdown {
            return Ok(());
        }
        self.kill_child_to_idle();

        let start = {
            let this = self.clone();
            tokio::spawn(async move { this.start_default().await })
        };
        let generation = self.start_generation.load(Ordering::SeqCst);
        if let Ok(Err(error)) = start.await {
            self.finish_failed_protocol_repair(owner, Some(&error));
            return Ok(());
        }
        if self.start_stale(generation) || self.state() != State::Running {
            self.finish_failed_protocol_repair(owner, None);
            return Ok(());
        }

        let restored = self.perform_restore(true, KernelRestoreSource::Auto).await;
        if self.start_stale(generation) || self.state() != State::Running {
            self.finish_failed_protocol_repair(owner, None);
            return Ok(());
        }
        if self.options.snapshot.is_some() && restored.is_none() {
            if owner.is_superseded() || !self.is_protocol_repair_owner(owner) {
                return Ok(());
            }
            self.append_kernel_diagnostic(
                "protocol repair restore failed; discarding replacement kernel",
            );
            self.kill_child_to_idle();
            // The snapshot is the declared culprit; the lazy path must not retry it.
            self.pending_restore.store(false, Ordering::SeqCst);
            return Ok(());
        }

        // Restore revives only the user namespace; live handles (rlm, bash, skills)
        // come from the runtime bootstrap, so a repaired kernel must re-run it.
        let Some(code) = self.options.bootstrap_code.clone() else {
            return Ok(());
        };
        let bootstrapped = self.bootstrap_repaired_kernel(&code).await;
        if self.start_stale(generation) || self.state() != State::Running {
            self.finish_failed_protocol_repair(owner, None);
            return Ok(());
        }
        if !bootstrapped {
            if owner.is_superseded() || !self.is_protocol_repair_owner(owner) {
                return Ok(());
            }
            self.append_kernel_diagnostic(
                "protocol repair bootstrap failed; discarding replacement kernel",
            );
            self.kill_child_to_idle();
        }
        Ok(())
    }

    fn is_protocol_repair_owner(&self, owner: &Arc<ProtocolRepairOwner>) -> bool {
        match self.protocol_repair_owner.lock().unwrap().as_ref() {
            Some(current) => Arc::ptr_eq(current, owner),
            None => false,
        }
    }

    /// Bounded bootstrap of a repaired kernel; false when it failed. Never throws.
    async fn bootstrap_repaired_kernel(self: &Arc<Self>, code: &str) -> bool {
        let request = json!({ "type": "execute", "code": code });
        let opts = ExecuteOptions {
            internal: true,
            protocol_repair: true,
            ..Default::default()
        };
        let outcome = self
            .enqueue_request(&request, code, opts, Some(REPAIR_STEP_TIMEOUT_MS), None)
            .await;
        match outcome {
            Ok(result) => {
                if result.result.status != ExecuteStatus::Ok {
                    let detail = match (&result.result.error, result.result.status) {
                        (Some(error), _) if error.evalue.is_empty() => result.result.stderr.clone(),
                        (Some(error), _) => error.evalue.clone(),
                        (None, _) => result.result.stderr.clone(),
                    };
                    let how = if result.result.status == ExecuteStatus::Aborted {
                        "timed out"
                    } else {
                        "failed"
                    };
                    self.append_kernel_diagnostic(&format!(
                        "protocol repair bootstrap {how}: {detail}"
                    ));
                    return false;
                }
                self.pending_rebootstrap.store(false, Ordering::SeqCst);
                true
            }
            Err(error) => {
                self.append_kernel_diagnostic(&format!(
                    "protocol repair bootstrap error: {}",
                    error_message(&error)
                ));
                false
            }
        }
    }

    /**
     * A fresh kernel started after a discarded repair has none of the runtime
     * bootstrap's live handles (rlm, bash, skills) and an empty namespace:
     * reprovision (restore, then bootstrap) before any user request. A failed
     * re-bootstrap discards the kernel again instead of serving user code on an
     * unprovisioned namespace.
     */
    /// `ensureKernelRebootstrapped(signal)`.
    ///
    /// Returns a boxed future, not an `async fn`: the re-bootstrap path is
    /// mutually recursive with `enqueueRequest` (the bootstrap cell goes back
    /// through the queue), and a recursive `async fn` would have to be boxed at
    /// the call site anyway. Boxing here states the obligation once, exactly like
    /// the TypeScript, whose promise type carries no layout recursion.
    fn ensure_kernel_rebootstrapped(
        self: &Arc<Self>,
        signal: &Option<AbortSignal>,
    ) -> pi_ai::types::BoxFuture<Result<(), KernelError>> {
        let this = self.clone();
        let signal = signal.clone();
        Box::pin(async move { this.ensure_kernel_rebootstrapped_body(&signal).await })
    }

    async fn ensure_kernel_rebootstrapped_body(
        self: &Arc<Self>,
        signal: &Option<AbortSignal>,
    ) -> Result<(), KernelError> {
        let code = self.options.bootstrap_code.clone();
        let needs_restore =
            self.options.snapshot.is_some() && self.pending_restore.load(Ordering::SeqCst);
        let needs_bootstrap = code.is_some() && self.pending_rebootstrap.load(Ordering::SeqCst);
        // An in-flight repair owns its kernel's restore/bootstrap sequence, and
        // a teardown's final snapshot must never trigger reprovisioning.
        if (!needs_restore && !needs_bootstrap)
            || self.protocol_repair_promise.lock().unwrap().is_some()
            || self.teardown_in_flight.load(Ordering::SeqCst) > 0
            || self.state() != State::Running
        {
            return Ok(());
        }
        let task = {
            let existing = self.rebootstrap_promise.lock().unwrap().clone();
            match existing {
                Some(task) => task,
                None => {
                    let task = Arc::new(SharedPromise::<bool>::new());
                    *self.rebootstrap_promise.lock().unwrap() = Some(task.clone());
                    let this = self.clone();
                    let started = task.clone();
                    tokio::spawn(async move {
                        let ok = this.reprovision_fresh_kernel(code).await;
                        started.settle(Ok(ok));
                        let mut guard = this.rebootstrap_promise.lock().unwrap();
                        let ours = guard
                            .as_ref()
                            .map(|current| Arc::ptr_eq(current, &started))
                            .unwrap_or(false);
                        if ours {
                            *guard = None;
                        }
                    });
                    task
                }
            }
        };
        // An aborted request never executes, so it may skip the wait; race the
        // signal like waitForProtocolRepair does instead of riding out the
        // bootstrap bound after a mid-wait abort.
        if let Some(signal) = signal {
            if signal.is_aborted() {
                return Ok(());
            }
            tokio::select! {
                biased;
                _ = wait_for_abort(Some(signal.clone())) => return Ok(()),
                _ = task.wait() => {}
            }
            if signal.is_aborted() {
                return Ok(());
            }
        }
        let ok = task.wait().await.unwrap_or(false); // bounded by REPAIR_STEP_TIMEOUT_MS
        if !ok {
            return Err(KernelError::new(
                "Kernel bootstrap failed after protocol repair",
            ));
        }
        Ok(())
    }

    /// Restore (one-shot, best-effort) then bootstrap the lazily started fresh kernel.
    async fn reprovision_fresh_kernel(self: &Arc<Self>, code: Option<String>) -> bool {
        if self.options.snapshot.is_some() && self.pending_restore.load(Ordering::SeqCst) {
            let _ = self.perform_restore(true, KernelRestoreSource::Auto).await; // clears pendingRestore on success
                                                                                 // Corrupted during the restore: the spawned repair owns the kernel now.
            if self.protocol_repair_promise.lock().unwrap().is_some()
                || self.state() != State::Running
            {
                return false;
            }
            // One attempt per discard: a clean restore failure falls back to an
            // empty namespace (ordinary startup semantics), never a retry loop.
            self.pending_restore.store(false, Ordering::SeqCst);
        }
        let Some(code) = code else {
            return true;
        };
        if !self.pending_rebootstrap.load(Ordering::SeqCst) {
            return true;
        }
        let ok = self.bootstrap_repaired_kernel(&code).await;
        if !ok && self.state() == State::Running {
            self.kill_child_to_idle();
        }
        ok
    }

    /// Kill the current child and settle at clean idle, so the next start spawns fresh.
    fn kill_child_to_idle(self: &Arc<Self>) {
        // The discarded kernel carried the runtime bootstrap and (possibly) the
        // restored namespace; a lazily started replacement must reprovision both.
        self.pending_rebootstrap.store(true, Ordering::SeqCst);
        self.pending_restore.store(true, Ordering::SeqCst);
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        self.cleanup_resources(Some(Signal::Kill));
        self.set_state(State::Idle);
    }

    fn finish_failed_protocol_repair(
        &self,
        owner: &Arc<ProtocolRepairOwner>,
        error: Option<&KernelError>,
    ) {
        if let Some(error) = error {
            self.append_kernel_diagnostic(&format!(
                "protocol repair start failed: {}",
                error_message(error)
            ));
        }
        if owner.is_superseded() || !self.is_protocol_repair_owner(owner) {
            return;
        }
        if self.state() == State::Shutdown {
            self.set_state(State::Idle);
        }
    }

    fn supersede_protocol_repair(&self) {
        if let Some(owner) = self.protocol_repair_owner.lock().unwrap().as_ref() {
            owner.superseded.store(true, Ordering::SeqCst);
        }
    }

    /// Wait until no protocol repair is pending; resolves early when the signal aborts.
    async fn wait_for_protocol_repair(&self, signal: &Option<AbortSignal>) {
        loop {
            if is_aborted(signal) {
                return;
            }
            let repair = self.protocol_repair_promise.lock().unwrap().clone();
            let Some(repair) = repair else {
                return;
            };
            match signal {
                // A rejected repair still clears the holder, and the loop re-reads it.
                None => {
                    let _ = repair.wait().await;
                }
                Some(signal) => {
                    tokio::select! {
                        biased;
                        _ = wait_for_abort(Some(signal.clone())) => return,
                        _ = repair.wait() => {}
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Child wiring and event decoding
// ---------------------------------------------------------------------------

impl KernelState {
    #[allow(clippy::needless_pass_by_value)]
    fn wire_child(
        self: &Arc<Self>,
        child: Arc<ChildState>,
        mut process: tokio::process::Child,
        stdout: Option<tokio::process::ChildStdout>,
        stderr: Option<tokio::process::ChildStderr>,
    ) {
        if let Some(stdout) = stdout {
            let this = self.clone();
            let child_state = child.clone();
            tokio::spawn(async move {
                let mut reader = stdout;
                let mut buffered = String::new();
                let mut pending: Vec<u8> = Vec::new();
                let mut chunk = vec![0u8; STREAM_CHUNK_BYTES];
                loop {
                    let read = tokio::select! {
                        biased;
                        _ = child_state.destroy_stdout.wait() => break,
                        result = reader.read(&mut chunk) => {
                            match result {
                                Ok(0) => break,
                                Ok(read) => read,
                                Err(_) => break,
                            }
                        }
                    };
                    if !this.is_current_child(&child_state) {
                        return;
                    }
                    buffered.push_str(&decode_utf8_chunk(&mut pending, &chunk[..read]));
                    loop {
                        if first_protocol_frame_too_large(&buffered) {
                            this.fail_protocol_frame(&child_state,
                                "kernel protocol frame exceeds 32 MiB; output was rejected");
                            return;
                        }
                        let Some(newline) = buffered.find('\n') else { break; };
                        if !this.is_current_child(&child_state) {
                            return;
                        }
                        let line = buffered[..newline].to_string();
                        buffered = buffered[newline + 1..].to_string();
                        if line.trim().is_empty() {
                            continue;
                        }
                        let event: Value = match serde_json::from_str(&line) {
                            Ok(event) => event,
                            Err(_) => {
                                this.fail_protocol_frame(
                                    &child_state,
                                    &format!(
                                        "unparseable protocol line: {}",
                                        truncate_chars(&line, 200)
                                    ),
                                );
                                return;
                            }
                        };
                        let Some(object) = event.as_object() else {
                            this.fail_protocol_frame(
                                &child_state,
                                &format!(
                                    "non-object protocol line: {}",
                                    truncate_chars(&line, 200)
                                ),
                            );
                            return;
                        };
                        if let Some(reason) = invalid_protocol_frame_reason(object) {
                            this.fail_protocol_frame(
                                &child_state,
                                &format!("{reason}: {}", truncate_chars(&line, 200)),
                            );
                            return;
                        }
                        if this.handle_event(object).is_err() {
                            return;
                        }
                    }
                }
            });
        }

        // The runtime dup2's fd 2 into its protocol pump before ready (repl.py
        // _setup_fds), so this pipe only ever carries pre-ready bytes; the write
        // budget caps what lands on disk, and once it is spent the handler keeps
        // draining but discards (a blocked pipe would wedge a pre-ready kernel).
        if let Some(stderr) = stderr {
            let this = self.clone();
            let child_state = child.clone();
            tokio::spawn(async move {
                let mut reader = stderr;
                let mut pending: Vec<u8> = Vec::new();
                let mut chunk = vec![0u8; STREAM_CHUNK_BYTES];
                let mut stderr_log = this.open_stderr_log();
                loop {
                    let read = tokio::select! {
                        biased;
                        _ = child_state.destroy_stderr.wait() => break,
                        result = reader.read(&mut chunk) => {
                            match result {
                                Ok(0) => break,
                                Ok(read) => read,
                                Err(_) => break,
                            }
                        }
                    };
                    let bytes = &chunk[..read];
                    let text = decode_utf8_chunk(&mut pending, bytes);
                    this.append_kernel_stderr_text(&text);
                    let Some(log) = stderr_log.as_mut() else {
                        continue;
                    };
                    if log.budget == STDERR_LOG_SPENT {
                        continue;
                    }
                    let result = (|| -> std::io::Result<()> {
                        if bytes.len() as u64 <= log.budget {
                            write_fully_sync(&mut log.file, bytes)?;
                            log.budget -= bytes.len() as u64;
                            Ok(())
                        } else {
                            write_fully_sync(
                                &mut log.file,
                                KERNEL_STDERR_LOG_BUDGET_MARKER.as_bytes(),
                            )?;
                            log.budget = STDERR_LOG_SPENT; // stderrLogWritable = false
                            Ok(())
                        }
                    })();
                    if let Err(error) = result {
                        stderr_log = None;
                        this.append_kernel_diagnostic(&format!(
                            "kernel stderr log write failed: {}",
                            error_message(&KernelError::new(error.to_string()))
                        ));
                    }
                }
                // A kernel that dies mid-character leaves bytes buffered in the decoder;
                // flush them so the tail keeps the truncated final character. Both events,
                // because each can be the only one to precede the tail build: 'end' beats
                // 'exit' on natural EOF (whose 'close' emission can land after it), while
                // a drain-destroyed stream skips 'end'. The second end() returns "".
                this.append_kernel_stderr_text(&flush_utf8_pending(&mut pending));
                this.append_kernel_stderr_text(&flush_utf8_pending(&mut pending));
                drop(stderr_log.take());
                child_state.stderr_closed.settle();
            });
        }

        // `child.once("exit", ...)`: one turn for the poll phase to deliver the
        // bytes the kernel wrote before dying, then destroy the stderr stream.
        {
            let this = self.clone();
            let child_state = child.clone();
            tokio::spawn(async move {
                let status = process.wait().await;
                match status {
                    Ok(status) => {
                        let signal = node_signal_name(&status);
                        child_state.exit.settle(status.code(), signal);
                    }
                    Err(_) => child_state.exit.settle(None, None),
                }
                // One turn for the poll phase to deliver the bytes the kernel wrote
                // before dying (the pipe buffer bounds them), then destroy: EOF may
                // never come, and anything later is a surviving grandchild's
                // post-mortem noise, not the kernel's last words.
                tokio::task::yield_now().await;
                child_state.destroy_stderr.settle();
                this.handle_child_exit(&child_state).await;
            });
        }
    }

    /// `waitForReady(child)`.
    async fn wait_for_ready(self: &Arc<Self>, child: &Arc<ChildState>) -> Result<f64, KernelError> {
        let ready = self.ready_deferred.lock().unwrap().clone();
        let Some(ready) = ready else {
            return Err(KernelError::new("Kernel ready state is missing"));
        };
        let exit_wait = async {
            // The final stderr chunks can still be in flight at 'exit'; wait for
            // the drained pipe so the tail includes the kernel's last words (the
            // ready timeout stays armed, bounding the wait).
            child.exit.wait().await;
            child.wait_for_stderr_close().await;
            let tail = tail_chars(
                &self.kernel_stderr.lock().unwrap(),
                STARTUP_STDERR_TAIL_CHARS,
            );
            let tail = if tail.is_empty() {
                "(empty)".to_string()
            } else {
                tail
            };
            Err(KernelError::new(format!(
                "Kernel exited before ready. stderr:\n{tail}"
            )))
        };
        let ready_wait = ready.wait();
        let timeout = async {
            tokio::time::sleep(Duration::from_millis(READY_TIMEOUT_MS)).await;
            let tail = tail_chars(
                &self.kernel_stderr.lock().unwrap(),
                STARTUP_STDERR_TAIL_CHARS,
            );
            let tail = if tail.is_empty() {
                "(empty)".to_string()
            } else {
                tail
            };
            Err(KernelError::new(format!(
                "Kernel did not become ready within {READY_TIMEOUT_MS}ms. stderr tail:\n{tail}"
            )))
        };
        let already_exited = child.exit.has_exited();
        if already_exited {
            return exit_wait.await;
        }
        tokio::select! {
            biased;
            result = ready_wait => result,
            result = exit_wait => result,
            result = timeout => result,
        }
    }

    /// Write one JSON-lines request frame; resolves when the OS accepted the bytes.
    async fn write_line(&self, request: &Map<String, Value>) -> Result<(), KernelError> {
        let child = self.child.lock().unwrap().clone();
        let Some(child) = child else {
            return Err(KernelError::new("Kernel stdin is not connected"));
        };
        if child.stdin_destroyed.load(Ordering::SeqCst) {
            return Err(KernelError::new("Kernel stdin is not connected"));
        }
        let text = format!("{}\n", Value::Object(request.clone()));
        let mut guard = child.stdin.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(KernelError::new("Kernel stdin is not connected"));
        };
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|error| KernelError::new(error.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|error| KernelError::new(error.to_string()))?;
        Ok(())
    }

    /// `handleEvent(event)`. Returns Err when the caller must stop pumping.
    fn handle_event(self: &Arc<Self>, event: &Map<String, Value>) -> Result<(), ()> {
        let kind = event
            .get("event")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if kind == "display" {
            if let Some(data) = event.get("data").filter(|value| value.is_object()) {
                if let Some(activity) = data.get(BASH_ACTIVITY_DISPLAY_MIME) {
                    if let Some(activity) = activity.as_object() {
                        let id = activity.get("id").and_then(|value| value.as_str());
                        let pid = activity.get("pid").and_then(|value| value.as_f64());
                        let active = activity.get("active").and_then(|value| value.as_bool());
                        let valid = id.map(is_bash_activity_id).unwrap_or(false)
                            && pid
                                .map(|pid| pid.fract() == 0.0 && pid > 0.0)
                                .unwrap_or(false)
                            && active.is_some();
                        if valid {
                            let id = id.unwrap().to_string();
                            let pid = pid.unwrap();
                            let active = active.unwrap();
                            if active {
                                let mut handles = self.background_bash_handles.lock().unwrap();
                                if !handles.iter().any(|(existing, _)| *existing == id) {
                                    handles.push((id, pid));
                                }
                            } else {
                                let mut handles = self.background_bash_handles.lock().unwrap();
                                let had_work = !handles.is_empty();
                                if handles
                                    .iter()
                                    .find(|(existing, _)| *existing == id)
                                    .map(|(_, existing_pid)| *existing_pid == pid)
                                    .unwrap_or(false)
                                {
                                    ordered_delete(&mut handles, &id);
                                }
                                let settled = had_work && handles.is_empty();
                                drop(handles);
                                if settled {
                                    self.notify_background_work_settled();
                                }
                            }
                        }
                    }
                    return Ok(());
                }
            }
        }
        if kind == "ready" {
            let formats = as_string_array_field(event.get("snapshotFormats"));
            *self.runtime_snapshot_formats.lock().unwrap() = formats;
            let protocol = event
                .get("protocol")
                .and_then(|value| value.as_f64())
                .unwrap_or(-1.0);
            let ready = self.ready_deferred.lock().unwrap().clone();
            if let Some(ready) = ready {
                ready.settle(Ok(protocol));
            }
            return Ok(());
        }
        if kind == "host_request" {
            if let Some(id) = event.get("id").and_then(|value| value.as_str()) {
                self.start_host_request(id.to_string(), event.get("data").cloned());
            }
            return Ok(());
        }

        let id = event
            .get("id")
            .and_then(|value| value.as_str())
            .map(|id| id.to_string());
        let execution = self.active_execution();
        let matches = match (&execution, &id) {
            (Some(execution), Some(id)) => execution.lock().unwrap().request_id == *id,
            _ => false,
        };
        if !matches {
            let data = event.get("data").filter(|value| value.is_object());
            if kind == "display" {
                if let Some(data) = data {
                    self.dispatch_late_sent_agent_message(
                        id.as_deref(),
                        data.get(AGENT_MESSAGE_DISPLAY_MIME),
                    );
                }
            } else if kind == "stdout" || kind == "stderr" {
                // Unowned output (null id, or another cell's id): never merge it into
                // the active cell's streams; buffer it as background output instead.
                self.append_background_output(
                    event
                        .get("text")
                        .and_then(|value| value.as_str())
                        .unwrap_or(""),
                );
            } else if kind == "done" && id.is_some() {
                let waiter = {
                    let mut waiters = self.pending_done_waiters.lock().unwrap();
                    let id = id.as_deref().unwrap();
                    let index = waiters.iter().position(|(existing, _)| existing == id);
                    index.map(|index| waiters.remove(index).1)
                };
                if let Some(waiter) = waiter {
                    waiter.settle();
                }
            } else if kind == "error" && id.is_none() {
                let evalue = event.get("evalue").map(node_string).unwrap_or_default();
                self.append_kernel_diagnostic(&format!("protocol error: {evalue}"));
            }
            return Ok(());
        }

        let execution = execution.unwrap();
        let settled = execution.lock().unwrap().settled;
        if settled && kind == "display" {
            if let Some(data) = event.get("data").filter(|value| value.is_object()) {
                if self.dispatch_late_sent_agent_message(
                    id.as_deref(),
                    data.get(AGENT_MESSAGE_DISPLAY_MIME),
                ) {
                    return Ok(());
                }
            }
        }
        if kind == "stdout" || kind == "stderr" {
            let text = event
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let on_stream = {
                let mut guard = execution.lock().unwrap();
                if kind == "stdout" {
                    if guard.stdout.chars().count() < guard.max_chars {
                        guard.stdout.push_str(text);
                        if guard.stdout.chars().count() > guard.max_chars {
                            let trimmed = truncate_chars(&guard.stdout.clone(), guard.max_chars);
                            guard.stdout = trimmed;
                            guard.stdout_truncated = true;
                        }
                    }
                } else if guard.stderr.chars().count() < guard.max_chars {
                    guard.stderr.push_str(text);
                    if guard.stderr.chars().count() > guard.max_chars {
                        let trimmed = truncate_chars(&guard.stderr.clone(), guard.max_chars);
                        guard.stderr = trimmed;
                        guard.stderr_truncated = true;
                    }
                }
                guard.opts.on_stream.clone()
            };
            if let Some(on_stream) = on_stream {
                let name = if kind == "stdout" {
                    StreamName::Stdout
                } else {
                    StreamName::Stderr
                };
                let callback: &dyn Fn(&str, StreamName) = &*on_stream;
                callback(text, name);
            }
        } else if kind == "result" {
            if let Some(text) = event.get("text").and_then(|value| value.as_str()) {
                execution.lock().unwrap().result = Some(text.to_string());
            }
        } else if kind == "display" {
            let empty = Map::new();
            let data = event
                .get("data")
                .and_then(|value| value.as_object())
                .unwrap_or(&empty);
            let diff = parse_diff_display(data.get(DIFF_DISPLAY_MIME).unwrap_or(&Value::Null));
            let attachment =
                parse_attachment_display(data.get(ATTACHMENT_DISPLAY_MIME).unwrap_or(&Value::Null));
            let sent_agent_message = parse_sent_agent_message(
                data.get(AGENT_MESSAGE_DISPLAY_MIME).unwrap_or(&Value::Null),
            );
            let mut guard = execution.lock().unwrap();
            if let Some(diff) = diff {
                guard.diffs.push(diff);
            }
            match attachment {
                Some(ParsedAttachment::Oversized) => {
                    let separator = if guard.stderr.is_empty() { "" } else { "\n" };
                    guard.stderr.push_str(&format!(
                        "{separator}attachment dropped: exceeds {MAX_ATTACHMENT_DATA_CHARS} base64 chars"
                    ));
                    guard.status = ExecuteStatus::Error;
                }
                Some(ParsedAttachment::Attachment(attachment)) => {
                    guard.attachments.push(attachment)
                }
                None => {}
            }
            if let Some(message) = sent_agent_message {
                guard.sent_agent_messages.push(message);
            }
        } else if kind == "error" {
            let error = ExecError {
                ename: event
                    .get("ename")
                    .and_then(|value| value.as_str())
                    .unwrap_or("Error")
                    .to_string(),
                evalue: event
                    .get("evalue")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string(),
                traceback: as_string_array_field(event.get("traceback")),
            };
            let mut guard = execution.lock().unwrap();
            guard.error = Some(error);
            guard.status = ExecuteStatus::Error;
        } else if kind == "done" {
            let mut guard = execution.lock().unwrap();
            guard.done_fields = Some(event.clone());
            if event.get("status").and_then(|value| value.as_str()) != Some("ok")
                && guard.status == ExecuteStatus::Ok
            {
                guard.status = ExecuteStatus::Error;
                // State requests report failures as a done reason without an error event.
                if guard.error.is_none() {
                    if let Some(reason) = event.get("reason").and_then(|value| value.as_str()) {
                        guard.error = Some(ExecError {
                            ename: "KernelError".to_string(),
                            evalue: reason.to_string(),
                            traceback: Vec::new(),
                        });
                    }
                }
            }
            drop(guard);
            self.finish_active_execution(&execution);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Execution queue
// ---------------------------------------------------------------------------

impl KernelState {
    async fn execute(
        self: &Arc<Self>,
        code: &str,
        opts: ExecuteOptions,
    ) -> Result<ExecuteResult, KernelError> {
        self.wait_for_protocol_repair(&opts.signal).await;
        let result = self.enqueue_execute(code, opts, None).await?;
        // Refresh the on-disk snapshot after real work so a later resume (or a
        // crash before graceful shutdown) revives the most recent namespace.
        if result.result.status == ExecuteStatus::Ok {
            self.schedule_snapshot();
        }
        Ok(result.result)
    }

    /// Queue and run a cell, serializing against all other executions.
    async fn enqueue_execute(
        self: &Arc<Self>,
        code: &str,
        opts: ExecuteOptions,
        execution_timeout_ms: Option<u64>,
    ) -> Result<InternalExecuteResult, KernelError> {
        let request = json!({ "type": "execute", "code": code });
        self.enqueue_request(&request, code, opts, execution_timeout_ms, None)
            .await
    }

    /// Queue one protocol request (execute or state op) behind every other request.
    ///
    /// Boxed rather than a plain `async fn`: the queue path is mutually recursive
    /// (`enqueue_request` -> `execute_queued` -> `execute_in_queue_slot` ->
    /// `enqueue_request` when a repair starts while a request waits for its slot).
    /// A recursive `async fn` needs indirection to have a finite layout, and the
    /// same indirection is what lets the compiler prove the spawned bootstrap
    /// future `Send`. The TypeScript promise has neither problem.
    fn enqueue_request(
        self: &Arc<Self>,
        request_fields: &Value,
        code: &str,
        opts: ExecuteOptions,
        execution_timeout_ms: Option<u64>,
        snapshot_metric: Option<SnapshotMetricState>,
    ) -> pi_ai::types::BoxFuture<Result<InternalExecuteResult, KernelError>> {
        let this = self.clone();
        let request_fields = request_fields.clone();
        let code = code.to_string();
        Box::pin(async move {
            this.enqueue_request_body(&request_fields, &code, opts, execution_timeout_ms, snapshot_metric)
                .await
        })
    }

    async fn enqueue_request_body(
        self: &Arc<Self>,
        request_fields: &Value,
        code: &str,
        mut opts: ExecuteOptions,
        execution_timeout_ms: Option<u64>,
        mut snapshot_metric: Option<SnapshotMetricState>,
    ) -> Result<InternalExecuteResult, KernelError> {
        if is_aborted(&opts.signal) {
            return Ok(aborted_result(0.0));
        }
        self.start_with_options(KernelStartOptions {
            on_bootstrap_progress: None,
            signal: opts.signal.clone(),
        })
        .await?;
        if self.state() == State::Shutdown {
            return Err(KernelError::new("Kernel has been shut down"));
        }
        if self.flushing_snapshot_for_dispose.load(Ordering::SeqCst) && !opts.internal {
            return Err(KernelError::new("Kernel is shutting down"));
        }
        if !opts.protocol_repair {
            self.ensure_kernel_rebootstrapped(&opts.signal).await?;
        }
        // Aborted while waiting on the re-bootstrap: settle now instead of parking
        // on the queue slot behind the still-running bootstrap.
        if is_aborted(&opts.signal) {
            return Ok(aborted_result(0.0));
        }
        // Re-check: a final flush may have started while this request awaited the
        // lazy re-bootstrap; admitting it now would splice it between the flush's
        // captured queue and the final snapshot, unbounding the teardown.
        if self.flushing_snapshot_for_dispose.load(Ordering::SeqCst) && !opts.internal {
            return Err(KernelError::new("Kernel is shutting down"));
        }

        let prev = self.execution_queue.lock().unwrap().clone();
        if !opts.internal {
            // The tail still holds the predecessor: stamp it before replacing the slot.
            let mut tail = self.execution_queue_tail_snapshot_metric.lock().unwrap();
            if let Some(metric) = tail.as_mut() {
                let mut timing = metric.timing.lock().unwrap();
                if timing.following_cell_queued_at.is_none() {
                    timing.following_cell_queued_at = safe_metric_now(Some(&metric.recorder));
                }
            }
        }
        let next = Arc::new(SharedPromise::<()>::new());
        *self.execution_queue.lock().unwrap() = next.clone();
        *self.execution_queue_tail_snapshot_metric.lock().unwrap() = snapshot_metric.clone();
        prev.wait().await;
        if let Some(metric) = snapshot_metric.as_mut() {
            metric.timing.lock().unwrap().dequeued_at = safe_metric_now(Some(&metric.recorder));
        }

        let started = now_ms();
        self.execute_queued(
            request_fields,
            code,
            &mut opts,
            execution_timeout_ms,
            &mut snapshot_metric,
            started,
            next,
        )
        .await
    }

    /// The body of the queue slot: `try/finally` around the actual request.
    async fn execute_queued(
        self: &Arc<Self>,
        request_fields: &Value,
        code: &str,
        opts: &mut ExecuteOptions,
        execution_timeout_ms: Option<u64>,
        snapshot_metric: &mut Option<SnapshotMetricState>,
        started: f64,
        slot: Arc<SharedPromise<()>>,
    ) -> Result<InternalExecuteResult, KernelError> {
        let outcome = self
            .execute_in_queue_slot(
                request_fields,
                code,
                opts,
                execution_timeout_ms,
                snapshot_metric,
                started,
                &slot,
            )
            .await;
        // `finally`: clear the tail metric and release the queue slot.
        self.release_metric_tail(snapshot_metric);
        slot.settle(Ok(()));
        outcome
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_in_queue_slot(
        self: &Arc<Self>,
        request_fields: &Value,
        code: &str,
        opts: &mut ExecuteOptions,
        execution_timeout_ms: Option<u64>,
        snapshot_metric: &mut Option<SnapshotMetricState>,
        started: f64,
        slot: &Arc<SharedPromise<()>>,
    ) -> Result<InternalExecuteResult, KernelError> {
        self.wait_for_active_execution_to_clear_for_reuse(&opts.signal)
            .await?;
        if is_aborted(&opts.signal) {
            return Ok(aborted_result(now_ms() - started));
        }
        if self.state() == State::Shutdown {
            return Err(KernelError::new("Kernel has been shut down"));
        }
        // A repair started while this request was queued or busy-waiting: release
        // the slot so the repair's own restore can run, then requeue behind it.
        if self.protocol_repair_promise.lock().unwrap().is_some() && !opts.protocol_repair {
            let retried_snapshot_metric = snapshot_metric.clone();
            self.release_metric_tail(snapshot_metric);
            *snapshot_metric = None;
            slot.settle(Ok(()));
            self.wait_for_protocol_repair(&opts.signal).await;
            return self
                .enqueue_request(
                    request_fields,
                    code,
                    opts.clone(),
                    execution_timeout_ms,
                    retried_snapshot_metric,
                )
                .await;
        }
        let Some(execution_timeout_ms) = execution_timeout_ms else {
            return self
                .execute_inner(request_fields, code, opts, started)
                .await;
        };

        let controller = AbortSignal::new();

        // `globalThis.setTimeout(() => controller.abort(), executionTimeoutMs)`.
        let timer_target = controller.clone();
        let timer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(execution_timeout_ms)).await;
            timer_target.abort(None);
        });
        let signal = match &opts.signal {
            Some(signal) => AbortSignal::any(vec![signal.clone(), controller.clone()]),
            None => controller.clone(),
        };
        let mut timed_opts = opts.clone();
        timed_opts.signal = Some(signal);
        let outcome = self
            .execute_inner(request_fields, code, &timed_opts, started)
            .await;
        timer.abort();
        outcome
    }

    fn release_metric_tail(&self, snapshot_metric: &Option<SnapshotMetricState>) {
        if let Some(metric) = snapshot_metric {
            let mut tail = self.execution_queue_tail_snapshot_metric.lock().unwrap();
            if tail.as_ref().is_some_and(|tail| Arc::ptr_eq(&tail.timing, &metric.timing)) {
                *tail = None;
            }
        }
    }

    async fn execute_inner(
        self: &Arc<Self>,
        request_fields: &Value,
        code: &str,
        opts: &ExecuteOptions,
        started: f64,
    ) -> Result<InternalExecuteResult, KernelError> {
        let max_chars = opts.max_output_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS);
        let request_id = uuid_v4();

        if is_aborted(&opts.signal) {
            return Ok(aborted_result(now_ms() - started));
        }
        if self.active_execution().is_some() {
            return Err(KernelError::new("Kernel already has an active execution"));
        }

        let (resolver, receiver) = tokio::sync::oneshot::channel();
        let pending_background_output = self.pending_background_output.lock().unwrap().clone();
        let pending_truncated = self
            .pending_background_output_truncated
            .load(Ordering::SeqCst);
        let execution = Arc::new(Mutex::new(ActiveExecution {
            request_id: request_id.clone(),
            code: code.to_string(),
            started,
            max_chars,
            opts: opts.clone(),
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            result: None,
            diffs: Vec::new(),
            attachments: Vec::new(),
            sent_agent_messages: Vec::new(),
            background_output: pending_background_output,
            background_output_truncated: pending_truncated,
            error: None,
            status: ExecuteStatus::Ok,
            done_fields: None,
            settled: false,
            interrupt_requested_at: None,
            resolver: Some(resolver),
        }));
        *self.pending_background_output.lock().unwrap() = String::new();
        self.pending_background_output_truncated
            .store(false, Ordering::SeqCst);

        // `onAbort` + the KERNEL_ABORT_GRACE_MS force-abort timer.
        // `if (execution.interruptRequestedAt !== undefined) return;` guards a second call.
        let interrupt_requested = Arc::new(AtomicBool::new(false));
        let on_abort: Arc<dyn Fn() + Send + Sync> = {
            let interrupt_requested = interrupt_requested.clone();
            let this = self.clone();
            let execution = execution.clone();
            let grace_target = self.clone();
            let grace_execution = execution.clone();
            Arc::new(move || {
                if interrupt_requested.swap(true, Ordering::SeqCst) {
                    return;
                }
                {
                    let mut guard = execution.lock().unwrap();
                    if guard.interrupt_requested_at.is_some() {
                        return;
                    }
                    guard.interrupt_requested_at = Some(now_ms());
                }
                let interrupt_target = this.clone();
                tokio::spawn(async move {
                    let _ = interrupt_target.interrupt().await;
                });
                let target = grace_target.clone();
                let execution = grace_execution.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(KERNEL_ABORT_GRACE_MS)).await;
                    if !ActiveExecution::same_as(&target.active_execution(), &execution) {
                        return;
                    }
                    let mut guard = execution.lock().unwrap();
                    guard.status = ExecuteStatus::Aborted;
                    drop(guard);
                    // The execution stays active until its done event arrives; clearing it
                    // early would let a new cell race the interrupted one (see busy-after-interrupt).
                    target.resolve_execution(&execution, false);
                });
            })
        };
        let abort_listener = opts.signal.as_ref().map(|signal| {
            let on_abort = on_abort.clone();
            signal.add_listener(move || {
                let callback: &dyn Fn() = &*on_abort;
                callback();
            })
        });
        if is_aborted(&opts.signal) {
            let callback: &dyn Fn() = &*on_abort;
            callback();
        }

        *self.active_execution.lock().unwrap() = Some(execution.clone());
        if !opts.internal {
            *self.last_cell_code.lock().unwrap() = Some(code.to_string());
        }

        let mut request = request_fields.clone();
        if let Some(object) = request.as_object_mut() {
            object.insert("id".to_string(), Value::String(request_id));
        }

        let object = request.as_object().cloned().unwrap_or_default();
        // One future, polled twice: the race, then the confirming await that the
        // TypeScript does with the same (already resolved) sendPromise.
        let mut send = Box::pin(self.write_line(&object));
        let settled_flag = {
            let execution = execution.clone();
            async move {
                loop {
                    if execution.lock().unwrap().settled {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        };
        // `await Promise.race([sendPromise, result.promise.then(() => undefined)])`.
        let send_result = tokio::select! {
            biased;
            result = send.as_mut() => Some(result),
            _ = settled_flag => None,
        };
        match send_result {
            Some(Err(error)) => {
                if ActiveExecution::same_as(&self.active_execution(), &execution) {
                    *self.active_execution.lock().unwrap() = None;
                }
                if let Some(listener) = abort_listener {
                    listener.remove();
                }
                return Err(error);
            }
            Some(Ok(())) => {
                // The TypeScript re-awaits the already-resolved sendPromise here
                // (`if (this.activeExecution === execution && execution.status !== "aborted")`).
                // A completed future has nothing left to report, so the check only
                // documents the condition; awaiting it twice is not valid in Rust.
                let _await_again = ActiveExecution::same_as(&self.active_execution(), &execution)
                    && execution.lock().unwrap().status != ExecuteStatus::Aborted;
            }
            None => {}
        }

        let result = receiver.await;
        if let Some(listener) = abort_listener {
            listener.remove();
        }
        match result {
            Ok(outcome) => outcome,
            Err(_) => Err(KernelError::new(
                "Kernel execution was dropped without settling",
            )),
        }
    }

    fn append_background_output(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        let execution = self.active_execution();
        if let Some(execution) = execution {
            let mut guard = execution.lock().unwrap();
            if guard.background_output.chars().count() >= MAX_BACKGROUND_OUTPUT_CHARS {
                guard.background_output_truncated = true;
                return;
            }
            guard.background_output.push_str(text);
            if guard.background_output.chars().count() > MAX_BACKGROUND_OUTPUT_CHARS {
                let trimmed = truncate_chars(
                    &guard.background_output.clone(),
                    MAX_BACKGROUND_OUTPUT_CHARS,
                );
                guard.background_output = trimmed;
                guard.background_output_truncated = true;
            }
            return;
        }
        let mut guard = self.pending_background_output.lock().unwrap();
        if guard.chars().count() >= MAX_BACKGROUND_OUTPUT_CHARS {
            self.pending_background_output_truncated
                .store(true, Ordering::SeqCst);
            return;
        }
        guard.push_str(text);
        if guard.chars().count() > MAX_BACKGROUND_OUTPUT_CHARS {
            let trimmed = truncate_chars(&guard.clone(), MAX_BACKGROUND_OUTPUT_CHARS);
            *guard = trimmed;
            self.pending_background_output_truncated
                .store(true, Ordering::SeqCst);
        }
    }

    fn finish_active_execution(&self, execution: &Arc<Mutex<ActiveExecution>>) {
        if !ActiveExecution::same_as(&self.active_execution(), execution) {
            return;
        }
        self.resolve_execution(execution, true);
    }

    fn resolve_execution(&self, execution: &Arc<Mutex<ActiveExecution>>, clear_active: bool) {
        let did_clear_active =
            clear_active && ActiveExecution::same_as(&self.active_execution(), execution);
        if clear_active && ActiveExecution::same_as(&self.active_execution(), execution) {
            *self.active_execution.lock().unwrap() = None;
        }
        let mut guard = execution.lock().unwrap();
        if !guard.settled {
            guard.settled = true;
            if let Some(handler) = guard.opts.on_late_sent_agent_message.clone() {
                drop(guard);
                self.register_late_sent_agent_message_handler(
                    execution.lock().unwrap().request_id.clone(),
                    handler,
                );
                guard = execution.lock().unwrap();
            }

            let mut stdout = guard.stdout.clone();
            let mut stderr = guard.stderr.clone();
            let mut result = guard.result.clone();
            let mut status = guard.status;
            if guard.stdout_truncated {
                stdout.push_str(&format!(
                    "\n[... output truncated at {} chars ...]",
                    guard.max_chars
                ));
            }
            if guard.stderr_truncated {
                stderr.push_str(&format!(
                    "\n[... output truncated at {} chars ...]",
                    guard.max_chars
                ));
            }
            if let Some(text) = result.as_ref() {
                if text.chars().count() > guard.max_chars {
                    result = Some(format!(
                        "{}\n[... output truncated at {} chars ...]",
                        truncate_chars(text, guard.max_chars),
                        guard.max_chars
                    ));
                }
            }

            if is_aborted(&guard.opts.signal) {
                status = ExecuteStatus::Aborted;
            }

            let mut background_output = guard.background_output.clone();
            if guard.background_output_truncated {
                background_output.push_str(&format!(
                    "\n[... background output truncated at {MAX_BACKGROUND_OUTPUT_CHARS} chars ...]"
                ));
            }

            let execution_result = ExecuteResult {
                stdout,
                stderr,
                result,
                diffs: if guard.diffs.is_empty() {
                    None
                } else {
                    Some(guard.diffs.clone())
                },
                attachments: if guard.attachments.is_empty() {
                    None
                } else {
                    Some(guard.attachments.clone())
                },
                sent_agent_messages: if guard.sent_agent_messages.is_empty() {
                    None
                } else {
                    Some(guard.sent_agent_messages.clone())
                },
                background_output: if background_output.is_empty() {
                    None
                } else {
                    Some(background_output)
                },
                status,
                error: guard.error.clone(),
                duration_ms: now_ms() - guard.started,
            };
            let done_fields = guard.done_fields.clone();
            guard.resolve(InternalExecuteResult {
                result: execution_result,
                done_fields,
            });
        }
        drop(guard);
        if did_clear_active {
            self.notify_active_execution_idle();
        }
    }

    fn dispatch_late_sent_agent_message(
        &self,
        request_id: Option<&str>,
        value: Option<&Value>,
    ) -> bool {
        let null = Value::Null;
        let Some(sent_agent_message) = parse_sent_agent_message(value.unwrap_or(&null)) else {
            return false;
        };
        let Some(request_id) = request_id else {
            return false;
        };
        let handler = {
            let mut handlers = self.late_sent_agent_message_handlers.lock().unwrap();
            let index = handlers.iter().position(|(id, _)| id == request_id);
            match index {
                Some(index) => {
                    // `map.delete` then `map.set`: the touched key moves to the end.
                    let entry = handlers.remove(index);
                    handlers.push(entry.clone());
                    Some(entry.1)
                }
                None => None,
            }
        };
        let Some(handler) = handler else {
            return false;
        };
        let callback: &dyn Fn(KernelSentAgentMessage) = &*handler;
        callback(sent_agent_message);
        true
    }

    fn register_late_sent_agent_message_handler(
        &self,
        request_id: String,
        handler: Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>,
    ) {
        let mut handlers = self.late_sent_agent_message_handlers.lock().unwrap();
        ordered_set(&mut handlers, request_id, handler);
        while handlers.len() > MAX_LATE_SENT_AGENT_MESSAGE_HANDLERS {
            if handlers.is_empty() {
                break;
            }
            handlers.remove(0);
        }
    }

    fn reject_active_execution(&self, error: KernelError) {
        let execution = self.active_execution();
        let Some(execution) = execution else {
            return;
        };
        *self.active_execution.lock().unwrap() = None;
        execution.lock().unwrap().reject(error);
        self.notify_active_execution_idle();
    }

    fn notify_active_execution_idle(&self) {
        self.active_execution_idle_epoch
            .fetch_add(1, Ordering::SeqCst);
        self.active_execution_idle.notify_waiters();
    }

    /// `waitForActiveExecutionToClear(signal, timeoutMs)`.
    async fn wait_for_active_execution_to_clear(
        &self,
        signal: &Option<AbortSignal>,
        timeout_ms: u64,
    ) -> bool {
        let epoch = self.active_execution_idle_epoch.load(Ordering::SeqCst);
        if self.active_execution().is_none() {
            return true;
        }
        let notified = self.active_execution_idle.notified();
        if self.active_execution_idle_epoch.load(Ordering::SeqCst) != epoch {
            return true;
        }
        let timeout = tokio::time::sleep(Duration::from_millis(timeout_ms));
        tokio::select! {
            biased;
            _ = notified => true,
            _ = wait_for_abort(signal.clone()) => false,
            _ = timeout => false,
        }
    }

    async fn reconcile_settled_active_execution(
        self: &Arc<Self>,
        signal: &Option<AbortSignal>,
    ) -> bool {
        let Some(execution) = self.active_execution() else {
            return true;
        };
        if !execution.lock().unwrap().settled || is_aborted(signal) {
            return false;
        }
        let existing = self.active_execution_reconciliation.lock().unwrap().clone();
        if let Some(existing) = existing {
            return existing.wait().await.unwrap_or(false);
        }
        let operation = Arc::new(SharedPromise::<bool>::new());
        *self.active_execution_reconciliation.lock().unwrap() = Some(operation.clone());
        let this = self.clone();
        let signal = signal.clone();
        let started = operation.clone();
        tokio::spawn(async move {
            let request_id = uuid_v4();
            let done = Latch::new();
            {
                let mut waiters = this.pending_done_waiters.lock().unwrap();
                ordered_set(&mut waiters, request_id.clone(), done.clone());
            }
            // The original done, cancellation, or disposal may arrive before the
            // barrier reply. Do not hold a cancelled caller until the deadline.
            let cleared =
                this.wait_for_active_execution_to_clear(&signal, KERNEL_BUSY_REUSE_WAIT_MS);
            let send = async {
                let mut request = Map::new();
                request.insert("type".to_string(), json!("execute"));
                request.insert("id".to_string(), json!(request_id));
                request.insert("code".to_string(), json!("None"));
                this.write_line(&request).await
            };
            let confirmed = {
                let send_and_done = async {
                    let sent = send.await;
                    done.wait().await;
                    sent.is_ok()
                };
                tokio::select! {
                    biased;
                    result = send_and_done => result,
                    _ = cleared => false,
                }
            };
            let outcome = if confirmed
                && !is_aborted(&signal)
                && ActiveExecution::same_as(&this.active_execution(), &execution)
            {
                // Requests are serialized in Python. A correlated barrier done proves
                // the previous request ended; elapsed time alone never proves that.
                this.finish_active_execution(&execution);
                true
            } else {
                this.active_execution().is_none()
            };
            started.settle(Ok(outcome));
            let mut guard = this.pending_done_waiters.lock().unwrap();
            ordered_delete(&mut guard, &request_id);
            drop(guard);
            let mut guard = this.active_execution_reconciliation.lock().unwrap();
            let ours = guard
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &started))
                .unwrap_or(false);
            if ours {
                *guard = None;
            }
        });
        operation.wait().await.unwrap_or(false)
    }

    async fn wait_for_active_execution_to_clear_for_reuse(
        self: &Arc<Self>,
        signal: &Option<AbortSignal>,
    ) -> Result<(), KernelError> {
        if self.active_execution().is_none() || is_aborted(signal) {
            return Ok(());
        }
        if self.state() == State::Shutdown {
            return Err(KernelError::new("Kernel has been shut down"));
        }
        let settled = self
            .active_execution()
            .map(|execution| execution.lock().unwrap().settled)
            .unwrap_or(false);
        if settled {
            if self.reconcile_settled_active_execution(signal).await {
                return Ok(());
            }
        } else {
            // The abort path requested the interrupt. Reuse must not send more.
            if self
                .wait_for_active_execution_to_clear(signal, KERNEL_BUSY_REUSE_WAIT_MS)
                .await
            {
                return Ok(());
            }
            let settled = self
                .active_execution()
                .map(|execution| execution.lock().unwrap().settled)
                .unwrap_or(false);
            if settled && self.reconcile_settled_active_execution(signal).await {
                return Ok(());
            }
        }
        if self.active_execution().is_some() && !is_aborted(signal) {
            return Err(kernel_busy_after_interrupt_error());
        }
        Ok(())
    }

    fn start_host_request(self: &Arc<Self>, request_id: String, data: Option<Value>) {
        {
            let mut handled = self.handled_host_request_ids.lock().unwrap();
            if handled.iter().any(|existing| *existing == request_id) {
                return;
            }
            handled.push(request_id.clone());
            while handled.len() > MAX_HANDLED_HOST_REQUEST_IDS {
                if handled.is_empty() {
                    break;
                }
                handled.remove(0);
            }
        }

        let task = Arc::new(SharedPromise::<()>::new());
        self.in_flight_host_requests
            .lock()
            .unwrap()
            .push(task.clone());
        let this = self.clone();
        let started = task.clone();
        tokio::spawn(async move {
            match this.handle_host_request(data).await {
                Ok(result) => {
                    let mut request = Map::new();
                    request.insert("type".to_string(), json!("host_reply"));
                    request.insert("id".to_string(), json!(request_id));
                    request.insert(
                        "data".to_string(),
                        json!({ "status": "ok", "result": result }),
                    );
                    if let Err(reply_error) = this.write_line(&request).await {
                        this.append_kernel_diagnostic(&format!(
                            "failed to send host request ok reply for {request_id}: {}",
                            error_message(&reply_error)
                        ));
                    }
                }
                Err(error) => {
                    this.append_kernel_diagnostic(&format!(
                        "host request failed for {request_id}: {}",
                        error_message(&error)
                    ));
                    let mut request = Map::new();
                    request.insert("type".to_string(), json!("host_reply"));
                    request.insert("id".to_string(), json!(request_id));
                    request.insert(
                        "data".to_string(),
                        json!({ "status": "error", "error": error_message(&error) }),
                    );
                    if let Err(reply_error) = this.write_line(&request).await {
                        this.append_kernel_diagnostic(&format!(
                            "failed to send host request error reply for {request_id}: {}",
                            error_message(&reply_error)
                        ));
                    }
                }
            }
            started.settle(Ok(()));
            let mut guard = this.in_flight_host_requests.lock().unwrap();
            let ours = guard
                .iter()
                .position(|existing| Arc::ptr_eq(existing, &started));
            if let Some(index) = ours {
                guard.remove(index);
            }
        });
    }

    async fn handle_host_request(&self, data: Option<Value>) -> Result<Value, KernelError> {
        let data = data.unwrap_or(Value::Null);
        if !is_record(&data) {
            return Err(KernelError::new("host request payload must be an object"));
        }
        let kind = data.get("type").and_then(|value| value.as_str());
        let Some(kind) = kind else {
            return Err(KernelError::new(
                "host request payload must have a string type",
            ));
        };
        if kind.is_empty() {
            return Err(KernelError::new(
                "host request payload must have a string type",
            ));
        }

        let handler = self
            .options
            .host_handlers
            .as_ref()
            .and_then(|handlers| handlers.get(kind))
            .cloned();
        let Some(handler) = handler else {
            return Err(KernelError::new(format!(
                "host request type \"{kind}\" is not available in this session"
            )));
        };
        // Tag the request with the cell that triggered it. A blocking call is still
        // the in-flight execution; detached spawns (asyncio.create_task) fire after
        // the scheduling cell goes idle, so fall back to that last cell's source.
        let cell_source_code = self
            .active_execution()
            .map(|execution| execution.lock().unwrap().code.clone())
            .or_else(|| self.last_cell_code.lock().unwrap().clone());
        let mut payload = data.as_object().cloned().unwrap_or_default();
        match cell_source_code {
            Some(code) => {
                payload.insert("cellSourceCode".to_string(), json!(cap_cell_source(&code)));
            }
            None => {
                payload.insert("cellSourceCode".to_string(), Value::Null);
            }
        }
        handler(Value::Object(payload)).await
    }

    async fn interrupt(&self) -> Result<(), KernelError> {
        let request_id = self
            .active_execution()
            .map(|execution| execution.lock().unwrap().request_id.clone());
        let Some(request_id) = request_id else {
            return Ok(());
        };
        let mut request = Map::new();
        request.insert("type".to_string(), json!("interrupt"));
        request.insert("id".to_string(), json!(request_id));
        self.write_line(&request).await
    }
}

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------

impl KernelState {
    fn cleanup_resources(self: &Arc<Self>, kill_signal: Option<Signal>) {
        let kill_signal = kill_signal.unwrap_or(Signal::Term);
        self.start_generation.fetch_add(1, Ordering::SeqCst); // any teardown invalidates in-flight starts
        self.clear_snapshot_timer();
        *self.execution_queue_tail_snapshot_metric.lock().unwrap() = None;
        self.runtime_snapshot_formats.lock().unwrap().clear();
        self.late_sent_agent_message_handlers
            .lock()
            .unwrap()
            .clear();
        self.pending_done_waiters.lock().unwrap().clear();
        let had_background_work = {
            let mut handles = self.background_bash_handles.lock().unwrap();
            let had_work = !handles.is_empty();
            handles.clear();
            had_work
        };
        if had_background_work {
            self.notify_background_work_settled();
        }
        // Stale pre-teardown background output must not surface after a restart.
        *self.pending_background_output.lock().unwrap() = String::new();
        self.pending_background_output_truncated
            .store(false, Ordering::SeqCst);
        self.reject_active_execution(KernelError::new("Kernel has been shut down"));
        let child = self.child.lock().unwrap().take();
        *self.ready_deferred.lock().unwrap() = None;
        if let Some(child) = child {
            // `child.stdin?.destroy(); child.stdout?.destroy();`
            child.stdin_destroyed.store(true, Ordering::SeqCst);
            child.destroy_stdout.settle();
            // An exited child keeps its stderr: the post-exit drain owns it, and
            // destroying here would drop its buffered last words. A still-alive
            // child may ignore the kill signal and never emit 'exit', so that
            // drain would never run - destroy now to bound the pipe's lifetime.
            if !child.exit.has_exited() {
                child.destroy_stderr.settle();
            }
            let pid = child.pid;
            let already_exited = child.exit.has_exited();
            let mut signaled = already_exited;
            let mut tree_cleanup_failed = false;
            if cfg!(windows) && pid.is_some() && !already_exited {
                // A venv python.exe can be a shim. Killing only it leaves the real REPL alive.
                let taskkill = std::path::Path::new(
                    &std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string()),
                )
                .join("System32")
                .join("taskkill.exe");
                let args = vec![
                    "/PID".to_string(),
                    pid.unwrap().to_string(),
                    "/T".to_string(),
                    "/F".to_string(),
                ];
                match spawn_sync_hidden_with_timeout(
                    &taskkill.to_string_lossy(),
                    &args,
                    SpawnOptions {
                        capture_stdout: false,
                        capture_stderr: false,
                        ..Default::default()
                    },
                    TASKKILL_TIMEOUT_MS,
                ) {
                    Ok(output) => {
                        // spawnSyncHidden already waited: `result.status` is the exit code.
                        let status = output.status.code();
                        signaled = status == Some(0);
                        if !signaled {
                            tree_cleanup_failed = true;
                            self.append_kernel_diagnostic(&format!(
                                "Windows kernel tree cleanup failed: exit {}",
                                node_option_string(&status)
                            ));
                        }
                    }
                    Err(error) => {
                        tree_cleanup_failed = true;
                        self.append_kernel_diagnostic(&format!(
                            "Windows kernel tree cleanup failed: {}",
                            error_message(&KernelError::new(error.to_string()))
                        ));
                    }
                }
            }
            if !signaled {
                // `child.kill(killSignal)`; without a pid there is nothing to signal.
                if let Some(pid) = pid {
                    signaled = crate::utils::child_process::signal_process_group_or_process(
                        pid as i32,
                        kill_signal,
                    );
                }
            }
            // Inactive only when the signal proved the pid still named our un-reaped child.
            // Killing a venv shim alone does not prove its CPython descendants died.
            // Preserve recovery evidence when tree cleanup failed.
            if let Some(pid) = pid {
                if signaled && !tree_cleanup_failed {
                    record_orphan_process_state(pid as i64, false);
                }
            }
            // A killed/crashed kernel cannot run its own shutdown hook, so the host
            // reaps the bash() process groups it journaled under this kernel pid.
            if let Some(pid) = pid {
                reap_kernel_orphan_processes(pid as i64);
            }
        }
        *self.start_promise.lock().unwrap() = None;
    }

    async fn wait_for_kernel_exit(&self) {
        let child = self.child.lock().unwrap().clone();
        let Some(child) = child else {
            return;
        };
        if child.exit.has_exited() {
            return;
        }
        child.exit.wait().await;
    }

    async fn wait_for_host_requests_to_settle(
        &self,
        tasks: &[Arc<SharedPromise<()>>],
        timeout_ms: u64,
    ) {
        let all = async {
            for task in tasks {
                task.wait().await;
            }
            "settled"
        };
        let timeout = async {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
            "timeout"
        };
        let result = tokio::select! {
            biased;
            result = all => result,
            result = timeout => result,
        };
        if result == "timeout" {
            self.append_kernel_diagnostic(&format!(
                "timed out waiting {timeout_ms}ms for {} host request task(s) during shutdown",
                tasks.len()
            ));
        }
    }

    /// Resolves true when this call performed the cleanup (false: a concurrent
    /// teardown won; a joiner's options are ignored - the first caller's policy wins).
    async fn shutdown(self: &Arc<Self>, opts: KernelShutdownOptions) -> Result<bool, KernelError> {
        let in_flight = self.graceful_shutdown_promise.lock().unwrap().clone();
        if let Some(in_flight) = in_flight {
            let _ = in_flight.wait().await;
            return Ok(false);
        }

        self.teardown_in_flight.fetch_add(1, Ordering::SeqCst);
        self.supersede_protocol_repair();
        let operation = Arc::new(SharedPromise::<bool>::new());
        *self.graceful_shutdown_promise.lock().unwrap() = Some(operation.clone());
        let this = self.clone();
        let started = operation.clone();
        let performed = this.perform_shutdown(opts).await;
        match performed {
            Ok(performed) => {
                started.settle(Ok(performed));
                self.teardown_in_flight.fetch_sub(1, Ordering::SeqCst);
                let mut guard = self.graceful_shutdown_promise.lock().unwrap();
                let ours = guard
                    .as_ref()
                    .map(|current| Arc::ptr_eq(current, &started))
                    .unwrap_or(false);
                if ours {
                    *guard = None;
                }
                Ok(performed)
            }
            Err(error) => {
                started.settle(Err(error.clone()));
                self.teardown_in_flight.fetch_sub(1, Ordering::SeqCst);
                let mut guard = self.graceful_shutdown_promise.lock().unwrap();
                let ours = guard
                    .as_ref()
                    .map(|current| Arc::ptr_eq(current, &started))
                    .unwrap_or(false);
                if ours {
                    *guard = None;
                }
                Err(error)
            }
        }
    }

    /// `shutdown()` with default options.
    async fn shutdown_default(self: &Arc<Self>) -> Result<bool, KernelError> {
        self.shutdown(KernelShutdownOptions::default()).await
    }

    async fn perform_shutdown(
        self: &Arc<Self>,
        opts: KernelShutdownOptions,
    ) -> Result<bool, KernelError> {
        if self.state() == State::Shutdown {
            self.delete_live_kernels();
            let graceful = *self.graceful_shutdown_generation.lock().unwrap();
            if graceful == Some(self.start_generation.load(Ordering::SeqCst)) {
                return Ok(false);
            }
            self.cleanup_resources(Some(Signal::Term));
            return Ok(true);
        }
        // Captured before any await: teardowns and newer starts bump the counter.
        let generation = self.start_generation.load(Ordering::SeqCst);
        if opts.snapshot {
            self.flush_snapshot_for_dispose().await;
            if self.start_stale(generation) {
                return Ok(false);
            }
        }
        // Protocol shutdown first: the runtime closes MCP servers and kills live bash()
        // process groups a bare hard-kill would leak.
        let protocol_shutdown_available = self.state() == State::Running;
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        *self.graceful_shutdown_generation.lock().unwrap() = Some(generation);

        let mut done_waiter_id: Option<String> = None;
        let mut performed_cleanup = false;
        let outcome = async {
            if opts.drain_host_requests {
                let in_flight = self.in_flight_host_requests.lock().unwrap().clone();
                if !in_flight.is_empty() {
                    self.wait_for_host_requests_to_settle(
                        &in_flight,
                        HOST_REQUEST_SHUTDOWN_TIMEOUT_MS,
                    )
                    .await;
                }
            }
            let child = self.child.lock().unwrap().clone();
            let stdin_ready = child
                .as_ref()
                .map(|child| {
                    child
                        .stdin
                        .try_lock()
                        .map(|guard| guard.is_some())
                        .unwrap_or(true)
                })
                .unwrap_or(false);
            if protocol_shutdown_available && !self.start_stale(generation) && stdin_ready {
                let request_id = uuid_v4();
                done_waiter_id = Some(request_id.clone());
                let done = Latch::new();
                {
                    let mut waiters = self.pending_done_waiters.lock().unwrap();
                    ordered_set(&mut waiters, request_id.clone(), done.clone());
                }
                let deadline =
                    tokio::time::sleep(Duration::from_millis(KERNEL_SHUTDOWN_TIMEOUT_MS));
                let send = async {
                    let mut request = Map::new();
                    request.insert("type".to_string(), json!("shutdown"));
                    request.insert("id".to_string(), json!(request_id));
                    self.write_line(&request).await
                };
                let kernel_exit = self.wait_for_kernel_exit();
                let graceful = async {
                    let sent = send.await;
                    if sent.is_err() {
                        return;
                    }
                    done.wait().await;
                };
                tokio::pin!(graceful);
                tokio::pin!(kernel_exit);
                tokio::pin!(deadline);
                tokio::select! {
                    biased;
                    _ = graceful.as_mut() => {}
                    _ = kernel_exit.as_mut() => {}
                    _ = deadline.as_mut() => {
                        return Err(KernelError::new(format!(
                            "Kernel did not shut down within {KERNEL_SHUTDOWN_TIMEOUT_MS}ms"
                        )));
                    }
                }
                tokio::select! {
                    biased;
                    _ = kernel_exit.as_mut() => {}
                    _ = deadline.as_mut() => {
                        return Err(KernelError::new(format!(
                            "Kernel did not shut down within {KERNEL_SHUTDOWN_TIMEOUT_MS}ms"
                        )));
                    }
                }
            }
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            self.append_kernel_diagnostic(&format!(
                "graceful shutdown failed (killing instead): {}",
                error_message(&error)
            ));
        }
        if let Some(id) = done_waiter_id {
            let mut waiters = self.pending_done_waiters.lock().unwrap();
            ordered_delete(&mut waiters, &id);
        }
        {
            let mut graceful = self.graceful_shutdown_generation.lock().unwrap();
            if *graceful == Some(generation) {
                *graceful = None;
            }
        }
        if !self.start_stale(generation) {
            self.cleanup_resources(Some(Signal::Term));
            performed_cleanup = true;
        }

        Ok(performed_cleanup)
    }

    async fn restart(self: &Arc<Self>) -> Result<(), KernelError> {
        // A final dispose flush owns the queue tail. Taking a slot now and joining
        // the in-flight shutdown would deadlock: the flush's snapshot waits on our
        // slot while we wait on the flush's shutdown.
        if self.flushing_snapshot_for_dispose.load(Ordering::SeqCst) {
            return Err(KernelError::new("Kernel is shutting down"));
        }
        let prev = self.execution_queue.lock().unwrap().clone();
        let next = Arc::new(SharedPromise::<()>::new());
        *self.execution_queue.lock().unwrap() = next.clone();
        prev.wait().await;

        let outcome = async {
            let performed_cleanup = self.shutdown_default().await?;
            if !performed_cleanup {
                return Ok(());
            }
            self.set_state(State::Idle);
            *self.kernel_stderr.lock().unwrap() = String::new();
            self.start_default().await
        }
        .await;
        next.settle(Ok(()));
        outcome
    }

    async fn kill(self: &Arc<Self>) -> Result<(), KernelError> {
        self.supersede_protocol_repair();
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        self.cleanup_resources(Some(Signal::Kill));
        Ok(())
    }

    fn dispose_sync(self: &Arc<Self>) {
        self.supersede_protocol_repair();
        self.set_state(State::Shutdown);
        self.delete_live_kernels();
        self.cleanup_resources(Some(Signal::Term));
    }
}

// ---------------------------------------------------------------------------
// Snapshots and restore
// ---------------------------------------------------------------------------

impl KernelState {
    /**
     * Serialize the user namespace to disk (best-effort, per-variable). No-op when
     * the kernel isn't running or no snapshot target was configured. Never throws.
     */
    async fn snapshot_state(self: &Arc<Self>) -> Option<SnapshotResult> {
        self.capture_snapshot(CaptureSnapshotOptions::default())
            .await
    }

    /// Persist the namespace, then remove variables above the per-variable cap.
    async fn prune_oversized_variables(self: &Arc<Self>) -> Option<SnapshotResult> {
        self.capture_snapshot(CaptureSnapshotOptions {
            execution_timeout_ms: Some(SNAPSHOT_EXECUTION_TIMEOUT_MS),
            prune_oversized: true,
        })
        .await
    }

    fn cas_root_path(&self) -> Option<String> {
        let cfg = self.options.snapshot.as_ref()?;
        Some(
            cfg.cas_root_path
                .clone()
                .unwrap_or_else(|| cas_snapshot_root_for_legacy_path(&cfg.path)),
        )
    }

    fn requires_cas_capability(&self, source: KernelRestoreSource) -> bool {
        let Some(cfg) = self.options.snapshot.as_ref() else {
            return false;
        };
        let Some(root) = self.cas_root_path() else {
            return false;
        };
        if source == KernelRestoreSource::Legacy {
            return false;
        }
        source == KernelRestoreSource::Current
            || source == KernelRestoreSource::Previous
            || cfg.format == Some(KernelSnapshotFormat::CasV2)
            || cas_snapshot_state_exists(&root)
    }

    fn record_snapshot_metric(
        &self,
        state: Option<&SnapshotMetricState>,
        outcome: PerformanceMetricOutcome,
        metadata: Option<&SnapshotPerformanceMetadata>,
    ) {
        let Some(state) = state else {
            return;
        };
        let ended_at = safe_metric_now(Some(&state.recorder));
        let (following_cell_queued_at, dequeued_at) = {
            let timing = state.timing.lock().unwrap();
            (timing.following_cell_queued_at, timing.dequeued_at)
        };
        let next_cell_start = match (following_cell_queued_at, dequeued_at) {
            (Some(queued), Some(dequeued)) => Some(queued.max(dequeued)),
            _ => None,
        };
        let event = PerformanceMetricEvent {
            operation: "snapshot".to_string(),
            component: Some("snapshot".to_string()),
            outcome: Some(outcome),
            measurements: vec![
                (
                    "total_ms",
                    crate::core::kernel::shared::elapsed_metric_ms(state.started_at, ended_at),
                ),
                (
                    "queue_ms",
                    crate::core::kernel::shared::elapsed_metric_ms(
                        state.started_at,
                        dequeued_at,
                    ),
                ),
                (
                    "serialization_ms",
                    metadata.and_then(|metadata| metadata.serialization_wall_ms),
                ),
                (
                    "serialization_cpu_ms",
                    metadata.and_then(|metadata| metadata.serialization_cpu_ms),
                ),
                (
                    "serialization_max_variable_ms",
                    metadata.and_then(|metadata| metadata.serialization_max_variable_ms),
                ),
                (
                    "serialization_slow_variables",
                    metadata.and_then(|metadata| metadata.serialization_slow_variables),
                ),
                (
                    "serialization_saved_ms",
                    metadata.and_then(|metadata| metadata.serialization_saved_ms),
                ),
                (
                    "serialization_skipped_ms",
                    metadata.and_then(|metadata| metadata.serialization_skipped_ms),
                ),
                ("write_ms", metadata.and_then(|metadata| metadata.write_ms)),
                (
                    "serialized_bytes",
                    metadata.and_then(|metadata| metadata.serialized_bytes),
                ),
                (
                    "written_bytes",
                    metadata.and_then(|metadata| metadata.written_bytes),
                ),
                (
                    "next_cell_delay_ms",
                    crate::core::kernel::shared::elapsed_metric_ms(next_cell_start, ended_at),
                ),
            ],
        };
        safe_record_performance_metric(Some(&state.recorder), event);
    }

    async fn capture_snapshot(
        self: &Arc<Self>,
        options: CaptureSnapshotOptions,
    ) -> Option<SnapshotResult> {
        let cfg = self.options.snapshot.clone()?;
        if !self.is_running() {
            return None;
        }
        let recorder = self.options.performance_metrics.clone();
        let metric_state = recorder.map(|recorder| SnapshotMetricState {
            started_at: safe_metric_now(Some(&recorder)),
            recorder,
            // Queue bookkeeping and the final recorder must observe the same
            // state, as they do with the shared object in the TypeScript runtime.
            timing: Arc::new(Mutex::new(SnapshotQueueTiming::default())),
        });
        let mut metric_metadata: Option<SnapshotPerformanceMetadata> = None;
        let mut metric_outcome = PerformanceMetricOutcome::Failure;
        let outcome = async {
            if self.requires_cas_capability(KernelRestoreSource::Auto)
                && !self
                    .runtime_snapshot_formats
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|format| format == "cas-v2")
            {
                self.append_kernel_diagnostic("state snapshot requires a CAS v2-capable Python runtime");
                return None;
            }
            let cas_root = self.cas_root_path();
            let request = json!({
                "type": "snapshot",
                "path": cfg.path.clone(),
                "manifest_path": cfg.manifest_path.clone(),
                "cas_root": match cas_root { Some(root) => Value::String(root), None => Value::Null },
                "snapshot_format": snapshot_format_field(cfg.format),
                "max_bytes": cfg.max_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_BYTES),
                "max_variable_bytes": cfg.max_variable_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES),
                "prune_oversized": options.prune_oversized,
            });
            let r = self
                .enqueue_request(
                    &request,
                    "",
                    ExecuteOptions {
                        internal: true,
                        ..Default::default()
                    },
                    options.execution_timeout_ms,
                    metric_state.clone(),
                )
                .await;
            let r = match r {
                Ok(r) => r,
                Err(error) => {
                    self.append_kernel_diagnostic(&format!("state snapshot error: {}", error_message(&error)));
                    return None;
                }
            };
            metric_metadata = r
                .done_fields
                .as_ref()
                .and_then(|fields| as_snapshot_performance_metadata(fields.get("metrics")));
            if r.result.status != ExecuteStatus::Ok || r.done_fields.is_none() {
                metric_outcome = if r.result.status == ExecuteStatus::Aborted {
                    PerformanceMetricOutcome::Cancelled
                } else {
                    PerformanceMetricOutcome::Failure
                };
                let how = if r.result.status == ExecuteStatus::Aborted {
                    "timed out"
                } else {
                    "failed"
                };
                let detail = match (&r.result.error, &r.result.stderr) {
                    (Some(error), _) if !error.evalue.is_empty() => error.evalue.clone(),
                    (_, stderr) => stderr.clone(),
                };
                let diagnostic = format!("state snapshot {how}: {detail}");
                let this = self.clone();
                this.append_kernel_diagnostic(&diagnostic);
                return None;
            }
            let fields = r.done_fields.clone().unwrap_or_default();
            let pruned = as_string_array_field(fields.get("pruned"));
            let format = if fields.get("format").and_then(|value| value.as_str()) == Some("cas-v2") {
                KernelSnapshotFormat::CasV2
            } else {
                KernelSnapshotFormat::Legacy
            };
            metric_outcome = PerformanceMetricOutcome::Success;
            Some(SnapshotResult {
                saved: as_string_array_field(fields.get("saved")),
                skipped: as_reason_array(fields.get("skipped")),
                pruned: if pruned.is_empty() { None } else { Some(pruned) },
                bytes: as_metric_number_field(fields.get("bytes")).unwrap_or(0.0) as u64,
                logical_bytes: as_metric_number_field(fields.get("logical_bytes"))
                    .or(metric_metadata.as_ref().and_then(|metadata| metadata.serialized_bytes))
                    .unwrap_or(0.0) as u64,
                written_bytes: as_metric_number_field(fields.get("written_bytes"))
                    .or(metric_metadata.as_ref().and_then(|metadata| metadata.written_bytes))
                    .unwrap_or(0.0) as u64,
                generation: fields
                    .get("generation")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string()),
                backward_readable: fields
                    .get("backward_readable")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(format == KernelSnapshotFormat::Legacy),
                metrics: metric_metadata.clone(),
                format,
                path: cfg.path.clone(),
            })
        }
        .await;
        self.record_snapshot_metric(
            metric_state.as_ref(),
            metric_outcome,
            metric_metadata.as_ref(),
        );
        outcome
    }

    /**
     * Revive a previously snapshotted namespace into the kernel. Call right after
     * start() and before the runtime bootstrap, which then refreshes live handles
     * (rlm, skills) over anything restored. Never throws.
     */
    async fn restore_state_with_options(
        self: &Arc<Self>,
        options: KernelRestoreOptions,
    ) -> Option<RestoreResult> {
        let source = options.source.unwrap_or(KernelRestoreSource::Auto);
        self.perform_restore(false, source).await
    }

    /// Repair restores bypass the repair gate and are bounded so a stalled kernel cannot wedge it.
    async fn perform_restore(
        self: &Arc<Self>,
        protocol_repair: bool,
        source: KernelRestoreSource,
    ) -> Option<RestoreResult> {
        let cfg = self.options.snapshot.clone()?;
        if self.start_default().await.is_err() {
            return None;
        }
        let cas_capable = self
            .runtime_snapshot_formats
            .lock()
            .unwrap()
            .iter()
            .any(|format| format == "cas-v2");
        if self.requires_cas_capability(source) && !cas_capable {
            let reason = "state restore requires a CAS v2-capable Python runtime".to_string();
            self.append_kernel_diagnostic(&reason);
            return if protocol_repair {
                None
            } else {
                Some(RestoreResult {
                    restored: Vec::new(),
                    failed: vec![SkippedVariable {
                        name: "<snapshot>".to_string(),
                        reason,
                    }],
                    format: None,
                    generation: None,
                    rolled_back: None,
                    unsaved_work_possible: None,
                    legacy_recovery: None,
                    path: cfg.path.clone(),
                })
            };
        }
        let cas_root = self.cas_root_path();
        let request = json!({
            "type": "restore",
            "path": cfg.path.clone(),
            "cas_root": match cas_root { Some(root) => Value::String(root), None => Value::Null },
            "source": source.as_str(),
            "max_bytes": cfg.max_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_BYTES),
            "max_variable_bytes": cfg.max_variable_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES),
        });
        let r = self
            .enqueue_request(
                &request,
                "",
                ExecuteOptions {
                    internal: true,
                    protocol_repair,
                    ..Default::default()
                },
                if protocol_repair {
                    Some(REPAIR_STEP_TIMEOUT_MS)
                } else {
                    None
                },
                None,
            )
            .await;
        let r = match r {
            Ok(r) => r,
            Err(error) => {
                self.append_kernel_diagnostic(&format!(
                    "state restore error: {}",
                    error_message(&error)
                ));
                return None;
            }
        };
        if r.result.status != ExecuteStatus::Ok || r.done_fields.is_none() {
            let reason = if r.result.status == ExecuteStatus::Aborted {
                "restore timed out".to_string()
            } else {
                let detail = match (&r.result.error, &r.result.stderr) {
                    (Some(error), _) if !error.evalue.is_empty() => error.evalue.clone(),
                    (_, stderr) => stderr.clone(),
                };
                if detail.is_empty() {
                    "restore failed".to_string()
                } else {
                    detail
                }
            };
            self.append_kernel_diagnostic(&format!("state restore {reason}"));
            return if protocol_repair {
                None
            } else {
                Some(RestoreResult {
                    restored: Vec::new(),
                    failed: vec![SkippedVariable {
                        name: "<snapshot>".to_string(),
                        reason,
                    }],
                    format: None,
                    generation: None,
                    rolled_back: None,
                    unsaved_work_possible: None,
                    legacy_recovery: None,
                    path: cfg.path.clone(),
                })
            };
        }
        let fields = r.done_fields.clone().unwrap_or_default();
        self.pending_restore.store(false, Ordering::SeqCst);
        Some(RestoreResult {
            restored: as_string_array_field(fields.get("restored")),
            failed: as_reason_array(fields.get("failed")),
            format: fields
                .get("format")
                .and_then(|value| value.as_str())
                .and_then(KernelSnapshotFormat::from_str),
            generation: fields
                .get("generation")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string()),
            rolled_back: match fields.get("rolled_back").and_then(|value| value.as_bool()) {
                Some(true) => Some(true),
                _ => None,
            },
            unsaved_work_possible: match fields
                .get("unsaved_work_possible")
                .and_then(|value| value.as_bool())
            {
                Some(true) => Some(true),
                _ => None,
            },
            legacy_recovery: match fields
                .get("legacy_recovery")
                .and_then(|value| value.as_bool())
            {
                Some(true) => Some(true),
                _ => None,
            },
            path: cfg.path.clone(),
        })
    }

    /// Export the selected committed CAS generation for an explicitly gated older runtime.
    async fn export_state_for_legacy_runtime(
        self: &Arc<Self>,
        source: LegacyExportSource,
    ) -> Option<SnapshotLegacyExportResult> {
        let cfg = self.options.snapshot.clone()?;
        let cas_root = self.cas_root_path()?;
        if !cas_snapshot_state_exists(&cas_root) {
            return None;
        }
        if self.start_default().await.is_err() {
            return None;
        }
        if !self
            .runtime_snapshot_formats
            .lock()
            .unwrap()
            .iter()
            .any(|format| format == "cas-v2")
        {
            self.append_kernel_diagnostic("legacy export requires a CAS v2-capable Python runtime");
            return None;
        }
        let request = json!({
            "type": "snapshot_export_legacy",
            "path": cfg.path.clone(),
            "manifest_path": cfg.manifest_path.clone(),
            "cas_root": cas_root,
            "source": source.as_str(),
            "max_bytes": cfg.max_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_BYTES),
            "max_variable_bytes": cfg.max_variable_bytes.unwrap_or(DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES),
        });
        let r = self
            .enqueue_request(
                &request,
                "",
                ExecuteOptions {
                    internal: true,
                    ..Default::default()
                },
                Some(SNAPSHOT_EXECUTION_TIMEOUT_MS),
                None,
            )
            .await;
        let r = match r {
            Ok(r) => r,
            Err(error) => {
                self.append_kernel_diagnostic(&format!(
                    "legacy export error: {}",
                    error_message(&error)
                ));
                return None;
            }
        };
        let fields = r.done_fields.clone().unwrap_or_default();
        let source_generation = fields
            .get("source_generation")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        if r.result.status != ExecuteStatus::Ok || source_generation.is_none() {
            let detail = match (&r.result.error, &r.result.stderr) {
                (Some(error), _) if !error.evalue.is_empty() => error.evalue.clone(),
                (_, stderr) => stderr.clone(),
            };
            self.append_kernel_diagnostic(&format!("legacy export failed: {detail}"));
            return None;
        }
        Some(SnapshotLegacyExportResult {
            exported: as_string_array_field(fields.get("exported")),
            bytes: as_metric_number_field(fields.get("bytes")).unwrap_or(0.0) as u64,
            source_generation: source_generation.unwrap_or_default(),
            source,
            backward_readable: true,
            path: cfg.path.clone(),
        })
    }

    /// Live user-defined top-level names, or null if the kernel isn't running. Never throws.
    async fn list_namespace_names(
        self: &Arc<Self>,
        signal: Option<AbortSignal>,
    ) -> Option<Vec<String>> {
        if !self.is_running() {
            return None;
        }
        let r = self
            .enqueue_request(
                &json!({ "type": "list_names" }),
                "",
                ExecuteOptions {
                    internal: true,
                    signal,
                    ..Default::default()
                },
                None,
                None,
            )
            .await;
        let r = match r {
            Ok(r) => r,
            Err(error) => {
                self.append_kernel_diagnostic(&format!(
                    "namespace listing error: {}",
                    error_message(&error)
                ));
                return None;
            }
        };
        if r.result.status != ExecuteStatus::Ok || r.done_fields.is_none() {
            let detail = match (&r.result.error, &r.result.stderr) {
                (Some(error), _) if !error.evalue.is_empty() => error.evalue.clone(),
                (_, stderr) => stderr.clone(),
            };
            self.append_kernel_diagnostic(&format!("namespace listing failed: {detail}"));
            return None;
        }
        Some(as_string_array_field(
            r.done_fields
                .as_ref()
                .and_then(|fields| fields.get("names")),
        ))
    }

    fn schedule_snapshot(self: &Arc<Self>) {
        let Some(cfg) = self.options.snapshot.as_ref() else {
            return;
        };
        let debounce = cfg.debounce_ms.unwrap_or(DEFAULT_SNAPSHOT_DEBOUNCE_MS);
        let mut timer = self.snapshot_timer.lock().unwrap();
        if let Some(existing) = timer.take() {
            existing.clear();
        }
        let this = self.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(debounce)).await;
            *this.snapshot_timer.lock().unwrap() = None;
            this.capture_snapshot(CaptureSnapshotOptions {
                execution_timeout_ms: Some(SNAPSHOT_EXECUTION_TIMEOUT_MS),
                prune_oversized: false,
            })
            .await;
        });
        *timer = Some(SnapshotTimerHandle { handle });
    }

    fn clear_snapshot_timer(&self) {
        let mut timer = self.snapshot_timer.lock().unwrap();
        if let Some(existing) = timer.take() {
            existing.clear();
        }
    }

    /// Boxed: `shutdown -> flush -> captureSnapshot -> enqueueRequest -> start` is a
    /// legal async recursion in the TypeScript, but rustc cannot compute the opaque
    /// return types of that cycle; the trait object ends it.
    fn flush_snapshot_for_dispose<'a>(
        self: &'a Arc<Self>,
    ) -> crate::core::kernel::shared::BoxFuture<'a, ()> {
        Box::pin(async move {
            // Concurrent teardowns (dispose vs a signal-handler shutdown) join one flush:
            // a second flusher would clear the execution guard while the first is still
            // snapshotting and enqueue a duplicate final snapshot behind it.
            let existing = self.snapshot_flush_for_dispose.lock().unwrap().clone();
            let flush = match existing {
                Some(flush) => flush,
                None => {
                    let flush = Arc::new(SharedPromise::<()>::new());
                    *self.snapshot_flush_for_dispose.lock().unwrap() = Some(flush.clone());
                    let this = self.clone();
                    let started = flush.clone();
                    tokio::spawn(async move {
                        this.run_snapshot_flush_for_dispose().await;
                        started.settle(Ok(()));
                        let mut guard = this.snapshot_flush_for_dispose.lock().unwrap();
                        let ours = guard
                            .as_ref()
                            .map(|current| Arc::ptr_eq(current, &started))
                            .unwrap_or(false);
                        if ours {
                            *guard = None;
                        }
                    });
                    flush
                }
            };
            let _ = flush.wait().await;
        })
    }

    async fn run_snapshot_flush_for_dispose(self: &Arc<Self>) {
        if self.options.snapshot.is_none() || !self.is_running() {
            return;
        }
        // A kernel that never restored the saved namespace must not overwrite it:
        // the on-disk snapshot is strictly fresher than this namespace.
        if self.pending_restore.load(Ordering::SeqCst) {
            return;
        }
        // Block new external executions so none can splice ahead of the final snapshot and stall dispose.
        self.flushing_snapshot_for_dispose
            .store(true, Ordering::SeqCst);
        let pending_executions = self.execution_queue.lock().unwrap().clone();
        if self.active_execution().is_some() {
            let this = self.clone();
            tokio::spawn(async move {
                let _ = this.interrupt().await;
            });
        }
        let queue_settled = tokio::select! {
            biased;
            _ = pending_executions.wait() => true,
            _ = tokio::time::sleep(Duration::from_millis(SNAPSHOT_EXECUTION_TIMEOUT_MS)) => false,
        };
        if queue_settled {
            self.capture_snapshot(CaptureSnapshotOptions {
                execution_timeout_ms: Some(SNAPSHOT_EXECUTION_TIMEOUT_MS),
                prune_oversized: false,
            })
            .await;
        }
        // Reset: a superseding start() can revive this kernel for new work.
        self.flushing_snapshot_for_dispose
            .store(false, Ordering::SeqCst);
    }

    fn is_running(&self) -> bool {
        self.state() == State::Running
    }

    fn is_defunct(&self) -> bool {
        self.state() == State::Shutdown
    }
}

// ---------------------------------------------------------------------------
// ReplKernelManager: the `KernelClient` surface
// ---------------------------------------------------------------------------

impl KernelClient for ReplKernelManager {
    fn owner_session_id(&self) -> Option<String> {
        self.state.owner_session_id()
    }

    fn is_running(&self) -> bool {
        self.state.is_running()
    }

    fn has_background_work(&self) -> bool {
        self.state.has_background_work()
    }

    fn is_defunct(&self) -> bool {
        self.state.is_defunct()
    }

    fn start<'a>(
        &'a self,
        options: KernelStartOptions,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Result<(), KernelError>> {
        let state = self.state.clone();
        Box::pin(async move { state.start_with_options(options).await })
    }

    fn execute<'a>(
        &'a self,
        code: String,
        opts: ExecuteOptions,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Result<ExecuteResult, KernelError>> {
        let state = self.state.clone();
        Box::pin(async move { state.execute(&code, opts).await })
    }

    fn shutdown<'a>(
        &'a self,
        opts: KernelShutdownOptions,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Result<bool, KernelError>> {
        let state = self.state.clone();
        Box::pin(async move { state.shutdown(opts).await })
    }

    fn restart<'a>(
        &'a self,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Result<(), KernelError>> {
        let state = self.state.clone();
        Box::pin(async move { state.restart().await })
    }

    fn kill<'a>(&'a self) -> crate::core::kernel::shared::BoxFuture<'a, ()> {
        let state = self.state.clone();
        Box::pin(async move {
            let _ = state.kill().await;
        })
    }

    fn dispose_sync(&self) {
        self.state.dispose_sync();
    }

    fn snapshot_state<'a>(
        &'a self,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Option<SnapshotResult>> {
        let state = self.state.clone();
        Box::pin(async move { state.snapshot_state().await })
    }

    fn prune_oversized_variables<'a>(
        &'a self,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Option<SnapshotResult>> {
        let state = self.state.clone();
        Box::pin(async move { state.prune_oversized_variables().await })
    }

    fn restore_state<'a>(
        &'a self,
        options: KernelRestoreOptions,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Option<RestoreResult>> {
        let state = self.state.clone();
        Box::pin(async move { state.restore_state_with_options(options).await })
    }

    fn export_state_for_legacy_runtime<'a>(
        &'a self,
        source: LegacyExportSource,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Option<SnapshotLegacyExportResult>> {
        let state = self.state.clone();
        Box::pin(async move { state.export_state_for_legacy_runtime(source).await })
    }

    fn list_namespace_names<'a>(
        &'a self,
        signal: Option<AbortSignal>,
    ) -> crate::core::kernel::shared::BoxFuture<'a, Option<Vec<String>>> {
        let state = self.state.clone();
        Box::pin(async move { state.list_namespace_names(signal).await })
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `{ executionTimeoutMs?: number; pruneOversized?: boolean }` for `captureSnapshot`.
#[derive(Debug, Clone, Default)]
struct CaptureSnapshotOptions {
    execution_timeout_ms: Option<u64>,
    prune_oversized: bool,
}

/// Latching one-shot notification: `settle` before a wait still releases it.
struct Latch {
    settled: AtomicBool,
    notify: tokio::sync::Notify,
}

impl Latch {
    fn new() -> Arc<Latch> {
        Arc::new(Latch {
            settled: AtomicBool::new(false),
            notify: tokio::sync::Notify::new(),
        })
    }

    fn settle(&self) {
        self.settled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    fn reset(&self) {
        self.settled.store(false, Ordering::SeqCst);
    }

    fn is_settled(&self) -> bool {
        self.settled.load(Ordering::SeqCst)
    }

    async fn wait(&self) {
        loop {
            if self.is_settled() {
                return;
            }
            let notified = self.notify.notified();
            if self.is_settled() {
                return;
            }
            notified.await;
        }
    }
}

/// `{ stdout: "", stderr: "", status: "aborted", durationMs: n }`.
fn aborted_result(duration_ms: f64) -> InternalExecuteResult {
    InternalExecuteResult::aborted(duration_ms)
}

/// `{ ...process.env, ...extra }`: later definitions replace earlier ones.
fn set_env_var(env: &mut Vec<(String, String)>, key: &str, value: &str) {
    for entry in env.iter_mut() {
        if entry.0 == key {
            entry.1 = value.to_string();
            return;
        }
    }
    env.push((key.to_string(), value.to_string()));
}

/// Node's string interpolation of a `number | null`: `null` prints as "null".
fn node_option_string(value: &Option<i32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "null".to_string(),
    }
}

/// `String(value)` for an arbitrary JSON value, as the diagnostic messages do.
fn node_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Node's `signal` field of the child `exit` event.
fn node_signal_name(status: &std::process::ExitStatus) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        let number = status.signal()?;
        let name = match number {
            libc::SIGHUP => "SIGHUP",
            libc::SIGINT => "SIGINT",
            libc::SIGQUIT => "SIGQUIT",
            libc::SIGKILL => "SIGKILL",
            libc::SIGTERM => "SIGTERM",
            libc::SIGALRM => "SIGALRM",
            libc::SIGUSR1 => "SIGUSR1",
            libc::SIGUSR2 => "SIGUSR2",
            _ => return Some(format!("SIG{number}")),
        };
        Some(name.to_string())
    }
    #[cfg(not(unix))]
    {
        let _ = status;
        None
    }
}

/// `cfg.format ?? "auto"` for the `snapshot_format` request field.
fn snapshot_format_field(format: Option<KernelSnapshotFormat>) -> &'static str {
    match format {
        Some(format) => format.as_str(),
        None => "auto",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn nine_kernel_shared_snapshot_timing_survives_older_release() {
        struct Recorder;
        impl PerformanceMetricRecorder for Recorder {
            fn session_id(&self) -> &str { "private-metric-test" }
            fn monotonic_now(&self) -> f64 { 1.0 }
            fn record(&self, _: crate::core::kernel::shared::PerformanceMetricEvent) {}
        }
        let older = SnapshotMetricState {
            recorder: Arc::new(Recorder), started_at: Some(0.0),
            timing: Arc::new(Mutex::new(SnapshotQueueTiming::default())),
        };
        let queued = older.clone();
        queued.timing.lock().unwrap().dequeued_at = Some(10.0);
        queued.timing.lock().unwrap().following_cell_queued_at = Some(20.0);
        assert_eq!(older.timing.lock().unwrap().dequeued_at, Some(10.0));
        assert_eq!(older.timing.lock().unwrap().following_cell_queued_at, Some(20.0));
        let newer = SnapshotMetricState { timing: Arc::new(Mutex::new(SnapshotQueueTiming::default())), ..older.clone() };
        let manager = new_repl_kernel_manager(KernelManagerOptions::default());
        *manager.state.execution_queue_tail_snapshot_metric.lock().unwrap() = Some(newer.clone());
        manager.state.release_metric_tail(&Some(older));
        assert!(manager.state.execution_queue_tail_snapshot_metric.lock().unwrap().as_ref()
            .is_some_and(|tail| Arc::ptr_eq(&tail.timing, &newer.timing)));
        manager.state.release_metric_tail(&Some(newer));
        assert!(manager.state.execution_queue_tail_snapshot_metric.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn first_execution_and_followup_reach_the_kernel_write() {
        let manager = new_repl_kernel_manager(KernelManagerOptions::default());
        manager.state.set_state(State::Running);
        let started = Arc::new(SharedPromise::new());
        started.settle(Ok(()));
        *manager.state.start_promise.lock().unwrap() = Some(started);

        for code in ["1", "2"] {
            let result = tokio::time::timeout(
                Duration::from_secs(1),
                manager.execute(code.to_string(), ExecuteOptions::default()),
            )
            .await
            .expect("execution must not remain parked on the queue");
            // `KernelError` is an enum, not a struct: read it through the canonical
            // `error_message` helper (shared.rs:60) exactly as the production paths do.
            assert_eq!(
                error_message(&result.unwrap_err()),
                "Kernel stdin is not connected"
            );
        }
    }

    #[test]
    fn invalid_protocol_frame_reason_accepts_known_kinds_only() {
        let mut ready = Map::new();
        ready.insert("event".into(), json!("ready"));
        assert_eq!(invalid_protocol_frame_reason(&ready), None);

        let mut unknown = Map::new();
        unknown.insert("event".into(), json!("surprise"));
        assert_eq!(
            invalid_protocol_frame_reason(&unknown),
            Some("unknown protocol event".to_string())
        );

        let mut missing_id = Map::new();
        missing_id.insert("event".into(), json!("done"));
        assert_eq!(
            invalid_protocol_frame_reason(&missing_id),
            Some("done frame without id".to_string())
        );

        let mut empty_id = Map::new();
        empty_id.insert("event".into(), json!("host_request"));
        empty_id.insert("id".into(), json!(""));
        assert_eq!(
            invalid_protocol_frame_reason(&empty_id),
            Some("host_request frame without id".to_string())
        );

        let mut ok = Map::new();
        ok.insert("event".into(), json!("host_request"));
        ok.insert("id".into(), json!("abc"));
        assert_eq!(invalid_protocol_frame_reason(&ok), None);
    }

    #[test]
    fn decode_utf8_chunk_holds_partial_characters() {
        let mut pending: Vec<u8> = Vec::new();
        // First two bytes of a three-byte character: nothing is emitted yet.
        assert_eq!(decode_utf8_chunk(&mut pending, &[0xE2, 0x82]), "");
        assert_eq!(pending.len(), 2);
        assert_eq!(decode_utf8_chunk(&mut pending, &[0xAC]), "\u{20AC}");
        assert!(pending.is_empty());
        assert_eq!(flush_utf8_pending(&mut pending), "");
    }

    #[test]
    fn utf8_flush_replaces_an_incomplete_tail() {
        let mut pending: Vec<u8> = Vec::new();
        assert_eq!(decode_utf8_chunk(&mut pending, &[0x41, 0xE2, 0x82]), "A");
        assert_eq!(flush_utf8_pending(&mut pending), "\u{FFFD}");
    }

    #[test]
    fn as_snapshot_performance_metadata_rejects_non_numbers() {
        let metadata = as_snapshot_performance_metadata(Some(&json!({
            "serialization_wall_ms": 12.5,
            "serialized_bytes": "x",
            "write_ms": -1,
            "serialization_max_variable_ms": 11.0,
            "serialization_slow_variables": 2,
            "serialization_saved_ms": true,
            "serialization_skipped_ms": -1,
        })))
        .expect("record");
        assert_eq!(metadata.serialization_wall_ms, Some(12.5));
        assert_eq!(metadata.serialized_bytes, None);
        assert_eq!(metadata.write_ms, None);
        assert_eq!(metadata.serialization_max_variable_ms, Some(11.0));
        assert_eq!(metadata.serialization_slow_variables, Some(2.0));
        assert_eq!(metadata.serialization_saved_ms, None);
        assert_eq!(metadata.serialization_skipped_ms, None);
        let old_metadata = as_snapshot_performance_metadata(Some(&json!({}))).expect("old metadata");
        assert_eq!(old_metadata.serialization_max_variable_ms, None);
        assert!(as_snapshot_performance_metadata(Some(&json!([1]))).is_none());
    }

    #[test]
    fn snapshot_format_field_defaults_to_auto() {
        assert_eq!(snapshot_format_field(None), "auto");
        assert_eq!(
            snapshot_format_field(Some(KernelSnapshotFormat::CasV2)),
            "cas-v2"
        );
        assert_eq!(
            snapshot_format_field(Some(KernelSnapshotFormat::Legacy)),
            "legacy"
        );
    }

    #[test]
    fn set_env_var_replaces_in_place() {
        let mut env = vec![("A".to_string(), "1".to_string())];
        set_env_var(&mut env, "A", "2");
        set_env_var(&mut env, "B", "3");
        assert_eq!(
            env,
            vec![
                ("A".to_string(), "2".to_string()),
                ("B".to_string(), "3".to_string())
            ]
        );
    }

    #[test]
    fn bash_activity_ids_match_the_runtime_shape() {
        assert!(is_bash_activity_id(&"a".repeat(32)));
        assert!(!is_bash_activity_id(&"A".repeat(32)));
        assert!(!is_bash_activity_id(&"a".repeat(31)));
    }

    #[test]
    fn tail_and_truncate_count_characters() {
        assert_eq!(truncate_chars("ab\u{20AC}d", 3), "ab\u{20AC}");
        assert_eq!(tail_chars("ab\u{20AC}d", 2), "\u{20AC}d");
        assert_eq!(tail_chars("short", 10), "short");
    }

    #[test]
    fn node_option_string_prints_null_like_node() {
        assert_eq!(node_option_string(&Some(0)), "0");
        assert_eq!(node_option_string(&None), "null");
    }

    #[test]
    fn aborted_result_matches_the_typescript_shape() {
        let result = aborted_result(4.0);
        assert_eq!(result.result.status, ExecuteStatus::Aborted);
        assert_eq!(result.result.duration_ms, 4.0);
        assert_eq!(result.result.stdout, "");
        assert_eq!(result.result.stderr, "");
        assert!(result.done_fields.is_none());
    }

    #[test]
    fn required_constants_match_the_typescript() {
        assert_eq!(REPL_PROTOCOL_VERSION, 3.0);
        assert_eq!(READY_TIMEOUT_MS, 30_000);
        assert_eq!(REPAIR_STEP_TIMEOUT_MS, 30_000);
        assert_eq!(MAX_HANDLED_HOST_REQUEST_IDS, 1024);
        assert_eq!(MAX_BACKGROUND_OUTPUT_CHARS, 64 * 1024);
        assert_eq!(MAX_KERNEL_STDERR_CHARS, 8 * 1024);
        assert_eq!(MAX_KERNEL_STDERR_LOG_BYTES, 5 * 1024 * 1024);
        assert_eq!(TASKKILL_TIMEOUT_MS, 5000);
        assert_eq!(PROTOCOL_EVENT_KINDS.len(), 8);
    }

    // --- sessionkernel_t04 fixtures -------------------------------------------------

    /// Process-global lock for tests that touch ambient env vars.
    static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn protocol_frame_limit_handles_delimiters_and_unicode() {
        let maximum = "x".repeat(MAX_PROTOCOL_FRAME_BYTES);
        assert!(!first_protocol_frame_too_large(&maximum));
        assert!(!first_protocol_frame_too_large(&format!("{maximum}\nnext")));
        assert!(first_protocol_frame_too_large(&format!("{maximum}x")));
        assert!(first_protocol_frame_too_large(&format!("{maximum}x\n")));
        assert!(first_protocol_frame_too_large(&"😀".repeat(MAX_PROTOCOL_FRAME_BYTES / 4 + 1)));
        assert!(!first_protocol_frame_too_large(&format!("ok\n{maximum}")));
        assert_eq!(cap_cell_source("small"), "small");
        let source = "😀".repeat(MAX_CELL_SOURCE_CHARS + 1);
        assert!(cap_cell_source(&source).starts_with(&"😀".repeat(MAX_CELL_SOURCE_CHARS)));
        assert!(cap_cell_source(&source).ends_with("[... cell source truncated ...]"));
    }

    #[cfg(unix)]
    #[test]
    fn stderr_logs_are_private_after_creation_reopen_and_rotation() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("session");
        let path = parent.join("kernel-stderr.log");
        let old = parent.join("kernel-stderr.log.old");
        let manager = new_repl_kernel_manager(KernelManagerOptions {
            stderr_log_path: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        });
        let state = manager.state();
        drop(state.open_stderr_log().unwrap());
        assert_eq!(std::fs::metadata(&parent).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::write(&old, "old").unwrap();
        std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o644)).unwrap();
        drop(state.open_stderr_log().unwrap());
        assert_eq!(std::fs::metadata(&old).unwrap().permissions().mode() & 0o777, 0o600);
        std::fs::OpenOptions::new().write(true).open(&path).unwrap()
            .set_len(MAX_KERNEL_STDERR_LOG_BYTES + 1).unwrap();
        drop(state.open_stderr_log().unwrap());
        for file in [&path, &old] {
            assert_eq!(std::fs::metadata(file).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert_eq!(std::fs::metadata(&parent).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn background_settlement_notifies_once_after_last_matching_handle_and_on_teardown() {
        let notifications = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = notifications.clone();
        let manager = new_repl_kernel_manager(KernelManagerOptions {
            on_background_work_settled: Some(Arc::new(move || { counter.fetch_add(1, Ordering::SeqCst); })),
            ..Default::default()
        });
        let state = manager.state();
        let activity = |id: &str, pid: u32, active: bool| {
            state.handle_event(json!({"event":"display", "data": {
                BASH_ACTIVITY_DISPLAY_MIME: {"id": id, "pid": pid, "active": active}
            }}).as_object().unwrap()).unwrap();
        };
        let a = "a".repeat(32);
        let b = "b".repeat(32);
        activity(&a, 1, true);
        activity(&b, 2, true);
        activity(&a, 1, false);
        activity(&b, 99, false);
        assert!(state.has_background_work());
        assert_eq!(notifications.load(Ordering::SeqCst), 0);
        activity(&b, 2, false);
        activity(&b, 2, false);
        assert!(!state.has_background_work());
        assert_eq!(notifications.load(Ordering::SeqCst), 1);
        activity(&a, 1, true);
        state.cleanup_resources(None);
        state.cleanup_resources(None);
        assert_eq!(notifications.load(Ordering::SeqCst), 2);
    }

    /// G2-03: a failed stderr-log rotation must not cost the log. TS keeps the
    /// existing file, logs `cannot rotate kernel stderr log: ...` and the write
    /// budget is the file's remaining capacity (repl-manager.ts:302-313).
    #[test]
    fn rotation_failure_preserves_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kernel-stderr.log");
        let big = "x".repeat(MAX_KERNEL_STDERR_LOG_BYTES as usize + 1024);
        std::fs::write(&path, big.as_bytes()).unwrap();
        // An un-removable `.old`: rename onto a directory always fails.
        std::fs::create_dir(dir.path().join("kernel-stderr.log.old")).unwrap();
        let manager = new_repl_kernel_manager(KernelManagerOptions {
            stderr_log_path: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        });
        let state = manager.state();
        let log = state
            .open_stderr_log()
            .expect("rotation failure must not disable the stderr log");
        assert_eq!(log.budget, 0, "budget is the file's remaining capacity, not a fresh allowance");
        assert!(
            state.kernel_stderr.lock().unwrap().contains("cannot rotate kernel stderr log"),
            "rotation failure must be reported as a rotation diagnostic"
        );
    }

    /// G2-01: cleanup_resources must bound the Windows tree kill at
    /// TASKKILL_TIMEOUT_MS and fall back to signaling the child (TS
    /// repl-manager.ts:1412-1435). A hung taskkill must never hang teardown.
    // Windows-only: the fixture spawns `ping -n`, waits through a Win32
    // process handle, and asserts the taskkill fallback path.
    #[cfg(windows)]
    #[test]
    fn cleanup_resources_taskkill_is_bounded_and_falls_back() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let journal_path = dir.path().join("orphans.jsonl");
        std::env::set_var(crate::core::orphan_process_journal::ORPHAN_PROCESS_JOURNAL_ENV, &journal_path);

        let (helper_handle, helper_pid) = spawn_helper(20);
        let _ = helper_handle;
        let manager = new_repl_kernel_manager(KernelManagerOptions::default());
        let state = manager.state();
        let child_state = Arc::new(ChildState {
            id: 1,
            pid: Some(helper_pid as u32),
            stdin: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_destroyed: std::sync::atomic::AtomicBool::new(false),
            exit: ExitState::new(),
            destroy_stdout: Latch::new(),
            destroy_stderr: Latch::new(),
            stderr_closed: Latch::new(),
        });
        *state.child.lock().unwrap() = Some(child_state);
        *state.state.lock().unwrap() = State::Running;

        let hang = |command: &str, _args: &[String], _options: &SpawnOptions| {
            if command.to_lowercase().ends_with("taskkill.exe") {
                // Stand-in for a hung taskkill: a child that outlives the
                // cleanup deadline. The unbounded baseline path waits on it
                // forever; the bounded fix kills it at the timeout.
                let mut stand_in = std::process::Command::new("ping");
                stand_in.args(["-n", "60", "127.0.0.1"]);
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    stand_in.creation_flags(0x08000000);
                }
                stand_in
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());
                Some(stand_in.spawn())
            } else {
                None
            }
        };
        crate::utils::child_process::set_sync_spawn_override_for_tests(Some(hang));
        let (done_sender, done_receiver) = std::sync::mpsc::channel::<Duration>();
        std::thread::spawn({
            let state = state.clone();
            move || {
                let start = Instant::now();
                state.cleanup_resources(None);
                let _ = done_sender.send(start.elapsed());
            }
        });
        let elapsed = match done_receiver.recv_timeout(Duration::from_secs(30)) {
            Ok(elapsed) => elapsed,
            Err(_) => {
                crate::utils::child_process::set_sync_spawn_override_for_tests(None);
                std::env::remove_var(crate::core::orphan_process_journal::ORPHAN_PROCESS_JOURNAL_ENV);
                panic!("cleanup_resources hung; the Windows taskkill tree kill is unbounded");
            }
        };
        crate::utils::child_process::set_sync_spawn_override_for_tests(None);
        assert!(
            elapsed < Duration::from_secs(30),
            "cleanup_resources must be bounded, took {elapsed:?}"
        );
        // The fallback kill must have delivered: the helper is gone.
        assert!(
            wait_process_gone(helper_pid, Duration::from_secs(10)),
            "fallback kill did not terminate the kernel child"
        );
        // A failed tree cleanup preserves the recovery evidence: no inactive journal record.
        let records = read_journal_records(&journal_path);
        assert!(
            !records.iter().any(|record| record.pid == helper_pid as i64 && !record.active),
            "failed tree cleanup must not mark the orphan record inactive"
        );
        std::env::remove_var(crate::core::orphan_process_journal::ORPHAN_PROCESS_JOURNAL_ENV);
    }

    #[cfg(windows)]
    fn spawn_helper(seconds: u32) -> (tokio::process::Child, i32) {
        let handle = crate::utils::child_process::spawn_hidden(
            "ping",
            &["-n".to_string(), format!("{seconds}"), "127.0.0.1".to_string()],
            SpawnOptions::default(),
        )
        .expect("helper spawn");
        let pid = handle.child.id().expect("helper pid") as i32;
        (handle.child, pid)
    }

    #[cfg(windows)]
    fn read_journal_records(path: &std::path::Path) -> Vec<crate::core::orphan_process_journal::OrphanProcessRecord> {
        match std::fs::read_to_string(path) {
            Ok(contents) => contents
                .lines()
                .filter_map(|line| serde_json::from_str::<crate::core::orphan_process_journal::OrphanProcessRecord>(line).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Test liveness oracle: a terminated process whose handle is still open
    /// (tokio reaper) reports `STILL_ACTIVE`-independent existence via
    /// `process_id_exists`, so check the exit code like the OS task list does.
    #[cfg(windows)]
    fn probe_process_running(pid: i32) -> bool {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        const STILL_ACTIVE: u32 = 259;
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if handle.is_null() {
                return false;
            }
            let mut exit_code: u32 = 0;
            let ok = GetExitCodeProcess(handle, &mut exit_code) != 0;
            CloseHandle(handle);
            ok && exit_code == STILL_ACTIVE
        }
    }

    #[cfg(windows)]
    fn wait_process_gone(pid: i32, deadline: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < deadline {
            if !probe_process_running(pid) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        !probe_process_running(pid)
    }
}
