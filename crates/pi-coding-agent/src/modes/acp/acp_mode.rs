//! Port of packages/coding-agent/src/modes/acp/acp-mode.ts
//!
//! ACP (Agent Client Protocol) mode: one prime-agent session served over an ACP
//! JSON-RPC stream.
//!
//! blocked_on: `@agentclientprotocol/sdk` has no Rust port in this repository, so
//! the exact SDK surface `acp-mode.ts` consumes is reproduced privately at the top
//! of this file: `ndJsonStream`, `agent(...).onRequest/.onNotification/.connect`,
//! `PROTOCOL_VERSION`, `methods`, `RequestError` and `errorToResult`. Every name,
//! code, message and ordering rule is copied from the pinned SDK (1.3.0) dist
//! output; nothing below is a redesign. The SDK's zod param validation
//! (`onRequest` parses params before the handler runs) is not reproduced because
//! this repository has no zod port; handlers read the fields they use.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxFuture;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::config::VERSION;
use crate::core::output_guard::{take_over_stdout, write_raw_stdout};
use crate::core::prompt_admission::wait_for_prompt_admission;
use crate::modes::acp::acp_events::{acp_updates_for_session_event, AcpEventMappingState, AcpSessionUpdate};
use crate::modes::acp::acp_mcp::{resolve_acp_mcp_servers, AcpMcpServer, AcpRequestError};
use crate::modes::acp::acp_meta::{
    prime_agent_meta, PrimeAgentAutonomousMeta, PrimeAgentCwdMeta, PrimeAgentQuiescenceMeta,
    PrimeAgentSessionMeta, OUTCOME_ERROR, OUTCOME_RESULT, PHASE_EVENT, PHASE_RESPONSE_BOUNDARY,
    PHASE_TERMINAL_QUIESCENCE, PRIME_AGENT_META_NAMESPACE,
};
use crate::modes::acp::acp_stop_reason::acp_stop_reason;
use crate::modes::agent_connection::in_process_agent_connection::{
    InProcessAgentConnection, InProcessHeadlessExtensionOptions, InProcessRuntimeHost,
};
use crate::modes::agent_connection::types::*;
use crate::modes::headless_completion::latest_autonomous_gate_attempt;

// ---------------------------------------------------------------------------
// Private ACP SDK seam (see the module note)
// ---------------------------------------------------------------------------

/// `acp.PROTOCOL_VERSION`.
pub const ACP_PROTOCOL_VERSION: i64 = 1;

/// `acp.methods`.
pub mod acp_methods {
    pub mod agent {
        pub mod session {
            pub const NEW: &str = "session/new";
            pub const PROMPT: &str = "session/prompt";
            pub const CLOSE: &str = "session/close";
            pub const CANCEL: &str = "session/cancel";
        }
        pub const INITIALIZE: &str = "initialize";
    }
    pub mod client {
        pub mod session {
            pub const UPDATE: &str = "session/update";
        }
    }
}

/// A JS `Promise<void>`-shaped handle.
///
/// `acp-mode.ts` stores promises, awaits them more than once, and compares them
/// by identity (`entry.promptTask === promptTask`); `watch` gives the same
/// settle-once/await-many/identity behaviour.
pub struct AcpPromise {
    inner: Arc<AcpPromiseInner>,
}

struct AcpPromiseInner {
    tx: watch::Sender<Option<Result<(), String>>>,
    rx: watch::Receiver<Option<Result<(), String>>>,
}

impl Clone for AcpPromise {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// The external `resolve`/`reject` pair of a JS promise executor.
#[derive(Clone)]
pub struct AcpPromiseResolver {
    tx: watch::Sender<Option<Result<(), String>>>,
}

impl AcpPromise {
    pub fn new() -> (AcpPromise, AcpPromiseResolver) {
        let (tx, rx) = watch::channel(None);
        let promise = AcpPromise {
            inner: Arc::new(AcpPromiseInner { tx: tx.clone(), rx }),
        };
        (promise, AcpPromiseResolver { tx })
    }

    pub fn resolved() -> AcpPromise {
        let (promise, resolver) = AcpPromise::new();
        resolver.resolve();
        promise
    }

    /// `void (async () => { ... })()`: the work starts immediately.
    pub fn spawn<F>(future: F) -> AcpPromise
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let (promise, resolver) = AcpPromise::new();
        tokio::spawn(async move {
            future.await;
            resolver.resolve();
        });
        promise
    }

    pub async fn wait(&self) -> Result<(), String> {
        let mut rx = self.inner.rx.clone();
        // The `Ref` borrow must end before `rx` drops, so settle into a local first.
        let settled = match rx.wait_for(|value| value.is_some()).await {
            Ok(value) => (*value).clone(),
            Err(_) => None,
        };
        settled.unwrap_or(Ok(()))
    }

    pub fn is_settled(&self) -> bool {
        self.inner.rx.borrow().is_some()
    }

    /// `left === right` for the promise objects `acp-mode.ts` compares.
    pub fn ptr_eq(&self, other: &AcpPromise) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl AcpPromiseResolver {
    pub fn resolve(&self) {
        let _ = self.tx.send(Some(Ok(())));
    }

    pub fn reject(&self, error: impl Into<String>) {
        let _ = self.tx.send(Some(Err(error.into())));
    }
}

/// `acp.RequestError`: a JSON-RPC error object.
#[derive(Debug, Clone, PartialEq)]
pub struct AcpRequestErrorInfo {
    pub code: i64,
    pub message: String,
    pub data: Value,
}

impl AcpRequestErrorInfo {
    /// `RequestError.invalidParams({ reason })`.
    pub fn invalid_params(reason: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: "Invalid params".to_string(),
            data: json!({ "reason": reason.into() }),
        }
    }

    /// `RequestError.internalError(data)`.
    pub fn internal_error(data: Value) -> Self {
        Self {
            code: -32603,
            message: "Internal error".to_string(),
            data,
        }
    }

    /// `RequestError.methodNotFound(method)`.
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("\"Method not found\": {method}"),
            data: json!({ "method": method }),
        }
    }

    fn to_error_response(&self) -> Value {
        json!({ "code": self.code, "message": self.message, "data": self.data })
    }
}

/// A handler failure: a `RequestError`, a thrown `Error`, or a thrown non-Error.
#[derive(Debug, Clone, PartialEq)]
pub enum AcpHandlerError {
    Request(AcpRequestErrorInfo),
    /// A thrown `Error`: `errorDetails(error)` is its `message`.
    Thrown(String),
    /// A thrown non-Error; `errorDetails` is `undefined`, so `{}` is used.
    NonError,
}

impl From<String> for AcpHandlerError {
    fn from(message: String) -> Self {
        AcpHandlerError::Thrown(message)
    }
}

impl From<&str> for AcpHandlerError {
    fn from(message: &str) -> Self {
        AcpHandlerError::Thrown(message.to_string())
    }
}

impl From<AcpRequestErrorInfo> for AcpHandlerError {
    fn from(error: AcpRequestErrorInfo) -> Self {
        AcpHandlerError::Request(error)
    }
}

impl From<AcpRequestError> for AcpHandlerError {
    /// `throw RequestError.invalidParams({ reason })`.
    fn from(error: AcpRequestError) -> Self {
        AcpHandlerError::Request(AcpRequestErrorInfo::invalid_params(error.reason))
    }
}

/// `errorToResult(error)`.
fn error_to_result(error: AcpHandlerError) -> AcpRequestErrorInfo {
    match error {
        AcpHandlerError::Request(request) => request,
        AcpHandlerError::NonError => AcpRequestErrorInfo::internal_error(json!({})),
        AcpHandlerError::Thrown(details) => {
            // `try { internalError(JSON.parse(details ?? "")) } catch { internalError({ details }) }`
            match serde_json::from_str::<Value>(&details) {
                Ok(parsed) => AcpRequestErrorInfo::internal_error(parsed),
                Err(_) => AcpRequestErrorInfo::internal_error(json!({ "details": details })),
            }
        }
    }
}

/// `isJsonRpcResponse(message, requestId)`.
fn is_json_rpc_response(message: &Value, request_id: &Value) -> bool {
    let Some(record) = message.as_object() else {
        return false;
    };
    record.get("jsonrpc") == Some(&json!("2.0"))
        && record.get("id") == Some(request_id)
        && !record.contains_key("method")
        && record.contains_key("result") != record.contains_key("error")
}

/// One incoming JSON-RPC message.
#[derive(Debug, Clone, PartialEq)]
pub struct AcpIncomingMessage {
    pub method: Option<String>,
    pub id: Value,
    pub params: Value,
    /// `true` for a request (has an `id`), `false` for a notification.
    pub is_request: bool,
}

/// `acp.ndJsonStream(output, input)`.
///
/// Writes go through [`AcpStreamWritable`] so the wrapper in `run_acp_mode_with_connection`
/// can observe the exact outgoing response bytes; reads come from a channel fed by
/// the caller's input stream.
pub struct AcpStream {
    pub writable: AcpStreamWritable,
    pub reader: mpsc::UnboundedReceiver<AcpIncomingMessage>,
}

impl AcpStream {
    /// The TypeScript `Stream` shape is a plain object literal, so callers build one
    /// from a writer plus a parsed-message receiver.
    pub fn new(writable: AcpStreamWritable, reader: mpsc::UnboundedReceiver<AcpIncomingMessage>) -> Self {
        Self { writable, reader }
    }
}

/// Outgoing half of an ACP stream; `write` is the observed boundary.
#[derive(Clone)]
pub struct AcpStreamWritable {
    sink: Arc<dyn Fn(Value) -> BoxFuture<Result<(), String>> + Send + Sync>,
}

impl AcpStreamWritable {
    pub fn new(sink: Arc<dyn Fn(Value) -> BoxFuture<Result<(), String>> + Send + Sync>) -> Self {
        Self { sink }
    }

    pub fn write(&self, message: Value) -> BoxFuture<Result<(), String>> {
        (self.sink)(message)
    }
}

/// `writeRawStdout(JSON.stringify(header) + "\n")` for an ACP frame.
fn write_acp_frame(message: &Value) {
    write_raw_stdout(&format!("{}\n", serde_json::to_string(message).unwrap_or_else(|_| "null".to_string())));
}

/// `acp.ndJsonStream(rawStdoutSink(), readStdStream)`.
///
/// blocked_on: JS `ReadableStream`/`WritableStream` have no direct Rust type, so
/// the port models both halves explicitly: `writer` is the observed outgoing
/// boundary and `reader` is the parsed incoming channel.
pub fn nd_json_stream(
    writer: AcpStreamWritable,
    reader: mpsc::UnboundedReceiver<AcpIncomingMessage>,
) -> AcpStream {
    AcpStream {
        writable: writer,
        reader,
    }
}

/// The default transport: ACP frames on raw stdout, JSON-RPC objects on stdin.
pub fn default_acp_stream() -> AcpStream {
    let (tx, rx) = mpsc::unbounded_channel::<AcpIncomingMessage>();
    // stdin is read on a blocking thread and decoded as NDJSON.
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };
            if tx.send(parse_incoming(value)).is_err() {
                break;
            }
        }
    });
    nd_json_stream(
        AcpStreamWritable::new(Arc::new(|message: Value| {
            write_acp_frame(&message);
            Box::pin(async { Ok(()) }) as BoxFuture<Result<(), String>>
        })),
        rx,
    )
}

/// Split a wire JSON-RPC message into the fields the handlers read.
fn parse_incoming(message: Value) -> AcpIncomingMessage {
    let method = message.get("method").and_then(Value::as_str).map(str::to_string);
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    AcpIncomingMessage {
        is_request: method.is_some() && !id.is_null(),
        method,
        id,
        params,
    }
}

// ---------------------------------------------------------------------------
// Module-private helpers ported from acp-mode.ts
// ---------------------------------------------------------------------------

/// `normalizeWindowsDriveLetter(path)`.
fn normalize_windows_drive_letter(path: &str) -> String {
    if std::env::consts::OS != "windows" {
        return path.to_string();
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let mut out = path.to_string();
        let lowered = out[..1].to_ascii_lowercase();
        out.replace_range(0..1, &lowered);
        return out;
    }
    path.to_string()
}

/// Node `path.resolve(path)`.
///
/// blocked_on: `utils/paths.ts` keeps its `resolve` private, so this module
/// carries the same lexical resolution as a private helper (`resolve_path` in
/// `crate::utils::paths`).
fn resolve_absolute(path: &str) -> String {
    let candidate = std::path::Path::new(path);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else if path.is_empty() {
        std::env::current_dir().unwrap_or_default()
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(candidate)
    };
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in absolute.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy());
            }
            Component::RootDir => prefix.push(std::path::MAIN_SEPARATOR),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
        }
    }
    let joined = parts.join(&std::path::MAIN_SEPARATOR.to_string());
    if prefix.is_empty() {
        joined
    } else {
        format!("{prefix}{joined}")
    }
}

/// `canonicalCwd(path)`.
fn canonical_cwd(path: &str) -> String {
    let resolved = resolve_absolute(path);
    let canonical = std::fs::canonicalize(&resolved)
        .map(|value| value.to_string_lossy().to_string())
        // Preserve the previous lexical comparison when a path is missing or inaccessible.
        .unwrap_or(resolved);
    normalize_windows_drive_letter(&canonical)
}

/// `sameCwd(left, right)`.
///
/// The TypeScript compares `statSync(..., { bigint: true }).dev/.ino`. Rust reads
/// the same two numbers through `MetadataExt`: `st_dev`/`st_ino` on unix, and on
/// Windows the volume serial number plus the file index from `windows-sys`.
fn same_cwd(left: &str, right: &str) -> bool {
    let canonical_left = canonical_cwd(left);
    let canonical_right = canonical_cwd(right);
    if canonical_left == canonical_right {
        return true;
    }
    match (file_identity(&canonical_left), file_identity(&canonical_right)) {
        (Some(left_identity), Some(right_identity)) => left_identity == right_identity,
        _ => false,
    }
}

/// `statSync(path, { bigint: true })` -> `(dev, ino)` or `None` when it throws.
///
/// Either half being zero makes the pair untrustworthy: Windows path-based stat
/// can report dev 0 with a real ino, and comparing ino alone would match distinct
/// directories on different volumes, since file IDs are volume-local.
fn file_identity(path: &str) -> Option<(u64, u64)> {
    let metadata = std::fs::metadata(path).ok()?;
    let (dev, ino) = metadata_identity(path, &metadata)?;
    if dev == 0 || ino == 0 {
        return None;
    }
    Some((dev, ino))
}

#[cfg(unix)]
fn metadata_identity(_path: &str, metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(windows)]
fn metadata_identity(path: &str, _metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    windows_file_identity(path)
}

#[cfg(not(any(unix, windows)))]
fn metadata_identity(_path: &str, _metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    None
}

/// Windows side of `statSync(..., { bigint: true })`: the `volume_serial_number`
/// and `file_index` accessors are unstable std, so read the same pair from the
/// file handle the way Node does (`BY_HANDLE_FILE_INFORMATION`).
#[cfg(windows)]
fn windows_file_identity(path: &str) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    let wide: Vec<u16> = std::ffi::OsStr::new(path).encode_wide().chain(std::iter::once(0)).collect();
    unsafe {
        // `FILE_FLAG_BACKUP_SEMANTICS` is what makes directories openable.
        let handle = CreateFileW(
            wide.as_ptr(),
            FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            std::ptr::null_mut(),
        );
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut info = BY_HANDLE_FILE_INFORMATION::default();
        let ok = GetFileInformationByHandle(handle, &mut info);
        CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        let ino = ((info.nFileIndexHigh as u64) << 32) | info.nFileIndexLow as u64;
        Some((info.dwVolumeSerialNumber as u64, ino))
    }
}

/// `AcpModeOptions`.
#[derive(Default)]
pub struct AcpModeOptions {
    /// Bind headless extensions once the connection is live (in-process mode).
    pub bind_headless_extensions: Option<Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>>,
    /// Transport override. Defaults to NDJSON over stdio.
    pub stream: Option<AcpStream>,
    /// Skip claiming stdout when the caller supplies its own transport.
    pub own_stdout: Option<bool>,
}

/// `ctx.client.notify(method, params)`.
pub trait AcpClientNotify: Send + Sync {
    fn notify(&self, method: &str, params: Value) -> BoxFuture<Result<Value, String>>;
}

/// `AcpPendingTerminal`.
pub struct AcpPendingTerminal {
    pub prompt_turn_id: i64,
    pub boundary: TurnBoundary,
    pub outcome: &'static str,
    pub abort: Arc<CancellationToken>,
    pub status: Mutex<Option<AgentAutonomousStatus>>,
    pub turn_failure: Mutex<Option<String>>,
    pub failure: Mutex<Option<String>>,
    /// `pending.task`, assigned by `finalizePendingTerminal`.
    pub task: Mutex<Option<AcpPromise>>,
}

impl AcpPendingTerminal {
    fn new(prompt_turn_id: i64, boundary: TurnBoundary, outcome: &'static str, abort: Arc<CancellationToken>) -> Self {
        Self {
            prompt_turn_id,
            boundary,
            outcome,
            abort,
            status: Mutex::new(None),
            turn_failure: Mutex::new(None),
            failure: Mutex::new(None),
            task: Mutex::new(None),
        }
    }

    fn task(&self) -> Option<AcpPromise> {
        self.task.lock().expect("pending task poisoned").clone()
    }

    fn failure(&self) -> Option<String> {
        self.failure.lock().expect("pending failure poisoned").clone()
    }

    fn turn_failure(&self) -> Option<String> {
        self.turn_failure.lock().expect("turn failure poisoned").clone()
    }

    fn status(&self) -> Option<AgentAutonomousStatus> {
        self.status.lock().expect("pending status poisoned").clone()
    }
}

/// `AcpInputPauseRelease`.
#[derive(Clone)]
pub struct AcpInputPauseRelease {
    pub promise: AcpPromise,
    pub resolver: AcpPromiseResolver,
}

impl AcpInputPauseRelease {
    fn new() -> Self {
        let (promise, resolver) = AcpPromise::new();
        Self { promise, resolver }
    }
}

/// `AcpSessionEntry`.
pub struct AcpSessionEntry {
    pub id: String,
    pub abort: Mutex<Option<Arc<CancellationToken>>>,
    pub cancelling: AtomicBool,
    pub cancel_task: Mutex<Option<AcpPromise>>,
    pub stop_failure: Mutex<Option<String>>,
    pub input_pause: Mutex<Option<AgentConnectionSessionInputPause>>,
    pub input_pause_key: Mutex<Option<String>>,
    pub input_pause_release: Mutex<Option<AcpInputPauseRelease>>,
    pub pending_terminal: Mutex<Option<Arc<AcpPendingTerminal>>>,
    pub prompt_task: Mutex<Option<AcpPromise>>,
    pub resolve_prompt_task: Mutex<Option<AcpPromiseResolver>>,
    pub unsubscribe: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    pub producer: Arc<AcpUpdateProducer>,
}

impl AcpSessionEntry {
    fn abort(&self) -> Option<Arc<CancellationToken>> {
        self.abort.lock().expect("abort poisoned").clone()
    }

    fn stop_failure(&self) -> Option<String> {
        self.stop_failure.lock().expect("stop failure poisoned").clone()
    }

    fn pending_terminal(&self) -> Option<Arc<AcpPendingTerminal>> {
        self.pending_terminal.lock().expect("pending terminal poisoned").clone()
    }

    fn prompt_task(&self) -> Option<AcpPromise> {
        self.prompt_task.lock().expect("prompt task poisoned").clone()
    }

    fn input_pause_release(&self) -> Option<AcpInputPauseRelease> {
        self.input_pause_release
            .lock()
            .expect("input pause release poisoned")
            .clone()
    }
}

/// `AcpUpdateProducer`.
///
/// The sole producer of ACP session updates for one ACP session. ACP
/// notifications are asynchronous, so assigning an id at each call site is
/// insufficient: detached calls can be observed out of order. This producer
/// serializes publication and stamps the *delivered* order. Its phase/outcome
/// fields are application metadata, deliberately independent of ACP stop
/// reasons such as `end_turn`.
pub struct AcpUpdateProducer {
    session_id: String,
    client: Arc<dyn AcpClientNotify>,
    event_sequence: AtomicI64,
    next_prompt_turn_id: AtomicI64,
    active_prompt_turn_id: AtomicI64,
    tail: Mutex<AcpPromise>,
    child_origin_turn_ids: Mutex<HashMap<String, i64>>,
    terminal_child_origin_turns: Mutex<HashSet<i64>>,
    response_committed_turns: Mutex<HashSet<i64>>,
    terminal_lifecycle_turns: Mutex<HashSet<i64>>,
    finished_prompt_turns: Mutex<HashSet<i64>>,
    admission_ready: AcpPromise,
    release_admission: AcpPromiseResolver,
    admission_open: AtomicBool,
    admission_closed: AtomicBool,
}

impl AcpUpdateProducer {
    pub fn new(session_id: String, client: Arc<dyn AcpClientNotify>) -> Arc<Self> {
        let (admission_ready, release_admission) = AcpPromise::new();
        let producer = Arc::new(AcpUpdateProducer {
            session_id,
            client,
            event_sequence: AtomicI64::new(0),
            next_prompt_turn_id: AtomicI64::new(0),
            active_prompt_turn_id: AtomicI64::new(0),
            tail: Mutex::new(AcpPromise::resolved()),
            child_origin_turn_ids: Mutex::new(HashMap::new()),
            terminal_child_origin_turns: Mutex::new(HashSet::new()),
            response_committed_turns: Mutex::new(HashSet::new()),
            terminal_lifecycle_turns: Mutex::new(HashSet::new()),
            finished_prompt_turns: Mutex::new(HashSet::new()),
            admission_ready,
            release_admission,
            admission_open: AtomicBool::new(false),
            admission_closed: AtomicBool::new(false),
        });
        // Subscribe before the initial snapshot, but do not let that subscription
        // publish a session-bound update before session/new has replied.
        producer
    }

    /// `commitSessionNewResponse()`.
    pub fn commit_session_new_response(&self) {
        if self.admission_closed.load(Ordering::SeqCst) {
            return;
        }
        self.admission_open.store(true, Ordering::SeqCst);
        self.release_admission.resolve();
    }

    /// `failSessionNewAdmission()`.
    pub fn fail_session_new_admission(&self) {
        if self.admission_open.load(Ordering::SeqCst) || self.admission_closed.load(Ordering::SeqCst) {
            return;
        }
        self.admission_closed.store(true, Ordering::SeqCst);
        self.release_admission.resolve();
    }

    /// `beginPrompt()`.
    pub fn begin_prompt(&self) -> i64 {
        let turn_id = self.next_prompt_turn_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.active_prompt_turn_id.store(turn_id, Ordering::SeqCst);
        turn_id
    }

    fn cleanup_turn(&self, turn_id: i64) {
        self.response_committed_turns
            .lock()
            .expect("committed turns poisoned")
            .remove(&turn_id);
        let still_origin = self
            .child_origin_turn_ids
            .lock()
            .expect("child origin turns poisoned")
            .values()
            .any(|origin_turn_id| *origin_turn_id == turn_id);
        if !still_origin {
            self.terminal_child_origin_turns
                .lock()
                .expect("terminal child origin turns poisoned")
                .remove(&turn_id);
        }
    }

    /// `beginTerminalLifecycle(turnId)`.
    pub fn begin_terminal_lifecycle(&self, turn_id: i64) {
        self.terminal_lifecycle_turns
            .lock()
            .expect("terminal lifecycle turns poisoned")
            .insert(turn_id);
    }

    /// `finishPrompt(turnId)`.
    pub fn finish_prompt(&self, turn_id: i64) {
        if self.active_prompt_turn_id.load(Ordering::SeqCst) == turn_id {
            self.active_prompt_turn_id.store(0, Ordering::SeqCst);
        }
        if self
            .terminal_lifecycle_turns
            .lock()
            .expect("terminal lifecycle turns poisoned")
            .contains(&turn_id)
        {
            self.finished_prompt_turns
                .lock()
                .expect("finished prompt turns poisoned")
                .insert(turn_id);
            return;
        }
        self.cleanup_turn(turn_id);
    }

    /// `finishTerminalLifecycle(turnId)`.
    pub fn finish_terminal_lifecycle(&self, turn_id: i64) {
        self.terminal_lifecycle_turns
            .lock()
            .expect("terminal lifecycle turns poisoned")
            .remove(&turn_id);
        let finished = self
            .finished_prompt_turns
            .lock()
            .expect("finished prompt turns poisoned")
            .remove(&turn_id);
        if finished {
            self.cleanup_turn(turn_id);
        }
    }

    /// `commitResponse(turnId)`.
    ///
    /// Cut a scoreable terminal boundary before it is queued. A subscription
    /// callback after this point is connection-scoped, never appended to a turn
    /// that an evaluator may treat as terminal.
    pub fn commit_response(&self, turn_id: i64) {
        self.response_committed_turns
            .lock()
            .expect("committed turns poisoned")
            .insert(turn_id);
    }

    /// `isResponseCommitted(turnId)`.
    pub fn is_response_committed(&self, turn_id: i64) -> bool {
        self.response_committed_turns
            .lock()
            .expect("committed turns poisoned")
            .contains(&turn_id)
    }

    /// `sealTerminal(turnId)`.
    pub fn seal_terminal(&self, turn_id: i64) {
        self.commit_response(turn_id);
        let is_child_origin = self
            .child_origin_turn_ids
            .lock()
            .expect("child origin turns poisoned")
            .values()
            .any(|origin_turn_id| *origin_turn_id == turn_id);
        if is_child_origin {
            self.terminal_child_origin_turns
                .lock()
                .expect("terminal child origin turns poisoned")
                .insert(turn_id);
        }
        if self.active_prompt_turn_id.load(Ordering::SeqCst) == turn_id {
            self.active_prompt_turn_id.store(0, Ordering::SeqCst);
        }
    }

    /// `turnForEvent(event)`.
    pub fn turn_for_event(&self, event: &AgentConnectionSessionEvent) -> i64 {
        if let AgentConnectionSessionEvent::RlmChildUpdate { child } = event {
            let known = self
                .child_origin_turn_ids
                .lock()
                .expect("child origin turns poisoned")
                .get(&child.id)
                .copied();
            let origin_turn_id = known.unwrap_or_else(|| self.active_prompt_turn_id.load(Ordering::SeqCst));
            let turn_id = if self
                .terminal_child_origin_turns
                .lock()
                .expect("terminal child origin turns poisoned")
                .contains(&origin_turn_id)
            {
                0
            } else {
                origin_turn_id
            };
            let child_finished = ["done", "error", "cancelled"].contains(&child.status.as_str());
            if child_finished {
                self.child_origin_turn_ids
                    .lock()
                    .expect("child origin turns poisoned")
                    .remove(&child.id);
                let still_origin = self
                    .child_origin_turn_ids
                    .lock()
                    .expect("child origin turns poisoned")
                    .values()
                    .any(|origin| *origin == origin_turn_id);
                if !still_origin {
                    self.terminal_child_origin_turns
                        .lock()
                        .expect("terminal child origin turns poisoned")
                        .remove(&origin_turn_id);
                }
            } else if known.is_none() {
                // Remember its initial origin, including connection scope, so a later
                // child update cannot be relabelled by a subsequent prompt.
                self.child_origin_turn_ids
                    .lock()
                    .expect("child origin turns poisoned")
                    .insert(child.id.clone(), origin_turn_id);
            }
            return turn_id;
        }
        self.active_prompt_turn_id.load(Ordering::SeqCst)
    }

    /// `publish(update, turnId, phase, outcome?)`.
    pub async fn publish(
        self: &Arc<Self>,
        update: AcpSessionUpdate,
        turn_id: i64,
        phase: &str,
        outcome: Option<&str>,
    ) -> bool {
        // Admission is synchronous through the tail assignment below: close either
        // rejects this call here or drains the update after it joins the queue.
        if self.admission_closed.load(Ordering::SeqCst) {
            return false;
        }
        let event_sequence = self.event_sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let mut correlated_update = update.clone();
        let prior_meta = correlated_update
            .get("_meta")
            .filter(|value| value.is_object())
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let prior_prime_meta = prior_meta
            .get(PRIME_AGENT_META_NAMESPACE)
            .filter(|value| value.is_object())
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut prime_meta = prior_prime_meta;
        prime_meta.insert("promptTurnId".to_string(), json!(turn_id));
        prime_meta.insert("eventSequence".to_string(), json!(event_sequence));
        prime_meta.insert("phase".to_string(), Value::String(phase.to_string()));
        if let Some(outcome) = outcome {
            prime_meta.insert("outcome".to_string(), Value::String(outcome.to_string()));
        }
        let mut meta = prior_meta;
        meta.insert(PRIME_AGENT_META_NAMESPACE.to_string(), Value::Object(prime_meta));
        correlated_update.insert("_meta".to_string(), Value::Object(meta));

        // Keep the chain alive after a failed notification, while preserving the
        // order of every later notification and allowing callers to await its drain.
        let (link, resolver) = AcpPromise::new();
        let previous = {
            let mut tail = self.tail.lock().expect("producer tail poisoned");
            let previous = tail.clone();
            *tail = link.clone();
            previous
        };
        let published = Arc::new(AtomicBool::new(false));
        let producer = Arc::clone(self);
        let params = json!({
            "sessionId": producer.session_id.clone(),
            "update": correlated_update,
        });
        let published_flag = published.clone();
        tokio::spawn(async move {
            let _ = previous.wait().await;
            // Drop only this update; a rejected queue tail would strand later updates.
            let _ = async {
                producer.admission_ready.wait().await?;
                if !producer.admission_open.load(Ordering::SeqCst) {
                    return Ok::<(), String>(());
                }
                producer
                    .client
                    .notify(acp_methods::client::session::UPDATE, params)
                    .await?;
                published_flag.store(true, Ordering::SeqCst);
                Ok(())
            }
            .await;
            resolver.resolve();
        });
        let _ = link.wait().await;
        published.load(Ordering::SeqCst)
    }

    /// `drain()`.
    pub async fn drain(&self) {
        let tail = self.tail.lock().expect("producer tail poisoned").clone();
        let _ = tail.wait().await;
    }

    /// `close()`.
    pub async fn close(&self) {
        self.admission_closed.store(true, Ordering::SeqCst);
        self.release_admission.resolve();
        self.drain().await;
        self.admission_open.store(false, Ordering::SeqCst);
    }
}


/// `promptContent(blocks)`.
///
/// Split ACP prompt blocks into the text and images prime-agent accepts.
///
/// Image and embedded-resource blocks are advertised in `initialize`, so they must
/// actually reach the model: dropping them silently would let a client believe a
/// pasted screenshot was accepted.
pub fn prompt_content(blocks: &[Value]) -> (String, Vec<pi_ai::types::ImageContent>) {
    let mut texts: Vec<String> = Vec::new();
    let mut images: Vec<pi_ai::types::ImageContent> = Vec::new();
    for block in blocks {
        let Some(typed) = block.as_object() else {
            continue;
        };
        let block_type = typed.get("type").and_then(Value::as_str);
        match block_type {
            Some("text") => {
                if let Some(text) = typed.get("text").and_then(Value::as_str) {
                    texts.push(text.to_string());
                }
            }
            Some("image") => {
                if let (Some(data), Some(mime_type)) = (
                    typed.get("data").and_then(Value::as_str),
                    typed.get("mimeType").and_then(Value::as_str),
                ) {
                    images.push(pi_ai::types::ImageContent::new(data, mime_type));
                }
            }
            Some("resource") => {
                let resource = typed.get("resource").and_then(Value::as_object);
                if let Some(text) = resource.and_then(|resource| resource.get("text")).and_then(Value::as_str) {
                    // Embedded text resources become context the model can read.
                    let uri = resource
                        .and_then(|resource| resource.get("uri"))
                        .and_then(Value::as_str)
                        .map(|uri| format!("{uri}\n"))
                        .unwrap_or_default();
                    texts.push(format!("{uri}{text}"));
                }
            }
            Some("resource_link") => {
                if let Some(uri) = typed.get("uri").and_then(Value::as_str) {
                    texts.push(uri.to_string());
                }
            }
            _ => {}
        }
    }
    (texts.join("\n"), images)
}

/// `autonomousMeta(status)`.
fn autonomous_meta(status: Option<&AgentAutonomousStatus>) -> Option<PrimeAgentAutonomousMeta> {
    let status = status?;
    if !status.enabled {
        return None;
    }
    let gate_attempt = latest_autonomous_gate_attempt(status);
    Some(PrimeAgentAutonomousMeta {
        enabled: status.enabled,
        // The meta struct carries the wire integers the reference serializes.
        continuations_used: status.continuations_used as i64,
        turns_used: status.turns_used as i64,
        tokens_used: status.tokens_used as i64,
        gate_attempt: if gate_attempt == 0.0 {
            None
        } else {
            Some(gate_attempt as i64)
        },
        gate_failure: status.last_gate_failure.as_ref().map(|failure| failure.exit_text.clone()),
        limit_reason: None,
    })
}

/// `outstandingSubagentCount(children)`.
fn outstanding_subagent_count(children: Option<&[AgentConnectionRlmChildAgentSnapshot]>) -> usize {
    children
        .unwrap_or(&[])
        .iter()
        .filter(|child| child.status == "queued" || child.status == "running")
        .count()
}

/// `quiescenceMeta(status, children)`.
fn quiescence_meta(
    status: &AgentAutonomousStatus,
    children: Option<&[AgentConnectionRlmChildAgentSnapshot]>,
) -> PrimeAgentQuiescenceMeta {
    PrimeAgentQuiescenceMeta {
        outstanding_subagents: outstanding_subagent_count(children),
        remaining_autonomous_continuations: if status.enabled {
            (status.limits.max_continuations - status.continuations_used).max(0.0) as i64
        } else {
            0
        },
    }
}

/// `TurnBoundary`.
///
/// The transcript as it stood before a turn started, recorded so the turn's own
/// messages can be told apart from everything older.
///
/// A pre-turn message *count* cannot do that job: auto-compaction can fire during
/// a turn and rebuild `state.messages` (it filters, slices, and re-materializes
/// persisted entries), so this turn's failure can end up at a lower index than
/// the count taken before prompting. Membership is tracked by things a rebuild
/// preserves instead - the message objects themselves, plus a content key for
/// transports that hand back fresh copies (daemon RPC re-parses JSON, so identity
/// does not survive it) and for compaction paths that re-materialize a kept
/// message from its persisted entry with its original timestamp.
#[derive(Default)]
pub struct TurnBoundary {
    /// Membership keys of the messages that existed before the turn.
    ///
    /// blocked_on: the TypeScript `WeakSet<object>` identity half has no Rust
    /// referent - `AgentConnection::get_messages` returns messages by value, so a
    /// transcript read cannot hand back the same objects twice. The content key is
    /// the port's membership test, and it is the half the TypeScript comment above
    /// says survives daemon RPC re-parsing and compaction re-materialization.
    keys: HashSet<String>,
}

/// `messageKey(message)`.
///
/// Key for a kept message: compaction drops messages, it does not rewrite them.
fn message_key(message: &Value) -> Option<String> {
    let record = message.as_object()?;
    let timestamp = record.get("timestamp").and_then(Value::as_f64)?;
    let key = json!([
        record.get("role").cloned().unwrap_or(Value::Null),
        timestamp,
        record.get("stopReason").cloned().unwrap_or(Value::Null),
        record.get("errorMessage").cloned().unwrap_or(Value::Null),
    ]);
    Some(serde_json::to_string(&key).unwrap_or_default())
}

/// `turnBoundary(messages)`.
fn turn_boundary(messages: &[pi_agent_core::types::AgentMessage]) -> TurnBoundary {
    let mut boundary = TurnBoundary::default();
    for message in messages {
        let value = serde_json::to_value(message).unwrap_or(Value::Null);
        if !value.is_object() {
            continue;
        }
        if let Some(key) = message_key(&value) {
            boundary.keys.insert(key);
        }
    }
    boundary
}

/// `isPreTurn(message, boundary)`.
fn is_pre_turn(message: &Value, boundary: &TurnBoundary) -> bool {
    if !message.is_object() {
        return false;
    }
    match message_key(message) {
        Some(key) => boundary.keys.contains(&key),
        None => false,
    }
}

/// `turnFailure(connection, boundary)`.
///
/// Error text from an assistant message this turn produced, when it failed.
///
/// `promptAndWait` resolves for a failed turn just as it does for a successful
/// one, so the outcome has to be read off the transcript. Only messages that were
/// not in the transcript before the turn are considered: scanning the whole
/// transcript would let an earlier failed turn reject a later turn that never
/// called the model (a handled slash command, say), reporting a stale error.
///
/// A transcript read that fails is not treated as success - that would restore
/// the silent-success behavior this exists to prevent.
async fn turn_failure(connection: &Arc<dyn AgentConnection>, boundary: &TurnBoundary) -> Option<String> {
    let messages = match connection.get_messages().await {
        Ok(messages) => messages,
        Err(_) => return Some("the model request failed".to_string()),
    };
    let values: Vec<Value> = messages
        .iter()
        .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
        .collect();
    for index in (0..values.len()).rev() {
        let message = &values[index];
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        // The newest assistant message predates the turn, so the turn appended none.
        if is_pre_turn(message, boundary) {
            return None;
        }
        if message.get("stopReason").and_then(Value::as_str) != Some("error") {
            return None;
        }
        let error_message = message.get("errorMessage").and_then(Value::as_str).unwrap_or_default();
        return Some(if error_message.is_empty() {
            "the model request failed".to_string()
        } else {
            error_message.to_string()
        });
    }
    None
}

/// `ctx.client` - the ACP client half of the connection the producer notifies through.
struct AcpStreamClient {
    writable: AcpStreamWritable,
}

impl AcpClientNotify for AcpStreamClient {
    fn notify(&self, method: &str, params: Value) -> BoxFuture<Result<Value, String>> {
        let writable = self.writable.clone();
        let method = method.to_string();
        Box::pin(async move {
            writable
                .write(json!({
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                }))
                .await?;
            Ok(Value::Null)
        })
    }
}

/// `pendingSessionNewResponse`.
struct PendingSessionNewResponse {
    request_id: Value,
    producer: Arc<AcpUpdateProducer>,
    entry: Arc<AcpSessionEntry>,
    input_pause: Option<AgentConnectionSessionInputPause>,
}

/// Shared mode state; the TypeScript keeps these in the enclosing closure scope.
struct AcpModeState {
    connection: Arc<dyn AgentConnection>,
    bind_headless_extensions: Option<Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>>,
    supports_mcp_servers: bool,
    acp_mcp_owner_id: String,
    acp_mcp_server_names: Mutex<Vec<String>>,
    session: Mutex<Option<Arc<AcpSessionEntry>>>,
    closed_input_pause: Mutex<Option<AgentConnectionSessionInputPause>>,
    closed_input_pause_key: Mutex<Option<String>>,
    session_new_in_flight: AtomicBool,
    session_close_in_flight: AtomicBool,
    session_close_task: Mutex<Option<AcpPromise>>,
    bound: AtomicBool,
    pending_session_new_response: Mutex<Option<PendingSessionNewResponse>>,
    /// `disposed` and `unsubscribe` of `disposeConnection()`.
    disposed: AtomicBool,
    unsubscribe: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl AcpModeState {
    fn session(&self) -> Option<Arc<AcpSessionEntry>> {
        self.session.lock().expect("session poisoned").clone()
    }

    fn closed_input_pause(&self) -> Option<AgentConnectionSessionInputPause> {
        self.closed_input_pause.lock().expect("closed pause poisoned").clone()
    }

    fn closed_input_pause_key(&self) -> Option<String> {
        self.closed_input_pause_key.lock().expect("closed pause key poisoned").clone()
    }

    fn session_new_in_flight(&self) -> bool {
        self.session_new_in_flight.load(Ordering::SeqCst)
    }

    fn session_close_in_flight(&self) -> bool {
        self.session_close_in_flight.load(Ordering::SeqCst)
    }

    /// `clearAcpMcpServers(serverNames = acpMcpServerNames)`.
    async fn clear_acp_mcp_servers(&self, server_names: Option<Vec<String>>) -> Result<(), String> {
        let server_names = match server_names {
            Some(server_names) => server_names,
            None => self
                .acp_mcp_server_names
                .lock()
                .expect("mcp server names poisoned")
                .clone(),
        };
        if !self.supports_mcp_servers {
            return Ok(());
        }
        self.connection
            .release_acp_mcp_servers(&self.acp_mcp_owner_id, server_names)
            .await?;
        self.acp_mcp_server_names
            .lock()
            .expect("mcp server names poisoned")
            .clear();
        Ok(())
    }

    /// `replaceAcpMcpServers(servers, cwd)`.
    async fn replace_acp_mcp_servers(&self, servers: &[AcpMcpServer], cwd: &str) -> Result<(), AcpHandlerError> {
        if !self
            .acp_mcp_server_names
            .lock()
            .expect("mcp server names poisoned")
            .is_empty()
        {
            // Retry a prior best-effort close before admitting another session,
            // including one that does not declare replacement MCP servers.
            let _ = self.clear_acp_mcp_servers(None).await;
        }
        if servers.is_empty()
            && self
                .acp_mcp_server_names
                .lock()
                .expect("mcp server names poisoned")
                .is_empty()
        {
            return Ok(());
        }
        // `supportsMcpServers` in the TypeScript also checks that the connection
        // exposes `replaceAcpMcpServers`; the Rust trait always has the method, so
        // `supports_acp_mcp_servers()` is the whole gate here.
        if !self.supports_mcp_servers {
            return Err(AcpRequestErrorInfo::invalid_params("MCP servers are unavailable in this ACP host").into());
        }
        let resolved = resolve_acp_mcp_servers(servers, cwd)?;
        let server_names: Vec<String> = resolved.iter().map(|server| server.name().to_string()).collect();
        let values: Vec<Value> = resolved
            .iter()
            .map(|server| serde_json::to_value(server).unwrap_or(Value::Null))
            .collect();
        match self
            .connection
            .replace_acp_mcp_servers(values, &self.acp_mcp_owner_id)
            .await
        {
            Ok(()) => {}
            Err(error) => {
                // The daemon may have applied the configuration before its acknowledgement
                // was lost. Always attempt owner-scoped cleanup before rejecting admission.
                let _ = self.clear_acp_mcp_servers(Some(server_names)).await;
                return Err(AcpHandlerError::Thrown(error));
            }
        }
        *self
            .acp_mcp_server_names
            .lock()
            .expect("mcp server names poisoned") = server_names;
        Ok(())
    }

    /// `disposeConnection()`: idempotent, unsubscribes first.
    async fn dispose_connection(&self) {
        if self.disposed.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(unsubscribe) = self.unsubscribe.lock().expect("unsubscribe poisoned").take() {
            unsubscribe();
        }
        let _ = self.connection.dispose().await;
    }

    /// `failPendingSessionNewResponse()`.
    fn fail_pending_session_new_response(&self) {
        let admission = self
            .pending_session_new_response
            .lock()
            .expect("pending admission poisoned")
            .take();
        if let Some(admission) = admission {
            admission.producer.fail_session_new_admission();
            if let Some(release) = admission.entry.input_pause_release() {
                release
                    .resolver
                    .reject("ACP session/new response was not delivered");
            }
        }
    }

    /// The wrapped stream's `write`: observe the outgoing response boundary.
    async fn observe_outgoing_message(&self, message: &Value) {
        let matches = {
            let guard = self
                .pending_session_new_response
                .lock()
                .expect("pending admission poisoned");
            guard
                .as_ref()
                .map(|admission| is_json_rpc_response(message, &admission.request_id))
                .unwrap_or(false)
        };
        if !matches {
            return;
        }
        let admission = self
            .pending_session_new_response
            .lock()
            .expect("pending admission poisoned")
            .take();
        let Some(admission) = admission else {
            return;
        };
        if let Some(input_pause) = admission.input_pause.clone() {
            match input_pause.release().await {
                Ok(()) => {
                    {
                        let mut entry_pause = admission.entry.input_pause.lock().expect("input pause poisoned");
                        if entry_pause.as_ref().map(|pause| Arc::ptr_eq(pause, &input_pause)).unwrap_or(false) {
                            *entry_pause = None;
                            *admission.entry.input_pause_key.lock().expect("input pause key poisoned") = None;
                        }
                    }
                    {
                        let mut closed = self.closed_input_pause.lock().expect("closed pause poisoned");
                        if closed.as_ref().map(|pause| Arc::ptr_eq(pause, &input_pause)).unwrap_or(false) {
                            *closed = None;
                            *self
                                .closed_input_pause_key
                                .lock()
                                .expect("closed pause key poisoned") = None;
                        }
                    }
                    if let Some(release) = admission.entry.input_pause_release() {
                        release.resolver.resolve();
                    }
                    *admission
                        .entry
                        .input_pause_release
                        .lock()
                        .expect("input pause release poisoned") = None;
                }
                Err(error) => {
                    *admission
                        .entry
                        .stop_failure
                        .lock()
                        .expect("stop failure poisoned") = Some(error.clone());
                    if let Some(release) = admission.entry.input_pause_release() {
                        release.resolver.reject(error);
                    }
                }
            }
        }
        admission.producer.commit_session_new_response();
    }

    /// `cancelOutstandingRlmChildren()`.
    async fn cancel_outstanding_rlm_children(&self) -> Result<(), String> {
        let children = self.connection.get_rlm_child_snapshots().await?;
        let cancellations = futures::future::join_all(
            children
                .iter()
                .map(|child| self.connection.cancel_rlm_child(child.id.as_str())),
        )
        .await;
        for result in cancellations {
            if let Err(reason) = result {
                return Err(reason);
            }
        }
        Ok(())
    }

    /// `abortConnectionWork()`.
    async fn abort_connection_work(&self) -> Result<(), String> {
        self.connection.abort_and_clear_queue().await.map(|_| ())
    }

    /// `acquireStopInputPause(entry)`.
    async fn acquire_stop_input_pause(
        &self,
        entry: &Arc<AcpSessionEntry>,
    ) -> Result<AgentConnectionSessionInputPause, String> {
        if let Some(release) = entry.input_pause_release() {
            let _ = release.promise.wait().await;
            *entry.input_pause_release.lock().expect("input pause release poisoned") = None;
        }
        let lease_key = entry
            .input_pause_key
            .lock()
            .expect("input pause key poisoned")
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        *entry.input_pause_key.lock().expect("input pause key poisoned") = Some(lease_key.clone());
        let pause = self.connection.acquire_session_input_pause(&lease_key).await?;
        *entry.input_pause.lock().expect("input pause poisoned") = Some(pause.clone());
        Ok(pause)
    }

    /// `stopSessionWork(pending?, promptTask?)`.
    ///
    /// Every step rejects in the TypeScript, so the port propagates the first
    /// failure the same way (`await` inside the caller's `try`).
    async fn stop_session_work(
        &self,
        pending: Option<Arc<AcpPendingTerminal>>,
        prompt_task: Option<AcpPromise>,
    ) -> Result<(), String> {
        self.abort_connection_work().await?;
        self.connection.wait_for_idle().await?;
        self.cancel_outstanding_rlm_children().await?;
        if let Some(pending) = pending {
            if let Some(task) = pending.task() {
                let _ = task.wait().await;
            }
        }
        if let Some(prompt_task) = prompt_task {
            let _ = prompt_task.wait().await;
        }
        Ok(())
    }
}

/// `runAcpMode(runtimeHost)`.
///
/// blocked_on: `AgentSessionRuntime` (core/agent-session-runtime.ts) belongs to
/// another slice, so the port takes the runtime host seam the in-process
/// connection already consumes.
pub async fn run_acp_mode(host: Arc<dyn InProcessRuntimeHost>) -> Result<(), String> {
    let connection = Arc::new(InProcessAgentConnection::new(host));
    let bind = {
        let connection = connection.clone();
        Arc::new(move || connection.bind_headless_extensions(InProcessHeadlessExtensionOptions::default()))
            as Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>
    };
    run_acp_mode_with_connection(connection as Arc<dyn AgentConnection>, AcpModeOptions {
        bind_headless_extensions: Some(bind),
        stream: None,
        own_stdout: None,
    })
    .await
}

/// `runAcpModeWithConnection(connection, options = {})`.
pub async fn run_acp_mode_with_connection(
    connection: Arc<dyn AgentConnection>,
    options: AcpModeOptions,
) -> Result<(), String> {
    // ACP owns stdout: any stray write corrupts the JSON-RPC stream.
    if options.own_stdout.unwrap_or(true) && options.stream.is_none() {
        take_over_stdout();
    }
    let supports_mcp_servers = connection.supports_acp_mcp_servers();
    let state = Arc::new(AcpModeState {
        connection: connection.clone(),
        bind_headless_extensions: options.bind_headless_extensions.clone(),
        supports_mcp_servers,
        acp_mcp_owner_id: uuid::Uuid::new_v4().to_string(),
        acp_mcp_server_names: Mutex::new(Vec::new()),
        session: Mutex::new(None),
        closed_input_pause: Mutex::new(None),
        closed_input_pause_key: Mutex::new(None),
        session_new_in_flight: AtomicBool::new(false),
        session_close_in_flight: AtomicBool::new(false),
        session_close_task: Mutex::new(None),
        bound: AtomicBool::new(false),
        pending_session_new_response: Mutex::new(None),
        disposed: AtomicBool::new(false),
        unsubscribe: Mutex::new(None),
    });

    // One ACP connection drives one AgentConnection, whose newSession() replaces
    // the live session rather than creating a parallel one. Tracking a single
    // session keeps every event unambiguously attributable; a second session/new
    // is refused rather than silently sharing conversation state, cwd, and queues.

    let mut options = options;
    let caller_owned_transport = options.stream.is_some();
    let base_stream = options.stream.take().unwrap_or_else(default_acp_stream);
    // ACP's public request handler only returns a response; it has no response
    // commit callback. Observe the outgoing response at the supplied stream
    // boundary instead. The SDK serializes every write, so opening the producer
    // after this write resolves puts buffered notifications strictly behind it.
    let base_writable = base_stream.writable.clone();
    let observing_state = state.clone();
    let stream = AcpStream {
        writable: AcpStreamWritable::new(Arc::new(move |message: Value| {
            let base_writable = base_writable.clone();
            let state = observing_state.clone();
            Box::pin(async move {
                if let Err(error) = base_writable.write(message.clone()).await {
                    state.fail_pending_session_new_response();
                    return Err(error);
                }
                state.observe_outgoing_message(&message).await;
                Ok(())
            }) as BoxFuture<Result<(), String>>
        })),
        reader: base_stream.reader,
    };

    let app = Arc::new(AcpAgentApp {
        state: state.clone(),
        client: Arc::new(AcpStreamClient {
            writable: stream.writable.clone(),
        }) as Arc<dyn AcpClientNotify>,
        writable: stream.writable.clone(),
    });

    // Signals: only the process host exits; a caller-supplied transport does not.
    let signal_cleanup_handlers = spawn_acp_signal_handlers(state.clone(), caller_owned_transport);

    let disconnect = app.connect(stream.reader).await;

    // Exit when the client disconnects (stdin EOF or a closed transport). Blocking
    // forever would leave an orphaned agent per run.
    let _ = disconnect.wait().await;
    let session = state.session();
    if let Some(session) = &session {
        if let Some(abort) = session.abort() {
            abort.cancel();
        }
        if let Some(unsubscribe) = session.unsubscribe.lock().expect("unsubscribe poisoned").take() {
            unsubscribe();
        }
        if let Some(pause) = session.input_pause.lock().expect("input pause poisoned").clone() {
            let _ = pause.release().await;
        }
    }
    *state.session.lock().expect("session poisoned") = None;
    if let Some(pause) = state.closed_input_pause() {
        let _ = pause.release().await;
    }
    *state.closed_input_pause.lock().expect("closed pause poisoned") = None;
    *state.closed_input_pause_key.lock().expect("closed pause key poisoned") = None;
    let _ = state.clear_acp_mcp_servers(None).await;
    state.dispose_connection().await;
    for cleanup in signal_cleanup_handlers {
        cleanup.abort();
    }
    // Only the real stdio entrypoint owns the process; a caller-supplied transport
    // (tests, embedding) must never have its host exited from under it.
    if caller_owned_transport {
        return Ok(());
    }
    // `return process.exit(0) as never`.
    std::process::exit(0);
}

/// `for (const signal of ["SIGINT","SIGTERM", ...(win32 ? [] : ["SIGHUP"])])`.
///
/// Returns one abort handle per registered handler; aborting it runs the
/// `signalCleanupHandlers` entry (`process.off(signal, handler)`).
fn spawn_acp_signal_handlers(
    state: Arc<AcpModeState>,
    caller_owned_transport: bool,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();
    for signal in acp_signal_names() {
        let handler = {
            let state = state.clone();
            let signal = signal.to_string();
            move || {
                let state = state.clone();
                let signal = signal.clone();
                tokio::spawn(async move {
                    crate::utils::shell::kill_tracked_detached_children();
                    let code = if signal == "SIGINT" {
                        130
                    } else if signal == "SIGHUP" {
                        129
                    } else {
                        143
                    };
                    dispose_connection_for_exit(&state).await;
                    // `process.exit(exitCode)`: a caller-supplied transport is not exited.
                    if !caller_owned_transport {
                        std::process::exit(code);
                    }
                });
            }
        };
        if let Some(handle) = listen_for_signal(signal, Arc::new(handler)) {
            handles.push(handle);
        }
    }
    handles
}

fn acp_signal_names() -> Vec<&'static str> {
    let mut signals = vec!["SIGINT", "SIGTERM"];
    if std::env::consts::OS != "windows" {
        signals.push("SIGHUP");
    }
    signals
}

/// `disposeConnection()` from the signal handler.
async fn dispose_connection_for_exit(state: &AcpModeState) {
    state.dispose_connection().await;
}

#[cfg(unix)]
fn listen_for_signal(
    signal: &'static str,
    handler: Arc<dyn Fn() + Send + Sync>,
) -> Option<tokio::task::JoinHandle<()>> {
    use tokio::signal::unix::{signal as unix_signal, SignalKind};
    let kind = match signal {
        "SIGINT" => SignalKind::interrupt(),
        "SIGTERM" => SignalKind::terminate(),
        "SIGHUP" => SignalKind::hangup(),
        _ => return None,
    };
    let mut stream = unix_signal(kind).ok()?;
    let handle = tokio::spawn(async move {
        if stream.recv().await.is_some() {
            handler();
        }
    });
    Some(handle)
}

#[cfg(not(unix))]
fn listen_for_signal(
    signal: &'static str,
    handler: Arc<dyn Fn() + Send + Sync>,
) -> Option<tokio::task::JoinHandle<()>> {
    // Node allows listening for all three names on Windows; SIGTERM/SIGHUP are
    // never delivered there, so the port registers the same handler only for
    // SIGINT through the console handler.
    if signal != "SIGINT" {
        return None;
    }
    let handle = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            handler();
        }
    });
    Some(handle)
}

/// `acp.agent({ name: "prime-agent" })` with the four registered handlers, plus
/// `connect(stream)`.
struct AcpAgentApp {
    state: Arc<AcpModeState>,
    client: Arc<dyn AcpClientNotify>,
    /// The same wrapped stream writable: responses and notifications share one
    /// serialized writer, exactly like the SDK connection's single writer.
    writable: AcpStreamWritable,
}

impl AcpAgentApp {
    /// `.onRequest("initialize", ...)`.
    async fn handle_initialize(&self) -> Value {
        let supports_mcp_servers = self.state.supports_mcp_servers;
        let mut agent_capabilities = json!({
            "loadSession": false,
            "promptCapabilities": { "image": true, "embeddedContext": true },
            "sessionCapabilities": { "close": {} },
        });
        if supports_mcp_servers {
            if let Some(object) = agent_capabilities.as_object_mut() {
                object.insert("mcpCapabilities".to_string(), json!({ "http": true }));
            }
        }
        json!({
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "agentCapabilities": agent_capabilities,
            "agentInfo": { "name": "prime-agent", "title": "Optimus", "version": VERSION },
            // Advertise prime-agent extras under a namespaced key: ACP reserves
            // every object root for future protocol fields.
            "_meta": prime_agent_meta(PrimeAgentSessionMeta::new()),
        })
    }

    /// `.onRequest("session/new", ...)`.
    async fn handle_session_new(&self, ctx: &AcpRequestContext) -> Result<Value, AcpHandlerError> {
        // Reserve the single-session slot before the first await. Otherwise two
        // concurrent requests can both pass the empty-slot check while cwd or
        // snapshot reads are in flight, then overwrite each other's session.
        if self.state.session().is_some()
            || self.state.session_new_in_flight()
            || self.state.session_close_in_flight()
        {
            return Err(AcpHandlerError::Thrown(
                "Optimus ACP mode hosts one session per connection; \
                 start another optimus-agent process for a second session"
                    .to_string(),
            ));
        }
        self.state.session_new_in_flight.store(true, Ordering::SeqCst);
        let result = self.session_new_inner(ctx).await;
        self.state.session_new_in_flight.store(false, Ordering::SeqCst);
        result
    }

    async fn session_new_inner(&self, ctx: &AcpRequestContext) -> Result<Value, AcpHandlerError> {
        let mut mcp_servers: Vec<AcpMcpServer> = Vec::new();
        if let Some(list) = ctx.params.get("mcpServers").and_then(Value::as_array) {
            for entry in list {
                mcp_servers.push(serde_json::from_value(entry.clone()).unwrap_or_default());
            }
        }
        if !mcp_servers.is_empty() && !self.state.supports_mcp_servers {
            return Err(AcpRequestErrorInfo::invalid_params("MCP servers are unavailable in this ACP host").into());
        }
        if !self.state.bound.load(Ordering::SeqCst) {
            // Only latch after a successful bind: a rejected bind must not leave
            // extensions permanently unavailable for the rest of the process.
            if let Some(bind) = &self.state.bind_headless_extensions {
                (bind)().await.map_err(AcpHandlerError::Thrown)?;
            }
            self.state.bound.store(true, Ordering::SeqCst);
        }
        // prime-agent's cwd is fixed at startup by the session it was launched
        // with, so a client-supplied cwd cannot be adopted after the fact.
        // Report the real cwd back in `_meta` rather than failing the request or
        // letting the client assume a directory the agent is not using.
        let requested_cwd = ctx.params.get("cwd").cloned();
        let actual_cwd = self.state.connection.get_state().await.ok().map(|state| state.cwd);
        let has_stdio = mcp_servers.iter().any(|server| server.command.is_some());
        if actual_cwd.is_none() && has_stdio {
            return Err(
                AcpRequestErrorInfo::invalid_params("Could not resolve the ACP session cwd for stdio MCP").into(),
            );
        }
        self.state
            .replace_acp_mcp_servers(&mcp_servers, actual_cwd.as_deref().unwrap_or(""))
            .await?;
        let cwd_mismatch = match (requested_cwd.as_ref().and_then(Value::as_str), actual_cwd.as_ref()) {
            (Some(requested), Some(actual)) if !requested.is_empty() => {
                if same_cwd(requested, actual) {
                    None
                } else {
                    Some(PrimeAgentCwdMeta {
                        requested: requested.to_string(),
                        actual: actual.clone(),
                    })
                }
            }
            _ => None,
        };
        let session_id = uuid::Uuid::new_v4().to_string();
        // Install the listener before fetching the snapshot. Child updates can arrive
        // while the snapshot request is in flight; the connection remains the
        // authoritative source used when quiescence is emitted below.
        let producer = AcpUpdateProducer::new(session_id.clone(), self.client.clone());
        let closed_input_pause = self.state.closed_input_pause();
        let input_pause_release = if closed_input_pause.is_some() {
            Some(AcpInputPauseRelease::new())
        } else {
            None
        };
        let entry = Arc::new(AcpSessionEntry {
            id: session_id.clone(),
            abort: Mutex::new(None),
            cancelling: AtomicBool::new(false),
            cancel_task: Mutex::new(None),
            stop_failure: Mutex::new(None),
            input_pause: Mutex::new(closed_input_pause.clone()),
            input_pause_key: Mutex::new(self.state.closed_input_pause_key()),
            input_pause_release: Mutex::new(input_pause_release),
            pending_terminal: Mutex::new(None),
            prompt_task: Mutex::new(None),
            resolve_prompt_task: Mutex::new(None),
            unsubscribe: Mutex::new(None),
            producer: producer.clone(),
        });
        // Subscribe for the session lifetime, not per prompt turn: prime-agent
        // subagents are fire-and-forget and keep reporting after the spawning turn
        // ends, so a turn-scoped subscription would drop their updates. One
        // mapping state per session keeps streaming bash output correlated with
        // the run that produced it.
        let mapping_state: Arc<Mutex<AcpEventMappingState>> = Arc::new(Mutex::new(AcpEventMappingState::default()));
        let observed_children: Arc<Mutex<HashMap<String, Value>>> = Arc::new(Mutex::new(HashMap::new()));
        let subscribe_producer = producer.clone();
        let subscribe_state = mapping_state.clone();
        let subscribe_children = observed_children.clone();
        let unsubscribe: Arc<dyn Fn() + Send + Sync> = Arc::from(self.state.connection.subscribe(Arc::new(move |event: AgentConnectionEvent| {
            let producer = subscribe_producer.clone();
            let mapping_state = subscribe_state.clone();
            let observed_children = subscribe_children.clone();
            Box::pin(async move {
                dispatch_acp_connection_event(producer, mapping_state, observed_children, event).await;
            })
        })));
        {
            // Reconcile after subscribing so updates cannot be lost while the snapshot
            // request is in flight. Do not turn a failed read into an empty roster.
            let initial_snapshot = self.state.connection.get_initial_snapshot().await;
            match initial_snapshot {
                Ok(snapshot) => {
                    for child in snapshot.children.unwrap_or_default() {
                        // Scope the guard so it cannot be held across the publish await.
                        let already_observed = {
                            let mut children = observed_children.lock().expect("observed children poisoned");
                            if children.contains_key(&child.id) {
                                true
                            } else {
                                children.insert(
                                    child.id.clone(),
                                    serde_json::to_value(&child).unwrap_or(Value::Null),
                                );
                                false
                            }
                        };
                        if already_observed {
                            continue;
                        }
                        let event = AgentConnectionSessionEvent::RlmChildUpdate { child };
                        let turn_id = producer.turn_for_event(&event);
                        let updates = {
                            let mut state = mapping_state.lock().expect("mapping poisoned");
                            acp_updates_for_session_event(&event, &mut state)
                        };
                        for update in updates {
                            producer.publish(update, turn_id, PHASE_EVENT, None).await;
                        }
                    }
                }
                Err(error) => {
                    producer.fail_session_new_admission();
                    unsubscribe();
                    let _ = self.state.clear_acp_mcp_servers(None).await;
                    return Err(AcpHandlerError::Thrown(error));
                }
            }
        }
        // Claim the single-session slot only once the subscription and snapshot are
        // ready, so a failed setup cannot leave it occupied and unusable.
        *entry.unsubscribe.lock().expect("unsubscribe poisoned") = Some(unsubscribe.clone());
        *self.state.unsubscribe.lock().expect("unsubscribe poisoned") = Some(unsubscribe);
        *self.state.session.lock().expect("session poisoned") = Some(entry.clone());
        let mut response = json!({ "sessionId": session_id });
        if let Some(cwd_mismatch) = cwd_mismatch {
            let mut meta = PrimeAgentSessionMeta::new();
            meta.cwd = Some(cwd_mismatch);
            response["_meta"] = Value::Object(prime_agent_meta(meta));
        }
        // The stream wrapper commits this gate after this exact response has
        // written. Buffered subscription updates retain producer order.
        let closed_pause = self.state.closed_input_pause();
        *self
            .state
            .pending_session_new_response
            .lock()
            .expect("pending admission poisoned") = Some(PendingSessionNewResponse {
            request_id: ctx.request_id.clone(),
            producer: entry.producer.clone(),
            entry: entry.clone(),
            input_pause: closed_pause,
        });
        Ok(response)
    }
}

/// The `void producer.publish(...)` subscription callback.
async fn dispatch_acp_connection_event(
    producer: Arc<AcpUpdateProducer>,
    mapping_state: Arc<Mutex<AcpEventMappingState>>,
    observed_children: Arc<Mutex<HashMap<String, Value>>>,
    event: AgentConnectionEvent,
) {
    // Heartbeats are connection-scoped, including if one races a prompt.
    // They therefore intentionally use origin turn 0.
    if matches!(event, AgentConnectionEvent::HeartbeatsChanged) {
        let mut meta = PrimeAgentSessionMeta::new();
        meta.heartbeats_changed = Some(true);
        let mut update = Map::new();
        update.insert("sessionUpdate".to_string(), json!("session_info_update"));
        update.insert("_meta".to_string(), Value::Object(prime_agent_meta(meta)));
        producer.publish(update, 0, PHASE_EVENT, None).await;
        return;
    }
    let AgentConnectionEvent::SessionEvent { event } = event else {
        return;
    };
    if let AgentConnectionSessionEvent::RlmChildUpdate { child } = &event {
        observed_children
            .lock()
            .expect("observed children poisoned")
            .insert(child.id.clone(), serde_json::to_value(child).unwrap_or(Value::Null));
    }
    let turn_id = producer.turn_for_event(&event);
    let updates = {
        let mut state = mapping_state.lock().expect("mapping poisoned");
        acp_updates_for_session_event(&event, &mut state)
    };
    for update in updates {
        producer.publish(update, turn_id, PHASE_EVENT, None).await;
    }
}

/// One ACP request context: `ctx.params`, `ctx.requestId`, `ctx.client`.
struct AcpRequestContext {
    params: Value,
    request_id: Value,
}

impl AcpAgentApp {
    /// `.onRequest("session/prompt", ...)`.
    async fn handle_session_prompt(&self, ctx: &AcpRequestContext) -> Result<Value, AcpHandlerError> {
        let params = &ctx.params;
        let requested_session_id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
        let prompt_blocks: Vec<Value> = params
            .get("prompt")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let entry = match self.state.session() {
            Some(entry) if entry.id == requested_session_id => entry,
            _ => {
                return Err(AcpHandlerError::Thrown(format!(
                    "Unknown ACP session: {requested_session_id}"
                )))
            }
        };
        if self.state.session_close_in_flight() {
            return Err(AcpHandlerError::Thrown(format!(
                "ACP session is closing: {requested_session_id}"
            )));
        }
        if entry.cancelling.load(Ordering::SeqCst) {
            return Err(AcpHandlerError::Thrown(format!(
                "ACP session is cancelling: {requested_session_id}"
            )));
        }
        if let Some(release) = entry.input_pause_release() {
            let _ = release.promise.wait().await;
        }
        // A prompt response precedes its correlated terminal update. Serialize the
        // next turn behind that lifecycle so it cannot overwrite terminal ownership.
        if let Some(pending) = entry.pending_terminal() {
            if let Some(task) = pending.task() {
                let _ = task.wait().await;
            }
        }
        if self.state.session().map(|current| !Arc::ptr_eq(&current, &entry)).unwrap_or(true) {
            return Err(AcpHandlerError::Thrown(format!(
                "Unknown ACP session: {requested_session_id}"
            )));
        }
        if self.state.session_close_in_flight() {
            return Err(AcpHandlerError::Thrown(format!(
                "ACP session is closing: {requested_session_id}"
            )));
        }
        // This prompt was admitted before the cancellation started; it is dropped
        // by the cancel rather than malformed, so report the protocol stop reason
        // instead of a request error.
        if entry.cancelling.load(Ordering::SeqCst) {
            return Ok(json!({ "stopReason": acp_stop_reason(true, None) }));
        }
        if let Some(stop_failure) = entry.stop_failure() {
            return Err(AcpHandlerError::Thrown(format!(
                "ACP session stop failed: {stop_failure}"
            )));
        }
        if let Some(pending) = entry.pending_terminal() {
            if let Some(failure) = pending.failure() {
                return Err(AcpHandlerError::Thrown(format!(
                    "ACP lifecycle reconciliation failed: {failure}"
                )));
            }
        }
        if entry.abort().is_some() {
            return Err(AcpHandlerError::Thrown(
                "A prompt turn is already running for this ACP session".to_string(),
            ));
        }

        let abort = Arc::new(CancellationToken::new());
        *entry.abort.lock().expect("abort poisoned") = Some(abort.clone());
        let (prompt_task, resolve_prompt_task) = AcpPromise::new();
        *entry.prompt_task.lock().expect("prompt task poisoned") = Some(prompt_task.clone());
        *entry.resolve_prompt_task.lock().expect("prompt resolve poisoned") = Some(resolve_prompt_task.clone());
        // Allocate the causal turn before the first await, not when an update is
        // delivered. This prevents late producer events becoming the next turn.
        let prompt_turn_id = entry.producer.begin_prompt();
        let mut response_boundary_emitted = false;
        let mut terminal_settlement_cancelled = false;
        let result = self
            .prompt_inner(
                &entry,
                &prompt_blocks,
                &abort,
                prompt_turn_id,
                &mut response_boundary_emitted,
                &mut terminal_settlement_cancelled,
            )
            .await;
        entry.producer.finish_prompt(prompt_turn_id);
        {
            let mut prompt_task_slot = entry.prompt_task.lock().expect("prompt task poisoned");
            if prompt_task_slot
                .as_ref()
                .map(|task| task.ptr_eq(&prompt_task))
                .unwrap_or(false)
            {
                *prompt_task_slot = None;
                *entry.resolve_prompt_task.lock().expect("prompt resolve poisoned") = None;
                resolve_prompt_task.resolve();
            }
        }
        {
            let abort_slot = entry.abort();
            let pending_uses_this_abort = entry
                .pending_terminal()
                .map(|pending| Arc::ptr_eq(&pending.abort, &abort))
                .unwrap_or(false);
            if abort_slot
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &abort))
                .unwrap_or(false)
                && !pending_uses_this_abort
            {
                *entry.abort.lock().expect("abort poisoned") = None;
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    async fn prompt_inner(
        &self,
        entry: &Arc<AcpSessionEntry>,
        prompt_blocks: &[Value],
        abort: &Arc<CancellationToken>,
        prompt_turn_id: i64,
        response_boundary_emitted: &mut bool,
        terminal_settlement_cancelled: &mut bool,
    ) -> Result<Value, AcpHandlerError> {
        let outcome = self
            .prompt_turn(entry, prompt_blocks, abort, prompt_turn_id, response_boundary_emitted, terminal_settlement_cancelled)
            .await;
        match outcome {
            Ok(value) => Ok(value),
            Err(error) => {
                if abort.is_cancelled() && !entry.producer.is_response_committed(prompt_turn_id) {
                    entry.producer.drain().await;
                    return Ok(json!({ "stopReason": acp_stop_reason(true, None) }));
                }
                // Failed prompt/snapshot admission gets one correlated error boundary;
                // it never gets an invented terminal-quiescence update.
                if !*response_boundary_emitted {
                    let mut meta = PrimeAgentSessionMeta::new();
                    meta.terminal_quiescence_expected = Some(false);
                    let mut update = Map::new();
                    update.insert("sessionUpdate".to_string(), json!("session_info_update"));
                    update.insert("_meta".to_string(), Value::Object(prime_agent_meta(meta)));
                    entry
                        .producer
                        .publish(update, prompt_turn_id, PHASE_RESPONSE_BOUNDARY, Some(OUTCOME_ERROR))
                        .await;
                }
                entry.producer.drain().await;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn prompt_turn(
        &self,
        entry: &Arc<AcpSessionEntry>,
        prompt_blocks: &[Value],
        abort: &Arc<CancellationToken>,
        prompt_turn_id: i64,
        response_boundary_emitted: &mut bool,
        terminal_settlement_cancelled: &mut bool,
    ) -> Result<Value, AcpHandlerError> {
        let cancelled = || json!({ "stopReason": acp_stop_reason(true, None) });
        let (text, images) = prompt_content(prompt_blocks);
        let prior_messages = turn_boundary(
            &self
                .state
                .connection
                .get_messages()
                .await
                .map_err(AcpHandlerError::Thrown)?,
        );
        if abort.is_cancelled() {
            entry.producer.drain().await;
            return Ok(cancelled());
        }
        // A follow-up prompt can arrive while injected work (subagent replies,
        // heartbeats) keeps the resident session busy. ACP has no native queue
        // field, so queue the host turn behind that work with follow-up
        // semantics instead of rejecting it as "Agent is already processing".
        let prompt_options = AgentConnectionPromptOptions {
            images: if images.is_empty() { None } else { Some(images) },
            streaming_behavior: Some("followUp".to_string()),
            queue_if_busy: Some(true),
            source: None,
        };
        let prompt_future = self.state.connection.prompt_and_wait(&text, Some(prompt_options));
        match wait_for_prompt_admission(prompt_future, Some((**abort).clone())).await {
            Ok(result) => result.map_err(AcpHandlerError::Thrown)?,
            Err(_) => {
                entry.producer.drain().await;
                return Ok(cancelled());
            }
        }
        if abort.is_cancelled() {
            entry.producer.drain().await;
            return Ok(cancelled());
        }
        let status = self
            .state
            .connection
            .wait_for_headless_completion(None)
            .await
            .map_err(AcpHandlerError::Thrown)?;
        if abort.is_cancelled() {
            entry.producer.drain().await;
            return Ok(cancelled());
        }
        let failure = turn_failure(&self.state.connection, &prior_messages).await;
        if abort.is_cancelled() {
            entry.producer.drain().await;
            return Ok(cancelled());
        }
        let autonomous = autonomous_meta(Some(&status));
        let live_children = self
            .state
            .connection
            .get_rlm_child_snapshots()
            .await
            .unwrap_or_default();
        if abort.is_cancelled() {
            entry.producer.drain().await;
            return Ok(cancelled());
        }
        let outcome = if failure.is_some() { OUTCOME_ERROR } else { OUTCOME_RESULT };
        let mut terminal_status = status.clone();
        let observed_quiescence = quiescence_meta(&status, Some(live_children.as_slice()));
        // The roster is telemetry at the response cut, not proof of terminality:
        // a child can publish a terminal status before its result reaches the parent.
        // Every turn therefore finalizes through the strong settlement barrier.
        entry.producer.commit_response(prompt_turn_id);
        let mut boundary_meta = PrimeAgentSessionMeta::new();
        boundary_meta.terminal_quiescence_expected = Some(true);
        let mut boundary_update = Map::new();
        boundary_update.insert("sessionUpdate".to_string(), json!("session_info_update"));
        boundary_update.insert("_meta".to_string(), Value::Object(prime_agent_meta(boundary_meta)));
        *response_boundary_emitted = entry
            .producer
            .publish(boundary_update, prompt_turn_id, PHASE_RESPONSE_BOUNDARY, Some(outcome))
            .await;
        if !*response_boundary_emitted {
            return Err(AcpHandlerError::Thrown(
                "Failed to publish ACP response boundary".to_string(),
            ));
        }
        let mut completion_meta = PrimeAgentSessionMeta::new();
        completion_meta.autonomous = autonomous;
        completion_meta.quiescence = Some(observed_quiescence);
        let mut completion_update = Map::new();
        completion_update.insert("sessionUpdate".to_string(), json!("session_info_update"));
        completion_update.insert("_meta".to_string(), Value::Object(prime_agent_meta(completion_meta)));
        let completion_update_emitted = entry
            .producer
            .publish(completion_update, prompt_turn_id, PHASE_EVENT, None)
            .await;
        if !completion_update_emitted {
            return Err(AcpHandlerError::Thrown(
                "Failed to publish ACP completion update".to_string(),
            ));
        }
        entry.producer.drain().await;
        if !abort.is_cancelled() {
            entry.producer.begin_terminal_lifecycle(prompt_turn_id);
            let pending = Arc::new(AcpPendingTerminal::new(
                prompt_turn_id,
                prior_messages,
                outcome,
                abort.clone(),
            ));
            *entry.pending_terminal.lock().expect("pending terminal poisoned") = Some(pending.clone());
            finalize_pending_terminal(entry.clone(), pending.clone(), self.state.connection.clone());
            if let Some(task) = pending.task() {
                let _ = task.wait().await;
            }
            *terminal_settlement_cancelled = abort.is_cancelled();
            if let Some(failure) = pending.failure() {
                return Err(AcpHandlerError::Thrown(format!(
                    "ACP lifecycle reconciliation failed: {failure}"
                )));
            }
            if let Some(turn_failure) = pending.turn_failure() {
                return Err(AcpHandlerError::Thrown(format!(
                    "Optimus turn failed: {turn_failure}"
                )));
            }
            terminal_status = pending.status().unwrap_or(status);
        }
        if let Some(failure) = failure {
            return Err(AcpHandlerError::Thrown(format!(
                "Optimus turn failed: {failure}"
            )));
        }
        Ok(json!({
            "stopReason": acp_stop_reason(*terminal_settlement_cancelled, Some(&terminal_status)),
        }))
    }
}

/// `finalizePendingTerminal(entry, pending)`.
///
/// The connection handle is threaded through as private plumbing: the TypeScript
/// settles through the closure's `connection`, which the entry object does not hold.
fn finalize_pending_terminal(
    entry: Arc<AcpSessionEntry>,
    pending: Arc<AcpPendingTerminal>,
    state_connection: Arc<dyn AgentConnection>,
) {
    let entry_for_task = entry.clone();
    let pending_for_task = pending.clone();
    let task = AcpPromise::spawn(async move {
        loop {
            let status = match state_connection
                .wait_for_headless_completion(Some(AgentConnectionHeadlessCompletionOptions {
                    wait_for_rlm_quiescence: Some(true),
                }))
                .await
            {
                Ok(status) => status,
                Err(error) => {
                    if !pending_for_task.abort.is_cancelled()
                        && pending_is_current(&entry_for_task, &pending_for_task)
                    {
                        *pending_for_task.failure.lock().expect("pending failure poisoned") = Some(error);
                    }
                    return;
                }
            };
            if pending_for_task.abort.is_cancelled() || !pending_is_current(&entry_for_task, &pending_for_task) {
                return;
            }
            let final_failure = turn_failure(&state_connection, &pending_for_task.boundary).await;
            if pending_for_task.abort.is_cancelled() || !pending_is_current(&entry_for_task, &pending_for_task) {
                return;
            }
            let live_children = state_connection.get_rlm_child_snapshots().await.unwrap_or_default();
            if pending_for_task.abort.is_cancelled() || !pending_is_current(&entry_for_task, &pending_for_task) {
                return;
            }
            let terminal_quiescence = quiescence_meta(&status, Some(live_children.as_slice()));
            if terminal_quiescence.outstanding_subagents != 0 {
                continue;
            }
            *pending_for_task.status.lock().expect("pending status poisoned") = Some(status.clone());
            *pending_for_task.turn_failure.lock().expect("turn failure poisoned") = final_failure.clone();

            entry_for_task.producer.seal_terminal(pending_for_task.prompt_turn_id);
            let autonomous = autonomous_meta(Some(&status));
            let mut meta = PrimeAgentSessionMeta::new();
            meta.autonomous = autonomous;
            meta.quiescence = Some(terminal_quiescence);
            let mut update = Map::new();
            update.insert("sessionUpdate".to_string(), json!("session_info_update"));
            update.insert("_meta".to_string(), Value::Object(prime_agent_meta(meta)));
            let publication = entry_for_task
                .producer
                .publish(
                    update,
                    pending_for_task.prompt_turn_id,
                    PHASE_TERMINAL_QUIESCENCE,
                    Some(if final_failure.is_some() {
                        OUTCOME_ERROR
                    } else {
                        pending_for_task.outcome
                    }),
                )
                .await;
            // Keep terminal ownership until this settlement task has fully drained.
            // A follow-up prompt awaits that task; clearing ownership at publication
            // admission would let it overlap the first prompt handler.
            if !publication {
                return;
            }
            entry_for_task.producer.drain().await;
            return;
        }
    });
    *pending.task.lock().expect("pending task poisoned") = Some(task.clone());
    // `.finally(() => { ... })`.
    let entry_for_finally = entry.clone();
    let pending_for_finally = pending.clone();
    tokio::spawn(async move {
        let _ = task.wait().await;
        entry_for_finally
            .producer
            .finish_terminal_lifecycle(pending_for_finally.prompt_turn_id);
        {
            let mut slot = entry_for_finally
                .pending_terminal
                .lock()
                .expect("pending terminal poisoned");
            let is_current = slot
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &pending_for_finally))
                .unwrap_or(false);
            if is_current {
                *slot = None;
            }
        }
        {
            let mut abort_slot = entry_for_finally.abort.lock().expect("abort poisoned");
            let is_current = abort_slot
                .as_ref()
                .map(|current| Arc::ptr_eq(current, &pending_for_finally.abort))
                .unwrap_or(false);
            if is_current {
                *abort_slot = None;
            }
        }
    });
}

/// `session !== entry || entry.pendingTerminal !== pending` guard.
fn pending_is_current(entry: &Arc<AcpSessionEntry>, pending: &Arc<AcpPendingTerminal>) -> bool {
    entry
        .pending_terminal()
        .map(|current| Arc::ptr_eq(&current, pending))
        .unwrap_or(false)
}


impl AcpAgentApp {
    /// `.onRequest("session/close", ...)`.
    async fn handle_session_close(&self, ctx: &AcpRequestContext) -> Result<Value, AcpHandlerError> {
        let requested_session_id = ctx.params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
        let Some(closing) = self.state.session() else {
            return Err(AcpHandlerError::Thrown(format!(
                "Unknown ACP session: {requested_session_id}"
            )));
        };
        if closing.id != requested_session_id {
            return Err(AcpHandlerError::Thrown(format!(
                "Unknown ACP session: {requested_session_id}"
            )));
        }
        // Stop real work, not just local bookkeeping: aborting only the local
        // controller leaves the agent running with nobody listening, so closing
        // must abort the connection the same way session/cancel does.
        if self.state.session_close_in_flight() {
            return Err(AcpHandlerError::Thrown(format!(
                "ACP session is already closing: {requested_session_id}"
            )));
        }
        self.state.session_close_in_flight.store(true, Ordering::SeqCst);
        let (close_task, finish_close) = AcpPromise::new();
        *self.state.session_close_task.lock().expect("close task poisoned") = Some(close_task);
        let result = self.session_close_inner(&closing).await;
        {
            let mut slot = self.state.session_close_task.lock().expect("close task poisoned");
            *slot = None;
        }
        finish_close.resolve();
        if self
            .state
            .session()
            .map(|current| Arc::ptr_eq(&current, &closing))
            .unwrap_or(false)
        {
            closing.cancelling.store(false, Ordering::SeqCst);
        }
        self.state.session_close_in_flight.store(false, Ordering::SeqCst);
        result
    }

    async fn session_close_inner(&self, closing: &Arc<AcpSessionEntry>) -> Result<Value, AcpHandlerError> {
        let cancel_task = closing.cancel_task.lock().expect("cancel task poisoned").clone();
        if let Some(cancel_task) = cancel_task {
            let _ = cancel_task.wait().await;
        }
        closing.cancelling.store(true, Ordering::SeqCst);
        if let Some(abort) = closing.abort() {
            abort.cancel();
        }
        let pending = closing.pending_terminal();
        let prompt_task = closing.prompt_task();
        let input_pause = match self.state.acquire_stop_input_pause(closing).await {
            Ok(input_pause) => input_pause,
            Err(error) => {
                closing.cancelling.store(false, Ordering::SeqCst);
                *closing.stop_failure.lock().expect("stop failure poisoned") = Some(error.clone());
                return Err(AcpHandlerError::Thrown(error));
            }
        };
        let input_pause_key = closing.input_pause_key.lock().expect("input pause key poisoned").clone();
        let Some(input_pause_key) = input_pause_key else {
            let error = "Missing ACP close input-pause key".to_string();
            *closing.stop_failure.lock().expect("stop failure poisoned") = Some(error.clone());
            return Err(AcpHandlerError::Thrown(error));
        };
        let outcome: Result<(), String> = async {
            self.state.stop_session_work(pending, prompt_task).await?;
            if let Some(unsubscribe) = closing.unsubscribe.lock().expect("unsubscribe poisoned").take() {
                unsubscribe();
            }
            // Keep the backing session fenced until a replacement ACP session is admitted.
            closing.producer.close().await;
            // Host credentials are already gone before kernel release runs. Do not
            // retain the ACP session slot if best-effort transport reaping fails.
            let _ = self.state.clear_acp_mcp_servers(None).await;
            {
                let mut closed = self.state.closed_input_pause.lock().expect("closed pause poisoned");
                *closed = Some(input_pause.clone());
            }
            *self
                .state
                .closed_input_pause_key
                .lock()
                .expect("closed pause key poisoned") = Some(input_pause_key.clone());
            {
                let mut entry_pause = closing.input_pause.lock().expect("input pause poisoned");
                if entry_pause
                    .as_ref()
                    .map(|pause| Arc::ptr_eq(pause, &input_pause))
                    .unwrap_or(false)
                {
                    *entry_pause = None;
                    *closing.input_pause_key.lock().expect("input pause key poisoned") = None;
                }
            }
            *closing.stop_failure.lock().expect("stop failure poisoned") = None;
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            *closing.stop_failure.lock().expect("stop failure poisoned") = Some(error.clone());
            return Err(AcpHandlerError::Thrown(error));
        }
        if self
            .state
            .session()
            .map(|current| Arc::ptr_eq(&current, closing))
            .unwrap_or(false)
        {
            *self.state.session.lock().expect("session poisoned") = None;
        }
        Ok(json!({}))
    }

    /// `.onNotification("session/cancel", ...)`.
    async fn handle_session_cancel(&self, ctx: &AcpRequestContext) -> Result<(), AcpHandlerError> {
        let requested_session_id = ctx.params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
        while self.state.session_close_in_flight() {
            let task = self.state.session_close_task.lock().expect("close task poisoned").clone();
            if let Some(task) = task {
                let _ = task.wait().await;
            }
        }
        // Only cancel the addressed session: aborting unconditionally would kill
        // whichever turn happens to be running, and leave the real turn's
        // AbortController unmarked so it reports a wrong stop reason.
        let Some(cancelling) = self.state.session() else {
            return Ok(());
        };
        if cancelling.id != requested_session_id {
            return Ok(());
        }
        if cancelling.cancelling.load(Ordering::SeqCst) {
            let task = cancelling.cancel_task.lock().expect("cancel task poisoned").clone();
            if let Some(task) = task {
                let _ = task.wait().await;
            }
            return Ok(());
        }
        let abort = cancelling.abort();
        if abort.is_none()
            && cancelling.stop_failure().is_none()
            && cancelling.input_pause_release().is_none()
        {
            return Ok(());
        }
        let pending = abort.as_ref().and_then(|abort| {
            cancelling
                .pending_terminal()
                .filter(|pending| Arc::ptr_eq(&pending.abort, abort))
        });
        let prompt_task = cancelling.prompt_task();
        cancelling.cancelling.store(true, Ordering::SeqCst);

        let cancel_state = self.state.clone();
        let cancel_entry = cancelling.clone();
        let abort_for_task = abort.clone();
        let (cancel_task, cancel_resolver) = AcpPromise::new();
        *cancelling.cancel_task.lock().expect("cancel task poisoned") = Some(cancel_task.clone());
        let cancel_task_for_task = cancel_task.clone();
        tokio::spawn(async move {
            if let Some(abort) = &abort_for_task {
                abort.cancel();
            }
            let outcome: Result<(), String> = async {
                let input_pause = cancel_state.acquire_stop_input_pause(&cancel_entry).await?;
                cancel_state.stop_session_work(pending.clone(), prompt_task).await?;
                input_pause.release().await?;
                {
                    let mut entry_pause = cancel_entry.input_pause.lock().expect("input pause poisoned");
                    if entry_pause
                        .as_ref()
                        .map(|pause| Arc::ptr_eq(pause, &input_pause))
                        .unwrap_or(false)
                    {
                        *entry_pause = None;
                        *cancel_entry.input_pause_key.lock().expect("input pause key poisoned") = None;
                    }
                }
                {
                    let mut closed = cancel_state.closed_input_pause.lock().expect("closed pause poisoned");
                    if closed
                        .as_ref()
                        .map(|pause| Arc::ptr_eq(pause, &input_pause))
                        .unwrap_or(false)
                    {
                        *closed = None;
                        *cancel_state
                            .closed_input_pause_key
                            .lock()
                            .expect("closed pause key poisoned") = None;
                    }
                }
                *cancel_entry.input_pause_release.lock().expect("input pause release poisoned") = None;
                *cancel_entry.stop_failure.lock().expect("stop failure poisoned") = None;
                if let Some(pending) = &pending {
                    let mut slot = cancel_entry.pending_terminal.lock().expect("pending terminal poisoned");
                    if slot
                        .as_ref()
                        .map(|current| Arc::ptr_eq(current, pending))
                        .unwrap_or(false)
                    {
                        *slot = None;
                    }
                }
                if let Some(abort) = &abort_for_task {
                    let mut slot = cancel_entry.abort.lock().expect("abort poisoned");
                    if slot
                        .as_ref()
                        .map(|current| Arc::ptr_eq(current, abort))
                        .unwrap_or(false)
                    {
                        *slot = None;
                    }
                }
                Ok(())
            }
            .await;
            if let Err(error) = outcome {
                *cancel_entry.stop_failure.lock().expect("stop failure poisoned") = Some(error);
            }
            let mut slot = cancel_entry.cancel_task.lock().expect("cancel task poisoned");
            if slot
                .as_ref()
                .map(|current| current.ptr_eq(&cancel_task_for_task))
                .unwrap_or(false)
            {
                *slot = None;
            }
            cancel_entry.cancelling.store(false, Ordering::SeqCst);
            cancel_resolver.resolve();
        });
        // `await cancelTask` in the handler: the notification is only handled once
        // the cancellation has fully settled.
        let _ = cancel_task.wait().await;
        Ok(())
    }
}

impl AcpAgentApp {
    /// `handle.connect(stream)`; the returned promise is `handle.closed`.
    ///
    /// blocked_on: `AcpConnection.closed` and the SDK router are driven here
    /// directly: requests are dispatched by method name, notifications likewise,
    /// and the connection is closed when the reader channel ends.
    async fn connect(self: &Arc<Self>, mut reader: mpsc::UnboundedReceiver<AcpIncomingMessage>) -> AcpPromise {
        let (closed, closed_resolver) = AcpPromise::new();
        let app = self.clone();
        tokio::spawn(async move {
            while let Some(message) = reader.recv().await {
                let Some(method) = message.method.clone() else {
                    continue;
                };
                if message.is_request {
                    let app = app.clone();
                    tokio::spawn(async move {
                        let ctx = AcpRequestContext {
                            params: message.params.clone(),
                            request_id: message.id.clone(),
                        };
                        match method.as_str() {
                            acp_methods::agent::INITIALIZE => {
                                let result = app.handle_initialize().await;
                                app.send_response(&ctx.request_id, result).await;
                            }
                            m if m == acp_methods::agent::session::NEW => {
                                let result = app.handle_session_new(&ctx).await;
                                app.send_outcome(&ctx.request_id, result).await;
                            }
                            m if m == acp_methods::agent::session::PROMPT => {
                                let result = app.handle_session_prompt(&ctx).await;
                                app.send_outcome(&ctx.request_id, result).await;
                            }
                            m if m == acp_methods::agent::session::CLOSE => {
                                let result = app.handle_session_close(&ctx).await;
                                app.send_outcome(&ctx.request_id, result).await;
                            }
                            other => {
                                let error = AcpRequestErrorInfo::method_not_found(other);
                                let message = json!({
                                    "jsonrpc": "2.0",
                                    "id": ctx.request_id,
                                    "error": error.to_error_response(),
                                });
                                let _ = app.writable.write(message).await;
                            }
                        }
                    });
                    continue;
                }
                if method == acp_methods::agent::session::CANCEL {
                    let app = app.clone();
                    tokio::spawn(async move {
                        let ctx = AcpRequestContext {
                            params: message.params.clone(),
                            request_id: message.id.clone(),
                        };
                        let _ = app.handle_session_cancel(&ctx).await;
                    });
                    continue;
                }
                // Unknown notifications are ignored, exactly as an unhandled
                // `onNotification` registration would leave them.
            }
            closed_resolver.resolve();
        });
        closed
    }

    /// `ctx.responder.respond(response)`.
    async fn send_response(&self, request_id: &Value, result: Value) {
        let message = json!({ "jsonrpc": "2.0", "id": request_id, "result": result });
        let _ = self.writable.write(message).await;
    }

    /// `ctx.responder.respondWithError` / `respond` through `errorToResult`.
    async fn send_outcome(&self, request_id: &Value, result: Result<Value, AcpHandlerError>) {
        match result {
            Ok(value) => self.send_response(request_id, value).await,
            Err(error) => {
                let error = error_to_result(error);
                let message = json!({ "jsonrpc": "2.0", "id": request_id, "error": error.to_error_response() });
                let _ = self.writable.write(message).await;
            }
        }
    }
}
