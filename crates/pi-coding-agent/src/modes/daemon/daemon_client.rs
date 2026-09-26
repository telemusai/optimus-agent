//! Port of packages/coding-agent/src/modes/daemon/daemon-client.ts
//!
//! Everything `daemon-client.ts` imports from `./daemon-protocol.js` comes from
//! `super::daemon_protocol`. What lives here is only what the TypeScript module
//! declares itself: `DaemonClient`, its errors, the reconnect types,
//! `DaemonHello`, `DaemonCommandBody`, the listener aliases and the private
//! `isDaemon*` guards the reader uses.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Map, Value};
use tokio::sync::{oneshot, Mutex, Notify};
use tokio::time::Instant;
use tokio_util::codec::{Framed, LinesCodec};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::utils::daemon_socket_path::normalize_socket_path;

use super::daemon_protocol::{
    daemon_command_compatibility, daemon_jev_mode_compatibility, is_daemon_mutating_command,
    meets_daemon_command_compatibility, DaemonCommandCompatibility, DaemonCompatibilityHello, DaemonProtocolInfo,
    DaemonResponse, DaemonServerCapability, DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION, DAEMON_PROTOCOL_NAME, DAEMON_PROTOCOL_VERSION,
};

/// `getDaemonLogPath` from config.ts (logs live under the agent dir).
// Private config.ts plumbing (config.ts belongs to another slice).
fn get_daemon_log_path(socket_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let normalized = normalize_socket_path(socket_path, None);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let digest = hasher.finalize();
    let hash: String = digest.iter().take(4).map(|byte| format!("{byte:02x}")).collect();
    let basename = Path::new(&normalized)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| normalized.clone());
    join_path(&get_logs_dir(), &format!("{basename}.{hash}.log"))
}

fn get_logs_dir() -> String {
    join_path(&get_agent_dir(), "logs")
}

fn get_agent_dir() -> String {
    if let Ok(env_dir) = std::env::var("PRIME_AGENT_CODING_AGENT_DIR") {
        if !env_dir.is_empty() {
            return expand_tilde_path(&env_dir);
        }
    }
    join_path(&home_dir(), ".prime/agent")
}

fn home_dir() -> String {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string())
}

fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        return join_path(&home_dir(), rest);
    }
    path.to_string()
}

fn join_path(base: &str, name: &str) -> String {
    let base = base.trim_end_matches(['/', '\\']);
    format!("{base}/{name}")
}

fn daemon_endpoint_details(socket_path: &str) -> String {
    format!(
        "Socket: {socket_path}. Daemon log: {}.",
        get_daemon_log_path(socket_path)
    )
}

/// `type DaemonHello = Extract<DaemonOutbound, { type: "daemon_hello" }>` as this
/// client consumes it: the greeting fields plus the raw frame.
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonHello {
    pub protocol: DaemonProtocolInfo,
    pub schema_revision: Option<u32>,
    pub server_capabilities: Vec<String>,
    pub raw: Value,
}

impl DaemonHello {
    pub fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.get("type")?.as_str()? != "daemon_hello" {
            return None;
        }
        let protocol_value = object.get("protocol")?;
        if !protocol_value.is_object() {
            return None;
        }
        let protocol = serde_json::from_value::<DaemonProtocolInfo>(protocol_value.clone()).ok()?;
        let schema_revision = object
            .get("schemaRevision")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let server_capabilities = object
            .get("serverCapabilities")
            .and_then(Value::as_array)
            .map(|entries| entries.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        Some(Self { protocol, schema_revision, server_capabilities, raw: value.clone() })
    }

    /// `hello.serverCapabilities?.includes(capability) === true`.
    pub fn supports(&self, capability: &str) -> bool {
        self.server_capabilities.iter().any(|entry| entry == capability)
    }
}

/// `meetsDaemonCommandCompatibility(hello, compatibility)` reads the greeting; the
/// protocol module takes its own view of it, so map the frame onto that view.
/// An unknown capability name can never match `includes`, so dropping it is exact.
pub(crate) fn compatibility_hello(hello: &DaemonHello) -> DaemonCompatibilityHello {
    DaemonCompatibilityHello {
        protocol: hello.protocol.clone(),
        schema_revision: hello.schema_revision,
        server_capabilities: Some(
            hello
                .server_capabilities
                .iter()
                .filter_map(|capability| {
                    serde_json::from_value::<DaemonServerCapability>(Value::String(capability.clone())).ok()
                })
                .collect(),
        ),
    }
}

/// The wire name of a capability (`DaemonServerCapability` is snake_case on the wire).
pub(crate) fn capability_name(capability: &DaemonServerCapability) -> String {
    serde_json::to_value(capability)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn is_daemon_hello(value: &Value) -> bool {
    value.as_object().is_some_and(|candidate| {
        candidate.get("type").and_then(Value::as_str) == Some("daemon_hello")
            && candidate.get("protocol").is_some_and(Value::is_object)
    })
}

pub(crate) fn is_daemon_response(value: &Value) -> bool {
    value.as_object().is_some_and(|candidate| {
        candidate.get("type").and_then(Value::as_str) == Some("response")
            && candidate.get("command").and_then(Value::as_str).is_some()
            && candidate.get("success").and_then(Value::as_bool).is_some()
    })
}

fn is_daemon_saved_session_info(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    candidate.get("path").and_then(Value::as_str).is_some()
        && candidate.get("id").and_then(Value::as_str).is_some()
        && candidate.get("cwd").and_then(Value::as_str).is_some()
        && candidate.get("created").and_then(Value::as_str).is_some()
        && candidate.get("modified").and_then(Value::as_str).is_some()
        && candidate.get("messageCount").and_then(Value::as_f64).is_some()
        && candidate.get("firstMessage").and_then(Value::as_str).is_some()
        && candidate.get("allMessagesText").and_then(Value::as_str).is_some()
        && candidate.get("agentStatus").is_none_or(is_daemon_saved_session_agent_status)
}

fn is_daemon_saved_session_agent_status(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    candidate.get("summary").and_then(Value::as_str).is_some()
        && candidate.get("basedOnMessageCount").and_then(Value::as_f64).is_some()
        && candidate
            .get("taskState")
            .is_none_or(|task_state| matches!(task_state.as_str(), Some("needs_input") | Some("completed")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonRequestAbortError {
    pub command_type: String,
}

impl DaemonRequestAbortError {
    pub fn message(&self) -> String {
        format!("Daemon request \"{}\" was cancelled", self.command_type)
    }
}

impl std::fmt::Display for DaemonRequestAbortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for DaemonRequestAbortError {}

pub(crate) fn daemon_request_abort_error(command_type: &str) -> DaemonClientError {
    DaemonClientError::Aborted(DaemonRequestAbortError { command_type: command_type.to_string() })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonSocketClosedError {
    pub socket_path: String,
    pub daemon_closing_reason: Option<String>,
    pub cause: Option<String>,
}

impl DaemonSocketClosedError {
    pub fn new(socket_path: &str, daemon_closing_reason: Option<&str>, cause: Option<&str>) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            daemon_closing_reason: daemon_closing_reason.map(str::to_string),
            cause: cause.map(str::to_string),
        }
    }

    pub fn message(&self) -> String {
        let reason_details = self
            .daemon_closing_reason
            .as_ref()
            .map(|reason| format!(" Reason: {reason}."))
            .unwrap_or_default();
        let cause_details = self
            .cause
            .as_ref()
            .map(|cause| format!(" Cause: {cause}."))
            .unwrap_or_default();
        format!(
            "Connection to the Prime Agent daemon closed.{reason_details}{cause_details} {}",
            daemon_endpoint_details(&self.socket_path)
        )
    }
}

impl std::fmt::Display for DaemonSocketClosedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for DaemonSocketClosedError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonCapabilityUnavailableError {
    pub command: String,
    pub capability: Option<String>,
    pub after_reconnect: bool,
}

impl DaemonCapabilityUnavailableError {
    pub fn new(command: &str, capability: Option<&str>, after_reconnect: bool) -> Self {
        Self {
            command: command.to_string(),
            capability: capability.map(str::to_string),
            after_reconnect,
        }
    }

    pub fn message(&self) -> String {
        match &self.capability {
            Some(capability) => format!("The running Prime Agent daemon does not support {capability}."),
            None => format!("The running Prime Agent daemon does not support {}.", self.command),
        }
    }
}

impl std::fmt::Display for DaemonCapabilityUnavailableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for DaemonCapabilityUnavailableError {}

/// The client-side failures. `DaemonClientError::Message` is a plain `Error`, the
/// other arms are the named classes `daemon-client.ts` throws.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonClientError {
    Message(String),
    SocketClosed(DaemonSocketClosedError),
    CapabilityUnavailable(DaemonCapabilityUnavailableError),
    Aborted(DaemonRequestAbortError),
}

impl DaemonClientError {
    pub fn message(&self) -> String {
        match self {
            DaemonClientError::Message(message) => message.clone(),
            DaemonClientError::SocketClosed(error) => error.message(),
            DaemonClientError::CapabilityUnavailable(error) => error.message(),
            DaemonClientError::Aborted(error) => error.message(),
        }
    }

    pub fn is_abort(&self) -> bool {
        matches!(self, DaemonClientError::Aborted(_))
    }

    pub fn is_capability_unavailable(&self) -> bool {
        matches!(self, DaemonClientError::CapabilityUnavailable(_))
    }
}

impl std::fmt::Display for DaemonClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for DaemonClientError {}

pub type DaemonClientResult<T> = Result<T, DaemonClientError>;

pub fn get_daemon_socket_close_reason(error: &DaemonClientError) -> Option<String> {
    match error {
        DaemonClientError::SocketClosed(closed) => closed.daemon_closing_reason.clone(),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonClientReconnectStatus {
    Reconnecting { error: String },
    Connected,
    Failed { error: String },
}

#[derive(Clone)]
pub struct DaemonClientReconnectOptions {
    pub recover_daemon: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
    pub timeout_ms: Option<u64>,
    pub on_status: Option<Arc<dyn Fn(DaemonClientReconnectStatus) + Send + Sync>>,
}

pub type DaemonClientMessageListener = Arc<dyn Fn(&Value) + Send + Sync>;
pub type DaemonClientCloseListener = Arc<dyn Fn(&DaemonClientError) + Send + Sync>;
pub type DaemonClientProgressListener = Arc<dyn Fn(&Value) + Send + Sync>;

/// `options?: { onProgress?, signal?, recoverable? }`.
#[derive(Clone, Default)]
pub struct DaemonClientRequestOptions {
    pub on_progress: Option<DaemonClientProgressListener>,
    pub signal: Option<CancellationToken>,
    /// False opts out of reconnect parking: a close rejects so the caller's own retry loop stays live.
    pub recoverable: Option<bool>,
}

const DEFAULT_DAEMON_REQUEST_TIMEOUT_MS: u64 = 30_000;
// Windows worker startup can exceed 30 seconds under antivirus scanning.
const WINDOWS_DAEMON_CREATE_TIMEOUT_MS: u64 = 120_000;

/// `defaultDaemonRequestTimeout(command)`. The worker command bodies this client
/// also carries are strings, so the type is taken as a name.
fn default_daemon_request_timeout(command_type: &str) -> u64 {
    if command_type == "create" && crate::utils::pi_user_agent::process_platform() == "win32" {
        WINDOWS_DAEMON_CREATE_TIMEOUT_MS
    } else {
        DEFAULT_DAEMON_REQUEST_TIMEOUT_MS
    }
}

const DEFAULT_RECONNECT_TIMEOUT_MS: u64 = 60_000;
const RECONNECT_CONNECT_TIMEOUT_MS: u64 = 1000;
const RECONNECT_HELLO_TIMEOUT_MS: u64 = 3000;
const MAX_RECONNECT_DELAY_MS: u64 = 2000;

/// LF-only JSONL framing: payload strings may contain U+2028/U+2029, so records
/// must split on `\n` only (mirrors `attachJsonlLineReader`).
pub const DAEMON_MAX_LINE_LENGTH: usize = 256 * 1024 * 1024;

/// `serializeJsonLine` (modes/rpc/jsonl.ts, ported in this crate).
pub(crate) fn serialize_json_line(value: &Value) -> String {
    crate::modes::rpc::jsonl::serialize_json_line(value)
}

/// `type DaemonWireCommandBody = DaemonCommandBody | DaemonWorkerCommandBody`:
/// the body is a JSON object plus the `type` discriminant, like the TypeScript
/// `{ ...command, id }` spread.
pub type DaemonCommandBody = Map<String, Value>;

fn command_body_type(body: &DaemonCommandBody) -> &str {
    body.get("type").and_then(Value::as_str).unwrap_or_default()
}

fn full_command_value(body: &DaemonCommandBody, id: &str) -> Value {
    let mut object = body.clone();
    object.insert("id".to_string(), Value::String(id.to_string()));
    Value::Object(object)
}

// JSON bodies retain missing/null fields and extension keys just like the TypeScript spread.
fn command_envelope_value(command: &Value, id: &str, client_id: Option<&str>, protocol_version: u32) -> Value {
    let mut envelope = Map::from_iter([
        ("type".to_string(), Value::String("command".to_string())),
        ("id".to_string(), Value::String(id.to_string())),
        ("protocol".to_string(), serde_json::json!({ "name": DAEMON_PROTOCOL_NAME, "version": protocol_version })),
    ]);
    if let Some(client_id) = client_id.filter(|client_id| !client_id.is_empty()) {
        envelope.insert("clientId".to_string(), Value::String(client_id.to_string()));
    }
    envelope.insert("command".to_string(), command.clone());
    Value::Object(envelope)
}

/// `getDaemonCommandCompatibilities(command)` for a JSON command body: the extra
/// requirement the body's fields trigger, then the table entry for its type.
pub(crate) fn command_compatibilities(body: &DaemonCommandBody) -> Vec<DaemonCommandCompatibility> {
    let command_type = command_body_type(body);
    let mut requirements: Vec<DaemonCommandCompatibility> = Vec::new();
    if matches!(command_type, "prompt" | "prompt_and_wait" | "steer" | "follow_up")
        && body.get("message").and_then(Value::as_str)
            .and_then(crate::core::slash_commands::parse_session_slash_command)
            .is_some_and(|command| command.name == "mode")
    {
        requirements.push(DaemonCommandCompatibility::gated(34, DaemonServerCapability::ExecutionMode));
        if body.get("message").and_then(Value::as_str)
            .and_then(crate::core::slash_commands::parse_session_slash_command)
            .is_some_and(|c| matches!(c.args.trim(), "node" | "toggle")) {
            requirements.push(DaemonCommandCompatibility::gated(35, DaemonServerCapability::NodeExecutionMode));
        }
    }
    let has_field = |key: &str| body.get(key).is_some_and(|value| !value.is_null());
    if (command_type == "attach" || command_type == "reattach") && has_field("recoveryConfig") {
        requirements.push(DaemonCommandCompatibility::gated(17, DaemonServerCapability::OwnedSessionRecoveryContext));
    }
    let carries_telemetry_policy = ((command_type == "attach" || command_type == "reattach")
        && has_field("telemetryDisabled"))
        || (command_type == "create"
            && body
                .get("config")
                .and_then(Value::as_object)
                .is_some_and(|config| config.contains_key("telemetryDisabled")));
    if carries_telemetry_policy {
        requirements.push(DaemonCommandCompatibility::revision(14));
    }
    if (command_type == "prompt" || command_type == "prompt_and_wait") && has_field("admissionId") {
        requirements.push(DaemonCommandCompatibility::gated(
            8,
            DaemonServerCapability::PromptAdmissionCancellation,
        ));
    }
    if command_type == "wait_for_headless_completion" && body.get("waitForRlmQuiescence") == Some(&Value::Bool(true)) {
        requirements.push(DaemonCommandCompatibility::gated(18, DaemonServerCapability::RlmQuiescenceBarrier));
    }
    if command_type == "cancel_prompt_admission" && body.get("cancelOwned") == Some(&Value::Bool(true)) {
        requirements.push(DaemonCommandCompatibility::gated(20, DaemonServerCapability::OwnedPromptCancellation));
    }
    if command_type == "jev_set_session_mode" {
        if let Some(requirement) = body.get("mode").and_then(Value::as_str).and_then(daemon_jev_mode_compatibility) {
            requirements.push(requirement);
        }
    }
    requirements.push(daemon_command_compatibility(command_type));
    requirements
}

/// `message as DaemonResponse` for a frame `isDaemonResponse` already accepted.
fn daemon_response_from_value(value: &Value) -> DaemonResponse {
    let candidate = value.as_object().expect("is_daemon_response checked the object");
    DaemonResponse {
        id: candidate.get("id").and_then(Value::as_str).map(str::to_string),
        type_: "response".to_string(),
        command: candidate
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        success: candidate.get("success").and_then(Value::as_bool).unwrap_or(false),
        data: candidate.get("data").cloned(),
        error: candidate.get("error").and_then(Value::as_str).map(str::to_string),
        error_info: candidate
            .get("errorInfo")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
    }
}

struct PendingDaemonRequest {
    command_type: String,
    timeout_ms: u64,
    on_progress: Option<DaemonClientProgressListener>,
    wire_data: String,
    awaiting_reconnect: bool,
    acknowledge_result: bool,
    recoverable: bool,
    compatibilities: Vec<DaemonCommandCompatibility>,
    deadline: Instant,
    result: Arc<StdMutex<Option<DaemonClientResult<DaemonResponse>>>>,
    wake: Arc<Notify>,
}

impl PendingDaemonRequest {
    fn settle(&self, result: DaemonClientResult<DaemonResponse>) {
        let mut slot = self.result.lock().expect("pending request slot poisoned");
        if slot.is_none() {
            *slot = Some(result);
        }
        // notify_one stores a permit when the waiter has not registered yet, so
        // a settle that races the select still wakes it.
        self.wake.notify_one();
    }
}

struct HelloWaiter {
    sender: oneshot::Sender<DaemonClientResult<DaemonHello>>,
}

#[cfg(unix)]
pub type DaemonSocketStream = tokio::net::UnixStream;
#[cfg(windows)]
pub type DaemonSocketStream = tokio::net::windows::named_pipe::NamedPipeClient;

type DaemonSink = SplitSink<Framed<DaemonSocketStream, LinesCodec>, String>;

struct ClientSocket {
    writer: Mutex<Option<DaemonSink>>,
    reader_task: StdMutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ClientSocket {
    async fn write_line(&self, line: String) -> std::io::Result<()> {
        let mut writer = self.writer.lock().await;
        match writer.as_mut() {
            Some(writer) => writer.send(line).await.map_err(std::io::Error::other),
            None => Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "socket closed")),
        }
    }

    fn destroy(&self) {
        let task = self.reader_task.lock().expect("reader task slot poisoned").take();
        if let Some(task) = task {
            task.abort();
        }
    }
}

/// The reader half of a connected socket: dispatch each line, then report the
/// close once the stream ends (`socket.on("close", ...)` in the TypeScript).
///
/// The client is held weakly, as before, so a live reader never keeps it alive.
async fn read_frames(
    weak: std::sync::Weak<DaemonClient>,
    mut reader: futures_util::stream::SplitStream<Framed<DaemonSocketStream, LinesCodec>>,
    socket: Arc<ClientSocket>,
) {
    while let Some(line) = reader.next().await {
        match line {
            Ok(line) => match weak.upgrade() {
                Some(client) => client.handle_line(&line).await,
                None => return,
            },
            Err(_) => break,
        }
    }
    let Some(client) = weak.upgrade() else {
        return;
    };
    notify_closed_boxed(client, socket).await;
}

/// `notifyClosed` can await `autoReconnect` -> `connect`, which spawns this same
/// reader future again. Boxing that one recursive await keeps the reader `Send`:
/// without the box, the reader type would have to contain itself.
fn notify_closed_boxed(
    client: Arc<DaemonClient>,
    socket: Arc<ClientSocket>,
) -> futures::future::BoxFuture<'static, ()> {
    Box::pin(async move {
        let reason = client.state.lock().await.daemon_closing_reason.clone();
        client
            .notify_closed(
                &socket,
                Some(DaemonSocketClosedError::new(
                    &client.socket_path,
                    reason.as_deref(),
                    None,
                )),
            )
            .await;
    })
}

pub(crate) async fn connect_daemon_socket(socket_path: &str) -> std::io::Result<DaemonSocketStream> {
    #[cfg(unix)]
    {
        tokio::net::UnixStream::connect(socket_path).await
    }
    #[cfg(windows)]
    {
        loop {
            match tokio::net::windows::named_pipe::ClientOptions::new().open(socket_path) {
                // The listener needs a scheduling turn to create its next instance.
                // connect() owns the deadline; cancellation drops this wait safely.
                Err(error) if error.raw_os_error() == Some(231) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                result => return result,
            }
        }
    }
}

#[derive(Default)]
struct ClientState {
    socket: Option<Arc<ClientSocket>>,
    pending: HashMap<String, PendingDaemonRequest>,
    hello_waiters: Vec<HelloWaiter>,
    request_id: u64,
    hello_message: Option<DaemonHello>,
    daemon_closing_reason: Option<String>,
    request_recovery_enabled: bool,
    reconnect_options: Option<DaemonClientReconnectOptions>,
}

/// The daemon transport a `DaemonAgentConnection` talks to.
///
/// The async members are boxed (`*_boxed`) because `DaemonClient`'s own
/// connect/request take `self: &Arc<Self>`; the boxed form keeps the trait
/// object-safe for `Arc<dyn DaemonTransportClient>`.
pub trait DaemonTransportClient: Send + Sync {
    fn hello(&self) -> Option<DaemonHello>;
    fn is_connected(&self) -> bool;
    fn supports_server_capability(&self, capability: &str) -> bool;
    fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync>;
    fn on_close(&self, listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync>;
    fn enable_request_recovery(&self);
    fn request_boxed(
        &self,
        command: DaemonCommandBody,
        timeout_ms: Option<u64>,
        options: DaemonClientRequestOptions,
    ) -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonResponse>>;
    fn wait_for_hello_boxed(
        &self,
        timeout_ms: u64,
    ) -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonHello>> {
        let _ = timeout_ms;
        Box::pin(async move {
            Err(DaemonClientError::Message(
                "Daemon transport does not implement waitForHello".to_string(),
            ))
        })
    }
    fn connect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        let _ = timeout_ms;
        Box::pin(async move {
            Err(DaemonClientError::Message(
                "Daemon transport does not implement connect".to_string(),
            ))
        })
    }
    fn reconnect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        let _ = timeout_ms;
        Box::pin(async move {
            Err(DaemonClientError::Message(
                "Daemon transport does not implement reconnect".to_string(),
            ))
        })
    }
    fn disconnect_for_reconnect_boxed(&self, reason: String) -> futures::future::BoxFuture<'static, ()> {
        let _ = reason;
        Box::pin(async move {})
    }
    fn reset_transport_for_reconnect_boxed(&self) -> futures::future::BoxFuture<'static, ()> {
        Box::pin(async move {})
    }
    /// True for `DaemonRoutedClient`, which already owns its direct link.
    fn is_routed_client(&self) -> bool {
        false
    }
    fn close(&self);
}

/// The daemon JSONL client: one socket, request/response correlation by id.
pub struct DaemonClient {
    socket_path: String,
    state: Mutex<ClientState>,
    listeners: Arc<StdMutex<HashMap<u64, DaemonClientMessageListener>>>,
    close_listeners: Arc<StdMutex<HashMap<u64, DaemonClientCloseListener>>>,
    next_listener_id: AtomicU64,
    protocol_client_id: String,
    closed: AtomicBool,
    quick_hello: StdMutex<Option<DaemonHello>>,
    quick_connected: AtomicBool,
    reconnect_busy: Mutex<()>,
    /// Set by `DaemonClient::create`, so the boxed trait methods can reach the
    /// `self: &Arc<Self>` implementations.
    self_ref: StdMutex<Option<std::sync::Weak<DaemonClient>>>,
}

impl DaemonClient {
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            state: Mutex::new(ClientState::default()),
            listeners: Arc::new(StdMutex::new(HashMap::new())),
            close_listeners: Arc::new(StdMutex::new(HashMap::new())),
            next_listener_id: AtomicU64::new(0),
            protocol_client_id: format!("daemon-client:{}", Uuid::new_v4()),
            closed: AtomicBool::new(false),
            quick_hello: StdMutex::new(None),
            quick_connected: AtomicBool::new(false),
            reconnect_busy: Mutex::new(()),
            self_ref: StdMutex::new(None),
        }
    }

    /// Plumbing helper: build the client with its own weak self reference so the
    /// `DaemonTransportClient` boxed methods can run the Arc-taking operations.
    pub fn create(socket_path: &str) -> Arc<Self> {
        let client = Arc::new(Self::new(socket_path));
        *client.self_ref.lock().expect("self reference slot poisoned") = Some(Arc::downgrade(&client));
        client
    }

    fn self_arc(&self) -> DaemonClientResult<Arc<Self>> {
        self.self_ref
            .lock()
            .expect("self reference slot poisoned")
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                DaemonClientError::Message(
                    "Prime Agent daemon client requires an Arc handle for this operation".to_string(),
                )
            })
    }

    pub fn socket_path(&self) -> &str {
        &self.socket_path
    }

    /// Synchronous getter mirroring `get hello()`.
    pub fn hello(&self) -> Option<DaemonHello> {
        self.quick_hello.lock().expect("hello slot poisoned").clone()
    }

    /// Synchronous getter mirroring `get isConnected()`.
    pub fn is_connected(&self) -> bool {
        self.quick_connected.load(Ordering::SeqCst)
    }

    pub fn supports_server_capability(&self, capability: &str) -> bool {
        self.hello().is_some_and(|hello| hello.supports(capability))
    }

    /// Wait for the daemon_hello greeting sent on connect.
    pub async fn wait_for_hello(&self, timeout_ms: u64) -> DaemonClientResult<DaemonHello> {
        let (sender, receiver) = oneshot::channel::<DaemonClientResult<DaemonHello>>();
        {
            let mut state = self.state.lock().await;
            if let Some(hello) = &state.hello_message {
                return Ok(hello.clone());
            }
            if state.socket.is_none() {
                return Err(DaemonClientError::Message(format!(
                    "Cannot wait for the Prime Agent daemon handshake because the daemon is not connected. {}",
                    daemon_endpoint_details(&self.socket_path)
                )));
            }
            state.hello_waiters.push(HelloWaiter { sender });
        }
        let details = daemon_endpoint_details(&self.socket_path);
        match tokio::time::timeout(Duration::from_millis(timeout_ms), receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(DaemonClientError::Message(format!(
                "Cannot wait for the Prime Agent daemon handshake because the daemon is not connected. {details}"
            ))),
            Err(_) => {
                let mut state = self.state.lock().await;
                state.hello_waiters.retain(|waiter| !waiter.sender.is_closed());
                Err(DaemonClientError::Message(format!(
                    "Timed out after {timeout_ms}ms waiting for the Prime Agent daemon handshake. {details}"
                )))
            }
        }
    }

    pub async fn connect(self: &Arc<Self>, timeout_ms: u64) -> DaemonClientResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(DaemonClientError::Message(
                "Prime Agent daemon client is closed".to_string(),
            ));
        }
        {
            let mut state = self.state.lock().await;
            if state.socket.is_some() {
                return Err(DaemonClientError::Message(format!(
                    "Prime Agent daemon client is already connected. {}",
                    daemon_endpoint_details(&self.socket_path)
                )));
            }
            state.hello_message = None;
            state.daemon_closing_reason = None;
        }
        *self.quick_hello.lock().expect("hello slot poisoned") = None;
        self.quick_connected.store(false, Ordering::SeqCst);

        let stream = match tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            connect_daemon_socket(&self.socket_path),
        )
        .await
        {
            Ok(Ok(stream)) => stream,
            Ok(Err(error)) => {
                return Err(DaemonClientError::Message(format!(
                    "Failed to connect to the Prime Agent daemon: {error}. {}",
                    daemon_endpoint_details(&self.socket_path)
                )));
            }
            Err(_) => {
                return Err(DaemonClientError::Message(format!(
                    "Timed out after {timeout_ms}ms connecting to the Prime Agent daemon. {}",
                    daemon_endpoint_details(&self.socket_path)
                )));
            }
        };

        let framed = Framed::new(stream, LinesCodec::new_with_max_length(DAEMON_MAX_LINE_LENGTH));
        let (writer, reader) = framed.split();
        let socket = Arc::new(ClientSocket {
            writer: Mutex::new(Some(writer)),
            reader_task: StdMutex::new(None),
        });
        let weak = Arc::downgrade(self);
        let reader_socket = socket.clone();
        let reader_task = tokio::spawn(read_frames(weak, reader, reader_socket));
        *socket.reader_task.lock().expect("reader task slot poisoned") = Some(reader_task);
        self.state.lock().await.socket = Some(socket);
        self.quick_connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub async fn reconnect(self: &Arc<Self>, timeout_ms: u64) -> DaemonClientResult<()> {
        if self.is_connected() {
            return Ok(());
        }
        self.connect(timeout_ms).await
    }

    pub async fn disconnect_for_reconnect(self: &Arc<Self>, reason: &str) {
        let socket = { self.state.lock().await.socket.clone() };
        let Some(socket) = socket else {
            return;
        };
        self.state.lock().await.daemon_closing_reason = Some(reason.to_string());
        self.notify_closed(
            &socket,
            Some(DaemonSocketClosedError::new(&self.socket_path, Some(reason), None)),
        )
        .await;
        socket.destroy();
    }

    /// Discard a partially recovered transport so the next retry can reconnect cleanly.
    pub async fn reset_transport_for_reconnect(&self) {
        let socket = { self.state.lock().await.socket.take() };
        let Some(socket) = socket else {
            return;
        };
        self.quick_connected.store(false, Ordering::SeqCst);
        let preserve = self.state.lock().await.request_recovery_enabled;
        self.reject_all(
            DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
                &self.socket_path,
                None,
                Some("reconnect attempt did not complete"),
            )),
            preserve,
        )
        .await;
        socket.destroy();
    }

    pub fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners.lock().expect("listeners poisoned").insert(id, listener);
        let registry = Arc::clone(&self.listeners);
        Box::new(move || {
            registry.lock().expect("listeners poisoned").remove(&id);
        })
    }

    pub fn on_close(&self, listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.close_listeners
            .lock()
            .expect("close listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.close_listeners);
        Box::new(move || {
            registry.lock().expect("close listeners poisoned").remove(&id);
        })
    }

    /// Keep in-flight command promises alive and resend their stable envelopes after reconnect.
    pub async fn enable_request_recovery(&self) {
        self.state.lock().await.request_recovery_enabled = true;
    }

    /// Reconnect a global/raw daemon client after supervisor replacement.
    pub async fn enable_auto_reconnect(&self, options: DaemonClientReconnectOptions) {
        let mut state = self.state.lock().await;
        state.request_recovery_enabled = true;
        state.reconnect_options = Some(options);
    }
}

impl DaemonClient {
    pub async fn request(
        self: &Arc<Self>,
        command: DaemonCommandBody,
        timeout_ms: Option<u64>,
        options: DaemonClientRequestOptions,
    ) -> DaemonClientResult<DaemonResponse> {
        let timeout_ms = timeout_ms.unwrap_or_else(|| default_daemon_request_timeout(command_body_type(&command)));
        if options.signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(daemon_request_abort_error(command_body_type(&command)));
        }
        {
            let state = self.state.lock().await;
            if state.socket.is_none() {
                return Err(DaemonClientError::Message(format!(
                    "Cannot send daemon command \"{}\" because the Prime Agent daemon is not connected. {}",
                    command_body_type(&command),
                    daemon_endpoint_details(&self.socket_path)
                )));
            }
        }
        let hello = match self.hello() {
            Some(hello) => hello,
            None => self.wait_for_hello(3000).await?,
        };
        if options.signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(daemon_request_abort_error(command_body_type(&command)));
        }
        // A JSON body has no typed `DaemonCommand` to gate, so the compatibility
        // table entry for its discriminator is the requirement set, exactly as
        // `getDaemonCommandCompatibilities` derives it for a plain body.
        let compatibilities = command_compatibilities(&command);
        if let Some(missing) = compatibilities
            .iter()
            .find(|compatibility| !meets_daemon_command_compatibility(&compatibility_hello(&hello), compatibility))
        {
            return Err(DaemonClientError::CapabilityUnavailable(
                DaemonCapabilityUnavailableError::new(
                    command_body_type(&command),
                    missing.capability.as_ref().map(capability_name).as_deref(),
                    false,
                ),
            ));
        }
        let envelope_protocol_version = hello.protocol.version.min(DAEMON_PROTOCOL_VERSION);
        let public_envelope = (envelope_protocol_version >= DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION)
            .then_some(envelope_protocol_version);
        self.request_wire(command, timeout_ms, options, public_envelope, compatibilities)
            .await
    }

    pub async fn authenticate_worker(
        self: &Arc<Self>,
        token: &str,
        timeout_ms: u64,
    ) -> DaemonClientResult<()> {
        let command = Map::from_iter([
            ("type".to_string(), Value::String("worker_auth".to_string())),
            ("token".to_string(), Value::String(token.to_string())),
        ]);
        let response = self
            .request_wire(command, timeout_ms, Default::default(), None, Vec::new())
            .await?;
        if !response.success {
            return Err(DaemonClientError::Message(
                response.error.unwrap_or_else(|| "worker authentication failed".to_string()),
            ));
        }
        Ok(())
    }

    pub async fn request_worker(
        self: &Arc<Self>,
        command: DaemonCommandBody,
        timeout_ms: u64,
    ) -> DaemonClientResult<DaemonResponse> {
        self.request_wire(command, timeout_ms, Default::default(), None, Vec::new())
            .await
    }

    pub async fn request_wire(
        self: &Arc<Self>,
        command: DaemonCommandBody,
        timeout_ms: u64,
        options: DaemonClientRequestOptions,
        public_envelope_protocol_version: Option<u32>,
        compatibilities: Vec<DaemonCommandCompatibility>,
    ) -> DaemonClientResult<DaemonResponse> {
        let socket = {
            let state = self.state.lock().await;
            match state.socket.clone() {
                Some(socket) => socket,
                None => {
                    return Err(DaemonClientError::Message(format!(
                        "Cannot send daemon command \"{}\" because the Prime Agent daemon is not connected. {}",
                        command_body_type(&command),
                        daemon_endpoint_details(&self.socket_path)
                    )));
                }
            }
        };

        let id = {
            let mut state = self.state.lock().await;
            state.request_id += 1;
            format!("daemon_{}", state.request_id)
        };
        let full_command = full_command_value(&command, &id);
        let wire_value = match public_envelope_protocol_version {
            Some(protocol_version) => command_envelope_value(
                &full_command,
                &id,
                Some(&self.protocol_client_id),
                protocol_version,
            ),
            None => full_command,
        };
        let wire_data = serialize_json_line(&wire_value);
        let acknowledge_result =
            public_envelope_protocol_version.is_some() && is_daemon_mutating_command(command_body_type(&command));

        let result = Arc::new(StdMutex::new(None));
        let wake = Arc::new(Notify::new());
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        {
            let mut state = self.state.lock().await;
            state.pending.insert(
                id.clone(),
                PendingDaemonRequest {
                    command_type: command_body_type(&command).to_string(),
                    timeout_ms,
                    on_progress: options.on_progress.clone(),
                    wire_data: wire_data.clone(),
                    awaiting_reconnect: false,
                    acknowledge_result,
                    recoverable: options.recoverable.unwrap_or(true),
                    compatibilities,
                    deadline,
                    result: Arc::clone(&result),
                    wake: Arc::clone(&wake),
                },
            );
        }
        if let Err(error) = socket.write_line(wire_data).await {
            let pending = self.state.lock().await.pending.remove(&id);
            if let Some(pending) = pending {
                pending.settle(Err(DaemonClientError::Message(error.to_string())));
            }
            return Err(DaemonClientError::Message(error.to_string()));
        }

        let details = daemon_endpoint_details(&self.socket_path);
        let command_type = command_body_type(&command).to_string();
        let abort = options.signal.clone();
        tokio::select! {
            _ = wake.notified() => {}
            _ = tokio::time::sleep_until(deadline) => {
                let pending = self.state.lock().await.pending.remove(&id);
                if let Some(pending) = pending {
                    pending.settle(Err(DaemonClientError::Message(format!(
                        "Timed out after {}ms waiting for the Prime Agent daemon response to \"{}\". {details}",
                        pending.timeout_ms, pending.command_type
                    ))));
                }
            }
            _ = async {
                match &abort {
                    Some(signal) => signal.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                let pending = self.state.lock().await.pending.remove(&id);
                if let Some(pending) = pending {
                    pending.settle(Err(daemon_request_abort_error(&pending.command_type)));
                }
            }
        }
        let settled = result.lock().expect("pending request slot poisoned").take();
        settled.unwrap_or_else(|| {
            Err(DaemonClientError::Message(format!(
                "Prime Agent daemon client closed before the operation completed. {details} (command \"{command_type}\")"
            )))
        })
    }

    pub async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let socket = {
            let mut state = self.state.lock().await;
            state.reconnect_options = None;
            state.socket.take()
        };
        self.quick_connected.store(false, Ordering::SeqCst);
        self.reject_all(
            DaemonClientError::Message(format!(
                "Prime Agent daemon client closed before the operation completed. {}",
                daemon_endpoint_details(&self.socket_path)
            )),
            false,
        )
        .await;
        if let Some(socket) = socket {
            socket.destroy();
        }
    }

    async fn handle_line(self: &Arc<Self>, line: &str) {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            return;
        };

        if let Some(hello) = DaemonHello::from_value(&message) {
            // `this.helloMessage = message` is assigned BEFORE the waiters resolve
            // (`daemon-client.ts:487-490`). Publishing the synchronous hello first
            // keeps `supportsServerCapability` consistent for a waiter that resumes
            // as soon as it is notified; otherwise it can read `None` and a
            // `--print` run fails with "does not support client_owned_sessions".
            *self.quick_hello.lock().expect("hello slot poisoned") = Some(hello.clone());
            {
                let mut state = self.state.lock().await;
                state.hello_message = Some(hello.clone());
                let waiters = std::mem::take(&mut state.hello_waiters);
                for waiter in waiters {
                    let _ = waiter.sender.send(Ok(hello.clone()));
                }
            }
            if self.is_connected() {
                let replays = self.take_reconnect_replays().await;
                for (id, pending) in replays {
                    if let Some(missing) = pending
                        .compatibilities
                        .iter()
                        .find(|compatibility| {
                            !meets_daemon_command_compatibility(&compatibility_hello(&hello), compatibility)
                        })
                    {
                        self.state.lock().await.pending.remove(&id);
                        pending.settle(Err(DaemonClientError::CapabilityUnavailable(
                            DaemonCapabilityUnavailableError::new(
                                &pending.command_type,
                                missing.capability.as_ref().map(capability_name).as_deref(),
                                true,
                            ),
                        )));
                        continue;
                    }
                    let socket = { self.state.lock().await.socket.clone() };
                    if let Some(socket) = socket {
                        let _ = socket.write_line(pending.wire_data.clone()).await;
                    }
                }
            }
        }
        if is_daemon_closing(&message) {
            if let Some(reason) = message.get("reason").and_then(Value::as_str) {
                self.state.lock().await.daemon_closing_reason = Some(reason.to_string());
            }
        }


        if is_daemon_response(&message) {
            if let Some(id) = message.get("id").and_then(Value::as_str) {
                let pending = self.state.lock().await.pending.remove(id);
                if let Some(pending) = pending {
                    let acknowledge = pending.acknowledge_result;
                    pending.settle(Ok(daemon_response_from_value(&message)));
                    if acknowledge {
                        self.acknowledge_command_result(id).await;
                    }
                    return;
                }
            }
        }
        if is_daemon_request_progress(&message) {
            if let Some(id) = message.get("id").and_then(Value::as_str) {
                let on_progress = {
                    let state = self.state.lock().await;
                    state.pending.get(id).and_then(|pending| pending.on_progress.clone())
                };
                if let Some(on_progress) = on_progress {
                    on_progress(&message);
                    return;
                }
            }
        }

        let listeners: Vec<DaemonClientMessageListener> = {
            let registry = self.listeners.lock().expect("listeners poisoned");
            registry.values().cloned().collect()
        };
        for listener in listeners {
            // A consumer failure must not interrupt protocol parsing for other clients.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(&message)));
        }
    }

    async fn take_reconnect_replays(&self) -> Vec<(String, PendingDaemonRequest)> {
        let mut state = self.state.lock().await;
        let ids: Vec<String> = state
            .pending
            .iter()
            .filter(|(_, pending)| pending.awaiting_reconnect)
            .map(|(id, _)| id.clone())
            .collect();
        let mut replays = Vec::new();
        for id in ids {
            if let Some(mut pending) = state.pending.remove(&id) {
                pending.awaiting_reconnect = false;
                pending.deadline = Instant::now() + Duration::from_millis(pending.timeout_ms);
                replays.push((id, pending));
            }
        }
        for (id, pending) in &replays {
            state.pending.insert(id.clone(), clone_pending(pending));
        }
        replays
    }

    async fn acknowledge_command_result(&self, command_id: &str) {
        let hello = self.hello();
        if !self.is_connected()
            || hello.is_none()
            || hello
                .as_ref()
                .is_some_and(|hello| hello.protocol.version < DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION)
        {
            return;
        }
        let hello = hello.expect("checked above");
        let id = {
            let mut state = self.state.lock().await;
            state.request_id += 1;
            format!("daemon_ack_{}", state.request_id)
        };
        let command = Map::from_iter([
            ("id".to_string(), Value::String(id.clone())),
            ("type".to_string(), Value::String("ack_result".to_string())),
            ("commandId".to_string(), Value::String(command_id.to_string())),
        ]);
        let protocol_version = hello.protocol.version.min(DAEMON_PROTOCOL_VERSION);
        let envelope =
            command_envelope_value(&Value::Object(command), &id, Some(&self.protocol_client_id), protocol_version);
        let socket = { self.state.lock().await.socket.clone() };
        if let Some(socket) = socket {
            let _ = socket.write_line(serialize_json_line(&envelope)).await;
        }
    }

    async fn reject_all(&self, error: DaemonClientError, preserve_pending_requests: bool) {
        let mut pending: Vec<PendingDaemonRequest> = Vec::new();
        {
            let mut state = self.state.lock().await;
            let ids: Vec<String> = state.pending.keys().cloned().collect();
            for id in ids {
                let Some(mut entry) = state.pending.remove(&id) else {
                    continue;
                };
                if preserve_pending_requests && entry.recoverable {
                    entry.awaiting_reconnect = true;
                    state.pending.insert(id, entry);
                    continue;
                }
                pending.push(entry);
            }
            let waiters = std::mem::take(&mut state.hello_waiters);
            for waiter in waiters {
                let _ = waiter.sender.send(Err(error.clone()));
            }
        }
        for entry in pending {
            entry.settle(Err(error.clone()));
        }
    }

    async fn notify_closed(self: &Arc<Self>, socket: &Arc<ClientSocket>, error: Option<DaemonSocketClosedError>) {
        {
            let mut state = self.state.lock().await;
            match &state.socket {
                Some(current) if Arc::ptr_eq(current, socket) => {}
                _ => return,
            }
            state.socket = None;
        }
        self.quick_connected.store(false, Ordering::SeqCst);
        let error = DaemonClientError::SocketClosed(error.unwrap_or_else(|| {
            DaemonSocketClosedError::new(&self.socket_path, None, None)
        }));
        let preserve = self.state.lock().await.request_recovery_enabled;
        self.reject_all(error.clone(), preserve).await;
        let listeners: Vec<DaemonClientCloseListener> = {
            let registry = self.close_listeners.lock().expect("close listeners poisoned");
            registry.values().cloned().collect()
        };
        for listener in listeners {
            listener(&error);
        }
        let reconnect_options = self.state.lock().await.reconnect_options.clone();
        if reconnect_options.is_some() && !self.closed.load(Ordering::SeqCst) {
            self.auto_reconnect(error).await;
        }
    }

    async fn auto_reconnect(self: &Arc<Self>, cause: DaemonClientError) {
        let _guard = self.reconnect_busy.lock().await;
        let options = self.state.lock().await.reconnect_options.clone();
        let Some(options) = options else {
            return;
        };
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        self.emit_reconnect_status(&options, DaemonClientReconnectStatus::Reconnecting { error: cause.message() });
        let deadline = Instant::now() + Duration::from_millis(options.timeout_ms.unwrap_or(DEFAULT_RECONNECT_TIMEOUT_MS));
        let mut attempt = 0u32;
        let mut last_error = cause;
        while !self.closed.load(Ordering::SeqCst) && Instant::now() < deadline {
            (options.recover_daemon)().await;
            if self.closed.load(Ordering::SeqCst) {
                return;
            }
            match self.connect(RECONNECT_CONNECT_TIMEOUT_MS).await {
                Ok(()) => match self.wait_for_hello(RECONNECT_HELLO_TIMEOUT_MS).await {
                    Ok(_) => {
                        self.emit_reconnect_status(&options, DaemonClientReconnectStatus::Connected);
                        return;
                    }
                    Err(error) => {
                        last_error = error;
                        self.reset_transport_for_reconnect().await;
                    }
                },
                Err(error) => {
                    last_error = error;
                    self.reset_transport_for_reconnect().await;
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let backoff = 100u64 * 2u64.pow(attempt.min(5));
            let delay_ms = remaining.as_millis() as u64;
            let delay_ms = delay_ms.min(MAX_RECONNECT_DELAY_MS).min(backoff);
            attempt += 1;
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        let failure = DaemonClientError::Message(format!(
            "Daemon reconnection failed: {}",
            last_error.message()
        ));
        self.reject_all(failure.clone(), false).await;
        self.emit_reconnect_status(
            &options,
            DaemonClientReconnectStatus::Failed { error: failure.message() },
        );
        self.state.lock().await.reconnect_options = None;
    }

    fn emit_reconnect_status(&self, options: &DaemonClientReconnectOptions, status: DaemonClientReconnectStatus) {
        if let Some(on_status) = &options.on_status {
            // UI status callbacks must never interrupt transport recovery.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_status(status)));
        }
    }
}

fn clone_pending(pending: &PendingDaemonRequest) -> PendingDaemonRequest {
    PendingDaemonRequest {
        command_type: pending.command_type.clone(),
        timeout_ms: pending.timeout_ms,
        on_progress: pending.on_progress.clone(),
        wire_data: pending.wire_data.clone(),
        awaiting_reconnect: pending.awaiting_reconnect,
        acknowledge_result: pending.acknowledge_result,
        recoverable: pending.recoverable,
        compatibilities: pending.compatibilities.clone(),
        deadline: pending.deadline,
        result: Arc::clone(&pending.result),
        wake: Arc::clone(&pending.wake),
    }
}

pub(crate) fn is_daemon_closing(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    candidate.get("type").and_then(Value::as_str) == Some("daemon_closing")
        && matches!(
            candidate.get("reason").and_then(Value::as_str),
            Some("shutdown") | Some("update")
        )
}

pub(crate) fn is_daemon_request_progress(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    if candidate.get("command").and_then(Value::as_str) != Some("list_saved_sessions")
        || candidate.get("id").and_then(Value::as_str).is_none()
    {
        return false;
    }
    match candidate.get("type").and_then(Value::as_str) {
        Some("session_list_progress") => {
            candidate.get("loaded").and_then(Value::as_f64).is_some()
                && candidate.get("total").and_then(Value::as_f64).is_some()
        }
        Some("session_list_item") => candidate
            .get("session")
            .is_some_and(is_daemon_saved_session_info),
        _ => false,
    }
}

impl DaemonTransportClient for DaemonClient {
    fn hello(&self) -> Option<DaemonHello> {
        DaemonClient::hello(self)
    }

    fn is_connected(&self) -> bool {
        DaemonClient::is_connected(self)
    }

    fn supports_server_capability(&self, capability: &str) -> bool {
        DaemonClient::supports_server_capability(self, capability)
    }

    fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
        DaemonClient::on_message(self, listener)
    }

    fn on_close(&self, listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync> {
        DaemonClient::on_close(self, listener)
    }

    fn enable_request_recovery(&self) {
        let state = self.state.try_lock();
        if let Ok(mut state) = state {
            state.request_recovery_enabled = true;
        }
    }

    fn request_boxed(
        &self,
        command: DaemonCommandBody,
        timeout_ms: Option<u64>,
        options: DaemonClientRequestOptions,
    ) -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonResponse>> {
        let client = self.self_arc();
        Box::pin(async move { client?.request(command, timeout_ms, options).await })
    }

    fn wait_for_hello_boxed(
        &self,
        timeout_ms: u64,
    ) -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonHello>> {
        let client = self.self_arc();
        Box::pin(async move { client?.wait_for_hello(timeout_ms).await })
    }

    fn connect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        let client = self.self_arc();
        Box::pin(async move { client?.connect(timeout_ms).await })
    }

    fn reconnect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        let client = self.self_arc();
        Box::pin(async move { client?.reconnect(timeout_ms).await })
    }

    fn disconnect_for_reconnect_boxed(&self, reason: String) -> futures::future::BoxFuture<'static, ()> {
        let client = self.self_arc();
        Box::pin(async move {
            if let Ok(client) = client {
                client.disconnect_for_reconnect(&reason).await;
            }
        })
    }

    fn reset_transport_for_reconnect_boxed(&self) -> futures::future::BoxFuture<'static, ()> {
        let client = self.self_arc();
        Box::pin(async move {
            if let Ok(client) = client {
                client.reset_transport_for_reconnect().await;
            }
        })
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.quick_connected.store(false, Ordering::SeqCst);
        let socket = self.state.try_lock().ok().and_then(|mut state| state.socket.take());
        if let Some(socket) = socket {
            socket.destroy();
        }
    }
}

#[cfg(test)]
#[path = "daemon_jev_compatibility_tests.rs"]
mod jev_compatibility_tests;

#[cfg(test)]
mod tests {
    use super::super::daemon_protocol::*;
    use super::*;

    #[test]
    fn jsonl_serialization_is_lf_terminated() {
        let line = serialize_json_line(&serde_json::json!({ "a": 1 }));
        assert!(line.ends_with('\n'));
        assert_eq!(line, "{\"a\":1}\n");
    }

    #[test]
    fn closing_and_progress_detection() {
        assert!(is_daemon_closing(&serde_json::json!({
            "type": "daemon_closing",
            "reason": "shutdown"
        })));
        assert!(!is_daemon_closing(&serde_json::json!({
            "type": "daemon_closing",
            "reason": "other"
        })));
        assert!(is_daemon_request_progress(&serde_json::json!({
            "type": "session_list_progress",
            "command": "list_saved_sessions",
            "id": "x",
            "loaded": 1,
            "total": 2
        })));
        assert!(!is_daemon_request_progress(&serde_json::json!({
            "type": "session_list_progress",
            "command": "list",
            "id": "x",
            "loaded": 1,
            "total": 2
        })));
    }

    #[test]
    fn hello_parsing_keeps_the_raw_frame_and_capabilities() {
        let frame = serde_json::json!({
            "type": "daemon_hello",
            "socketPath": "/tmp/prime-agent.sock",
            "protocol": { "name": "prime-agent.daemon", "version": 7 },
            "schemaRevision": 29,
            "appVersion": "9.9.9",
            "clientId": "client-1",
            "serverCapabilities": ["session_input_admission"]
        });
        let hello = DaemonHello::from_value(&frame).expect("hello");
        assert_eq!(hello.protocol.version, 7);
        assert_eq!(hello.schema_revision, Some(29));
        assert!(hello.supports("session_input_admission"));
        assert!(!hello.supports("history_ranges"));
        assert_eq!(hello.raw, frame);
        assert!(is_daemon_hello(&frame));
        assert!(DaemonHello::from_value(&serde_json::json!({ "type": "response" })).is_none());
    }

    #[test]
    fn compatibility_gating_reads_the_hello_capabilities() {
        let hello = DaemonHello::from_value(&serde_json::json!({
            "type": "daemon_hello",
            "protocol": { "name": "prime-agent.daemon", "version": 7 },
            "schemaRevision": 29,
            "serverCapabilities": ["history_ranges"]
        }))
        .expect("hello");
        let view = compatibility_hello(&hello);
        assert!(meets_daemon_command_compatibility(
            &view,
            &daemon_command_compatibility("get_history_range")
        ));
        assert!(!meets_daemon_command_compatibility(
            &view,
            &daemon_command_compatibility("delete_rlm_subagent")
        ));
    }

    #[test]
    fn body_requirements_match_the_typescript_gates() {
        let plain = Map::from_iter([("type".to_string(), Value::String("get_history_range".to_string()))]);
        let plain_requirements = command_compatibilities(&plain);
        assert_eq!(plain_requirements.len(), 1);
        assert_eq!(plain_requirements[0].min_schema_revision, Some(29));

        // A plain body has no `admissionId` field, so no cancellation gate applies.
        let prompt = Map::from_iter([
            ("type".to_string(), Value::String("prompt".to_string())),
            ("activeSessionId".to_string(), Value::String("active-1".to_string())),
        ]);
        assert_eq!(command_compatibilities(&prompt).len(), 1);

        let mut telemetry = Map::from_iter([("type".to_string(), Value::String("attach".to_string()))]);
        telemetry.insert("telemetryDisabled".to_string(), Value::Bool(true));
        let requirements = command_compatibilities(&telemetry);
        assert_eq!(requirements.len(), 2);
        assert_eq!(requirements[0].min_schema_revision, Some(14));
    }

    #[test]
    fn full_command_value_spreads_the_id_over_the_body() {
        let body = Map::from_iter([
            ("type".to_string(), Value::String("list".to_string())),
            ("cwd".to_string(), Value::String("/tmp".to_string())),
        ]);
        assert_eq!(
            full_command_value(&body, "daemon_1"),
            serde_json::json!({ "type": "list", "cwd": "/tmp", "id": "daemon_1" })
        );
    }

    #[test]
    fn response_parsing_requires_the_response_discriminants() {
        let ok = serde_json::json!({
            "id": "daemon_1", "type": "response", "command": "list", "success": true, "data": { "x": 1 }
        });
        assert!(is_daemon_response(&ok));
        let response = daemon_response_from_value(&ok);
        assert_eq!(response.id.as_deref(), Some("daemon_1"));
        assert!(response.success);
        assert_eq!(response.data, Some(serde_json::json!({ "x": 1 })));
        assert!(!is_daemon_response(&serde_json::json!({ "type": "response", "command": "list" })));
    }

    #[test]
    fn endpoint_details_include_the_socket_and_log_path() {
        let details = daemon_endpoint_details("\\\\\\\\.\\\\pipe\\\\prime-agent-daemon");
        assert!(details.contains("Socket: "));
        assert!(details.contains("Daemon log: "));
    }

    #[test]
    fn capability_unavailable_message_matches_the_typescript() {
        let with_capability =
            DaemonCapabilityUnavailableError::new("get_history_range", Some("history_ranges"), false);
        assert_eq!(
            with_capability.message(),
            "The running Prime Agent daemon does not support history_ranges."
        );
        let without = DaemonCapabilityUnavailableError::new("bogus", None, false);
        assert_eq!(without.message(), "The running Prime Agent daemon does not support bogus.");
    }

    #[test]
    fn abort_error_uses_the_typescript_wording() {
        let error = daemon_request_abort_error("prompt");
        assert_eq!(error.message(), "Daemon request \"prompt\" was cancelled");
        assert!(error.is_abort());
    }

    #[test]
    fn socket_closed_error_carries_reason_and_cause() {
        let error = DaemonSocketClosedError::new("/tmp/sock", Some("update"), Some("reset"));
        assert!(error.message().starts_with("Connection to the Prime Agent daemon closed."));
        assert!(error.message().contains(" Reason: update."));
        assert!(error.message().contains(" Cause: reset."));
        assert_eq!(
            get_daemon_socket_close_reason(&DaemonClientError::SocketClosed(error)),
            Some("update".to_string())
        );
    }

    #[test]
    fn agent_dir_prefers_the_environment_override() {
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", "/tmp/agent-dir");
        assert_eq!(get_agent_dir(), "/tmp/agent-dir");
        assert_eq!(get_logs_dir(), "/tmp/agent-dir/logs");
        std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR");
        assert_eq!(expand_tilde_path("/abs/path"), "/abs/path");
    }
}
