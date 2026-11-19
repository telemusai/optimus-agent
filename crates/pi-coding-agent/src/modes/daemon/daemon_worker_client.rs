//! Port of packages/coding-agent/src/modes/daemon/daemon-worker-client.ts
//!
//! The private-framed channel from `modes/session-worker/private-framing.ts` is
//! inlined here (that module belongs to another slice): 8-byte prefix of two
//! big-endian u32 lengths, then the JSON header, then the payload.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::sync::{oneshot, Mutex};

use super::daemon_protocol::DaemonResponse;
use super::daemon_client::DaemonHello;
use super::daemon_client::{
    is_daemon_closing, is_daemon_response, serialize_json_line, DaemonClientError, DaemonClientMessageListener,
    DaemonClientRequestOptions, DaemonClientResult, DaemonCommandBody, DaemonSocketClosedError,
};
use super::daemon_worker_protocol::{
    is_daemon_worker_frame_header, worker_command, DaemonPeerTransportTicket,
};

const FRAME_PREFIX_BYTES: usize = 8;
pub const DEFAULT_PRIVATE_FRAME_MAX_HEADER_BYTES: usize = 1024 * 1024;
pub const DEFAULT_PRIVATE_FRAME_MAX_PAYLOAD_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct PrivateFrame {
    pub header: Value,
    pub payload: Vec<u8>,
}

fn assert_frame_length(name: &str, value: usize, maximum: usize) -> Result<(), String> {
    if value > maximum {
        return Err(format!("Invalid private frame {name}: {value}"));
    }
    Ok(())
}

pub fn encode_private_frame(header: &Value, payload: &[u8]) -> Result<Vec<u8>, String> {
    let header_buffer = serde_json::to_vec(header).map_err(|error| error.to_string())?;
    assert_frame_length("header length", header_buffer.len(), DEFAULT_PRIVATE_FRAME_MAX_HEADER_BYTES)?;
    assert_frame_length("payload length", payload.len(), DEFAULT_PRIVATE_FRAME_MAX_PAYLOAD_BYTES)?;
    if header_buffer.is_empty() {
        return Err("Private frame header cannot be empty".to_string());
    }
    let mut frame = Vec::with_capacity(FRAME_PREFIX_BYTES + header_buffer.len() + payload.len());
    frame.extend_from_slice(&(header_buffer.len() as u32).to_be_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&header_buffer);
    frame.extend_from_slice(payload);
    Ok(frame)
}

#[derive(Default)]
pub struct PrivateFrameDecoder {
    buffered: Vec<u8>,
}

impl PrivateFrameDecoder {
    pub fn new() -> Self {
        Self { buffered: Vec::new() }
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buffered.len()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<PrivateFrame>, String> {
        if !chunk.is_empty() {
            self.buffered.extend_from_slice(chunk);
        }
        let mut frames: Vec<PrivateFrame> = Vec::new();
        let mut offset = 0usize;
        while self.buffered.len() - offset >= FRAME_PREFIX_BYTES {
            let header_length =
                u32::from_be_bytes(self.buffered[offset..offset + 4].try_into().expect("4 bytes")) as usize;
            let payload_length =
                u32::from_be_bytes(self.buffered[offset + 4..offset + 8].try_into().expect("4 bytes")) as usize;
            assert_frame_length("header length", header_length, DEFAULT_PRIVATE_FRAME_MAX_HEADER_BYTES)?;
            assert_frame_length("payload length", payload_length, DEFAULT_PRIVATE_FRAME_MAX_PAYLOAD_BYTES)?;
            if header_length == 0 {
                return Err("Private frame header cannot be empty".to_string());
            }
            let frame_length = FRAME_PREFIX_BYTES + header_length + payload_length;
            if self.buffered.len() - offset < frame_length {
                break;
            }
            let header_start = offset + FRAME_PREFIX_BYTES;
            let payload_start = header_start + header_length;
            let decoded: Value = serde_json::from_slice(&self.buffered[header_start..payload_start])
                .map_err(|error| format!("Invalid private frame header JSON: {error}"))?;
            if !decoded.is_object() || !is_daemon_worker_frame_header(&decoded) {
                return Err("Invalid private frame routing header".to_string());
            }
            frames.push(PrivateFrame {
                header: decoded,
                payload: self.buffered[payload_start..payload_start + payload_length].to_vec(),
            });
            offset += frame_length;
        }
        if offset > 0 {
            self.buffered.drain(..offset);
        }
        Ok(frames)
    }

    pub fn finish(&self) -> Result<(), String> {
        if !self.buffered.is_empty() {
            return Err(format!(
                "Private frame channel ended with {} incomplete bytes",
                self.buffered.len()
            ));
        }
        Ok(())
    }
}

pub type DaemonWorkerFrameListener = Arc<dyn Fn(&PrivateFrame) + Send + Sync>;
pub type DaemonWorkerCloseListener = Arc<dyn Fn(&DaemonClientError) + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonWorkerAuthenticationError {
    pub message: String,
}

impl std::fmt::Display for DaemonWorkerAuthenticationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DaemonWorkerAuthenticationError {}

/// A probe (hello/response/connect) timed out; recovery treats the worker as live-but-slow, never dead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonWorkerProbeTimeoutError {
    pub message: String,
}

impl std::fmt::Display for DaemonWorkerProbeTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DaemonWorkerProbeTimeoutError {}

/// A direct transport closed or rejected a write. Preserve the originating I/O
/// class so routing can distinguish a socket close from a worker-level error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonWorkerTransportError {
    pub operation: &'static str,
    pub kind: std::io::ErrorKind,
    pub source: String,
}

impl DaemonWorkerTransportError {
    fn write(error: std::io::Error) -> Self {
        Self { operation: "write", kind: error.kind(), source: error.to_string() }
    }

    pub fn message(&self) -> String {
        format!("Daemon worker socket {} failed ({:?}): {}", self.operation, self.kind, self.source)
    }

    fn to_client_error(&self, socket_path: &str) -> DaemonClientError {
        DaemonClientError::SocketClosed(DaemonSocketClosedError::new(socket_path, None, Some(&self.message())))
    }
}

impl std::fmt::Display for DaemonWorkerTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.message()) }
}

impl std::error::Error for DaemonWorkerTransportError {}

struct PendingWorkerRequest {
    result: Arc<StdMutex<Option<DaemonClientResult<DaemonResponse>>>>,
    wake: Arc<tokio::sync::Notify>,
}

impl PendingWorkerRequest {
    fn settle(&self, result: DaemonClientResult<DaemonResponse>) {
        let mut slot = self.result.lock().expect("worker pending slot poisoned");
        if slot.is_none() {
            *slot = Some(result);
        }
        // A request has one waiter. Store a permit if its response arrives
        // before request_wire starts waiting (notify_waiters would lose it).
        self.wake.notify_one();
    }
}

struct WorkerHelloWaiter {
    sender: oneshot::Sender<DaemonClientResult<DaemonHello>>,
}

/// Direct worker-socket client used by the routed client and worker probes.
pub struct DaemonWorkerClient {
    socket_path: String,
    writer: Mutex<Option<WorkerWriteHalf>>,
    pending: Mutex<HashMap<String, PendingWorkerRequest>>,
    hello_waiters: Mutex<Vec<WorkerHelloWaiter>>,
    frame_listeners: Arc<StdMutex<HashMap<u64, DaemonWorkerFrameListener>>>,
    message_listeners: Arc<StdMutex<HashMap<u64, DaemonClientMessageListener>>>,
    close_listeners: Arc<StdMutex<HashMap<u64, DaemonWorkerCloseListener>>>,
    next_listener_id: AtomicU64,
    request_id: AtomicU64,
    hello_message: StdMutex<Option<DaemonHello>>,
    connected: AtomicBool,
    direct_peer: AtomicBool,
    direct_closing_reason: StdMutex<Option<String>>,
    reader_task: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    #[cfg(test)]
    close_calls: AtomicU64,
}

fn worker_socket_connect_error(socket_path: &str, error: std::io::Error) -> DaemonClientError {
    DaemonClientError::Message(format!(
        "Failed to connect to daemon worker socket {socket_path}: {error}"
    ))
}

impl DaemonWorkerClient {
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            writer: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
            hello_waiters: Mutex::new(Vec::new()),
            frame_listeners: Arc::new(StdMutex::new(HashMap::new())),
            message_listeners: Arc::new(StdMutex::new(HashMap::new())),
            close_listeners: Arc::new(StdMutex::new(HashMap::new())),
            next_listener_id: AtomicU64::new(0),
            request_id: AtomicU64::new(0),
            hello_message: StdMutex::new(None),
            connected: AtomicBool::new(false),
            direct_peer: AtomicBool::new(false),
            direct_closing_reason: StdMutex::new(None),
            reader_task: StdMutex::new(None),
            #[cfg(test)]
            close_calls: AtomicU64::new(0),
        }
    }

    pub fn hello(&self) -> Option<DaemonHello> {
        self.hello_message.lock().expect("worker hello slot poisoned").clone()
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn close_call_count(&self) -> u64 {
        self.close_calls.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn install_test_write_failure(&self, hello: DaemonHello, kind: std::io::ErrorKind) {
        *self.hello_message.lock().expect("worker hello slot poisoned") = Some(hello);
        self.direct_peer.store(true, Ordering::SeqCst);
        self.connected.store(true, Ordering::SeqCst);
        *self.writer.try_lock().expect("test writer lock") = Some(WorkerWriteHalf::Failing(kind));
    }

    pub fn supports_server_capability(&self, capability: &str) -> bool {
        self.hello().is_some_and(|hello| hello.supports(capability))
    }

    pub async fn connect(self: &Arc<Self>, timeout_ms: u64) -> DaemonClientResult<()> {
        if self.is_connected() {
            return Err(DaemonClientError::Message(
                "Daemon worker client is already connected".to_string(),
            ));
        }
        let connect = async {
            #[cfg(unix)]
            {
                tokio::net::UnixStream::connect(&self.socket_path)
                    .await
                    .map(WorkerStream::Unix)
            }
            #[cfg(windows)]
            {
                tokio::net::windows::named_pipe::ClientOptions::new()
                    .open(&self.socket_path)
                    .map(WorkerStream::Pipe)
            }
        };
        let stream = match tokio::time::timeout(Duration::from_millis(timeout_ms), connect).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(error)) => return Err(worker_socket_connect_error(&self.socket_path, error)),
            Err(_) => {
                return Err(DaemonClientError::Message(format!(
                    "Timed out connecting to daemon worker socket: {}",
                    self.socket_path
                )));
            }
        };
        let (read_half, write_half) = match stream {
            #[cfg(unix)]
            WorkerStream::Unix(stream) => {
                let (read, write) = stream.into_split();
                (WorkerReadHalf::Unix(read), WorkerWriteHalf::Unix(write))
            }
            #[cfg(windows)]
            WorkerStream::Pipe(pipe) => {
                let (read, write) = tokio::io::split(pipe);
                (WorkerReadHalf::Pipe(read), WorkerWriteHalf::Pipe(write))
            }
        };
        *self.writer.lock().await = Some(write_half);
        self.connected.store(true, Ordering::SeqCst);

        let weak = Arc::downgrade(self);
        let reader_task = tokio::spawn(async move {
            let mut read_half = read_half;
            let mut decoder = PrivateFrameDecoder::new();
            let mut buffer = vec![0u8; 64 * 1024];
            let mut eof_error: Option<String> = None;
            loop {
                match read_half.read(&mut buffer).await {
                    Ok(0) => {
                        // `daemon_server.rs` and `private-framing.ts` treat a half-written
                        // trailing frame as an error, not a silent EOF, so the truncation
                        // becomes the close cause instead of a generic socket-closed.
                        eof_error = decoder.finish().err();
                        break;
                    }
                    Ok(read) => {
                        let frames = match decoder.push(&buffer[..read]) {
                            Ok(frames) => frames,
                            Err(error) => {
                                if let Some(client) = weak.upgrade() {
                                    client
                                        .close_with_error(DaemonClientError::Message(error))
                                        .await;
                                }
                                return;
                            }
                        };
                        let Some(client) = weak.upgrade() else {
                            return;
                        };
                        for frame in frames {
                            client.handle_frame(&frame).await;
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Some(client) = weak.upgrade() {
                let error = client.direct_close_error(DaemonClientError::Message(
                    eof_error.unwrap_or_else(|| "Daemon worker socket closed".to_string()),
                ));
                client.close_with_error(error).await;
            }
        });
        *self.reader_task.lock().expect("worker reader slot poisoned") = Some(reader_task);
        Ok(())
    }

    pub async fn wait_for_hello(&self, timeout_ms: u64) -> DaemonClientResult<DaemonHello> {
        if let Some(hello) = self.hello() {
            return Ok(hello);
        }
        if !self.is_connected() {
            return Err(DaemonClientError::Message(
                "Daemon worker client is not connected".to_string(),
            ));
        }
        let (sender, receiver) = oneshot::channel::<DaemonClientResult<DaemonHello>>();
        self.hello_waiters.lock().await.push(WorkerHelloWaiter { sender });
        match tokio::time::timeout(Duration::from_millis(timeout_ms), receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(DaemonClientError::Message(
                "Daemon worker client is not connected".to_string(),
            )),
            Err(_) => Err(DaemonClientError::Message(
                DaemonWorkerProbeTimeoutError {
                    message: "Timed out waiting for daemon worker hello".to_string(),
                }
                .message,
            )),
        }
    }

    pub fn on_frame(&self, listener: DaemonWorkerFrameListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.frame_listeners
            .lock()
            .expect("frame listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.frame_registry());
        Box::new(move || {
            registry.lock().expect("frame listeners poisoned").remove(&id);
        })
    }

    pub fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.message_listeners
            .lock()
            .expect("message listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.message_registry());
        Box::new(move || {
            registry.lock().expect("message listeners poisoned").remove(&id);
        })
    }

    pub fn on_close(&self, listener: DaemonWorkerCloseListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.close_listeners
            .lock()
            .expect("close listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.close_registry());
        Box::new(move || {
            registry.lock().expect("close listeners poisoned").remove(&id);
        })
    }

    fn frame_registry(&self) -> Arc<StdMutex<HashMap<u64, DaemonWorkerFrameListener>>> {
        Arc::clone(&self.frame_listeners)
    }

    fn message_registry(&self) -> Arc<StdMutex<HashMap<u64, DaemonClientMessageListener>>> {
        Arc::clone(&self.message_listeners)
    }

    fn close_registry(&self) -> Arc<StdMutex<HashMap<u64, DaemonWorkerCloseListener>>> {
        Arc::clone(&self.close_listeners)
    }

    /// Progress/recovery options are supervisor-transport features; a direct
    /// request fails fast instead of replaying (no double execution).
    pub async fn request(
        &self,
        command: DaemonCommandBody,
        timeout_ms: u64,
        _options: DaemonClientRequestOptions,
    ) -> DaemonClientResult<DaemonResponse> {
        self.request_wire(command, timeout_ms).await
    }

    pub async fn request_worker(
        &self,
        command: DaemonCommandBody,
        timeout_ms: u64,
    ) -> DaemonClientResult<DaemonResponse> {
        self.request_wire(command, timeout_ms).await
    }

    pub async fn authenticate_worker(
        &self,
        token: &str,
        owner: &[(&str, Value)],
        timeout_ms: u64,
    ) -> DaemonClientResult<DaemonResponse> {
        let mut command = worker_command("worker_auth", &[("token", Value::String(token.to_string()))]);
        for (key, value) in owner {
            command.insert((*key).to_string(), value.clone());
        }
        let response = self.request_worker(command, timeout_ms).await?;
        if !response.success {
            return Err(DaemonClientError::Message(
                DaemonWorkerAuthenticationError {
                    message: response.error.unwrap_or_default(),
                }
                .message,
            ));
        }
        Ok(response)
    }

    pub async fn authenticate_peer(
        &self,
        ticket: &DaemonPeerTransportTicket,
        timeout_ms: u64,
    ) -> DaemonClientResult<()> {
        let command = worker_command("peer_auth", &[
            ("grantId", Value::String(ticket.grant_id.clone())),
            ("token", Value::String(ticket.token.clone())),
            ("workerInstanceId", Value::String(ticket.worker_instance_id.clone())),
            ("purpose", Value::String(ticket.purpose.clone())),
        ]);
        let response = self.request_wire(command, timeout_ms).await?;
        if !response.success {
            return Err(DaemonClientError::Message(response.error.unwrap_or_default()));
        }
        self.direct_peer.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub async fn close(self: &Arc<Self>) {
        #[cfg(test)]
        self.close_calls.fetch_add(1, Ordering::SeqCst);
        self.transition_closed();
        self.abort_reader();
        self.finish_close(DaemonClientError::Message("Daemon worker client closed".to_string())).await;
    }

    /// Synchronous close performs the visible state transition immediately and
    /// schedules the lock-owned teardown. Unlike `try_lock`, no cleanup phase is
    /// skipped when a request or writer is busy.
    pub fn close_now(self: &Arc<Self>) {
        #[cfg(test)]
        self.close_calls.fetch_add(1, Ordering::SeqCst);
        self.transition_closed();
        self.abort_reader();
        let this = Arc::clone(self);
        let close = async move {
            this.finish_close(DaemonClientError::Message("Daemon worker client closed".to_string())).await;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(close);
        } else {
            futures::executor::block_on(close);
        }
    }

    fn transition_closed(&self) -> bool {
        let was_connected = self.connected.swap(false, Ordering::SeqCst);
        self.direct_peer.store(false, Ordering::SeqCst);
        *self.direct_closing_reason.lock().expect("closing reason poisoned") = None;
        was_connected
    }

    fn abort_reader(&self) {
        if let Some(task) = self.reader_task.lock().expect("worker reader slot poisoned").take() {
            task.abort();
        }
    }

    async fn finish_close(&self, error: DaemonClientError) {
        *self.writer.lock().await = None;
        self.reject_all(error).await;
    }

    async fn request_wire(&self, command: DaemonCommandBody, timeout_ms: u64) -> DaemonClientResult<DaemonResponse> {
        if !self.is_connected() {
            return Err(DaemonClientError::Message(
                "Daemon worker client is not connected".to_string(),
            ));
        }
        let id = format!("worker_{}", self.request_id.fetch_add(1, Ordering::SeqCst) + 1);
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
        let mut full_command = command;
        full_command.insert("id".to_string(), Value::String(id.clone()));
        let payload = serialize_json_line(&Value::Object(full_command));
        let header = serde_json::json!({
            "kind": "command",
            "requestId": id,
            "commandType": command_type,
        });
        let result = Arc::new(StdMutex::new(None));
        let wake = Arc::new(tokio::sync::Notify::new());
        self.pending.lock().await.insert(
            id.clone(),
            PendingWorkerRequest {
                result: Arc::clone(&result),
                wake: Arc::clone(&wake),
            },
        );
        let frame = match encode_private_frame(&header, payload.as_bytes()) {
            Ok(frame) => frame,
            Err(error) => {
                self.pending.lock().await.remove(&id);
                return Err(DaemonClientError::Message(error));
            }
        };
        if let Err(error) = self.write_frame(&frame).await {
            let error = DaemonWorkerTransportError::write(error).to_client_error(&self.socket_path);
            // A failed raw write is a transport close, not an ordinary worker
            // error. Settle this request once and close every other pending waiter
            // with the same routable SocketClosed class.
            let was_connected = self.transition_closed();
            self.abort_reader();
            let pending = self.pending.lock().await.remove(&id);
            if let Some(pending) = pending { pending.settle(Err(error.clone())); }
            self.finish_close(error.clone()).await;
            if was_connected { self.notify_close_listeners(&error); }
        }
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
        tokio::select! {
            _ = wake.notified() => {}
            _ = tokio::time::sleep_until(deadline) => {
                let pending = self.pending.lock().await.remove(&id);
                if let Some(pending) = pending {
                    pending.settle(Err(DaemonClientError::Message(
                        DaemonWorkerProbeTimeoutError {
                            message: format!(
                                "Timed out waiting for daemon worker response to {}",
                                command_type
                            ),
                        }
                        .message,
                    )));
                }
            }
        }
        let settled = result.lock().expect("worker pending slot poisoned").take();
        settled.unwrap_or_else(|| {
            Err(DaemonClientError::Message(
                "Daemon worker client is not connected".to_string(),
            ))
        })
    }

    async fn write_frame(&self, frame: &[u8]) -> std::io::Result<()> {
        let mut writer = self.writer.lock().await;
        match writer.as_mut() {
            Some(writer) => match writer {
                #[cfg(unix)]
                WorkerWriteHalf::Unix(writer) => {
                    writer.write_all(frame).await?;
                    writer.flush().await
                }
                #[cfg(windows)]
                WorkerWriteHalf::Pipe(writer) => {
                    writer.write_all(frame).await?;
                    writer.flush().await
                }
                #[cfg(test)]
                WorkerWriteHalf::Failing(kind) => Err(std::io::Error::new(*kind, "injected transport failure")),
            },
            None => Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "socket closed")),
        }
    }

    async fn handle_frame(self: &Arc<Self>, frame: &PrivateFrame) {
        let Some(candidate) = frame.header.as_object() else {
            return;
        };
        if candidate.get("kind").and_then(Value::as_str) != Some("outbound") {
            return;
        }
        let outbound_type = candidate.get("outboundType").and_then(Value::as_str).unwrap_or_default();
        if outbound_type == "response" {
            if let Some(request_id) = candidate.get("requestId").and_then(Value::as_str) {
                let pending = self.pending.lock().await.remove(request_id);
                if let Some(pending) = pending {
                    match serde_json::from_slice::<Value>(&frame.payload) {
                        Ok(response) if is_daemon_response(&response) => {
                            match serde_json::from_value::<DaemonResponse>(response) {
                                Ok(response) => pending.settle(Ok(response)),
                                Err(_) => pending.settle(Err(DaemonClientError::Message(
                                    "Invalid daemon worker response".to_string(),
                                ))),
                            }
                            return;
                        }
                        Ok(_) => {
                            pending.settle(Err(DaemonClientError::Message(
                                "Invalid daemon worker response".to_string(),
                            )));
                            return;
                        }
                        Err(error) => {
                            pending.settle(Err(DaemonClientError::Message(format!(
                                "Invalid daemon worker response: {error}"
                            ))));
                            return;
                        }
                    }
                }
            }
        }
        if outbound_type == "daemon_hello" {
            if let Ok(parsed) = serde_json::from_slice::<Value>(&frame.payload) {
                if let Some(hello) = DaemonHello::from_value(&parsed) {
                    *self.hello_message.lock().expect("worker hello slot poisoned") = Some(hello.clone());
                    let waiters = std::mem::take(&mut *self.hello_waiters.lock().await);
                    for waiter in waiters {
                        let _ = waiter.sender.send(Ok(hello.clone()));
                    }
                }
            }
        }
        let listeners: Vec<DaemonWorkerFrameListener> = {
            let registry = self.frame_listeners.lock().expect("frame listeners poisoned");
            registry.values().cloned().collect()
        };
        for listener in listeners {
            listener(frame);
        }
        if self.direct_peer.load(Ordering::SeqCst)
            && outbound_type != "daemon_hello"
            && outbound_type != "response"
        {
            self.emit_direct_outbound(frame).await;
        }
    }

    /// A malformed outbound frame closes the direct link (the routed client
    /// falls back) instead of throwing into consumers.
    async fn emit_direct_outbound(self: &Arc<Self>, frame: &PrivateFrame) {
        let Some(candidate) = frame.header.as_object() else {
            return;
        };
        if candidate.get("kind").and_then(Value::as_str) != Some("outbound") {
            return;
        }
        let parsed = (|| -> Result<Value, String> {
            if let Some(encoding) = candidate.get("payloadEncoding").and_then(Value::as_str) {
                if encoding != "jsonl" {
                    return Err(format!(
                        "Direct worker sent an unsupported payload encoding: {encoding}"
                    ));
                }
            }
            let parsed: Value = serde_json::from_slice(&frame.payload).map_err(|error| error.to_string())?;
            if parsed.as_object().and_then(|object| object.get("type")).and_then(Value::as_str).is_none() {
                return Err("Direct worker sent an invalid outbound payload".to_string());
            }
            Ok(parsed)
        })();
        let message = match parsed {
            Ok(message) => message,
            Err(error) => {
                let error = self.direct_close_error(DaemonClientError::Message(error));
                self.close_with_error(error).await;
                return;
            }
        };
        if is_daemon_closing(&message) {
            if let Some(reason) = message.get("reason").and_then(Value::as_str) {
                *self.direct_closing_reason.lock().expect("closing reason poisoned") =
                    Some(reason.to_string());
            }
        }
        let listeners: Vec<DaemonClientMessageListener> = {
            let registry = self.message_listeners.lock().expect("message listeners poisoned");
            registry.values().cloned().collect()
        };
        for listener in listeners {
            listener(&message);
        }
    }

    fn direct_close_error(&self, cause: DaemonClientError) -> DaemonClientError {
        if self.direct_peer.load(Ordering::SeqCst) {
            DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
                &self.socket_path,
                self.direct_closing_reason
                    .lock()
                    .expect("closing reason poisoned")
                    .as_deref(),
                Some(&cause.message()),
            ))
        } else {
            cause
        }
    }

    async fn reject_all(&self, error: DaemonClientError) {
        let pending: Vec<PendingWorkerRequest> = {
            let mut pending = self.pending.lock().await;
            let keys: Vec<String> = pending.keys().cloned().collect();
            keys.into_iter().filter_map(|key| pending.remove(&key)).collect()
        };
        for entry in pending {
            entry.settle(Err(error.clone()));
        }
        let waiters = std::mem::take(&mut *self.hello_waiters.lock().await);
        for waiter in waiters {
            let _ = waiter.sender.send(Err(error.clone()));
        }
    }

    fn notify_close_listeners(&self, error: &DaemonClientError) {
        let listeners: Vec<DaemonWorkerCloseListener> = {
            let registry = self.close_listeners.lock().expect("close listeners poisoned");
            registry.values().cloned().collect()
        };
        for listener in listeners { listener(error); }
    }

    async fn close_with_error(self: &Arc<Self>, error: DaemonClientError) {
        let was_connected = self.transition_closed();
        self.finish_close(error.clone()).await;
        if was_connected { self.notify_close_listeners(&error); }
    }
}

#[cfg(unix)]
enum WorkerStream {
    Unix(tokio::net::UnixStream),
}

#[cfg(windows)]
enum WorkerStream {
    Pipe(tokio::net::windows::named_pipe::NamedPipeClient),
}

#[cfg(unix)]
enum WorkerWriteHalf {
    Unix(tokio::net::unix::OwnedWriteHalf),
    #[cfg(test)]
    Failing(std::io::ErrorKind),
}

#[cfg(windows)]
enum WorkerWriteHalf {
    Pipe(tokio::io::WriteHalf<tokio::net::windows::named_pipe::NamedPipeClient>),
    #[cfg(test)]
    Failing(std::io::ErrorKind),
}

#[cfg(unix)]
enum WorkerReadHalf {
    Unix(tokio::net::unix::OwnedReadHalf),
}

#[cfg(windows)]
enum WorkerReadHalf {
    Pipe(tokio::io::ReadHalf<tokio::net::windows::named_pipe::NamedPipeClient>),
}

impl WorkerReadHalf {
    async fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        use tokio::io::AsyncReadExt;
        match self {
            #[cfg(unix)]
            WorkerReadHalf::Unix(reader) => reader.read(buffer).await,
            #[cfg(windows)]
            WorkerReadHalf::Pipe(reader) => reader.read(buffer).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_waiters(client: &DaemonWorkerClient) -> (
        Arc<StdMutex<Option<DaemonClientResult<DaemonResponse>>>>,
        oneshot::Receiver<DaemonClientResult<DaemonHello>>,
    ) {
        let result = Arc::new(StdMutex::new(None));
        let wake = Arc::new(tokio::sync::Notify::new());
        client.pending.try_lock().unwrap().insert(
            "request".into(),
            PendingWorkerRequest { result: Arc::clone(&result), wake },
        );
        let (sender, receiver) = oneshot::channel();
        client.hello_waiters.try_lock().unwrap().push(WorkerHelloWaiter { sender });
        (result, receiver)
    }

    #[tokio::test]
    async fn async_close_waits_for_pending_and_hello_cleanup() {
        let client = Arc::new(DaemonWorkerClient::new("unused"));
        client.connected.store(true, Ordering::SeqCst);
        let (result, receiver) = close_waiters(&client);
        let pending = client.pending.lock().await;
        let hello = client.hello_waiters.lock().await;
        let closing = tokio::spawn({
            let client = Arc::clone(&client);
            async move { client.close().await }
        });
        tokio::task::yield_now().await;
        assert!(!closing.is_finished());
        drop(pending);
        drop(hello);
        tokio::time::timeout(Duration::from_secs(1), closing).await.expect("close must finish").unwrap();
        assert!(result.lock().unwrap().take().unwrap().is_err());
        assert!(receiver.await.unwrap().is_err());
        assert!(client.pending.lock().await.is_empty());
        assert!(client.hello_waiters.lock().await.is_empty());
        assert!(client.writer.lock().await.is_none());
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn synchronous_close_completes_deferred_pending_and_hello_cleanup() {
        let client = Arc::new(DaemonWorkerClient::new("unused"));
        client.connected.store(true, Ordering::SeqCst);
        let (result, receiver) = close_waiters(&client);
        let pending = client.pending.lock().await;
        let hello = client.hello_waiters.lock().await;
        client.close_now();
        assert!(!client.is_connected());
        drop(pending);
        drop(hello);
        tokio::time::timeout(Duration::from_secs(1), receiver).await.expect("deferred close must finish").unwrap().unwrap_err();
        assert!(result.lock().unwrap().take().unwrap().is_err());
        assert!(client.pending.lock().await.is_empty());
        assert!(client.hello_waiters.lock().await.is_empty());
        assert!(client.writer.lock().await.is_none());
    }

    #[tokio::test]
    async fn a_response_before_waiting_keeps_its_notification() {
        let request = PendingWorkerRequest {
            result: Arc::new(StdMutex::new(None)),
            wake: Arc::new(tokio::sync::Notify::new()),
        };
        request.settle(Err(DaemonClientError::Message("first response".to_string())));
        tokio::time::timeout(Duration::from_millis(100), request.wake.notified())
            .await.expect("the response notification is retained");
        request.settle(Err(DaemonClientError::Message("second response".to_string())));
        assert_eq!(request.result.lock().unwrap().as_ref().unwrap().as_ref().unwrap_err().to_string(), "first response");
    }

    #[tokio::test]
    async fn raw_broken_pipe_is_a_structured_socket_closed_error() {
        let client = Arc::new(DaemonWorkerClient::new("/tmp/direct-worker.sock"));
        let hello = DaemonHello::from_value(&serde_json::json!({
            "type":"daemon_hello", "protocol":{"name":"test", "version":1}
        })).expect("hello");
        client.install_test_write_failure(hello, std::io::ErrorKind::BrokenPipe);
        let close_count = Arc::new(AtomicU64::new(0));
        let close_errors = Arc::new(StdMutex::new(Vec::new()));
        let _unsubscribe = client.on_close({
            let close_count = Arc::clone(&close_count);
            let close_errors = Arc::clone(&close_errors);
            Arc::new(move |error| {
                close_count.fetch_add(1, Ordering::SeqCst);
                close_errors.lock().expect("captured close errors").push(error.clone());
            })
        });
        let error = client.request_worker(serde_json::Map::from_iter([
            ("type".into(), Value::String("attach".into())),
        ]), 1000).await.expect_err("write fails");
        let DaemonClientError::SocketClosed(closed) = &error else {
            panic!("raw transport failure must be SocketClosed: {error:?}");
        };
        let cause = closed.cause.as_deref().expect("transport cause");
        assert!(cause.contains("BrokenPipe"), "{cause}");
        assert!(cause.contains("write"), "{cause}");
        assert!(!client.is_connected());
        assert!(client.pending.lock().await.is_empty());
        assert!(client.writer.lock().await.is_none());
        assert_eq!(close_count.load(Ordering::SeqCst), 1, "raw write close notifies exactly once");
        let notified = close_errors.lock().expect("captured close errors").clone();
        assert_eq!(notified, vec![error.clone()], "listener receives the same SocketClosed BrokenPipe error");
    }

    #[test]
    fn frame_round_trip() {
        let header = serde_json::json!({
            "kind": "outbound",
            "outboundType": "session_event",
            "activeSessionId": "abc"
        });
        let frame = encode_private_frame(&header, b"{\"a\":1}\n").expect("frame encodes");
        let mut decoder = PrivateFrameDecoder::new();
        let frames = decoder.push(&frame).expect("frame decodes");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"{\"a\":1}\n");
        assert_eq!(frames[0].header["outboundType"], "session_event");
        assert!(decoder.finish().is_ok());
    }

    #[test]
    fn decoder_waits_for_a_complete_frame() {
        let header = serde_json::json!({ "kind": "command", "requestId": "r", "commandType": "worker_auth" });
        let frame = encode_private_frame(&header, b"payload").expect("frame encodes");
        let mut decoder = PrivateFrameDecoder::new();
        assert!(decoder.push(&frame[..4]).expect("partial push").is_empty());
        assert_eq!(decoder.buffered_bytes(), 4);
        let frames = decoder.push(&frame[4..]).expect("rest decodes");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"payload");
    }

    #[test]
    fn decoder_rejects_an_invalid_routing_header() {
        let header = serde_json::json!({ "kind": "nope" });
        let frame = encode_private_frame(&header, b"").expect("frame encodes");
        let mut decoder = PrivateFrameDecoder::new();
        assert_eq!(
            decoder.push(&frame),
            Err("Invalid private frame routing header".to_string())
        );
    }

    #[test]
    fn decoder_reports_incomplete_bytes_on_finish() {
        let header = serde_json::json!({ "kind": "command", "requestId": "r", "commandType": "x" });
        let frame = encode_private_frame(&header, b"abc").expect("frame encodes");
        let mut decoder = PrivateFrameDecoder::new();
        decoder.push(&frame[..10]).expect("partial push");
        assert_eq!(
            decoder.finish(),
            Err("Private frame channel ended with 10 incomplete bytes".to_string())
        );
    }

    #[test]
    fn empty_header_is_rejected() {
        assert_eq!(
            encode_private_frame(&serde_json::json!({}), &[]).map(|_| ()),
            Ok(())
        );
    }
}
