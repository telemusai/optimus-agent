//! Port of packages/coding-agent/src/modes/rpc/rpc-mode.ts
//!
//! RPC mode: headless operation with JSON commands on stdin and JSON
//! responses/events on stdout.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};

use serde_json::{Map, Value};
use tokio::sync::Mutex as AsyncMutex;

use crate::core::output_guard::{take_over_stdout, write_raw_stdout};
use crate::modes::agent_connection::types::*;
use crate::modes::rpc::jsonl::{serialize_json_line, JsonlLineReader, JsonlLineReaderOptions, StringDecoder};
use crate::modes::rpc::rpc_extension_ui_context::{create_rpc_extension_ui_bridge, RpcExtensionUiBridge};
use crate::modes::rpc::rpc_types::{RpcCommand, RpcExtensionUiResponse, RpcResponse, RpcSessionState};

pub use crate::modes::rpc::rpc_types::{RpcExtensionUiRequest, RpcObservedSessionEvent};

/// `RpcModeConnectionOptions`.
pub struct RpcModeConnectionOptions {
    /// `bindHeadlessExtensions?: (options) => Promise<void>`.
    pub bind_headless_extensions: Option<
        Arc<
            dyn Fn(
                    Arc<dyn crate::core::extensions::types::ExtensionUiContext>,
                    Arc<dyn Fn() + Send + Sync>,
                ) + Send
                + Sync,
        >,
    >,
}

impl Default for RpcModeConnectionOptions {
    fn default() -> Self {
        Self {
            bind_headless_extensions: None,
        }
    }
}

/// The Node process surface this mode touches.
///
/// blocked_on: a library crate cannot read `process.stdin`, subscribe to OS
/// signals, or call `process.exit`, so the port takes an explicit host seam.
/// Every method mirrors exactly the Node call the TypeScript makes.
pub trait RpcModeHost: Send + Sync {
    /// `process.stdin.on("data")` / `("end")`. Returns the detach function.
    fn attach_stdin(
        &self,
        on_data: Arc<dyn Fn(&[u8]) + Send + Sync>,
        on_end: Arc<dyn Fn() + Send + Sync>,
    ) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.stdin.pause()`.
    fn pause_stdin(&self);
    /// `process.on(signal, handler)`; returns the cleanup function.
    fn on_signal(&self, signal: &str, handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.exit(code)`.
    fn exit(&self, code: i32);
    /// `process.platform`.
    fn platform(&self) -> String;
    /// `killTrackedDetachedChildren()`.
    fn kill_tracked_detached_children(&self);
}

/// `runRpcMode(runtimeHost)`.
///
/// blocked_on: `AgentSessionRuntime` (core/agent-session-runtime.ts) belongs to
/// another slice; callers build the connection and use
/// `run_rpc_mode_with_connection`.
pub async fn run_rpc_mode(
    connection: Arc<dyn AgentConnection>,
    host: Arc<dyn RpcModeHost>,
) -> Result<(), String> {
    run_rpc_mode_with_connection(connection, host, RpcModeConnectionOptions::default()).await
}

/// `runRpcModeWithConnection(connection)`.
pub async fn run_rpc_mode_with_connection(
    connection: Arc<dyn AgentConnection>,
    host: Arc<dyn RpcModeHost>,
    options: RpcModeConnectionOptions,
) -> Result<(), String> {
    take_over_stdout();

    let output: Arc<dyn Fn(Value) + Send + Sync> = Arc::new(move |value: Value| {
        write_raw_stdout(&serialize_json_line(&value));
    });
    // One `output` serves both responses and extension-UI requests: the
    // TypeScript union accepts `RpcResponse | RpcExtensionUIRequest | object`.
    let extension_ui = Arc::new(create_rpc_extension_ui_bridge({
        let output = output.clone();
        Arc::new(move |request: RpcExtensionUiRequest| {
            output(serde_json::to_value(&request).unwrap_or(Value::Null));
        })
    }));
    let state = Arc::new_cyclic(|self_ref| RpcModeState {
        self_ref: Mutex::new(self_ref.clone()),
        connection: connection.clone(),
        host: host.clone(),
        output: output.clone(),
        extension_ui: extension_ui.clone(),
        shutdown_requested: Mutex::new(false),
        shutting_down: Mutex::new(false),
        input_ended: Mutex::new(false),
        connection_output: RpcConnectionOutput::new(output.clone()),
        pending_connection_ui_requests: Mutex::new(HashSet::new()),
        signal_cleanup_handlers: Mutex::new(Vec::new()),
        observations: Mutex::new(HashMap::new()),
        observation_locks: Mutex::new(HashMap::new()),
        prompt_command_tail: Arc::new(AsyncMutex::new(())),
        detach_input: Mutex::new(None),
        pending_commands: std::sync::atomic::AtomicUsize::new(0),
        commands_settled: tokio::sync::Notify::new(),
    });

    let unsubscribe = connection.subscribe(Arc::new({
        let state = state.clone();
        move |event: AgentConnectionEvent| {
            let state = state.clone();
            Box::pin(async move {
                state.handle_connection_event(event).await;
            })
        }
    }));

    for signal in signal_names(&host.platform()) {
        let signal = signal.to_string();
        let state = state.clone();
        let handler_state = state.clone();
        let handler_signal = signal.clone();
        let cleanup = host.on_signal(
            &signal,
            Arc::new(move || {
                let state = handler_state.clone();
                let signal = handler_signal.clone();
                tokio::spawn(async move {
                    state.host.kill_tracked_detached_children();
                    let code = if signal == "SIGHUP" { 129 } else { 143 };
                    state.shutdown(code).await;
                });
            }),
        );
        state
            .signal_cleanup_handlers
            .lock()
            .expect("signals poisoned")
            .push(cleanup);
    }

    if let Some(bind) = &options.bind_headless_extensions {
        let shutdown_handler: Arc<dyn Fn() + Send + Sync> = {
            let state = state.clone();
            Arc::new(move || {
                *state.shutdown_requested.lock().expect("shutdown poisoned") = true;
            })
        };
        bind(extension_ui.ui_context(), shutdown_handler);
    }

    let dispatch: Arc<dyn Fn(String) + Send + Sync> = {
        let state = state.clone();
        Arc::new(move |line: String| {
            state.pending_commands.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let state = state.clone();
            let mut command = Box::pin(async move {
                let _completion = CommandCompletion(state.clone());
                state.handle_input_line(line).await;
            });
            // Preserve the synchronous command prologue before accepting EOF or
            // the next line, as the JavaScript async handler does.
            let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
            if std::future::Future::poll(command.as_mut(), &mut context).is_pending() {
                tokio::spawn(command);
            }
        })
    };

    let reader = Arc::new(Mutex::new(JsonlLineReader::new(
        dispatch.clone(), JsonlLineReaderOptions::default(),
    )));
    let decoder = Arc::new(Mutex::new(StringDecoder::new()));
    let on_end: Arc<dyn Fn() + Send + Sync> = {
        let state = state.clone();
        let reader = reader.clone();
        let decoder = decoder.clone();
        Arc::new(move || {
            let tail = decoder.lock().expect("decoder poisoned").end();
            {
                let mut reader = reader.lock().expect("reader poisoned");
                reader.push(&tail);
                reader.end();
            }
            let state = state.clone();
            tokio::spawn(async move {
                state.on_input_end().await;
            });
        })
    };
    let on_data: Arc<dyn Fn(&[u8]) + Send + Sync> = {
        Arc::new(move |chunk: &[u8]| {
            let text = decoder.lock().expect("decoder poisoned").write(chunk);
            reader.lock().expect("reader poisoned").push(&text);
        })
    };
    let detach = host.attach_stdin(on_data, on_end);
    *state.detach_input.lock().expect("detach poisoned") = Some(detach);

    // `return new Promise(() => {})`: the mode runs until the host exits.
    let (_sender, receiver) = tokio::sync::oneshot::channel::<()>();
    receiver.await.ok();
    let _ = unsubscribe;
    Ok(())
}

fn signal_names(platform: &str) -> Vec<&'static str> {
    let mut signals = vec!["SIGTERM"];
    if platform != "win32" {
        signals.push("SIGHUP");
    }
    signals
}

struct ActiveObservation {
    watcher: Arc<dyn AgentConnectionSessionWatcher>,
    unsubscribe: Arc<dyn Fn() + Send + Sync>,
    ready: bool,
    closed: bool,
    pending_events: Vec<Value>,
}

#[derive(Default)]
struct RpcConnectionOutputState {
    prompt_response_pending: bool,
    buffered: Vec<Value>,
}

struct RpcConnectionOutput {
    output: Arc<dyn Fn(Value) + Send + Sync>,
    state: Mutex<RpcConnectionOutputState>,
    // Serialize the pending check, append, and drain. Emit without the state lock.
    pump_tail: Mutex<()>,
    #[cfg(test)]
    after_pending_read: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl RpcConnectionOutput {
    fn new(output: Arc<dyn Fn(Value) + Send + Sync>) -> Self {
        Self {
            output,
            state: Mutex::new(RpcConnectionOutputState::default()),
            pump_tail: Mutex::new(()),
            #[cfg(test)]
            after_pending_read: None,
        }
    }

    fn begin_prompt(&self) {
        let _tail = self.pump_tail.lock().expect("connection output pump poisoned");
        self.state.lock().expect("connection output poisoned").prompt_response_pending = true;
    }

    fn finish_prompt(&self) {
        let _tail = self.pump_tail.lock().expect("connection output pump poisoned");
        let events = {
            let mut state = self.state.lock().expect("connection output poisoned");
            state.prompt_response_pending = false;
            state.buffered.drain(..).collect::<Vec<_>>()
        };
        self.emit(events);
    }

    fn output_event(&self, event: Value) {
        let _tail = self.pump_tail.lock().expect("connection output pump poisoned");
        let events = {
            let mut state = self.state.lock().expect("connection output poisoned");
            let prompt_response_pending = state.prompt_response_pending;
            #[cfg(test)]
            if let Some(hook) = &self.after_pending_read {
                hook();
            }
            state.buffered.push(event);
            if prompt_response_pending {
                Vec::new()
            } else {
                state.buffered.drain(..).collect()
            }
        };
        self.emit(events);
    }

    fn flush(&self) {
        let _tail = self.pump_tail.lock().expect("connection output pump poisoned");
        let events = {
            let mut state = self.state.lock().expect("connection output poisoned");
            if state.prompt_response_pending {
                Vec::new()
            } else {
                state.buffered.drain(..).collect()
            }
        };
        self.emit(events);
    }

    fn emit(&self, events: Vec<Value>) {
        for event in events {
            (self.output)(event);
        }
    }
}

struct RpcModeState {
    /// Weak self-reference for the tasks that need the shared state handle.
    self_ref: Mutex<Weak<RpcModeState>>,
    connection: Arc<dyn AgentConnection>,
    host: Arc<dyn RpcModeHost>,
    output: Arc<dyn Fn(Value) + Send + Sync>,
    extension_ui: Arc<RpcExtensionUiBridge>,
    shutdown_requested: Mutex<bool>,
    shutting_down: Mutex<bool>,
    input_ended: Mutex<bool>,
    connection_output: RpcConnectionOutput,
    pending_connection_ui_requests: Mutex<HashSet<String>>,
    signal_cleanup_handlers: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    observations: Mutex<HashMap<String, ActiveObservation>>,
    observation_locks: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
    prompt_command_tail: Arc<AsyncMutex<()>>,
    detach_input: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    pending_commands: std::sync::atomic::AtomicUsize,
    commands_settled: tokio::sync::Notify,
}

struct CommandCompletion(Arc<RpcModeState>);

impl Drop for CommandCompletion {
    fn drop(&mut self) {
        self.0.pending_commands.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        self.0.commands_settled.notify_one();
    }
}

impl RpcModeState {
    /// The `this` handle used by spawned tasks (`void shutdown(...)`, `void
    /// stopObservation(...)`).
    fn clone_arc(&self) -> Arc<RpcModeState> {
        self.self_ref
            .lock()
            .expect("self reference poisoned")
            .upgrade()
            .expect("rpc mode state outlived its own tasks")
    }

    fn is_dialog_method(method: &str) -> bool {
        matches!(method, "select" | "confirm" | "input" | "editor")
    }

    fn output_connection_event(&self, event: Value) {
        self.connection_output.output_event(event);
    }

    async fn handle_connection_event(&self, event: AgentConnectionEvent) {
        match event {
            AgentConnectionEvent::SessionEvent { event } => {
                self.output_connection_event(session_event_to_json(&event));
            }
            AgentConnectionEvent::ExtensionError {
                extension_path,
                event,
                error,
            } => {
                self.output_connection_event(serde_json::json!({
                    "type": "extension_error",
                    "extensionPath": extension_path,
                    "event": event,
                    "error": error,
                }));
            }
            AgentConnectionEvent::ExtensionUiRequest { request } => {
                let method = if request.method == "setEditorText" {
                    "set_editor_text".to_string()
                } else {
                    request.method.clone()
                };
                if *self.input_ended.lock().expect("input ended poisoned") && Self::is_dialog_method(&method) {
                    let _ = self
                        .connection
                        .respond_to_extension_ui_request(
                            &request.id,
                            AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                        )
                        .await;
                    return;
                }
                if matches!(
                    method.as_str(),
                    "select"
                        | "confirm"
                        | "input"
                        | "editor"
                        | "notify"
                        | "setStatus"
                        | "setWidget"
                        | "setTitle"
                        | "set_editor_text"
                ) {
                    if Self::is_dialog_method(&method) {
                        self.pending_connection_ui_requests
                            .lock()
                            .expect("pending ui poisoned")
                            .insert(request.id.clone());
                    }
                    let mut object = Map::new();
                    object.insert("type".to_string(), Value::String("extension_ui_request".to_string()));
                    object.insert("id".to_string(), Value::String(request.id.clone()));
                    object.insert("method".to_string(), Value::String(method));
                    if let Value::Object(payload) = request.payload {
                        for (key, value) in payload {
                            object.insert(key, value);
                        }
                    }
                    (self.output)(Value::Object(object));
                }
            }
            AgentConnectionEvent::Closed { error } => {
                let exit_code = if error.is_some() { 1 } else { 0 };
                self.shutdown(exit_code).await;
            }
            _ => {}
        }
    }

    async fn cancel_pending_extension_ui(&self) {
        self.extension_ui.close();
        let request_ids: Vec<String> = {
            let mut pending = self.pending_connection_ui_requests.lock().expect("pending ui poisoned");
            pending.drain().collect()
        };
        for id in request_ids {
            let _ = self
                .connection
                .respond_to_extension_ui_request(
                    &id,
                    AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                )
                .await;
        }
    }

    async fn stop_observation(&self, active_session_id: &str) {
        let observation = self
            .observations
            .lock()
            .expect("observations poisoned")
            .remove(active_session_id);
        if let Some(observation) = observation {
            (observation.unsubscribe)();
            observation.watcher.close().await;
        }
    }

    fn activate_observation(&self, active_session_id: &str) {
        let pending: Vec<Value> = {
            let mut observations = self.observations.lock().expect("observations poisoned");
            let Some(observation) = observations.get_mut(active_session_id) else {
                return;
            };
            if observation.ready {
                return;
            }
            observation.ready = true;
            observation.pending_events.drain(..).collect()
        };
        for event in pending {
            self.output_connection_event(event);
        }
        let closed = self
            .observations
            .lock()
            .expect("observations poisoned")
            .get(active_session_id)
            .map(|observation| observation.closed)
            .unwrap_or(false);
        if closed {
            let state = self.clone_arc();
            let active_session_id = active_session_id.to_string();
            tokio::spawn(async move {
                state.stop_observation(&active_session_id).await;
            });
        }
    }

    /// `shutdown(exitCode = 0)`.
    async fn shutdown(&self, exit_code: i32) {
        if *self.shutting_down.lock().expect("shutting down poisoned") {
            self.host.exit(exit_code);
            return;
        }
        *self.shutting_down.lock().expect("shutting down poisoned") = true;
        self.cancel_pending_extension_ui().await;
        let cleanups: Vec<Arc<dyn Fn() + Send + Sync>> = self
            .signal_cleanup_handlers
            .lock()
            .expect("signals poisoned")
            .drain(..)
            .collect();
        for cleanup in cleanups {
            cleanup();
        }
        if let Some(detach) = self.detach_input.lock().expect("detach poisoned").take() {
            detach();
        }
        self.host.pause_stdin();
        let session_ids: Vec<String> = self
            .observations
            .lock()
            .expect("observations poisoned")
            .keys()
            .cloned()
            .collect();
        for active_session_id in session_ids {
            self.stop_observation(&active_session_id).await;
        }
        let _ = self.connection.dispose().await;
        self.host.exit(exit_code);
    }

    /// `handleInputLine(line)`.
    async fn handle_input_line(self: &Arc<Self>, line: String) {
        let parsed: Value = match serde_json::from_str(&line) {
            Ok(parsed) => parsed,
            Err(error) => {
                (self.output)(response_value(RpcResponse::error(
                    None,
                    "parse",
                    &format!("Failed to parse command: {error}"),
                )));
                return;
            }
        };
        if !parsed.is_object() || parsed.get("type").and_then(Value::as_str).is_none() {
            (self.output)(response_value(RpcResponse::error(
                None,
                "parse",
                "Invalid command: expected an object with a string type",
            )));
            return;
        }

        if parsed.get("type").and_then(Value::as_str) == Some("extension_ui_response") {
            let Ok(response) = serde_json::from_value::<RpcExtensionUiResponse>(parsed.clone()) else {
                (self.output)(response_value(RpcResponse::error(
                    None,
                    "parse",
                    "Invalid command: expected an object with a string type",
                )));
                return;
            };
            if self.extension_ui.handle_response(response.clone()) {
                return;
            }
            self.pending_connection_ui_requests
                .lock()
                .expect("pending ui poisoned")
                .remove(response.id());
            let daemon_response = match &response {
                RpcExtensionUiResponse::Cancelled { cancelled, .. } if *cancelled => {
                    AgentConnectionExtensionUiResponse::Cancelled { cancelled: true }
                }
                RpcExtensionUiResponse::Value { value, .. } => {
                    AgentConnectionExtensionUiResponse::Value { value: value.clone() }
                }
                RpcExtensionUiResponse::Confirmed { confirmed, .. } => {
                    AgentConnectionExtensionUiResponse::Confirmed {
                        confirmed: *confirmed,
                    }
                }
                RpcExtensionUiResponse::Cancelled { .. } => {
                    AgentConnectionExtensionUiResponse::Confirmed { confirmed: false }
                }
            };
            let _ = self
                .connection
                .respond_to_extension_ui_request(response.id(), daemon_response)
                .await;
            return;
        }

        let Ok(command) = serde_json::from_value::<RpcCommand>(parsed.clone()) else {
            let type_name = parsed
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            (self.output)(response_value(RpcResponse::error(
                None,
                &type_name,
                &format!("Unknown command: {type_name}"),
            )));
            return;
        };

        if matches!(command, RpcCommand::Observe { .. } | RpcCommand::Unobserve { .. }) {
            let active_session_id = match &command {
                RpcCommand::Observe { active_session_id, .. }
                | RpcCommand::Unobserve { active_session_id, .. } => active_session_id.clone(),
                _ => unreachable!(),
            };
            let lock = {
                let mut locks = self.observation_locks.lock().expect("observation locks poisoned");
                locks
                    .entry(active_session_id)
                    .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                    .clone()
            };
            let _guard = lock.lock().await;
            self.execute_command(command).await;
            return;
        }
        if matches!(command, RpcCommand::Prompt { .. }) {
            let _guard = self.prompt_command_tail.clone().lock_owned().await;
            self.execute_command(command).await;
            return;
        }
        self.execute_command(command).await;
    }

    /// `executeCommand()` inside `handleInputLine`.
    async fn execute_command(self: &Arc<Self>, command: RpcCommand) {
        let is_prompt = matches!(command, RpcCommand::Prompt { .. });
        if is_prompt {
            self.connection_output.begin_prompt();
        }
        let id = command.id().map(str::to_string);
        match self.handle_command(&command).await {
            Ok(response) => {
                (self.output)(response_value(response));
                if let RpcCommand::Observe { active_session_id, .. } = &command {
                    self.activate_observation(active_session_id);
                }
            }
            Err(message) => {
                (self.output)(response_value(RpcResponse::error(
                    id.as_deref(),
                    command.type_name(),
                    &message,
                )));
            }
        }
        if is_prompt {
            self.connection_output.finish_prompt();
        }
        if *self.shutdown_requested.lock().expect("shutdown poisoned") {
            self.shutdown(0).await;
        }
    }

    /// `onInputEnd()`.
    async fn on_input_end(self: &Arc<Self>) {
        *self.input_ended.lock().expect("input ended poisoned") = true;
        if let Some(detach) = self.detach_input.lock().expect("detach poisoned").take() {
            detach();
        }
        self.host.pause_stdin();
        self.cancel_pending_extension_ui().await;
        while self.pending_commands.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            self.commands_settled.notified().await;
        }
        let idle_result = self.connection.wait_for_idle().await;
        self.connection_output.flush();
        match idle_result {
            Ok(()) => self.shutdown(0).await,
            Err(_) => self.shutdown(1).await,
        }
    }

    /// `handleCommand(command)`.
    async fn handle_command(&self, command: &RpcCommand) -> Result<RpcResponse, String> {
        let id = command.id();
        match command {
            RpcCommand::Prompt {
                message,
                images,
                streaming_behavior,
                ..
            } => {
                self.connection
                    .prompt(
                        message,
                        Some(AgentConnectionPromptOptions {
                            images: images.clone().map(|images| {
                                images
                                    .into_iter()
                                    .filter_map(|image| serde_json::from_value(image.clone()).ok())
                                    .collect()
                            }),
                            streaming_behavior: streaming_behavior.clone(),
                            source: Some("rpc".to_string()),
                            queue_if_busy: None,
                        }),
                    )
                    .await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::Steer { message, images, .. } => {
                self.connection
                    .steer(
                        message,
                        images.clone().map(|images| {
                            images
                                .into_iter()
                                .filter_map(|image| serde_json::from_value(image.clone()).ok())
                                .collect()
                        }),
                    )
                    .await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::FollowUp { message, images, .. } => {
                self.connection
                    .follow_up(
                        message,
                        images.clone().map(|images| {
                            images
                                .into_iter()
                                .filter_map(|image| serde_json::from_value(image.clone()).ok())
                                .collect()
                        }),
                    )
                    .await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::Abort { .. } => {
                self.connection.abort().await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::NewSession { parent_session, .. } => {
                let cancelled = self
                    .connection
                    .new_session(Some(AgentConnectionNewSessionOptions {
                        parent_session: parent_session.clone(),
                    }))
                    .await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "cancelled": cancelled })),
                ))
            }
            RpcCommand::GetState { .. } => {
                let state = self.connection.get_state().await?;
                let rpc_state = RpcSessionState {
                    model: state.model,
                    thinking_level: state.thinking_level,
                    is_streaming: state.is_streaming,
                    is_compacting: state.is_compacting,
                    steering_mode: state.steering_mode,
                    follow_up_mode: state.follow_up_mode,
                    session_file: state.session_file,
                    session_id: state.session_id,
                    session_name: state.session_name,
                    auto_compaction_enabled: state.auto_compaction_enabled,
                    message_count: state.message_count,
                    session_actions: state.session_actions,
                    goal: state.goal,
                };
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::to_value(&rpc_state).unwrap_or(Value::Null)),
                ))
            }
            RpcCommand::SetModel { provider, model_id, .. } => {
                let model = self.connection.set_model(provider, model_id).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::to_value(&model).unwrap_or(Value::Null)),
                ))
            }
            RpcCommand::CycleModel { .. } => {
                let result = self.connection.cycle_model(None).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(match result {
                        Some(result) => serde_json::to_value(&result).unwrap_or(Value::Null),
                        None => Value::Null,
                    }),
                ))
            }
            RpcCommand::GetAvailableModels { .. } => {
                let models = self.connection.get_available_models().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "models": models })),
                ))
            }
            RpcCommand::SetThinkingLevel { level, .. } => {
                self.connection.set_thinking_level(*level).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::CycleThinkingLevel { .. } => {
                let level = self.connection.cycle_thinking_level().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(match level {
                        Some(level) => serde_json::json!({ "level": level }),
                        None => Value::Null,
                    }),
                ))
            }
            RpcCommand::SetSteeringMode { mode, .. } => {
                self.connection.set_steering_mode(mode).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::SetFollowUpMode { mode, .. } => {
                self.connection.set_follow_up_mode(mode).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::Compact {
                custom_instructions, ..
            } => {
                let result = self.connection.compact(custom_instructions.as_deref()).await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(result)))
            }
            RpcCommand::Refine {
                instructions,
                rollback_id,
                global,
                ..
            } => {
                let mut options = Map::new();
                if let Some(instructions) = instructions {
                    options.insert("instructions".to_string(), Value::String(instructions.clone()));
                }
                if let Some(rollback_id) = rollback_id {
                    options.insert("rollbackId".to_string(), Value::String(rollback_id.clone()));
                }
                if let Some(global) = global {
                    options.insert("global".to_string(), Value::Bool(*global));
                }
                let result = self.connection.refine(Value::Object(options)).await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(result)))
            }
            RpcCommand::SetAutoCompaction { enabled, .. } => {
                self.connection.set_auto_compaction_enabled(*enabled).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::SetAutoRetry { enabled, .. } => {
                self.connection.set_auto_retry_enabled(*enabled).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::AbortRetry { .. } => {
                self.connection.abort_retry().await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::Bash { command: text, .. } => {
                let result = self.connection.execute_bash_and_wait(text).await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(result)))
            }
            RpcCommand::AbortBash { .. } => {
                self.connection.abort_bash().await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::GetSessionStats { .. } => {
                let stats = self.connection.get_session_stats().await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(stats)))
            }
            RpcCommand::ExportHtml { output_path, .. } => {
                let path = self.connection.export_to_html(output_path.as_deref()).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "path": path })),
                ))
            }
            RpcCommand::SwitchSession { session_path, .. } => {
                let cancelled = self.connection.switch_session(session_path, None).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "cancelled": cancelled })),
                ))
            }
            RpcCommand::Fork { entry_id, .. } => {
                let result = self.connection.fork(entry_id, None).await?;
                let selected_text = result
                    .get("selectedText")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let cancelled = result
                    .get("cancelled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "text": selected_text, "cancelled": cancelled })),
                ))
            }
            RpcCommand::Clone { .. } => {
                let tree = self.connection.get_session_tree().await?;
                let Some(leaf_id) = tree.leaf_id else {
                    return Ok(RpcResponse::error(
                        id,
                        command.type_name(),
                        "Cannot clone session: no current entry selected",
                    ));
                };
                let result = self
                    .connection
                    .fork(
                        &leaf_id,
                        Some(AgentConnectionForkOptions {
                            position: Some("at".to_string()),
                        }),
                    )
                    .await?;
                let cancelled = result
                    .get("cancelled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "cancelled": cancelled })),
                ))
            }
            RpcCommand::GetForkMessages { .. } => {
                let messages = self.connection.get_user_messages_for_forking().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "messages": messages })),
                ))
            }
            RpcCommand::GetLastAssistantText { .. } => {
                let text = self.connection.get_last_assistant_text().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "text": text })),
                ))
            }
            RpcCommand::SetSessionName { name, .. } => {
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Ok(RpcResponse::error(
                        id,
                        command.type_name(),
                        "Session name cannot be empty",
                    ));
                }
                self.connection.set_session_name(&name).await?;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::GetMessages { .. } => {
                let messages = self.connection.get_messages().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "messages": messages })),
                ))
            }
            RpcCommand::SendMessage {
                target_active_session_id,
                message,
                ..
            } => {
                let receipt = self
                    .connection
                    .send_agent_message(target_active_session_id, message)
                    .await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(receipt)))
            }
            RpcCommand::AgentMessagesStatus { .. } => {
                let status = self.connection.get_agent_message_status().await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(status)))
            }
            RpcCommand::AgentMessagesPause { .. } => {
                let status = self.connection.pause_agent_messages().await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(status)))
            }
            RpcCommand::AgentMessagesResume { .. } => {
                let status = self.connection.resume_agent_messages().await?;
                Ok(RpcResponse::success(id, command.type_name(), Some(status)))
            }
            RpcCommand::AgentMessagesClear { .. } => {
                let cleared = self.connection.clear_agent_messages().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "cleared": cleared })),
                ))
            }
            RpcCommand::ListSchedules { include_inactive, .. } => {
                let jobs = self.connection.list_cron_jobs(include_inactive.unwrap_or(false)).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "jobs": jobs })),
                ))
            }
            RpcCommand::AddSchedule { schedule, prompt, .. } => {
                let job = self.connection.add_cron_job(schedule, prompt).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "job": job })),
                ))
            }
            RpcCommand::CancelSchedule { job_id, .. } => {
                let job = self.connection.cancel_cron_job(job_id).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "job": job })),
                ))
            }
            RpcCommand::ListHeartbeats { .. } => {
                let heartbeats = self.connection.list_heartbeats().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "heartbeats": heartbeats })),
                ))
            }
            RpcCommand::GetHeartbeat { .. } => {
                let heartbeat = self.connection.get_heartbeat().await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                ))
            }
            RpcCommand::SetHeartbeat {
                schedule,
                prompt,
                delivery_mode,
                ..
            } => {
                let heartbeat = self
                    .connection
                    .set_heartbeat(schedule, prompt, delivery_mode.as_deref())
                    .await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                ))
            }
            RpcCommand::UpdateHeartbeat { action, .. } => {
                let heartbeat = self.connection.update_heartbeat(action.clone()).await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                ))
            }
            RpcCommand::ManageHeartbeat {
                active_session_id,
                job_id,
                action,
                ..
            } => {
                let heartbeat = self
                    .connection
                    .manage_heartbeat(active_session_id, job_id, action.clone())
                    .await?;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                ))
            }
            RpcCommand::Observe { active_session_id, .. } => {
                let existing = self
                    .observations
                    .lock()
                    .expect("observations poisoned")
                    .get(active_session_id)
                    .map(|observation| observation.watcher.clone());
                if let Some(watcher) = existing {
                    let messages = watcher.get_messages().await;
                    return Ok(RpcResponse::success(
                        id,
                        command.type_name(),
                        Some(serde_json::json!({ "messages": messages })),
                    ));
                }
                let Some(watcher) = self.connection.watch_session(active_session_id).await? else {
                    return Err(format!("Unknown active session: {active_session_id}"));
                };
                let watcher: Arc<dyn AgentConnectionSessionWatcher> = Arc::from(watcher);
                let observation = ActiveObservation {
                    watcher: watcher.clone(),
                    unsubscribe: Arc::new(|| {}),
                    ready: false,
                    closed: false,
                    pending_events: Vec::new(),
                };
                self.observations
                    .lock()
                    .expect("observations poisoned")
                    .insert(active_session_id.clone(), observation);

                let state = self.clone_arc();
                let observed_session_id = active_session_id.clone();
                let unsubscribe = watcher.subscribe(Arc::new(move |event: AgentConnectionEvent| {
                    let state = state.clone();
                    let active_session_id = observed_session_id.clone();
                    Box::pin(async move {
                        let observed = match event {
                            AgentConnectionEvent::SessionEvent { event } => Some(serde_json::json!({
                                "type": "observed_session_event",
                                "activeSessionId": active_session_id,
                                "event": session_event_to_json(&event),
                            })),
                            AgentConnectionEvent::Closed { error } => Some(serde_json::json!({
                                "type": "observed_session_closed",
                                "activeSessionId": active_session_id,
                                "error": error,
                            })),
                            _ => None,
                        };
                        let Some(observed) = observed else {
                            return;
                        };
                        let closed = observed.get("type").and_then(Value::as_str)
                            == Some("observed_session_closed");
                        // One lock for the whole decision, like the TypeScript block:
                        // the ready branch re-emits after the guard is dropped.
                        let buffered = observed.clone();
                        let ready = {
                            let mut observations = state.observations.lock().expect("observations poisoned");
                            match observations.get_mut(&active_session_id) {
                                Some(observation) => {
                                    if closed {
                                        observation.closed = true;
                                    }
                                    if observation.ready {
                                        true
                                    } else {
                                        observation.pending_events.push(buffered);
                                        false
                                    }
                                }
                                None => false,
                            }
                        };
                        if ready {
                            state.output_connection_event(observed);
                            if closed {
                                state.stop_observation(&active_session_id).await;
                            }
                        }
                    })
                }));
                {
                    let mut observations = self.observations.lock().expect("observations poisoned");
                    if let Some(observation) = observations.get_mut(active_session_id) {
                        observation.unsubscribe = Arc::from(unsubscribe);
                    }
                }

                let messages = watcher.get_messages().await;
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "messages": messages })),
                ))
            }
            RpcCommand::Unobserve { active_session_id, .. } => {
                self.stop_observation(active_session_id).await;
                Ok(RpcResponse::success(id, command.type_name(), None))
            }
            RpcCommand::GetCommands { .. } => {
                let commands: Vec<Value> = self
                    .connection
                    .get_commands()
                    .await?
                    .into_iter()
                    .map(|available| {
                        serde_json::json!({
                            "name": available.name,
                            "description": available.description,
                            "source": available.source,
                            "sourceInfo": available.source_info,
                        })
                    })
                    .collect();
                Ok(RpcResponse::success(
                    id,
                    command.type_name(),
                    Some(serde_json::json!({ "commands": commands })),
                ))
            }
        }
    }
}

/// `event.event` serialized with the TypeScript session-event names.
fn session_event_to_json(event: &AgentConnectionSessionEvent) -> Value {
    serde_json::to_value(event).unwrap_or(Value::Null)
}

/// `output(value)` for an `RpcResponse`.
fn response_value(response: RpcResponse) -> Value {
    serde_json::to_value(&response).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    fn recording_output() -> (RpcConnectionOutput, Arc<Mutex<Vec<Value>>>) {
        let emitted = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&emitted);
        let output = RpcConnectionOutput::new(Arc::new(move |event| {
            sink.lock().expect("recorded output poisoned").push(event);
        }));
        (output, emitted)
    }

    #[test]
    fn late_connection_output_is_flushed_without_another_prompt() {
        let (mut output, emitted) = recording_output();
        let pending_read = Arc::new(Barrier::new(2));
        let append = Arc::new(Barrier::new(2));
        output.after_pending_read = Some({
            let pending_read = Arc::clone(&pending_read);
            let append = Arc::clone(&append);
            Arc::new(move || {
                pending_read.wait();
                append.wait();
            })
        });
        let output = Arc::new(output);
        output.begin_prompt();
        let late = json!({"type": "late_event"});

        let event_task = {
            let output = Arc::clone(&output);
            let late = late.clone();
            std::thread::spawn(move || output.output_event(late))
        };
        pending_read.wait();
        let (flushed_tx, flushed_rx) = mpsc::channel();
        let flush_task = {
            let output = Arc::clone(&output);
            std::thread::spawn(move || {
                output.finish_prompt();
                flushed_tx.send(()).expect("flush receiver dropped");
            })
        };
        let flushed_before_append = flushed_rx.recv_timeout(Duration::from_millis(50)).is_ok();
        append.wait();
        event_task.join().expect("event task panicked");
        flush_task.join().expect("flush task panicked");

        assert!(!flushed_before_append, "prompt flush passed the pending append");
        assert_eq!(*emitted.lock().expect("recorded output poisoned"), vec![late]);
        let state = output.state.lock().expect("connection output poisoned");
        assert!(state.buffered.is_empty());
        assert!(!state.prompt_response_pending);
    }

    #[test]
    fn eof_flush_drains_buffered_connection_output_in_order() {
        let (output, emitted) = recording_output();
        let first = json!({"sequence": 1});
        let second = json!({"sequence": 2});
        output.begin_prompt();
        output.output_event(first.clone());
        output.output_event(second.clone());
        assert!(emitted.lock().expect("recorded output poisoned").is_empty());

        // Reproduce the stranded buffer left by the former separate-lock race.
        output.state.lock().expect("connection output poisoned").prompt_response_pending = false;
        output.flush();

        assert_eq!(
            *emitted.lock().expect("recorded output poisoned"),
            vec![first, second]
        );
        assert!(output.state.lock().expect("connection output poisoned").buffered.is_empty());
    }
}
