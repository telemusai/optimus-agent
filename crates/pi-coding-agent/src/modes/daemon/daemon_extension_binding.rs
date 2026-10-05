//! Port of packages/coding-agent/src/modes/daemon/daemon-extension-binding.ts
//!
//! blocked_on: needs core::extensions::{ExtensionUIContext,
//! ExtensionCommandContextActions, ExtensionUIDialogOptions,
//! ExtensionWidgetOptions, WorkingIndicatorOptions}, core::rlm_runtime::SubagentRuntimeHost,
//! modes::agent_connection::snapshot::{createAgentConnectionState, AgentSessionRuntime
//! snapshot source}, interactive::theme::theme, and the AgentSession binding
//! methods (`setExecEnvProvider`, `subscribe`, `bindExtensions`, `reload`,
//! `waitForIdle`, `navigateTree`) plus `AgentSessionRuntime::{setRuntimeEnvScope,
//! setSubagentRuntimeHost, setRebindSession, newSession, fork, switchSession}`.
//! The extension UI surface and the broadcast/rebind flow are ported here; the
//! session-owned calls go through `DaemonExtensionBindingSession`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::oneshot;

use super::active_session_state::{ActiveSessionExtensionUiRequest, ActiveSessionState, DaemonExtensionUIResponse};
use super::daemon_protocol::is_daemon_dialog_extension_ui_request;
use super::daemon_client_env::{exec_env_for_session, with_client_env, EnvMap};

/// Minimal local view of the `DaemonOutbound` variants this binding emits.
/// blocked_on: the full union lives in modes/daemon/daemon-protocol.ts
/// (ca-daemon-a); these four variants keep the same camelCase wire fields.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonOutbound {
    SessionEvent {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        event: Value,
    },
    SessionReplaced {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        state: Value,
        messages: Vec<Value>,
    },
    ExtensionError {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "extensionPath", skip_serializing_if = "Option::is_none", default)]
        extension_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        event: Option<String>,
        error: String,
    },
    ExtensionUiRequest {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        id: String,
        method: String,
        payload: Value,
    },
}

/// The session/runtime calls the binding makes. The session slice owns them.
#[async_trait::async_trait]
pub trait DaemonExtensionBindingSession: Send + Sync {
    fn set_exec_env_provider(&self, client_env: Option<EnvMap>);
    fn set_runtime_env_scope(&self, client_env: Option<EnvMap>);
    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>);
    /// `session.subscribe(listener)`; the returned handle unsubscribes.
    fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>);
    async fn bind_extensions(&self, binding: ExtensionBindingInput);
    async fn wait_for_idle(&self);
    async fn reload(&self);
}

/// `session.bindExtensions({ uiContext, commandContextActions, shutdownHandler, onError })`.
pub struct ExtensionBindingInput {
    pub ui_context: Arc<ExtensionUiContext>,
    pub shutdown_handler: Arc<dyn Fn() + Send + Sync>,
    pub on_error: Arc<dyn Fn(&ExtensionBindingError) + Send + Sync>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtensionBindingError {
    pub extension_path: Option<String>,
    pub event: Option<String>,
    pub error: String,
}

#[derive(Clone)]
pub struct ActiveSessionBindingCallbacks {
    pub get_session: Arc<dyn Fn() -> Option<Arc<dyn DaemonExtensionBindingSession>> + Send + Sync>,
    pub broadcast: Arc<dyn Fn(&ActiveSessionState, DaemonOutbound) + Send + Sync>,
    /// `createConnectionState`; None keeps the daemon default (another slice).
    pub create_connection_state: Option<Arc<dyn Fn(&ActiveSessionState) -> Value + Send + Sync>>,
    pub session_replaced: Option<Arc<dyn Fn(&ActiveSessionState) + Send + Sync>>,
    pub shutdown: Arc<dyn Fn() + Send + Sync>,
    pub subagent_runtime_host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>,
}

/// message_update events carry the full partial assistant message twice; the
/// nested copy is dropped before serialization.
pub fn slim_session_event_for_wire(event: &Value) -> Value {
    let Some(candidate) = event.as_object() else {
        return event.clone();
    };
    if candidate.get("type").and_then(Value::as_str) != Some("message_update") {
        return event.clone();
    }
    let mut slim = candidate.clone();
    if let Some(assistant_message_event) = slim.get_mut("assistantMessageEvent") {
        if let Some(object) = assistant_message_event.as_object_mut() {
            object.shift_remove("partial");
        }
    }
    Value::Object(slim)
}

/// `bindActiveSessionState`.
/// The caller holds `Arc<StdMutex<ActiveSessionState>>`; this takes the same handle rather than a
/// `&mut` borrow so the lock is held only for the synchronous prologue below. `bindActiveSessionState`
/// (daemon-extension-binding.ts:49-97) mutates the state synchronously and awaits only
/// `bindExtensions`, so a `&mut ActiveSessionState` parameter cannot be `Send` when this body awaits.
pub fn bind_active_session_state<'a>(
    state: &'a Arc<StdMutex<ActiveSessionState>>,
    session: Arc<dyn DaemonExtensionBindingSession>,
    callbacks: ActiveSessionBindingCallbacks,
) -> futures::future::BoxFuture<'a, ()> {
    Box::pin(async move {
    // Synchronous prologue: every mutation and subscription happens under this one lock, which is
    // released before the first await. Every runtime rebuild re-loads extensions, which capture
    // client env synchronously at that moment.
    let (ui_context, error_active_session_id, error_broadcast) = {
        let mut state = state.lock().expect("active session poisoned");
        session.set_exec_env_provider(state.client_env.clone());
        session.set_runtime_env_scope(state.client_env.clone());

        if let Some(unsubscribe) = state.unsubscribe.take() {
            unsubscribe();
        }
        if let Some(host) = &callbacks.subagent_runtime_host {
            session.set_subagent_runtime_host(Some(host.clone()));
        }
        let active_session_id = state.active_session_id.clone();
        let broadcast = Arc::clone(&callbacks.broadcast);
        state.unsubscribe = Some(session.subscribe(Arc::new(move |event: &Value| {
            broadcast(
                &broadcast_state(&active_session_id),
                DaemonOutbound::SessionEvent {
                    active_session_id: active_session_id.clone(),
                    event: slim_session_event_for_wire(event),
                },
            );
        })));


        (
            Arc::new(ExtensionUiContext::new(&state, Arc::clone(&callbacks.broadcast))),
            state.active_session_id.clone(),
            Arc::clone(&callbacks.broadcast),
        )
    };
    *ui_context.live_state.lock().expect("live state poisoned") = Arc::downgrade(state);
    let rebind_state = Arc::downgrade(state);
    let rebind_callbacks = callbacks.clone();
    session.set_rebind_session(Arc::new(move || {
        let state = rebind_state.upgrade();
        let session = (rebind_callbacks.get_session)();
        let callbacks = rebind_callbacks.clone();
        Box::pin(async move {
            let (Some(state), Some(session)) = (state, session) else { return; };
            Box::pin(bind_active_session_state(&state, session, callbacks.clone())).await;
            // Daemon callbacks resolve this identity through the live registry. Do
            // not hold its state lock while invoking them: broadcasts acquire it.
            let state = {
                let state = state.lock().expect("active session poisoned");
                let mut snapshot = broadcast_state(&state.active_session_id);
                snapshot.runtime.session.messages = state.runtime.session.messages.clone();
                snapshot
            };
            if let Some(replaced) = &callbacks.session_replaced { replaced(&state); }
            let connection_state = callbacks.create_connection_state.as_ref().map(|create| create(&state)).unwrap_or(Value::Null);
            let messages = state.runtime.session.messages.iter().filter_map(|message| serde_json::to_value(message).ok()).collect();
            (callbacks.broadcast)(&state, DaemonOutbound::SessionReplaced { active_session_id: state.active_session_id.clone(), state: connection_state, messages });
        })
    }));
    session
        .bind_extensions(ExtensionBindingInput {
            ui_context,
            shutdown_handler: Arc::clone(&callbacks.shutdown),
            on_error: Arc::new(move |error: &ExtensionBindingError| {
                error_broadcast(
                    &broadcast_state(&error_active_session_id),
                    DaemonOutbound::ExtensionError {
                        active_session_id: error_active_session_id.clone(),
                        extension_path: error.extension_path.clone(),
                        event: error.event.clone(),
                        error: error.error.clone(),
                    },
                );
            }),
        })
        .await;
    })
}

/// `session.subscribe` listeners receive only the session id; the daemon
/// re-reads live state through the registry, so this stand-in carries the id.
fn broadcast_state(active_session_id: &str) -> ActiveSessionState {
    ActiveSessionState::new(
        active_session_id,
        super::active_session_state::AgentSessionRuntime::default(),
    )
}

/// `createExtensionUIContext`.
pub struct ExtensionUiContext {
    clients: Arc<StdMutex<Vec<Arc<StdMutex<super::active_session_state::DaemonSocketClient>>>>>,
    live_state: StdMutex<std::sync::Weak<StdMutex<ActiveSessionState>>>,

    requests: Arc<StdMutex<std::collections::HashMap<String, oneshot::Sender<DaemonExtensionUIResponse>>>>,
    next_request_id: AtomicU64,
    emit: Arc<dyn Fn(&str, Value) -> String + Send + Sync>,
    client_env: Option<EnvMap>,
}

impl ExtensionUiContext {
    pub fn new(
        state: &ActiveSessionState,
        broadcast: Arc<dyn Fn(&ActiveSessionState, DaemonOutbound) + Send + Sync>,
    ) -> Self {
        let active_session_id = state.active_session_id.clone();
        let clients = Arc::new(StdMutex::new(state.clients.clone()));
        let requests = Arc::new(StdMutex::new(std::collections::HashMap::new()));
        let emit = {
            let active_session_id = active_session_id.clone();
            Arc::new(move |method: &str, payload: Value| -> String {
                let id = new_extension_ui_request_id();
                broadcast(
                    &broadcast_state(&active_session_id),
                    DaemonOutbound::ExtensionUiRequest {
                        active_session_id: active_session_id.clone(),
                        id: id.clone(),
                        method: method.to_string(),
                        payload,
                    },
                );
                id
            }) as Arc<dyn Fn(&str, Value) -> String + Send + Sync>
        };
        Self {
            clients,
            live_state: StdMutex::new(std::sync::Weak::new()),
            requests,
            next_request_id: AtomicU64::new(0),
            emit,
            client_env: state.client_env.clone(),
        }
    }

    pub fn client_env(&self) -> Option<&EnvMap> {
        self.client_env.as_ref()
    }

    fn emit_ui_request(&self, method: &str, payload: Value) -> String {
        let _ = self.next_request_id.fetch_add(1, Ordering::SeqCst);
        (self.emit)(method, payload)
    }

    pub fn has_extension_ui_client_for_method(&self, method: &str) -> bool {
        let live = self.live_state.lock().expect("live state poisoned").upgrade();
        let clients = match live {
            Some(state) => state.lock().expect("active session poisoned").clients.clone(),
            None => self.clients.lock().expect("clients poisoned").clone(),
        };
        if !is_daemon_dialog_extension_ui_request(method) {
            return !clients.is_empty();
        }
        clients
            .iter()
            .any(|client| client.lock().expect("daemon client poisoned").supports_extension_ui)
    }

    /// `dialogRequest(method, payload, opts, fallback, resolveResponse)`.
    pub async fn dialog_request(
        &self,
        method: &str,
        payload: Value,
        timeout: Option<u64>,
        aborted: bool,
        fallback: Option<String>,
        confirmed_fallback: bool,
        response_kind: DialogResponseKind,
    ) -> DialogResult {
        if aborted {
            return DialogResult::from_fallback(fallback, confirmed_fallback, response_kind);
        }
        if !self.has_extension_ui_client_for_method(method) {
            return DialogResult::from_fallback(fallback, confirmed_fallback, response_kind);
        }
        let request_id = self.emit_ui_request(method, payload);
        let (sender, receiver) = oneshot::channel::<DaemonExtensionUIResponse>();
        self.requests
            .lock()
            .expect("requests poisoned")
            .insert(request_id.clone(), sender);
        let timeout_duration = timeout.map(Duration::from_millis);
        let response = match timeout_duration {
            Some(duration) => match tokio::time::timeout(duration, receiver).await {
                Ok(Ok(response)) => Some(response),
                _ => None,
            },
            None => receiver.await.ok(),
        };
        self.requests.lock().expect("requests poisoned").remove(&request_id);
        match response {
            Some(response) => DialogResult::from_response(response, response_kind),
            None => DialogResult::from_fallback(fallback, confirmed_fallback, response_kind),
        }
    }

    pub fn select(&self, title: &str, values: Vec<String>, timeout: Option<u64>) -> DialogFuture {
        let payload = serde_json::json!({ "title": title, "options": values, "timeout": timeout });
        self.spawn_dialog("select", payload, timeout, None, false, DialogResponseKind::Value)
    }

    pub fn confirm(&self, title: &str, message: &str, timeout: Option<u64>) -> DialogFuture {
        let payload = serde_json::json!({ "title": title, "message": message, "timeout": timeout });
        self.spawn_dialog("confirm", payload, timeout, None, false, DialogResponseKind::Confirmed)
    }

    pub fn input(&self, title: &str, placeholder: Option<&str>, timeout: Option<u64>) -> DialogFuture {
        let payload = serde_json::json!({ "title": title, "placeholder": placeholder, "timeout": timeout });
        self.spawn_dialog("input", payload, timeout, None, false, DialogResponseKind::Value)
    }

    pub fn editor(&self, title: &str, prefill: Option<&str>) -> DialogFuture {
        let payload = serde_json::json!({ "title": title, "prefill": prefill });
        self.spawn_dialog("editor", payload, None, None, false, DialogResponseKind::Value)
    }

    fn spawn_dialog(
        &self,
        method: &str,
        payload: Value,
        timeout: Option<u64>,
        fallback: Option<String>,
        confirmed_fallback: bool,
        response_kind: DialogResponseKind,
    ) -> DialogFuture {
        if !self.has_extension_ui_client_for_method(method) {
            return Box::pin(async move { DialogResult::from_fallback(fallback, confirmed_fallback, response_kind) });
        }
        let request_id = self.emit_ui_request(method, payload);
        let (sender, receiver) = oneshot::channel::<DaemonExtensionUIResponse>();
        let live = self.live_state.lock().expect("live state poisoned").clone();
        if let Some(state) = live.upgrade() {
            register_extension_ui_request(&mut state.lock().expect("active session poisoned"), &request_id, Box::new(move |response| { let _ = sender.send(response); }));
        } else {
            self.requests.lock().expect("requests poisoned").insert(request_id.clone(), sender);
        }
        let cleanup = DialogCleanup { live, requests: self.requests.clone(), request_id };
        Box::pin(async move {
            let _cleanup = cleanup;
            let response = match timeout.map(Duration::from_millis) {
                Some(duration) => match tokio::time::timeout(duration, receiver).await {
                    Ok(Ok(response)) => Some(response),
                    _ => None,
                },
                None => receiver.await.ok(),
            };
            match response {
                Some(response) => DialogResult::from_response(response, response_kind),
                None => DialogResult::from_fallback(fallback, confirmed_fallback, response_kind),
            }
        })
    }

    pub fn notify(&self, message: &str, notify_type: Option<&str>) -> String {
        self.emit_ui_request("notify", serde_json::json!({ "message": message, "notifyType": notify_type }))
    }

    pub fn set_status(&self, key: &str, text: &str) -> String {
        self.emit_ui_request("setStatus", serde_json::json!({ "statusKey": key, "statusText": text }))
    }

    pub fn set_working_message(&self, message: &str) -> String {
        self.emit_ui_request("setWorkingMessage", serde_json::json!({ "message": message }))
    }

    pub fn set_working_visible(&self, visible: bool) -> String {
        self.emit_ui_request("setWorkingVisible", serde_json::json!({ "visible": visible }))
    }

    pub fn set_working_indicator(&self, options: Option<Value>) -> String {
        self.emit_ui_request("setWorkingIndicator", serde_json::json!({ "options": options }))
    }

    pub fn set_hidden_thinking_label(&self, label: &str) -> String {
        self.emit_ui_request("setHiddenThinkingLabel", serde_json::json!({ "label": label }))
    }

    /// `setWidget(key, content, options)`: only undefined/array content is emitted.
    pub fn set_widget(&self, key: &str, content: Option<Vec<String>>, placement: Option<&str>) -> Option<String> {
        match content {
            None => Some(self.emit_ui_request(
                "setWidget",
                serde_json::json!({
                    "widgetKey": key,
                    "widgetLines": Value::Null,
                    "widgetPlacement": placement,
                }),
            )),
            Some(lines) => Some(self.emit_ui_request(
                "setWidget",
                serde_json::json!({
                    "widgetKey": key,
                    "widgetLines": lines,
                    "widgetPlacement": placement,
                }),
            )),
        }
    }

    pub fn set_title(&self, title: &str) -> String {
        self.emit_ui_request("setTitle", serde_json::json!({ "title": title }))
    }

    pub fn paste_to_editor(&self, text: &str) -> String {
        self.emit_ui_request("setEditorText", serde_json::json!({ "text": text }))
    }

    pub fn set_editor_text(&self, text: &str) -> String {
        self.emit_ui_request("setEditorText", serde_json::json!({ "text": text }))
    }

    pub fn get_editor_text(&self) -> String {
        String::new()
    }

    pub fn get_all_themes(&self) -> Vec<Value> {
        Vec::new()
    }

    pub fn get_theme(&self) -> Option<Value> {
        None
    }

    pub fn set_theme(&self) -> Result<(), String> {
        Err("Theme switching is not supported in daemon mode".to_string())
    }

    pub fn get_tools_expanded(&self) -> bool {
        false
    }

    pub fn set_tools_expanded(&self) {}
}

struct DialogCleanup {
    live: std::sync::Weak<StdMutex<ActiveSessionState>>,
    requests: Arc<StdMutex<std::collections::HashMap<String, oneshot::Sender<DaemonExtensionUIResponse>>>>,
    request_id: String,
}

impl Drop for DialogCleanup {
    fn drop(&mut self) {
        if let Some(state) = self.live.upgrade() {
            state.lock().expect("active session poisoned").extension_ui_requests.remove(&self.request_id);
        }
        self.requests.lock().expect("requests poisoned").remove(&self.request_id);
    }
}

fn canonical_dialog(
    context: &ExtensionUiContext,
    method: &str,
    mut payload: serde_json::Map<String, Value>,
    opts: Option<crate::core::extensions::types::ExtensionUIDialogOptions>,
    kind: DialogResponseKind,
) -> DialogFuture {
    let signal = opts.as_ref().and_then(|opts| opts.signal.clone());
    if signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
        return Box::pin(async move { DialogResult::from_fallback(None, false, kind) });
    }
    let timeout = opts.as_ref().and_then(|opts| opts.timeout);
    if let Some(timeout) = timeout { payload.insert("timeout".to_string(), Value::from(timeout)); }
    let timeout = timeout.map(|timeout| if !timeout.is_finite() || timeout < 1.0 || timeout > i32::MAX as f64 { 1 } else { timeout as u64 });
    let dialog = context.spawn_dialog(method, Value::Object(payload), timeout, None, false, kind);
    Box::pin(async move {
        if let Some(signal) = signal {
            tokio::select! {
                result = dialog => result,
                _ = signal.cancelled() => DialogResult::from_fallback(None, false, kind),
            }
        } else { dialog.await }
    })
}

impl crate::core::extensions::types::ExtensionUiContext for ExtensionUiContext {
    fn select(&self, title: String, options: Vec<String>, opts: Option<crate::core::extensions::types::ExtensionUIDialogOptions>) -> futures::future::BoxFuture<'static, Option<String>> {
        let payload = serde_json::json!({"title": title, "options": options}).as_object().unwrap().clone();
        let dialog = canonical_dialog(self, "select", payload, opts, DialogResponseKind::Value);
        Box::pin(async move { dialog.await.value })
    }
    fn confirm(&self, title: String, message: String, opts: Option<crate::core::extensions::types::ExtensionUIDialogOptions>) -> futures::future::BoxFuture<'static, bool> {
        let payload = serde_json::json!({"title": title, "message": message}).as_object().unwrap().clone();
        let dialog = canonical_dialog(self, "confirm", payload, opts, DialogResponseKind::Confirmed);
        Box::pin(async move { dialog.await.confirmed })
    }
    fn input(&self, title: String, placeholder: Option<String>, opts: Option<crate::core::extensions::types::ExtensionUIDialogOptions>) -> futures::future::BoxFuture<'static, Option<String>> {
        let mut payload = serde_json::Map::new(); payload.insert("title".to_string(), Value::String(title));
        if let Some(placeholder) = placeholder { payload.insert("placeholder".to_string(), Value::String(placeholder)); }
        let dialog = canonical_dialog(self, "input", payload, opts, DialogResponseKind::Value);
        Box::pin(async move { dialog.await.value })
    }
    fn editor(&self, title: String, prefill: Option<String>) -> futures::future::BoxFuture<'static, Option<String>> {
        let mut payload = serde_json::Map::new(); payload.insert("title".to_string(), Value::String(title));
        if let Some(prefill) = prefill { payload.insert("prefill".to_string(), Value::String(prefill)); }
        let dialog = canonical_dialog(self, "editor", payload, None, DialogResponseKind::Value);
        Box::pin(async move { dialog.await.value })
    }
    fn notify(&self, message: String, kind: Option<String>) { let mut payload = serde_json::json!({"message":message}); if let Some(kind) = kind { payload["notifyType"] = Value::String(kind); } self.emit_ui_request("notify", payload); }
    fn on_terminal_input(&self, _handler: crate::core::extensions::types::TerminalInputHandler) -> Arc<dyn Fn() + Send + Sync> { Arc::new(|| {}) }
    fn set_status(&self, key: String, text: Option<String>) { let mut payload = serde_json::json!({"statusKey":key}); if let Some(text) = text { payload["statusText"] = Value::String(text); } self.emit_ui_request("setStatus", payload); }
    fn set_working_message(&self, message: Option<String>) { let mut payload = serde_json::json!({}); if let Some(message) = message { payload["message"] = Value::String(message); } self.emit_ui_request("setWorkingMessage", payload); }
    fn set_working_visible(&self, visible: bool) { ExtensionUiContext::set_working_visible(self, visible); }
    fn set_working_indicator(&self, options: Option<crate::core::extensions::types::WorkingIndicatorOptions>) { let mut payload = serde_json::json!({}); if let Some(options) = options { payload["options"] = serde_json::to_value(options).expect("working indicator is serializable"); } self.emit_ui_request("setWorkingIndicator", payload); }
    fn set_hidden_thinking_label(&self, label: Option<String>) { let mut payload = serde_json::json!({}); if let Some(label) = label { payload["label"] = Value::String(label); } self.emit_ui_request("setHiddenThinkingLabel", payload); }
    fn set_widget_strings(&self, key: String, content: Option<Vec<String>>, options: Option<crate::core::extensions::types::ExtensionWidgetOptions>) { let mut payload = serde_json::json!({"widgetKey":key}); if let Some(lines) = content { payload["widgetLines"] = serde_json::json!(lines); } if let Some(placement) = options.and_then(|options| options.placement) { payload["widgetPlacement"] = serde_json::to_value(placement).expect("widget placement is serializable"); } self.emit_ui_request("setWidget", payload); }
    fn set_widget_factory(&self, _key: String, _content: Option<crate::core::extensions::types::WidgetFactory>, _options: Option<crate::core::extensions::types::ExtensionWidgetOptions>) {}
    fn set_footer(&self, _factory: Option<crate::core::extensions::types::FooterFactory>) {}
    fn set_header(&self, _factory: Option<crate::core::extensions::types::HeaderFactory>) {}
    fn set_title(&self, title: String) { ExtensionUiContext::set_title(self, &title); }
    fn custom(&self, _factory: crate::core::extensions::types::CustomComponentFactory, _options: Option<Value>) -> crate::core::extensions::types::CustomComponentResult { Box::pin(async { None }) }
    fn paste_to_editor(&self, text: String) { ExtensionUiContext::paste_to_editor(self, &text); }
    fn set_editor_text(&self, text: String) { ExtensionUiContext::set_editor_text(self, &text); }
    fn get_editor_text(&self) -> String { ExtensionUiContext::get_editor_text(self) }
    fn add_autocomplete_provider(&self, _factory: crate::core::extensions::types::AutocompleteProviderFactory) {}
    fn set_editor_component(&self, _factory: Option<crate::core::extensions::types::EditorFactory>) {}
    fn get_editor_component(&self) -> Option<crate::core::extensions::types::EditorFactory> { None }
    fn theme(&self) -> crate::core::extensions::types::Theme { crate::core::extensions::types::Theme { name: crate::modes::interactive::theme::theme::theme().name.clone(), ..Default::default() } }
    fn get_all_themes(&self) -> Vec<crate::core::extensions::types::ThemeInfo> { Vec::new() }
    fn get_theme(&self, _name: String) -> Option<crate::core::extensions::types::Theme> { None }
    fn set_theme(&self, _theme: Value) -> crate::core::extensions::types::SetThemeResult { crate::core::extensions::types::SetThemeResult { success: false, error: Some("Theme switching is not supported in daemon mode".to_string()) } }
    fn get_tools_expanded(&self) -> bool { false }
    fn set_tools_expanded(&self, _expanded: bool) {}
}

pub type DialogFuture = futures::future::BoxFuture<'static, DialogResult>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogResponseKind {
    Value,
    Confirmed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DialogResult {
    pub value: Option<String>,
    pub confirmed: bool,
}

impl DialogResult {
    fn from_response(response: DaemonExtensionUIResponse, kind: DialogResponseKind) -> Self {
        match kind {
            DialogResponseKind::Value => match response {
                DaemonExtensionUIResponse::Value(value) => DialogResult {
                    value: Some(value),
                    confirmed: false,
                },
                _ => DialogResult {
                    value: None,
                    confirmed: false,
                },
            },
            DialogResponseKind::Confirmed => match response {
                DaemonExtensionUIResponse::Confirmed(confirmed) => DialogResult {
                    value: None,
                    confirmed,
                },
                _ => DialogResult {
                    value: None,
                    confirmed: false,
                },
            },
        }
    }

    fn from_fallback(value: Option<String>, confirmed: bool, kind: DialogResponseKind) -> Self {
        match kind {
            DialogResponseKind::Value => DialogResult {
                value,
                confirmed: false,
            },
            DialogResponseKind::Confirmed => DialogResult {
                value: None,
                confirmed,
            },
        }
    }
}

fn new_extension_ui_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Register a UI response resolver on the active session state.
pub fn register_extension_ui_request(
    state: &mut ActiveSessionState,
    request_id: &str,
    resolve: Box<dyn FnOnce(DaemonExtensionUIResponse) + Send>,
) {
    state
        .extension_ui_requests
        .insert(request_id.to_string(), ActiveSessionExtensionUiRequest { resolve });
}

/// `createCommandContextActions` result surface (session-owned calls).
pub struct CommandContextActions {
    pub wait_for_idle: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
    pub reload: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
}

/// `withClientEnv(state.clientEnv, () => state.runtime.session.reload())`.
pub async fn reload_with_client_env(
    session: Arc<dyn DaemonExtensionBindingSession>,
    client_env: Option<EnvMap>,
) {
    with_client_env(client_env.as_ref(), || async move {
        session.reload().await;
    })
    .await;
}

/// `session.setExecEnvProvider(() => execEnvForSession(state.clientEnv))`.
pub fn exec_env_provider(client_env: Option<&EnvMap>) -> std::collections::HashMap<String, Option<String>> {
    exec_env_for_session(client_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_update_events_lose_the_nested_partial_copy() {
        let event = serde_json::json!({
            "type": "message_update",
            "message": { "role": "assistant" },
            "assistantMessageEvent": { "type": "text_delta", "partial": { "role": "assistant" } }
        });
        let slim = slim_session_event_for_wire(&event);
        assert!(slim.get("message").is_some());
        assert!(slim["assistantMessageEvent"].get("partial").is_none());
        assert_eq!(
            slim["assistantMessageEvent"]["type"].as_str(),
            Some("text_delta")
        );
    }

    #[test]
    fn other_events_pass_through_untouched() {
        let event = serde_json::json!({ "type": "turn_end", "partial": 1 });
        assert_eq!(slim_session_event_for_wire(&event), event);
    }

    #[test]
    fn dialog_results_map_values_and_confirmations() {
        assert_eq!(
            DialogResult::from_response(
                DaemonExtensionUIResponse::Value("picked".to_string()),
                DialogResponseKind::Value
            )
            .value
            .as_deref(),
            Some("picked")
        );
        assert!(DialogResult::from_response(
            DaemonExtensionUIResponse::Confirmed(true),
            DialogResponseKind::Confirmed
        )
        .confirmed);
        assert_eq!(
            DialogResult::from_response(
                DaemonExtensionUIResponse::Cancelled,
                DialogResponseKind::Value
            )
            .value,
            None
        );
        assert_eq!(
            DialogResult::from_fallback(Some("fallback".to_string()), false, DialogResponseKind::Value)
                .value
                .as_deref(),
            Some("fallback")
        );
    }

    #[tokio::test]
    async fn a_dialog_without_a_supporting_client_uses_the_fallback() {
        let state = ActiveSessionState::new(
            "active-1",
            super::super::active_session_state::AgentSessionRuntime::default(),
        );
        let context = ExtensionUiContext::new(&state, Arc::new(|_, _| {}));
        assert!(!context.has_extension_ui_client_for_method("select"));
        let result = context
            .dialog_request("select", serde_json::json!({}), None, false, None, false, DialogResponseKind::Value)
            .await;
        assert_eq!(result.value, None);
        let result = context
            .dialog_request(
                "confirm",
                serde_json::json!({}),
                None,
                true,
                None,
                false,
                DialogResponseKind::Confirmed,
            )
            .await;
        assert!(!result.confirmed);
    }

    #[tokio::test]
    async fn canonical_dialog_uses_live_clients_and_response_registry() {
        let state = Arc::new(StdMutex::new(ActiveSessionState::new("active-1", super::super::active_session_state::AgentSessionRuntime::default())));
        let emitted = Arc::new(StdMutex::new(Vec::new()));
        let captured = emitted.clone();
        let context = ExtensionUiContext::new(&state.lock().unwrap(), Arc::new(move |_, outbound| captured.lock().unwrap().push(outbound)));
        *context.live_state.lock().unwrap() = Arc::downgrade(&state);
        // The client attaches after the extension context was created.
        state.lock().unwrap().clients.push(Arc::new(StdMutex::new(super::super::active_session_state::DaemonSocketClient::new("client-1", true))));
        let dialog = crate::core::extensions::types::ExtensionUiContext::select(&context, "Pick".to_string(), vec!["one".to_string()], None);
        let request_id = match &emitted.lock().unwrap()[0] { DaemonOutbound::ExtensionUiRequest { id, payload, .. } => { assert!(payload.get("timeout").is_none()); id.clone() }, other => panic!("unexpected outbound: {other:?}") };
        let request = state.lock().unwrap().extension_ui_requests.remove(&request_id).unwrap();
        (request.resolve)(DaemonExtensionUIResponse::Value("one".to_string()));
        assert_eq!(dialog.await.as_deref(), Some("one"));
        let pending = crate::core::extensions::types::ExtensionUiContext::input(&context, "Input".to_string(), None, None);
        assert_eq!(state.lock().unwrap().extension_ui_requests.len(), 1);
        drop(pending);
        assert!(state.lock().unwrap().extension_ui_requests.is_empty());
    }

    #[tokio::test]
    async fn a_supporting_client_gets_the_request_and_the_timeout_falls_back() {
        let mut state = ActiveSessionState::new(
            "active-1",
            super::super::active_session_state::AgentSessionRuntime::default(),
        );
        state
            .clients
            .push(Arc::new(StdMutex::new(
                super::super::active_session_state::DaemonSocketClient::new("client-1", true),
            )));
        let emitted = Arc::new(StdMutex::new(Vec::new()));
        let captured = Arc::clone(&emitted);
        let context = ExtensionUiContext::new(
            &state,
            Arc::new(move |_, outbound| {
                captured.lock().expect("emitted poisoned").push(outbound);
            }),
        );
        assert!(context.has_extension_ui_client_for_method("select"));
        let result = context
            .dialog_request(
                "select",
                serde_json::json!({ "title": "t" }),
                Some(1),
                false,
                None,
                false,
                DialogResponseKind::Value,
            )
            .await;
        assert_eq!(result.value, None);
        let emitted = emitted.lock().expect("emitted poisoned");
        assert_eq!(emitted.len(), 1);
        match &emitted[0] {
            DaemonOutbound::ExtensionUiRequest { method, .. } => assert_eq!(method, "select"),
            other => panic!("unexpected outbound: {other:?}"),
        }
    }

    #[test]
    fn non_dialog_methods_need_any_client() {
        let mut state = ActiveSessionState::new(
            "active-1",
            super::super::active_session_state::AgentSessionRuntime::default(),
        );
        state
            .clients
            .push(Arc::new(StdMutex::new(
                super::super::active_session_state::DaemonSocketClient::new("client-1", false),
            )));
        let context = ExtensionUiContext::new(&state, Arc::new(|_, _| {}));
        assert!(context.has_extension_ui_client_for_method("notify"));
        assert!(!context.has_extension_ui_client_for_method("select"));
    }

    #[test]
    fn theme_switching_is_rejected_in_daemon_mode() {
        let state = ActiveSessionState::new(
            "active-1",
            super::super::active_session_state::AgentSessionRuntime::default(),
        );
        let context = ExtensionUiContext::new(&state, Arc::new(|_, _| {}));
        assert_eq!(
            context.set_theme().expect_err("unsupported"),
            "Theme switching is not supported in daemon mode"
        );
        assert!(!context.get_tools_expanded());
        assert_eq!(context.get_editor_text(), "");
    }
}
