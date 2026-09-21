//! Live extension objects cross to the UI owner as typed events, never JSON.
use super::*;
use crate::core::extensions::types as ext;
use crate::modes::rpc::{
    rpc_extension_ui_context::{create_rpc_extension_ui_bridge, RpcExtensionUiBridge},
    rpc_types::RpcExtensionUiResponse,
};
use pi_ai::types::BoxFuture;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) enum Event {
    Widget(
        String,
        Option<ext::WidgetFactory>,
        Option<ext::ExtensionWidgetOptions>,
    ),
    Footer(Option<ext::FooterFactory>),
    Header(Option<ext::HeaderFactory>),
    Custom(
        String,
        Arc<dyn ext::Component>,
        Option<Value>,
        tokio::sync::oneshot::Sender<Option<Value>>,
    ),
    Done(String, Value),
    Paste(String),
    ToolsExpanded(bool),
    Reset,
    /// The in-process runtime was REBOUND to a different session (fork, /new,
    /// in-chat /resume). Unlike [`Event::Reset`] (same-session reload), the old
    /// session's host-published state must never survive into the new
    /// session's first frame, so the receiver blanket-resets the surfaces.
    RuntimeRebound,
}

pub(super) struct Bridge {
    rpc: RpcExtensionUiBridge,
    delegate: Arc<dyn ext::ExtensionUiContext>,
    send: mpsc::Sender<HostEvent>,
    pub editor_text: Mutex<String>,
    pub tools_expanded: AtomicBool,
    pub footer_data: Arc<crate::core::footer_data_provider::FooterDataProvider>,
    input: Arc<Mutex<indexmap::IndexMap<String, ext::TerminalInputHandler>>>,
    closed: tokio_util::sync::CancellationToken,
}

impl Bridge {
    pub fn new(send: mpsc::Sender<HostEvent>, cwd: &str) -> Arc<Self> {
        let output = send.clone();
        let rpc = create_rpc_extension_ui_bridge(Arc::new(move |request| {
            let method = if request.method == "set_editor_text" {
                "setEditorText".into()
            } else {
                request.method
            };
            let _ = output.send(HostEvent::Connection(
                wire::AgentConnectionEvent::ExtensionUiRequest {
                    request: wire::AgentConnectionExtensionUiRequest {
                        id: request.id,
                        method,
                        payload: request.payload,
                    },
                },
            ));
        }));
        let delegate = rpc.ui_context();
        Arc::new(Self {
            rpc,
            delegate,
            send,
            editor_text: Mutex::new(String::new()),
            tools_expanded: AtomicBool::new(false),
            footer_data: Arc::new(crate::core::footer_data_provider::FooterDataProvider::new(
                cwd,
            )),
            input: Arc::new(Mutex::new(indexmap::IndexMap::new())),
            closed: tokio_util::sync::CancellationToken::new(),
        })
    }
    pub fn response(&self, reply: &ExtensionReply) -> bool {
        let response = match &reply.1 {
            wire::AgentConnectionExtensionUiResponse::Value { value } => {
                RpcExtensionUiResponse::Value {
                    type_: "extension_ui_response".into(),
                    id: reply.0.clone(),
                    value: value.clone(),
                }
            }
            wire::AgentConnectionExtensionUiResponse::Confirmed { confirmed } => {
                RpcExtensionUiResponse::Confirmed {
                    type_: "extension_ui_response".into(),
                    id: reply.0.clone(),
                    confirmed: *confirmed,
                }
            }
            wire::AgentConnectionExtensionUiResponse::Cancelled { cancelled } => {
                RpcExtensionUiResponse::Cancelled {
                    type_: "extension_ui_response".into(),
                    id: reply.0.clone(),
                    cancelled: *cancelled,
                }
            }
        };
        self.rpc.handle_response(response)
    }
    pub fn filter_input(&self, mut data: String) -> Option<String> {
        let listeners: Vec<_> = self
            .input
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect();
        for listener in listeners {
            if let Some(result) = listener(data.clone()) {
                if result.consume == Some(true) {
                    return None;
                }
                if let Some(next) = result.data {
                    data = next;
                }
            }
        }
        Some(data)
    }
    pub fn reset(&self) {
        self.input.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.footer_data.clear_extension_statuses();
    }
    pub fn close(&self) {
        self.closed.cancel();
        self.rpc.close();
        self.reset();
        self.footer_data.dispose();
    }
    fn emit(&self, method: &str, payload: Value) {
        let _ = self.send.send(HostEvent::Connection(
            wire::AgentConnectionEvent::ExtensionUiRequest {
                request: wire::AgentConnectionExtensionUiRequest {
                    id: uuid::Uuid::new_v4().to_string(),
                    method: method.into(),
                    payload,
                },
            },
        ));
    }
    pub fn tui(&self) -> Arc<dyn ext::Tui> {
        Arc::new(RenderProxy(self.send.clone()))
    }
}
struct RenderProxy(mpsc::Sender<HostEvent>);
impl ext::Tui for RenderProxy {
    fn request_render(&self) {
        let _ = self.0.send(HostEvent::Render);
    }
}
struct Keys;
impl ext::KeybindingsManager for Keys {
    fn get_keys(&self, binding: &str) -> Vec<String> {
        pi_tui::keybindings::get_keybindings().get_keys(binding)
    }
}

fn convert_theme(theme: &crate::modes::interactive::theme::theme::Theme) -> ext::Theme {
    ext::Theme {
        name: theme.name.clone(),
        source_path: theme.source_path.clone(),
        source_info: theme
            .source_info
            .as_ref()
            .and_then(|s| Some(json!({"path":s.path,"source":s.source,"scope":s.scope})))
            .and_then(|v| serde_json::from_value(v).ok()),
        extra: crate::modes::interactive::theme::theme::get_resolved_theme_colors(
            theme.name.as_deref(),
        )
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect(),
    }
}

impl ext::ExtensionUiContext for Bridge {
    fn select(
        &self,
        title: String,
        options: Vec<String>,
        opts: Option<ext::ExtensionUIDialogOptions>,
    ) -> BoxFuture<Option<String>> {
        self.delegate.select(title, options, opts)
    }
    fn confirm(
        &self,
        title: String,
        message: String,
        opts: Option<ext::ExtensionUIDialogOptions>,
    ) -> BoxFuture<bool> {
        self.delegate.confirm(title, message, opts)
    }
    fn input(
        &self,
        title: String,
        placeholder: Option<String>,
        opts: Option<ext::ExtensionUIDialogOptions>,
    ) -> BoxFuture<Option<String>> {
        self.delegate.input(title, placeholder, opts)
    }
    fn editor(&self, title: String, prefill: Option<String>) -> BoxFuture<Option<String>> {
        self.delegate.editor(title, prefill)
    }
    fn notify(&self, message: String, kind: Option<String>) {
        self.delegate.notify(message, kind);
    }
    fn on_terminal_input(&self, handler: ext::TerminalInputHandler) -> Arc<dyn Fn() + Send + Sync> {
        let id = uuid::Uuid::new_v4().to_string();
        let input = self.input.clone();
        input
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.clone(), handler);
        Arc::new(move || {
            input
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .shift_remove(&id);
        })
    }
    fn set_status(&self, key: String, text: Option<String>) {
        self.footer_data.set_extension_status(&key, text.as_deref());
        self.delegate.set_status(key, text);
    }
    fn set_working_message(&self, message: Option<String>) {
        self.emit("setWorkingMessage", json!({"message":message}));
    }
    fn set_working_visible(&self, visible: bool) {
        self.emit("setWorkingVisible", json!({"visible":visible}));
    }
    fn set_working_indicator(&self, options: Option<ext::WorkingIndicatorOptions>) {
        self.emit("setWorkingIndicator", json!({"options":options}));
    }
    fn set_hidden_thinking_label(&self, label: Option<String>) {
        self.emit("setHiddenThinkingLabel", json!({"label":label}));
    }
    fn set_widget_strings(
        &self,
        key: String,
        content: Option<Vec<String>>,
        options: Option<ext::ExtensionWidgetOptions>,
    ) {
        self.delegate.set_widget_strings(key, content, options);
    }
    fn set_widget_factory(
        &self,
        key: String,
        factory: Option<ext::WidgetFactory>,
        options: Option<ext::ExtensionWidgetOptions>,
    ) {
        let _ = self
            .send
            .send(HostEvent::Extension(Event::Widget(key, factory, options)));
    }
    fn set_footer(&self, factory: Option<ext::FooterFactory>) {
        let _ = self.send.send(HostEvent::Extension(Event::Footer(factory)));
    }
    fn set_header(&self, factory: Option<ext::HeaderFactory>) {
        let _ = self.send.send(HostEvent::Extension(Event::Header(factory)));
    }
    fn set_title(&self, title: String) {
        self.delegate.set_title(title);
    }
    fn custom(
        &self,
        factory: ext::CustomComponentFactory,
        options: Option<Value>,
    ) -> ext::CustomComponentResult {
        let id = uuid::Uuid::new_v4().to_string();
        let done_id = id.clone();
        let send = self.send.clone();
        let done_send = send.clone();
        let completed = AtomicBool::new(false);
        let done = Arc::new(move |value| {
            if !completed.swap(true, Ordering::SeqCst) {
                let _ = done_send.send(HostEvent::Extension(Event::Done(done_id.clone(), value)));
            }
        });
        let tui = self.tui();
        let theme = convert_theme(&theme());
        let closed = self.closed.clone();
        Box::pin(async move {
            let component = tokio::select! { component = factory(tui, theme, Arc::new(Keys), done) => component, _ = closed.cancelled() => return None };
            let (reply, wait) = tokio::sync::oneshot::channel();
            if send
                .send(HostEvent::Extension(Event::Custom(
                    id,
                    component.clone(),
                    options,
                    reply,
                )))
                .is_err()
            {
                component.dispose();
                return None;
            }
            tokio::select! { value = wait => value.ok().flatten(), _ = closed.cancelled() => None }
        })
    }
    fn paste_to_editor(&self, text: String) {
        let _ = self.send.send(HostEvent::Extension(Event::Paste(text)));
    }
    fn set_editor_text(&self, text: String) {
        *self.editor_text.lock().unwrap_or_else(|e| e.into_inner()) = text.clone();
        self.emit("setEditorText", json!({"text":text}));
    }
    fn get_editor_text(&self) -> String {
        self.editor_text
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    fn add_autocomplete_provider(&self, factory: ext::AutocompleteProviderFactory) {
        self.delegate.add_autocomplete_provider(factory);
    }
    fn set_editor_component(&self, factory: Option<ext::EditorFactory>) {
        self.delegate.set_editor_component(factory);
    }
    fn get_editor_component(&self) -> Option<ext::EditorFactory> {
        self.delegate.get_editor_component()
    }
    fn theme(&self) -> ext::Theme {
        convert_theme(&theme())
    }
    fn get_all_themes(&self) -> Vec<ext::ThemeInfo> {
        crate::modes::interactive::theme::theme::get_available_themes_with_paths()
            .into_iter()
            .map(|t| ext::ThemeInfo {
                name: t.name,
                path: t.path,
            })
            .collect()
    }
    fn get_theme(&self, name: String) -> Option<ext::Theme> {
        crate::modes::interactive::theme::theme::get_theme_by_name(&name).map(|t| convert_theme(&t))
    }
    fn set_theme(&self, value: Value) -> ext::SetThemeResult {
        let name = value
            .as_str()
            .or_else(|| value.get("name").and_then(Value::as_str));
        let Some(name) = name else {
            return ext::SetThemeResult {
                success: false,
                error: Some("A registered theme name is required".into()),
            };
        };
        let result = crate::modes::interactive::theme::theme::set_theme(name, false);
        let _ = self.send.send(HostEvent::Render);
        ext::SetThemeResult {
            success: result.success,
            error: result.error,
        }
    }
    fn get_tools_expanded(&self) -> bool {
        self.tools_expanded.load(Ordering::Relaxed)
    }
    fn set_tools_expanded(&self, expanded: bool) {
        self.tools_expanded.store(expanded, Ordering::Relaxed);
        let _ = self
            .send
            .send(HostEvent::Extension(Event::ToolsExpanded(expanded)));
    }
}

pub(super) struct ComponentAdapter(pub Arc<dyn ext::Component>, pub bool);
impl TuiComponent for ComponentAdapter {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.0.render(width.max(1.0) as usize)
    }
    fn invalidate(&mut self) {
        self.0.invalidate();
    }
    fn handle_input(&mut self, data: &str) {
        self.0.handle_input(data);
    }
    fn as_focusable(&mut self) -> Option<&mut dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}
impl pi_tui::tui::Focusable for ComponentAdapter {
    fn focused(&self) -> bool {
        self.1
    }
    fn set_focused(&mut self, focused: bool) {
        self.1 = focused;
        self.0.set_focused(focused);
    }
}
impl Drop for ComponentAdapter {
    fn drop(&mut self) {
        self.0.dispose();
    }
}

pub(super) fn overlay_options(value: Option<&Value>) -> pi_tui::tui::OverlayOptions {
    use pi_tui::tui::{OverlayAnchor, OverlayOptions, SizeValue};
    let Some(value) = value else {
        return OverlayOptions::default();
    };
    let value = value.get("overlayOptions").unwrap_or(value);
    let size = |name: &str| {
        value.get(name).and_then(|v| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .map(SizeValue::Number)
                .or_else(|| v.as_str().map(|s| SizeValue::Percent(s.into())))
        })
    };
    OverlayOptions {
        width: size("width"),
        max_height: size("maxHeight"),
        row: size("row"),
        col: size("col"),
        min_width: value.get("minWidth").and_then(Value::as_f64),
        offset_x: value.get("offsetX").and_then(Value::as_i64),
        offset_y: value.get("offsetY").and_then(Value::as_i64),
        anchor: value
            .get("anchor")
            .and_then(Value::as_str)
            .map(|anchor| match anchor {
                "top-left" => OverlayAnchor::TopLeft,
                "top-right" => OverlayAnchor::TopRight,
                "bottom-left" => OverlayAnchor::BottomLeft,
                "bottom-right" => OverlayAnchor::BottomRight,
                "top-center" => OverlayAnchor::TopCenter,
                "bottom-center" => OverlayAnchor::BottomCenter,
                "left-center" => OverlayAnchor::LeftCenter,
                "right-center" => OverlayAnchor::RightCenter,
                _ => OverlayAnchor::Center,
            }),
        non_capturing: value
            .get("nonCapturing")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        scrollback: value
            .get("scrollback")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        suspend_fullscreen_mouse: value
            .get("suspendFullscreenMouse")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        ..Default::default()
    }
}
