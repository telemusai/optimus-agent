//! Native adapters for the agents-view controller and daemon roster.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_tui::tui::{
    Component, Focusable, FullscreenOptions, InputListenerResult, TuiStopOptions, TUI,
};
use serde_json::Value;

use super::native_wire::normalize_browser_numbers;

use super::agents_view_mode as view;
use super::roster_store as roster;
use crate::core::settings_manager::SettingsManager;
use crate::main_entry::{AgentsViewSeamOptions, InteractiveModeSeamOptions};
use crate::modes::agent_connection::types as wire;
use crate::modes::daemon::daemon_client::{self, DaemonClient};
use crate::modes::interactive::components::custom_editor::{CustomEditor, CustomEditorOptions};
use crate::modes::interactive::theme::theme::{self as palette, theme};

thread_local! {
    // Trait adapters are Send, but all component access remains on the owner thread.
    static UI: RefCell<Option<NativeUi>> = const { RefCell::new(None) };
}

struct Frame(Vec<String>);
impl Component for Frame {
    fn render(&mut self, _width: f64) -> Vec<String> {
        self.0.clone()
    }
    fn invalidate(&mut self) {}
}

struct NativeUi {
    tui: Rc<RefCell<TUI>>,
    editor: CustomEditor,
    content: Rc<RefCell<Frame>>,
    dock: Rc<RefCell<Frame>>,
    input: Rc<RefCell<Vec<String>>>,
    submissions: Rc<RefCell<Vec<String>>>,
    started: bool,
}

impl NativeUi {
    fn new(settings: &SettingsManager) -> Self {
        let tui = Rc::new(RefCell::new(TUI::new(
            Box::new(pi_tui::terminal::ProcessTerminal::new()),
            Some(settings.get_show_hardware_cursor()),
        )));
        tui.borrow_mut()
            .set_clear_on_shrink(settings.get_clear_on_shrink());
        let mut editor = CustomEditor::new(
            tui.clone(),
            editor_theme(),
            CustomEditorOptions {
                padding_x: Some(settings.get_editor_padding_x()),
                autocomplete_max_visible: Some(settings.get_autocomplete_max_visible()),
                placeholder: Some(view::SEARCH_PROMPT_PLACEHOLDER.into()),
                placeholder_color: Some(Box::new(|text| theme().fg("dim", text))),
                ..Default::default()
            },
        );
        editor.editor_mut().set_focused(true);
        let submissions = Rc::new(RefCell::new(Vec::new()));
        let submitted = submissions.clone();
        editor.editor_mut().on_submit = Some(Box::new(move |text| {
            submitted.borrow_mut().push(text.to_string())
        }));
        let input = Rc::new(RefCell::new(Vec::new()));
        let queued = input.clone();
        tui.borrow_mut().add_input_listener(Box::new(move |data| {
            queued.borrow_mut().push(data.to_string());
            InputListenerResult {
                consume: true,
                data: None,
            }
        }));
        let content = Rc::new(RefCell::new(Frame(Vec::new())));
        let dock = Rc::new(RefCell::new(Frame(Vec::new())));
        tui.borrow_mut().add_child(content.clone());
        tui.borrow_mut().add_child(dock.clone());
        Self {
            tui,
            editor,
            content,
            dock,
            input,
            submissions,
            started: false,
        }
    }

    fn start(&mut self) {
        if self.started {
            return;
        }
        self.tui.borrow_mut().start();
        self.tui.borrow_mut().enter_fullscreen(FullscreenOptions {
            scroll: vec![self.content.clone()],
            dock: self.dock.clone(),
            mouse: false,
            viewport_controls: false,
        });
        self.started = true;
    }

    fn pause(&mut self) {
        if self.started {
            self.tui.borrow_mut().stop(TuiStopOptions::default());
            self.started = false;
        }
    }
}

fn with_ui<T>(f: impl FnOnce(&mut NativeUi) -> T) -> T {
    UI.with(|ui| {
        f(ui.borrow_mut()
            .as_mut()
            .expect("agents UI is initialized on its owner thread"))
    })
}

struct UiGuard;
impl Drop for UiGuard {
    fn drop(&mut self) {
        UI.with(|ui| {
            if let Some(mut ui) = ui.borrow_mut().take() {
                ui.pause();
            }
        });
        palette::stop_theme_watcher();
    }
}

struct NativeTerminal;
impl view::AgentsViewTerminal for NativeTerminal {
    fn rows(&self) -> usize {
        with_ui(|ui| ui.tui.borrow().terminal.rows())
    }
    fn columns(&self) -> usize {
        with_ui(|ui| ui.tui.borrow().terminal.columns())
    }
    fn request_render(&self, force: bool) {
        with_ui(|ui| {
            if force {
                ui.tui.borrow_mut().request_render_forced();
            } else {
                ui.tui.borrow_mut().request_render();
            }
        });
    }
    fn set_title(&self, title: &str) {
        with_ui(|ui| ui.tui.borrow_mut().terminal.set_title(title));
    }
    fn poll_input(&self) -> Result<Option<Vec<String>>, String> {
        with_ui(|ui| {
            ui.start();
            if !ui
                .tui
                .borrow_mut()
                .terminal
                .poll_input()
                .map_err(|error| error.to_string())?
            {
                return Ok(None);
            }
            ui.tui.borrow_mut().drain_input();
            Ok(Some(std::mem::take(&mut *ui.input.borrow_mut())))
        })
    }
    fn present(&self, lines: Vec<String>, dock: Vec<String>) -> Result<(), String> {
        with_ui(|ui| {
            let changed = ui.content.borrow().0 != lines || ui.dock.borrow().0 != dock;
            ui.content.borrow_mut().0 = lines;
            ui.dock.borrow_mut().0 = dock;
            ui.start();
            if changed {
                ui.tui.borrow_mut().request_render();
            }
            ui.tui
                .borrow_mut()
                .run_pending_render(chrono::Utc::now().timestamp_millis() as f64);
        });
        Ok(())
    }
}

struct NativeEditor;
impl view::AgentsViewEditor for NativeEditor {
    fn set_text(&mut self, text: &str) {
        with_ui(|ui| ui.editor.editor_mut().set_text(text));
    }
    fn get_text(&self) -> String {
        with_ui(|ui| ui.editor.editor().get_text())
    }
    fn get_expanded_text(&self) -> String {
        with_ui(|ui| ui.editor.editor().get_expanded_text())
    }
    fn set_placeholder(&mut self, text: &str) {
        with_ui(|ui| ui.editor.set_placeholder(Some(text.into())));
    }
    fn render(&mut self, width: usize) -> Vec<String> {
        with_ui(|ui| {
            let rows = ui.tui.borrow().terminal.rows();
            ui.editor.editor_mut().set_terminal_rows(rows);
            ui.editor.render(width as f64)
        })
    }
    fn invalidate(&mut self) {
        with_ui(|ui| ui.editor.invalidate());
    }
    fn get_lines(&self) -> Vec<String> {
        with_ui(|ui| ui.editor.editor().get_lines())
    }
    fn get_cursor(&self) -> (usize, usize) {
        with_ui(|ui| ui.editor.editor().get_cursor())
    }
    fn handle_input(&mut self, data: &str) -> bool {
        with_ui(|ui| ui.editor.handle_input(data));
        true
    }
    fn focus(&mut self) {
        with_ui(|ui| ui.editor.editor_mut().set_focused(true));
    }
    fn is_focused(&self) -> bool {
        with_ui(|ui| ui.editor.editor().focused())
    }
    fn take_submissions(&mut self) -> Vec<String> {
        with_ui(|ui| std::mem::take(&mut *ui.submissions.borrow_mut()))
    }
}

struct NativeTheme;
impl view::AgentsViewTheme for NativeTheme {
    fn fg(&self, color: &str, text: &str) -> String {
        theme().fg(color, text)
    }
    fn bg(&self, color: &str, text: &str) -> String {
        theme().bg(color, text)
    }
    fn bold(&self, text: &str) -> String {
        theme().bold(text)
    }
    fn italic(&self, text: &str) -> String {
        theme().italic(text)
    }
    fn selection_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        theme().get_selection_background_color()
    }
}

struct UiServices {
    cwd: String,
    theme: String,
    themes: Vec<String>,
    hardware_cursor: bool,
    clear_on_shrink: bool,
    padding: usize,
    autocomplete_rows: usize,
}
impl view::AgentsViewUiServices for UiServices {
    fn get_initial_cwd(&self) -> String {
        self.cwd.clone()
    }
    fn get_theme(&self) -> String {
        self.theme.clone()
    }
    fn get_themes(&self) -> Vec<String> {
        self.themes.clone()
    }
    fn get_show_hardware_cursor(&self) -> bool {
        self.hardware_cursor
    }
    fn get_clear_on_shrink(&self) -> bool {
        self.clear_on_shrink
    }
    fn get_editor_padding_x(&self) -> usize {
        self.padding
    }
    fn get_autocomplete_max_visible(&self) -> usize {
        self.autocomplete_rows
    }
}

pub(crate) struct NativeTransport(Arc<DaemonClient>);
impl NativeTransport {
    pub(crate) fn new(socket: &str) -> Self {
        Self(DaemonClient::create(socket))
    }
    fn hello_value(&self, hello: daemon_client::DaemonHello) -> roster::DaemonHello {
        roster::DaemonHello {
            socket_path: self.0.socket_path().into(),
            server_capabilities: hello.server_capabilities,
            client_id: hello
                .raw
                .get("clientId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            schema_revision: hello.schema_revision.map(i64::from),
        }
    }
}

fn outbound(value: &Value) -> roster::DaemonOutbound {
    match value.get("type").and_then(Value::as_str) {
        Some("roster_update") => roster::DaemonOutbound::RosterUpdate {
            changed: value
                .get("changed")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|entry| serde_json::from_value(normalize_browser_numbers(entry.clone())).ok())
                .collect(),
            removed: value
                .get("removed")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok()),
            resync: value.get("resync").and_then(Value::as_bool),
        },
        Some("heartbeats_changed") => roster::DaemonOutbound::HeartbeatsChanged,
        _ => roster::DaemonOutbound::Other,
    }
}

impl roster::DaemonTransport for NativeTransport {
    fn fresh_transport(&self) -> Option<Arc<dyn roster::DaemonTransport>> {
        Some(Arc::new(Self::new(self.0.socket_path())))
    }
    fn hello(&self) -> Option<roster::DaemonHello> {
        self.0.hello().map(|hello| self.hello_value(hello))
    }
    fn is_connected(&self) -> bool {
        self.0.is_connected()
    }
    fn supports_server_capability(&self, capability: &str) -> bool {
        self.0.supports_server_capability(capability)
    }
    fn wait_for_hello(
        &self,
        timeout_ms: u64,
    ) -> roster::TransportFuture<Result<roster::DaemonHello, String>> {
        let client = self.0.clone();
        Box::pin(async move {
            let hello = client
                .wait_for_hello(timeout_ms)
                .await
                .map_err(|error| error.message())?;
            Ok(NativeTransport(client).hello_value(hello))
        })
    }
    fn request(
        &self,
        command: Value,
        timeout_ms: u64,
        options: roster::DaemonClientRequestOptions,
    ) -> roster::TransportFuture<Result<roster::DaemonResponse, String>> {
        let client = self.0.clone();
        Box::pin(async move {
            let body = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be an object".to_string())?;
            let response = client
                .request(
                    body,
                    Some(timeout_ms),
                    daemon_client::DaemonClientRequestOptions {
                        on_progress: options.on_progress.map(Arc::from),
                        recoverable: options.recoverable,
                        signal: None,
                    },
                )
                .await
                .map_err(|error| error.message())?;
            Ok(roster::DaemonResponse {
                command: response.command,
                success: response.success,
                data: response.data.map(normalize_browser_numbers),
                error: response.error,
            })
        })
    }
    fn on_message(&self, listener: roster::MessageListener) -> Box<dyn Fn() + Send + Sync> {
        self.0
            .on_message(Arc::new(move |value| listener(&outbound(value))))
    }
    fn on_close(&self, listener: Box<dyn Fn(&str) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.0
            .on_close(Arc::new(move |error| listener(&error.message())))
    }
    fn connect(&self, timeout_ms: u64) -> roster::TransportFuture<Result<(), String>> {
        let client = self.0.clone();
        Box::pin(async move {
            client
                .connect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }
    fn reconnect(&self, timeout_ms: u64) -> roster::TransportFuture<Result<(), String>> {
        let client = self.0.clone();
        Box::pin(async move {
            client
                .reconnect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }
    fn close(&self) {
        let client = self.0.clone();
        tokio::spawn(async move {
            client.close().await;
        });
    }
    fn socket_path(&self) -> String {
        self.0.socket_path().into()
    }
}

type Attached = Arc<Mutex<Option<Arc<dyn wire::AgentConnection>>>>;
struct ConnectionFactory {
    socket: String,
    config: crate::core::agent_session_config::AgentSessionRuntimeConfig,
    attached: Attached,
}
struct ConnectionHandle(Arc<dyn wire::AgentConnection>);
impl view::DaemonAgentConnectionHandle for ConnectionHandle {
    fn prompt(
        &self,
        message: &str,
        behavior: Option<&str>,
    ) -> view::TransportFuture<Result<(), String>> {
        let connection = self.0.clone();
        let message = message.to_string();
        let behavior = behavior.map(str::to_string);
        Box::pin(async move {
            connection
                .prompt(
                    &message,
                    Some(wire::AgentConnectionPromptOptions {
                        streaming_behavior: behavior,
                        ..Default::default()
                    }),
                )
                .await
        })
    }
    fn dispose(&self) -> view::TransportFuture<Result<(), String>> {
        let connection = self.0.clone();
        Box::pin(async move { connection.dispose().await })
    }
}
impl view::DaemonAgentConnectionFactory for ConnectionFactory {
    fn attach(
        &self,
        client: roster::DaemonTransportClient,
        active: &str,
        options: view::AttachOptions,
    ) -> view::TransportFuture<Result<Arc<dyn view::DaemonAgentConnectionHandle>, String>> {
        let socket = self.socket.clone();
        let mut config = self.config.clone();
        let active = active.to_string();
        let attached = self.attached.clone();
        config.telemetry_disabled = options.telemetry_disabled.or(config.telemetry_disabled);
        Box::pin(async move {
            let result = crate::main_entry::create_daemon_client_connection(
                crate::main_entry::CreateDaemonClientConnectionOptions {
                    socket_path: socket,
                    config,
                    session_path: None,
                    continue_recent: None,
                    active_session_id: Some(active),
                    client_owned: Some(false),
                    no_session: None,
                    supports_extension_ui: options.supports_extension_ui,
                    defer_session_events: true,
                },
            )
            .await;
            client.close();
            let (connection, _) = result?;
            let connection: Arc<dyn wire::AgentConnection> = connection;
            *attached.lock().expect("attached connection poisoned") = Some(connection.clone());
            Ok(Arc::new(ConnectionHandle(connection))
                as Arc<dyn view::DaemonAgentConnectionHandle>)
        })
    }
}

struct NativeInteractive {
    options: view::InteractiveModeOptions,
    attached: Attached,
}
impl view::InteractiveModeHandle for NativeInteractive {
    fn run(&mut self) -> view::TransportFuture<Result<view::InteractiveRunResult, String>> {
        let options = self.options.clone();
        let attached = self.attached.clone();
        Box::pin(async move {
            let connection = attached
                .lock()
                .expect("attached connection poisoned")
                .take()
                .ok_or_else(|| "No attached agents-view session".to_string())?;
            let mut source = options.source_summary.clone();
            source.rlm_depth = options.session_depth;
            with_ui(NativeUi::pause);
            let result = crate::modes::interactive::native_host::run_interactive_mode_for_agents(
                InteractiveModeSeamOptions {
                    daemon_socket_path: options.daemon_socket_path,
                    migrated_providers: options.migrated_providers.unwrap_or_default(),
                    model_fallback_message: view::combine_agents_view_startup_notices(&[
                        options.model_fallback_message.as_deref(),
                        options.startup_notice.as_deref(),
                    ]),
                    initial_message: None,
                    initial_images: None,
                    initial_messages: Vec::new(),
                    verbose: options.verbose.unwrap_or(false),
                    return_to_agents_view: true,
                    session_depth: options.session_depth.map(|depth| depth as f64),
                    session_has_children: options.session_has_children.unwrap_or(false),
                    connection: Some(connection),
                    runtime: None,
                },
            )
            .await?;
            let Some(result) = result else {
                return Ok(view::InteractiveRunResult {
                    kind: "exit".into(),
                    source,
                });
            };
            source.active_session_id = result.source.active_session_id;
            source.id = source
                .active_session_id
                .clone()
                .unwrap_or_else(|| result.source.session_id.clone());
            source.session_id = result.source.session_id;
            source.session_file = result.source.session_file;
            source.session_name = result.source.session_name;
            source.cwd = result.source.cwd;
            Ok(view::InteractiveRunResult {
                kind: result.type_.as_str().into(),
                source,
            })
        })
    }
    fn teardown_session_ui(
        &mut self,
        _preserve_alt_screen: bool,
    ) -> view::TransportFuture<Result<(), String>> {
        // The interactive host owns a terminal guard that restores state on all exits.
        let attached = self.attached.clone();
        Box::pin(async move {
            let connection = attached
                .lock()
                .expect("attached connection poisoned")
                .take();
            if let Some(connection) = connection {
                connection.dispose().await?;
            }
            Ok(())
        })
    }
}

fn editor_theme() -> pi_tui::components::editor::EditorTheme {
    let source = palette::get_editor_theme();
    pi_tui::components::editor::EditorTheme {
        border_color: Rc::new(move |text| (source.border_color)(text)),
        background_color: source
            .background_color
            .map(|color| Rc::new(move |text: &str| color(text)) as Rc<dyn Fn(&str) -> String>),
        autocomplete_background_color: Some(Rc::new(move |text| {
            (source.autocomplete_background_color)(text)
        })),
        command_color: Some(Rc::new(move |text| (source.command_color)(text))),
        select_list: pi_tui::components::select_list::SelectListTheme {
            selected_prefix: Box::new(|text| theme().fg("accent", text)),
            selected_text: Box::new(|text| theme().fg("accent", text)),
            description: Box::new(|text| theme().fg("muted", text)),
            argument_hint: None,
            source_tag: None,
            scroll_info: Box::new(|text| theme().fg("dim", text)),
            no_match: Box::new(|text| theme().fg("muted", text)),
        },
    }
}

pub(crate) async fn run_agents_view_mode(options: AgentsViewSeamOptions) -> Result<(), String> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(run_on_owner_thread(options)))
        .await
        .map_err(|error| format!("Agents terminal failed: {error}"))?
}

async fn run_on_owner_thread(options: AgentsViewSeamOptions) -> Result<(), String> {
    let cwd = options.config.cwd.clone().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let settings = SettingsManager::create(&cwd, Some(&options.agent_dir));
    crate::core::keybindings::KeybindingsManager::create(Some(&options.agent_dir)).install();
    palette::init_theme(settings.get_theme().as_deref(), true);
    let services = Arc::new(UiServices {
        cwd,
        theme: settings.get_theme().unwrap_or_else(|| palette::get_default_theme().into()),
        themes: palette::get_available_themes(),
        hardware_cursor: settings.get_show_hardware_cursor(),
        clear_on_shrink: settings.get_clear_on_shrink(),
        padding: settings.get_editor_padding_x() as usize,
        autocomplete_rows: settings.get_autocomplete_max_visible() as usize,
    });
    UI.with(|ui| *ui.borrow_mut() = Some(NativeUi::new(&settings)));
    let _guard = UiGuard;
    let attached = Arc::new(Mutex::new(None));
    let factory = ConnectionFactory {
        socket: options.socket_path.clone(),
        config: options.config.clone(),
        attached: attached.clone(),
    };
    let interactive: view::InteractiveModeFactory = Box::new(move |options| {
        Box::new(NativeInteractive {
            options,
            attached: attached.clone(),
        })
    });
    let socket = options.socket_path.clone();
    let recover = Arc::new(move || {
        let socket = socket.clone();
        Box::pin(async move {
            crate::cli::daemon_launch::ensure_interactive_daemon_running(&socket, None).await
        }) as view::TransportFuture<Result<(), String>>
    });
    view::run_agents_view_mode(
        view::AgentsViewModeOptions {
            socket_path: Some(options.socket_path.clone()),
            config: view::AgentsViewRuntimeConfig {
                cwd: options.config.cwd,
                session_dir: options.config.session_dir,
                telemetry_disabled: options.config.telemetry_disabled,
            },
            ui_services: services,
            migrated_providers: Some(options.migrated_providers),
            model_fallback_message: options.model_fallback_message,
            startup_model_id: options.startup_model_id,
            verbose: Some(options.verbose),
            reconnect_timeout_ms: None,
            initial_session: options
                .initial_session
                .map(|session| serde_json::to_value(session).and_then(|value| serde_json::from_value(normalize_browser_numbers(value))))
                .transpose()
                .map_err(|error| format!("Invalid initial agents session: {error}"))?,
            initial_scope_key: options.initial_scope_key,
        },
        Arc::new(NativeTerminal),
        Box::new(NativeEditor),
        Arc::new(NativeTheme),
        Arc::new(NativeTransport::new(&options.socket_path)),
        &factory,
        interactive,
        Some(recover),
        None,
    )
    .await
}


#[cfg(test)]
pub(super) fn capture_for_test(name: &str, mode: &mut view::AgentsViewMode<'_>, columns: usize, rows: usize) {
    use view::AgentsViewTerminal;
    struct CaptureTerminal { bytes: Rc<RefCell<String>>, columns: usize, rows: usize, alternate: bool }
    impl pi_tui::terminal::Terminal for CaptureTerminal {
        fn start(&mut self, _: Box<dyn Fn(String)>, _: Box<dyn Fn()>) {}
        fn stop(&mut self, _: pi_tui::terminal::TerminalStopOptions) {}
        fn drain_input(&mut self, _: u64, _: u64) {}
        fn write(&mut self, value: &str) { self.bytes.borrow_mut().push_str(value); }
        fn columns(&self) -> usize { self.columns }
        fn rows(&self) -> usize { self.rows }
        fn kitty_protocol_active(&self) -> bool { false }
        fn move_by(&mut self, _: i64) {}
        fn hide_cursor(&mut self) {}
        fn show_cursor(&mut self) {}
        fn clear_line(&mut self) {}
        fn clear_from_cursor(&mut self) {}
        fn clear_screen(&mut self) {}
        fn enter_alt_screen(&mut self) { self.alternate = true; }
        fn leave_alt_screen(&mut self) { self.alternate = false; }
        fn alt_screen_active(&self) -> bool { self.alternate }
        fn set_mouse_tracking(&mut self, _: bool) {}
        fn mouse_tracking_active(&self) -> bool { false }
        fn set_title(&mut self, _: &str) {}
        fn set_progress(&mut self, _: bool) {}
    }
    let bytes = Rc::new(RefCell::new(String::new()));
    let native = NativeUi::new(&SettingsManager::in_memory(Default::default()));
    native.tui.borrow_mut().terminal = Box::new(CaptureTerminal {
        bytes: bytes.clone(), columns, rows, alternate: false,
    });
    UI.with(|ui| *ui.borrow_mut() = Some(native));
    let _guard = UiGuard;
    NativeTerminal.present(mode.render_view(columns), mode.render_dock(columns)).unwrap();
    let frame = bytes.borrow().clone();
    assert!(frame.contains("\x1b[1;1H"), "actual TUI full-frame paint expected");
    if let Ok(directory) = std::env::var("OPTIMUS_UI_PROOF_DIR") {
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(std::path::Path::new(&directory).join(name), frame).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use view::AgentsViewEditor;

    #[test]
    fn editor_accepts_terminal_bytes_and_submits_the_edited_text() {
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        palette::init_theme(Some("prime"), false);
        UI.with(|ui| {
            *ui.borrow_mut() = Some(NativeUi::new(&SettingsManager::in_memory(
                Default::default(),
            )))
        });
        let _guard = UiGuard;
        let mut editor = NativeEditor;
        editor.handle_input("hello");
        editor.handle_input("\x1b[D");
        editor.handle_input("!");
        assert_eq!(editor.get_text(), "hell!o");
        assert!(editor
            .render(80)
            .iter()
            .any(|line| pi_tui::utils::strip_ansi(line).contains("hell!o")));
        editor.handle_input("\r");
        assert_eq!(editor.take_submissions(), vec!["hell!o"]);
        assert!(editor.take_submissions().is_empty());
        assert!(editor.get_text().is_empty());
    }

    #[test]
    fn roster_wire_push_preserves_changes_removals_and_resync() {
        let entry = roster::AgentRosterEntry {
            agent_id: "agent-1".into(),
            summary: super::super::agents_view_state::SessionSummary::new(
                "agent-1",
                "session-1",
                "/work",
            ),
            ..Default::default()
        };
        let event = outbound(&serde_json::json!({
            "type": "roster_update", "changed": [entry.clone()], "removed": ["old-agent"], "resync": true,
        }));
        assert_eq!(
            event,
            roster::DaemonOutbound::RosterUpdate {
                changed: vec![entry],
                removed: Some(vec!["old-agent".into()]),
                resync: Some(true),
            }
        );
        assert_eq!(
            outbound(&serde_json::json!({"type":"heartbeats_changed"})),
            roster::DaemonOutbound::HeartbeatsChanged
        );
    }
}
