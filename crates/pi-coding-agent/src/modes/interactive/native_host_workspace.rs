//! One native workspace for startup, empty state and attached conversations.
#[cfg(test)]
#[path = "native_host_workspace_tests.rs"]
mod tests;
use super::*;
use crate::core::agent_session_config::AgentSessionRuntimeConfig;
use crate::main_entry::{AgentsViewSeamOptions, CreateDaemonClientConnectionOptions};
use crate::modes::daemon::daemon_client::DaemonClient;
use crate::modes::daemon::daemon_session_list::SessionSummary;
use crate::modes::interactive::session_sidebar::{self as sidebar, Dialog, DialogKind, Pane, State};
use pi_tui::tui::Focusable;
use pi_tui::utils::wrap_text_with_ansi;
use serde_json::{json, Value};

pub(super) struct Runtime {
    pub state: Rc<RefCell<State>>,
    socket: Option<String>,
    config: AgentSessionRuntimeConfig,
    send: mpsc::Sender<Reply>,
    receive: mpsc::Receiver<Reply>,
    refresh: Option<tokio::task::JoinHandle<()>>,
    operation: Option<tokio::task::JoinHandle<()>>,
    refreshed: Instant,
    catalog_refreshed: Option<Instant>,
    dialog: Option<Rc<RefCell<Dialog>>>,
    location: Option<Rc<RefCell<LocationPane>>>,
    overlay: Option<pi_tui::tui::OverlayHandle>,
    pub next: Option<Arc<dyn wire::AgentConnection>>,
    folder_generation: u64,
    #[cfg(test)]
    clipboard: Option<mpsc::Sender<String>>,
}

enum Reply {
    Catalog(Result<Vec<SessionSummary>, String>, bool),
    Opened(Result<Arc<dyn wire::AgentConnection>, String>),
    FolderValidated(Result<String, String>, u64),
    Changed(Result<String, String>),
    Copied(Result<String, String>),
}

/// UI-009: read-only full-location overlay. The paths wrap instead of clipping,
/// and confirm copies the full repo path without truncation.
struct LocationPane {
    repo: Option<String>,
    session_file: Option<String>,
    status: Option<Result<String, String>>,
}

impl LocationPane {
    fn copy_text(&self) -> Option<String> {
        self.repo.as_ref().filter(|path| !path.trim().is_empty())
            .or(self.session_file.as_ref()).cloned()
    }
}

impl TuiComponent for LocationPane {
    fn render(&mut self, width: f64) -> Vec<String> {
        let inner = (width as usize).saturating_sub(4).max(1);
        let mut lines = vec![theme().fg("accent", " Full location"), String::new()];
        match (&self.repo, &self.session_file) {
            (Some(repo), Some(file)) => {
                lines.push(theme().fg("muted", "Project/repo path"));
                lines.extend(wrap_text_with_ansi(repo, inner));
                lines.push(String::new());
                lines.push(theme().fg("muted", "Saved chat file"));
                lines.extend(wrap_text_with_ansi(file, inner));
            }
            (Some(repo), None) => {
                lines.push(theme().fg("muted", "Project/repo path"));
                lines.extend(wrap_text_with_ansi(repo, inner));
            }
            (None, Some(file)) => {
                lines.push(theme().fg("muted", "Saved chat file"));
                lines.extend(wrap_text_with_ansi(file, inner));
            }
            (None, None) => lines.push(theme().fg("muted", "Nothing is selected.")),
        }
        if let Some(status) = &self.status {
            lines.push(String::new());
            let themed = match status {
                Ok(text) => theme().fg("success", text),
                Err(error) => theme().fg("error", error),
            };
            lines.extend(wrap_text_with_ansi(&themed, inner));
        }
        lines.push(String::new());
        lines.push(theme().fg("dim", &format!(
            "{} Copy path   {} Close",
            sidebar::key_label("tui.select.confirm"),
            sidebar::key_label("tui.select.cancel")
        )));
        let border = theme().fg("border", &"─".repeat(inner));
        let mut output = vec![format!("┌{border}┐")];
        output.extend(lines.into_iter().map(|line| format!("│{}│", truncate_to_width(&line, inner as f64, "…", true))));
        output.push(format!("└{border}┘"));
        output
    }
    fn invalidate(&mut self) {}
}

impl Runtime {
    pub fn new(socket: Option<String>, config: AgentSessionRuntimeConfig, agent_dir: &str) -> Self {
        let cwd = config.cwd.clone().unwrap_or_else(|| std::env::current_dir().unwrap_or_default().to_string_lossy().into_owned());
        let (send, receive) = mpsc::channel();
        Self { state: Rc::new(RefCell::new(State::new(cwd, std::path::Path::new(agent_dir).join("session-sidebar.json")))),
            socket, config, send, receive, refresh: None, operation: None, refreshed: Instant::now(),
            catalog_refreshed: None, dialog: None, location: None, overlay: None, next: None, folder_generation: 0,
            #[cfg(test)] clipboard: None }
    }

    pub fn focus(&mut self, editor: &Rc<RefCell<CustomEditor>>) {
        self.state.borrow_mut().focused = true;
        editor.borrow_mut().editor_mut().set_focused(false);
    }

    /// UI-006: the editor's Down-arrow path ends on the sub-agent summary line;
    /// confirming it hands the keyboard to the sessions sidebar with the
    /// selection aimed at a running child of the chat open in the main pane, so
    /// the next confirm opens that sub-agent chat. Aiming lives in the sidebar
    /// state (its row model owns linkage and activity facts).
    pub fn focus_subagents(&mut self, editor: &Rc<RefCell<CustomEditor>>, ui: &Rc<RefCell<TUI>>) {
        if self.state.borrow_mut().focus_current_running_child() {
            ui.borrow_mut().set_fullscreen_sidebar_hidden(false);
            self.focus(editor);
        } else {
            self.state.borrow_mut().focused = false;
            self.state.borrow_mut().status = "No running child of this chat is available.".into();
            editor.borrow_mut().editor_mut().set_focused(true);
        }
    }

    pub fn set_current(&mut self, state: &wire::AgentConnectionState) {
        let changed = self.state.borrow().active.as_ref() != Some(&state.session_id);
        self.state.borrow_mut().set_current(SessionSummary {
            id: state.active_session_id.clone().unwrap_or_else(|| state.session_id.clone()),
            session_id: state.session_id.clone(), active_session_id: state.active_session_id.clone(),
            session_name: state.session_name.clone(), session_file: state.session_file.clone(),
            cwd: state.cwd.clone(), is_session_active: true, is_streaming: state.is_streaming,
            is_compacting: state.is_compacting, ..Default::default()
        });
        // Never copy another chat's cwd into new-session configuration.
        if changed { self.request_refresh(true); }
    }

    fn request_refresh(&mut self, force: bool) {
        if self.refresh.is_some() || (!force && self.refreshed.elapsed() < Duration::from_secs(5)) { return; }
        let Some(socket) = self.socket.clone() else { return; };
        let full = force || self.catalog_refreshed.is_none_or(|at| at.elapsed() >= Duration::from_secs(60));
        let session_dir = self.config.session_dir.clone();
        let send = self.send.clone();
        self.refreshed = Instant::now();
        self.refresh = Some(tokio::spawn(async move {
            let mut command = json!({"type":"list", "all":full});
            if let Some(dir) = session_dir { command["sessionDir"] = dir.into(); }
            let result = request(&socket, command).await.and_then(session_list);
            let _ = send.send(Reply::Catalog(result, full));
        }));
    }

    pub fn poll(&mut self, ui: &Rc<RefCell<TUI>>) {
        let mut changed = false;
        while let Ok(reply) = self.receive.try_recv() {
            changed = true;
            match reply {
                Reply::Catalog(result, full) => {
                    self.refresh = None;
                    match result {
                        Ok(sessions) => {
                            self.state.borrow_mut().update(sessions, full);
                            if full { self.catalog_refreshed = Some(Instant::now()); }
                        }
                        Err(error) => self.state.borrow_mut().status = format!("Session list: {error}"),
                    }
                }
                Reply::Opened(result) => {
                    self.operation = None;
                    self.state.borrow_mut().busy = false;
                    match result {
                        Ok(connection) => { self.next = Some(connection); self.close_dialog(); }
                        Err(error) => self.report_error(error),
                    }
                }
                Reply::FolderValidated(result, generation) => {
                    if generation != self.folder_generation || self.dialog.is_none() { continue; }
                    self.operation = None;
                    self.state.borrow_mut().busy = false;
                    // Only the still-open dialog may commit its validated folder.
                    let store = self.state.borrow().store.clone();
                    let result = result.and_then(|path| sidebar::persist_folder(&store, path.clone()).map(|folders| (folders, path)));
                    match result {
                        Ok((folders, path)) => { self.state.borrow_mut().folders_added(folders, &path); self.close_dialog(); }
                        Err(error) => self.report_error(error),
                    }
                }
                Reply::Changed(result) => {
                    self.operation = None;
                    self.state.borrow_mut().busy = false;
                    match result {
                        Ok(status) => { self.state.borrow_mut().status = status; self.close_dialog(); self.request_refresh(true); }
                        Err(error) => self.report_error(error),
                    }
                }
                Reply::Copied(result) => {
                    // The copy runs off the input thread; its outcome lands here.
                    if let Some(location) = &self.location { location.borrow_mut().status = Some(result); }
                }
            }
        }
        self.request_refresh(false);
        if changed { ui.borrow_mut().request_render(); }
    }

    fn report_error(&mut self, error: String) {
        self.state.borrow_mut().status = error.clone();
        if let Some(dialog) = &self.dialog { let mut dialog = dialog.borrow_mut(); dialog.error = error; dialog.busy = false; }
    }

    fn close_dialog(&mut self) {
        self.folder_generation = self.folder_generation.wrapping_add(1);
        if let Some(handle) = self.overlay.take() { handle.hide(); }
        self.dialog = None;
        self.location = None;
    }

    fn show_dialog(&mut self, kind: DialogKind, ui: &Rc<RefCell<TUI>>) {
        self.close_dialog();
        let dialog = Rc::new(RefCell::new(Dialog::new(kind)));
        self.overlay = Some(ui.borrow_mut().show_overlay(dialog.clone(), pi_tui::tui::OverlayOptions {
            width: Some(pi_tui::tui::SizeValue::Number(72.0)), ..Default::default()
        }));
        self.dialog = Some(dialog);
    }

    /// UI-009: the full location wraps instead of clipping, whatever the pane
    /// width, and confirm copies the untruncated repo path.
    fn show_location(&mut self, ui: &Rc<RefCell<TUI>>) {
        self.close_dialog();
        let (repo, session_file) = {
            let state = self.state.borrow();
            (state.selected_cwd(), state.selected_session_file())
        };
        let pane = Rc::new(RefCell::new(LocationPane { repo, session_file, status: None }));
        self.overlay = Some(ui.borrow_mut().show_overlay(pane.clone(), pi_tui::tui::OverlayOptions {
            width: Some(pi_tui::tui::SizeValue::Number(72.0)), ..Default::default()
        }));
        self.location = Some(pane);
    }

    pub fn input(&mut self, data: &str, editor_at_start: bool, ui: &Rc<RefCell<TUI>>) -> bool {
        if pi_tui::keys::is_key_release(data) { return true; }
        let keys = pi_tui::keybindings::get_keybindings();
        // UI-010: transport-gated sidebar shortcuts. `is_unambiguous_ctrl_combo`
        // keeps the reserved editing bytes (raw Enter/Backspace/Tab/DEL) from
        // acting as toggles on any transport, so the defaults live only where
        // Ctrl+H/Ctrl+M arrive self-identified, and user remaps to other raw
        // combos stay live. The toggles also work while an overlay is open.
        if pi_tui::tui::is_unambiguous_ctrl_combo(data)
            && (keys.matches(data, "app.sidebar.toggleVisibility")
                || keys.matches(data, "app.sidebar.toggleSide"))
        {
            if keys.matches(data, "app.sidebar.toggleVisibility") {
                let hidden = ui.borrow_mut().toggle_fullscreen_sidebar_hidden();
                let mut state = self.state.borrow_mut();
                if hidden { state.focused = false; }
                state.status = if hidden {
                    format!("Sidebar hidden. {} shows it.", sidebar::key_label("app.sidebar.toggleVisibility"))
                } else {
                    "Sidebar shown.".to_string()
                };
            } else {
                let side = ui.borrow_mut().toggle_fullscreen_sidebar_side();
                let label = match side {
                    pi_tui::tui::FullscreenSidebarSide::Left => "left",
                    pi_tui::tui::FullscreenSidebarSide::Right => "right",
                };
                self.state.borrow_mut().status = format!("Sidebar moved to the {label}.");
            }
            return true;
        }
        if let Some(location) = self.location.clone() {
            if keys.matches(data, "tui.select.cancel") {
                self.close_dialog();
            } else if keys.matches(data, "tui.select.confirm") {
                let text = location.borrow().copy_text();
                if let Some(text) = text {
                    location.borrow_mut().status = Some(Ok("Copying full path…".into()));
                    let send = self.send.clone();
                    #[cfg(test)]
                    if let Some(clipboard) = &self.clipboard {
                        clipboard.send(text.clone()).unwrap();
                        let _ = send.send(Reply::Copied(Ok(format!("Copied. {text}"))));
                        return true;
                    }
                    tokio::spawn(async move {
                        let result = crate::utils::clipboard::copy_to_clipboard(&text).await
                            .map(|_| format!("Copied. {text}"))
                            .map_err(|_| "Copy failed in this terminal. Read the full path above and copy it manually.".to_string());
                        let _ = send.send(Reply::Copied(result));
                    });
                } else {
                    location.borrow_mut().status = Some(Err("Nothing is selected to copy.".into()));
                }
            }
            return true;
        }
        if let Some(dialog) = self.dialog.clone() {
            if keys.matches(data, "tui.select.cancel") {
                if matches!(dialog.borrow().kind, DialogKind::AddFolder) {
                    if let Some(task) = self.operation.take() { task.abort(); }
                    self.state.borrow_mut().busy = false;
                    self.close_dialog();
                } else if !dialog.borrow().busy { self.close_dialog(); }
            } else if dialog.borrow().busy { return true; }
            else if keys.matches(data, "tui.select.confirm") {
                let kind = dialog.borrow().kind.clone();
                let value = dialog.borrow().input.get_value().to_string();
                self.confirm_dialog(kind, value);
            } else { dialog.borrow_mut().input.handle_input(data); }
            return true;
        }
        if let Some(mouse) = pi_tui::mouse::parse_sgr_mouse_event(data) {
            // `fullscreen_sidebar_hit` is edge- and hide-aware: a hidden or
            // unmounted pane is never a hit, and the right edge only claims its
            // own columns (UI-010).
            if ui.borrow().fullscreen_sidebar_hit(mouse.x, mouse.y) {
                let header_height = ui.borrow().fullscreen_header_height();
                let mut state = self.state.borrow_mut();
                state.focused = true;
                if pi_tui::mouse::is_wheel_up(&mouse) {
                    state.move_by(-3);
                } else if pi_tui::mouse::is_wheel_down(&mouse) {
                    state.move_by(3);
                } else if mouse.press && !mouse.motion && mouse.button == pi_tui::mouse::MOUSE_BUTTON_LEFT {
                    let row = (mouse.y - header_height as i64 - 1) as usize;
                    let pane_height = ui.borrow().terminal.rows().saturating_sub(header_height);
                    if row == state.location_footer_row() || (pane_height >= 12 && row + 1 == pane_height) {
                        drop(state);
                        self.show_location(ui);
                    } else if row < state.location_footer_row() {
                        state.pointer(row);
                    }
                }
                return true;
            }
            return false;
        }
        // A hidden pane must not capture keys or trap focus (UI-010).
        if ui.borrow().fullscreen_sidebar_hidden() {
            self.state.borrow_mut().focused = false;
            return false;
        }
        if !self.state.borrow().focused {
            if editor_at_start && keys.matches(data, "app.sidebar.focus") {
                self.state.borrow_mut().focused = true;
                return true;
            }
            return false;
        }
        if keys.matches(data, "app.sidebar.chat") || keys.matches(data, "tui.select.cancel") {
            self.state.borrow_mut().focused = false;
        } else if keys.matches(data, "tui.select.up") { self.state.borrow_mut().move_by(-1); }
        else if keys.matches(data, "tui.select.down") { self.state.borrow_mut().move_by(1); }
        else if keys.matches(data, "tui.select.pageUp") { self.state.borrow_mut().move_by(-8); }
        else if keys.matches(data, "tui.select.pageDown") { self.state.borrow_mut().move_by(8); }
        else if keys.matches(data, "app.sidebar.addFolder") { self.show_dialog(DialogKind::AddFolder, ui); }
        else if keys.matches(data, "app.sidebar.location") { self.show_location(ui); }
        else if keys.matches(data, "app.agents.rename") {
            let session = self.state.borrow().selected_session();
            if let Some(session) = session { self.show_dialog(DialogKind::Rename(session), ui); }
        } else if keys.matches(data, "app.agents.delete") {
            let session = self.state.borrow().selected_session();
            if let Some(session) = session { self.show_dialog(DialogKind::Delete(session), ui); }
        } else if keys.matches(data, "app.agents.new") {
            let cwd = self.state.borrow().selected_cwd();
            if let Some(cwd) = cwd { self.open(None, Some(cwd)); }
        } else if keys.matches(data, "tui.select.confirm") {
            let session = self.state.borrow_mut().confirm();
            if let Some(session) = session {
                if self.state.borrow().active.as_ref() == Some(&sidebar::identity(&session)) {
                    self.state.borrow_mut().focused = false;
                } else { self.open(Some(session), None); }
            }
        } else if keys.matches(data, "app.exit") || keys.matches(data, "app.clear") { return false; }
        true
    }

    pub fn open(&mut self, session: Option<SessionSummary>, cwd: Option<String>) {
        if self.operation.is_some() { return; }
        let Some(socket) = self.socket.clone() else { self.report_error("Session navigation requires a daemon connection.".into()); return; };
        let mut config = self.config.clone();
        config.cwd = cwd;
        let send = self.send.clone();
        self.state.borrow_mut().busy = true;
        self.state.borrow_mut().status = "Opening chat…".into();
        self.operation = Some(tokio::spawn(async move {
            let result = open_connection(socket, config, session).await;
            let _ = send.send(Reply::Opened(result));
        }));
    }

    fn confirm_dialog(&mut self, kind: DialogKind, value: String) {
        if self.operation.is_some() { return; }
        if let DialogKind::Rename(_) = &kind {
            if value.trim().is_empty() { self.report_error("Enter a chat name.".into()); return; }
        }
        let send = self.send.clone();
        if let Some(dialog) = &self.dialog { dialog.borrow_mut().busy = true; }
        self.state.borrow_mut().busy = true;
        if matches!(kind, DialogKind::AddFolder) {
            let generation = self.folder_generation;
            self.operation = Some(tokio::spawn(async move {
                let result = tokio::task::spawn_blocking(move || sidebar::validate_folder(&value))
                    .await.unwrap_or_else(|error| Err(error.to_string()));
                let _ = send.send(Reply::FolderValidated(result, generation));
            }));
        } else {
            let Some(socket) = self.socket.clone() else {
                self.state.borrow_mut().busy = false;
                self.report_error("This action requires a daemon connection.".into()); return;
            };
            self.operation = Some(tokio::spawn(async move {
                let result = change_session(&socket, kind, &value).await;
                let _ = send.send(Reply::Changed(result));
            }));
        }
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if let Some(task) = self.refresh.take() { task.abort(); }
        // Explicit mutations are not retried or cancelled after commit. All RPCs have deadlines.
        self.close_dialog();
    }
}

async fn request(socket: &str, command: Value) -> Result<Value, String> {
    let client = DaemonClient::create(socket);
    let result = async {
        client.connect(3000).await.map_err(|e| e.message())?;
        let response = client.request(command.as_object().cloned().ok_or("Invalid sidebar command")?,
            Some(30_000), Default::default()).await.map_err(|e| e.message())?;
        if !response.success { return Err(response.error.unwrap_or_else(|| "Session command failed".into())); }
        Ok(response.data.unwrap_or(Value::Null))
    }.await;
    client.close().await;
    result
}

fn session_list(data: Value) -> Result<Vec<SessionSummary>, String> {
    let sessions = data.get("sessions").and_then(Value::as_array).ok_or("Invalid session list")?;
    sessions.iter().map(|session| serde_json::from_value(
        crate::modes::agents_view::native_wire::normalize_browser_numbers(session.clone())
    ).map_err(|_| "Invalid session summary".into())).collect()
}

async fn open_connection(socket: String, config: AgentSessionRuntimeConfig, session: Option<SessionSummary>) -> Result<Arc<dyn wire::AgentConnection>, String> {
    if let Some(cwd) = &config.cwd { sidebar::validate_folder(cwd)?; }
    if let Some(session) = &session {
        if session.active_session_id.is_none() && session.session_file.is_none() { return Err("No saved chat or active session is available.".into()); }
        if session.active_session_id.is_none() { sidebar::validate_folder(&session.cwd)?; }
    }
    let options = CreateDaemonClientConnectionOptions {
        socket_path: socket, config,
        session_path: session.as_ref().and_then(|s| s.session_file.clone()),
        active_session_id: session.as_ref().and_then(|s| s.active_session_id.clone()),
        continue_recent: None, client_owned: Some(false), no_session: None,
        supports_extension_ui: Some(true), defer_session_events: true,
    };
    let fallback = options.session_path.clone();
    match crate::main_entry::create_daemon_client_connection(options.clone()).await {
        Ok((connection, _)) => Ok(connection),
        Err(error) if options.active_session_id.is_some() && fallback.is_some()
            && (error.to_string().contains("Unknown active session") || error.to_string().starts_with("Session is recovering")) => {
            let mut options = options; options.active_session_id = None;
            crate::main_entry::create_daemon_client_connection(options).await
                .map(|(connection, _)| connection as Arc<dyn wire::AgentConnection>).map_err(|e| e.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

async fn change_session(socket: &str, kind: DialogKind, value: &str) -> Result<String, String> {
    match kind {
        DialogKind::Rename(session) => {
            let command = if let Some(active) = session.active_session_id {
                json!({"type":"rename", "activeSessionId":active, "name":value.trim()})
            } else { json!({"type":"rename_saved_session", "sessionPath":session.session_file.ok_or("No saved session")?, "name":value.trim()}) };
            request(socket, command).await?;
            Ok("Chat renamed".into())
        }
        DialogKind::Delete(session) => {
            if let Some(active) = session.active_session_id {
                request(socket, json!({"type":"kill", "activeSessionId":active})).await?;
                return Ok("Session stopped. History kept; Delete again only when inactive.".into());
            }
            let file = session.session_file.ok_or("No saved session to delete")?;
            let latest = session_list(request(socket, json!({"type":"list"})).await?)?;
            if latest.iter().any(|s| s.session_file.as_deref().is_some_and(|p| sidebar::path_key(p) == sidebar::path_key(&file)) && s.active_session_id.is_some()) {
                return Err("Session became active. Stop it before deleting.".into());
            }
            let result = request(socket, json!({"type":"delete_saved_session", "sessionPath":file})).await?;
            if result.get("ok").and_then(Value::as_bool) != Some(true) {
                return Err(result.get("error").and_then(Value::as_str).unwrap_or("Delete was not confirmed").into());
            }
            Ok("Saved chat deleted. Repository files unchanged.".into())
        }
        DialogKind::AddFolder => unreachable!(),
    }
}

pub(super) async fn attached_loop(mut options: InteractiveModeSeamOptions, benchmark: bool, runtime: &mut Runtime) -> Result<Option<InteractiveModeRunResult>, String> {
    loop {
        let result = run_terminal(options.clone(), benchmark, runtime).await?;
        let Some(connection) = runtime.next.take() else { return Ok(result); };
        options.connection = Some(connection);
        options.runtime = None;
        options.initial_message = None;
        options.initial_images = None;
        options.initial_messages.clear();
        options.return_to_agents_view = false;
        options.session_depth = None;
        options.session_has_children = false;
    }
}

pub(crate) async fn startup(options: AgentsViewSeamOptions) -> Result<(), String> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(async move {
        let mut workspace = Runtime::new(Some(options.socket_path.clone()), options.config.clone(), &options.agent_dir);
        let cwd = options.config.cwd.clone().unwrap_or_default();
        let settings = crate::core::settings_manager::SettingsManager::create(&cwd, Some(&options.agent_dir));
        crate::core::keybindings::KeybindingsManager::create(Some(&options.agent_dir)).install();
        crate::modes::interactive::theme::theme::init_theme(settings.get_theme().as_deref(), false);
        if let Some(session) = options.initial_session { workspace.open(Some(session), None); }
        let Some(connection) = empty(&mut workspace).await? else { return Ok(()); };
        attached_loop(InteractiveModeSeamOptions {
            daemon_socket_path: Some(options.socket_path), migrated_providers: options.migrated_providers,
            model_fallback_message: options.model_fallback_message, initial_message: None, initial_images: None,
            initial_messages: Vec::new(), verbose: options.verbose, return_to_agents_view: false,
            session_depth: None, session_has_children: false, connection: Some(connection), runtime: None,
        }, false, &mut workspace).await.map(|_| ())
    })).await.map_err(|e| format!("Workspace terminal failed: {e}"))?
}

struct EmptyChat;
impl TuiComponent for EmptyChat {
    fn render(&mut self, _width: f64) -> Vec<String> {
        vec![String::new(), theme().fg("muted", "No chat selected."),
            theme().fg("dim", "Select a chat and press Enter, or create one in the highlighted folder.")]
    }
    fn invalidate(&mut self) {}
}
struct EmptyHeader(String);
impl TuiComponent for EmptyHeader {
    fn render(&mut self, width: f64) -> Vec<String> { self.render_with_height(width, 7) }
    fn render_with_height(&mut self, width: f64, height: usize) -> Vec<String> {
        native_neon::render_header(width as usize, height, &native_neon::HeaderData {
            cwd: &self.0, session: "No chat selected", model: "No chat selected", phase: "READY", jev: None, clock: "",
        })
    }
    fn invalidate(&mut self) {}
}

async fn empty(workspace: &mut Runtime) -> Result<Option<Arc<dyn wire::AgentConnection>>, String> {
    let ui = Rc::new(RefCell::new(TUI::new(Box::new(pi_tui::terminal::ProcessTerminal::new()), Some(true))));
    let input = Rc::new(RefCell::new(Vec::<String>::new()));
    let queued = input.clone();
    ui.borrow_mut().add_input_listener(Box::new(move |data| {
        if !pi_tui::keys::is_key_release(data) { queued.borrow_mut().push(data.to_string()); }
        InputListenerResult { consume: true, data: None }
    }));
    ui.borrow_mut().start();
    let _guard = TerminalGuard(ui.clone());
    ui.borrow_mut().enter_fullscreen(pi_tui::tui::FullscreenOptions {
        scroll: vec![Rc::new(RefCell::new(EmptyChat))],
        dock: Rc::new(RefCell::new(TuiText::new("Choose a chat from Sessions".into(), 1, 0, None))),
        mouse: true, viewport_controls: false,
    });
    ui.borrow_mut().set_fullscreen_sidebar(Some(Rc::new(RefCell::new(Pane(workspace.state.clone())))));
    ui.borrow_mut().set_fullscreen_header(Some(Rc::new(RefCell::new(EmptyHeader(workspace.state.borrow().selected_cwd().unwrap_or_default())))));
    workspace.request_refresh(true);
    loop {
        if !ui.borrow_mut().terminal.poll_input().map_err(|e| e.to_string())? { return Ok(None); }
        ui.borrow_mut().drain_input();
        for data in std::mem::take(&mut *input.borrow_mut()) {
            if !workspace.input(&data, true, &ui) {
                let keys = pi_tui::keybindings::get_keybindings();
                if keys.matches(&data, "app.exit") || keys.matches(&data, "app.clear") { return Ok(None); }
            }
            ui.borrow_mut().request_render();
        }
        workspace.poll(&ui);
        if let Some(connection) = workspace.next.take() { return Ok(Some(connection)); }
        ui.borrow_mut().run_pending_render(now_ms());
        tokio::time::sleep(Duration::from_millis(16)).await;
    }
}
