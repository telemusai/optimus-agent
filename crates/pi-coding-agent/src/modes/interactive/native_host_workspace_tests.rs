use super::*;
use pi_ai::types::{AssistantMessage, ContentBlock, Message, TextContent, ThinkingContent, UserContent, UserMessage};
use pi_tui::utils::{strip_ansi, visible_width};
use crate::modes::interactive::theme::theme::{get_markdown_theme, init_theme};

fn init() {
    crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
    init_theme(Some("neon"), false);
}
fn summary(id: &str, cwd: &str) -> SessionSummary {
    SessionSummary { id: id.into(), session_id: id.into(), cwd: cwd.into(),
        session_name: Some(format!("Chat {id}")), modified: Some("2026-09-24T00:00:00Z".into()), ..Default::default() }
}
fn runtime(temp: &tempfile::TempDir) -> Runtime {
    Runtime::new(None, AgentSessionRuntimeConfig { cwd: Some(temp.path().to_string_lossy().into_owned()), ..Default::default() }, &temp.path().to_string_lossy())
}

#[test]
fn ui_repair_workspace_focus_browse_hover_and_shortcuts_preserve_chat_and_cursor() {
    init();
    let temp = tempfile::tempdir().unwrap();
    let mut runtime = runtime(&temp);
    let h = super::super::ui_tests::FrameHarness::new("active");
    h.editor.borrow_mut().editor_mut().set_text("keep this draft");
    h.editor.borrow_mut().handle_input("\x1b[D");
    let cursor = h.editor.borrow().editor().get_cursor();
    let cwd = temp.path().to_string_lossy().to_string();
    runtime.state.borrow_mut().update(vec![summary("active", &cwd), summary("other", "C:/other-folder")], true);
    runtime.state.borrow_mut().active = Some("active".into());
    runtime.state.borrow_mut().focused = false;
    assert!(!runtime.input("\x1b[D", false, &h.ui), "Left inside a prompt must keep editing");
    assert!(!runtime.input("\x01", false, &h.ui), "Ctrl+A outside sidebar belongs to the editor");
    assert!(runtime.input("\x1b[D", true, &h.ui));
    for key in ["\x1b[B", "\x1b[A", "\x1b[B"] { assert!(runtime.input(key, true, &h.ui)); }
    runtime.state.borrow_mut().pointer(1);
    assert_eq!(runtime.state.borrow().active.as_deref(), Some("active"));
    runtime.state.borrow_mut().set_current(summary("active", &cwd));
    assert!(runtime.state.borrow().focused, "same-session refresh must not steal sidebar focus");
    assert!(runtime.next.is_none() && runtime.operation.is_none());
    assert_eq!(h.editor.borrow().editor().get_text(), "keep this draft");
    assert_eq!(h.editor.borrow().editor().get_cursor(), cursor);
    assert!(runtime.input("\x1b[C", true, &h.ui));
    assert!(!runtime.state.borrow().focused);
    h.editor.borrow_mut().handle_input("\x01");
    assert_eq!(h.editor.borrow().editor().get_cursor(), (0, 0), "normal editor Ctrl+A remains line-start");
    assert_eq!(h.editor.borrow().editor().get_text(), "keep this draft");

    runtime.state.borrow_mut().focused = true;
    assert!(runtime.input("\x01", true, &h.ui));
    let dialog = runtime.dialog.as_ref().unwrap().clone();
    dialog.borrow_mut().input.handle_input(r"C:\Windows");
    assert!(strip_ansi(&dialog.borrow_mut().render(72.0).join("\n")).contains("Folder/repo path"));
    runtime.input("\x1b", true, &h.ui);
    assert!(runtime.dialog.is_none());
    assert!(!runtime.state.borrow().store.exists());
    runtime.state.borrow_mut().selected = Some(sidebar::Item::Session("other".into()));
    runtime.input("\x12", true, &h.ui);
    assert!(matches!(runtime.dialog.as_ref().unwrap().borrow().kind, DialogKind::Rename(_)));
    runtime.input("\x1b", true, &h.ui);
    runtime.input("\x18", true, &h.ui);
    assert!(matches!(runtime.dialog.as_ref().unwrap().borrow().kind, DialogKind::Delete(_)));
    assert!(runtime.operation.is_none(), "Ctrl+X must only open confirmation");
    runtime.input("\x1b", true, &h.ui);
    assert_eq!(runtime.state.borrow().active.as_deref(), Some("active"));
    // New resolves selected session's saved cwd, not current process or active chat.
    assert_eq!(runtime.state.borrow().selected_cwd().as_deref(), Some("C:/other-folder"));
}

#[tokio::test]
async fn ui_repair_add_folder_invalid_retry_persist_and_cancel_after_enter() {
    init();
    let temp = tempfile::tempdir().unwrap();
    let folder = temp.path().join("folder"); std::fs::create_dir(&folder).unwrap();
    let mut runtime = runtime(&temp);
    let h = super::super::ui_tests::FrameHarness::new("folder-test");
    runtime.input("\x01", true, &h.ui);
    runtime.dialog.as_ref().unwrap().borrow_mut().input.handle_input("not an absolute path");
    runtime.input("\r", true, &h.ui);
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.operation.is_some() { tokio::task::yield_now().await; runtime.poll(&h.ui); }
    }).await.unwrap();
    let dialog = runtime.dialog.as_ref().unwrap().clone();
    assert!(!dialog.borrow().error.is_empty());
    assert_eq!(dialog.borrow().input.get_value(), "not an absolute path");
    dialog.borrow_mut().input.set_value(folder.to_string_lossy().into_owned());
    runtime.input("\r", true, &h.ui);
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.operation.is_some() { tokio::task::yield_now().await; runtime.poll(&h.ui); }
    }).await.unwrap();
    assert!(runtime.dialog.is_none());
    assert!(runtime.state.borrow().store.exists());
    assert!(runtime.next.is_none());
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 0);
    let before = std::fs::read(&runtime.state.borrow().store).unwrap();
    runtime.input("\x01", true, &h.ui);
    runtime.dialog.as_ref().unwrap().borrow_mut().input.handle_input(&temp.path().to_string_lossy());
    let generation = runtime.folder_generation;
    runtime.input("\r", true, &h.ui);
    runtime.input("\x1b", true, &h.ui);
    // Even a racing validation reply cannot commit a cancelled dialog.
    runtime.send.send(Reply::FolderValidated(Ok(temp.path().to_string_lossy().into_owned()), generation)).unwrap();
    runtime.poll(&h.ui);
    assert!(runtime.dialog.is_none());
    assert_eq!(std::fs::read(&runtime.state.borrow().store).unwrap(), before);
}

#[test]
fn ui_repair_prompt_surface_and_only_assistant_prose_changes_colour() {
    init();
    let green = if theme().color_mode() == pi_tui::terminal_colors::TerminalColorMode::Truecolor { "\x1b[38;2;19;161;14m" } else { "\x1b[38;5;34m" };
    let grey = if theme().color_mode() == pi_tui::terminal_colors::TerminalColorMode::Truecolor { "\x1b[48;2;28;28;28m" } else { "\x1b[48;5;234m" };
    for width in [20, 40, 100] {
        let mut user = UserMessageComponent::new("A submitted prompt that wraps.\nSecond paragraph with `code` and **emphasis**.", get_markdown_theme(), &|_| false);
        let lines = user.render(width as f64);
        assert!(lines.len() >= 4);
        assert!(lines.iter().all(|line| line.contains(grey) && visible_width(line) == width));
        assert!(lines.iter().any(|line| line.contains(&theme().get_fg_ansi("userMessageText"))));
        assert!(!lines.join("\n").contains(green));
    }
    let mut assistant = AssistantMessageComponent::new(Some(AssistantMessage {
        content: vec![ContentBlock::Thinking(ThinkingContent::new("Thinking fixture")),
            ContentBlock::Text(TextContent::new("Normal AI prose.\n\n`inline_code`\n\n```python\nprint('syntax')\n```"))],
        error_message: Some("Fixture error".into()), stop_reason: "error".into(), ..Default::default()
    }), false, get_markdown_theme(), "Thinking…", Default::default());
    let lines = assistant.render(100.0);
    assert!(lines.iter().find(|l| strip_ansi(l).contains("Normal AI prose.")).unwrap().contains(green));
    for text in ["Thinking fixture", "inline_code", "print(", "Fixture error"] {
        let line = lines.iter().find(|line| strip_ansi(line).contains(text)).unwrap();
        // A prefix can restore prose after semantic segments; actual semantic colour must remain present.
        if text == "Thinking fixture" { assert!(line.contains(&theme().get_fg_ansi("thinkingText"))); }
        if text == "inline_code" { assert!(line.contains(&theme().get_fg_ansi("mdCode"))); }
        if text == "Fixture error" { assert!(line.contains(&theme().get_fg_ansi("error"))); }
        if text == "print(" { assert!(!line.contains(green), "code block was recoloured"); }
    }
    let notice = crate::modes::interactive::components::agent_message::agent_message_summary_line("Agent message received", "child fixture", Some("unchanged preview"));
    assert!(!notice.contains(green));
    assert!(notice.contains(&theme().get_fg_ansi("muted")));
}

struct ProofTerminal { output: Rc<RefCell<String>>, width: usize, height: usize }
impl pi_tui::terminal::Terminal for ProofTerminal {
    fn start(&mut self, _: Box<dyn Fn(String)>, _: Box<dyn Fn()>) {}
    fn stop(&mut self, _: pi_tui::terminal::TerminalStopOptions) {}
    fn drain_input(&mut self, _: u64, _: u64) {}
    fn write(&mut self, data: &str) { self.output.borrow_mut().push_str(data); }
    fn columns(&self) -> usize { self.width }
    fn rows(&self) -> usize { self.height }
    fn kitty_protocol_active(&self) -> bool { false }
    fn move_by(&mut self, _: i64) {}
    fn hide_cursor(&mut self) {}
    fn show_cursor(&mut self) {}
    fn clear_line(&mut self) {}
    fn clear_from_cursor(&mut self) {}
    fn clear_screen(&mut self) {}
    fn enter_alt_screen(&mut self) {}
    fn leave_alt_screen(&mut self) {}
    fn alt_screen_active(&self) -> bool { true }
    fn set_mouse_tracking(&mut self, _: bool) {}
    fn mouse_tracking_active(&self) -> bool { true }
    fn set_title(&mut self, _: &str) {}
    fn set_progress(&mut self, _: bool) {}
}

fn proof(name: &str, data: &str) {
    if let Ok(dir) = std::env::var("OPTIMUS_UI_PROOF_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(name), data).unwrap();
    }
}

#[test]
fn ui_repair_combined_sidebar_graphics_and_statsdock_render_proof() {
    use base64::Engine;
    use pi_tui::terminal_image::{CellDimensions, ImageProtocol, TerminalCapabilities, is_image_line, reset_capabilities_cache, set_capabilities, set_cell_dimensions};
    init();
    let temp = tempfile::tempdir().unwrap();
    let workspace = runtime(&temp);
    workspace.state.borrow_mut().update(vec![summary("Current task", "C:/work/alpha"), summary("Other project", "D:/work/beta")], true);
    workspace.state.borrow_mut().active = Some("Current task".into());
    let output = Rc::new(RefCell::new(String::new()));
    let ui = Rc::new(RefCell::new(TUI::new(Box::new(ProofTerminal { output: output.clone(), width: 160, height: 42 }), Some(true))));
    let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("Current task")));
    mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "Current task".into(), session_name: Some("Current task".into()), cwd: "C:/work/alpha".into(),
        context_usage: local::ContextUsage { tokens: Some(14_000.0), context_window: 100_000.0, percent: Some(14.0) },
        ..Default::default()
    });
    let editor = Rc::new(RefCell::new(CustomEditor::new(ui.clone(), editor_theme(), CustomEditorOptions::default())));
    editor.borrow_mut().editor_mut().set_text("Draft survives next to a managed image and the stats dock.");
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().sidebar = Some(workspace.state.clone());
    transcript.borrow_mut().message(AgentMessage::Message(Message::User(UserMessage::new(UserContent::Text("Attach the fixture image.".into()), 1))), false);
    transcript.borrow_mut().tool_start("img-fixture", "ipython", json!({"code":"print(await attach_image('quest.png'))"}));
    let mut jpeg = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(1200, 642, image::Rgb([0, 244, 119])).write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
    transcript.borrow_mut().tool_result("img-fixture", &json!({
        "content": [{"type":"image", "mimeType":"image/jpeg",
        "data": base64::engine::general_purpose::STANDARD.encode(jpeg.into_inner())}]
    }), false, false);
    transcript.borrow().tools.get("img-fixture").unwrap().borrow_mut().set_expanded(true);
    set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
    set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
    transcript.borrow().stats_panel.show(Rc::new(RefCell::new(TuiText::new("Stats fixture: 2,900 tokens (astra)".into(), 1, 0, None))));
    ui.borrow_mut().set_focus(Some(editor.clone()));
    ui.borrow_mut().start();
    native_settings::fullscreen(true, &mode, &editor, &ui, &transcript);
    ui.borrow_mut().do_render();
    let frame = output.borrow().clone();
    let plain = strip_ansi(&frame);
    for expected in ["Sessions", "Current task", "alpha", "beta", "Stats fixture: 2,900 tokens (astra)", "Draft survives next to"] {
        assert!(plain.contains(expected), "missing {expected} in combined frame");
    }
    // The ProofTerminal concatenates the frame with explicit row cursor moves
    // (`\x1b[R;CH`), not newlines, so rows are extracted by cursor position.
    fn cursor_moves(frame: &str) -> Vec<(usize, usize)> {
        let b = frame.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while let Some(rel) = frame[i..].find("\x1b[") {
            let s = i + rel;
            let mut j = s + 2;
            let mut first = 0;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; first += 1; }
            if first == 0 || j >= b.len() || b[j] != b';' { i = s + 2; continue; }
            j += 1;
            let mut second = 0;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; second += 1; }
            if second == 0 || j >= b.len() || b[j] != b'H' { i = s + 2; continue; }
            out.push((s, j + 1));
            i = j + 1;
        }
        out
    }
    // Persist the frame before assertions so the ANSI capture survives any failure.
    proof("combined-sidebar-image-statsdock.ansi", &frame);
    let moves = cursor_moves(&frame);
    let payload_at = frame.find("\x1bP").expect("sixel payload rendered");
    let gfx_start = moves.iter().rev().find(|&(s, _)| *s < payload_at).map(|&(s, _)| s).expect("cursor move before the graphics row");
    let gfx_end = moves.iter().find(|&(s, _)| *s > payload_at).map(|&(s, _)| s).unwrap_or(frame.len());
    let graphics_row = &frame[gfx_start..gfx_end];
    assert!(is_image_line(graphics_row), "graphics row carries the managed image");
    let gfx_payload_at = graphics_row.find("\x1bP").unwrap();
    let gfx_label_at = ["Ctrl+N New", "Ctrl+A Add folder", "Current task", "alpha", "beta", "Sessions"]
        .iter().find_map(|label| graphics_row.find(label));
    assert!(gfx_label_at.is_some(), "sidebar label must survive on the graphics row: {:?}...", &graphics_row[..graphics_row.len().min(160)]);
    assert!(gfx_label_at.unwrap() < gfx_payload_at, "image must start beyond the sidebar columns");
    let stats_at = frame.find("Stats fixture").expect("stats dock row content");
    let stats_start = moves.iter().rev().find(|&(s, _)| *s < stats_at).map(|&(s, _)| s).expect("cursor move before the stats dock row");
    let stats_end = moves.iter().find(|&(s, _)| *s > stats_at).map(|&(s, _)| s).unwrap_or(frame.len());
    let stats_row = &frame[stats_start..stats_end];
    let stats_visible = visible_width(&strip_ansi(stats_row));
    assert!(stats_visible <= 160, "stats dock row stays within the terminal width (chat-width mount), got {stats_visible}");
    let row_stats_at = stats_row.find("Stats fixture").unwrap();
    let prefix_visible = visible_width(&strip_ansi(&stats_row[..row_stats_at]));
    let sidebar_width = ui.borrow().fullscreen_sidebar_width();
    assert!(prefix_visible >= sidebar_width, "stats dock content starts after the sidebar region, prefix {prefix_visible} vs sidebar {sidebar_width}");
    assert!(stats_visible - prefix_visible <= 160 - sidebar_width, "dock content fits the chat width");
    assert!(frame.contains("\x1b[42;"), "editor cursor row includes the sidebar offset");
    let editor_at = frame.find("Draft survives").expect("editor text rendered");
    assert!(payload_at < stats_at && stats_at < editor_at, "graphics row above the stats dock, stats dock above the editor");
    transcript.borrow().stats_panel.clear();
    ui.borrow_mut().do_render();
    let output_str = output.borrow().clone();
    let latest = output_str.rsplit_once("\x1b[2J").map(|(_, tail)| tail.to_string()).unwrap_or(output_str);
    assert!(!strip_ansi(&latest).contains("Stats fixture"), "cleared dock leaves the latest frame");
    reset_capabilities_cache();
}

#[test]
fn ui_repair_offline_native_render_proof_active_empty_and_dialog() {
    use crate::core::messages::{create_async_bash_completion_message, AsyncBashCompletionDetails};
    init();
    let temp = tempfile::tempdir().unwrap();
    let mut workspace = runtime(&temp);
    workspace.state.borrow_mut().update(vec![summary("Current task", "C:/work/alpha"), summary("Other project", "D:/work/beta")], true);
    workspace.state.borrow_mut().active = Some("Current task".into());
    workspace.state.borrow_mut().selected = Some(sidebar::Item::Session("Other project".into()));
    let output = Rc::new(RefCell::new(String::new()));
    let ui = Rc::new(RefCell::new(TUI::new(Box::new(ProofTerminal { output: output.clone(), width: 160, height: 42 }), Some(true))));
    let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("Current task")));
    mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "Current task".into(), session_name: Some("Current task".into()), cwd: "C:/work/alpha".into(),
        context_usage: local::ContextUsage { tokens: Some(14_000.0), context_window: 100_000.0, percent: Some(14.0) },
        heartbeat: Some(local::AgentCronJob { source: Some("heartbeat".into()), status: "active".into(), session_id: "Current task".into(), ..Default::default() }),
        ..Default::default()
    });
    let editor = Rc::new(RefCell::new(CustomEditor::new(ui.clone(), editor_theme(), CustomEditorOptions::default())));
    editor.borrow_mut().editor_mut().set_text("Draft stays in the current chat while browsing another folder.");
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().sidebar = Some(workspace.state.clone());
    transcript.borrow_mut().message(AgentMessage::Message(Message::User(UserMessage::new(UserContent::Text(
        "Please review this project. Keep the working directory and saved history intact.\nThis submitted prompt has a padded faded-grey background.".into()), 1))), false);
    transcript.borrow_mut().tool_start("fixture", "ipython", json!({"code":"print('offline fixture')"}));
    transcript.borrow_mut().tool_result("fixture", &json!({"content":[{"type":"text","text":"offline fixture"}]}), false, false);
    transcript.borrow_mut().message(AgentMessage::Custom(create_async_bash_completion_message(AsyncBashCompletionDetails { pid: 33504, command: "hidden/full/command".into(), exit_code: 0 }, 1)), false);
    transcript.borrow_mut().message(AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::Thinking(ThinkingContent::new("**Reviewing the fixture**")), ContentBlock::Text(TextContent::new("The current chat stays open while you browse. Only this normal AI prose uses the reference green.\n\n`inline code keeps its colour`\n\n```python\nprint('syntax stays unchanged')\n```"))], ..Default::default()
    })), false);
    ui.borrow_mut().set_focus(Some(editor.clone()));
    ui.borrow_mut().start();
    native_settings::fullscreen(true, &mode, &editor, &ui, &transcript);
    ui.borrow_mut().do_render();
    let active = output.borrow().clone();
    let plain = strip_ansi(&active);
    for expected in ["Sessions", "alpha", "beta", "Current task", "Ctrl+A Add folder", "Ctrl+X Delete", "Shell finished", "HB", crate::config::VERSION, "reference green"] {
        assert!(plain.contains(expected), "missing {expected}: {plain}");
    }
    assert!(!plain.contains("hidden/full/command"));
    proof("workspace-active.ansi", &active);
    workspace.show_dialog(DialogKind::AddFolder, &ui);
    workspace.dialog.as_ref().unwrap().borrow_mut().input.handle_input(r"C:\work\existing-folder");
    ui.borrow_mut().do_render();
    proof("workspace-dialog.ansi", &output.borrow());
    workspace.close_dialog();
    transcript.borrow_mut().replace(Vec::new());
    ui.borrow_mut().do_render();
    proof("workspace-empty-chat.ansi", &output.borrow());
    // Render the actual no-selection startup components, not the retired agents page.
    ui.borrow_mut().exit_fullscreen(pi_tui::tui::ExitFullscreenOptions { flush: false, leave_alt_screen: false });
    ui.borrow_mut().enter_fullscreen(pi_tui::tui::FullscreenOptions {
        scroll: vec![Rc::new(RefCell::new(EmptyChat))], dock: Rc::new(RefCell::new(TuiText::new("Choose a chat from Sessions".into(), 1, 0, None))), mouse: true, viewport_controls: false,
    });
    ui.borrow_mut().set_fullscreen_sidebar(Some(Rc::new(RefCell::new(Pane(workspace.state.clone())))));
    ui.borrow_mut().set_fullscreen_header(Some(Rc::new(RefCell::new(EmptyHeader("C:/work/alpha".into())))));
    ui.borrow_mut().do_render();
    proof("workspace-startup.ansi", &output.borrow());
    assert!(strip_ansi(&output.borrow()).contains("No chat selected."));
}


struct WorkspaceDaemon {
    socket: String,
    commands: Arc<Mutex<Vec<Value>>>,
    rows: Arc<Mutex<Vec<SessionSummary>>>,
    task: tokio::task::JoinHandle<()>,
}

async fn serve_workspace_fixture<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    socket: S, commands: Arc<Mutex<Vec<Value>>>, rows: Arc<Mutex<Vec<SessionSummary>>>,
) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let (reader, mut writer) = tokio::io::split(socket);
    let protocol = crate::modes::daemon::daemon_protocol::daemon_protocol_info();
    let hello = json!({"type":"daemon_hello", "protocol":protocol, "clientId":"offline-workspace-fixture", "serverCapabilities":["attach_snapshot","event_sequence"]});
    writer.write_all(format!("{hello}\n").as_bytes()).await.unwrap();
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        // The native public client may send blank liveness/handshake lines.
        if line.trim().is_empty() { continue; }
        let envelope: Value = serde_json::from_str(&line).unwrap();
        let command = envelope.get("command").filter(|value| value.is_object()).unwrap_or(&envelope);
        // Result acknowledgement is transport bookkeeping, not a user mutation.
        if command["type"] == "ack_result" { continue; }
        let mut application_command = command.clone();
        application_command.as_object_mut().unwrap().remove("id");
        commands.lock().unwrap().push(application_command);
        let active = command["activeSessionId"].as_str();
        let summary = rows.lock().unwrap().iter().find(|row| row.active_session_id.as_deref() == active && active.is_some()).cloned();
        let data = match command["type"].as_str().unwrap() {
            "list" => json!({"sessions":*rows.lock().unwrap()}),
            "get_state" => serde_json::to_value(summary.as_ref().expect("exact selected active identity")).unwrap(),
            "attach" => {
                let summary = summary.unwrap();
                json!({"activeSessionId":summary.active_session_id,"snapshot":{"summary":summary,"state":{"sessionId":summary.session_id,"cwd":summary.cwd},"messages":[],"lastEventSequence":0}})
            }
            "kill" => json!({}),
            "delete_saved_session" => json!({"ok":true,"method":"unlink"}),
            "detach" => json!({}),
            other => panic!("unexpected fixture command {other}"),
        };
        let response = json!({"type":"response","id":envelope["id"],"command":command["type"],"success":true,"data":data});
        if writer.write_all(format!("{response}\n").as_bytes()).await.is_err() { return; }
    }
}

impl WorkspaceDaemon {
    fn new(rows: Vec<SessionSummary>) -> Self {
        let commands = Arc::new(Mutex::new(Vec::new()));
        let rows = Arc::new(Mutex::new(rows));
        let captured = commands.clone(); let served = rows.clone();
        #[cfg(windows)]
        let (socket, task) = {
            use tokio::net::windows::named_pipe::ServerOptions;
            let socket = format!(r"\\.\pipe\optimus-sidebar-fixture-{}", uuid::Uuid::new_v4());
            let mut server = ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
            let path = socket.clone();
            let task = tokio::spawn(async move {
                let mut clients = tokio::task::JoinSet::new();
                loop {
                    server.connect().await.unwrap();
                    let connected = server;
                    server = ServerOptions::new().create(&path).unwrap();
                    clients.spawn(serve_workspace_fixture(connected, captured.clone(), served.clone()));
                }
            });
            (socket, task)
        };
        #[cfg(unix)]
        let (socket, task) = {
            let socket = std::env::temp_dir().join(format!("sidebar-{}.sock", uuid::Uuid::new_v4())).to_string_lossy().into_owned();
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let task = tokio::spawn(async move {
                let mut clients = tokio::task::JoinSet::new();
                loop {
                    let (connected, _) = listener.accept().await.unwrap();
                    clients.spawn(serve_workspace_fixture(connected, captured.clone(), served.clone()));
                }
            });
            (socket, task)
        };
        Self { socket, commands, rows, task }
    }
}
impl Drop for WorkspaceDaemon {
    fn drop(&mut self) {
        self.task.abort();
        #[cfg(unix)]
        let _ = std::fs::remove_file(&self.socket);
    }
}

fn running_child(id: &str, parent: &str, cwd: &str) -> SessionSummary {
    SessionSummary { active_session_id: Some(format!("active-{id}")), parent_session_id: Some(parent.into()),
        runtime_kind: Some("subagent".into()), lifecycle: "live".into(), activity: "working".into(),
        is_session_active: true, is_streaming: true, ..summary(id, cwd) }
}

#[tokio::test]
async fn sidebar_followup_down_focuses_running_direct_child_and_opens_in_workspace() {
    init();
    let temp = tempfile::tempdir().unwrap(); let cwd = temp.path().to_string_lossy().into_owned();
    let current = SessionSummary { active_session_id: Some("active-parent".into()), ..summary("parent", &cwd) };
    let child = running_child("child", "parent", &cwd);
    let unrelated = running_child("unrelated", "other-parent", &cwd);
    let grandchild = running_child("grandchild", "child", &cwd);
    let idle = SessionSummary { activity: "idle".into(), is_streaming: false, is_session_active: false, ..running_child("idle", "parent", &cwd) };
    let daemon = WorkspaceDaemon::new(vec![current.clone(), unrelated.clone(), idle.clone(), grandchild.clone(), child.clone()]);
    let mut runtime = runtime(&temp); runtime.socket = Some(daemon.socket.clone());
    runtime.state.borrow_mut().update(vec![current.clone(), unrelated, idle, grandchild, child.clone()], true);
    runtime.state.borrow_mut().set_current(current);
    runtime.state.borrow_mut().focused = false;
    runtime.state.borrow_mut().selected = Some(sidebar::Item::Session("unrelated".into()));
    let h = super::super::ui_tests::FrameHarness::new("parent");
    h.mode.borrow_mut().options.return_to_agents_view = false;
    h.mode.borrow_mut().replace_subagent_summary(Some(&[local::AgentConnectionRlmChildAgentSnapshot {
        id: "child".into(), parent_id: None, active_session_id: child.active_session_id.clone(), status: "running".into(), ..Default::default()
    }]));
    let mut bar = native_subagents::Bar::new(h.mode.clone());
    let actions = Rc::new(RefCell::new(Vec::new()));
    h.editor.borrow_mut().editor_mut().set_text("first line\nlast line");
    h.editor.borrow_mut().handle_input("\x1b[A");
    assert!(!bar.input("\x1b[B", &h.editor, &actions), "Down in multiline input remains editor navigation");
    h.editor.borrow_mut().handle_input("\x1b[B");
    let cursor = h.editor.borrow().editor().get_cursor();
    assert!(!runtime.input("\x1b[B", false, &h.ui));
    assert!(bar.input("\x1b[B", &h.editor, &actions), "Down at editor end focuses the summary even with retired view disabled");
    assert!(bar.input("\r", &h.editor, &actions));
    assert!(matches!(actions.borrow_mut().pop(), Some(InputAction::Subagents)));
    h.ui.borrow_mut().set_fullscreen_sidebar_hidden(true);
    runtime.focus_subagents(&h.editor, &h.ui);
    assert!(runtime.state.borrow().focused);
    assert!(!h.ui.borrow().fullscreen_sidebar_hidden());
    assert_eq!(runtime.state.borrow().selected_session().unwrap().session_id, "child");
    assert!(runtime.input("\r", false, &h.ui));
    tokio::time::timeout(Duration::from_secs(8), async {
        while runtime.operation.is_some() { tokio::task::yield_now().await; runtime.poll(&h.ui); }
    }).await.unwrap();
    let connection = runtime.next.take().expect("existing workspace receives the attached child connection");
    let calls = daemon.commands.lock().unwrap().clone();
    assert!(calls.iter().any(|command| command["type"] == "attach" && command["activeSessionId"] == "active-child"));
    assert!(!calls.iter().any(|command| command["type"] == "create"));
    assert_eq!(h.editor.borrow().editor().get_text(), "first line\nlast line");
    assert_eq!(h.editor.borrow().editor().get_cursor(), cursor);
    connection.dispose().await.unwrap();
    runtime.state.borrow_mut().update(vec![summary("parent", &cwd), running_child("unrelated", "other", &cwd)], true);
    runtime.focus_subagents(&h.editor, &h.ui);
    assert!(!runtime.state.borrow().focused, "no running child must not focus an unrelated row");
}

#[tokio::test]
async fn sidebar_followup_active_saved_delete_routes_and_cancel_are_identity_safe() {
    init(); let temp = tempfile::tempdir().unwrap(); let mut runtime = runtime(&temp);
    let h = super::super::ui_tests::FrameHarness::new("delete-current");
    let saved = SessionSummary { session_file: Some(temp.path().join("selected.jsonl").to_string_lossy().into_owned()), ..summary("saved", &temp.path().to_string_lossy()) };
    let daemon = WorkspaceDaemon::new(vec![]); runtime.socket = Some(daemon.socket.clone());
    for active in [false, true] {
        let row = SessionSummary { active_session_id: active.then(|| "selected-active".into()), ..saved.clone() };
        runtime.show_dialog(DialogKind::Delete(row), &h.ui);
        assert!(runtime.input("\x1b", false, &h.ui));
        assert!(runtime.dialog.is_none()); assert!(runtime.operation.is_none());
    }
    assert!(daemon.commands.lock().unwrap().is_empty(), "cancel emits no mutation");
    let active = SessionSummary { active_session_id: Some("selected-active".into()), ..saved.clone() };
    let result = change_session(&daemon.socket, DialogKind::Delete(active.clone()), "").await.unwrap();
    assert!(result.contains("History kept"));
    assert_eq!(daemon.commands.lock().unwrap().as_slice(), &[json!({"type":"kill","activeSessionId":"selected-active"})]);
    daemon.commands.lock().unwrap().clear();
    let result = change_session(&daemon.socket, DialogKind::Delete(saved.clone()), "").await.unwrap();
    assert!(result.contains("Saved chat deleted"));
    let calls = daemon.commands.lock().unwrap().clone();
    assert_eq!(calls.len(), 2); assert_eq!(calls[0]["type"], "list");
    assert_eq!(calls[1], json!({"type":"delete_saved_session","sessionPath":saved.session_file}));
    daemon.commands.lock().unwrap().clear(); *daemon.rows.lock().unwrap() = vec![active];
    let error = change_session(&daemon.socket, DialogKind::Delete(saved), "").await.unwrap_err();
    assert!(error.contains("became active"));
    assert_eq!(daemon.commands.lock().unwrap().len(), 1, "active race never emits delete");
}

#[test]
fn sidebar_followup_shortcuts_full_location_copy_click_and_state() {
    use pi_tui::tui::FullscreenSidebarSide;
    init(); let temp = tempfile::tempdir().unwrap(); let mut runtime = runtime(&temp);
    let h = super::super::ui_tests::FrameHarness::new("location");
    let full = r"C:\Users\offline-fixture\projects\distinct-project-with-a-long-name\nested-folder\source";
    let file = r"C:\Users\offline-fixture\profiles\sessions\history\selected-chat.jsonl";
    runtime.state.borrow_mut().update(vec![SessionSummary { session_file: Some(file.into()), ..summary("location", full) }], true);
    runtime.state.borrow_mut().selected = Some(sidebar::Item::Session("location".into()));
    h.ui.borrow_mut().set_fullscreen_sidebar(Some(Rc::new(RefCell::new(Pane(runtime.state.clone())))));
    h.editor.borrow_mut().editor_mut().set_text("keep draft");
    let selected = runtime.state.borrow().selected.clone();
    for data in ["\x1b[109;5u", "\x1b[27;5;109~"] { assert!(runtime.input(data, false, &h.ui)); }
    assert_eq!(h.ui.borrow().fullscreen_sidebar_side(), FullscreenSidebarSide::Left);
    assert!(runtime.input("\x1b[104;5u", false, &h.ui)); assert_eq!(h.ui.borrow().fullscreen_sidebar_width(), 0);
    assert!(!runtime.input("\r", false, &h.ui)); assert!(!runtime.input("\x08", false, &h.ui)); assert!(!runtime.input("\x7f", false, &h.ui));
    assert!(runtime.input("\x1b[104;5:3u", false, &h.ui)); assert!(h.ui.borrow().fullscreen_sidebar_hidden(), "release never toggles");
    assert!(runtime.input("\x1b[27;5;104~", false, &h.ui));
    assert_eq!(runtime.state.borrow().selected, selected); assert_eq!(h.editor.borrow().editor().get_text(), "keep draft");
    runtime.state.borrow_mut().focused = true;
    assert!(runtime.input("\x1b[108;6u", false, &h.ui));
    let pane = runtime.location.as_ref().unwrap().clone();
    let text = strip_ansi(&pane.borrow_mut().render(30.0).join("\n")).replace(['│','\n',' '], "");
    assert!(text.contains(full)); assert!(text.contains(file));
    let (send, receive) = mpsc::channel(); runtime.clipboard = Some(send);
    assert!(runtime.input("\r", false, &h.ui));
    assert_eq!(receive.try_recv().unwrap(), full, "copy boundary gets the whole selected path, not display text");
    runtime.poll(&h.ui); assert!(pane.borrow().status.as_ref().unwrap().is_ok());
    runtime.input("\x1b", false, &h.ui);
    for side in [FullscreenSidebarSide::Left, FullscreenSidebarSide::Right] {
        h.ui.borrow_mut().set_fullscreen_sidebar_side(side); h.ui.borrow_mut().do_render();
        let bounds = h.ui.borrow().fullscreen_sidebar_bounds().unwrap();
        let header = h.ui.borrow().fullscreen_header_height();
        let y = header + runtime.state.borrow().location_footer_row() + 1;
        let x = bounds.col + 2;
        for event in [format!("\x1b[<0;{x};{y}m"), format!("\x1b[<32;{x};{y}M"), format!("\x1b[<2;{x};{y}M")] {
            runtime.input(&event, false, &h.ui); assert!(runtime.location.is_none());
        }
        assert!(runtime.input(&format!("\x1b[<0;{x};{y}M"), false, &h.ui)); assert!(runtime.location.is_some());
        runtime.input("\x1b", false, &h.ui);
    }
    h.ui.borrow_mut().set_fullscreen_sidebar_hidden(true);
    assert!(!runtime.input("\x1b[<0;79;20M", false, &h.ui));
    // A safe raw custom binding remains supported; raw editing bytes stay editing.
    use crate::core::keybindings::KeybindingSetting;
    let remaps = indexmap::IndexMap::from([
        ("app.sidebar.toggleVisibility".to_string(), KeybindingSetting::Single("ctrl+g".to_string())),
        ("app.sidebar.toggleSide".to_string(), KeybindingSetting::Single("ctrl+q".to_string())),
    ]);
    crate::core::keybindings::KeybindingsManager::new(remaps, None).install();
    assert!(runtime.input("\x07", false, &h.ui)); assert!(!h.ui.borrow().fullscreen_sidebar_hidden());
    assert!(runtime.input("\x11", false, &h.ui)); assert_eq!(h.ui.borrow().fullscreen_sidebar_side(), FullscreenSidebarSide::Left);
    for data in ["\r", "\x08", "\x7f"] { runtime.state.borrow_mut().focused = false; assert!(!runtime.input(data, false, &h.ui)); }
    h.editor.borrow_mut().handle_input("\x7f"); assert_eq!(h.editor.borrow().editor().get_text(), "keep draf");
    init();
}


#[test]
fn sidebar_followup_fresh_combined_native_capture_left_right_hidden_location_and_child() {
    use base64::Engine;
    use pi_tui::terminal_image::{CellDimensions, ImageProtocol, TerminalCapabilities, reset_capabilities_cache, set_capabilities, set_cell_dimensions};
    use pi_tui::tui::FullscreenSidebarSide;
    init(); let temp = tempfile::tempdir().unwrap(); let mut workspace = runtime(&temp);
    let cwd = r"C:\Users\offline-fixture\projects\Atlas-workbench\source";
    let beta = r"D:\work\Borealis-project\source";
    let current = SessionSummary { session_name:Some("Current chat".into()), active_session_id:Some("active-current".into()), session_file:Some(r"C:\Users\offline-fixture\profiles\sessions\atlas\current-chat.jsonl".into()), ..summary("current", cwd) };
    let child = SessionSummary { session_name:Some("Running child".into()), ..running_child("child", "current", cwd) };
    let active_other = SessionSummary { session_name:Some("Other active chat".into()), active_session_id:Some("active-other".into()), ..summary("other", beta) };
    workspace.state.borrow_mut().update(vec![current.clone(), child, active_other, summary("Saved inactive chat", beta)], true);
    workspace.state.borrow_mut().set_current(current);
    workspace.state.borrow_mut().selected = Some(sidebar::Item::Session("other".into()));
    workspace.state.borrow_mut().focused = false;
    let output = Rc::new(RefCell::new(String::new()));
    let ui = Rc::new(RefCell::new(TUI::new(Box::new(ProofTerminal { output:output.clone(), width:160, height:42 }), Some(true))));
    let mode = Rc::new(RefCell::new(super::super::tests::stash_mode("current")));
    mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState { session_id:"current".into(), session_name:Some("Current chat".into()), cwd:cwd.into(), ..Default::default() });
    mode.borrow_mut().replace_subagent_summary(Some(&[local::AgentConnectionRlmChildAgentSnapshot { id:"child".into(), status:"running".into(), active_session_id:Some("active-child".into()), ..Default::default() }]));
    let editor = Rc::new(RefCell::new(CustomEditor::new(ui.clone(), editor_theme(), CustomEditorOptions::default())));
    editor.borrow_mut().editor_mut().set_text("Draft preserved while sidebar moves, hides, and selects a child.");
    let bar = Rc::new(RefCell::new(native_subagents::Bar::new(mode.clone())));
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().sidebar = Some(workspace.state.clone()); transcript.borrow_mut().subagents = Some(bar.clone());
    transcript.borrow_mut().message(AgentMessage::Message(Message::User(UserMessage::new(UserContent::Text("Offline workspace capture. No provider or live session is connected.".into()), 1))), false);
    transcript.borrow_mut().message(AgentMessage::Message(Message::Assistant(AssistantMessage { content:vec![ContentBlock::Text(TextContent::new("Project groups keep their identities. Every active chat has an asterisk. Only the current chat name is red."))], ..Default::default() })), false);
    ui.borrow_mut().set_focus(Some(editor.clone())); ui.borrow_mut().start();
    native_settings::fullscreen(true, &mode, &editor, &ui, &transcript);
    let capture = |name: &str| {
        output.borrow_mut().clear(); ui.borrow_mut().request_render_forced(); ui.borrow_mut().do_render();
        let frame = output.borrow().clone();
        assert!(frame.contains("\x1b[2J"), "capture is one forced full frame, not accumulated output");
        proof(name, &frame); frame
    };
    for (side, name) in [(FullscreenSidebarSide::Left,"repair25-left.ansi"), (FullscreenSidebarSide::Right,"repair25-right.ansi")] {
        ui.borrow_mut().set_fullscreen_sidebar_side(side);
        let frame = capture(name); let plain = strip_ansi(&frame);
        for expected in ["Atlas-workbench", "Borealis-project", "Current chat", "Other active chat", "Running child", "Location / copy"] { assert!(plain.contains(expected), "{name} missing {expected}"); }
        assert_eq!(plain.matches('*').count(), 3, "only active rows have asterisks");
    }
    ui.borrow_mut().set_fullscreen_sidebar_hidden(true);
    let hidden = capture("repair25-hidden.ansi"); assert!(!strip_ansi(&hidden).contains("Sessions"));
    assert_eq!(ui.borrow().fullscreen_sidebar_width(), 0);
    ui.borrow_mut().set_fullscreen_sidebar_hidden(false);
    ui.borrow_mut().set_fullscreen_sidebar_side(FullscreenSidebarSide::Left);
    let actions = Rc::new(RefCell::new(Vec::new()));
    assert!(bar.borrow_mut().input("\x1b[B", &editor, &actions));
    capture("repair25-child-summary-focus.ansi");
    assert!(bar.borrow_mut().input("\r", &editor, &actions));
    workspace.focus_subagents(&editor, &ui);
    assert_eq!(workspace.state.borrow().selected_session().unwrap().session_id, "child");
    capture("repair25-child-selected.ansi");
    workspace.state.borrow_mut().selected = Some(sidebar::Item::Session("current".into()));
    workspace.show_location(&ui);
    let location = capture("repair25-location.ansi");
    assert!(strip_ansi(&location).contains("Full location"));
    workspace.close_dialog();
    transcript.borrow_mut().tool_start("sidebar-image", "ipython", json!({"code":"offline managed image fixture"}));
    let mut png = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(320, 144, image::Rgb([28, 140, 168])).write_to(&mut png, image::ImageFormat::Png).unwrap();
    transcript.borrow_mut().tool_result("sidebar-image", &json!({"content":[{"type":"image","mimeType":"image/png","data":base64::engine::general_purpose::STANDARD.encode(png.into_inner())}]}), false, false);
    transcript.borrow().tools.get("sidebar-image").unwrap().borrow_mut().set_expanded(true);
    set_capabilities(TerminalCapabilities { images:Some(ImageProtocol::Sixel), true_color:true, hyperlinks:true });
    set_cell_dimensions(CellDimensions { width_px:9, height_px:18 });
    ui.borrow_mut().set_fullscreen_sidebar_side(FullscreenSidebarSide::Right);
    let graphics = capture("repair25-right-graphics.ansi");
    let payload = graphics.find("\x1bP").expect("actual managed image payload");
    let pane_column = ui.borrow().fullscreen_sidebar_bounds().unwrap().col + 1;
    assert!(graphics[payload..].contains(&format!("\x1b[{pane_column}G")), "sidebar row painted after unmodified managed payload");
    reset_capabilities_cache();
    proof("repair25-capture-dimensions.json", r#"{"columns":160,"rows":42,"cellWidthPx":9,"cellHeightPx":18,"source":"actual combined native TUI forced single frames","offline":true}"#);
}
