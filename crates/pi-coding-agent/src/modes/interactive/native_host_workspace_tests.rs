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
