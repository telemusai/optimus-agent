//! Regressions through the native event adapter, fullscreen composition and paint.
use super::*;
use pi_ai::types::{AssistantMessage, ContentBlock, Message, TextContent, ToolCall, UserContent, UserMessage};
use pi_tui::terminal::{Terminal, TerminalStopOptions};
use pi_tui::tui::Focusable;

type InputHandler = Rc<RefCell<Option<Box<dyn Fn(String)>>>>;

struct FrameTerminal {
    frame: Rc<RefCell<Vec<String>>>,
    raw: Rc<RefCell<String>>,
    width: Rc<Cell<usize>>,
    height: Rc<Cell<usize>>,
    input: InputHandler,
    mouse_tracking: bool,
}

impl Terminal for FrameTerminal {
    fn start(&mut self, input: Box<dyn Fn(String)>, _: Box<dyn Fn()>) { *self.input.borrow_mut() = Some(input); }
    fn stop(&mut self, _: TerminalStopOptions) {}
    fn drain_input(&mut self, _: u64, _: u64) {}
    fn write(&mut self, data: &str) {
        self.raw.borrow_mut().push_str(data);
        let mut frame = self.frame.borrow_mut();
        frame.resize(self.height.get(), String::new());
        if data.contains("\x1b[2J") { frame.fill(String::new()); }
        let positions = regex::Regex::new(r"\x1b\[(\d+);1H\x1b\[2K").unwrap();
        let updates: Vec<_> = positions.captures_iter(data).collect();
        for (index, capture) in updates.iter().enumerate() {
            let row = capture[1].parse::<usize>().unwrap() - 1;
            let start = capture.get(0).unwrap().end();
            let end = updates.get(index + 1).map(|next| next.get(0).unwrap().start()).unwrap_or(data.len());
            if row < frame.len() { frame[row] = pi_tui::utils::strip_ansi(&data[start..end]); }
        }
    }
    fn columns(&self) -> usize { self.width.get() }
    fn rows(&self) -> usize { self.height.get() }
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
    fn set_mouse_tracking(&mut self, enabled: bool) { self.mouse_tracking = enabled; }
    fn mouse_tracking_active(&self) -> bool { self.mouse_tracking }
    fn set_title(&mut self, _: &str) {}
    fn set_progress(&mut self, _: bool) {}
}

pub(super) struct FrameHarness {
    pub(super) mode: Rc<RefCell<InteractiveMode>>,
    pub(super) transcript: Rc<RefCell<Transcript>>,
    pub(super) editor: Rc<RefCell<CustomEditor>>,
    pub(super) ui: Rc<RefCell<TUI>>,
    pub(super) width: Rc<Cell<usize>>,
    pub(super) height: Rc<Cell<usize>>,
    frame: Rc<RefCell<Vec<String>>>,
    raw: Rc<RefCell<String>>,
    input: InputHandler,
}

impl FrameHarness {
    pub(super) fn new(session: &str) -> Self {
        let mode = Rc::new(RefCell::new(super::tests::stash_mode(session)));
        mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
            session_id: session.into(), is_streaming: true, ..Default::default()
        });
        Self::with_mode(mode, None)
    }

    fn with_mode(mode: Rc<RefCell<InteractiveMode>>, editor: Option<Rc<RefCell<CustomEditor>>>) -> Self {
        let frame = Rc::new(RefCell::new(Vec::new()));
        let raw = Rc::new(RefCell::new(String::new()));
        let width = Rc::new(Cell::new(80));
        let height = Rc::new(Cell::new(24));
        let input = Rc::new(RefCell::new(None));
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(FrameTerminal {
            frame: frame.clone(), raw: raw.clone(), width: width.clone(), height: height.clone(), input: input.clone(), mouse_tracking: false,
        }), None)));
        let editor = editor.unwrap_or_else(|| Rc::new(RefCell::new(CustomEditor::new(ui.clone(), editor_theme(), CustomEditorOptions::default()))));
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        ui.borrow_mut().set_focus(Some(editor.clone()));
        ui.borrow_mut().start();
        native_settings::fullscreen(true, &mode, &editor, &ui, &transcript);
        Self { mode, transcript, editor, ui, width, height, frame, raw, input }
    }

    pub(super) fn paint(&self) -> Vec<String> {
        self.ui.borrow_mut().do_render();
        self.frame.borrow().clone()
    }

    fn capture(&self, name: &str) -> Vec<String> {
        self.raw.borrow_mut().clear();
        self.ui.borrow_mut().request_render_forced();
        let frame = self.paint();
        if let Ok(directory) = std::env::var("OPTIMUS_UI_PROOF_DIR") {
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(std::path::Path::new(&directory).join(name), self.raw.borrow().as_bytes()).unwrap();
        }
        frame
    }

    pub(super) fn key(&self, data: &str) {
        self.input.borrow().as_ref().unwrap()(data.to_string());
        self.ui.borrow_mut().drain_input();
    }

    fn event(&self, event: wire::AgentConnectionSessionEvent) {
        apply_event(&self.mode, &self.transcript, event);
    }

    fn streaming_update(&self, message: AgentMessage) {
        let AgentMessage::Message(Message::Assistant(partial)) = &message else { panic!("assistant update required") };
        let assistant_message_event = if matches!(partial.content.last(), Some(ContentBlock::ToolCall(_))) {
            pi_ai::types::AssistantMessageEvent::ToolCallDelta { content_index: 0, delta: "arguments".into(), partial: partial.clone() }
        } else {
            pi_ai::types::AssistantMessageEvent::TextDelta { content_index: 0, delta: "grows".into(), partial: partial.clone() }
        };
        self.event(wire::AgentConnectionSessionEvent::MessageUpdate { message, assistant_message_event });
    }

    fn queue(&self, steering: &[&str], follow_up: &[&str]) {
        self.event(wire::AgentConnectionSessionEvent::SessionActionUpdate { actions: serde_json::json!({
            "steering": steering, "followUps": follow_up, "queuedCount": steering.len() + follow_up.len(), "active": "running"
        }) });
    }
}

pub(super) fn painted_transcript(mode: Rc<RefCell<InteractiveMode>>, editor: Rc<RefCell<CustomEditor>>) -> String {
    FrameHarness::with_mode(mode, Some(editor)).paint().join("\n")
}

fn user(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::User(UserMessage::new(UserContent::Text(text.into()), 1)))
}

fn assistant(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))], ..Default::default()
    }))
}

fn unused_connection() -> Arc<dyn wire::AgentConnection> {
    let client = crate::modes::daemon::daemon_client::DaemonClient::create("unused-ui-test-socket");
    Arc::new(crate::modes::agent_connection::daemon_agent_connection::DaemonAgentConnection::new(
        Arc::new(crate::main_entry::MainEntryDaemonTransport::new(client)), "active".into(), Default::default(),
    ))
}


#[test]
fn ui014_full_width_chat_and_summary_render_without_a_session_side_panel() {
    let h = FrameHarness::new("ui014-full-width");
    h.width.set(120);
    h.height.set(32);
    h.mode.borrow_mut().apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "ui014-full-width".into(), active_session_id: Some("parent".into()),
        session_name: Some("Full-width parent chat".into()), cwd: "C:/synthetic/project".into(),
        ..Default::default()
    });
    h.mode.borrow_mut().replace_subagent_summary(Some(&[
        local::AgentConnectionRlmChildAgentSnapshot { id: "direct-child".into(),
            active_session_id: Some("child".into()), status: "running".into(), ..Default::default() }
    ]));
    let bar = Rc::new(RefCell::new(native_subagents::Bar::new(h.mode.clone())));
    h.transcript.borrow_mut().subagents = Some(bar.clone());
    // Production installs the bar before entering fullscreen. This harness
    // starts with a dock, so leave it before composing the production dock.
    h.ui.borrow_mut().exit_fullscreen(pi_tui::tui::ExitFullscreenOptions {
        flush: false, leave_alt_screen: false,
    });
    native_settings::fullscreen(true, &h.mode, &h.editor, &h.ui, &h.transcript);
    h.transcript.borrow_mut().message(user("Restore the session list and keep the chat full width."), false);
    let line = format!("FULL_LEFT {} FULL_RIGHT", "content ".repeat(11));
    h.transcript.borrow_mut().message(assistant(&line), false);
    h.editor.borrow_mut().editor_mut().set_text("keep this parent draft");
    let frame = h.capture("ui014-chat-120x32.ansi");
    assert!(frame.iter().any(|row| row.contains("FULL_LEFT") && row.contains("FULL_RIGHT")), "{frame:?}");
    assert!(frame.iter().all(|row| !row.contains("Location / copy")));
    let actions = Rc::new(RefCell::new(Vec::new()));
    assert!(bar.borrow_mut().input("\x1b[B", &h.editor, &actions));
    let focused = h.capture("ui014-child-summary-focus-120x32.ansi");
    assert!(focused.iter().any(|row| row.contains("1 agent")), "{focused:?}");
    assert!(focused.iter().any(|row| row.contains("1 running") && row.contains("open")), "focused count and open hint: {focused:?}");
    assert!(bar.borrow_mut().input("\x1b[A", &h.editor, &actions));
    assert!(h.editor.borrow().editor().focused());
    h.width.set(40);
    h.height.set(18);
    let narrow = h.capture("ui014-chat-narrow-40x18.ansi");
    assert_eq!(narrow.len(), 18);
    assert!(narrow.iter().all(|row| pi_tui::utils::visible_width(row) <= 40));
    assert_eq!(h.editor.borrow().editor().get_text(), "keep this parent draft");
}

#[test]
fn native_live_queue_events_update_painted_previews_without_resync_or_draft_loss() {
    let h = FrameHarness::new("live-queue");
    h.editor.borrow_mut().editor_mut().set_text("preserve this draft");
    h.editor.borrow_mut().handle_input("\x1b[D");
    let cursor = h.editor.borrow().editor().get_cursor();
    h.event(wire::AgentConnectionSessionEvent::MessageStart { message: assistant("HELD_OPEN_REPLY") });
    let mut snapshot = h.mode.borrow().connection_state.clone().unwrap();
    snapshot.session_actions.steering = vec!["initial human steer".into()];
    h.mode.borrow_mut().apply_connection_state_snapshot(snapshot);
    assert!(h.paint().join("\n").contains("Steering: initial human steer"));

    h.queue(&["initial human steer", "Agent message received: child report", "third steer", "fourth steer", "fifth steer"], &["later follow-up"]);
    let painted = h.paint().join("\n");
    for text in ["initial human steer", "child report", "third steer", "fourth steer", "2 more queued messages"] {
        assert!(painted.contains(text), "{painted}");
    }
    assert!(!painted.contains("fifth steer"));
    assert!(painted.find("initial human steer") < painted.find("child report"));
    h.streaming_update(assistant("HELD_OPEN_REPLY grows"));
    assert!(h.paint().join("\n").contains("2 more queued messages"));

    h.queue(&["Agent message received: child report", "third steer"], &["later follow-up"]);
    let painted = h.paint().join("\n");
    assert!(!painted.contains("initial human steer"));
    assert!(painted.contains("Follow-up: later follow-up"), "{painted}");
    h.height.set(12);
    assert!(h.paint().join("\n").contains("later follow-up"));
    h.queue(&[], &[]);
    let painted = h.paint().join("\n");
    assert!(!painted.contains("Steering:") && !painted.contains("Follow-up:"));
    assert_eq!(h.editor.borrow().editor().get_text(), "preserve this draft");
    assert_eq!(h.editor.borrow().editor().get_cursor(), cursor);
    assert!(h.mode.borrow().is_agent_streaming(), "previews must appear before completion");
}

#[test]
fn native_two_of_eighty_attach_places_live_output_immediately_above_editor() {
    for session in ["running-root", "running-child", "reopened-root"] {
        let h = FrameHarness::new(session);
        let mut history = native_history::HistoryRuntime::new(unused_connection());
        let window = wire::AgentConnectionHistoryWindow {
            version: 1.0, generation: "g".into(), representation: "model".into(), tip_entry_id: Some("79".into()),
            total_message_count: 80.0, start_index: 78.0, entry_ids: vec!["78".into(), "79".into()], has_older: true, order: "chronological".into(),
        };
        apply_history_snapshot(Some(window), vec![user("RECENT_78"), user("RECENT_79")], Some(assistant("LIVE_TAIL")), &h.transcript, &h.editor, &mut history,
            Some(native_history::ViewportFill { width: 80, rows: 24 }));
        let frame = h.paint();
        let live = frame.iter().position(|line| line.contains("LIVE_TAIL")).unwrap();
        assert!(frame.iter().any(|line| line.contains("Showing 2 of 80")));
        assert!(live > frame.len() / 2, "short tail was top aligned: {frame:?}");
        assert!(h.ui.borrow().get_scroll_info().unwrap().following);
        assert_eq!(h.ui.borrow().get_scroll_info().unwrap().lines_below, 0);
    }
}

#[test]
fn native_streaming_tool_arguments_follow_the_live_edge_without_duplicate_rows() {
    let h = FrameHarness::new("streaming-tool");
    h.mode.borrow_mut().tool_output_expanded = true;
    h.editor.borrow_mut().editor_mut().set_text("tool draft");
    let update = |code: &str| AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::ToolCall(ToolCall::new("tool-1", "ipython", serde_json::json!({"code": code}).as_object().unwrap().clone()))],
        ..Default::default()
    }));
    h.event(wire::AgentConnectionSessionEvent::MessageStart { message: update("print('FIRST_ARG')") });
    h.paint();
    let code = format!("{}\nprint('LATEST_ARG')", (0..70).map(|line| format!("print('tool argument line {line}')")).collect::<Vec<_>>().join("\n"));
    h.streaming_update(update(&code));
    assert!(h.paint().join("\n").contains("LATEST_ARG"));
    assert_eq!(h.transcript.borrow().tools.len(), 1);
    h.width.set(48);
    assert!(h.paint().join("\n").contains("LATEST_ARG"));
    assert!(h.ui.borrow().get_scroll_info().unwrap().following);
    assert_eq!(h.editor.borrow().editor().get_text(), "tool draft");
}

#[test]
fn native_history_prepend_and_streaming_preserve_reading_anchor_then_follow_resumes() {
    let h = FrameHarness::new("history-anchor");
    h.editor.borrow_mut().editor_mut().set_text("history draft");
    let mut history = native_history::HistoryRuntime::new(unused_connection());
    let messages: Vec<_> = (0..80).map(|index| user(&format!("MESSAGE_{index:03} {}", "wrapped content ".repeat(10)))).collect();
    apply_history_snapshot(None, messages, Some(assistant("LIVE_STREAM")), &h.transcript, &h.editor, &mut history,
        Some(native_history::ViewportFill { width: 80, rows: 24 }));
    assert!(h.paint().iter().any(|line| line.contains("LIVE_STREAM")));
    h.key("\x1b[5~");
    let before = h.paint();
    assert!(!h.ui.borrow().get_scroll_info().unwrap().following);
    history.request(&h.mode.borrow());
    history.poll(&h.mode, &h.transcript, &h.ui);
    let after = h.paint();
    assert_eq!(&after[..12], &before[..12], "backfill jumped the reading position");
    h.streaming_update(assistant(&"LIVE_STREAM grows ".repeat(50)));
    assert_eq!(&h.paint()[..12], &before[..12], "streaming pulled a history reader down");

    let top = h.ui.borrow().get_scroll_info().unwrap().lines_above;
    let anchor = h.transcript.borrow().viewport_anchors[top..].iter().flatten().next().cloned().unwrap();
    h.width.set(48);
    h.paint();
    let resized_top = h.ui.borrow().get_scroll_info().unwrap().lines_above;
    let resized = h.transcript.borrow().viewport_anchors[resized_top..].iter().flatten().next().cloned().unwrap();
    assert_eq!(resized.key, anchor.key);
    assert!(resized.offset <= anchor.offset && anchor.offset - resized.offset < 48);
    h.key("\x1b[<65;2;3M");
    h.paint();
    h.ui.borrow_mut().scroll_to_bottom();
    assert!(h.paint().join("\n").contains("LIVE_STREAM"));
    assert!(h.ui.borrow().get_scroll_info().unwrap().following);
    assert_eq!(h.editor.borrow().editor().get_text(), "history draft");
}

/// The reported defect: a 313-message session attached with "Showing 2 of 313"
/// and a blank viewport. The painted first frame must fill the visible page,
/// keep the newest message anchored at the live edge, and keep the rest of the
/// conversation behind PageUp instead of loading it unconditionally.
#[test]
fn native_313_message_attach_fills_the_viewport_and_anchors_the_newest_message() {
    let h = FrameHarness::new("fill-313");
    let mut history = native_history::HistoryRuntime::new(unused_connection());
    let count = 313usize;
    let messages: Vec<_> = (0..count).map(|index| user(&format!("MESSAGE_{index:03}"))).collect();
    let window = wire::AgentConnectionHistoryWindow {
        version: 1.0, generation: "g".into(), representation: "model".into(), tip_entry_id: Some("312".into()),
        total_message_count: count as f64, start_index: 0.0,
        entry_ids: (0..count).map(|index| index.to_string()).collect(),
        has_older: false, order: "chronological".into(),
    };
    apply_history_snapshot(Some(window), messages, Some(assistant("LIVE_TAIL")), &h.transcript, &h.editor, &mut history,
        Some(native_history::ViewportFill { width: 80, rows: 24 }));
    let frame = h.paint();
    let painted = frame.join("\n");
    let info = h.ui.borrow().get_scroll_info().unwrap();
    assert!(info.following, "the newest message stays the live anchor");
    assert_eq!(info.lines_below, 0);
    // One page above the viewport is already rendered (the prefetch page).
    assert!(info.lines_above >= 24, "a prefetch page must sit above the viewport: {info:?}");
    assert!(painted.contains("MESSAGE_312"), "the newest message must be on screen: {painted:?}");
    assert!(painted.contains("LIVE_TAIL"));
    assert!(!painted.contains("MESSAGE_000"), "the attach must not load the whole conversation");
    let rendered_lines = h.transcript.borrow_mut().render(80.0).len();
    assert!(rendered_lines >= 48, "the transcript must cover about two pages: {rendered_lines}");
    assert!(rendered_lines < count * 3, "the fill must stay bounded: {rendered_lines}");
    let rendered = pi_tui::utils::strip_ansi(&h.transcript.borrow_mut().render(80.0).join("\n"));
    assert!(rendered.contains("of 313 messages."));
}
