use super::*;
use pi_ai::types::{
    AssistantMessage, ContentBlock, Message, TextContent, ThinkingContent, UserContent, UserMessage,
};
use pi_tui::fullscreen::FullscreenViewport;

fn mode() -> Rc<RefCell<InteractiveMode>> {
    let mut mode = super::super::tests::stash_mode("neon-offline-fixture");
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    mode.fullscreen_enabled = true;
    mode.apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "neon-offline-fixture".into(),
        cwd: "~/agents/optimus-agent".into(),
        session_name: Some("Add tests for @filepath".into()),
        model: Some(pi_ai::types::Model {
            id: "offline-fixture".into(),
            name: "Offline fixture".into(),
            ..Default::default()
        }),
        context_usage: local::ContextUsage {
            tokens: Some(146_000.0),
            context_window: 1_000_000.0,
            percent: Some(14.6),
        },
        ..Default::default()
    });
    Rc::new(RefCell::new(mode))
}

fn user(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text(text.into()),
        1_790_200_000_000,
    )))
}
fn assistant(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        timestamp: 1_790_200_005_000,
        ..Default::default()
    }))
}

#[test]
fn neon_header_and_timeline_fit_unicode_and_tiny_windows() {
    let _mode = mode();
    let palette = crate::modes::interactive::theme::theme::load_theme_from_path(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../resources/agent/src/modes/interactive/theme/neon.json"
        ),
        Some(crate::modes::interactive::theme::theme::TerminalColorMode::Truecolor),
    )
    .unwrap();
    assert_eq!(palette.get_fg_ansi("accent"), "\x1b[38;2;0;244;119m");
    assert_eq!(palette.get_fg_ansi("thinkingText"), "\x1b[38;2;226;1;234m");
    assert_eq!(palette.get_fg_ansi("text"), "\x1b[38;2;179;188;199m");
    let data = HeaderData {
        cwd: "~/projects/界界界",
        session: "Long title 👩‍💻 ",
        model: "fixture-model",
        phase: "READY",
        jev: Some("Jev Full · fallback"),
        clock: "12:34:56",
    };
    for width in [
        1, 8, 24, 35, 36, 79, 80, 89, 90, 100, 109, 110, 120, 159, 160, 240,
    ] {
        for budget in 0..12 {
            let rows = render_header(width, budget, &data);
            assert!(rows.len() <= budget);
            assert!(
                rows.iter().all(|r| visible_width(r) == width),
                "header {width}: {rows:?}"
            );
        }
        let t = Timeline::new(width);
        let rows = pi_tui::utils::wrap_text_with_ansi(
            "hello 世界 👩‍💻 with a long output that must remain readable",
            t.content_width(),
        );
        for row in rows {
            let decorated = t.line(
                &row,
                Some(&RowMeta::new(Kind::Thinking, Some(1_790_200_000_000))),
                true,
            );
            assert_eq!(visible_width(&decorated), width, "{width}: {decorated:?}");
        }
    }
    assert!(render_header(160, 7, &data)
        .join("\n")
        .contains("telemus.ai"));
    assert!(!render_header(80, 3, &data)
        .join("\n")
        .contains("telemus.ai"));
    let header = render_header(160, 7, &data);
    assert!(strip_ansi(&header[0]).trim().is_empty());
    let strip = strip_ansi(&header[4]);
    assert!(strip.contains("● fixture-model  │  ● Jev Full · fallback"));
    assert!(header[4].contains(&format!(
        "{}● Jev Full · fallback",
        theme().get_fg_ansi("warning")
    )));
    assert!(strip_ansi(&header[1]).trim_end().ends_with("telemus.ai"));
    let disabled = HeaderData {
        jev: Some("Jev Off"),
        ..data
    };
    assert!(render_header(120, 7, &disabled)
        .join("\n")
        .contains("○ Jev Off"));
}

#[test]
fn neon_metadata_keeps_recorded_times_streaming_identity_and_tool_outcomes() {
    let mode = mode();
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().message(user("hello"), false);
    let mut thinking = AssistantMessage {
        content: vec![ContentBlock::Thinking(ThinkingContent::new(
            "Checking the implementation.",
        ))],
        timestamp: 1_790_200_005_000,
        ..Default::default()
    };
    transcript.borrow_mut().message(
        AgentMessage::Message(Message::Assistant(thinking.clone())),
        true,
    );
    let key = transcript.borrow().row_keys[&1].clone();
    assert_eq!(transcript.borrow().row_metadata[&key].kind, Kind::Thinking);
    thinking
        .content
        .push(ContentBlock::Text(TextContent::new("Here is the result.")));
    transcript
        .borrow_mut()
        .message(AgentMessage::Message(Message::Assistant(thinking)), false);
    assert_eq!(transcript.borrow().row_metadata[&key].kind, Kind::Assistant);
    assert_eq!(transcript.borrow().rows.len(), 2);
    let event: wire::AgentConnectionSessionEvent = serde_json::from_value(serde_json::json!({
        "type": "tool_execution_start", "toolCallId":"fixture-tool", "toolName":"read", "args":{"path":"README.md"}
    })).unwrap();
    apply_event(&mode, &transcript, event);
    assert!(transcript.borrow().row_metadata["tool:fixture-tool"]
        .started
        .is_some());
    for _ in 0..2 {
        transcript.borrow_mut().tool_result(
            "fixture-tool",
            &serde_json::json!({"content":[{"type":"text","text":"fixture output"}]}),
            true,
            false,
        );
        let t = transcript.borrow();
        assert_eq!(t.row_metadata["tool:fixture-tool"].kind, Kind::Error);
        assert!(t.row_metadata["tool:fixture-tool"].elapsed.is_some());
    }
    transcript
        .borrow_mut()
        .replace_history(vec![user("older recorded message")], 1.0);
    let t = transcript.borrow();
    assert_eq!(
        t.history
            .as_ref()
            .unwrap()
            .row_metadata
            .values()
            .next()
            .unwrap()
            .timestamp,
        Some(1_790_200_000_000)
    );
    assert_eq!(RowMeta::new(Kind::Notice, Some(0)).timestamp, None);
}

#[test]
fn neon_copy_excludes_gutter_and_border_and_preserves_unicode() {
    let _mode = mode();
    let t = Timeline::new(120);
    let mut viewport = FullscreenViewport::new();
    let rows = ["hello 世界", "second 👩‍💻 line"].map(|s| {
        t.line(
            s,
            Some(&RowMeta::new(Kind::User, Some(1_790_200_000_000))),
            true,
        )
    });
    viewport.set_transcript_presentation(vec![Some((t.left, t.width - t.right)); 2], t.padding());
    viewport.compose_frame_with_header(
        &rows,
        &["input".into()],
        5,
        &[],
        &[],
        false,
        &["header".into()],
    );
    assert!(viewport.begin_selection(1, 0));
    viewport.extend_selection(2, 120);
    let copied = viewport.end_selection().unwrap();
    assert_eq!(copied, "hello 世界\nsecond 👩‍💻 line");
}

#[test]
fn neon_meter_uses_actual_usage_and_omits_unknown_or_nonfinite_values() {
    let _mode = mode();
    for percent in [Some(f64::NAN), Some(f64::INFINITY), None] {
        assert_eq!(
            strip_ansi(&context_meter(
                Some(&local::ContextUsage {
                    percent,
                    ..Default::default()
                }),
                "unknown",
                160
            )),
            "unknown"
        );
    }
    let usage = local::ContextUsage {
        percent: Some(14.6),
        ..Default::default()
    };
    assert_eq!(
        strip_ansi(&context_meter(Some(&usage), "146k (15%)", 160)),
        "[━·········] 146k (15%)"
    );
    assert_eq!(
        strip_ansi(&context_meter(Some(&usage), "146k (15%)", 80)),
        "146k (15%)"
    );
}

#[test]
fn neon_frame_fixture_keeps_editor_jev_history_and_theme_switching() {
    use crate::modes::interactive::components::subagent_summary_line::{
        RightStatus, SubagentSummaryCounts, SubagentSummaryLine,
    };
    let mode = mode();
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    transcript.borrow_mut().replace(vec![
        user("Please inspect the project and add tests for the failing parser."),
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Thinking(ThinkingContent::new(
                "I’ll inspect the parser, reproduce the failure, then run the focused tests.",
            ))],
            timestamp: 1_790_200_005_000,
            ..Default::default()
        })),
    ]);
    transcript.borrow_mut().tool_start(
        "read-fixture",
        "read",
        serde_json::json!({"path":"src/parser.rs"}),
    );
    transcript.borrow_mut().tool_result("read-fixture", &serde_json::json!({"content":[{"type":"text","text":"pub fn parse(input: &str) -> Result<Value, Error> {\n    parse_value(input.trim())\n}"}]}), false, false);
    transcript
        .borrow_mut()
        .row_metadata
        .get_mut("tool:read-fixture")
        .unwrap()
        .elapsed = Some(Duration::from_millis(1200));
    transcript.borrow_mut().message(assistant("Added coverage for empty input and Unicode.\n\nThe focused parser tests pass. The changes are ready for review."), false);
    let ui = Rc::new(RefCell::new(TUI::new(
        Box::new(pi_tui::terminal::ProcessTerminal::new()),
        Some(false),
    )));
    let editor = Rc::new(RefCell::new(CustomEditor::new(
        ui.clone(),
        editor_theme(),
        CustomEditorOptions::default(),
    )));
    editor
        .borrow_mut()
        .editor_mut()
        .set_text("Add tests for @filepath");
    let mut agents = SubagentSummaryLine::default();
    agents.set_always_visible(true);
    agents.set_subagent_counts(SubagentSummaryCounts {
        total: 11,
        running: 1,
        idle: 2,
        inactive: 8,
        ..Default::default()
    });
    agents.set_right_status(Some(RightStatus {
        full: "Jev Full · fallback · 413 req · 603.8k in/52.2k out · 242ms".into(),
        compact: "Jev Full · fallback · 413 req".into(),
        minimal: "Jev Full · fallback".into(),
    }));
    let dock = Rc::new(RefCell::new(pi_tui::tui::Container::new()));
    dock.borrow_mut().add_child(editor.clone());
    dock.borrow_mut().add_child(Rc::new(RefCell::new(agents)));
    dock.borrow_mut()
        .add_child(Rc::new(RefCell::new(Tray(mode.clone(), editor))));
    let mut dock = Dock(dock);
    let mut header = Header(mode.clone(), Rc::downgrade(&transcript));
    for (width, height) in [
        (160usize, 48usize),
        (120, 36),
        (100, 30),
        (80, 24),
        (40, 12),
    ] {
        let rows = transcript.borrow_mut().render(width as f64);
        if width >= 100 {
            assert!(
                rows.join("\n").contains("1.2s"),
                "elapsed time must survive padded tool rows"
            );
        }
        let dock_rows = dock.render(width as f64);
        let budget = (height / 4).min(height.saturating_sub(
            pi_tui::fullscreen::clipped_fullscreen_dock_height(dock_rows.len(), height)
                + pi_tui::fullscreen::FULLSCREEN_MIN_TRANSCRIPT_ROWS,
        ));
        let header_rows = header.render_with_height(width as f64, budget);
        let mut viewport = FullscreenViewport::new();
        viewport.set_transcript_presentation(
            transcript.borrow().get_selection_columns(),
            transcript.borrow().get_fullscreen_padding(),
        );
        let frame = viewport.compose_frame_with_header(
            &rows,
            &dock_rows,
            height,
            &[],
            &transcript.borrow().get_viewport_anchors(),
            true,
            &header_rows,
        );
        assert_eq!(frame.len(), height);
        assert!(frame.iter().all(|r| visible_width(r) <= width));
        let plain = frame
            .iter()
            .map(|r| strip_ansi(r))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(plain.contains("Add tests for @filepath"));
        assert!(plain.contains("Jev Full"));
        if width >= 100 {
            assert!(plain.contains("146k"));
        }
        if let Ok(dir) = std::env::var("OPTIMUS_NEON_CAPTURE_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/{width}x{height}.ansi"), frame.join("\n")).unwrap();
            std::fs::write(format!("{dir}/{width}x{height}.txt"), plain).unwrap();
        }
    }
    // Rendering a different theme immediately removes the fixed chrome/margins.
    crate::modes::interactive::theme::theme::set_theme("prime", false);
    assert!(header.render(120.0).is_empty());
    transcript.borrow_mut().render(120.0);
    assert!(transcript.borrow().get_selection_columns().is_empty());
    assert!(!dock
        .render(120.0)
        .iter()
        .any(|r| strip_ansi(r).starts_with('└')));
    // Switching back, and toggling inline mode, keep the original text available.
    crate::modes::interactive::theme::theme::set_theme("neon", false);
    mode.borrow_mut().fullscreen_enabled = false;
    assert!(transcript
        .borrow_mut()
        .render(120.0)
        .join("\n")
        .contains("focused parser tests pass"));
    assert!(transcript.borrow().get_selection_columns().is_empty());
}

#[test]
fn neon_padded_tool_duration_is_visible_but_not_copied() {
    let mode = mode();
    let mut transcript = Transcript::new(mode);
    transcript.tool_start("timed", "read", serde_json::json!({"path":"src/parser.rs"}));
    transcript.tool_result(
        "timed",
        &serde_json::json!({"content":[{"type":"text","text":"source code"}]}),
        false,
        false,
    );
    transcript
        .row_metadata
        .get_mut("tool:timed")
        .unwrap()
        .elapsed = Some(Duration::from_millis(1200));
    let rows = transcript.render(120.0);
    assert!(rows.join("\n").contains("1.2s"));
    let mut viewport = FullscreenViewport::new();
    viewport.set_transcript_presentation(
        transcript.get_selection_columns(),
        transcript.get_fullscreen_padding(),
    );
    viewport.compose_frame(&rows, &[], rows.len(), &[]);
    assert!(viewport.begin_selection(0, 0));
    viewport.extend_selection(rows.len() as i64 - 1, 120);
    let text = viewport.end_selection().unwrap();
    assert!(text.contains("source code"));
    assert!(!text.contains("1.2s"));
    assert!(!text.contains('│'));
}

#[test]
fn neon_canvas_preserves_cursor_and_hyperlink_control_sequences() {
    let _mode = mode();
    let cursor = pi_tui::tui::CURSOR_MARKER;
    let link = "\x1b]8;;https://telemus.ai\x07telemus.ai\x1b]8;;\x07";
    let painted = surface(&format!("draft {cursor}{link}"), 80);
    assert!(painted.contains(cursor));
    assert!(painted.contains(link));
    assert_eq!(visible_width(&painted), 80);
}
