use super::*;
use pi_ai::types::{AssistantMessage, TextContent, ThinkingContent, ToolCall};
use pi_tui::utils::visible_width;

fn message(content: Vec<ContentBlock>, output: f64) -> AssistantMessage {
    AssistantMessage {
        content,
        usage: pi_ai::types::Usage {
            output,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn live_preview_streams_decoded_code_then_final_usage_without_thinking_or_json() {
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let mut preview = LivePreview::default();
    for (code, tokens) in [
        ("print(\"世", 0.0),
        ("print(\"世界\")\nanswer = 42", 0.0),
        ("print(\"世界\")\nanswer = 42", 19.0),
    ] {
        preview.update(
            &message(
                vec![ContentBlock::ToolCall(ToolCall {
                    id: "call-1".into(),
                    name: "ipython".into(),
                    arguments: serde_json::json!({"code":code})
                        .as_object()
                        .unwrap()
                        .clone(),
                    ..Default::default()
                })],
                tokens,
            ),
            tokens == 0.0,
        );
        let rendered = strip_ansi(&preview.render(120, 5).join("\n"));
        assert!(
            rendered.contains(code.lines().last().unwrap()),
            "{rendered}"
        );
        assert!(!rendered.contains("\"code\":"));
        assert!(rendered.contains(if tokens == 0.0 {
            "LIVE · tokens pending"
        } else {
            "LAST OUTPUT · 19 tokens"
        }));
    }
    preview.update(
        &message(
            vec![ContentBlock::Thinking(ThinkingContent::new(
                "private reasoning",
            ))],
            0.0,
        ),
        true,
    );
    let rendered = strip_ansi(&preview.render(120, 5).join("\n"));
    assert!(!rendered.contains("private reasoning"));
    assert!(!rendered.contains("answer = 42"));
    assert!(rendered.contains("Waiting for text or code"));
}

#[test]
fn live_preview_bounds_memory_sanitizes_output_and_fits_all_terminal_sizes() {
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let mut preview = LivePreview::default();
    preview.update(
        &message(
            vec![ContentBlock::Text(TextContent::new(format!(
                "{}\x1b[2J\x07 tail 👩‍💻",
                "界".repeat(100_000)
            )))],
            f64::NAN,
        ),
        true,
    );
    assert!(preview.tail.chars().count() <= MAX_TAIL_CHARS);
    assert!(!preview.tail.contains('\x1b'));
    assert!(!preview.tail.contains('\x07'));
    assert_eq!(preview.output_tokens, None);
    for width in [0, 1, 40, 59, 60, 80, 120, 160] {
        for budget in 0..14 {
            let height = preview.height(width, budget);
            let rows = preview.render(width, height);
            assert_eq!(rows.len(), height);
            assert!(
                rows.iter().all(|row| visible_width(row) == width),
                "{width}: {rows:?}"
            );
            assert!(height == 0 || budget - height >= 3);
        }
    }
}

#[test]
fn live_preview_event_path_and_header_capture_preserve_editor_and_clear_on_session_change() {
    let mut controller = super::super::tests::stash_mode("live-preview-fixture");
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    controller.fullscreen_enabled = true;
    let mode = Rc::new(RefCell::new(controller));
    let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
    let mut header = native_neon::Header(mode.clone(), Rc::downgrade(&transcript));
    let ui = Rc::new(RefCell::new(TUI::new(
        Box::new(pi_tui::terminal::ProcessTerminal::new()),
        Some(false),
    )));
    let editor = Rc::new(RefCell::new(CustomEditor::new(
        ui,
        editor_theme(),
        CustomEditorOptions::default(),
    )));
    editor
        .borrow_mut()
        .editor_mut()
        .set_text("Draft stays editable");
    let assistant = message(vec![ContentBlock::Text(TextContent::new("Building the parser…\nfn parse(input: &str) {\n    let value = input.trim();\n    println!(\"{value}\");\n}"))], 37.0);
    apply_event(
        &mode,
        &transcript,
        wire::AgentConnectionSessionEvent::MessageStart {
            message: AgentMessage::Message(Message::Assistant(assistant.clone())),
        },
    );
    for (width, height) in [(160usize, 48usize), (120, 36), (80, 24), (40, 12)] {
        let dock = editor.borrow_mut().render(width as f64);
        let budget = (height / 4).min(height.saturating_sub(
            pi_tui::fullscreen::clipped_fullscreen_dock_height(dock.len(), height)
                + pi_tui::fullscreen::FULLSCREEN_MIN_TRANSCRIPT_ROWS,
        ));
        let rows = header.render_with_height(width as f64, budget);
        assert!(rows.len() <= budget);
        assert!(rows.iter().all(|row| visible_width(row) == width));
        let mut viewport = pi_tui::fullscreen::FullscreenViewport::new();
        let frame = viewport.compose_frame_with_header(
            &transcript.borrow_mut().render(width as f64),
            &dock,
            height,
            &[],
            &[],
            true,
            &rows,
        );
        assert_eq!(frame.len(), height);
        assert!(frame.iter().all(|row| visible_width(row) <= width));
        let plain = strip_ansi(&frame.join("\n"));
        assert!(plain.contains("Draft stays editable"), "{plain}");
        if width >= 60 {
            assert!(plain.contains("LIVE · 37 tokens"), "{plain}");
        }
        if let Ok(dir) = std::env::var("OPTIMUS_NEON_CAPTURE_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/live-{width}x{height}.ansi"),
                frame.join("\n"),
            )
            .unwrap();
            std::fs::write(format!("{dir}/live-{width}x{height}.txt"), plain).unwrap();
        }
    }
    apply_event(
        &mode,
        &transcript,
        wire::AgentConnectionSessionEvent::MessageEnd {
            message: AgentMessage::Message(Message::Assistant(assistant)),
        },
    );
    assert!(strip_ansi(&header.render(120.0).join("\n")).contains("LAST OUTPUT · 37 tokens"));
    transcript.borrow_mut().replace(Vec::new());
    assert!(!strip_ansi(&header.render(120.0).join("\n")).contains("LAST OUTPUT"));
}
