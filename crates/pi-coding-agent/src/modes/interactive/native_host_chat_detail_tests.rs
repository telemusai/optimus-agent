use super::*;
use crate::core::settings_manager::{ChatDetail, SettingsManager};
use pi_ai::types::{AssistantMessage, ContentBlock, Message, TextContent, ThinkingContent};

fn persisted_mode(root: &std::path::Path) -> Rc<RefCell<InteractiveMode>> {
    let fixture = super::tests::stash_mode("chat-detail-offline");
    let mut services = fixture.ui_services;
    services.settings_manager = Arc::new(std::sync::Mutex::new(SettingsManager::create(
        &root.join("project").to_string_lossy(),
        Some(&root.join("agent").to_string_lossy()),
    )));
    let mut options = fixture.options;
    options.ui_services = Some(services);
    Rc::new(RefCell::new(InteractiveMode::new(options).unwrap()))
}

fn thinking(label: &str) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![
            ContentBlock::Thinking(ThinkingContent::new(format!(
                "**Reasoning recap**\n\n{label}"
            ))),
            ContentBlock::Text(TextContent::new("Visible answer")),
        ],
        ..Default::default()
    }))
}

fn assert_detail(mode: &InteractiveMode, detail: ChatDetail) {
    assert_eq!(mode.tool_output_expanded, detail == ChatDetail::All);
    assert_eq!(mode.agent_messages_expanded, detail == ChatDetail::All);
    assert_eq!(mode.edit_diffs_expanded, detail != ChatDetail::Overview);
    assert_eq!(mode.hide_thinking_block, detail == ChatDetail::Overview);
}

#[test]
fn chat_detail_cycles_persist_and_apply_to_restored_and_new_content() {
    let root = tempfile::tempdir().unwrap();
    let mode = persisted_mode(root.path());
    assert_detail(&mode.borrow(), ChatDetail::Details);
    let messages = vec![thinking("RESTORED_THINKING")];
    let trace = serde_json::to_vec(&messages).unwrap();
    let trace_path = root.path().join("saved-session.jsonl");
    std::fs::write(&trace_path, &trace).unwrap();
    let mut transcript = Transcript::new(mode.clone());
    transcript.replace_history(messages.clone(), 1.0);
    transcript.tool_start(
        "python",
        "ipython",
        serde_json::json!({"code":"print('offline fixture')"}),
    );
    let output = (0..30)
        .map(|n| format!("output line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    transcript.tool_result(
        "python",
        &serde_json::json!({"content":[{"type":"text","text":output}]}),
        false,
        false,
    );

    for detail in [ChatDetail::All, ChatDetail::Overview, ChatDetail::Details] {
        mode.borrow_mut().toggle_tool_output_expansion();
        transcript.apply_chat_detail();
        assert_detail(&mode.borrow(), detail);
        let text = pi_tui::utils::strip_ansi(&transcript.render(120.0).join("\n"));
        assert_eq!(
            text.contains("RESTORED_THINKING"),
            detail != ChatDetail::Overview
        );
        assert_eq!(text.contains("output line 29"), detail == ChatDetail::All);
        let reopened = persisted_mode(root.path());
        assert_detail(&reopened.borrow(), detail);
        let mut restored = Transcript::new(reopened);
        restored.replace_history(messages.clone(), 1.0);
        restored.message(thinking("NEW_THINKING"), true);
        let text = pi_tui::utils::strip_ansi(&restored.render(120.0).join("\n"));
        for label in ["RESTORED_THINKING", "NEW_THINKING"] {
            assert_eq!(text.contains(label), detail != ChatDetail::Overview);
        }
    }
    assert_eq!(std::fs::read(trace_path).unwrap(), trace);
    assert_eq!(serde_json::to_vec(&messages).unwrap(), trace);
}

#[test]
fn extension_detail_override_does_not_change_the_saved_choice() {
    let root = tempfile::tempdir().unwrap();
    let mode = persisted_mode(root.path());
    mode.borrow_mut().set_tools_expanded(true);
    assert_detail(&mode.borrow(), ChatDetail::All);
    assert!(!mode
        .borrow()
        .with_settings(|s| s.get_global_settings())
        .contains_key("chatDetail"));
    mode.borrow_mut().toggle_tool_output_expansion();
    assert_detail(&mode.borrow(), ChatDetail::Overview);
    mode.borrow_mut().set_tools_expanded(true);
    assert_detail(&mode.borrow(), ChatDetail::All);
    assert_detail(&persisted_mode(root.path()).borrow(), ChatDetail::Overview);
}

#[test]
fn jev_details_follow_ctrl_o_for_live_restored_and_new_results() {
    use pi_ai::types::{ImageOrTextContent, ToolCall, ToolResultMessage};
    use serde_json::json;

    let root = tempfile::tempdir().unwrap();
    let mode = persisted_mode(root.path());
    let args = json!({"state":"JEV_REQUEST_CONTEXT", "questions":{
        "coin":{"type":"choice", "instructions":"Choose", "criteria":{"heads":"Heads","tails":"Tails"}}
    }});
    let output =
        json!({"category":"dynamic", "answers":{"coin":{"choice":"JEV_RAW_ANSWER"}}}).to_string();
    let messages = vec![
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::ToolCall(ToolCall::new(
                "history",
                "jev_decide",
                args.as_object().unwrap().clone(),
            ))],
            ..Default::default()
        })),
        AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
            "history",
            "jev_decide",
            vec![ImageOrTextContent::Text(TextContent::new(&output))],
            false,
            0,
        ))),
    ];
    let original = serde_json::to_vec(&messages).unwrap();
    let mut transcript = Transcript::new(mode.clone());
    transcript.replace_history(messages.clone(), 2.0);

    for (index, detail) in [
        ChatDetail::Details,
        ChatDetail::All,
        ChatDetail::Overview,
        ChatDetail::Details,
    ]
    .into_iter()
    .enumerate()
    {
        if index > 0 {
            mode.borrow_mut().toggle_tool_output_expansion();
            transcript.apply_chat_detail();
        }
        let id = format!("live-{index}");
        transcript.tool_start(&id, "jev_decide", args.clone());
        transcript.tool_result(
            &id,
            &json!({"content":[{"type":"text", "text":output}]}),
            false,
            false,
        );
        let text = pi_tui::utils::strip_ansi(&transcript.render(120.0).join("\n"));
        let expected = if detail == ChatDetail::All {
            index + 2
        } else {
            0
        };
        assert_eq!(text.matches("JEV_REQUEST_CONTEXT").count(), expected);
        assert_eq!(text.matches("JEV_RAW_ANSWER").count(), expected);
        assert_eq!(text.matches("Jev decision").count(), index + 2);

        let reopened = persisted_mode(root.path());
        assert_detail(&reopened.borrow(), detail);
        let mut restored = Transcript::new(reopened);
        restored.replace_history(messages.clone(), 2.0);
        let text = pi_tui::utils::strip_ansi(&restored.render(120.0).join("\n"));
        assert_eq!(text.contains("JEV_RAW_ANSWER"), detail == ChatDetail::All);
    }
    assert_eq!(serde_json::to_vec(&messages).unwrap(), original);
}
