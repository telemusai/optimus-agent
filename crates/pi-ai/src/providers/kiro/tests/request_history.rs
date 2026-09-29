use super::*;

fn assistant(ids: &[&str]) -> Message {
    Message::Assistant(AssistantMessage {
        content: ids
            .iter()
            .map(|id| ContentBlock::ToolCall(ToolCall::new(*id, "node", Default::default())))
            .collect(),
        stop_reason: "toolUse".into(),
        ..Default::default()
    })
}

fn result(id: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage::new(
        id,
        "node",
        vec![TextContent::new(text).into()],
        false,
        0,
    ))
}

fn question() -> Message {
    Message::User(UserMessage::new(
        UserContent::Text("<side_question>Are the subagents Sonnet?</side_question>".into()),
        0,
    ))
}

fn wire(messages: Vec<Message>) -> Vec<Value> {
    let mut ctx = context();
    ctx.messages.extend(messages);
    let before = serde_json::to_value(&ctx.messages).unwrap();
    let body = request::build(&model(), &ctx, &options(), "KIRO_CLI", None).unwrap();
    assert_eq!(
        serde_json::to_value(&ctx.messages).unwrap(),
        before,
        "request repair must not change the parent snapshot"
    );
    let mut history = body["conversationState"]["history"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    history.push(body["conversationState"]["currentMessage"].clone());
    history
}

fn assert_pairs(history: &[Value]) {
    for (index, entry) in history.iter().enumerate() {
        if let Some(calls) = entry["assistantResponseMessage"]["toolUses"].as_array() {
            let results = history
                .get(index + 1)
                .and_then(|next| {
                    next["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array()
                })
                .expect("tool calls need results in the immediately following user message");
            let ids = calls
                .iter()
                .map(|c| c["toolUseId"].as_str().unwrap())
                .collect::<std::collections::HashSet<_>>();
            assert_eq!(results.len(), ids.len());
            assert_eq!(
                results
                    .iter()
                    .map(|r| r["toolUseId"].as_str().unwrap())
                    .collect::<std::collections::HashSet<_>>(),
                ids
            );
        }
        if let Some(results) =
            entry["userInputMessage"]["userInputMessageContext"]["toolResults"].as_array()
        {
            let calls = index
                .checked_sub(1)
                .and_then(|i| history[i]["assistantResponseMessage"]["toolUses"].as_array())
                .expect("tool results must follow their assistant call");
            assert!(results.iter().all(|result| calls
                .iter()
                .any(|call| call["toolUseId"] == result["toolUseId"])));
        }
    }
}

#[test]
fn pending_tool_call_is_closed_before_a_side_question_or_continuation() {
    for messages in [
        vec![assistant(&["pending"])],
        vec![assistant(&["pending"]), question()],
    ] {
        let history = wire(messages);
        assert_pairs(&history);
        let current = &history.last().unwrap()["userInputMessage"];
        let result = &current["userInputMessageContext"]["toolResults"][0];
        assert_eq!(result["toolUseId"], "pending");
        assert_eq!(result["status"], "error");
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("snapshot"));
    }
}

#[test]
fn partial_results_keep_real_output_and_images_while_filling_only_the_gap() {
    let mut complete = result("complete", "actual output");
    let Message::ToolResult(ref mut message) = complete else {
        unreachable!()
    };
    message
        .content
        .push(ImageContent::new("aW1hZ2U=", "image/png").into());
    let history = wire(vec![
        assistant(&["complete", "pending"]),
        complete,
        question(),
    ]);
    assert_pairs(&history);
    let current = &history.last().unwrap()["userInputMessage"];
    let results = current["userInputMessageContext"]["toolResults"]
        .as_array()
        .unwrap();
    assert_eq!(results[0]["status"], "success");
    assert_eq!(results[0]["content"][0]["text"], "actual output");
    assert_eq!(results[1]["toolUseId"], "pending");
    assert_eq!(results[1]["status"], "error");
    assert_eq!(current["images"][0]["source"]["bytes"], "aW1hZ2U=");
    assert!(current["content"]
        .as_str()
        .unwrap()
        .contains("Are the subagents Sonnet?"));
}

#[test]
fn pending_calls_are_closed_at_assistant_and_side_followup_boundaries() {
    let history = wire(vec![
        assistant(&["first"]),
        assistant(&["second"]),
        question(),
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new("Earlier side answer"))],
            ..Default::default()
        }),
        question(),
    ]);
    assert_pairs(&history);
    assert!(history
        .iter()
        .any(|entry| entry["assistantResponseMessage"]["content"] == "Earlier side answer"));
}

#[test]
fn duplicate_late_and_orphan_results_remain_text_without_unmatched_wire_results() {
    let history = wire(vec![
        assistant(&["call"]),
        result("call", "original"),
        result("call", "duplicate"),
        question(),
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new("Side answer"))],
            ..Default::default()
        }),
        result("call", "late"),
        result("missing", "orphan"),
    ]);
    assert_pairs(&history);
    let serialized = serde_json::to_string(&history).unwrap();
    for text in ["original", "duplicate", "late", "orphan"] {
        assert!(serialized.contains(text));
    }
    assert!(
        history.last().unwrap()["userInputMessage"]["userInputMessageContext"]["toolResults"]
            .is_null()
    );
}

#[test]
fn a_later_main_snapshot_uses_the_real_result_instead_of_a_placeholder() {
    let history = wire(vec![
        assistant(&["call"]),
        result("call", "finished"),
        question(),
    ]);
    assert_pairs(&history);
    let result =
        &history.last().unwrap()["userInputMessage"]["userInputMessageContext"]["toolResults"][0];
    assert_eq!(result["status"], "success");
    assert_eq!(result["content"][0]["text"], "finished");
}

#[tokio::test]
async fn pending_side_question_snapshot_is_repaired_in_the_actual_http_request() {
    let (url, server) = fixture(
        [
            frame(json!({"content":"The subagents are Sonnet."})),
            frame(json!({"contextUsagePercentage":1})),
        ]
        .concat(),
        200,
        false,
    )
    .await;
    let mut model = model();
    model.base_url = url;
    let mut ctx = context();
    ctx.messages
        .extend([assistant(&["running-node-call"]), question()]);
    let before = serde_json::to_value(&ctx.messages).unwrap();
    let response = stream_simple_kiro(&model, &ctx, Some(&options()))
        .result()
        .await;
    assert_eq!(response.stop_reason, "stop");
    assert_eq!(
        response.content[0].as_text().unwrap().text,
        "The subagents are Sonnet."
    );
    assert_eq!(serde_json::to_value(&ctx.messages).unwrap(), before);
    let (_, payload) = server.await.unwrap();
    let mut history = payload["conversationState"]["history"]
        .as_array()
        .unwrap()
        .clone();
    history.push(payload["conversationState"]["currentMessage"].clone());
    assert_pairs(&history);
    assert!(history.last().unwrap()["userInputMessage"]["content"]
        .as_str()
        .unwrap()
        .contains("<side_question>"));
}
