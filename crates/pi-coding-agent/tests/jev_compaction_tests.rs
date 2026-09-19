use std::sync::Arc;

use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{AgentMessage, AgentState, CustomAgentMessage, CustomMessageContent};
use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{
    AssistantMessage, ContentBlock, ImageContent, ImageOrTextContent, Message, TextContent,
    ThinkingContent, ToolCall, ToolResultMessage, UserContent, UserMessage,
};
use pi_coding_agent::core::jev_compaction::{apply_context, prepare_context, PreparedCompaction};
use pi_jev::compaction::{CompactionConfig, CompactionSkip, TRUNCATION_MARKER};
use pi_jev::{
    Answer, DecisionCategory, DecisionOutcome, DecisionRecord, JevLimits, JevMode, JevStats,
    JevSystemOne, MockJevTransport, MockStep, SecretString, SystemOne,
};
use serde_json::json;

fn user(text: &str) -> AgentMessage {
    UserMessage::new(UserContent::Text(text.into()), 1).into()
}
fn assistant(text: &str) -> AgentMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        ..Default::default()
    }
    .into()
}
fn call(id: &str, name: &str) -> AgentMessage {
    AssistantMessage { content: vec![ContentBlock::ToolCall(ToolCall::new(id, name, json!({"code":"from pathlib import Path\nprint(Path(\"src/parser.rs\").read_text())","path":"input.txt"}).as_object().unwrap().clone()))], stop_reason: "toolUse".into(), ..Default::default() }.into()
}
fn result(id: &str, name: &str, text: &str) -> AgentMessage {
    ToolResultMessage::new(
        id,
        name,
        vec![ImageOrTextContent::Text(TextContent::new(text))],
        false,
        2,
    )
    .into()
}
fn transcript(name: &str) -> Vec<AgentMessage> {
    let mut messages = vec![
        user("Keep exact constraints. Diagnose the parser."),
        call("old-call", name),
        result("old-call", name, &"old source output\n".repeat(1200)),
    ];
    for i in 0..7 {
        messages.push(assistant(&format!("Recent reasoning {i}")));
    }
    messages
}
fn outcome(prepared: &PreparedCompaction, call: f64, result: f64) -> DecisionOutcome {
    DecisionOutcome {
        records: prepared
            .plan
            .questions
            .keys()
            .map(|id| DecisionRecord {
                question_id: id.clone(),
                category: DecisionCategory::ContextRelevance,
                answer: Answer::Noul {
                    noul: if id.contains(".call_") { call } else { result },
                },
                response_model: Some("mock".into()),
                requested_model: "jev-latest".into(),
                applied: false,
            })
            .collect(),
        skips: vec![],
        response_model: Some("mock".into()),
        usage: Default::default(),
        applied: false,
        attempts: 1,
    }
}
fn tool_text(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Message(Message::ToolResult(result)) => result
            .content
            .iter()
            .filter_map(|block| match block {
                ImageOrTextContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect(),
        _ => panic!("not a tool result"),
    }
}

#[test]
fn deletes_only_complete_unpinned_read_pairs_preserving_all_prose() {
    let messages = transcript("read_file");
    let original = messages.clone();
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    let compacted = apply_context(messages, &prepared, &outcome(&prepared, 0.1, 0.1)).unwrap();
    assert_eq!(compacted.messages.len(), original.len() - 2);
    assert_eq!(compacted.messages[0], original[0]);
    assert_eq!(&compacted.messages[1..], &original[3..]);
    assert_eq!(compacted.stats.calls_removed, 1);
    assert!(compacted.stats.reduction_ratio > 0.8);
}

#[test]
fn native_ipython_keeps_call_and_only_truncates_old_successful_output() {
    let messages = transcript("ipython");
    let original = messages.clone();
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    let compacted = apply_context(messages, &prepared, &outcome(&prepared, 0.1, 0.1)).unwrap();
    assert_eq!(compacted.messages.len(), original.len());
    assert_eq!(compacted.messages[1], original[1]);
    assert_eq!(&compacted.messages[3..], &original[3..]);
    assert!(tool_text(&compacted.messages[2]).contains(TRUNCATION_MARKER));
    assert!(tool_text(&original[2]).len() > 10_000);
    assert_eq!(compacted.stats.calls_removed, 0);
    assert_eq!(compacted.stats.results_truncated, 1);
    assert!(matches!(
        prepare_context(&compacted.messages, &CompactionConfig::default()),
        Err(CompactionSkip::NoCandidates)
    ));
}

#[test]
fn unicode_multi_block_result_head_is_one_budget_and_metadata_survives() {
    let mut messages = transcript("ipython");
    let AgentMessage::Message(Message::ToolResult(result)) = &mut messages[2] else {
        unreachable!()
    };
    result.content = vec![
        ImageOrTextContent::Text(TextContent::new("é界🙂".repeat(2000))),
        ImageOrTextContent::Text(TextContent::new("second".repeat(1000))),
    ];
    result.details = Some(json!({"status":"ok","artifact":"saved"}));
    let before = result.clone();
    for head in [0, 7] {
        let config = CompactionConfig {
            truncate_head_chars: head,
            ..Default::default()
        };
        let prepared = prepare_context(&messages, &config).unwrap();
        let compacted =
            apply_context(messages.clone(), &prepared, &outcome(&prepared, 0.9, 0.1)).unwrap();
        let AgentMessage::Message(Message::ToolResult(after)) = &compacted.messages[2] else {
            unreachable!()
        };
        assert_eq!(after.details, before.details);
        assert_eq!(after.tool_call_id, before.tool_call_id);
        assert_eq!(after.timestamp, before.timestamp);
        assert_eq!(after.content.len(), 1);
        let text = tool_text(&compacted.messages[2]);
        assert!(text.chars().count() <= head + 140);
        assert!(!text.contains("re-run"));
        if head == 7 {
            assert!(text.starts_with("é界🙂é界🙂é\n"));
        }
    }
}

#[test]
fn first_recent_error_image_edit_and_coordination_results_are_pinned() {
    let original = transcript("ipython");
    let mut cases = Vec::new();
    let mut first = original.clone();
    first.remove(0);
    cases.push(first);
    cases.push(original[..3].to_vec());
    let mut error = original.clone();
    if let AgentMessage::Message(Message::ToolResult(result)) = &mut error[2] {
        result.is_error = true;
    }
    cases.push(error);
    let mut image = original.clone();
    if let AgentMessage::Message(Message::ToolResult(result)) = &mut image[2] {
        result
            .content
            .push(ImageOrTextContent::Image(ImageContent::new(
                "base64",
                "image/png",
            )));
    }
    cases.push(image);
    cases.push(transcript("edit"));
    cases.push(transcript("bash"));
    cases.push(transcript("unknown_tool"));
    for details in [
        json!({"diffs":[{"path":"edited.rs"}]}),
        json!({"sentAgentMessages":[{}]}),
        json!({"status":"error"}),
        json!({"stderr":"warning"}),
        json!({"kernelRestarted":true}),
    ] {
        let mut messages = original.clone();
        if let AgentMessage::Message(Message::ToolResult(result)) = &mut messages[2] {
            result.details = Some(details);
        }
        cases.push(messages);
    }
    for messages in cases {
        assert!(matches!(
            prepare_context(&messages, &CompactionConfig::default()),
            Err(CompactionSkip::NoCandidates)
        ));
    }
}

#[test]
fn native_python_mutation_code_is_not_eligible_even_without_error_flag() {
    for code in [
        "await edit(path='x')",
        "p.write_text('x')",
        "h=bash('cat x')",
        "await agent_message.send('x')",
        "await rlm('x')",
        "print(source_text)",
        "fn()",
        "from pathlib import Path\nprint(Path(\"AGENTS.md\").read_text())",
        "from pathlib import Path\nprint(Path(\"src/parser.rs\").read_text()); mutate()",
    ] {
        let mut messages = transcript("ipython");
        if let AgentMessage::Message(Message::Assistant(assistant)) = &mut messages[1] {
            if let ContentBlock::ToolCall(call) = &mut assistant.content[0] {
                call.arguments.insert("code".into(), json!(code));
            }
        }
        assert!(matches!(
            prepare_context(&messages, &CompactionConfig::default()),
            Err(CompactionSkip::NoCandidates)
        ));
    }
}

#[test]
fn parallel_tool_calls_remove_only_selected_pair_and_keep_signed_assistant_blocks() {
    let mut messages = transcript("read_file");
    if let AgentMessage::Message(Message::Assistant(assistant)) = &mut messages[1] {
        assistant.content.insert(
            0,
            ContentBlock::Text(TextContent::new("Keep this explanation")),
        );
        assistant.content.push(ContentBlock::ToolCall(ToolCall::new(
            "second-call",
            "read_file",
            Default::default(),
        )));
    }
    messages.insert(3, result("second-call", "read_file", &"b".repeat(10_000)));
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    let mut answers = outcome(&prepared, 0.1, 0.1);
    for record in &mut answers.records {
        if record.question_id.ends_with("t2") {
            record.answer = Answer::Noul { noul: 0.9 };
        }
    }
    let compacted = apply_context(messages.clone(), &prepared, &answers).unwrap();
    let AgentMessage::Message(Message::Assistant(assistant)) = &compacted.messages[1] else {
        unreachable!()
    };
    assert_eq!(assistant.content.len(), 2);
    assert_eq!(
        assistant.content[0].as_text().unwrap().text,
        "Keep this explanation"
    );
    assert_eq!(
        assistant.content[1].as_tool_call().unwrap().id,
        "second-call"
    );
    let mut signed = transcript("read_file");
    if let AgentMessage::Message(Message::Assistant(assistant)) = &mut signed[1] {
        let mut thinking = ThinkingContent::new("opaque reasoning");
        thinking.thinking_signature = Some("opaque".into());
        assistant
            .content
            .insert(0, ContentBlock::Thinking(thinking));
    }
    let prepared = prepare_context(&signed, &CompactionConfig::default()).unwrap();
    let compacted =
        apply_context(signed.clone(), &prepared, &outcome(&prepared, 0.1, 0.1)).unwrap();
    assert_eq!(compacted.messages[1], signed[1]);
    assert_eq!(compacted.stats.calls_removed, 0);
}

#[test]
fn malformed_pairings_and_stale_same_length_content_never_apply() {
    let original = transcript("read_file");
    let mut duplicate_call = original.clone();
    duplicate_call.insert(3, original[1].clone());
    let mut duplicate_result = original.clone();
    duplicate_result.insert(3, original[2].clone());
    let mut reversed = original.clone();
    reversed.swap(1, 2);
    let mut orphan = original.clone();
    orphan.remove(1);
    let mut wrong_name = original.clone();
    if let AgentMessage::Message(Message::ToolResult(result)) = &mut wrong_name[2] {
        result.tool_name = "other".into();
    }
    for messages in [
        duplicate_call,
        duplicate_result,
        reversed,
        orphan,
        wrong_name,
    ] {
        assert_eq!(
            prepare_context(&messages, &CompactionConfig::default()).unwrap_err(),
            CompactionSkip::InvalidPair
        );
    }
    let prepared = prepare_context(&original, &CompactionConfig::default()).unwrap();
    let mut changed = original.clone();
    if let AgentMessage::Message(Message::ToolResult(result)) = &mut changed[2] {
        if let ImageOrTextContent::Text(text) = &mut result.content[0] {
            text.text = text.text.replace("old", "new");
        }
    }
    assert_eq!(
        apply_context(changed, &prepared, &outcome(&prepared, 0.1, 0.1)).unwrap_err(),
        CompactionSkip::StaleContext
    );
}

#[test]
fn missing_invalid_uncertain_or_low_reduction_answers_leave_original_usable() {
    let messages = transcript("ipython");
    let original = messages.clone();
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    let mut missing = outcome(&prepared, 0.1, 0.1);
    missing.records.pop();
    for answers in [
        missing,
        outcome(&prepared, 0.5, 0.1),
        outcome(&prepared, 0.9, 0.9),
    ] {
        assert!(apply_context(messages.clone(), &prepared, &answers).is_err());
        assert_eq!(messages, original);
    }
}

#[test]
fn state_never_contains_tool_inputs_results_or_private_custom_messages() {
    let mut messages = transcript("read_file");
    messages[0] = user("password=hunter2-never-send");
    messages.insert(
        3,
        AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "memory".into(),
            content: CustomMessageContent::Text("PRIVATE_CUSTOM_MEMORY".into()),
            display: false,
            details: None,
            timestamp: 4,
        }),
    );
    if let AgentMessage::Message(Message::Assistant(assistant)) = &mut messages[1] {
        if let ContentBlock::ToolCall(call) = &mut assistant.content[0] {
            call.arguments
                .insert("password".into(), json!("TOOL_ARG_SECRET"));
        }
    }
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    let state = serde_json::to_string(&prepared.plan.state).unwrap();
    for secret in [
        "hunter2-never-send",
        "PRIVATE_CUSTOM_MEMORY",
        "TOOL_ARG_SECRET",
        "old source output",
    ] {
        assert!(!state.contains(secret));
    }
    let compacted =
        apply_context(messages.clone(), &prepared, &outcome(&prepared, 0.1, 0.1)).unwrap();
    assert_eq!(compacted.messages[0], messages[0]);
    assert!(compacted.messages.contains(&messages[3]));
}

#[test]
fn candidate_count_and_small_projection_benefit_are_bounded_before_network() {
    let mut messages = vec![user("task")];
    for i in 0..20 {
        messages.push(call(&format!("c{i}"), "read_file"));
        messages.push(result(&format!("c{i}"), "read_file", &"x".repeat(6000)));
    }
    messages.extend((0..7).map(|_| assistant("recent")));
    let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
    assert_eq!(prepared.baseline.calls_evaluated, 8);
    assert_eq!(prepared.plan.questions.len(), 16);
    let mut small = transcript("read_file");
    small[0] = user(&"constraints".repeat(20_000));
    assert_eq!(
        prepare_context(&small, &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::InsufficientReduction
    );
}

#[tokio::test]
async fn real_native_agent_provider_context_is_compacted_while_agent_history_is_unchanged() {
    let original = transcript("ipython");
    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(format!("jev-compaction-{}", uuid::Uuid::new_v4())),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let model = provider.get_model();
    provider.set_responses(vec![FauxResponseStep::Factory(Arc::new(|context, _, _, model| Box::pin(async move {
        let output = context.messages.iter().find_map(|message| match message {
            Message::ToolResult(result) => Some(result), _ => None,
        }).expect("retained tool result");
        assert!(matches!(&output.content[0], ImageOrTextContent::Text(text) if text.text.contains(TRUNCATION_MARKER)));
        assert_eq!(output.tool_call_id, "old-call");
        let mut response = faux_assistant_message(FauxAssistantContent::Text("done".into()), None);
        response.api = model.api.clone(); response.provider = model.provider.clone(); response.model = model.id.clone(); response
    })))]);
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Body(json!({
        "model":"mock", "answers": {"compaction.call_t1":{"type":"noul","noul":0.1}, "compaction.result_t1":{"type":"noul","noul":0.1}}
    }).to_string())]));
    let client = Arc::new(
        JevSystemOne::new(
            JevMode::Active,
            SecretString::new("synthetic"),
            transport,
            JevLimits::default(),
            Arc::new(JevStats::default()),
        )
        .unwrap(),
    );
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            messages: original.clone(),
            model,
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(|_| {
            Box::pin(async { Some("synthetic-faux-key".into()) })
        })),
        convert_to_llm: Some(Arc::new(|messages| {
            Box::pin(async move {
                pi_coding_agent::core::messages::convert_to_llm(&messages, &Default::default())
            })
        })),
        transform_context: Some(Arc::new(move |messages, _| {
            let client = client.clone();
            Box::pin(async move {
                let prepared = prepare_context(&messages, &CompactionConfig::default()).unwrap();
                let bundle = pi_jev::DecisionBundle {
                    session_id: "native-fixture".into(),
                    turn: 1,
                    stage: "compaction".into(),
                    state: prepared.plan.state.clone(),
                    model: "jev-latest".into(),
                    questions: prepared.plan.questions.clone(),
                    question_categories: prepared
                        .plan
                        .questions
                        .keys()
                        .map(|id| (id.clone(), DecisionCategory::ContextRelevance))
                        .collect(),
                };
                let answers = client.decide(bundle).await;
                apply_context(messages, &prepared, &answers)
                    .unwrap()
                    .messages
            })
        })),
        ..Default::default()
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent.prompt(PromptInput::Text {
            input: "Continue".into(),
            images: vec![],
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        &agent.state().messages[..original.len()],
        original.as_slice()
    );
    assert_eq!(provider.call_count(), 1);
    provider.unregister();
}

#[test]
fn instruction_and_prompt_reads_remain_verbatim() {
    for path in [
        "AGENTS.md",
        "docs/guide.md",
        "system-prompt.txt",
        "instructions.txt",
    ] {
        let mut messages = transcript("read_file");
        if let AgentMessage::Message(Message::Assistant(assistant)) = &mut messages[1] {
            if let ContentBlock::ToolCall(call) = &mut assistant.content[0] {
                call.arguments.insert("path".into(), json!(path));
            }
        }
        assert!(matches!(
            prepare_context(&messages, &CompactionConfig::default()),
            Err(CompactionSkip::NoCandidates)
        ));
    }
}

#[test]
fn incomplete_calls_and_provider_checkpoints_are_never_rewritten() {
    let mut incomplete = transcript("read_file");
    incomplete.remove(2);
    assert_eq!(
        prepare_context(&incomplete, &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::NoCandidates
    );
    let checkpoint = pi_ai::compaction::ProviderCompactionCheckpoint {
        version: 1,
        provider: "fixture".into(),
        api: "faux".into(),
        model: "fixture".into(),
        base_url: "https://invalid.test".into(),
        endpoint: None,
        items: vec![json!({"type":"message","opaque":"unchanged"})
            .as_object()
            .unwrap()
            .clone()],
        estimated_tokens: 100.0,
    };
    let mut messages = transcript("read_file");
    if let AgentMessage::Message(Message::User(user)) = &mut messages[0] {
        user.provider_context = Some(checkpoint.clone());
    }
    assert_eq!(
        prepare_context(&messages, &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::ProtectedContext
    );
    messages[0] = AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
        summary: "summary".into(),
        provider_context: Some(checkpoint),
        tokens_before: 100.0,
        retained_message_count: None,
        custom_instructions: None,
        harness_digest: None,
        timestamp: 0,
    });
    assert_eq!(
        prepare_context(&messages, &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::ProtectedContext
    );
}
