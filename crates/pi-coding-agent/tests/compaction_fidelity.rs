use pi_agent_core::types::AgentMessage;
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Message, Model, TextContent,
    UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, default_summary_call_runner, prepare_compaction,
    CompactionSessionEntry,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

const SUMMARY: &str = "## Goal\nFinish build.\n## Constraints & Preferences\nKeep user data.\n## Progress\nBuild passed.\n## Key Decisions\nReview first.\n## Next Steps\nPublish.\n## Critical Context\nUse src/main.rs.";

fn entry(id: &str, assistant: bool, text: &str) -> CompactionSessionEntry {
    let message = if assistant {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new(text))],
            ..Default::default()
        })
    } else {
        Message::User(UserMessage::new(UserContent::Text(text.into()), 0))
    };
    CompactionSessionEntry::Message {
        id: id.into(),
        parent_id: None,
        message: AgentMessage::Message(message),
    }
}

#[tokio::test]
async fn prepared_summary_uses_retained_state_and_reattaches_file_inventory_once() {
    let previous = format!("{SUMMARY}\n\n<read-files>\nold-read.rs\n</read-files>\n\n<modified-files>\nsrc/main.rs\n</modified-files>");
    let entries = vec![
        CompactionSessionEntry::Compaction {
            id: "prior".into(),
            parent_id: None,
            summary: previous,
            first_kept_entry_id: "older".into(),
            tokens_before: 2000.0,
            details: Some(json!({"readFiles":["old-read.rs"],"modifiedFiles":["src/main.rs"]})),
            from_hook: None,
            custom_instructions: None,
            harness_digest: None,
            timestamp: "".into(),
        },
        entry(
            "older",
            false,
            "Keep user data. Investigate the build failure.",
        ),
        entry("old-answer", true, "Build still fails."),
        entry("retained-user", false, "Please recheck."),
        entry(
            "retained-answer",
            true,
            &format!(
                "{}LATEST: build now passes; publish remains pending.",
                "界".repeat(3000)
            ),
        ),
    ];
    let mut settings = default_compaction_settings();
    settings.keep_recent_tokens = 1.0;
    let prep = prepare_compaction(&entries, &settings, &|_| vec![]).unwrap();
    assert_eq!(prep.previous_summary.as_deref(), Some(SUMMARY));
    let anchor = prep.retained_state_anchor.as_ref().unwrap();
    assert_eq!(anchor.chars().count(), 2000);
    assert!(anchor.ends_with("LATEST: build now passes; publish remains pending."));

    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    register_api_provider_simple(
        ApiProviderSimple {
            api: "compaction-fidelity-fixture".into(),
            compact: None,
            supports_compaction: None,
            stream: Arc::new(|_, _, _| panic!("unexpected base stream")),
            stream_simple: Arc::new(move |_, context, _| {
                let text = context
                    .messages
                    .iter()
                    .map(|message| match message {
                        Message::User(user) => user.content.text(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                seen.lock().unwrap().push(text.clone());
                let response = if text.contains("## Original Request") {
                    "## Original Request\nRecheck.\n## Early Progress\nEarlier failure.\n## Context for Suffix\nLatest build passed."
                } else {
                    SUMMARY
                };
                let message = AssistantMessage {
                    content: vec![ContentBlock::Text(TextContent::new(response))],
                    stop_reason: "stop".into(),
                    ..Default::default()
                };
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Done {
                    reason: "stop".into(),
                    message,
                });
                stream.end(None);
                stream
            }),
        },
        None,
    );
    let mut model = Model::new(
        "fixture",
        "fixture",
        "compaction-fidelity-fixture",
        "fixture",
        "http://fixture.invalid",
    );
    model.context_window = 200000.0;
    model.max_tokens = 32000.0;
    let result = compact(
        &prep,
        &model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        None,
        None,
    )
    .await
    .unwrap();
    let seen = requests.lock().unwrap();
    assert!(!seen.is_empty());
    for text in seen.iter() {
        assert!(text.contains("LATEST: build now passes; publish remains pending."));
        assert!(text.contains("assistant claims do not override them"));
        assert!(!text.contains("<read-files>"));
        assert!(!text.contains("<modified-files>"));
    }
    assert_eq!(result.summary.matches("<modified-files>").count(), 1);
    assert_eq!(result.details.unwrap().modified_files, vec!["src/main.rs"]);
}
