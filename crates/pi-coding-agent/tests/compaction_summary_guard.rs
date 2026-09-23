use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{AssistantMessage, ContentBlock, Message, Model, TextContent, UserContent, UserMessage};
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, CompactionPreparation, SummaryCallRunner,
};
use pi_coding_agent::core::compaction::utils::create_file_ops;

const VALID: &str = "## Goal\nStop safely.\n## Constraints & Preferences\nNone.\n## Progress\nStopped.\n## Key Decisions\nWait.\n## Next Steps\nReboot.\n## Critical Context\nSaved.";
const PREFIX: &str = "## Original Request\nStop.\n## Early Progress\nSaved.\n## Context for Suffix\nReady.";

fn response(text: &str, stop: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: stop.to_string(),
        ..Default::default()
    }
}

async fn summarize(message: AssistantMessage, split: bool) -> Result<String, String> {
    // No provider registry, filesystem, credentials, sockets or live session: the runner
    // substitutes a completed response at the existing per-call decoration boundary.
    let runner: SummaryCallRunner = Arc::new(move |_| {
        let message = message.clone();
        Box::pin(async move { Ok(message) })
    });
    let messages = vec![AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text("Stop and preserve this task.".to_string()), 0,
    )))];
    let preparation = CompactionPreparation {
        first_kept_entry_id: "retained-turn".to_string(),
        messages_to_summarize: if split { Vec::new() } else { messages.clone() },
        turn_prefix_messages: if split { messages } else { Vec::new() },
        is_split_turn: split,
        tokens_before: 250_000.0,
        retained_state_anchor: None, previous_summary: None,
        file_ops: create_file_ops(),
        settings: default_compaction_settings(),
    };
    let mut model = Model::new("fixture", "fixture", "faux", "faux", "https://fixture.invalid");
    model.context_window = 1_000_000.0;
    model.max_tokens = 32_000.0;
    compact(&preparation, &model, "unused", None, None, None, runner, None, None)
        .await.map(|result| result.summary)
}

#[tokio::test]
async fn summary_guard_rejects_refusal_empty_error_and_malformed_handoffs() {
    for (text, stop) in [
        ("I'm sorry, but I cannot assist with that request.", "stop"),
        ("", "stop"),
        ("Provider error: unavailable", "stop"),
        ("## Goal\nStop safely.", "stop"),
        ("## Goal\n\n## Constraints & Preferences\n\n## Progress\n\n## Key Decisions\n\n## Next Steps\n\n## Critical Context\n", "stop"),
        (VALID, "length"),
        (VALID, "error"),
        (VALID, "aborted"),
        (VALID, "toolUse"),
    ] {
        assert!(summarize(response(text, stop), false).await.is_err(), "accepted {stop}: {text}");
    }
    let mut refusal = response(VALID, "stop");
    refusal.stop_reason_raw = Some("refusal".to_string());
    assert!(summarize(refusal, false).await.is_err());
    let mut filtered = response(VALID, "stop");
    filtered.stop_reason_raw = Some("content_filter".to_string());
    assert!(summarize(filtered, false).await.is_err());
    let mut hidden_error = response(VALID, "stop");
    hidden_error.error_message = Some("provider rejected this request".to_string());
    assert!(summarize(hidden_error, false).await.is_err());
    assert!(summarize(response(&format!("{VALID}\n```\nunfinished fence"), "stop"), false).await.is_err());
}

#[tokio::test]
async fn summary_guard_preserves_short_valid_summary_and_split_turn_contract() {
    assert_eq!(summarize(response(VALID, "stop"), false).await.unwrap(), VALID);
    // A repeated section heading is a formatting slip; the handoff stays usable.
    let duplicated = format!("{VALID}\n## Goal\nDuplicated.");
    assert_eq!(summarize(response(&duplicated, "stop"), false).await.unwrap(), duplicated);
    assert!(summarize(response(PREFIX, "stop"), true).await.unwrap().contains(PREFIX));
    assert!(summarize(response(VALID, "stop"), true).await.is_err());
    assert!(summarize(response(PREFIX, "stop"), false).await.is_err());
    let formatted = VALID.replace("## ", "### **").replace("\n", "\r\n");
    let formatted = formatted.lines().map(|line| if line.starts_with("### **") {
        format!("{line}** ###")
    } else { line.to_string() }).collect::<Vec<_>>().join("\r\n");
    assert!(summarize(response(&formatted, "stop"), false).await.is_ok());
    assert!(summarize(response(&format!("```markdown\n{VALID}\n```"), "stop"), false).await.is_ok());
}
