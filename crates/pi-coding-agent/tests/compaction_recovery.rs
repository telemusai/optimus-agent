//! W5 wave-2 recovery tests: compaction correctness under continuation.
//!
//! Owner: `core/compaction/*`, `core/rlm_continuation.rs`, `core/rlm_max_depth.rs`
//! (audit W5, wave-2 fix list items 3a-3e). The scenarios drive the REAL
//! compaction pipeline (`prepare_compaction` -> `compact` -> `append_compaction`
//! -> `build_session_context_with_entry_ids`) over the REAL `SessionManager`
//! JSONL persistence, plus two full `AgentSession` end-to-end runs on the faux
//! provider that exercise the auto-compaction trigger itself.
//!
//! Coverage map (audit W5 test plan items 12/13 plus the wave-2 brief):
//! - (a) near-limit auto-compaction triggers and the compacted context retains
//!   task goals, outstanding constraints, subagent identity and evidence
//!   references - `near_limit_auto_compaction_*`,
//!   `agent_session_auto_compaction_end_to_end`.
//! - (b) manual compaction mid-session then continue -
//!   `manual_compaction_mid_session_then_continue`.
//! - (c) restart after compaction (persist, reopen, continue), including the
//!   A7 `deadlineFallback` entry shape - `restart_after_compaction_*`.
//! - (d) tool-output compaction preserves tool result identity/count -
//!   `tool_output_compaction_*`.
//! - (e) compaction while a subagent is active does not lose the child's
//!   pending result - `compaction_with_active_subagent_*`,
//!   `agent_session_rlm_child_*`.
//!
//! No network, no real provider: summary wire calls go to a fixture API
//! registered through `register_api_provider_simple`; agent turns go to the
//! faux provider. All writable state stays under an isolated temp root.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, CustomMessageContent};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Message, Model, TextContent, ToolCall,
    ToolResultMessage, Usage, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, default_summary_call_runner, estimate_context_tokens,
    find_cut_point, prepare_compaction, should_compact_for_model, CompactionSessionEntry,
    CompactionSettings,
};
use pi_coding_agent::core::rlm_continuation::{
    parse_rlm_continuation_state, RlmContinuationState, RLM_CONTINUATION_STATE_CUSTOM_TYPE,
};
use pi_coding_agent::core::session_manager::{
    CustomMessageEntryContent, NewSessionOptions, SessionEntry, SessionManager,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// V00 isolation harness (same contract as transcript_t06.rs)
// ---------------------------------------------------------------------------

fn state_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let root = std::env::var_os("OPTIMUS_PARITY_STATE_ROOT")
        .map(|raw| {
            assert!(
                !raw.is_empty(),
                "OPTIMUS_PARITY_STATE_ROOT must not be empty"
            );
            std::path::absolute(raw).expect("absolute state root")
        })
        .unwrap_or_else(|| {
            ROOT.get_or_init(|| {
                tempfile::Builder::new()
                    .prefix("optimus-compaction-recovery-")
                    .tempdir()
                    .expect("private state root")
            })
            .path()
            .to_path_buf()
        });
    let lowered = root.to_string_lossy().to_lowercase();
    assert!(
        !lowered.contains(".prime"),
        "state root must not touch .prime"
    );
    std::fs::create_dir_all(&root).expect("create state root");
    root
}

fn case_dir(tag: &str) -> PathBuf {
    let dir = state_root()
        .join(tag)
        .join(format!("{}-{}", std::process::id(), unique_tag()));
    std::fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn unique_tag() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static TAG_SEQ: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{}-{}", nanos, TAG_SEQ.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// A summary body that satisfies `validate_summary` (all six sections, each
/// with content) so the fixture summarizer is a usable handoff.
const VALID_SUMMARY: &str = "## Goal\nFinish the migration.\n## Constraints & Preferences\nKeep user data.\n## Progress\n### Done\n- [x] Audit read.\n\n### In Progress\n- [ ] Section 3.\n\n### Blocked\n- None.\n\n## Key Decisions\n- Incremental rollout.\n## Next Steps\n1. Publish the report.\n## Critical Context\nSee reports/audit-2026.md#findings.";

fn fixture_model(api: &str, context_window: f64, max_tokens: f64) -> Model {
    let mut model = Model::new(api, api, api, "fixture", "https://fixture.invalid");
    model.context_window = context_window;
    model.max_tokens = max_tokens;
    model
}

/// Registers a fixture API whose `stream_simple` captures the full text of
/// every request's first user message and answers with `VALID_SUMMARY`.
/// Returns the model bound to the fixture plus the captured requests.
fn summarizer_fixture(api: &str) -> (Model, Arc<Mutex<Vec<String>>>) {
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&captured);
    register_api_provider_simple(
        ApiProviderSimple {
            api: api.into(),
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
                seen.lock().unwrap().push(text);
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Done {
                    reason: "stop".into(),
                    message: AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent::new(VALID_SUMMARY))],
                        stop_reason: "stop".into(),
                        ..Default::default()
                    },
                });
                stream
            }),
            compact: None,
            supports_compaction: None,
        },
        None,
    );
    (fixture_model(api, 1_000_000.0, 64_000.0), captured)
}

/// Mirrors the production converter (`compaction_session_entry_from`,
/// core/agent_session.rs): a session-file entry becomes the
/// `CompactionSessionEntry` the compaction kernel consumes.
fn compaction_entry(entry: &SessionEntry) -> Option<CompactionSessionEntry> {
    let entry_type = entry.get("type").and_then(Value::as_str)?;
    let id = entry.get("id").and_then(Value::as_str)?.to_string();
    let parent_id = entry
        .get("parentId")
        .and_then(Value::as_str)
        .map(str::to_string);
    match entry_type {
        "message" => Some(CompactionSessionEntry::Message {
            id,
            parent_id,
            message: serde_json::from_value::<AgentMessage>(
                entry.get("message").cloned().unwrap_or(Value::Null),
            )
            .ok()?,
        }),
        "custom_message" => Some(CompactionSessionEntry::CustomMessage {
            id,
            parent_id,
            custom_type: entry
                .get("customType")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content: serde_json::from_value(entry.get("content").cloned().unwrap_or(Value::Null))
                .unwrap_or(CustomMessageContent::Text(String::new())),
            details: entry.get("details").cloned(),
            display: entry
                .get("display")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            timestamp: entry
                .get("timestamp")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_default(),
        }),
        "compaction" => Some(CompactionSessionEntry::Compaction {
            id,
            parent_id,
            summary: entry
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            first_kept_entry_id: entry
                .get("firstKeptEntryId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            tokens_before: entry
                .get("tokensBefore")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            details: entry.get("details").cloned(),
            from_hook: entry.get("fromHook").and_then(Value::as_bool),
            custom_instructions: entry
                .get("customInstructions")
                .and_then(Value::as_str)
                .map(str::to_string),
            harness_digest: entry
                .get("harnessDigest")
                .and_then(Value::as_str)
                .map(str::to_string),
            timestamp: entry
                .get("timestamp")
                .map(|value| value.to_string())
                .unwrap_or_default(),
        }),
        _ => Some(CompactionSessionEntry::Other {
            id,
            parent_id,
            entry_type: entry_type.to_string(),
        }),
    }
}

/// The full branch as compaction sees it, in path order.
fn path_entries(branch: &[SessionEntry]) -> Vec<CompactionSessionEntry> {
    branch.iter().filter_map(compaction_entry).collect()
}

/// The live-context analog of `agent.messages_without_harness_digests()`: the
/// custom entries become their custom agent messages, exactly like
/// `get_message_from_entry_for_compaction`.
fn live_messages(branch: &[SessionEntry]) -> Vec<AgentMessage> {
    branch
        .iter()
        .filter_map(compaction_entry)
        .filter_map(|entry| match entry {
            CompactionSessionEntry::Compaction { .. } => None,
            CompactionSessionEntry::Message { message, .. } => Some(message),
            CompactionSessionEntry::CustomMessage {
                custom_type,
                content,
                details,
                display,
                ..
            } => Some(AgentMessage::Custom(
                pi_agent_core::types::CustomAgentMessage::Custom {
                    custom_type,
                    content,
                    display,
                    details,
                    timestamp: 0,
                },
            )),
            _ => None,
        })
        .collect()
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text(text.into()),
        0,
    )))
}

fn assistant_message(text: &str, usage_input: f64) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: "stop".into(),
        usage: Usage {
            input: usage_input,
            ..Default::default()
        },
        ..Default::default()
    }))
}

fn compaction_settings(reserve_tokens: f64, keep_recent_tokens: f64) -> CompactionSettings {
    let mut settings = default_compaction_settings();
    settings.reserve_tokens = reserve_tokens;
    settings.keep_recent_tokens = keep_recent_tokens;
    settings
}

/// Appends one durable turn (user + assistant with a chosen usage anchor).
fn seed_turn(
    manager: &mut SessionManager,
    user_text: &str,
    assistant_text: &str,
    usage_input: f64,
) -> (String, String) {
    let user_id = manager.append_message(user_message(user_text)).unwrap();
    let assistant_id = manager
        .append_message(assistant_message(assistant_text, usage_input))
        .unwrap();
    (user_id, assistant_id)
}

fn tool_call(id: &str, name: &str, path: &str) -> ContentBlock {
    let mut arguments = serde_json::Map::new();
    arguments.insert("path".into(), Value::String(path.into()));
    ContentBlock::ToolCall(ToolCall::new(id, name, arguments))
}

fn tool_result(id: &str, name: &str, text: &str) -> AgentMessage {
    AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
        id,
        name,
        vec![pi_ai::types::ImageOrTextContent::Text(TextContent::new(
            text,
        ))],
        false,
        0,
    )))
}

fn text_of(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Message(Message::User(user)) => user.content.text(),
        AgentMessage::Message(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn tool_call_ids(message: &AgentMessage) -> Vec<String> {
    let AgentMessage::Message(Message::Assistant(assistant)) = message else {
        return Vec::new();
    };
    assistant
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call.id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(message: &AgentMessage) -> Vec<String> {
    let AgentMessage::Message(Message::ToolResult(result)) = message else {
        return Vec::new();
    };
    vec![result.tool_call_id.clone()]
}

/// The compaction entry appended last is the one the context must lead with.
fn branch_compaction_id(manager: &SessionManager) -> String {
    let branch = manager.get_branch(None);
    branch
        .iter()
        .rev()
        .find(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .and_then(|entry| entry.get("id").and_then(Value::as_str))
        .expect("compaction entry")
        .to_string()
}

// ---------------------------------------------------------------------------
// (a) Near-limit auto-compaction retains goals, constraints, subagent identity
//     and evidence references
// ---------------------------------------------------------------------------

#[tokio::test]
async fn near_limit_auto_compaction_retains_goals_constraints_subagent_and_evidence() {
    let dir = case_dir("near-limit");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();

    let goal = "TASK GOAL: migrate the billing service to API v2 before Friday.";
    let constraint = "CONSTRAINT: never delete production data; dry-run every cutover first.";
    let evidence = "Evidence reference: reports/audit-2026.md#findings (line 40).";
    let subagent = "Subagent reviewer-1 owns section 3 of the migration.";
    let child_report = "Subagent reviewer-1 finished section 3; artifact evidence/reviewer-1.json";

    // Turn 1 (summarized region): the goal, constraint, evidence reference and
    // subagent identity all live here. The usage anchor is the LAST
    // assistant, so this turn carries a small usage; the boundary-crossing
    // usage belongs to turn 2's assistant below.
    //
    // The child report lands MID-TURN, between the user request and the
    // parent's reply: agent messages are admitted asynchronously, and that is
    // the durable steering shape. Placement also pins the cut-point contract:
    // `find_cut_point` pulls a cut that would land directly after a custom
    // entry back ONTO it, so an entry immediately before the retained turn
    // would be dragged into the tail instead of being summarized.
    let user1_id = manager
        .append_message(user_message(&format!(
            "{goal}\n{constraint}\n{evidence}\n{subagent}"
        )))
        .unwrap();
    manager
        .append_custom_message_entry(
            "agent_message",
            &CustomMessageEntryContent::Text(child_report.into()),
            true,
            Some(json!({
                "id": "agentmsg-child-1",
                "message": child_report,
                "fromRelationship": "child",
                "fromName": "reviewer-1"
            })),
        )
        .unwrap();
    manager
        .append_message(assistant_message("Started the migration plan.", 12.0))
        .unwrap();

    // Turn 2 (retained tail): a large user message so the keep-recent walk
    // stops here and the cut lands on the user message, plus the assistant
    // reply that becomes both the usage anchor (17_000 crosses the boundary)
    // and the retained-state anchor for the summary instructions.
    let (user2_id, assistant2_id) = seed_turn(
        &mut manager,
        &format!(
            "Continue the migration. {}",
            "status: section 3 done. ".repeat(120)
        ),
        "Section 3 verified; publishing the report next.",
        17_000.0,
    );

    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let live = live_messages(&branch);

    let model = fixture_model("recovery-near-limit", 20_000.0, 8_000.0);
    let settings = compaction_settings(4_000.0, 20.0);

    // Near-limit trigger: the boundary is min(cap, window - reserve) = 16_000.
    let tokens = estimate_context_tokens(&live).tokens;
    assert!(
        tokens >= 17_000.0,
        "usage-anchored estimate {tokens} must sit above the boundary"
    );
    assert!(!should_compact_for_model(15_999.0, &model, &settings));
    assert!(should_compact_for_model(16_000.0, &model, &settings));
    assert!(should_compact_for_model(tokens, &model, &settings));

    let preparation = prepare_compaction(&entries, &settings, &|_| tokens)
        .expect("history exists below the boundary, so a preparation must resolve");
    assert_eq!(preparation.first_kept_entry_id, user2_id);
    assert!(!preparation.is_split_turn, "cut lands on the user turn");
    assert!(
        preparation.messages_to_summarize.iter().any(|message| {
            matches!(message, AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { custom_type, .. })
                if custom_type == "agent_message")
        }),
        "the subagent report must be summarized, not silently dropped"
    );

    let (summary_model, captured) = summarizer_fixture("recovery-near-limit-summary");
    let result = compact(
        &preparation,
        &summary_model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        None,
        None,
    )
    .await
    .expect("fixture summarizer succeeds");

    // The summarizer request must carry every retention anchor: task goal,
    // outstanding constraint, evidence reference, subagent identity and the
    // subagent's own report content.
    let conversation = captured.lock().unwrap().join("\n<chunk>\n");
    assert!(
        conversation.contains(goal),
        "task goal missing from summary input"
    );
    assert!(
        conversation.contains(constraint),
        "constraint missing from summary input"
    );
    assert!(
        conversation.contains("reports/audit-2026.md#findings"),
        "evidence reference missing from summary input"
    );
    assert!(
        conversation.contains("reviewer-1"),
        "subagent identity missing"
    );
    assert!(
        conversation.contains("evidence/reviewer-1.json"),
        "subagent report artifact missing"
    );
    assert!(
        conversation.contains("<retained-state>"),
        "retained-state anchor must be attached to the summary instructions"
    );
    assert!(
        conversation.contains("Section 3 verified; publishing the report next."),
        "retained-state anchor must carry the kept assistant state"
    );

    // Commit the compaction and rebuild the model context.
    let details = result
        .details
        .as_ref()
        .and_then(|details| serde_json::to_value(details).ok());
    manager
        .append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            result.tokens_before,
            details,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    manager.flush_now().unwrap();

    let context = manager.build_session_context_with_entry_ids(None);
    assert_eq!(context.messages.len(), 3, "summary + user2 + assistant2");
    match &context.messages[0] {
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::CompactionSummary {
            summary,
            ..
        }) => {
            assert_eq!(summary, &result.summary);
        }
        other => panic!("context must be summary-first, got {:?}", other.role()),
    }
    assert_eq!(
        context.entry_ids,
        vec![
            branch_compaction_id(&manager),
            user2_id.clone(),
            assistant2_id.clone(),
        ]
    );
    // The retained assistant still carries its stale pre-compaction usage
    // (17_000 input), so a raw estimate over the compacted context would stay
    // above the boundary; production fences that stale anchor with the
    // compaction timestamp (`threshold_context_tokens_from_state`) and the
    // next assistant reply re-anchors the estimate. Continue the session and
    // verify the re-anchored estimate drops below the boundary again.
    let (user3_id, assistant3_id) = seed_turn(&mut manager, "Final check.", "All clear.", 100.0);
    let continued = manager.build_session_context_with_entry_ids(None);
    assert_eq!(continued.messages.len(), 5, "summary + turn2 + turn3");
    let after = estimate_context_tokens(&continued.messages).tokens;
    assert!(
        after < 16_000.0,
        "post-compaction anchored estimate {after} must be below the boundary"
    );
    assert!(!context.entry_ids.contains(&user1_id));
    assert!(manager.get_entry(&user1_id).is_some());
    assert!(continued.entry_ids.contains(&user3_id));
    assert!(continued.entry_ids.contains(&assistant3_id));
}

// ---------------------------------------------------------------------------
// (b) Manual compaction mid-session, then continue
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manual_compaction_mid_session_then_continue() {
    let dir = case_dir("manual-continue");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();

    let (user1_id, assistant1_id) = seed_turn(
        &mut manager,
        "TASK GOAL: stabilize the parser. CONSTRAINT: keep the public API unchanged.",
        "Reproduced the failure.",
        17_000.0,
    );
    let (user2_id, assistant2_id) = seed_turn(
        &mut manager,
        &format!("Continue. {}", "context: parser fixed. ".repeat(100)),
        "Parser fix verified.",
        12.0,
    );

    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let live = live_messages(&branch);
    let tokens = estimate_context_tokens(&live).tokens;

    let settings = compaction_settings(4_000.0, 20.0);
    let preparation = prepare_compaction(&entries, &settings, &|_| tokens).unwrap();
    assert_eq!(preparation.first_kept_entry_id, user2_id);

    let (model, _captured) = summarizer_fixture("recovery-manual-summary");
    let instructions = "Focus: preserve the constraint and the parser acceptance evidence.";
    let result = compact(
        &preparation,
        &model,
        "unused",
        Some(instructions),
        None,
        None,
        default_summary_call_runner(None),
        None,
        None,
    )
    .await
    .unwrap();

    let details = result
        .details
        .as_ref()
        .and_then(|details| serde_json::to_value(details).ok());
    manager
        .append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            result.tokens_before,
            details,
            None,
            Some(instructions),
            None,
            None,
        )
        .unwrap();
    manager.flush_now().unwrap();

    // Mid-session continue: a fresh turn lands after the compaction.
    let (user3_id, assistant3_id) = seed_turn(&mut manager, "Final check.", "All clear.", 5.0);
    let context = manager.build_session_context_with_entry_ids(None);
    assert_eq!(context.messages.len(), 5, "summary + turn2 + turn3");
    match &context.messages[0] {
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::CompactionSummary {
            summary,
            custom_instructions,
            ..
        }) => {
            assert_eq!(summary, &result.summary);
            assert_eq!(custom_instructions.as_deref(), Some(instructions));
        }
        other => panic!("context must be summary-first, got {:?}", other.role()),
    }
    assert_eq!(
        context.entry_ids,
        vec![
            branch_compaction_id(&manager),
            user2_id.clone(),
            assistant2_id.clone(),
            user3_id.clone(),
            assistant3_id.clone(),
        ]
    );
    // The summarized turn is gone from the model context but stays durable.
    assert!(!context.entry_ids.contains(&user1_id));
    assert!(!context.entry_ids.contains(&assistant1_id));
    assert!(manager.get_entry(&user1_id).is_some());
    assert!(manager.get_entry(&assistant1_id).is_some());
    assert_eq!(text_of(&context.messages[3]), "Final check.");
    assert_eq!(text_of(&context.messages[4]), "All clear.");
}

// ---------------------------------------------------------------------------
// (c) Restart after compaction: persist, reopen, continue
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restart_after_compaction_rebuilds_identical_context() {
    let dir = case_dir("restart");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();

    seed_turn(
        &mut manager,
        "TASK GOAL: verify restart recovery.",
        "First pass done.",
        17_000.0,
    );
    let (user2_id, _) = seed_turn(
        &mut manager,
        &format!("Continue. {}", "restart: retained. ".repeat(100)),
        "Retained tail verified.",
        12.0,
    );

    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let live = live_messages(&branch);
    let tokens = estimate_context_tokens(&live).tokens;
    let settings = compaction_settings(4_000.0, 20.0);
    let preparation = prepare_compaction(&entries, &settings, &|_| tokens).unwrap();
    assert_eq!(preparation.first_kept_entry_id, user2_id);

    let (model, _captured) = summarizer_fixture("recovery-restart-summary");
    let result = compact(
        &preparation,
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
    let details = result
        .details
        .as_ref()
        .and_then(|details| serde_json::to_value(details).ok());
    manager
        .append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            result.tokens_before,
            details,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    // Continue after the compaction, then persist everything.
    seed_turn(&mut manager, "Post-compaction turn.", "Committed.", 5.0);
    manager.flush_now().unwrap();

    let before = manager.build_session_context_with_entry_ids(None);
    let session_file = manager.get_session_file().expect("durable session file");
    drop(manager);

    // Restart 1: reopen the persisted file.
    let mut reopened = SessionManager::open(&session_file, None, None).unwrap();
    let after = reopened.build_session_context_with_entry_ids(None);
    assert_eq!(
        before.messages, after.messages,
        "reopened context must equal the pre-restart context"
    );
    assert_eq!(before.entry_ids, after.entry_ids);
    // 5 messages: summary + user2 + assistant2 + user3 + assistant3.
    assert_eq!(after.messages.len(), 5, "summary + turn2 + turn3");

    // Restart 2: a second reopen after another flush is idempotent.
    reopened.flush_now().unwrap();
    let session_file_again = reopened.get_session_file().unwrap();
    assert_eq!(session_file_again, session_file);
    drop(reopened);
    let reopened_again = SessionManager::open(&session_file, None, None).unwrap();
    let after_again = reopened_again.build_session_context_with_entry_ids(None);
    assert_eq!(after_again.messages, after.messages);
    assert_eq!(after_again.entry_ids, after.entry_ids);
}

/// W5 test-plan item 12: a `deadlineFallback: true` compaction entry (A7
/// keep-recent-tail truncation) must reopen into the same summary-first
/// context shape, and the fallback flag must survive the round trip.
#[tokio::test]
async fn restart_after_deadline_fallback_compaction_keeps_summary_first_context() {
    let dir = case_dir("deadline-fallback");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();
    seed_turn(
        &mut manager,
        "TASK GOAL: survive a compaction deadline.",
        "Slow work.",
        17_000.0,
    );
    let (user2_id, _) = seed_turn(
        &mut manager,
        &format!("Continue. {}", "deadline: retained. ".repeat(100)),
        "Retained tail.",
        12.0,
    );
    let fallback_summary = "## Goal\nCompaction deadline fallback: truncated to the retained tail.\n## Constraints & Preferences\n- See the retained recent tail.\n## Progress\n### Done\n- [x] History dropped after the deadline.\n\n### In Progress\n- [ ] Continue from the tail.\n\n### Blocked\n- Summary generation exceeded the deadline.\n\n## Key Decisions\n- **Keep-recent-tail truncation**: applied.\n## Next Steps\n1. Continue the work.\n## Critical Context\n- The transcript keeps the full record.";
    manager
        .append_compaction(
            fallback_summary,
            &user2_id,
            17_000.0,
            Some(json!({"deadlineFallback": true})),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    seed_turn(&mut manager, "After the fallback.", "Committed.", 5.0);
    manager.flush_now().unwrap();

    let session_file = manager.get_session_file().unwrap();
    let before = manager.build_session_context_with_entry_ids(None);
    drop(manager);

    let reopened = SessionManager::open(&session_file, None, None).unwrap();
    let after = reopened.build_session_context_with_entry_ids(None);
    assert_eq!(before.messages, after.messages);
    assert_eq!(before.entry_ids, after.entry_ids);
    // Summary-first: the fallback summary leads the model context.
    match &after.messages[0] {
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::CompactionSummary {
            summary,
            ..
        }) => assert_eq!(summary, fallback_summary),
        other => panic!(
            "deadline fallback must stay summary-first, got {:?}",
            other.role()
        ),
    }
    // The flag round-trips through the JSONL entry.
    let branch = reopened.get_branch(None);
    let compaction = branch
        .iter()
        .rev()
        .find(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .unwrap();
    assert_eq!(
        compaction["details"]["deadlineFallback"],
        Value::Bool(true),
        "A7 deadlineFallback flag must survive persist + reopen"
    );
}

// ---------------------------------------------------------------------------
// (d) Tool-output compaction preserves tool result identity and count
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_output_compaction_preserves_result_identity_and_count() {
    let dir = case_dir("tool-identity");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();

    // Turn 1 (summarized): one assistant turn with three tool calls, each
    // answered by exactly one result. One result is oversized so the
    // summarizer-side truncation path also runs.
    manager
        .append_message(user_message("Inspect the build."))
        .unwrap();
    let assistant1 = AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![
            tool_call("call-a", "read_file", "src/a.rs"),
            tool_call("call-b", "read_file", "src/b.rs"),
            tool_call("call-c", "bash", "run_checks.sh"),
        ],
        stop_reason: "toolUse".into(),
        ..Default::default()
    }));
    manager.append_message(assistant1).unwrap();
    manager
        .append_message(tool_result(
            "call-a",
            "read_file",
            &"contents of a; ".repeat(40),
        ))
        .unwrap();
    manager
        .append_message(tool_result(
            "call-b",
            "read_file",
            &"contents of b; ".repeat(40),
        ))
        .unwrap();
    manager
        .append_message(tool_result(
            "call-c",
            "bash",
            &"exit 0; checks passed; ".repeat(400),
        ))
        .unwrap();

    // Turn 2 (retained): assistant with two calls, both answered.
    let (user2_id, _) = seed_turn(
        &mut manager,
        "Now fix the failing test.",
        "Working on it.",
        12.0,
    );
    let assistant2 = AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![
            tool_call("call-d", "edit", "src/c.rs"),
            tool_call("call-e", "read_file", "src/d.rs"),
        ],
        stop_reason: "toolUse".into(),
        ..Default::default()
    }));
    let assistant2_id = manager.append_message(assistant2).unwrap();
    manager
        .append_message(tool_result("call-d", "edit", "edited src/c.rs"))
        .unwrap();
    manager
        .append_message(tool_result("call-e", "read_file", "read src/d.rs"))
        .unwrap();
    // A trailing user turn large enough that the walk-back crosses
    // keepRecentTokens inside turn 1's results, so the cut resolves to user2.
    manager
        .append_message(user_message(&format!(
            "Finish up. {}",
            "final: waiting. ".repeat(120)
        )))
        .unwrap();

    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let live = live_messages(&branch);
    let tokens = estimate_context_tokens(&live).tokens;
    let settings = compaction_settings(4_000.0, 1_000.0);
    let preparation = prepare_compaction(&entries, &settings, &|_| tokens).unwrap();
    assert_eq!(preparation.first_kept_entry_id, user2_id);

    // The summarized region keeps tool result identity: three distinct results,
    // each exactly once, matching the three calls.
    let summarized_results: Vec<String> = preparation
        .messages_to_summarize
        .iter()
        .flat_map(tool_result_ids)
        .collect();
    assert_eq!(
        summarized_results.len(),
        3,
        "no duplicated or missing results"
    );
    let mut unique = summarized_results.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 3);
    let summarized_calls: Vec<String> = preparation
        .messages_to_summarize
        .iter()
        .flat_map(tool_call_ids)
        .collect();
    assert_eq!(summarized_calls.len(), 3);
    for id in ["call-a", "call-b", "call-c"] {
        assert!(summarized_calls.contains(&id.to_string()), "call {id} kept");
        assert_eq!(
            summarized_results
                .iter()
                .filter(|result| result == &id)
                .count(),
            1,
            "result {id} appears exactly once"
        );
    }

    let (model, captured) = summarizer_fixture("recovery-tool-summary");
    let result = compact(
        &preparation,
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
    let conversation = captured.lock().unwrap().join("\n<chunk>\n");
    // Every result is serialized exactly once, with a matched call number.
    assert_eq!(
        conversation.matches("[Tool result #1 read_file]").count(),
        1
    );
    assert_eq!(
        conversation.matches("[Tool result #2 read_file]").count(),
        1
    );
    assert_eq!(conversation.matches("[Tool result #3 bash]").count(), 1);
    assert!(
        !conversation.contains("call outside excerpt"),
        "a result whose call is in the same excerpt must never render unmatched"
    );
    assert!(
        conversation.contains("characters truncated"),
        "the oversized result is truncated for the summary, not dropped or duplicated"
    );

    // Commit and rebuild: the retained region keeps call/result pairs intact.
    let details = result
        .details
        .as_ref()
        .and_then(|details| serde_json::to_value(details).ok());
    manager
        .append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            result.tokens_before,
            details,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    manager.flush_now().unwrap();
    let context = manager.build_session_context_with_entry_ids(None);

    let context_calls: Vec<String> = context.messages.iter().flat_map(tool_call_ids).collect();
    let context_results: Vec<String> = context.messages.iter().flat_map(tool_result_ids).collect();
    assert_eq!(
        context_calls,
        vec!["call-d".to_string(), "call-e".to_string()]
    );
    assert_eq!(
        context_results,
        vec!["call-d".to_string(), "call-e".to_string()]
    );
    // No summarized results leak into the compacted context, and no orphan
    // results (a result whose call was summarized away) remain.
    for summarized in ["call-a", "call-b", "call-c"] {
        assert!(!context_results.contains(&summarized.to_string()));
        assert!(!context_calls.contains(&summarized.to_string()));
    }
    assert!(context.entry_ids.contains(&assistant2_id));
    assert_eq!(context.messages[0].role(), "compactionSummary");
}

/// The cut point must never land on a tool result: for every keep-recent
/// budget in a sweep, the first kept entry is a turn boundary (or the
/// assistant whose results follow it), never a bare toolResult.
#[test]
fn cut_points_never_land_on_a_tool_result_across_budgets() {
    let mut entries: Vec<CompactionSessionEntry> = Vec::new();
    let mut next = 0usize;
    let mut push_message = |entries: &mut Vec<CompactionSessionEntry>, message: AgentMessage| {
        next += 1;
        entries.push(CompactionSessionEntry::Message {
            id: format!("e{next}"),
            parent_id: None,
            message,
        });
    };
    push_message(&mut entries, user_message("Inspect."));
    push_message(
        &mut entries,
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![
                tool_call("c1", "read_file", "a.rs"),
                tool_call("c2", "bash", "run.sh"),
            ],
            stop_reason: "toolUse".into(),
            ..Default::default()
        })),
    );
    push_message(&mut entries, tool_result("c1", "read_file", "a"));
    push_message(&mut entries, tool_result("c2", "bash", "b"));
    push_message(
        &mut entries,
        user_message(&format!("Long turn. {}", "filler. ".repeat(80))),
    );
    push_message(
        &mut entries,
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![tool_call("c3", "edit", "c.rs")],
            stop_reason: "toolUse".into(),
            ..Default::default()
        })),
    );
    push_message(&mut entries, tool_result("c3", "edit", "c"));
    push_message(&mut entries, user_message("Wrap up."));

    for keep_recent in (1..=400usize).step_by(7) {
        let cut = find_cut_point(&entries, 0, entries.len(), keep_recent as f64);
        let first_kept = &entries[cut.first_kept_entry_index];
        let CompactionSessionEntry::Message { message, .. } = first_kept else {
            continue;
        };
        assert_ne!(
            message.role(),
            "toolResult",
            "keepRecent={keep_recent} cut at a tool result"
        );
        // A cut at an assistant with tool calls keeps its results by
        // construction: they follow it in the entry list.
        if message.role() == "assistant" {
            let calls = tool_call_ids(message);
            if !calls.is_empty() {
                let kept: Vec<CompactionSessionEntry> =
                    entries[cut.first_kept_entry_index..].to_vec();
                for id in calls {
                    let matched = kept.iter().any(|entry| {
                        matches!(entry,
                            CompactionSessionEntry::Message { message, .. }
                                if tool_result_ids(message) == vec![id.clone()])
                    });
                    assert!(matched, "cut orphans result {id}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (e) Compaction while a subagent is active keeps the child's pending result
// ---------------------------------------------------------------------------

/// The durable continuation state an active RLM child carries while it is
/// mid-task: one unreplied parent task plus a queued length-recovery
/// continuation.
fn active_child_state() -> Value {
    json!({
        "version": 1,
        "tasks": [{
            "id": "agentmsg-parent-1",
            "receivedAt": 100.0,
            "replied": false
        }],
        "continuationCount": 1,
        "lastStopReason": "length",
        "lastSourceKey": "20:length:7:partial",
        "taskHadLength": true,
        "pendingContinuation": {
            "sourceKey": "20:length:7:partial",
            "attempt": 1,
            "previousStopReason": "length",
            "messageTimestamp": 30,
            "messageText": "RLM child length recovery 1/3.",
            "phase": "queued"
        }
    })
}

/// The state after the child finished but the result was not delivered yet:
/// the pending result is retained on the task and as the global pending
/// result, exactly where `record_rlm_terminal_result` leaves it.
fn terminal_child_state() -> Value {
    json!({
        "version": 1,
        "tasks": [{
            "id": "agentmsg-parent-1",
            "receivedAt": 100.0,
            "replied": false,
            "result": {
                "status": "complete",
                "text": "Section 3 fixed; report pending",
                "partial": true,
                "stopReason": "length"
            },
            "resultDelivery": {
                "attemptId": "attempt-1",
                "reason": "acknowledgement_pending"
            }
        }],
        "continuationCount": 2,
        "lastStopReason": "length",
        "terminalStatus": "complete",
        "taskHadLength": true,
        "pendingResult": {
            "status": "complete",
            "text": "Section 3 fixed; report pending",
            "partial": true,
            "stopReason": "length"
        }
    })
}

#[tokio::test]
async fn compaction_with_active_subagent_preserves_the_pending_result() {
    let dir = case_dir("subagent-pending");
    let dir_string = dir.to_string_lossy().to_string();
    let mut manager = SessionManager::create(&dir_string, Some(&dir_string)).unwrap();
    // A child session: the header records rlmDepth 1.
    manager
        .new_session(Some(&NewSessionOptions {
            rlm_depth: Some(1),
            ..Default::default()
        }))
        .unwrap();

    // Turn 1 (summarized region): earlier work the compaction folds away.
    seed_turn(
        &mut manager,
        "Earlier: audit the billing migration.",
        "Audit started.",
        12.0,
    );
    // Turn 2 (retained tail): the parent task is in the conversation, and the
    // child is mid-work when the threshold compaction fires. The assistant's
    // 17_000 usage anchors the estimate above the boundary.
    let (user2_id, _) = seed_turn(
        &mut manager,
        &format!(
            "TASK GOAL: fix the billing migration. CONSTRAINT: no downtime. {}",
            "work: section 3 in progress. ".repeat(100)
        ),
        "Section 3 in progress.",
        17_000.0,
    );
    // The in-flight continuation ledger, persisted exactly the way
    // `persist_rlm_continuation_state` writes it (custom entry at the leaf).
    manager
        .append_custom_entry(
            RLM_CONTINUATION_STATE_CUSTOM_TYPE,
            Some(active_child_state()),
        )
        .unwrap();

    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let live = live_messages(&branch);
    let tokens = estimate_context_tokens(&live).tokens;
    let model = fixture_model("recovery-child-boundary", 20_000.0, 8_000.0);
    let settings = compaction_settings(4_000.0, 20.0);
    assert!(should_compact_for_model(tokens, &model, &settings));

    let preparation = prepare_compaction(&entries, &settings, &|_| tokens).unwrap();
    assert_eq!(preparation.first_kept_entry_id, user2_id);
    // The ledger lives in a `type: "custom"` entry: `compaction_session_entry_from`
    // maps it to `Other`, so it is never summarizer input and never model
    // context. Its durability is branch-level, which the restart checks below
    // verify end to end.
    assert!(
        !preparation
            .messages_to_summarize
            .iter()
            .any(|message| matches!(message,
                AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { custom_type, .. })
                    if custom_type == RLM_CONTINUATION_STATE_CUSTOM_TYPE)),
        "the continuation ledger is bookkeeping, not conversation"
    );

    let (summary_model, _captured) = summarizer_fixture("recovery-child-summary");
    let result = compact(
        &preparation,
        &summary_model,
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
    let details = result
        .details
        .as_ref()
        .and_then(|details| serde_json::to_value(details).ok());
    manager
        .append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            result.tokens_before,
            details,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // The production recovery path re-persists the ledger after compaction
    // (`run_auto_compaction` -> `queue_pending_rlm_continuation` ->
    // `persist_rlm_continuation_state`): a fresh custom entry lands at the
    // leaf, now carrying the compaction reason.
    let mut recovered = active_child_state();
    recovered["compactionReason"] = json!("threshold");
    manager
        .append_custom_entry(RLM_CONTINUATION_STATE_CUSTOM_TYPE, Some(recovered.clone()))
        .unwrap();
    manager.flush_now().unwrap();

    let session_file = manager.get_session_file().unwrap();
    drop(manager);

    // Restart: reopen and restore the ledger the way
    // `restore_rlm_continuation_state` does - last matching custom entry on
    // the branch wins.
    let reopened = SessionManager::open(&session_file, None, None).unwrap();
    let branch = reopened.get_branch(None);
    let last_state = branch
        .iter()
        .rev()
        .find(|entry| {
            entry.get("type").and_then(Value::as_str) == Some("custom")
                && entry.get("customType").and_then(Value::as_str)
                    == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
        })
        .expect("continuation ledger survives compaction + restart");
    let state =
        parse_rlm_continuation_state(&last_state["data"]).expect("the restored ledger parses");
    let task = &state.tasks[0];
    assert_eq!(task.id, "agentmsg-parent-1");
    assert!(!task.replied, "the parent task is still open");
    assert!(
        task.result.is_none(),
        "an active child has no terminal result yet"
    );
    let pending = state.pending_continuation.as_ref().unwrap();
    assert_eq!(pending.phase, "queued");
    assert_eq!(state.compaction_reason.as_deref(), Some("threshold"));
    // Subagent identity survives restart: the reopened header keeps depth 1.
    let header = reopened.get_header().unwrap();
    assert_eq!(header["rlmDepth"], Value::from(1));

    // The compacted model context never contains the ledger (bookkeeping is
    // not conversation), and it stays summary-first.
    let context = reopened.build_session_context_with_entry_ids(None);
    let ledger_messages = context
        .messages
        .iter()
        .filter(|message| matches!(message,
            AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { custom_type, .. })
                if custom_type == RLM_CONTINUATION_STATE_CUSTOM_TYPE))
        .count();
    assert_eq!(
        ledger_messages, 0,
        "the continuation ledger must never leak into the model context"
    );
    assert_eq!(context.messages[0].role(), "compactionSummary");

    // Round-trip stability of the terminal shape: a child whose result was
    // recorded but not delivered parses back identically.
    let terminal = parse_rlm_continuation_state(&terminal_child_state()).unwrap();
    let round_trip =
        parse_rlm_continuation_state(&serde_json::to_value(&terminal).unwrap()).unwrap();
    assert_eq!(terminal, round_trip);
    assert_eq!(
        terminal.tasks[0].result.as_ref().unwrap().text,
        "Section 3 fixed; report pending"
    );
    assert_eq!(terminal.pending_result.as_ref().unwrap().status, "complete");
    let _state: RlmContinuationState = terminal;
}

// ---------------------------------------------------------------------------
// End-to-end: the auto-compaction trigger itself through a real AgentSession
// ---------------------------------------------------------------------------

mod e2e {
    use super::*;
    use pi_ai::providers::faux::{
        faux_assistant_message, register_faux_provider, FauxAssistantContent,
        FauxAssistantMessageOptions, FauxModelDefinition, FauxResponseFactory, FauxResponseStep,
        RegisterFauxProviderOptions,
    };
    use pi_ai::types::{Context, Model};
    use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
    use pi_coding_agent::core::agent_session_services::{
        create_agent_session_from_services, create_agent_session_services,
        AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
        CreateAgentSessionServicesOptions,
    };
    use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
    use pi_coding_agent::core::messages::CustomMessage;
    use pi_coding_agent::core::model_registry::ModelRegistry;
    use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
    use pi_coding_agent::core::settings_manager::SettingsManager;
    use std::time::{Duration, Instant};

    const API_KEY: &str = "compaction-recovery-fixture-key";

    /// Summary text with every required section; the goal, constraint and
    /// evidence lines are echoed from the summarizer input so the compacted
    /// context demonstrably retains them.
    /// The body every summary wire call returns. The turn-prefix format (a
    /// split turn) needs its own three sections, or `validate_summary`
    /// rejects the handoff.
    fn summary_text(request_text: &str) -> String {
        if request_text.contains("This is the PREFIX of a turn") {
            return format!(
                "## Original Request\n{}\n## Early Progress\n- Section 1 audited.\n## Context for Suffix\n- {}",
                anchor_lines(request_text),
                "reports/migration-audit.md stays available."
            );
        }
        format!(
            "## Goal\n{}\n## Constraints & Preferences\n{}\n## Progress\n### Done\n- [x] Section 1 audited.\n\n### In Progress\n- [ ] Section 2.\n\n### Blocked\n- None.\n\n## Key Decisions\n- Incremental rollout.\n## Next Steps\n1. Publish the report.\n## Critical Context\n{}",
            anchor_lines(request_text),
            "None.",
            "reports/migration-audit.md stays available."
        )
    }

    /// The anchors each summary chunk re-anchors on: a chunk only sees the
    /// previous summary through `<previous-summary>` chaining, so the goal,
    /// constraint and evidence must be self-contained markers.
    fn anchor_lines(request_text: &str) -> String {
        let goal = if request_text.contains("TASK GOAL: ship the migration report") {
            "TASK GOAL: ship the migration report before Friday."
        } else if request_text.contains("TASK GOAL: fix the billing migration") {
            "TASK GOAL: fix the billing migration and report back to the parent."
        } else {
            "Continue the migration work."
        };
        let constraint = if request_text.contains("never drop the payments ledger") {
            "CONSTRAINT: never drop the payments ledger."
        } else {
            "None."
        };
        format!("{goal}\n{constraint}")
    }

    struct E2eFixture {
        session: Arc<AgentSession>,
        _dir: tempfile::TempDir,
        summary_requests: Arc<Mutex<Vec<String>>>,
    }

    /// One real `AgentSession` on the faux provider with a small context
    /// window, compaction enabled, and a factory that answers agent turns and
    /// summary wire calls (distinguished by the `<conversation>` wrapper).
    async fn fixture(label: &str, rlm_depth: i64) -> E2eFixture {
        fixture_with_checkpoint(label, rlm_depth, false).await
    }

    async fn fixture_with_checkpoint(
        label: &str,
        rlm_depth: i64,
        with_checkpoint: bool,
    ) -> E2eFixture {
        let root = tempfile::Builder::new()
            .prefix(&format!("compaction-recovery-{label}-"))
            .tempdir()
            .unwrap();
        let cwd = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let cwd_string = cwd.to_string_lossy().to_string();
        let agent_dir_string = agent_dir.to_string_lossy().to_string();

        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some(format!("recovery-{label}-{}", unique_tag())),
            models: Some(vec![FauxModelDefinition {
                id: "recovery-model".into(),
                name: Some("Recovery fixture model".into()),
                reasoning: Some(false),
                input: Some(vec!["text".into()]),
                cost: None,
                context_window: Some(8_192.0),
                max_tokens: Some(8_192.0),
            }]),
            tokens_per_second: Some(0.0),
            ..Default::default()
        }));
        let model = provider.get_model();
        if with_checkpoint {
            // Reuse the faux streams, replacing only this unique API's native
            // compaction operation with a production-shaped checkpoint result.
            let registered = pi_ai::api_registry::get_api_provider(&model.api)
                .expect("faux API registration");
            let compact: pi_ai::types::CompactFunction =
                Arc::new(|model: &Model, _context: &Context, _options| {
                    let checkpoint = pi_ai::compaction::ProviderCompactionCheckpoint {
                        version: 1,
                        provider: model.provider.clone(),
                        api: model.api.clone(),
                        model: model.id.clone(),
                        base_url: model.base_url.clone(),
                        endpoint: None,
                        items: vec![json!({
                            "type": "compaction",
                            "encrypted_content": "recovery-checkpoint-Ω"
                        })
                        .as_object()
                        .cloned()
                        .expect("checkpoint item object")],
                        estimated_tokens: 1.0,
                    };
                    Box::pin(async move {
                        Some(pi_ai::compaction::ProviderCompactionResult {
                            checkpoint,
                            usage: None,
                        })
                    })
                });
            register_api_provider_simple(
                ApiProviderSimple {
                    api: model.api.clone(),
                    stream: registered.stream,
                    stream_simple: registered.stream_simple,
                    compact: Some(compact),
                    supports_compaction: Some(Arc::new(|_| true)),
                },
                Some(format!("recovery-checkpoint-{}", unique_tag())),
            );
        }

        let summary_requests: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&summary_requests);
        let factory: FauxResponseFactory =
            Arc::new(move |context: &Context, _options, _state, _model: &Model| {
                let request_text = context
                    .messages
                    .iter()
                    .map(|message| match message {
                        Message::User(user) => user.content.text(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let text = if request_text.contains("<conversation>") {
                    seen.lock().unwrap().push(request_text.clone());
                    summary_text(&request_text)
                } else {
                    "ack: work started.".to_string()
                };
                Box::pin(async move {
                    faux_assistant_message(
                        FauxAssistantContent::Text(text),
                        Some(FauxAssistantMessageOptions {
                            stop_reason: Some("stop".into()),
                            ..Default::default()
                        }),
                    )
                })
            });
        // Every stream call pops one queued step; the continuation chains in
        // the child scenario make several calls, so queue generously.
        let steps: Vec<FauxResponseStep> = (0..64)
            .map(|_| FauxResponseStep::Factory(factory.clone()))
            .collect();
        provider.set_responses(steps);

        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {
                    "enabled": true,
                    "reserveTokens": 512.0,
                    "keepRecentTokens": 5.0
                },
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));

        let mut session_manager =
            SessionManager::create(&cwd_string, Some(&agent_dir_string)).unwrap();
        if rlm_depth > 0 {
            session_manager
                .new_session(Some(&NewSessionOptions {
                    rlm_depth: Some(rlm_depth),
                    ..Default::default()
                }))
                .unwrap();
        }
        let session_manager = Arc::new(Mutex::new(session_manager));

        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            AuthStorageData::new(),
            None,
        )));
        let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(AuthStorageData::new(), None),
        )));
        auth_storage
            .lock()
            .await
            .set_runtime_api_key(&model.provider, API_KEY);
        model_registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, API_KEY);

        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd_string.clone(),
            agent_dir: Some(agent_dir_string.clone()),
            auth_storage: Some(Arc::clone(&auth_storage)),
            settings_manager: Some(Arc::clone(&settings)),
            model_registry: Some(Arc::clone(&model_registry)),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd_string.clone(),
                agent_dir: agent_dir_string.clone(),
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            }),
        })
        .await
        .expect("services");

        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::clone(&session_manager),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                rlm_depth: Some(rlm_depth),
                no_tools: Some("all".to_string()),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                ..Default::default()
            },
        })
        .await
        .expect("agent session");
        E2eFixture {
            session: created.session,
            _dir: root,
            summary_requests,
        }
    }

    async fn wait_until(
        description: &str,
        timeout: Duration,
        mut predicate: impl FnMut() -> bool,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if predicate() {
                return true;
            }
            if Instant::now() >= deadline {
                eprintln!("timed out waiting for {description}");
                return false;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// (a) end-to-end: a prompt that pushes the faux usage estimate past the
    /// trigger boundary makes the real session auto-compact, and the compacted
    /// context is summary-first with the anchors retained.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_session_auto_compaction_end_to_end() {
        let fixture = fixture("auto", 0).await;
        let session = &fixture.session;
        let prompt = format!(
            "TASK GOAL: ship the migration report before Friday. CONSTRAINT: never drop the payments ledger. Evidence reference: reports/migration-audit.md. {}",
            "planning notes: the ledger export needs a dry run first. ".repeat(700)
        );
        session
            .prompt_and_wait(&prompt, None)
            .await
            .expect("prompt accepted");

        // The threshold compaction runs at agent end; wait for the durable
        // compaction entry to land.
        let compacted = wait_until("compaction entry", Duration::from_secs(60), || {
            session
                .session_manager
                .lock()
                .unwrap()
                .get_branch(None)
                .iter()
                .any(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        })
        .await;
        assert!(compacted, "auto-compaction never produced a durable entry");

        // The summarizer saw the anchors: goal, constraint, evidence. With the
        // faux window (8_192) the serialized conversation exceeds one chunk,
        // so the anchors must appear somewhere across the chained chunks.
        let requests = fixture.summary_requests.lock().unwrap().clone();
        assert!(
            !requests.is_empty(),
            "the auto-compaction made no summary wire calls"
        );
        let joined = requests.join("\n<chunk>\n");
        assert!(joined.contains("TASK GOAL: ship the migration report"));
        assert!(joined.contains("CONSTRAINT: never drop the payments ledger"));
        assert!(joined.contains("Evidence reference: reports/migration-audit.md"));
        // The chunk chain is visible in the captured requests.
        assert!(
            joined.contains("<previous-summary>"),
            "chained chunks must re-anchor on the previous summary"
        );

        // The live context is summary-first and under the boundary again.
        let messages = session.agent.messages();
        match messages.first() {
            Some(AgentMessage::Custom(
                pi_agent_core::types::CustomAgentMessage::CompactionSummary { summary, .. },
            )) => {
                assert!(
                    summary.contains("TASK GOAL: ship the migration report"),
                    "goal retained"
                );
                assert!(
                    summary.contains("never drop the payments ledger"),
                    "constraint retained"
                );
                assert!(
                    summary.contains("reports/migration-audit.md"),
                    "evidence retained"
                );
            }
            other => panic!(
                "live context must be summary-first after auto-compaction, got {:?}",
                other.map(|message| message.role())
            ),
        }
        // The retained assistant still carries its pre-compaction usage, so a
        // raw estimate over the compacted context stays above the boundary;
        // production fences that stale anchor with the compaction timestamp
        // (`threshold_context_tokens_from_state`), which is why the session
        // does not immediately re-compact. The compacted context itself is
        // summary + one assistant turn.
        assert_eq!(
            messages.len(),
            2,
            "compacted context must be summary + retained assistant"
        );
    }

    /// (e) end-to-end: an RLM child mid-task threshold-compacts and the
    /// continuation ledger survives with the parent task still open.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_session_rlm_child_pending_work_survives_threshold_compaction() {
        let fixture = fixture("child", 1).await;
        let session = &fixture.session;
        assert_eq!(session.rlm_depth(), 1);

        // The delivered message IS the custom agent-session message: the whole
        // oversized body goes into its content so the model context crosses
        // the compaction boundary.
        let task_text = format!(
            "Fix the billing migration and report back to the parent. TASK GOAL: ship the migration report. CONSTRAINT: never drop the payments ledger. Evidence reference: reports/migration-audit.md. {}",
            "child work notes: section 3 is in progress. ".repeat(700)
        );
        let custom = CustomMessage {
            role: "custom".into(),
            custom_type: "agent_message".into(),
            content: CustomMessageContent::Text(task_text.clone()),
            display: true,
            details: Some(json!({
                "id": "agentmsg-parent-1",
                "message": task_text,
                "fromRelationship": "parent",
                "fromName": "parent-session"
            })),
            timestamp: 1_000,
        };
        session
            .accept_agent_message_prompt(
                &task_text,
                Some(PromptOptions {
                    custom_message: Some(custom),
                    agent_message_id: Some("agentmsg-parent-1".into()),
                    ..Default::default()
                }),
            )
            .await
            .expect("parent task delivered");

        // The child turn crosses the boundary, the threshold compaction runs,
        // and the continuation machinery keeps re-persisting the ledger.
        let settled = wait_until(
            "ledger with an open task after compaction",
            Duration::from_secs(90),
            || {
                let manager = session.session_manager.lock().unwrap();
                let branch = manager.get_branch(None);
                let has_compaction = branch
                    .iter()
                    .any(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"));
                if !has_compaction {
                    return false;
                }
                let ledger = branch.iter().rev().find(|entry| {
                    entry.get("type").and_then(Value::as_str) == Some("custom")
                        && entry.get("customType").and_then(Value::as_str)
                            == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
                });
                match ledger {
                    Some(entry) => parse_rlm_continuation_state(&entry["data"])
                        .map(|state| {
                            state.tasks.len() == 1 && state.tasks[0].id == "agentmsg-parent-1"
                        })
                        .unwrap_or(false),
                    None => false,
                }
            },
        )
        .await;
        assert!(
            settled,
            "the child's continuation ledger never settled after compaction"
        );

        // Durable outcome: flush, reopen the child session file and restore the
        // ledger exactly like `restore_rlm_continuation_state`.
        let (session_file, branch) = {
            let mut manager = session.session_manager.lock().unwrap();
            manager.flush_now().unwrap();
            (manager.get_session_file(), manager.get_branch(None))
        };
        let session_file = session_file.expect("durable child session file");
        let ledger = branch
            .iter()
            .rev()
            .find(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("custom")
                    && entry.get("customType").and_then(Value::as_str)
                        == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
            })
            .expect("ledger entry");
        let state = parse_rlm_continuation_state(&ledger["data"]).unwrap();
        assert_eq!(state.tasks[0].id, "agentmsg-parent-1");
        // The task is either still open or already replied with a recorded
        // result; either way compaction did not erase it. What must never
        // happen is a lost result: once recorded, the text stays.
        if state.tasks[0].replied {
            let result = state.tasks[0]
                .result
                .as_ref()
                .expect("a replied task keeps its recorded result");
            assert!(!result.text.trim().is_empty());
        }

        let reopened = SessionManager::open(&session_file, None, None).unwrap();
        let reopened_branch = reopened.get_branch(None);
        let reopened_ledger = reopened_branch
            .iter()
            .rev()
            .find(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("custom")
                    && entry.get("customType").and_then(Value::as_str)
                        == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
            })
            .expect("ledger survives restart");
        let reopened_state = parse_rlm_continuation_state(&reopened_ledger["data"]).unwrap();
        assert_eq!(reopened_state.tasks[0].id, "agentmsg-parent-1");
        // Whatever the delivery state, the reopened ledger matches the
        // pre-restart one: compaction + restart loses nothing.
        assert_eq!(reopened_state.tasks, state.tasks);
        // The child identity survives restart.
        let header = reopened.get_header().unwrap();
        assert_eq!(header["rlmDepth"], Value::from(1));
    }

    fn append_restore_message(session: &Arc<AgentSession>, message: AgentMessage) {
        session
            .session_manager
            .lock()
            .unwrap()
            .append_message(message.clone())
            .expect("append restore fixture message");
        session.agent.update_state(Box::new(move |state| {
            state.messages.push(message);
        }));
    }

    async fn seed_restore_phase(session: &Arc<AgentSession>, phase: &str) {
        append_restore_message(
            session,
            user_message(&format!(
                "TASK GOAL: ship the migration report before Friday. CONSTRAINT: never drop the payments ledger. Unicode café 東京 🦀 phase {phase}. Evidence reference: reports/résumé-Ω.json. {}",
                "protected context with Unicode ångström Ω 東京. ".repeat(30)
            )),
        );
        session
            .send_custom_message(
                CustomMessage {
                    role: "custom".into(),
                    custom_type: "agent_message".into(),
                    content: CustomMessageContent::Text(format!(
                        "Protected {phase} reviewer evidence: artifacts/résumé-Ω.json"
                    )),
                    display: true,
                    details: Some(json!({
                        "id": format!("agentmsg-pr4-{phase}-Ω"),
                        "fromName": format!("reviewer-{phase}-Ω")
                    })),
                    timestamp: 1_000,
                },
                Some(false),
                None,
            )
            .await
            .expect("append protected restore evidence");

        let call_id = format!("{phase}-call");
        let orphan_id = format!("{phase}-orphan");
        append_restore_message(
            session,
            AgentMessage::Message(Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Text(TextContent::new(format!("Completed phase {phase}."))),
                    tool_call(&call_id, "read_file", "src/résumé-Ω.rs"),
                ],
                stop_reason: "toolUse".into(),
                ..Default::default()
            })),
        );
        append_restore_message(
            session,
            tool_result(&call_id, "read_file", &format!("matched {phase} result Ω")),
        );
        append_restore_message(
            session,
            tool_result(&orphan_id, "read_file", &format!("orphan {phase} result 東京")),
        );
        append_restore_message(
            session,
            user_message(&format!(
                "Retain the {phase} tail. {}",
                "tail marker with Unicode déjà vu Ω. ".repeat(45)
            )),
        );
    }

    fn restore_reopen_snapshot(manager: &SessionManager) -> Value {
        let branch = manager.get_branch(None);
        let branch_values: Vec<Value> = branch
            .iter()
            .map(|entry| Value::Object(entry.clone()))
            .collect();
        let compactions: Vec<Value> = branch_values
            .iter()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
            .cloned()
            .collect();
        let checkpoint_presence: Vec<bool> = compactions
            .iter()
            .map(|entry| {
                entry
                    .get("details")
                    .and_then(|details| details.get("providerCheckpoint"))
                    .is_some()
            })
            .collect();
        let newest_compaction = compactions.last().cloned().unwrap_or(Value::Null);
        let durable_messages = live_messages(&branch);
        let live = manager.build_session_context_with_entry_ids(None);
        let live_messages = serde_json::to_value(&live.messages).expect("serialize live messages");
        let live_tool_call_ids: Vec<String> = live.messages.iter().flat_map(tool_call_ids).collect();
        let live_tool_result_ids: Vec<String> =
            live.messages.iter().flat_map(tool_result_ids).collect();
        json!({
            "header": manager.get_header(),
            "branch": branch_values,
            "compactions": compactions,
            "checkpointPresence": checkpoint_presence,
            "newestCompaction": newest_compaction,
            "durableToolCallIds": durable_messages.iter().flat_map(tool_call_ids).collect::<Vec<_>>(),
            "durableToolResultIds": durable_messages.iter().flat_map(tool_result_ids).collect::<Vec<_>>(),
            "live": {
                "messages": live_messages,
                "toolCallIds": live_tool_call_ids,
                "toolResultIds": live_tool_result_ids,
                "entryIds": live.entry_ids
            }
        })
    }

    fn assert_restore_snapshot(snapshot: &Value, with_checkpoint: bool) {
        let compactions = snapshot["compactions"].as_array().expect("two compactions");
        assert_eq!(compactions.len(), 2);
        assert_eq!(snapshot["newestCompaction"], compactions[1]);
        assert_ne!(compactions[0]["id"], compactions[1]["id"]);
        assert_eq!(snapshot["checkpointPresence"], json!([with_checkpoint, with_checkpoint]));
        assert_eq!(
            snapshot["branch"].as_array().and_then(|branch| branch.last())
                .and_then(|entry| entry.get("type")).and_then(Value::as_str),
            Some("message"),
            "the newest compaction must not be the trailing leaf"
        );
        assert_eq!(snapshot["durableToolCallIds"], json!(["early-call", "late-call"]));
        assert_eq!(
            snapshot["durableToolResultIds"],
            json!(["early-call", "early-orphan", "late-call", "late-orphan"])
        );
        assert_eq!(
            snapshot["branch"].as_array().expect("durable branch").iter().filter(|entry| {
                entry["type"] == "custom_message" && entry["customType"] == "agent_message"
            }).count(),
            2,
            "both protected custom entries survive"
        );
        assert!(snapshot.to_string().contains("café 東京 🦀"), "Unicode evidence survives");
    }

    fn recovery_log_tail(path: &PathBuf) -> String {
        let bytes = std::fs::read(path).unwrap_or_default();
        String::from_utf8_lossy(&bytes[bytes.len().saturating_sub(16 * 1024)..]).into_owned()
    }

    fn run_restore_reopen_child(
        case_root: &PathBuf,
        session_file: &PathBuf,
        variant: &str,
    ) -> Value {
        use std::process::{Command, Stdio};

        let child_root = case_root.join(variant);
        let child_temp = child_root.join("temp");
        std::fs::create_dir_all(&child_temp).expect("create private child root");
        let result_path = child_root.join("reopened.json");
        let stdout_path = child_root.join("child.stdout.log");
        let stderr_path = child_root.join("child.stderr.log");
        let stdout = std::fs::File::create(&stdout_path).expect("create child stdout log");
        let stderr = std::fs::File::create(&stderr_path).expect("create child stderr log");
        let mut child = Command::new(std::env::current_exe().expect("current test executable"))
            .current_dir(&child_root)
            .args([
                "--exact",
                "e2e::restore_branch_projection_survives_fresh_process_reopen",
                "--test-threads=1",
                "--nocapture",
            ])
            .env_clear()
            .env("SystemRoot", std::env::var("SystemRoot").unwrap_or_default())
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", &child_root)
            .env("USERPROFILE", &child_root)
            .env("TEMP", &child_temp)
            .env("TMP", &child_temp)
            .env("TMPDIR", &child_temp)
            .env("OPTIMUS_PARITY_STATE_ROOT", &child_root)
            .env("OPTIMUS_PR4_RECOVERY_CHILD", "1")
            .env("OPTIMUS_PR4_CHILD_SESSION", session_file)
            .env("OPTIMUS_PR4_CHILD_RESULT", &result_path)
            .env("OPTIMUS_PR4_CHILD_VARIANT", variant)
            .env("OPTIMUS_PR4_PARENT_PID", std::process::id().to_string())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("spawn fresh recovery child");

        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(status) = child.try_wait().expect("poll recovery child") {
                assert!(
                    status.success(),
                    "fresh recovery child failed ({status}); logs: {}\n{}\nstdout:\n{}\nstderr:\n{}",
                    stdout_path.display(), stderr_path.display(),
                    recovery_log_tail(&stdout_path), recovery_log_tail(&stderr_path)
                );
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "fresh recovery child timed out; killed and reaped; logs: {}\n{}\nstdout:\n{}\nstderr:\n{}",
                    stdout_path.display(), stderr_path.display(),
                    recovery_log_tail(&stdout_path), recovery_log_tail(&stderr_path)
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        }

        let artifact: Value = serde_json::from_str(
            &std::fs::read_to_string(&result_path).expect("fresh child must write its result"),
        )
        .expect("complete fresh-child JSON result");
        assert_eq!(artifact["schemaVersion"], json!(1));
        assert_eq!(artifact["variant"], json!(variant));
        assert_eq!(artifact["reopened"], Value::Bool(true));
        assert_eq!(artifact["sessionFile"], json!(session_file.to_string_lossy()));
        assert_eq!(artifact["parentPid"].as_u64(), Some(std::process::id() as u64));
        assert_ne!(
            artifact["childPid"].as_u64(), Some(std::process::id() as u64),
            "reopen proof must come from a different process"
        );
        artifact
    }

    /// The parent executes both real restore tails. A fresh child only reopens
    /// the resulting JSONL; it does not execute the edited restore blocks.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_branch_projection_survives_fresh_process_reopen() {
        if std::env::var("OPTIMUS_PR4_RECOVERY_CHILD").unwrap_or_default() == "1" {
            let session_file = PathBuf::from(
                std::env::var_os("OPTIMUS_PR4_CHILD_SESSION").expect("child session path"),
            );
            let result_path = PathBuf::from(
                std::env::var_os("OPTIMUS_PR4_CHILD_RESULT").expect("child result path"),
            );
            let variant = std::env::var("OPTIMUS_PR4_CHILD_VARIANT").expect("child variant");
            let with_checkpoint = match variant.as_str() {
                "ordinary" => false,
                "checkpoint" => true,
                other => panic!("unexpected child variant {other}"),
            };
            let parent_pid = std::env::var("OPTIMUS_PR4_PARENT_PID")
                .expect("parent pid").parse::<u32>().expect("numeric parent pid");
            let reopened = SessionManager::open(&session_file, None, None)
                .expect("fresh child reopens production JSONL");
            let snapshot = restore_reopen_snapshot(&reopened);
            assert_restore_snapshot(&snapshot, with_checkpoint);
            drop(reopened);
            let artifact = json!({
                "schemaVersion": 1,
                "variant": variant,
                "reopened": true,
                "sessionFile": session_file.to_string_lossy(),
                "parentPid": parent_pid,
                "childPid": std::process::id(),
                "snapshot": snapshot
            });
            std::fs::write(
                &result_path,
                serde_json::to_vec_pretty(&artifact).expect("serialize reopen result"),
            )
            .expect("write concrete fresh-child result");
            return;
        }

        let process_root = case_dir("restore-fresh-process");
        for (variant, with_checkpoint) in [("ordinary", false), ("checkpoint", true)] {
            let fixture = fixture_with_checkpoint(variant, 0, with_checkpoint).await;
            let session = Arc::clone(&fixture.session);
            seed_restore_phase(&session, "early").await;
            session.compact_with_options(Some("first bounded restore"), false)
                .await.expect("first real compaction and restore");
            seed_restore_phase(&session, "late").await;
            session.compact_with_options(Some("second bounded restore"), false)
                .await.expect("second real compaction and restore");
            append_restore_message(
                &session,
                user_message(&format!("Trailing non-compaction after the newest {variant} summary Ω")),
            );

            let (session_file, expected, live_messages) = {
                let mut manager = session.session_manager.lock().unwrap();
                manager.flush_now().expect("flush both compactions");
                let expected = restore_reopen_snapshot(&manager);
                assert_restore_snapshot(&expected, with_checkpoint);
                let live_messages = manager.build_session_context_with_entry_ids(None).messages;
                (PathBuf::from(manager.get_session_file().expect("durable session file")), expected, live_messages)
            };
            assert_eq!(
                live_messages, session.agent.messages(),
                "the parent must preserve live context after both real restores"
            );
            session.dispose_async(Some(false)).await;

            let actual = run_restore_reopen_child(&process_root, &session_file, variant);
            assert_eq!(
                actual["snapshot"], expected,
                "fresh-process reopen must preserve the complete projection for {variant}"
            );
        }
    }
}
