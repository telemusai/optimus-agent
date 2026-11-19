//! W39 D28: native provider compaction must not silently reset the typed
//! durable chain.
//!
//! The native arm of `compact_with_metrics` (a compaction-capable model plus
//! a provider context) replaces the conversation text with the server-side
//! checkpoint. Before D28 it also returned `typed_state: None`, so the typed
//! ledger the episode had merged (the previous compaction's durable chain
//! plus the leaving window) was dropped from the persisted details: one
//! native compaction silently reset `details.typedState`.
//!
//! These tests drive the real pipeline (`prepare_compaction` -> `compact` ->
//! `append_compaction` -> `prepare_compaction` again) over the real
//! `SessionManager` JSONL persistence with a fixture API that is BOTH a
//! deterministic text summarizer (obedient to the protected block, same
//! fixture shape as rot2_typed_integration.rs) and a native compaction
//! provider (registered `compact` handler plus `supports_compaction`).
//!
//! No network. All state under a fresh temp dir per test.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, CustomMessageContent};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
use pi_ai::compaction::{CompactionOptions, CompactionRequestError, ProviderCompactionCheckpoint, ProviderCompactionResult};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Message, Model, TextContent,
    ToolResultMessage, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, default_summary_call_runner, prepare_compaction,
    CompactionResult, CompactionSessionEntry, CompactionSettings, TASK_STATE_MODE_TYPED,
};
use pi_coding_agent::core::compaction::typed_state::TypedInformationKind;
use pi_coding_agent::core::session_manager::{SessionManager, SessionEntry};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Session seeding helpers (mirror rot2_typed_integration.rs)
// ---------------------------------------------------------------------------

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text(text.into()),
        0,
    )))
}

fn assistant_message(text: &str) -> AgentMessage {
    AgentMessage::Message(Message::Assistant(AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: "stop".into(),
        ..Default::default()
    }))
}

fn tool_result_message(id: &str, name: &str, text: &str) -> AgentMessage {
    AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
        id,
        name,
        vec![pi_ai::types::ImageOrTextContent::Text(TextContent::new(text))],
        false,
        0,
    )))
}

fn seed_user(manager: &mut SessionManager, text: &str) {
    manager.append_message(user_message(text)).unwrap();
}

fn seed_assistant(manager: &mut SessionManager, text: &str) {
    manager.append_message(assistant_message(text)).unwrap();
}

fn seed_tool(manager: &mut SessionManager, call_id: &str, name: &str, text: &str) {
    manager
        .append_message(tool_result_message(call_id, name, text))
        .unwrap();
}

/// Mirrors the production converter (`compaction_session_entry_from`,
/// core/agent_session.rs), same as the rot2_typed_integration suite: a
/// session-file entry becomes the `CompactionSessionEntry` the kernel
/// consumes.
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

fn path_entries(branch: &[SessionEntry]) -> Vec<CompactionSessionEntry> {
    branch.iter().filter_map(compaction_entry).collect()
}

fn typed_settings() -> CompactionSettings {
    let mut settings = default_compaction_settings();
    settings.keep_recent_tokens = 20.0;
    settings.task_state = Some(TASK_STATE_MODE_TYPED.to_string());
    settings
}

/// The cycle-1 window: the material the W4 classifier retains, same shape
/// as the rot2_typed_integration seed (task, user constraints, a pending
/// action, a cargo-test tool result, a subagent tool report).
fn seed_cycle1_window(manager: &mut SessionManager) {
    seed_user(
        manager,
        "TASK GOAL: migrate the billing service to the new ledger. \
CONSTRAINT: never delete the payments ledger ARCH-KEEP-LEDGER. \
CONSTRAINT: the API must stay frozen ARCH-FROZEN-API. \
Also don't forget to update the changelog after the migration ARCH-TODO-CHANGELOG.",
    );
    seed_assistant(manager, "Started the migration plan for the billing service.");
    seed_tool(
        manager,
        "call-bash-1",
        "bash",
        "running cargo test\ntest result: ok. 4785 passed; 0 failed; 0 ignored\nfinished in 13.2s",
    );
    seed_tool(
        manager,
        "call-agent-1",
        "agent",
        "Subagent api-reviewer finished: no blocking findings, 2 nits. RLM_CHILD_STATUS: complete",
    );
    seed_user(manager, "status update: continuing the integration work on the migration plan.");
    seed_user(manager, "status update: still working through the checklist for the migration.");
}

/// Cycle-2 material: one new durable constraint plus window filler.
fn seed_cycle2_window(manager: &mut SessionManager) {
    seed_user(
        manager,
        "CONSTRAINT: the rollout must stay canary until sign-off ARCH-2-CANARY.",
    );
    seed_assistant(manager, "Canary rollout plan drafted for the billing migration.");
    seed_user(manager, "status update: preparing the canary checklist for cycle two.");
    seed_user(manager, "status update: verifying the canary thresholds for the rollout.");
}

/// Cycle-3 material: one new durable constraint plus window filler.
fn seed_cycle3_window(manager: &mut SessionManager) {
    seed_user(
        manager,
        "CONSTRAINT: keep the runbook pinned to the current release ARCH-3-RUNBOOK.",
    );
    seed_assistant(manager, "Runbook pinned and reviewed for the third cycle.");
    seed_user(manager, "status update: walking through the third cycle checklist items.");
    seed_user(manager, "status update: finishing the third cycle verification pass.");
}

/// A session manager over a fresh temp dir. The returned guard keeps the
/// dir alive; drop both at the test's end.
fn new_manager(prefix: &str) -> (SessionManager, tempfile::TempDir) {
    let root = tempfile::Builder::new().prefix(prefix).tempdir().unwrap();
    let dir = root.path().to_string_lossy().to_string();
    let manager = SessionManager::create(&dir, Some(&dir)).unwrap();
    (manager, root)
}

// ---------------------------------------------------------------------------
// Deterministic dual-mode fixture: text summarizer + native compaction
// ---------------------------------------------------------------------------

/// Parse the protected block out of a summarizer request (typed shape) and
/// route its item lines into the six summary sections `validate_summary`
/// requires. Same parsing as the rot2_typed_integration fixture.
fn fixture_summary(request: &str) -> String {
    let mut goal: Vec<String> = Vec::new();
    let mut constraints: Vec<String> = Vec::new();
    let mut next: Vec<String> = Vec::new();
    let mut context: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut section = String::new();
    for line in request.lines() {
        if line.starts_with("## Typed Protected State") {
            in_block = true;
            continue;
        }
        if !in_block {
            continue;
        }
        if line.starts_with("Copy every") {
            // The instruction appended after the block; the block is complete.
            break;
        }
        if let Some(item) = line.strip_prefix("- ") {
            let section = section.as_str();
            let bullet = format!("- {item}");
            match section {
                "ACTIVE CONSTRAINTS" | "SAFETY BOUNDARIES" | "ACCEPTANCE CRITERIA" => {
                    constraints.push(bullet);
                }
                "CURRENT TASK" => goal.push(bullet),
                "PENDING ACTIONS" => next.push(bullet),
                _ => context.push(bullet),
            }
        } else if !line.trim().is_empty() {
            section = line.trim().trim_matches('*').to_string();
        }
    }
    let mut out = String::new();
    out.push_str("## Goal\n");
    if goal.is_empty() {
        out.push_str("- Continue the user's task from the retained state.\n");
    } else {
        for line in goal {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.push_str("\n## Constraints & Preferences\n");
    if constraints.is_empty() {
        out.push_str("- None retained by the protected block.\n");
    } else {
        for line in constraints {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.push_str("\n## Progress\n### Done\n- [x] Prior turns summarized by the fixture.\n\n### In Progress\n- [ ] Continuing the task.\n\n### Blocked\n- None.\n");
    out.push_str("\n## Key Decisions\n- None recorded by the protected block.\n");
    out.push_str("\n## Next Steps\n1. Continue the work from the retained state.\n");
    for line in next {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str("\n## Critical Context\n");
    if context.is_empty() {
        out.push_str("- None retained by the protected block.\n");
    } else {
        for line in context {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

struct DualFixture {
    model: Model,
    /// Full text of every text-summarizer request.
    summary_requests: Arc<Mutex<Vec<String>>>,
    /// Number of native compaction calls served.
    native_calls: Arc<Mutex<usize>>,
}

/// A server-side checkpoint the kernel accepts: `is_compaction_checkpoint`
/// on the serialized value and `compaction_matches_model` against the
/// fixture model.
fn native_checkpoint(model: &Model) -> ProviderCompactionCheckpoint {
    ProviderCompactionCheckpoint {
        api: model.api.clone(),
        base_url: "https://fixture.invalid".into(),
        endpoint: None,
        estimated_tokens: 64.0,
        items: vec![json!({
            "role": "assistant",
            "content": "server-side compaction of the cycle window",
        })
        .as_object()
        .unwrap()
        .clone()],
        model: model.id.clone(),
        provider: model.provider.clone(),
        version: 1,
    }
}

/// Registers one fixture API that is both a deterministic text summarizer
/// (obedient to the typed protected block) and a native compaction
/// provider. The compaction handler answers every call with a valid
/// checkpoint for the fixture model.
fn dual_fixture(api: &str) -> DualFixture {
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let native_calls: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let seen = Arc::clone(&captured);
    let served = Arc::clone(&native_calls);

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
                seen.lock().unwrap().push(text.clone());
                let summary = fixture_summary(&text);
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Done {
                    reason: "stop".into(),
                    message: AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent::new(summary))],
                        stop_reason: "stop".into(),
                        ..Default::default()
                    },
                });
                stream
            }),
            compact: Some(Arc::new(move |model, _context, _options| {
                let model = model.clone();
                let served = Arc::clone(&served);
                Box::pin(async move {
                    *served.lock().unwrap() += 1;
                    Ok(Some(ProviderCompactionResult {
                        checkpoint: native_checkpoint(&model),
                        usage: None,
                    }))
                })
                    as Pin<Box<dyn std::future::Future<Output = Result<Option<ProviderCompactionResult>, CompactionRequestError>> + Send>>
            })),
            supports_compaction: Some(Arc::new(|_| true)),
        },
        None,
    );
    let mut model = Model::new(api, api, api, "fixture", "https://fixture.invalid");
    model.context_window = 1_000_000.0;
    model.max_tokens = 64_000.0;
    DualFixture {
        model,
        summary_requests: captured,
        native_calls,
    }
}

/// One compaction attempt over the current session state. `native` selects
/// the arm: `false` runs the text summarizer path (no provider context),
/// `true` passes a provider context so the native arm runs.
async fn run_compaction(
    manager: &mut SessionManager,
    settings: &CompactionSettings,
    model: &Model,
    native: bool,
) -> Result<CompactionResult, String> {
    let branch = manager.get_branch(None);
    let entries = path_entries(&branch);
    let preparation =
        prepare_compaction(&entries, settings, &|_| 40_000.0).expect("prepare_compaction");
    let context = Context::default();
    let options: Option<&CompactionOptions> = None;
    let provider_context = if native {
        Some((&context, options))
    } else {
        None
    };
    compact(
        &preparation,
        model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        None,
        provider_context,
    )
    .await
}

fn persist_compaction(manager: &mut SessionManager, result: &CompactionResult) {
    let details = result
        .details
        .as_ref()
        .map(|details| serde_json::to_value(details).expect("serialize details"));
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
}

// ---------------------------------------------------------------------------
// D28: the durable chain across a native compaction
// ---------------------------------------------------------------------------

/// Two typed-mode text compactions with a native provider compaction
/// between them: the third must still merge the durable chain, and the
/// native entry itself must persist the typed ledger it received instead
/// of silently dropping it (`typed_state: None` was the D28 defect).
#[tokio::test]
async fn native_compaction_between_typed_cycles_keeps_the_durable_chain() {
    let (mut manager, _root) = new_manager("rot2-native-chain-");
    let settings = typed_settings();
    let fixture = dual_fixture("rot2-w39-d28-native");
    let model = fixture.model;

    // Cycle 1: typed text compaction over the seeded window.
    seed_cycle1_window(&mut manager);
    let summary_calls_before = fixture.summary_requests.lock().unwrap().len();
    let result1 = run_compaction(&mut manager, &settings, &model, false)
        .await
        .expect("the obedient fixture must pass cycle-1 verification");
    let details1 = result1.details.as_ref().expect("cycle-1 details");
    let cycle1_ledger = details1
        .typed_state
        .as_ref()
        .expect("cycle 1 must persist its typed ledger");
    let cycle1_constraint = cycle1_ledger
        .records
        .iter()
        .find(|record| {
            record.kind == TypedInformationKind::UserConstraint
                && record.text.contains("payments ledger")
        })
        .expect("the cycle-1 constraint record");
    assert_eq!(*fixture.native_calls.lock().unwrap(), 0);
    assert!(
        fixture.summary_requests.lock().unwrap().len() > summary_calls_before,
        "cycle 1 must go through the text summarizer"
    );
    persist_compaction(&mut manager, &result1);

    // Cycle 2: NATIVE provider compaction over new material.
    seed_cycle2_window(&mut manager);
    let summary_calls_before = fixture.summary_requests.lock().unwrap().len();
    let result2 = run_compaction(&mut manager, &settings, &model, true)
        .await
        .expect("the native fixture must answer cycle 2");
    let details2 = result2.details.as_ref().expect("cycle-2 details");
    assert!(
        details2.provider_checkpoint.is_some(),
        "cycle 2 must take the native arm"
    );
    assert_eq!(*fixture.native_calls.lock().unwrap(), 1);
    assert_eq!(
        fixture.summary_requests.lock().unwrap().len(),
        summary_calls_before,
        "the native arm must not call the text summarizer"
    );
    assert!(
        result2.summary.contains("Conversation compacted on the server"),
        "the native arm replaces the conversation text with server boilerplate"
    );
    // D28 core: the native entry must persist the typed ledger it received
    // (the previous durable chain merged with the leaving window), not
    // silently drop it.
    let cycle2_ledger = details2
        .typed_state
        .as_ref()
        .expect("the native arm must persist the typed ledger it received");
    assert!(
        cycle2_ledger
            .records
            .iter()
            .any(|record| record.id == cycle1_constraint.id),
        "the ledger the native arm received must still carry the cycle-1 constraint by id"
    );
    assert!(
        cycle2_ledger
            .records
            .iter()
            .any(|record| record.text.contains("ARCH-2-CANARY")),
        "the ledger the native arm received must carry the cycle-2 window constraint"
    );
    assert!(
        details2.typed_verification.is_none(),
        "no model-written summary exists on the native path, so nothing is verified"
    );
    // v1 parity, documented and deliberately unfixed (v1 is
    // deprecated-by-evidence): the native path also drops the v1 `taskState`
    // ledger it received. See the D28 row in reports/context-rot-r2/BUG-LEDGER.md.
    assert!(details2.task_state.is_none());
    persist_compaction(&mut manager, &result2);

    // Cycle 3: typed text compaction over new material again. The durable
    // chain must survive the native boundary: the cycle-1 record carries
    // forward by id, and both later windows are present.
    seed_cycle3_window(&mut manager);
    let result3 = run_compaction(&mut manager, &settings, &model, false)
        .await
        .expect("the obedient fixture must pass cycle-3 verification");
    let details3 = result3.details.as_ref().expect("cycle-3 details");
    let cycle3_ledger = details3
        .typed_state
        .as_ref()
        .expect("cycle 3 must persist its merged typed ledger");
    assert!(
        cycle3_ledger
            .records
            .iter()
            .any(|record| record.id == cycle1_constraint.id),
        "the cycle-1 constraint must survive the native compaction by id"
    );
    assert!(
        cycle3_ledger
            .records
            .iter()
            .any(|record| record.text.contains("ARCH-2-CANARY")),
        "the cycle-2 constraint must survive into the third merge"
    );
    assert!(
        cycle3_ledger
            .records
            .iter()
            .any(|record| record.text.contains("ARCH-3-RUNBOOK")),
        "the cycle-3 window constraint must be present in the merged ledger"
    );
}

/// The persisted native entry must round-trip its ledger through the real
/// session file: a reopened session still exposes `details.typedState` on
/// the native compaction, so the durable chain survives a restart too.
#[tokio::test]
async fn native_compaction_entry_round_trips_the_ledger_through_the_session_file() {
    let (mut manager, root) = new_manager("rot2-native-restart-");
    let settings = typed_settings();
    let fixture = dual_fixture("rot2-w39-d28-restart");
    let model = fixture.model;

    seed_cycle1_window(&mut manager);
    let result1 = run_compaction(&mut manager, &settings, &model, false)
        .await
        .expect("cycle 1 must verify");
    persist_compaction(&mut manager, &result1);
    seed_cycle2_window(&mut manager);
    let result2 = run_compaction(&mut manager, &settings, &model, true)
        .await
        .expect("the native fixture must answer");
    let native_details = result2
        .details
        .as_ref()
        .expect("native details")
        .clone();
    let persisted_ledger = native_details
        .typed_state
        .as_ref()
        .expect("the native arm must persist the ledger it received");
    let constraint_id = persisted_ledger
        .records
        .iter()
        .find(|record| {
            record.kind == TypedInformationKind::UserConstraint
                && record.text.contains("payments ledger")
        })
        .expect("the cycle-1 constraint record")
        .id
        .clone();
    persist_compaction(&mut manager, &result2);
    seed_cycle3_window(&mut manager);
    manager.flush_now().unwrap();

    let session_file = manager.get_session_file().expect("durable session file");
    drop(manager);

    let reopened = SessionManager::open(&session_file, None, None).unwrap();
    let branch = reopened.get_branch(None);
    let compactions: Vec<&SessionEntry> = branch
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
        .collect();
    assert_eq!(compactions.len(), 2, "both compactions must be persisted");
    let native_entry = compactions[1];
    let details = native_entry
        .get("details")
        .expect("the native entry must persist details");
    assert!(
        details.get("providerCheckpoint").is_some(),
        "the native entry must carry its provider checkpoint"
    );
    let ledger = details
        .get("typedState")
        .expect("the native entry must round-trip the typed ledger it received");
    let ids: Vec<&str> = ledger["records"]
        .as_array()
        .expect("records array")
        .iter()
        .map(|record| record["id"].as_str().expect("record id"))
        .collect();
    assert!(
        ids.contains(&constraint_id.as_str()),
        "the reopened native entry must still carry the cycle-1 constraint id"
    );
    drop(root);
}

