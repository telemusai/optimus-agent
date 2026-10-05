//! C1: fallback contract for the compaction parallel sites.
//!
//! Every data-parallel site in the local compaction pipeline must produce
//! byte-identical output to its sequential implementation, and any failure in
//! the parallel path (including a worker panic) must fall back to running the
//! unchanged sequential implementation for the whole operation.
//!
//! These tests drive each site above its size threshold with three modes:
//! - normal (parallel path active),
//! - fault injection (admission fails -> sequential fallback),
//! - panic injection (a worker panics mid-flight -> sequential fallback),
//! and assert the outputs are identical in all three modes, with the fallback
//! counter proving the fallback path was actually taken.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    AssistantMessage, ContentBlock, ImageOrTextContent, Message, TextContent, ToolCall,
    ToolResultMessage, UserContent, UserMessage,
};
use pi_coding_agent::core::compaction::compaction::{
    default_compaction_settings, estimate_context_tokens, prepare_compaction,
    CompactionSessionEntry,
};
use pi_coding_agent::core::compaction::parallel::{
    parallel_fallback_count, set_parallel_fault_injection, set_parallel_panic_injection,
};
use pi_coding_agent::core::compaction::utils::serialize_conversation;
use pi_coding_agent::core::messages::convert_to_llm;
use pi_coding_agent::core::model_tool_output_policy::ModelToolOutputPolicyOptions;
use pi_coding_agent::core::session_manager::SessionManager;

/// Deterministic pseudo-random ASCII text (no dependencies).
fn fill(seed: u64, len: usize) -> String {
    let mut state = seed | 1;
    let mut out = String::with_capacity(len);
    const WORDS: [&str; 8] = [
        "alpha", "bravo", "delta", "gamma", "omega", "sigma", "tau", "zeta",
    ];
    while out.len() < len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push_str(WORDS[((state >> 33) as usize) % WORDS.len()]);
        out.push(' ');
    }
    out.truncate(len);
    out
}

/// A conversation above every site gate (1600 messages, ~30 MB: >= 24 MB with
/// >= 4 KB average parts, >= 1536 entries, >= 4096-byte middle entry), with
/// the message shapes that exercise the serializers: plain text, thinking,
/// tool calls, tool results (including oversized ones that truncate), and
/// duplicate call ids.
fn large_messages() -> Vec<AgentMessage> {
    let mut messages: Vec<AgentMessage> = Vec::new();
    for index in 0..1600 {
        match index % 6 {
            0 => messages.push(AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text(fill(index as u64 + 1, 20_000)),
                1_000_000 + index as i64,
            )))),
            1 | 4 => {
                let mut content = vec![ContentBlock::Thinking(pi_ai::types::ThinkingContent::new(
                    fill(index as u64 + 2, 6_000),
                ))];
                content.push(ContentBlock::Text(TextContent::new(fill(
                    index as u64 + 3,
                    14_000,
                ))));
                messages.push(AgentMessage::Message(Message::Assistant(AssistantMessage {
                    content,
                    stop_reason: "error".to_string(),
                    ..Default::default()
                })));
            }
            2 => {
                let mut arguments = serde_json::Map::new();
                arguments.insert(
                    "path".to_string(),
                    serde_json::Value::String(format!("src/file{index}.rs")),
                );
                arguments.insert(
                    "content".to_string(),
                    serde_json::Value::String(fill(index as u64 + 4, 2_000)),
                );
                messages.push(AgentMessage::Message(Message::Assistant(AssistantMessage {
                    content: vec![ContentBlock::ToolCall(ToolCall::new(
                        // Duplicate ids across messages: numbering and
                        // "call outside excerpt" resolution must not wobble.
                        format!("call{}", index % 30),
                        if index % 4 == 0 { "edit" } else { "read" }.to_string(),
                        arguments,
                    ))],
                    stop_reason: "error".to_string(),
                    ..Default::default()
                })));
            }
            3 | 5 => messages.push(AgentMessage::Message(Message::ToolResult(
                ToolResultMessage::new(
                    format!("call{}", (index * 7 + 3) % 40),
                    if index % 3 == 0 { "ipython" } else { "read" }.to_string(),
                    vec![ImageOrTextContent::Text(TextContent::new(fill(
                        index as u64 + 5,
                        20_000,
                    )))],
                    index % 10 == 0,
                    1_000_000 + index as i64,
                ),
            ))),
            _ => unreachable!(),
        }
    }
    messages
}

fn large_entries(messages: &[AgentMessage]) -> Vec<CompactionSessionEntry> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| CompactionSessionEntry::Message {
            id: format!("e{index:06}"),
            parent_id: if index == 0 {
                None
            } else {
                Some(format!("e{:06}", index - 1))
            },
            message: message.clone(),
        })
        .collect()
}

fn large_llm(messages: &[AgentMessage]) -> Vec<Message> {
    convert_to_llm(messages, &ModelToolOutputPolicyOptions::default())
}

fn estimate_site(messages: &[AgentMessage]) -> f64 {
    estimate_context_tokens(messages).tokens
}

fn prepare_site(entries: &[CompactionSessionEntry]) -> Option<String> {
    let mut settings = default_compaction_settings();
    settings.enabled = true;
    settings.keep_recent_tokens = 100.0;
    prepare_compaction(entries, &settings, &|entries: &[CompactionSessionEntry]| {
        let mut total = 0.0;
        for entry in entries {
            if let CompactionSessionEntry::Message { message, .. } = entry {
                total += estimate_context_tokens(std::slice::from_ref(message)).tokens;
            }
        }
        total
    })
    .map(|preparation| {
        format!(
            "{:?}|{:?}|{}|{}",
            preparation.first_kept_entry_id,
            preparation.messages_to_summarize.len(),
            preparation.is_split_turn,
            preparation.tokens_before
        )
    })
}

fn serialize_site(messages: &[AgentMessage]) -> String {
    serialize_conversation(&large_llm(messages))
}

/// Serializes the two tests: the injection flags and the fallback counter
/// are process-global.
fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(Default::default).lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn restore_site(messages: &[AgentMessage]) -> String {
    let cwd = std::env::temp_dir().join("compaction-parallel-fallback-test");
    let session_dir = cwd.join("sessions");
    std::fs::create_dir_all(&session_dir).expect("mkdir");
    let mut manager = SessionManager::in_memory(
        Some(&cwd.to_string_lossy()),
        Some(&session_dir.to_string_lossy()),
    )
    .expect("in-memory manager");
    for message in messages {
        manager.append_message(message.clone()).expect("append");
    }
    let branch = manager.get_branch(None);
    let context = manager.build_session_context(None);
    format!(
        "{}|{}|{:?}|{:?}",
        branch.len(),
        context.messages.len(),
        context.thinking_level,
        context
            .messages
            .iter()
            .map(|message| message.role().to_string())
            .collect::<Vec<_>>()
    )
}

#[test]
fn parallel_sites_match_sequential_output_and_fall_back_cleanly() {
    let _guard = test_lock();
    // Capture the expected panic-injection report (the injected worker panic
    // is caught and is the point of the exercise); real assertion failures
    // still surface through the captured message.
    static PANIC_LOG: std::sync::OnceLock<std::sync::Mutex<String>> =
        std::sync::OnceLock::new();
    let panic_log = PANIC_LOG.get_or_init(Default::default);
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        panic_log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_str(&format!("{info}\n"));
    }));
    let messages = large_messages();
    let entries = large_entries(&messages);

    // Baseline: parallel path active (inputs exceed every threshold).
    set_parallel_fault_injection(false);
    set_parallel_panic_injection(false);
    let parallel = (
        estimate_site(&messages),
        prepare_site(&entries),
        serialize_site(&messages),
        restore_site(&messages),
    );

    // Fault injection: admission fails, sequential fallback runs.
    let fallbacks_before_fault = parallel_fallback_count();
    set_parallel_fault_injection(true);
    let faulted = (
        estimate_site(&messages),
        prepare_site(&entries),
        serialize_site(&messages),
        restore_site(&messages),
    );
    let fault_fallbacks = parallel_fallback_count() - fallbacks_before_fault;
    set_parallel_fault_injection(false);

    // Panic injection: a worker panics mid-flight, the fallback re-runs the
    // whole operation sequentially.
    let fallbacks_before_panic = parallel_fallback_count();
    set_parallel_panic_injection(true);
    let panicked = (
        estimate_site(&messages),
        prepare_site(&entries),
        serialize_site(&messages),
        restore_site(&messages),
    );
    let panic_fallbacks = parallel_fallback_count() - fallbacks_before_panic;
    set_parallel_panic_injection(false);
    let captured = std::mem::take(
        &mut *panic_log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    );
    // Restore the default hook before asserting so assertion failures print
    // normally; only the injected worker panics were meant to be captured.
    std::panic::set_hook(prev_hook);

    assert_eq!(parallel, faulted, "fault-injected fallback must match parallel output");
    assert_eq!(parallel, panicked, "panic-injected fallback must match parallel output");
    assert!(
        fault_fallbacks >= 4,
        "fault injection must have forced sequential fallbacks (got {fault_fallbacks})"
    );
    assert!(
        panic_fallbacks >= 4,
        "worker panic must have forced sequential fallbacks (got {panic_fallbacks})"
    );
    let expected_panics = captured.matches("panic injection").count();
    assert!(
        expected_panics >= 4,
        "panic injection must have fired in workers (captured {expected_panics} reports)"
    );
}

#[test]
fn small_inputs_stay_sequential_without_fallback() {
    let _guard = test_lock();
    // Below every threshold the sites run their sequential path without
    // counting a fallback.
    let mut messages = Vec::new();
    for index in 0..8 {
        messages.push(AgentMessage::Message(Message::User(UserMessage::new(
            UserContent::Text(fill(index as u64, 100)),
            index as i64,
        ))));
    }
    let before = parallel_fallback_count();
    let _ = estimate_site(&messages);
    let _ = serialize_site(&messages);
    let _ = restore_site(&messages);
    assert_eq!(
        parallel_fallback_count(),
        before,
        "small inputs must not count as fallbacks"
    );
}
