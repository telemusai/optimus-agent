//! Port of packages/coding-agent/src/core/compaction/utils.ts
//!
//! Shared utilities for compaction and branch summarization.

use std::collections::{BTreeSet, HashMap};

use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::types::Message;
use serde_json::Value;

/// `FileOperations`; `Set<string>` becomes `BTreeSet<String>` (the TypeScript
/// sorts every consumer of these sets before it renders them).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileOperations {
    pub read: BTreeSet<String>,
    pub written: BTreeSet<String>,
    pub edited: BTreeSet<String>,
}

pub fn create_file_ops() -> FileOperations {
    FileOperations {
        read: BTreeSet::new(),
        written: BTreeSet::new(),
        edited: BTreeSet::new(),
    }
}

/// Extract file operations from tool calls in an assistant message.
pub fn extract_file_ops_from_message(message: &AgentMessage, file_ops: &mut FileOperations) {
    if let AgentMessage::Message(Message::ToolResult(result)) = message {
        if result.tool_name == "ipython" {
            if let Some(diffs) = result.details.as_ref().and_then(|details| details.get("diffs")).and_then(Value::as_array) {
                for diff in diffs {
                    if let Some(path) = diff.get("path").and_then(Value::as_str).filter(|path| !path.is_empty()) {
                        file_ops.edited.insert(path.to_string());
                    }
                }
            }
        }
        return;
    }
    let AgentMessage::Message(Message::Assistant(assistant)) = message else {
        return;
    };
    for block in &assistant.content {
        let pi_ai::types::ContentBlock::ToolCall(tool_call) = block else {
            continue;
        };
        let args: &serde_json::Map<String, Value> = &tool_call.arguments;
        let path = match args.get("path") {
            Some(Value::String(path)) => path.clone(),
            _ => continue,
        };
        if path.is_empty() {
            continue;
        }
        match tool_call.name.as_str() {
            "edit" => {
                file_ops.edited.insert(path);
            }
            _ => {}
        }
    }
}

/// Compute final file lists from file operations.
/// Returns readFiles (files only read, not modified) and modifiedFiles.
pub fn compute_file_lists(file_ops: &FileOperations) -> (Vec<String>, Vec<String>) {
    let mut modified: BTreeSet<String> = file_ops.edited.clone();
    modified.extend(file_ops.written.iter().cloned());
    let read_only: Vec<String> = file_ops
        .read
        .iter()
        .filter(|path| !modified.contains(*path))
        .take(200)
        .cloned()
        .collect();
    let modified_files: Vec<String> = modified.into_iter().take(200).collect();
    (read_only, modified_files)
}

/// Bound only rendered metadata; structured file tracking retains its existing inventory.
const FILE_LIST_MAX_CHARS: usize = 6000;

fn budget_file_section(tag: &str, files: &[String], remaining: &mut usize) -> String {
    if files.is_empty() { return String::new(); }
    let omitted = "[additional paths omitted; see structured compaction details]";
    let overhead = format!("\n\n<{tag}>\n\n</{tag}>").chars().count();
    if *remaining < overhead + omitted.len() { return String::new(); }
    let mut body = String::new();
    for (index, path) in files.iter().enumerate() {
        let separator = usize::from(!body.is_empty());
        let reserve = if index + 1 < files.len() { omitted.len() + 1 } else { 0 };
        if overhead + body.chars().count() + separator + path.chars().count() + reserve > *remaining {
            if !body.is_empty() { body.push('\n'); }
            body.push_str(omitted);
            break;
        }
        if !body.is_empty() { body.push('\n'); }
        body.push_str(path);
    }
    let section = format!("\n\n<{tag}>\n{body}\n</{tag}>");
    *remaining -= section.chars().count();
    section
}

/// Modified paths take precedence within the combined display budget.
pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut remaining = FILE_LIST_MAX_CHARS;
    let modified = budget_file_section("modified-files", modified_files, &mut remaining);
    let read = budget_file_section("read-files", read_files, &mut remaining);
    format!("{read}{modified}")
}

/// Remove mechanically appended inventories before iterating a prose summary.
/// File identities remain in CompactionDetails and are reattached after generation.
pub fn strip_file_operations(summary: &str) -> String {
    static BLOCKS: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?ms)^<read-files>\r?\n.*?^</read-files>[ \t]*\r?$|^<modified-files>\r?\n.*?^</modified-files>[ \t]*\r?$").unwrap()
    });
    BLOCKS.replace_all(summary, "").trim().to_string()
}

/// Maximum characters for a tool result in serialized summaries.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// Truncate text to a maximum character length for summarization.
/// Keeps both the beginning and the error-heavy tail inside the same budget.
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    // `String::chars` counts UTF-16-independent scalar values; the TypeScript
    // `slice` counts UTF-16 code units. Plain ASCII/Unicode text matches.
    // Byte counts never exceed char counts, so short ASCII text skips the scan.
    if text.len() <= max_chars {
        return text.to_string();
    }
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }
    let tail_chars = 500.min(max_chars / 4);
    let marker_max = format!("[... {total_chars} characters truncated; first {max_chars} and last {tail_chars} kept ...]").len() + 4;
    if max_chars <= marker_max + tail_chars {
        return text[char_boundary_before(text, max_chars)..].to_string();
    }
    let head_chars = max_chars - tail_chars - marker_max;
    let elided = total_chars - head_chars - tail_chars;
    let head = &text[..char_boundary_after(text, head_chars)];
    let tail = &text[char_boundary_before(text, tail_chars)..];
    format!("{head}\n\n[... {elided} characters truncated; first {head_chars} and last {tail_chars} kept ...]\n\n{tail}")
}

/// Byte offset where the first `count` chars end.
fn char_boundary_after(text: &str, count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    text.char_indices()
        .nth(count - 1)
        .map_or(text.len(), |(byte, ch)| byte + ch.len_utf8())
}

/// Byte offset where the last `count` chars start.
fn char_boundary_before(text: &str, count: usize) -> usize {
    if count == 0 {
        return text.len();
    }
    text.char_indices()
        .rev()
        .nth(count - 1)
        .map_or(0, |(byte, _)| byte)
}

/// Serialize LLM messages to text for summarization.
/// This prevents the model from treating it as a conversation to continue.
/// Call convertToLlm() first to handle custom message types.
///
/// Tool results are truncated to keep the summarization request within
/// reasonable token budgets. Full content is not needed for summarization.
pub fn serialize_conversation(messages: &[Message]) -> String {
    // Two passes with identical output to the historical single pass:
    //
    // Pass 1 (sequential, cheap) plans the tool-call numbering exactly as the
    // single pass did: numbers are assigned in message order, and a tool
    // result resolves against the numbering state at its own position, so a
    // result whose call appears later renders "call outside excerpt".
    //
    // Pass 2 renders each message's parts independently (a pure function of
    // the message plus its plan entry); large conversations render the parts
    // in parallel and concatenate them in message order. Parts are always
    // non-empty, so joining with "\n\n" reproduces the appended stream
    // byte-for-byte. Any parallel failure falls back to the sequential
    // single-pass renderer below.
    if let Some(serialized) = serialize_conversation_parallel(messages) {
        return serialized;
    }
    serialize_conversation_sequential(messages)
}

/// Per-message plan entry produced by the numbering pass.
enum SerializePlan {
    User,
    Assistant { call_numbers: Vec<usize> },
    ToolResult { identity: Option<usize> },
}

fn serialize_conversation_parallel(messages: &[Message]) -> Option<String> {
    use crate::core::compaction::parallel::{try_parallel_map, SERIALIZE_MESSAGES_THRESHOLD};
    let mut calls: HashMap<&str, usize> = HashMap::new();
    let mut next_call = 1usize;
    let mut plan: Vec<SerializePlan> = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::User(_) => plan.push(SerializePlan::User),
            Message::Assistant(assistant) => {
                let mut call_numbers = Vec::new();
                for block in &assistant.content {
                    if let pi_ai::types::ContentBlock::ToolCall(tool_call) = block {
                        let index = next_call;
                        next_call += 1;
                        calls.insert(tool_call.id.as_str(), index);
                        call_numbers.push(index);
                    }
                }
                plan.push(SerializePlan::Assistant { call_numbers });
            }
            Message::ToolResult(tool_result) => {
                plan.push(SerializePlan::ToolResult {
                    identity: calls.get(tool_result.tool_call_id.as_str()).copied(),
                });
            }
        }
    }
    let part_lists: Vec<Vec<String>> = try_parallel_map(
        messages,
        SERIALIZE_MESSAGES_THRESHOLD,
        |message, index| serialize_message_parts(message, &plan[index]),
    )?;
    let count: usize = part_lists.iter().map(Vec::len).sum();
    let mut parts: Vec<String> = Vec::with_capacity(count);
    for list in part_lists {
        parts.extend(list);
    }
    Some(parts.join("\n\n"))
}

/// Render one message's serialized parts (0 to 3 non-empty strings).
fn serialize_message_parts(message: &Message, plan: &SerializePlan) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    match message {
        Message::User(user) => {
            let content = user.content.text();
            if !content.is_empty() {
                parts.push(format!("[User]: {content}"));
            }
        }
        Message::Assistant(assistant) => {
            let SerializePlan::Assistant { call_numbers } = plan else {
                return parts;
            };
            let mut text_parts: Vec<&str> = Vec::new();
            let mut thinking_parts: Vec<&str> = Vec::new();
            let mut tool_calls: Vec<String> = Vec::new();
            let mut next_number = 0usize;
            for block in &assistant.content {
                match block {
                    pi_ai::types::ContentBlock::Text(text) => text_parts.push(&text.text),
                    pi_ai::types::ContentBlock::Thinking(thinking) => {
                        thinking_parts.push(&thinking.thinking)
                    }
                    pi_ai::types::ContentBlock::ToolCall(tool_call) => {
                        let args_str = tool_call
                            .arguments
                            .iter()
                            .map(|(key, value)| {
                                format!(
                                    "{key}={}",
                                    serde_json::to_string(value).unwrap_or_default()
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let index = call_numbers[next_number];
                        next_number += 1;
                        tool_calls.push(format!("#{index} {}({args_str})", tool_call.name));
                    }
                }
            }
            if !thinking_parts.is_empty() {
                parts.push(format!(
                    "[Assistant thinking]: {}",
                    thinking_parts.join("\n")
                ));
            }
            if !text_parts.is_empty() {
                parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
            }
            if !tool_calls.is_empty() {
                parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
            }
        }
        Message::ToolResult(tool_result) => {
            let SerializePlan::ToolResult { identity } = plan else {
                return parts;
            };
            let content: String = tool_result
                .content
                .iter()
                .filter_map(|block| match block {
                    pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                    pi_ai::types::ImageOrTextContent::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join("");
            if !content.is_empty() {
                let identity = match identity {
                    Some(index) => format!("#{index} {}", tool_result.tool_name),
                    None => format!("{} (call outside excerpt)", tool_result.tool_name),
                };
                parts.push(format!(
                    "[Tool result {}{}]: {}",
                    identity,
                    if tool_result.is_error { " ERROR" } else { "" },
                    truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)
                ));
            }
        }
    }
    parts
}

/// The historical single-pass renderer, kept unchanged as the sequential path
/// and the fallback for the parallel renderer.
fn serialize_conversation_sequential(messages: &[Message]) -> String {
    let mut calls: HashMap<String, usize> = HashMap::new();
    let mut next_call = 1usize;
    // The parts are appended directly: collecting them first and joining
    // would copy the whole conversation once more into the final string.
    // Every appended part is non-empty, so separators land exactly where
    // `parts.join("\n\n")` placed them.
    let mut out = String::new();

    for message in messages {
        match message {
            Message::User(user) => {
                let content = user.content.text();
                if !content.is_empty() {
                    push_part(&mut out, "[User]: ");
                    out.push_str(&content);
                }
            }
            Message::Assistant(assistant) => {
                let mut text_parts: Vec<&str> = Vec::new();
                let mut thinking_parts: Vec<&str> = Vec::new();
                let mut tool_calls: Vec<String> = Vec::new();

                for block in &assistant.content {
                    match block {
                        pi_ai::types::ContentBlock::Text(text) => text_parts.push(&text.text),
                        pi_ai::types::ContentBlock::Thinking(thinking) => {
                            thinking_parts.push(&thinking.thinking)
                        }
                        pi_ai::types::ContentBlock::ToolCall(tool_call) => {
                            let args_str = tool_call
                                .arguments
                                .iter()
                                .map(|(key, value)| {
                                    format!(
                                        "{key}={}",
                                        serde_json::to_string(value).unwrap_or_default()
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            let index = next_call;
                            next_call += 1;
                            calls.insert(tool_call.id.clone(), index);
                            tool_calls.push(format!("#{index} {}({args_str})", tool_call.name));
                        }
                    }
                }

                if !thinking_parts.is_empty() {
                    push_part(&mut out, "[Assistant thinking]: ");
                    out.push_str(&thinking_parts.join("\n"));
                }
                if !text_parts.is_empty() {
                    push_part(&mut out, "[Assistant]: ");
                    out.push_str(&text_parts.join("\n"));
                }
                if !tool_calls.is_empty() {
                    push_part(&mut out, "[Assistant tool calls]: ");
                    out.push_str(&tool_calls.join("; "));
                }
            }
            Message::ToolResult(tool_result) => {
                let content: String = tool_result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                        pi_ai::types::ImageOrTextContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if !content.is_empty() {
                    let identity = match calls.get(&tool_result.tool_call_id) {
                        Some(index) => format!("#{index} {}", tool_result.tool_name),
                        None => format!("{} (call outside excerpt)", tool_result.tool_name),
                    };
                    push_part(&mut out, "[Tool result ");
                    out.push_str(&identity);
                    out.push_str(if tool_result.is_error { " ERROR" } else { "" });
                    out.push_str("]: ");
                    out.push_str(&truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS));
                }
            }
        }
    }

    out
}

/// Start a new `\n\n`-separated part of the serialized conversation.
fn push_part(out: &mut String, prefix: &str) {
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(prefix);
}

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// `extractFileOpsFromMessage` accepts the coding-agent custom assistant shape too;
/// this helper mirrors the TypeScript `"content" in message` guard for callers that
/// pass a custom message.
pub fn message_has_content(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Message(_) => true,
        AgentMessage::Custom(CustomAgentMessage::Custom { .. }) => true,
        AgentMessage::Custom(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{
        AssistantMessage, ContentBlock, ImageOrTextContent, TextContent, ToolCall, UserContent,
        UserMessage,
    };
    use serde_json::json;

    fn assistant_with_tool_call(name: &str, path: Option<&str>) -> AgentMessage {
        let mut arguments = serde_json::Map::new();
        if let Some(path) = path {
            arguments.insert("path".to_string(), json!(path));
        }
        AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::ToolCall(ToolCall::new(
                "call-1", name, arguments,
            ))],
            ..Default::default()
        }))
    }

    #[test]
    fn edit_tool_calls_record_edited_paths_only() {
        let mut ops = create_file_ops();
        extract_file_ops_from_message(
            &assistant_with_tool_call("edit", Some("src/a.ts")),
            &mut ops,
        );
        extract_file_ops_from_message(
            &assistant_with_tool_call("read", Some("src/b.ts")),
            &mut ops,
        );
        extract_file_ops_from_message(&assistant_with_tool_call("edit", None), &mut ops);
        assert_eq!(
            ops.edited.iter().cloned().collect::<Vec<_>>(),
            vec!["src/a.ts"]
        );
        assert!(ops.read.is_empty());
        assert!(ops.written.is_empty());
    }

    #[test]
    fn compute_file_lists_sorts_and_excludes_modified_reads() {
        let mut ops = create_file_ops();
        ops.read.insert("b.ts".to_string());
        ops.read.insert("a.ts".to_string());
        ops.edited.insert("b.ts".to_string());
        ops.written.insert("c.ts".to_string());
        let (read_files, modified_files) = compute_file_lists(&ops);
        assert_eq!(read_files, vec!["a.ts".to_string()]);
        assert_eq!(modified_files, vec!["b.ts".to_string(), "c.ts".to_string()]);
    }

    #[test]
    fn format_file_operations_matches_typescript_layout() {
        assert_eq!(format_file_operations(&[], &[]), "");
        assert_eq!(
            format_file_operations(&["a.ts".to_string()], &[]),
            "\n\n<read-files>\na.ts\n</read-files>"
        );
        assert_eq!(
            format_file_operations(&["a.ts".to_string()], &["b.ts".to_string()]),
            "\n\n<read-files>\na.ts\n</read-files>\n\n<modified-files>\nb.ts\n</modified-files>"
        );
    }

    #[test]
    fn serialize_conversation_renders_each_role() {
        let messages = vec![
            Message::User(UserMessage::new(
                UserContent::Blocks(vec![ImageOrTextContent::Text(TextContent::new("hello"))]),
                1,
            )),
            Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(pi_ai::types::ThinkingContent::new("why")),
                    ContentBlock::Text(TextContent::new("answer")),
                    ContentBlock::ToolCall(ToolCall::new("id", "edit", {
                        let mut map = serde_json::Map::new();
                        map.insert("path".to_string(), json!("a.ts"));
                        map
                    })),
                ],
                ..Default::default()
            }),
            Message::ToolResult(pi_ai::types::ToolResultMessage::new(
                "id",
                "edit",
                vec![ImageOrTextContent::Text(TextContent::new("ok"))],
                false,
                2,
            )),
        ];
        let text = serialize_conversation(&messages);
        assert_eq!(
            text,
            "[User]: hello\n\n[Assistant thinking]: why\n\n[Assistant]: answer\n\n[Assistant tool calls]: #1 edit(path=\"a.ts\")\n\n[Tool result #1 edit]: ok"
        );
    }

    #[test]
    fn serialize_conversation_truncates_long_tool_results() {
        let long = "x".repeat(TOOL_RESULT_MAX_CHARS + 10);
        let messages = vec![Message::ToolResult(pi_ai::types::ToolResultMessage::new(
            "id",
            "bash",
            vec![ImageOrTextContent::Text(TextContent::new(long))],
            false,
            1,
        ))];
        let text = serialize_conversation(&messages);
        assert!(text.contains("characters truncated; first"));
        assert!(text.ends_with(&"x".repeat(500)));
        assert!(text.chars().count() <= TOOL_RESULT_MAX_CHARS + "[Tool result bash (call outside excerpt)]: ".len());
    }

    #[test]
    fn file_inventories_are_bounded_and_not_repeated_in_summary_input() {
        let read = vec!["read.rs".to_string(); 200];
        let modified: Vec<String> = (0..200).map(|index| format!("src/{index}/{}.rs", "界".repeat(80))).collect();
        let rendered = format_file_operations(&read, &modified);
        assert!(rendered.chars().count() <= FILE_LIST_MAX_CHARS);
        assert!(rendered.contains(&modified[0]));
        assert!(rendered.contains("additional paths omitted"));
        let prose = "Keep the user's constraint. The function is read-files().";
        assert_eq!(strip_file_operations(&format!("{prose}{rendered}")), prose);
        assert_eq!(strip_file_operations(&format!("{prose}{rendered}").replace('\n', "\r\n")), prose);
        assert_eq!(strip_file_operations("Narrative <read-files> inline </read-files>"),
            "Narrative <read-files> inline </read-files>");
        assert_eq!(modified.len(), 200);
    }

    #[test]
    fn truncate_for_summary_matches_the_char_vector_reference() {
        // The pre-optimization implementation, kept as the oracle: it
        // materialized the whole text as Vec<char> before slicing.
        fn reference(text: &str, max_chars: usize) -> String {
            let chars: Vec<char> = text.chars().collect();
            if chars.len() <= max_chars {
                return text.to_string();
            }
            let tail_chars = 500.min(max_chars / 4);
            let marker_max = format!("[... {} characters truncated; first {max_chars} and last {tail_chars} kept ...]", chars.len()).len() + 4;
            if max_chars <= marker_max + tail_chars {
                return chars[chars.len() - max_chars..].iter().collect();
            }
            let head_chars = max_chars - tail_chars - marker_max;
            let elided = chars.len() - head_chars - tail_chars;
            let head: String = chars[..head_chars].iter().collect();
            let tail: String = chars[chars.len() - tail_chars..].iter().collect();
            format!("{head}\n\n[... {elided} characters truncated; first {head_chars} and last {tail_chars} kept ...]\n\n{tail}")
        }
        let unicode_tail = format!("BEGIN{}\nERROR: failed at final step", "界".repeat(4000));
        let ascii = "y".repeat(9000);
        let mixed = format!("{}{}{}", "a".repeat(2000), "é界𝔘".repeat(500), "z".repeat(3000));
        let samples: [&str; 6] = ["", "abc", "界", "ab界", "界界界界", "hello world"];
        for text in samples {
            for max in [0usize, 1, 2, 3, 4, 5, 10, 100] {
                assert_eq!(
                    truncate_for_summary(text, max),
                    reference(text, max),
                    "text {text:?} max {max}"
                );
            }
        }
        for text in [&unicode_tail, &ascii, &mixed] {
            for max in [0usize, 1, 2, 3, 10, 100, 999, 1500, 2000, 2001, 5000, 9000] {
                assert_eq!(
                    truncate_for_summary(text, max),
                    reference(text, max),
                    "len {} max {max}",
                    text.len()
                );
            }
        }
    }

    #[test]
    fn serialize_conversation_matches_the_parts_join_reference() {
        // The pre-optimization implementation, kept as the oracle: it collected
        // every part into a Vec<String> and joined at the end.
        fn reference(messages: &[Message]) -> String {
            let mut parts: Vec<String> = Vec::new();
            let mut calls: HashMap<String, usize> = HashMap::new();
            let mut next_call = 1usize;
            for message in messages {
                match message {
                    Message::User(user) => {
                        let content = user.content.text();
                        if !content.is_empty() {
                            parts.push(format!("[User]: {content}"));
                        }
                    }
                    Message::Assistant(assistant) => {
                        let mut text_parts: Vec<String> = Vec::new();
                        let mut thinking_parts: Vec<String> = Vec::new();
                        let mut tool_calls: Vec<String> = Vec::new();
                        for block in &assistant.content {
                            match block {
                                ContentBlock::Text(text) => text_parts.push(text.text.clone()),
                                ContentBlock::Thinking(thinking) => {
                                    thinking_parts.push(thinking.thinking.clone())
                                }
                                ContentBlock::ToolCall(tool_call) => {
                                    let args_str = tool_call
                                        .arguments
                                        .iter()
                                        .map(|(key, value)| {
                                            format!(
                                                "{key}={}",
                                                serde_json::to_string(value).unwrap_or_default()
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    let index = next_call;
                                    next_call += 1;
                                    calls.insert(tool_call.id.clone(), index);
                                    tool_calls.push(format!("#{index} {}({args_str})", tool_call.name));
                                }
                            }
                        }
                        if !thinking_parts.is_empty() {
                            parts.push(format!(
                                "[Assistant thinking]: {}",
                                thinking_parts.join("\n")
                            ));
                        }
                        if !text_parts.is_empty() {
                            parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
                        }
                        if !tool_calls.is_empty() {
                            parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                        }
                    }
                    Message::ToolResult(tool_result) => {
                        let content: String = tool_result
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ImageOrTextContent::Text(text) => Some(text.text.clone()),
                                ImageOrTextContent::Image(_) => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        if !content.is_empty() {
                            let identity = match calls.get(&tool_result.tool_call_id) {
                                Some(index) => format!("#{index} {}", tool_result.tool_name),
                                None => format!("{} (call outside excerpt)", tool_result.tool_name),
                            };
                            parts.push(format!(
                                "[Tool result {identity}{}]: {}",
                                if tool_result.is_error { " ERROR" } else { "" },
                                truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)
                            ));
                        }
                    }
                }
            }
            parts.join("\n\n")
        }
        let mut arguments = serde_json::Map::new();
        arguments.insert("path".to_string(), json!("src/界.rs"));
        arguments.insert(
            "nested".to_string(),
            json!({"rows": [1, 2, 3], "note": "细节"}),
        );
        let messages = vec![
            Message::User(UserMessage::new(
                UserContent::Blocks(vec![
                    ImageOrTextContent::Text(TextContent::new("hello 界")),
                    ImageOrTextContent::Text(TextContent::new("second block")),
                ]),
                1,
            )),
            Message::User(UserMessage::new(UserContent::Text(String::new()), 2)),
            Message::Assistant(AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(pi_ai::types::ThinkingContent::new("why")),
                    ContentBlock::Text(TextContent::new("answer")),
                    ContentBlock::ToolCall(ToolCall::new("a", "ipython", arguments)),
                    ContentBlock::Text(TextContent::new("after")),
                    ContentBlock::ToolCall(ToolCall::new("b", "bash", Default::default())),
                ],
                ..Default::default()
            }),
            Message::ToolResult(pi_ai::types::ToolResultMessage::new(
                "b",
                "bash",
                vec![ImageOrTextContent::Text(TextContent::new("x".repeat(5000)))],
                false,
                3,
            )),
            Message::ToolResult(pi_ai::types::ToolResultMessage::new(
                "a",
                "ipython",
                vec![ImageOrTextContent::Text(TextContent::new(format!(
                    "BEGIN{}\nERROR: failed at final step",
                    "界".repeat(4000)
                )))],
                true,
                4,
            )),
            Message::ToolResult(pi_ai::types::ToolResultMessage::new(
                "missing",
                "read",
                vec![ImageOrTextContent::Text(TextContent::new("orphan"))],
                false,
                5,
            )),
            Message::ToolResult(pi_ai::types::ToolResultMessage::new(
                "img",
                "view",
                vec![ImageOrTextContent::Image(Default::default())],
                false,
                6,
            )),
        ];
        assert_eq!(serialize_conversation(&messages), reference(&messages));
    }

    #[test]
    fn tool_results_correlate_out_of_order_and_label_errors() {
        use pi_ai::types::ToolResultMessage;
        let messages = vec![
            Message::Assistant(AssistantMessage { content: vec![
                ContentBlock::ToolCall(ToolCall::new("a", "ipython", Default::default())),
                ContentBlock::ToolCall(ToolCall::new("b", "ipython", Default::default())),
            ], ..Default::default() }),
            Message::ToolResult(ToolResultMessage::new("b", "ipython", vec![ImageOrTextContent::Text(TextContent::new("failed"))], true, 1)),
            Message::ToolResult(ToolResultMessage::new("a", "ipython", vec![ImageOrTextContent::Text(TextContent::new("passed"))], false, 2)),
        ];
        let serialized = serialize_conversation(&messages);
        assert!(serialized.contains("#1 ipython(); #2 ipython()"));
        assert!(serialized.contains("[Tool result #2 ipython ERROR]: failed"));
        assert!(serialized.contains("[Tool result #1 ipython]: passed"));
    }

    #[test]
    fn summary_keeps_unicode_error_tail_and_structured_python_edits() {
        let output = format!("BEGIN{}\nERROR: failed at final step", "界".repeat(4000));
        let summary = truncate_for_summary(&output, TOOL_RESULT_MAX_CHARS);
        assert!(summary.starts_with("BEGIN"));
        assert!(summary.ends_with("ERROR: failed at final step"));
        assert!(summary.chars().count() <= TOOL_RESULT_MAX_CHARS);
        let mut result = pi_ai::types::ToolResultMessage::new("id", "ipython", vec![], false, 1);
        result.details = Some(json!({"diffs": [{"path": "src/a.rs"}, {"path": ""}, 4, {"path": false}]}));
        let mut ops = create_file_ops();
        extract_file_ops_from_message(&AgentMessage::Message(Message::ToolResult(result)), &mut ops);
        assert_eq!(compute_file_lists(&ops).1, vec!["src/a.rs"]);
        for index in 0..250 { ops.edited.insert(format!("file-{index:03}")); }
        assert_eq!(compute_file_lists(&ops).1.len(), 200);
    }
}