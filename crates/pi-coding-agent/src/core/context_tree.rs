//! Port of packages/coding-agent/src/core/context-tree.ts

use std::collections::{BTreeMap, BTreeSet};

use pi_ai::types::{AssistantMessage, Usage};
use serde_json::Value;

use crate::core::compaction::compaction::{calculate_context_tokens, estimate_context_tokens};
use crate::core::usage::{add_assistant_usage, clone_usage, empty_usage, subtract_assistant_usage};

/// Resolves a model's context window so disk-only nodes can report utilization.
pub type ContextWindowResolver = std::sync::Arc<dyn Fn(&str, &str) -> Option<f64> + Send + Sync>;

/// Context utilization shape from `core/extensions/types.ts` (`ContextUsage`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsage {
    /// Estimated context tokens, or null if unknown (e.g. right after compaction, before next LLM response).
    pub tokens: Option<f64>,
    pub context_window: f64,
    /// Context usage as percentage of context window, or null if tokens is unknown.
    pub percent: Option<f64>,
}

/// One agent in the context overview: the main session or an RLM (sub-)agent.
/// `ownUsage` excludes descendants; `totalUsage` includes completed descendants, matching /usage.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextTreeNode {
    /// "root" for the session itself; sub-xxxx for an RLM child.
    pub id: String,
    pub label: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ContextTreeModel>,
    pub own_usage: Usage,
    pub total_usage: Usage,
    /// This node's own auxiliary spend, already included above; unknown calls remain explicit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textmas_usage: Option<crate::core::occ::textmas::TextMasUsageSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_usage: Option<ContextUsage>,
    pub children: Vec<ContextTreeNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContextTreeModel {
    pub provider: String,
    pub id: String,
}

/// `RlmChildAgentStatus` from core/agent-session.ts (another slice).
pub const RLM_CHILD_AGENT_STATUSES: [&str; 5] = ["queued", "running", "done", "error", "cancelled"];

/// Entry shapes read by this module.
///
/// blocked_on: needs core::session_manager::{FileEntry, SessionEntry,
/// loadEntriesFromFile, buildSessionContext}.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextTreeEntry {
    Message {
        id: String,
        parent_id: Option<String>,
        message: pi_agent_core::types::AgentMessage,
    },
    ModelChange {
        id: String,
        parent_id: Option<String>,
        provider: String,
        model_id: String,
    },
    Compaction {
        id: String,
        parent_id: Option<String>,
        usage: Option<Usage>,
    },
    BranchSummary {
        id: String,
        parent_id: Option<String>,
        usage: Option<Usage>,
    },
    ChildUsageAttributed {
        id: String,
        parent_id: Option<String>,
        target_id: String,
        child_usage: Usage,
    },
    TextMasUsage {
        id: String,
        parent_id: Option<String>,
        record: crate::core::occ::textmas::TextMasUsageRecord,
    },
    Other {
        id: String,
        parent_id: Option<String>,
        entry_type: String,
    },
}

impl ContextTreeEntry {
    pub fn id(&self) -> &str {
        match self {
            ContextTreeEntry::Message { id, .. }
            | ContextTreeEntry::ModelChange { id, .. }
            | ContextTreeEntry::Compaction { id, .. }
            | ContextTreeEntry::BranchSummary { id, .. }
            | ContextTreeEntry::ChildUsageAttributed { id, .. }
            | ContextTreeEntry::TextMasUsage { id, .. }
            | ContextTreeEntry::Other { id, .. } => id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            ContextTreeEntry::Message { parent_id, .. }
            | ContextTreeEntry::ModelChange { parent_id, .. }
            | ContextTreeEntry::Compaction { parent_id, .. }
            | ContextTreeEntry::BranchSummary { parent_id, .. }
            | ContextTreeEntry::ChildUsageAttributed { parent_id, .. }
            | ContextTreeEntry::TextMasUsage { parent_id, .. }
            | ContextTreeEntry::Other { parent_id, .. } => parent_id.as_deref(),
        }
    }

    pub fn entry_type(&self) -> &str {
        match self {
            ContextTreeEntry::Message { .. } => "message",
            ContextTreeEntry::ModelChange { .. } => "model_change",
            ContextTreeEntry::Compaction { .. } => "compaction",
            ContextTreeEntry::BranchSummary { .. } => "branch_summary",
            ContextTreeEntry::ChildUsageAttributed { .. } => "child_usage_attributed",
            ContextTreeEntry::TextMasUsage { .. } => "custom",
            ContextTreeEntry::Other { entry_type, .. } => entry_type,
        }
    }
}

fn is_assistant_entry(entry: &ContextTreeEntry) -> Option<&AssistantMessage> {
    let ContextTreeEntry::Message { message, .. } = entry else {
        return None;
    };
    match message {
        pi_agent_core::types::AgentMessage::Message(pi_ai::types::Message::Assistant(
            assistant,
        )) => Some(assistant),
        _ => None,
    }
}

fn read_user_message_text(content: &pi_ai::types::UserContent) -> String {
    match content {
        pi_ai::types::UserContent::Text(text) => text.clone(),
        pi_ai::types::UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                pi_ai::types::ImageOrTextContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn compact_label(text: &str, max_length: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = compact.chars().collect();
    if chars.len() <= max_length {
        return compact;
    }
    let keep = max_length.saturating_sub(3);
    let head: String = chars[..keep].iter().collect();
    format!("{}...", head.trim_end())
}

/// Usage totals for one agent: `totalUsage` sums the branch's assistant usage
/// (attributed aggregates, so descendants are included), `ownUsage` removes the
/// attributions targeting those assistants. Attribution entries are matched by
/// target across ALL entries, not just the branch: attributions rewrite the
/// target assistant's usage no matter which branch they were appended on, so a
/// fork that keeps the assistant but drops the attribution entry must still
/// subtract it.
///
/// Totals are deliberately cumulative across compactions: compaction shrinks
/// the model-facing context, not what the session has spent, so assistants
/// dropped from the resolved context still count here.
pub fn compute_own_and_total_usage(
    branch: &[ContextTreeEntry],
    all_entries: &[ContextTreeEntry],
) -> (Usage, Usage) {
    let mut total_usage = empty_usage();
    let mut branch_assistant_ids: BTreeSet<String> = BTreeSet::new();
    let mut textmas = crate::core::occ::textmas::TextMasUsageLedger::default();
    for entry in branch {
        if let ContextTreeEntry::TextMasUsage { record, .. } = entry { textmas.observe(record.clone()); }
        if let Some(assistant) = is_assistant_entry(entry) {
            branch_assistant_ids.insert(entry.id().to_string());
            add_assistant_usage(&mut total_usage, &assistant.usage);
        } else if matches!(
            entry,
            ContextTreeEntry::Compaction { .. } | ContextTreeEntry::BranchSummary { .. }
        ) {
            let usage = match entry {
                ContextTreeEntry::Compaction { usage, .. } => usage,
                ContextTreeEntry::BranchSummary { usage, .. } => usage,
                _ => &None,
            };
            if let Some(usage) = usage {
                add_assistant_usage(&mut total_usage, usage);
            }
        }
    }
    add_assistant_usage(&mut total_usage, &textmas.total());
    let mut own_usage = clone_usage(&total_usage);
    for entry in all_entries {
        if let ContextTreeEntry::ChildUsageAttributed {
            target_id,
            child_usage,
            ..
        } = entry
        {
            if branch_assistant_ids.contains(target_id) {
                subtract_assistant_usage(&mut own_usage, child_usage);
            }
        }
    }
    (own_usage, total_usage)
}

/// Explain this node's own auxiliary subtotal without including any descendant usage.
pub fn textmas_usage_summary(branch: &[ContextTreeEntry]) -> Option<crate::core::occ::textmas::TextMasUsageSummary> {
    let mut ledger = crate::core::occ::textmas::TextMasUsageLedger::default();
    for entry in branch {
        if let ContextTreeEntry::TextMasUsage { record, .. } = entry { ledger.observe(record.clone()); }
    }
    ledger.summary()
}

/// Current context utilization from persisted entries, mirroring
/// AgentSession.getContextUsage(): unknown right after a compaction until the
/// next assistant response, otherwise the last assistant usage plus an
/// estimate for trailing messages (tool results, queued user input) that have
/// not hit the model yet.
fn compute_context_usage_from_entries(
    all_entries: &[ContextTreeEntry],
    branch: &[ContextTreeEntry],
    context_window: Option<f64>,
    build_session_context: &dyn Fn(&[ContextTreeEntry]) -> Vec<pi_agent_core::types::AgentMessage>,
) -> Option<ContextUsage> {
    let context_window = context_window?;
    if context_window <= 0.0 {
        return None;
    }

    let mut latest_compaction_index: Option<usize> = None;
    for (index, entry) in branch.iter().enumerate().rev() {
        if matches!(entry, ContextTreeEntry::Compaction { .. }) {
            latest_compaction_index = Some(index);
            break;
        }
    }

    if let Some(latest_compaction_index) = latest_compaction_index {
        let mut has_post_compaction_usage = false;
        for entry in branch.iter().skip(latest_compaction_index + 1).rev() {
            let Some(assistant) = is_assistant_entry(entry) else {
                continue;
            };
            if assistant.stop_reason == pi_ai::types::STOP_REASON_ABORTED
                || assistant.stop_reason == pi_ai::types::STOP_REASON_ERROR
            {
                continue;
            }
            if calculate_context_tokens(&assistant.usage) > 0.0 {
                has_post_compaction_usage = true;
            }
            break;
        }
        if !has_post_compaction_usage {
            return Some(ContextUsage {
                tokens: None,
                context_window,
                percent: None,
            });
        }
    }

    let estimate = estimate_context_tokens(&build_session_context(all_entries));
    if estimate.tokens <= 0.0 {
        return None;
    }
    Some(ContextUsage {
        tokens: Some(estimate.tokens),
        context_window,
        percent: Some((estimate.tokens / context_window) * 100.0),
    })
}

/// Entries on the current branch, root to leaf, mirroring
/// SessionManager.getBranch(): the leaf is the last appended entry and the
/// branch is its parentId chain. Keeps forked/abandoned paths out of usage
/// sums so disk nodes match what a live session would report.
pub fn branch_entries(entries: &[ContextTreeEntry]) -> Vec<ContextTreeEntry> {
    if entries.is_empty() {
        return Vec::new();
    }
    let by_id: BTreeMap<String, &ContextTreeEntry> = entries
        .iter()
        .map(|entry| (entry.id().to_string(), entry))
        .collect();
    let mut branch: Vec<ContextTreeEntry> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut current: Option<&ContextTreeEntry> = entries.last();
    while let Some(entry) = current {
        if seen.contains(entry.id()) {
            break;
        }
        seen.insert(entry.id().to_string());
        branch.push(entry.clone());
        current = entry
            .parent_id()
            .and_then(|parent| by_id.get(parent).copied());
    }
    branch.reverse();
    branch
}

/// Terminal status for a persisted child, inferred from how its last assistant
/// turn ended: errored and aborted runs should not render as successful.
pub fn status_from_branch(entries: &[ContextTreeEntry]) -> &'static str {
    for entry in entries.iter().rev() {
        let Some(assistant) = is_assistant_entry(entry) else {
            continue;
        };
        if assistant.stop_reason == pi_ai::types::STOP_REASON_ERROR {
            return "error";
        }
        if assistant.stop_reason == pi_ai::types::STOP_REASON_ABORTED {
            return "cancelled";
        }
        return "done";
    }
    "done"
}

fn find_session_file(dir: &str) -> Option<String> {
    let mut newest: Option<(String, std::time::SystemTime)> = None;
    let names = std::fs::read_dir(dir).ok()?;
    for name in names.flatten() {
        let file_name = name.file_name().to_string_lossy().to_string();
        if !file_name.ends_with(".jsonl") {
            continue;
        }
        let path = std::path::Path::new(dir).join(&file_name);
        let Ok(metadata) = std::fs::metadata(&path) else {
            // Skip unreadable files.
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let candidate = (path.to_string_lossy().to_string(), modified);
        if newest
            .as_ref()
            .map(|(_, current)| modified > *current)
            .unwrap_or(true)
        {
            newest = Some(candidate);
        }
    }
    newest.map(|(path, _)| path)
}

fn list_child_session_dirs(rlm_session_dir: &str) -> Vec<String> {
    let Ok(names) = std::fs::read_dir(rlm_session_dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<(String, std::time::SystemTime)> = Vec::new();
    for name in names.flatten() {
        let file_name = name.file_name().to_string_lossy().to_string();
        if !file_name.starts_with("sub-") {
            continue;
        }
        let path = std::path::Path::new(rlm_session_dir).join(&file_name);
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let modified = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
        dirs.push((path.to_string_lossy().to_string(), modified));
    }
    dirs.sort_by(|a, b| a.1.cmp(&b.1));
    dirs.into_iter().map(|(path, _)| path).collect()
}

/// Reads one persisted session file into `ContextTreeEntry` values.
///
/// blocked_on: needs core::session_manager::loadEntriesFromFile. The caller
/// supplies the loader so this module does not depend on another slice.
pub type SessionEntryLoader = std::sync::Arc<dyn Fn(&str) -> Vec<ContextTreeEntry> + Send + Sync>;

/// Build a context node for a completed RLM child from its persisted session
/// dir (sub-xxxx/). Children that already attributed grandchild usage carry the
/// aggregate on their assistant messages (applyChildUsageAttributions), so own
/// usage is recovered by subtracting the attribution entries. Returns undefined
/// when the dir holds no readable session.
pub fn load_context_tree_child_from_disk(
    child_session_dir: &str,
    resolve_context_window: &ContextWindowResolver,
    load_entries: &SessionEntryLoader,
    build_session_context: &dyn Fn(&[ContextTreeEntry]) -> Vec<pi_agent_core::types::AgentMessage>,
) -> Option<ContextTreeNode> {
    let session_file = find_session_file(child_session_dir)?;
    let all_entries = load_entries(&session_file);
    let branch = branch_entries(&all_entries);
    if branch.is_empty() {
        return None;
    }

    let (own_usage, total_usage) = compute_own_and_total_usage(&branch, &all_entries);

    let mut model: Option<ContextTreeModel> = None;
    for entry in &branch {
        if let ContextTreeEntry::ModelChange {
            provider, model_id, ..
        } = entry
        {
            model = Some(ContextTreeModel {
                provider: provider.clone(),
                id: model_id.clone(),
            });
        }
    }

    let mut label = String::new();
    for entry in &branch {
        if let ContextTreeEntry::Message { message, .. } = entry {
            if let pi_agent_core::types::AgentMessage::Message(pi_ai::types::Message::User(user)) =
                message
            {
                label = compact_label(&read_user_message_text(&user.content), 80);
                if !label.is_empty() {
                    break;
                }
            }
        }
    }

    let context_window = model
        .as_ref()
        .and_then(|model| resolve_context_window(&model.provider, &model.id));

    Some(ContextTreeNode {
        id: std::path::Path::new(child_session_dir)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default(),
        label: if label.is_empty() {
            "child agent".to_string()
        } else {
            label
        },
        status: status_from_branch(&branch).to_string(),
        model,
        own_usage,
        total_usage,
        textmas_usage: textmas_usage_summary(&branch),
        context_usage: compute_context_usage_from_entries(
            &all_entries,
            &branch,
            context_window,
            build_session_context,
        ),
        children: load_context_tree_children_from_disk(
            Some(child_session_dir),
            resolve_context_window,
            None,
            load_entries,
            build_session_context,
        ),
    })
}

/// Build context nodes for all persisted RLM children under an RLM session
/// dir, recursing into nested sub-* dirs for grandchildren. `skipIds`
/// excludes children that are already represented live.
pub fn load_context_tree_children_from_disk(
    rlm_session_dir: Option<&str>,
    resolve_context_window: &ContextWindowResolver,
    skip_ids: Option<&BTreeSet<String>>,
    load_entries: &SessionEntryLoader,
    build_session_context: &dyn Fn(&[ContextTreeEntry]) -> Vec<pi_agent_core::types::AgentMessage>,
) -> Vec<ContextTreeNode> {
    let Some(rlm_session_dir) = rlm_session_dir else {
        return Vec::new();
    };
    if !std::path::Path::new(rlm_session_dir).exists() {
        return Vec::new();
    }
    let mut nodes: Vec<ContextTreeNode> = Vec::new();
    for child_dir in list_child_session_dirs(rlm_session_dir) {
        let id = std::path::Path::new(&child_dir)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        if skip_ids.map(|skip| skip.contains(&id)).unwrap_or(false) {
            continue;
        }
        if let Some(node) = load_context_tree_child_from_disk(
            &child_dir,
            resolve_context_window,
            load_entries,
            build_session_context,
        ) {
            nodes.push(node);
        }
    }
    nodes
}

/// Parses one JSONL line of a session file into `ContextTreeEntry`.
///
/// blocked_on: needs core::session_manager::rehydrateSessionFileEntry. Only the
/// fields this module reads are extracted; the remaining entry kinds map to
/// `Other` so usage totals stay correct.
pub fn context_tree_entry_from_json(value: &Value) -> Option<ContextTreeEntry> {
    let object = value.as_object()?;
    let entry_type = object.get("type").and_then(Value::as_str)?;
    let id = object.get("id").and_then(Value::as_str)?.to_string();
    let parent_id = object
        .get("parentId")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(record) = crate::core::occ::textmas::usage_record(object) {
        return Some(ContextTreeEntry::TextMasUsage { id, parent_id, record });
    }
    match entry_type {
        "message" => {
            let message = object.get("message")?;
            let message =
                serde_json::from_value::<pi_agent_core::types::AgentMessage>(message.clone())
                    .ok()?;
            Some(ContextTreeEntry::Message {
                id,
                parent_id,
                message,
            })
        }
        "model_change" => Some(ContextTreeEntry::ModelChange {
            id,
            parent_id,
            provider: object.get("provider").and_then(Value::as_str)?.to_string(),
            model_id: object.get("modelId").and_then(Value::as_str)?.to_string(),
        }),
        "compaction" => Some(ContextTreeEntry::Compaction {
            id,
            parent_id,
            usage: object
                .get("usage")
                .and_then(|usage| serde_json::from_value::<Usage>(usage.clone()).ok()),
        }),
        "branch_summary" => Some(ContextTreeEntry::BranchSummary {
            id,
            parent_id,
            usage: object
                .get("usage")
                .and_then(|usage| serde_json::from_value::<Usage>(usage.clone()).ok()),
        }),
        "child_usage_attributed" => Some(ContextTreeEntry::ChildUsageAttributed {
            id,
            parent_id,
            target_id: object.get("targetId").and_then(Value::as_str)?.to_string(),
            child_usage: serde_json::from_value::<Usage>(object.get("childUsage")?.clone()).ok()?,
        }),
        other => Some(ContextTreeEntry::Other {
            id,
            parent_id,
            entry_type: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{AssistantMessage, Message, UserContent, UserMessage};
    use serde_json::json;

    fn assistant_entry(
        id: &str,
        parent: Option<&str>,
        usage: Usage,
        stop_reason: &str,
    ) -> ContextTreeEntry {
        ContextTreeEntry::Message {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            message: pi_agent_core::types::AgentMessage::Message(Message::Assistant(
                AssistantMessage {
                    usage,
                    stop_reason: stop_reason.to_string(),
                    ..Default::default()
                },
            )),
        }
    }

    fn usage(input: f64, output: f64, total: f64) -> Usage {
        let mut usage = Usage::default();
        usage.input = input;
        usage.output = output;
        usage.total_tokens = total;
        usage
    }

    fn user_entry(id: &str, parent: Option<&str>, text: &str) -> ContextTreeEntry {
        ContextTreeEntry::Message {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            message: pi_agent_core::types::AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text(text.to_string()),
                1,
            ))),
        }
    }

    #[test]
    fn own_usage_subtracts_attributions_for_branch_assistants() {
        let branch = vec![
            user_entry("u", None, "hello"),
            assistant_entry("a1", Some("u"), usage(100.0, 10.0, 110.0), "stop"),
        ];
        let all_entries = vec![
            branch[0].clone(),
            branch[1].clone(),
            ContextTreeEntry::ChildUsageAttributed {
                id: "attr".to_string(),
                parent_id: Some("a1".to_string()),
                target_id: "a1".to_string(),
                child_usage: usage(40.0, 4.0, 44.0),
            },
            ContextTreeEntry::ChildUsageAttributed {
                id: "attr-other".to_string(),
                parent_id: None,
                target_id: "not-in-branch".to_string(),
                child_usage: usage(1.0, 1.0, 2.0),
            },
        ];
        let (own, total) = compute_own_and_total_usage(&branch, &all_entries);
        assert_eq!(total.input, 100.0);
        assert_eq!(own.input, 60.0);
        assert_eq!(own.output, 6.0);
        assert_eq!(own.total_tokens, 66.0);
    }

    #[test]
    fn compaction_usage_counts_toward_totals() {
        let branch = vec![
            assistant_entry("a1", None, usage(10.0, 1.0, 11.0), "stop"),
            ContextTreeEntry::Compaction {
                id: "c1".to_string(),
                parent_id: Some("a1".to_string()),
                usage: Some(usage(5.0, 0.0, 5.0)),
            },
            ContextTreeEntry::BranchSummary {
                id: "b1".to_string(),
                parent_id: Some("c1".to_string()),
                usage: Some(usage(2.0, 0.0, 2.0)),
            },
        ];
        let (own, total) = compute_own_and_total_usage(&branch, &branch);
        assert_eq!(total.input, 17.0);
        assert_eq!(own.input, 17.0);
    }

    #[test]
    fn branch_entries_follow_the_parent_chain_and_drop_forks() {
        let entries = vec![
            user_entry("root", None, "root"),
            user_entry("forked", Some("root"), "fork"),
            assistant_entry("kept", Some("root"), usage(1.0, 1.0, 2.0), "stop"),
        ];
        let branch = branch_entries(&entries);
        let ids: Vec<&str> = branch.iter().map(|entry| entry.id()).collect();
        assert_eq!(ids, vec!["root", "kept"]);
    }

    #[test]
    fn branch_entries_stop_at_cycles() {
        let entries = vec![
            ContextTreeEntry::Other {
                id: "a".to_string(),
                parent_id: Some("b".to_string()),
                entry_type: "label".to_string(),
            },
            ContextTreeEntry::Other {
                id: "b".to_string(),
                parent_id: Some("a".to_string()),
                entry_type: "label".to_string(),
            },
        ];
        let branch = branch_entries(&entries);
        assert_eq!(branch.len(), 2);
    }

    #[test]
    fn status_from_branch_reads_the_last_assistant_turn() {
        let done = vec![assistant_entry("a", None, usage(1.0, 0.0, 1.0), "stop")];
        assert_eq!(status_from_branch(&done), "done");
        let error = vec![assistant_entry("a", None, usage(1.0, 0.0, 1.0), "error")];
        assert_eq!(status_from_branch(&error), "error");
        let cancelled = vec![assistant_entry("a", None, usage(1.0, 0.0, 1.0), "aborted")];
        assert_eq!(status_from_branch(&cancelled), "cancelled");
        assert_eq!(status_from_branch(&[user_entry("u", None, "x")]), "done");
    }

    #[test]
    fn context_usage_is_unknown_right_after_compaction() {
        let branch = vec![
            user_entry("u", None, "hello"),
            ContextTreeEntry::Compaction {
                id: "c".to_string(),
                parent_id: Some("u".to_string()),
                usage: None,
            },
        ];
        let build = |_entries: &[ContextTreeEntry]| Vec::new();
        let usage =
            compute_context_usage_from_entries(&branch, &branch, Some(1000.0), &build).unwrap();
        assert!(usage.tokens.is_none());
        assert!(usage.percent.is_none());
        assert_eq!(usage.context_window, 1000.0);
    }

    #[test]
    fn context_usage_uses_trailing_estimates_after_a_post_compaction_turn() {
        let branch = vec![
            ContextTreeEntry::Compaction {
                id: "c".to_string(),
                parent_id: None,
                usage: None,
            },
            assistant_entry("a", Some("c"), usage(100.0, 20.0, 120.0), "stop"),
        ];
        let build = |entries: &[ContextTreeEntry]| {
            entries
                .iter()
                .filter_map(|entry| match entry {
                    ContextTreeEntry::Message { message, .. } => Some(message.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let usage =
            compute_context_usage_from_entries(&branch, &branch, Some(1000.0), &build).unwrap();
        assert_eq!(usage.tokens, Some(120.0));
        assert_eq!(usage.percent, Some(12.0));
        assert!(compute_context_usage_from_entries(&branch, &branch, None, &build).is_none());
        assert!(compute_context_usage_from_entries(&branch, &branch, Some(0.0), &build).is_none());
    }

    #[test]
    fn compact_label_collapses_whitespace_and_truncates() {
        assert_eq!(compact_label("  a   b  ", 80), "a b");
        assert_eq!(compact_label(&"x".repeat(100), 10), "xxxxxxx...");
    }

    #[test]
    fn json_entries_map_to_the_local_entry_shape() {
        let entry = context_tree_entry_from_json(&json!({
            "type": "model_change",
            "id": "m1",
            "parentId": "u1",
            "provider": "openai",
            "modelId": "gpt-5",
        }))
        .unwrap();
        assert_eq!(entry.entry_type(), "model_change");
        assert_eq!(entry.parent_id(), Some("u1"));

        let attributed = context_tree_entry_from_json(&json!({
            "type": "child_usage_attributed",
            "id": "c1",
            "parentId": null,
            "targetId": "a1",
            "childUsage": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        }))
        .unwrap();
        match attributed {
            ContextTreeEntry::ChildUsageAttributed { child_usage, .. } => {
                assert_eq!(child_usage.input, 1.0);
                assert_eq!(child_usage.output, 2.0);
            }
            other => panic!("unexpected entry {other:?}"),
        }

        let other = context_tree_entry_from_json(&json!({
            "type": "label",
            "id": "l1",
            "parentId": null,
        }))
        .unwrap();
        assert_eq!(other.entry_type(), "label");
        assert!(context_tree_entry_from_json(&json!({"type": "message"})).is_none());
    }

    #[test]
    fn disk_loading_reads_the_newest_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let child = dir.path().join("sub-1");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(
            child.join("s.jsonl"),
            b"{\"type\":\"model_change\",\"id\":\"m\",\"parentId\":null,\"provider\":\"openai\",\"modelId\":\"gpt-5\"}\n{\"type\":\"message\",\"id\":\"u\",\"parentId\":\"m\",\"message\":{\"role\":\"user\",\"content\":\"hello\",\"timestamp\":1}}\n",
        )
        .unwrap();
        let loader: SessionEntryLoader = std::sync::Arc::new(|path: &str| {
            std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter_map(|value| context_tree_entry_from_json(&value))
                .collect()
        });
        let resolver: ContextWindowResolver = std::sync::Arc::new(|_provider, _model| Some(1000.0));
        let build = |entries: &[ContextTreeEntry]| {
            entries
                .iter()
                .filter_map(|entry| match entry {
                    ContextTreeEntry::Message { message, .. } => Some(message.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let node =
            load_context_tree_child_from_disk(child.to_str().unwrap(), &resolver, &loader, &build)
                .unwrap();
        assert_eq!(node.id, "sub-1");
        assert_eq!(node.label, "hello");
        assert_eq!(node.status, "done");
        assert_eq!(node.context_usage.unwrap().context_window, 1000.0);

        let empty = dir.path().join("sub-2");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(load_context_tree_child_from_disk(
            empty.to_str().unwrap(),
            &resolver,
            &loader,
            &build
        )
        .is_none());
    }

    #[test]
    fn children_listing_sorts_and_skips_ids() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["sub-b", "sub-a", "other"] {
            std::fs::create_dir_all(dir.path().join(name)).unwrap();
        }
        for name in ["sub-a", "sub-b"] {
            std::fs::write(
                dir.path().join(name).join("s.jsonl"),
                b"{\"type\":\"message\",\"id\":\"u\",\"parentId\":null,\"message\":{\"role\":\"user\",\"content\":\"x\",\"timestamp\":1}}\n",
            )
            .unwrap();
        }
        let loader: SessionEntryLoader = std::sync::Arc::new(|path: &str| {
            std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter_map(|value| context_tree_entry_from_json(&value))
                .collect()
        });
        let resolver: ContextWindowResolver = std::sync::Arc::new(|_provider, _model| None);
        let build = |_entries: &[ContextTreeEntry]| Vec::new();
        let nodes = load_context_tree_children_from_disk(
            Some(dir.path().to_str().unwrap()),
            &resolver,
            None,
            &loader,
            &build,
        );
        assert_eq!(nodes.len(), 2);
        let mut skip: BTreeSet<String> = BTreeSet::new();
        skip.insert("sub-a".to_string());
        let skipped = load_context_tree_children_from_disk(
            Some(dir.path().to_str().unwrap()),
            &resolver,
            Some(&skip),
            &loader,
            &build,
        );
        assert_eq!(skipped.len(), 1);
        assert!(
            load_context_tree_children_from_disk(None, &resolver, None, &loader, &build).is_empty()
        );
        assert!(load_context_tree_children_from_disk(
            Some("missing-dir-xyz"),
            &resolver,
            None,
            &loader,
            &build
        )
        .is_empty());
    }
}
