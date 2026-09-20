//! Reversible Jev compaction of provider context, not durable session history.
//!
//! `/compact`, automatic summaries and provider checkpoints remain unchanged.
//! The adapter keeps all user/custom messages, errors, edits and recent pairs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::types::{ContentBlock, ImageOrTextContent, Message, ToolCall, UserContent};
use pi_jev::compaction::{
    self, CallAction, CompactionConfig, CompactionPlan, CompactionSkip, HistoryExcerpt,
    PairCandidate, MAX_CANDIDATES, MAX_HISTORY_ENTRIES, MIN_CANDIDATE_CHARS,
};
use pi_jev::types::{DecisionBundle, DecisionCategory, DecisionOutcome, DEFAULT_MODEL};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::core::extensions::types::ExtensionContext;

const MAX_MESSAGES: usize = 4096;
const MAX_BLOCKS: usize = 16384;
const MAX_CONTEXT_BYTES: usize = 16 * 1024 * 1024;
const COMPACTION_DEADLINE: Duration = Duration::from_millis(2500);

type StatusEntries = HashMap<String, (Instant, Value)>;

fn statuses() -> &'static Mutex<StatusEntries> {
    static STATUS: OnceLock<Mutex<StatusEntries>> = OnceLock::new();
    STATUS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Latest observed compaction only; unseen or expired sessions have no status.
pub fn session_status(session_id: &str) -> Option<Value> {
    let mut entries = statuses()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    entries.retain(|_, (observed, _)| observed.elapsed() < Duration::from_secs(1800));
    entries.get(session_id).map(|(_, value)| value.clone())
}

pub fn clear_session_status(session_id: &str) {
    statuses()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(session_id);
}

fn note_status(session_id: &str, stats: Option<&CompactionStats>, fallback: Option<&str>) {
    if session_id.len() > 128 {
        return;
    }
    let mut value = stats
        .and_then(|stats| serde_json::to_value(stats).ok())
        .unwrap_or_else(|| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "applied".into(),
            json!(fallback.is_none() && stats.is_some_and(|stats| stats.reduction_ratio > 0.0)),
        );
        object.insert("fallback_reason".into(), json!(fallback));
    }
    let mut entries = statuses()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    entries.retain(|_, (observed, _)| observed.elapsed() < Duration::from_secs(1800));
    if entries.len() >= 64 && !entries.contains_key(session_id) {
        if let Some(oldest) = entries
            .iter()
            .min_by_key(|(_, (observed, _))| *observed)
            .map(|(id, _)| id.clone())
        {
            entries.remove(&oldest);
        }
    }
    entries.insert(session_id.to_string(), (Instant::now(), value));
}

/// Character counts include JSON framing and metadata; token counts are estimates.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CompactionStats {
    pub messages_before: usize,
    pub messages_after: usize,
    pub chars_before: usize,
    pub chars_after: usize,
    pub estimated_tokens_before: usize,
    pub estimated_tokens_after: usize,
    pub calls_evaluated: usize,
    pub calls_removed: usize,
    pub results_removed: usize,
    pub results_truncated: usize,
    pub reduction_ratio: f64,
    pub protected_messages: usize,
    pub eligible_reduction_ratio: f64,
}

#[derive(Debug, Clone)]
struct NativePair {
    id: String,
    call_index: usize,
    block_index: usize,
    result_index: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedCompaction {
    pub plan: CompactionPlan,
    pub baseline: CompactionStats,
    config: CompactionConfig,
    pairs: Vec<NativePair>,
    fingerprint: [u8; 32],
    eligible_chars_before: usize,
}

#[derive(Debug, Clone)]
pub struct CompactedContext {
    pub messages: Vec<AgentMessage>,
    pub stats: CompactionStats,
}

/// Streaming fingerprint: no second copy of the transcript or tool arguments.
#[derive(Default)]
struct FingerprintWriter {
    hasher: Sha256,
    bytes: usize,
    chars: usize,
}
impl Write for FingerprintWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self.bytes.saturating_add(bytes.len());
        if self.bytes > MAX_CONTEXT_BYTES {
            return Err(io::Error::other("compaction input limit"));
        }
        self.chars += bytes.iter().filter(|byte| **byte & 0xc0 != 0x80).count();
        self.hasher.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn fingerprint(messages: &[AgentMessage]) -> Result<([u8; 32], usize), CompactionSkip> {
    if messages.len() > MAX_MESSAGES {
        return Err(CompactionSkip::InputLimit);
    }
    let mut writer = FingerprintWriter::default();
    serde_json::to_writer(&mut writer, messages).map_err(|_| CompactionSkip::InputLimit)?;
    Ok((writer.hasher.finalize().into(), writer.chars))
}

fn preflight_values(values: &[Value]) -> Result<(), CompactionSkip> {
    if values.len() > MAX_MESSAGES {
        return Err(CompactionSkip::InputLimit);
    }
    let mut writer = FingerprintWriter::default();
    serde_json::to_writer(&mut writer, values).map_err(|_| CompactionSkip::InputLimit)
}

fn pinned(index: usize, len: usize, recent: usize) -> bool {
    index == 0 || index >= len.saturating_sub(recent)
}

fn has_signed_blocks(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Thinking(thinking) => {
            thinking.thinking_signature.is_some() || thinking.redacted == Some(true)
        }
        ContentBlock::Text(text) => text.text_signature.is_some(),
        ContentBlock::ToolCall(call) => call.thought_signature.is_some(),
    })
}

/// Only known reads may lose calls. Python accepts one explicit source-read
/// shape and retains its call, so an opaque cell never loses side-effect evidence.
fn tool_policy(call: &ToolCall) -> Option<bool> {
    let name = call
        .name
        .rsplit('.')
        .next()
        .unwrap_or(&call.name)
        .to_ascii_lowercase();
    match name.as_str() {
        "read" | "read_file" | "readfile" => {
            let path = call
                .arguments
                .get("path")
                .or_else(|| call.arguments.get("file_path"))?
                .as_str()?;
            (!protected_path(path)).then_some(true)
        }
        "search" | "grep" | "glob" | "find" | "list_files" | "websearch" => {
            if ["path", "file_path", "pattern", "query"]
                .iter()
                .filter_map(|key| call.arguments.get(*key).and_then(Value::as_str))
                .any(protected_path)
            {
                None
            } else {
                Some(true)
            }
        }
        "ipython" => {
            let code = call.arguments.get("code")?.as_str()?;
            if code.len() > 4096 {
                return None;
            }
            let expression = code
                .trim()
                .strip_prefix("from pathlib import Path\n")?
                .trim();
            let literal = expression
                .strip_prefix("print(Path(")?
                .strip_suffix(").read_text())")?
                .trim();
            let quote = literal.chars().next()?;
            if !matches!(quote, '\'' | '"') || literal.len() < 3 || !literal.ends_with(quote) {
                return None;
            }
            let path = &literal[1..literal.len() - 1];
            if path
                .chars()
                .any(|ch| matches!(ch, '\\' | '\'' | '"' | '\n' | '\r'))
                || protected_path(path)
            {
                return None;
            }
            Some(false)
        }
        _ => None,
    }
}

fn protected_path(path: &str) -> bool {
    if path.is_empty() || path.len() > 1024 {
        return true;
    }
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".md")
        || lower.ends_with(".mdx")
        || lower.contains("agents")
        || lower.contains("instruction")
        || lower.contains("prompt")
}

fn protected_result_details(details: Option<&Value>) -> bool {
    let Some(details) = details.and_then(Value::as_object) else {
        return false;
    };
    details.get("kernelRestarted") == Some(&Value::Bool(true))
        || details
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| status != "ok")
        || ["stderr", "backgroundOutput"].iter().any(|key| {
            details
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty())
        })
        || ["diffs", "attachments", "sentAgentMessages"]
            .iter()
            .any(|key| {
                details
                    .get(*key)
                    .and_then(Value::as_array)
                    .is_some_and(|items| !items.is_empty())
            })
}

fn excerpt_from_blocks<'a>(texts: impl Iterator<Item = &'a str>) -> String {
    let mut excerpt = String::new();
    let mut remaining = pi_jev::redact::MAX_SCAN_CHARS;
    for text in texts {
        if remaining == 0 {
            break;
        }
        if !excerpt.is_empty() {
            excerpt.push('\n');
            remaining -= 1;
        }
        for ch in text.chars().take(remaining) {
            excerpt.push(ch);
            remaining -= 1;
        }
    }
    pi_jev::redact::bounded_excerpt(&excerpt, 400)
}

pub fn prepare_context(
    messages: &[AgentMessage],
    config: &CompactionConfig,
) -> Result<PreparedCompaction, CompactionSkip> {
    config.validate()?;
    let (source_fingerprint, chars_before) = fingerprint(messages)?;
    // A checkpoint summarizes an immutable prefix. Only complete pairs after
    // the last checkpoint are eligible; opaque checkpoint items never leave here.
    let protected_end = messages.iter().rposition(|message| matches!(message,
        AgentMessage::Message(Message::User(user)) if user.provider_context.is_some()
    ) || matches!(message,
        AgentMessage::Custom(CustomAgentMessage::CompactionSummary { provider_context: Some(_), .. })
    )).map_or(0, |index| index + 1);
    let eligible_chars_before = if protected_end == 0 { chars_before } else { fingerprint(&messages[protected_end..])?.1 };
    let mut calls: BTreeMap<&str, (usize, usize, &ToolCall)> = BTreeMap::new();
    let mut results: BTreeMap<&str, usize> = BTreeMap::new();
    let mut history = Vec::new();
    let mut blocks_seen = 0usize;
    for (index, message) in messages.iter().enumerate() {
        if index < protected_end { continue; }
        let text = match message {
            AgentMessage::Message(Message::User(user)) => {
                match &user.content {
                    UserContent::Text(text) => pi_jev::redact::bounded_excerpt(text, 400),
                    UserContent::Blocks(blocks) => {
                        blocks_seen += blocks.len();
                        excerpt_from_blocks(blocks.iter().filter_map(|block| match block {
                            ImageOrTextContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        }))
                    }
                }
            }
            AgentMessage::Message(Message::Assistant(assistant)) => {
                blocks_seen += assistant.content.len();
                if blocks_seen > MAX_BLOCKS {
                    return Err(CompactionSkip::InputLimit);
                }
                for (block_index, block) in assistant.content.iter().enumerate() {
                    if let ContentBlock::ToolCall(call) = block {
                        if call.id.is_empty()
                            || call.id.len() > 256
                            || call.name.len() > 128
                            || calls.insert(&call.id, (index, block_index, call)).is_some()
                        {
                            return Err(CompactionSkip::InvalidPair);
                        }
                    }
                }
                excerpt_from_blocks(assistant.content.iter().filter_map(|block| match block {
                    ContentBlock::Text(text) => Some(text.text.as_str()),
                    _ => None,
                }))
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                blocks_seen += result.content.len();
                if result.tool_call_id.is_empty()
                    || result.tool_call_id.len() > 256
                    || results.insert(&result.tool_call_id, index).is_some()
                {
                    return Err(CompactionSkip::InvalidPair);
                }
                String::new()
            }
            // Custom messages may contain hidden memory and private tool output.
            // They are retained locally and represented only by their role.
            AgentMessage::Custom(_) => String::new(),
        };
        if blocks_seen > MAX_BLOCKS {
            return Err(CompactionSkip::InputLimit);
        }
        if !text.is_empty() {
            if history.len() >= MAX_HISTORY_ENTRIES {
                return Err(CompactionSkip::InputLimit);
            }
            history.push(HistoryExcerpt {
                index,
                role: if message.role() == "user" {
                    "user"
                } else {
                    "assistant"
                },
                text,
                pinned: pinned(index, messages.len(), config.preserve_recent_messages),
            });
        }
    }
    // Pin the latest actual user request in the redacted assessment too.
    if let Some(user) = history.iter_mut().rev().find(|entry| entry.role == "user") {
        user.pinned = true;
    }
    for (id, result_index) in &results {
        let Some((call_index, _, call)) = calls.get(id) else {
            // A result may finish a call already covered by the checkpoint.
            // Pin that result while continuing to consider independent pairs.
            let covered_call = messages[..protected_end].iter().any(|message| match message {
                AgentMessage::Message(Message::Assistant(assistant)) => assistant.content.iter().any(|block|
                    matches!(block, ContentBlock::ToolCall(call) if call.id == *id)),
                _ => false,
            });
            if covered_call { continue; }
            return Err(CompactionSkip::InvalidPair);
        };
        let AgentMessage::Message(Message::ToolResult(result)) = &messages[*result_index] else {
            return Err(CompactionSkip::InvalidPair);
        };
        if result_index <= call_index || result.tool_name != call.name {
            return Err(CompactionSkip::InvalidPair);
        }
    }
    let mut eligible = Vec::new();
    for (id, (call_index, block_index, call)) in &calls {
        let Some(result_index) = results.get(id).copied() else {
            continue;
        };
        if pinned(*call_index, messages.len(), config.preserve_recent_messages)
            || pinned(
                result_index,
                messages.len(),
                config.preserve_recent_messages,
            )
        {
            continue;
        }
        let Some(drop_allowed) = tool_policy(call) else {
            continue;
        };
        let AgentMessage::Message(Message::ToolResult(result)) = &messages[result_index] else {
            continue;
        };
        let AgentMessage::Message(Message::Assistant(assistant)) = &messages[*call_index] else {
            continue;
        };
        if result.is_error
            || assistant.stop_reason == "error"
            || assistant.stop_reason == "aborted"
            || protected_result_details(result.details.as_ref())
            || result.content.is_empty()
            || result.content.iter().any(|block| match block {
                ImageOrTextContent::Image(_) => true,
                ImageOrTextContent::Text(text) => {
                    text.text_signature.is_some()
                        || text.text.contains(compaction::TRUNCATION_MARKER)
                }
            })
        {
            continue;
        }
        let result_chars: usize = result
            .content
            .iter()
            .filter_map(|block| match block {
                ImageOrTextContent::Text(text) => Some(text.text.chars().count()),
                _ => None,
            })
            .sum();
        if result_chars < MIN_CANDIDATE_CHARS || result_chars <= config.truncate_head_chars + 160 {
            continue;
        }
        eligible.push((
            *call_index,
            *block_index,
            result_index,
            call,
            result_chars,
            drop_allowed && !has_signed_blocks(&assistant.content),
        ));
    }
    // Spend the finite request budget on the largest eligible outputs.
    eligible.sort_by_key(|(call, _, _, _, chars, _)| (std::cmp::Reverse(*chars), *call));
    eligible.truncate(MAX_CANDIDATES);
    eligible.sort_by_key(|(call, block, _, _, _, _)| (*call, *block));
    if eligible.is_empty() {
        return Err(CompactionSkip::NoCandidates);
    }
    let reclaimable: usize = eligible
        .iter()
        .map(|(_, _, _, _, chars, _)| chars.saturating_sub(config.truncate_head_chars + 160))
        .sum();
    if reclaimable as f64 / (eligible_chars_before.max(1) as f64) < config.minimum_reduction_ratio {
        return Err(CompactionSkip::InsufficientReduction);
    }
    let mut pairs = Vec::new();
    let mut candidates = Vec::new();
    for (index, (call_index, block_index, result_index, call, result_chars, allow_drop_call)) in
        eligible.into_iter().enumerate()
    {
        let id = format!("t{}", index + 1);
        pairs.push(NativePair {
            id: id.clone(),
            call_index,
            block_index,
            result_index,
        });
        candidates.push(PairCandidate {
            id,
            tool: pi_jev::redact::bounded_excerpt(&call.name, 64),
            result_chars,
            allow_drop_call,
        });
    }
    let plan = compaction::prepare(&history, &candidates, config)?;
    Ok(PreparedCompaction {
        baseline: CompactionStats {
            messages_before: messages.len(),
            messages_after: messages.len(),
            chars_before,
            chars_after: chars_before,
            estimated_tokens_before: chars_before.div_ceil(4),
            estimated_tokens_after: chars_before.div_ceil(4),
            calls_evaluated: pairs.len(),
            protected_messages: protected_end,
            ..Default::default()
        },
        plan,
        config: config.clone(),
        pairs,
        fingerprint: source_fingerprint,
        eligible_chars_before,
    })
}

pub fn apply_context(
    mut messages: Vec<AgentMessage>,
    prepared: &PreparedCompaction,
    outcome: &DecisionOutcome,
) -> Result<CompactedContext, CompactionSkip> {
    if fingerprint(&messages)?.0 != prepared.fingerprint {
        return Err(CompactionSkip::StaleContext);
    }
    let decisions = compaction::decisions(&prepared.plan, outcome)?;
    let mut removed_results = BTreeSet::new();
    let mut removed_calls: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut stats = prepared.baseline.clone();
    for (pair, decision) in prepared.pairs.iter().zip(decisions.iter()) {
        if pair.id != decision.id {
            return Err(CompactionSkip::InvalidAnswers);
        }
        match decision.action {
            CallAction::Keep => {}
            CallAction::DropCall => {
                removed_calls
                    .entry(pair.call_index)
                    .or_default()
                    .insert(pair.block_index);
                removed_results.insert(pair.result_index);
                stats.calls_removed += 1;
                stats.results_removed += 1;
            }
            CallAction::TruncateResult => {
                let AgentMessage::Message(Message::ToolResult(result)) =
                    &mut messages[pair.result_index]
                else {
                    return Err(CompactionSkip::StaleContext);
                };
                // Keep a single bounded head across all text blocks, not one head per block.
                let count: usize = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ImageOrTextContent::Text(text) => Some(text.text.chars().count()),
                        _ => None,
                    })
                    .sum();
                let mut remaining = prepared.config.truncate_head_chars;
                let mut head = String::new();
                for block in &result.content {
                    if let ImageOrTextContent::Text(text) = block {
                        let prefix: String = text.text.chars().take(remaining).collect();
                        remaining -= prefix.chars().count();
                        head.push_str(&prefix);
                    }
                }
                let omitted = count.saturating_sub(prepared.config.truncate_head_chars);
                let marker = format!("{}{omitted} characters omitted from this tool result; original content remains in session history.]", compaction::TRUNCATION_MARKER);
                if !head.is_empty() {
                    head.push('\n');
                }
                head.push_str(&marker);
                result.content = vec![ImageOrTextContent::Text(pi_ai::types::TextContent::new(
                    head,
                ))];
                stats.results_truncated += 1;
            }
        }
    }
    let mut output = Vec::with_capacity(messages.len());
    for (index, mut message) in messages.into_iter().enumerate() {
        if removed_results.contains(&index) {
            continue;
        }
        if let Some(blocks) = removed_calls.get(&index) {
            let AgentMessage::Message(Message::Assistant(assistant)) = &mut message else {
                return Err(CompactionSkip::StaleContext);
            };
            let mut index = 0;
            assistant.content.retain(|_| {
                let keep = !blocks.contains(&index);
                index += 1;
                keep
            });
            if assistant.content.is_empty() {
                continue;
            }
        }
        output.push(message);
    }
    let (_, chars_after) = fingerprint(&output)?;
    stats.messages_after = output.len();
    stats.chars_after = chars_after;
    stats.estimated_tokens_after = chars_after.div_ceil(4);
    stats.reduction_ratio =
        stats.chars_before.saturating_sub(chars_after) as f64 / stats.chars_before.max(1) as f64;
    stats.eligible_reduction_ratio = stats.chars_before.saturating_sub(chars_after) as f64 / prepared.eligible_chars_before.max(1) as f64;
    if chars_after >= stats.chars_before
        || stats.eligible_reduction_ratio < prepared.config.minimum_reduction_ratio
    {
        return Err(CompactionSkip::InsufficientReduction);
    }
    Ok(CompactedContext {
        messages: output,
        stats,
    })
}

/// Called only at the provider-context seam. Durable AgentState and entries are untouched.
pub async fn compact_context(
    ctx: Arc<dyn ExtensionContext>,
    values: Vec<Value>,
    signal: Option<CancellationToken>,
) -> Vec<Value> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = pi_jev::config::JevSettingsStore::new(crate::config::get_agent_dir()).load();
    if !settings.effective_compaction_enabled(&session_id)
        || signal.as_ref().is_some_and(CancellationToken::is_cancelled)
    {
        return values;
    }
    if preflight_values(&values).is_err() {
        note_status(&session_id, None, Some(CompactionSkip::InputLimit.as_str()));
        crate::core::jev_bridge::record_compaction_skip(
            ctx.clone(),
            CompactionSkip::InputLimit.as_str(),
            json!({}),
        );
        return values;
    }
    let Ok(messages) = values
        .iter()
        .cloned()
        .map(serde_json::from_value::<AgentMessage>)
        .collect::<Result<Vec<_>, _>>()
    else {
        note_status(&session_id, None, Some("invalid_compaction_messages"));
        crate::core::jev_bridge::record_compaction_skip(
            ctx.clone(),
            "invalid_compaction_messages",
            json!({}),
        );
        return values;
    };
    let prepared = match prepare_context(&messages, &settings.compaction) {
        Ok(prepared) => prepared,
        Err(reason @ (CompactionSkip::NoCandidates | CompactionSkip::InsufficientReduction)) => {
            note_status(&session_id, None, Some(reason.as_str()));
            return values;
        }
        Err(reason) => {
            note_status(&session_id, None, Some(reason.as_str()));
            crate::core::jev_bridge::record_compaction_skip(
                ctx.clone(),
                reason.as_str(),
                json!({}),
            );
            return values;
        }
    };
    let started = tokio::time::Instant::now();
    let mut batch_decisions = Vec::new();
    let mut aggregate: Option<DecisionOutcome> = None;
    for questions in &prepared.plan.batches {
        let remaining = COMPACTION_DEADLINE.saturating_sub(started.elapsed());
        let mut state = prepared.plan.state.clone();
        if let Some(object) = state.as_object_mut() {
            object.insert(
                "_jev_policy_generation".into(),
                json!(crate::core::jev_bridge::compaction_policy_generation(
                    &settings,
                    &session_id
                )),
            );
        }
        let bundle = DecisionBundle {
            session_id: session_id.clone(),
            turn: 0,
            stage: "compaction".into(),
            state,
            model: DEFAULT_MODEL.into(),
            question_categories: questions
                .keys()
                .map(|id| (id.clone(), DecisionCategory::ContextRelevance))
                .collect(),
            questions: questions.clone(),
        };
        let decision = tokio::time::timeout(
            remaining,
            crate::core::jev_bridge::decide_compaction(ctx.clone(), bundle, signal.clone()),
        )
        .await
        .ok()
        .flatten();
        let Some(decision) = decision else {
            note_status(
                &session_id,
                Some(&prepared.baseline),
                Some("incomplete_compaction_batch"),
            );
            for previous in &batch_decisions {
                record_batch(previous, None, Some("incomplete_compaction_batch"));
            }
            return values;
        };
        if !decision.can_apply() {
            note_status(
                &session_id,
                Some(&prepared.baseline),
                Some("stale_compaction_context"),
            );
            record_batch(&decision, None, Some("stale_compaction_context"));
            for previous in &batch_decisions {
                record_batch(previous, None, Some("stale_compaction_context"));
            }
            return values;
        }
        if let Some(reason) = decision.fallback_reason() {
            note_status(&session_id, Some(&prepared.baseline), Some(reason));
            record_batch(&decision, Some(&prepared.baseline), Some(reason));
            for previous in &batch_decisions {
                record_batch(previous, None, Some(reason));
            }
            return values;
        }
        if let Some(aggregate) = aggregate.as_mut() {
            if aggregate.response_model != decision.outcome.response_model {
                note_status(
                    &session_id,
                    Some(&prepared.baseline),
                    Some("compaction_model_changed"),
                );
                record_batch(&decision, None, Some("compaction_model_changed"));
                for previous in &batch_decisions {
                    record_batch(previous, None, Some("compaction_model_changed"));
                }
                return values;
            }
            aggregate.records.extend(decision.outcome.records.clone());
            aggregate.skips.extend(decision.outcome.skips.clone());
        } else {
            aggregate = Some(decision.outcome.clone());
        }
        batch_decisions.push(decision);
    }
    let Some(outcome) = aggregate else {
        return values;
    };
    let current = pi_jev::config::JevSettingsStore::new(crate::config::get_agent_dir()).load();
    if !current.effective_compaction_enabled(&session_id)
        || current.compaction != settings.compaction
        || signal.as_ref().is_some_and(CancellationToken::is_cancelled)
        || batch_decisions.iter().any(|decision| !decision.can_apply())
    {
        note_status(
            &session_id,
            Some(&prepared.baseline),
            Some("stale_compaction_context"),
        );
        for decision in &batch_decisions {
            record_batch(decision, None, Some("stale_compaction_context"));
        }
        return values;
    }
    match apply_context(messages, &prepared, &outcome) {
        Ok(compacted) if batch_decisions.iter().all(|decision| decision.can_apply()) => {
            let Ok(output) = compacted
                .messages
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
            else {
                return values;
            };
            note_status(&session_id, Some(&compacted.stats), None);
            for (index, decision) in batch_decisions.iter().enumerate() {
                record_batch(
                    decision,
                    (index + 1 == batch_decisions.len()).then_some(&compacted.stats),
                    None,
                );
            }
            output
        }
        result => {
            let reason = result.err().unwrap_or(CompactionSkip::StaleContext);
            note_status(&session_id, Some(&prepared.baseline), Some(reason.as_str()));
            for (index, decision) in batch_decisions.iter().enumerate() {
                record_batch(
                    decision,
                    (index + 1 == batch_decisions.len()).then_some(&prepared.baseline),
                    Some(reason.as_str()),
                );
            }
            values
        }
    }
}

fn record_batch(
    decision: &crate::core::jev_bridge::CompactionDecision,
    stats: Option<&CompactionStats>,
    fallback: Option<&str>,
) {
    let applied = fallback.is_none() && stats.is_some_and(|stats| stats.reduction_ratio > 0.0);
    let mut stats = stats
        .and_then(|stats| serde_json::to_value(stats).ok())
        .unwrap_or_else(|| json!({"batch_only": true}));
    if let Some(object) = stats.as_object_mut() {
        object.insert("applied".into(), json!(applied));
    }
    decision.record_compaction(stats, fallback);
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn oversized_value_preflight_borrows_without_cloning_the_payload() {
        let value = json!({"role":"toolResult", "content":"x".repeat(MAX_CONTEXT_BYTES + 1)});
        let values = vec![value];
        let original = values[0]["content"].as_str().unwrap().as_ptr();
        assert_eq!(preflight_values(&values), Err(CompactionSkip::InputLimit));
        assert_eq!(values[0]["content"].as_str().unwrap().as_ptr(), original);
        assert_eq!(
            values[0]["content"].as_str().unwrap().len(),
            MAX_CONTEXT_BYTES + 1
        );
        assert_eq!(
            preflight_values(&vec![Value::Null; MAX_MESSAGES + 1]),
            Err(CompactionSkip::InputLimit)
        );
    }

    #[test]
    fn latest_metadata_is_bounded_sparse_expiring_and_removable() {
        let prefix = format!("compact-test-{}", uuid::Uuid::new_v4());
        assert!(session_status(&prefix).is_none());
        note_status(&prefix, None, Some(CompactionSkip::InvalidConfig.as_str()));
        let sparse = session_status(&prefix).unwrap();
        assert_eq!(sparse["applied"], false);
        assert_eq!(sparse["fallback_reason"], "invalid_compaction_config");
        assert!(sparse.get("estimated_tokens_before").is_none());
        clear_session_status(&prefix);
        assert!(session_status(&prefix).is_none());
        for index in 0..70 {
            note_status(
                &format!("{prefix}-{index}"),
                None,
                Some("compaction_input_limit"),
            );
        }
        assert!(statuses().lock().unwrap().len() <= 64);
        let stale = format!("{prefix}-stale");
        statuses().lock().unwrap().insert(
            stale.clone(),
            (
                Instant::now() - Duration::from_secs(1801),
                json!({"applied":false}),
            ),
        );
        assert!(session_status(&stale).is_none());
        for index in 0..70 {
            clear_session_status(&format!("{prefix}-{index}"));
        }
    }
}
