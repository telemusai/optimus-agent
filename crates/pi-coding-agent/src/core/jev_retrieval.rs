//! Request-local relevance adapters. Stored messages and memories are never changed.
//!
//! Context candidates are successful historical results of the explicit read/search
//! tool names below, with one exact paired call. The newest six messages, current
//! user suffix, instructions, signed/provider context, unknown tools and all custom
//! messages remain pinned. Relevance replaces only a result's text with a marker;
//! it never removes the paired call or grants tool execution authority.

use std::collections::{BTreeMap, BTreeSet};
use std::time::SystemTime;

use pi_jev::active::ActiveDecision;
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::filtering::{
    candidate_questions, dropped_candidate_indices_with_options, FilteringOptions,
    MAX_FILTER_CANDIDATES, MAX_FILTER_EXCERPT_CHARS,
};
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};

use crate::core::memory::evidence::MemoryOrigin;
use crate::core::memory::search::{MemoryHit, MemoryScope};

pub const OMITTED_CONTEXT: &str = "[Optional historical read/search result omitted for this request; original result remains in session history.]";
const MAX_SCAN_MESSAGES: usize = 512;
const MAX_SCAN_BLOCKS: usize = 64;
const MIN_RECENT_MESSAGES: usize = 6;
const READ_SEARCH_TOOLS: [&str; 6] = ["read", "read_file", "search", "grep", "find", "file_search"];

pub struct PreparedRelevance {
    pub state: Value,
    pub category: DecisionCategory,
    pub candidate_indices: Vec<usize>,
    candidate_token_estimates: Vec<usize>,
    options: FilteringOptions,
}

impl PreparedRelevance {
    pub(crate) fn new(
        category: DecisionCategory,
        query: &str,
        candidates: Vec<(usize, String, usize)>,
    ) -> Self {
        let key = candidate_key(category);
        let descriptions: Vec<Value> = candidates.iter().enumerate().map(|(ordinal, (_, excerpt, _))|
            json!({"id": ordinal.to_string(), "excerpt": pi_jev::redact::bounded_excerpt(excerpt, MAX_FILTER_EXCERPT_CHARS)})
        ).collect();
        let mut state = json!({"user_text_excerpt": pi_jev::redact::bounded_excerpt(query, MAX_FILTER_EXCERPT_CHARS)});
        state[key] = Value::Array(descriptions);
        Self {
            state,
            category,
            candidate_token_estimates: candidates
                .iter()
                .map(|(_, _, bytes)| estimate_tokens(*bytes))
                .collect(),
            candidate_indices: candidates.into_iter().map(|(index, _, _)| index).collect(),
            options: FilteringOptions::default(),
        }
    }

    pub fn configure(&mut self, options: &FilteringOptions) {
        self.options = options.clone();
        let limit = if options.validate().is_ok() {
            options.max_candidates
        } else {
            0
        };
        self.candidate_indices.truncate(limit);
        self.candidate_token_estimates.truncate(limit);
        if let Some(candidates) = self.state[candidate_key(self.category)].as_array_mut() {
            candidates.truncate(limit);
        }
    }

    /// Local estimates describe candidate text only, not provider-billed tokens.
    /// Replaced context includes its placeholder cost; memory removal has none.
    pub fn action_metadata(&self, removals: &[usize]) -> BTreeMap<String, String> {
        let mut removed = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for (index, tokens) in self
            .candidate_indices
            .iter()
            .zip(&self.candidate_token_estimates)
        {
            before = before.saturating_add(*tokens);
            if removals.contains(index) {
                removed += 1;
                if self.category == DecisionCategory::ContextRelevance {
                    after = after.saturating_add(estimate_tokens(OMITTED_CONTEXT.len()));
                }
            } else {
                after = after.saturating_add(*tokens);
            }
        }
        BTreeMap::from([
            (
                "candidate_count".to_string(),
                self.candidate_indices.len().to_string(),
            ),
            (
                "retained_count".to_string(),
                (self.candidate_indices.len() - removed).to_string(),
            ),
            ("removed_count".to_string(), removed.to_string()),
            (
                "estimated_candidate_tokens_before".to_string(),
                before.to_string(),
            ),
            (
                "estimated_candidate_tokens_after".to_string(),
                after.to_string(),
            ),
            (
                "token_estimate_basis".to_string(),
                "candidate_utf8_bytes_div_4".to_string(),
            ),
        ])
    }

    pub fn questions(&self) -> Vec<PreparedQuestion> {
        candidate_questions(&self.state, self.category, candidate_key(self.category))
            .unwrap_or_default()
    }

    pub fn removals(
        &self,
        decisions: &[ActiveDecision],
        request_id: &str,
        turn: u64,
    ) -> Vec<usize> {
        dropped_candidate_indices_with_options(
            decisions,
            self.category,
            self.candidate_indices.len(),
            request_id,
            turn,
            SystemTime::now(),
            &self.options,
        )
        .into_iter()
        .map(|index| self.candidate_indices[index])
        .collect()
    }
}

fn estimate_tokens(bytes: usize) -> usize {
    bytes / 4 + usize::from(bytes % 4 != 0)
}

fn candidate_key(category: DecisionCategory) -> &'static str {
    match category {
        DecisionCategory::ContextRelevance => "context_candidates",
        DecisionCategory::CodeSearchRelevance => "code_search_candidates",
        DecisionCategory::MemoryRelevance => "memory_candidates",
        _ => "unsupported_candidates",
    }
}

fn pinned(value: &Value) -> bool {
    value.as_object().is_some_and(pinned_fields)
}

fn pinned_fields(fields: &serde_json::Map<String, Value>) -> bool {
    [
        "pinned",
        "mandatory",
        "current",
        "currentTask",
        "isError",
        "error",
        "edited",
        "instructions",
    ]
    .iter()
    .any(|key| {
        fields
            .get(*key)
            .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
    })
}

fn memory_eligible(hit: &MemoryHit) -> bool {
    hit.scope != MemoryScope::Session
        && hit.entry.kind.as_str() == "memory"
        && !matches!(hit.entry.source.as_str(), "user" | "manual")
        && !pinned_fields(&hit.entry.metadata)
        && !hit.sources.is_empty()
        && !hit
            .sources
            .iter()
            .any(|source| source.origin == MemoryOrigin::User)
}

pub fn prepare_memory(hits: &[MemoryHit], query: &str) -> PreparedRelevance {
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    for (index, hit) in hits.iter().enumerate() {
        if !seen.insert(&hit.id) {
            return PreparedRelevance::new(DecisionCategory::MemoryRelevance, query, Vec::new());
        }
        if candidates.len() < MAX_FILTER_CANDIDATES && memory_eligible(hit) {
            // Neither source paths nor full entry bodies are sent.
            candidates.push((
                index,
                pi_jev::redact::bounded_excerpt(&hit.entry.content, MAX_FILTER_EXCERPT_CHARS),
                hit.entry.content.len(),
            ));
        }
    }
    PreparedRelevance::new(DecisionCategory::MemoryRelevance, query, candidates)
}

pub fn apply_memory(hits: Vec<MemoryHit>, removals: &[usize]) -> Vec<MemoryHit> {
    hits.into_iter()
        .enumerate()
        .filter_map(|(index, hit)| {
            if removals.contains(&index) && memory_eligible(&hit) {
                None
            } else {
                Some(hit)
            }
        })
        .collect()
}

fn signed(value: &Value) -> bool {
    [
        "textSignature",
        "thoughtSignature",
        "signature",
        "providerContext",
    ]
    .iter()
    .any(|key| value.get(*key).is_some())
        || value
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks.iter().any(|block| {
                    [
                        "textSignature",
                        "thoughtSignature",
                        "signature",
                        "providerContext",
                    ]
                    .iter()
                    .any(|key| block.get(*key).is_some())
                })
            })
}

fn text_bytes(content: &Value) -> usize {
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block["text"].as_str())
        .map(str::len)
        .fold(0usize, usize::saturating_add)
}

fn text_excerpt(content: &Value) -> Option<String> {
    let blocks = content.as_array()?;
    if blocks.is_empty()
        || blocks.len() > MAX_SCAN_BLOCKS
        || blocks
            .iter()
            .any(|block| block["type"] != "text" || !block["text"].is_string())
    {
        return None;
    }
    let mut text = String::new();
    for block in blocks {
        let remaining = MAX_FILTER_EXCERPT_CHARS.saturating_sub(text.chars().count());
        if remaining == 0 {
            break;
        }
        text.extend(block["text"].as_str()?.chars().take(remaining));
    }
    Some(pi_jev::redact::bounded_excerpt(
        &text,
        MAX_FILTER_EXCERPT_CHARS,
    ))
}

pub(crate) fn query_from_messages(messages: &[Value]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .map(|message| {
            message["content"]
                .as_str()
                .map(|text| pi_jev::redact::bounded_excerpt(text, MAX_FILTER_EXCERPT_CHARS))
                .or_else(|| text_excerpt(&message["content"]))
                .unwrap_or_default()
        })
        .unwrap_or_default()
}

fn safe_read_call(call: &Value) -> bool {
    if signed(call) {
        return false;
    }
    let Some(name) = call["name"].as_str() else {
        return false;
    };
    if !READ_SEARCH_TOOLS.contains(&name) {
        return false;
    }
    let Some(arguments) = call["arguments"].as_object() else {
        return false;
    };
    // Instructions and policy files are never optional context. Unknown read
    // paths are pinned rather than guessed. Arguments remain local.
    if matches!(name, "read" | "read_file") {
        let Some(path) = arguments
            .get("path")
            .or_else(|| arguments.get("file_path"))
            .and_then(Value::as_str)
        else {
            return false;
        };
        if path.len() > 4096 {
            return false;
        }
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".md")
            || lower.ends_with(".mdc")
            || lower.contains("instruction")
            || lower.contains("prompt")
            || lower.contains("agents.")
            || lower.contains("claude.")
        {
            return false;
        }
    }
    !pinned_fields(arguments)
}

/// An exact, successful read/search result with one matching earlier call.
/// The current user suffix and newest six messages remain pinned. Oversized
/// histories fail open rather than hiding duplicate IDs outside a scan window.
fn eligible_context_indices(messages: &[Value]) -> Vec<usize> {
    if messages.len() > MAX_SCAN_MESSAGES
        || messages.iter().any(|message| {
            message.get("providerContext").is_some()
                || (message["role"] == "user" && signed(message))
        })
    {
        return Vec::new();
    }
    let Some(current_user) = messages
        .iter()
        .rposition(|message| message["role"] == "user")
    else {
        return Vec::new();
    };
    let candidate_end = current_user.min(messages.len().saturating_sub(MIN_RECENT_MESSAGES));
    let mut candidates = Vec::new();
    for index in 0..candidate_end {
        let message = &messages[index];
        if message["role"] != "toolResult"
            || message["isError"] != false
            || pinned(message)
            || pinned(&message["details"])
            || signed(message)
        {
            continue;
        }
        let Some(name) = message["toolName"].as_str() else {
            continue;
        };
        let Some(id) = message["toolCallId"].as_str().filter(|id| !id.is_empty()) else {
            continue;
        };
        if !READ_SEARCH_TOOLS.contains(&name) {
            continue;
        }
        let Some(excerpt) = text_excerpt(&message["content"]) else {
            continue;
        };
        if excerpt.is_empty()
            || excerpt.starts_with(OMITTED_CONTEXT)
            || text_bytes(&message["content"]) <= OMITTED_CONTEXT.len()
        {
            continue;
        }
        let mut matching = 0usize;
        let mut safe = false;
        for (call_index, prior) in messages.iter().enumerate() {
            if prior["role"] != "assistant" {
                continue;
            }
            if let Some(content) = prior["content"].as_array() {
                if content.len() > MAX_SCAN_BLOCKS {
                    return Vec::new();
                }
                for call in content {
                    if call["type"] == "toolCall" && call["id"] == id {
                        matching += 1;
                        safe = call_index < index
                            && !signed(prior)
                            && call["name"] == name
                            && safe_read_call(call);
                    }
                }
            }
        }
        let results = messages
            .iter()
            .filter(|other| other["role"] == "toolResult" && other["toolCallId"] == id)
            .count();
        if matching == 1 && results == 1 && safe {
            candidates.push(index);
            if candidates.len() == MAX_FILTER_CANDIDATES {
                break;
            }
        }
    }
    candidates
}

pub fn prepare_context(messages: &[Value]) -> PreparedRelevance {
    let candidates = eligible_context_indices(messages)
        .into_iter()
        .filter_map(|index| {
            text_excerpt(&messages[index]["content"]).map(|excerpt| {
                let bytes = text_bytes(&messages[index]["content"]);
                (index, excerpt, bytes)
            })
        })
        .collect();
    PreparedRelevance::new(
        DecisionCategory::ContextRelevance,
        &query_from_messages(messages),
        candidates,
    )
}

pub fn apply_context(mut messages: Vec<Value>, removals: &[usize]) -> Vec<Value> {
    let eligible = eligible_context_indices(&messages);
    for index in removals.iter().filter(|index| eligible.contains(index)) {
        messages[*index]["content"] = json!([{"type":"text", "text":OMITTED_CONTEXT}]);
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation(name: &str, path: &str) -> Vec<Value> {
        vec![
            json!({"role":"user","content":"old request"}),
            json!({"role":"assistant","content":[{"type":"toolCall","id":"c1","name":name,"arguments":{"path":path}}]}),
            json!({"role":"toolResult","toolCallId":"c1","toolName":name,"isError":false,"content":[{"type":"text","text":"old successful result ".repeat(16)}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 1"}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 2"}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 3"}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 4"}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 5"}]}),
            json!({"role":"assistant","content":[{"type":"text","text":"old note 6"}]}),
            json!({"role":"user","content":"current request"}),
        ]
    }

    #[test]
    fn short_results_never_expand_or_trigger_relevance_work() {
        let mut messages = conversation("search", "");
        messages[2]["content"][0]["text"] = json!("short result");
        assert!(prepare_context(&messages).questions().is_empty());
        assert_eq!(apply_context(messages.clone(), &[2]), messages);
        messages[2]["content"][0]["text"] = json!("x".repeat(OMITTED_CONTEXT.len()));
        assert!(prepare_context(&messages).questions().is_empty());
    }

    #[test]
    fn context_filter_keeps_pairs_users_and_restores_from_source() {
        let original = conversation("read", "src/main.rs");
        let prepared = prepare_context(&original);
        assert_eq!(prepared.candidate_indices, vec![2]);
        let filtered = apply_context(original.clone(), &[2]);
        assert_eq!(filtered.len(), original.len());
        assert_eq!(filtered[0], original[0]);
        assert_eq!(filtered[1], original[1]);
        assert_eq!(filtered[3], original[3]);
        assert_eq!(filtered[2]["toolCallId"], original[2]["toolCallId"]);
        assert_eq!(filtered[2]["content"][0]["text"], OMITTED_CONTEXT);
        assert_eq!(apply_context(original.clone(), &[]), original);
        assert!(prepare_context(&filtered).candidate_indices.is_empty());
    }

    #[test]
    fn unknown_current_error_edit_instruction_and_custom_are_pinned() {
        for (name, path) in [
            ("ipython", "code"),
            ("edit", "src.rs"),
            ("read", "AGENTS.md"),
            ("read_file", "system-prompt.txt"),
        ] {
            let messages = conversation(name, path);
            assert!(prepare_context(&messages).candidate_indices.is_empty());
            assert_eq!(apply_context(messages.clone(), &[2]), messages);
        }
        for key in ["pinned", "currentTask", "edited", "instructions"] {
            let mut messages = conversation("search", "");
            messages[2]["details"] = json!({key:true});
            assert!(prepare_context(&messages).candidate_indices.is_empty());
        }
        let mut error = conversation("grep", "");
        error[2]["isError"] = json!(true);
        assert!(prepare_context(&error).candidate_indices.is_empty());
        let mut current = conversation("grep", "");
        current.pop();
        assert!(prepare_context(&current).candidate_indices.is_empty());
        let mut custom = conversation("grep", "");
        custom[2]["role"] = json!("custom");
        assert!(prepare_context(&custom).candidate_indices.is_empty());
    }

    #[test]
    fn recent_signed_checkpoint_and_oversized_histories_are_pinned() {
        let messages = conversation("search", "");
        let recent = vec![
            messages[0].clone(),
            messages[1].clone(),
            messages[2].clone(),
            messages.last().unwrap().clone(),
        ];
        assert!(prepare_context(&recent).questions().is_empty());
        for (index, key) in [
            (1, "textSignature"),
            (2, "thoughtSignature"),
            (0, "providerContext"),
        ] {
            let mut signed_messages = messages.clone();
            signed_messages[index][key] = json!("opaque");
            assert!(prepare_context(&signed_messages).questions().is_empty());
            assert_eq!(
                apply_context(signed_messages.clone(), &[2]),
                signed_messages
            );
        }
        let mut signed_block = messages.clone();
        signed_block[1]["content"][0]["thoughtSignature"] = json!("opaque");
        assert!(prepare_context(&signed_block).questions().is_empty());
        let mut oversized = messages.clone();
        oversized.extend((0..MAX_SCAN_MESSAGES).map(|_| json!({"role":"assistant","content":[]})));
        assert!(prepare_context(&oversized).questions().is_empty());
        let mut duplicate_after = messages.clone();
        duplicate_after.push(messages[1].clone());
        assert!(prepare_context(&duplicate_after).questions().is_empty());
    }

    #[test]
    fn unmatched_duplicate_and_multimodal_results_are_pinned() {
        let mut messages = conversation("search", "");
        messages[2]["toolCallId"] = json!("unknown");
        assert!(prepare_context(&messages).candidate_indices.is_empty());
        let mut messages = conversation("search", "");
        messages.insert(3, messages[2].clone());
        assert!(prepare_context(&messages).candidate_indices.is_empty());
        let mut messages = conversation("search", "");
        messages[2]["content"][0] = json!({"type":"image","data":"abc"});
        assert!(prepare_context(&messages).candidate_indices.is_empty());
    }

    fn memory_hit(id: &str) -> MemoryHit {
        serde_json::from_value(json!({
            "id":id,"scope":"project","score":1.0,"matched":["fixture"],"freshness":"unknown",
            "sources":[{"id":"source","origin":"tool","sha256":"digest"}],
            "entry":{"id":id,"kind":"memory","title":"Fixture","content":"Optional fixture information.",
                "path":"general","metadata":{},"source":"refine","created_at":"2026-01-01","updated_at":"2026-01-01","version":1}
        })).unwrap()
    }

    #[test]
    fn memory_filter_removes_only_retrieved_optional_entries_and_is_reversible() {
        let hits = vec![
            memory_hit("project:memory:a"),
            memory_hit("project:memory:b"),
        ];
        let settings = crate::core::memory::store::default_memory_settings();
        let original = crate::core::memory::search::recall_memory(&hits, &settings, None);
        assert_eq!(original.ids.len(), 2);
        let prepared = prepare_memory(&hits, "fixture");
        assert_eq!(prepared.candidate_indices, vec![0, 1]);
        let decisions: Vec<ActiveDecision> = ["drop", "keep"]
            .iter()
            .enumerate()
            .map(|(index, value)| ActiveDecision {
                category: DecisionCategory::MemoryRelevance,
                question_id: format!("memory_relevance.{index}"),
                value: value.to_string(),
                confidence: 0.99,
                response_model: None,
                request_id: "r1".to_string(),
                turn: 1,
                decided_at: SystemTime::now(),
            })
            .collect();
        let removals = prepared.removals(&decisions, "r1", 1);
        let filtered = apply_memory(hits.clone(), &removals);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], hits[1]);
        let rendered = crate::core::memory::search::recall_memory(&filtered, &settings, None);
        assert_eq!(rendered.ids, vec![hits[1].id.clone()]);
        let restored = crate::core::memory::search::recall_memory(&hits, &settings, None);
        assert_eq!(restored.text, original.text);
        assert!(prepared.removals(&decisions[..1], "r1", 1).is_empty());
        assert!(prepared.removals(&decisions, "other", 1).is_empty());
    }

    #[test]
    fn pinned_session_user_instruction_unknown_and_duplicate_memories_stay() {
        let mut session = memory_hit("session");
        session.scope = MemoryScope::Session;
        let mut user = memory_hit("user");
        user.sources[0].origin = MemoryOrigin::User;
        let mut unknown = memory_hit("unknown");
        unknown.sources.clear();
        let mut manual = memory_hit("manual");
        manual.entry.source = "manual".to_string();
        let mut prompt = memory_hit("prompt");
        prompt.entry.kind = serde_json::from_value(json!("prompt")).unwrap();
        let mut pinned_hit = memory_hit("pinned");
        pinned_hit
            .entry
            .metadata
            .insert("pinned".to_string(), json!(true));
        let hits = vec![session, user, unknown, manual, prompt, pinned_hit];
        assert!(prepare_memory(&hits, "fixture").questions().is_empty());
        assert_eq!(apply_memory(hits.clone(), &[0, 1, 2, 3, 4, 5]), hits);
        let hit = memory_hit("duplicate");
        assert!(prepare_memory(&[hit.clone(), hit], "fixture")
            .questions()
            .is_empty());
    }

    #[test]
    fn native_recall_renderer_filters_injection_without_mutating_store() {
        // The native memory lock identity lowercases paths, including on Unix.
        let dir = tempfile::Builder::new()
            .prefix(&format!("jev-memory-{}", uuid::Uuid::new_v4()))
            .rand_bytes(0)
            .tempdir()
            .unwrap();
        let project = dir.path().join("project");
        let agent = dir.path().join("agent");
        std::fs::create_dir_all(&project).unwrap();
        let service = crate::core::memory::service::MemoryService::new(
            &project.to_string_lossy(),
            &agent.to_string_lossy(),
            None,
        )
        .unwrap();
        let before = serde_json::to_value(service.store.read().unwrap()).unwrap();
        let hits = vec![memory_hit("first"), memory_hit("second")];
        let original = service.render_recall(&hits);
        let filtered = service.render_recall(&apply_memory(hits.clone(), &[0]));
        assert_eq!(filtered.ids, vec!["second"]);
        assert_eq!(service.render_recall(&hits).text, original.text);
        assert_eq!(
            serde_json::to_value(service.store.read().unwrap()).unwrap(),
            before
        );
    }

    #[test]
    fn candidate_caps_keep_unscored_entries_and_reject_invalid_config() {
        let hits: Vec<MemoryHit> = (0..20)
            .map(|index| memory_hit(&format!("item-{index}")))
            .collect();
        let mut prepared = prepare_memory(&hits, "fixture");
        assert_eq!(prepared.candidate_indices.len(), MAX_FILTER_CANDIDATES);
        prepared.configure(&FilteringOptions {
            max_candidates: 2,
            ..Default::default()
        });
        assert_eq!(prepared.questions().len(), 2);
        prepared.configure(&FilteringOptions {
            max_candidates: 0,
            ..Default::default()
        });
        assert!(prepared.questions().is_empty());
    }

    #[test]
    fn local_candidate_metrics_match_selected_indices_and_placeholder_cost() {
        let hits = vec![memory_hit("first"), memory_hit("second")];
        let mut prepared = prepare_memory(&hits, "fixture");
        let baseline = prepared.action_metadata(&[]);
        assert_eq!(baseline["candidate_count"], "2");
        assert_eq!(baseline["retained_count"], "2");
        assert_eq!(baseline["removed_count"], "0");
        let actual = prepared.action_metadata(&[0, 0, 99]);
        assert_eq!(actual["retained_count"], "1");
        assert_eq!(actual["removed_count"], "1");
        assert_eq!(
            actual["estimated_candidate_tokens_after"],
            estimate_tokens(hits[1].entry.content.len()).to_string()
        );
        assert_eq!(
            actual["estimated_candidate_tokens_before"],
            baseline["estimated_candidate_tokens_before"]
        );
        assert!(prepared
            .state
            .get("estimated_candidate_tokens_before")
            .is_none());
        prepared.configure(&FilteringOptions {
            max_candidates: 1,
            ..Default::default()
        });
        assert_eq!(prepared.action_metadata(&[0])["candidate_count"], "1");
        assert_eq!(
            prepared.action_metadata(&[0])["estimated_candidate_tokens_after"],
            "0"
        );
        let context = prepare_context(&conversation("search", ""));
        assert_eq!(
            context.action_metadata(&[2])["estimated_candidate_tokens_after"],
            estimate_tokens(OMITTED_CONTEXT.len()).to_string()
        );
    }

    #[test]
    fn payload_redacts_and_does_not_include_arguments() {
        let mut messages = conversation("search", "secret-path");
        messages[2]["content"][0]["text"] = json!(format!(
            "password=hunter2-the-password {}",
            "data ".repeat(40)
        ));
        let state = prepare_context(&messages).state.to_string();
        assert!(!state.contains("secret-path"));
        assert!(!state.contains("hunter2-the-password"));
        assert!(!state.contains("arguments"));
    }
}
