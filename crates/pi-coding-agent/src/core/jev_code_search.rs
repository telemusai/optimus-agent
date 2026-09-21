//! Explicit search candidates at the provider-context boundary. Retrieval stays local.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use pi_jev::filtering::FilteringOptions;
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};

use super::jev_retrieval::PreparedRelevance;

pub(crate) const MAX_PRESENTATION_BYTES: usize = 512 * 1024;
const MAX_SCORED: usize = 64;

/// One recognized `rlm.code-search/1` presentation in the request window.
///
/// `scored` lists the non-pinned eligible candidates (envelope index + excerpt)
/// in envelope order, bounded by `MAX_SCORED`. Reranking reorders only these,
/// among their own positions; pinned candidates are position anchors and every
/// original value stays in session history and the Python variable.
pub struct SearchPresentation {
    pub message_index: usize,
    pub batches: Vec<PreparedRelevance>,
    pub fingerprint: String,
    pub scored: Vec<(usize, String)>,
    envelope: Value,
}

impl SearchPresentation {
    /// Legacy projection: drop optional candidates only. Behavior is unchanged.
    pub fn project(&self, removals: &[usize]) -> Option<Value> {
        self.project_with_order(removals, None, None, None)
    }

    /// Filter (set) then rerank (order) projection for the request copy only.
    ///
    /// `removals` are envelope indices of optional candidates to drop, exactly
    /// as [`SearchPresentation::project`] accepts them. `order` is the typed
    /// rerank result: a permutation of `0..k` over the RETAINED scored
    /// candidates' ordinals (scored candidates are listed by
    /// [`SearchPresentation::rerank_inputs`], in envelope order). The first `k`
    /// retained eligible positions receive the reordered values; any further
    /// retained eligible candidates keep their original relative order
    /// (`jev_rerank_scope: "prefix"`). Pinned candidates are position anchors
    /// and never move. An invalid or incomplete order is IGNORED (filter-only
    /// projection): a failed rerank retains the original surviving order.
    /// Returns `None` when nothing changes, so no untruthful metadata is added.
    pub fn project_with_order(
        &self,
        removals: &[usize],
        order: Option<&[usize]>,
        model: Option<&str>,
        scope: Option<&str>,
    ) -> Option<Value> {
        let candidates = self.envelope["candidates"].as_array()?;
        let eligible: BTreeSet<usize> = self
            .batches
            .iter()
            .flat_map(|batch| batch.candidate_indices.iter().copied())
            .collect();
        let scored_positions: Vec<usize> = self
            .scored
            .iter()
            .map(|(index, _)| *index)
            .filter(|index| eligible.contains(index))
            .collect();
        let dropped: BTreeSet<usize> = removals
            .iter()
            .copied()
            .filter(|index| eligible.contains(index))
            .collect();
        // Retained scored candidates, in envelope order (their ordinals are
        // exactly 0..retained.len() in rerank_inputs order).
        let retained: Vec<usize> = scored_positions
            .iter()
            .copied()
            .filter(|index| !dropped.contains(index))
            .collect();
        // Resolve the reorder: `order` must be a permutation of 0..k for some
        // k <= retained.len(). Anything else fails open to the original order.
        let mut ordered_retained: Vec<usize> = retained.clone();
        let mut order_applied = false;
        if let Some(order) = order {
            let k = order.len();
            if k > 0 && k <= retained.len() && (0..k).all(|ordinal| order.contains(&ordinal)) {
                let mut head: Vec<usize> = order.iter().map(|ordinal| retained[*ordinal]).collect();
                let tail: Vec<usize> = retained[k..].to_vec();
                head.extend(tail);
                ordered_retained = head;
                order_applied = ordered_retained != retained;
            }
        }
        if dropped.is_empty() && !order_applied {
            return None;
        }
        if dropped.len() == candidates.len() {
            return None;
        }
        let mut next_value = ordered_retained.into_iter();
        let projected: Vec<Value> = candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !dropped.contains(index))
            .map(|(index, value)| {
                if eligible.contains(&index) {
                    next_value
                        .next()
                        .map(|candidate| candidates[candidate].clone())
                        .unwrap_or_else(|| value.clone())
                } else {
                    value.clone()
                }
            })
            .collect();
        let mut envelope = self.envelope.clone();
        envelope["candidates"] = Value::Array(projected);
        if !dropped.is_empty() {
            envelope["jev_omitted_candidates"] = json!(dropped.len());
        }
        if order_applied {
            envelope["jev_reranked"] = json!(true);
            envelope["jev_rerank_candidates"] = json!(retained.len());
            envelope["jev_rerank_scope"] = json!(scope.unwrap_or("all"));
            if let Some(model) = model {
                envelope["jev_rerank_model"] = json!(model);
            }
            envelope["jev_rerank_notice"] = json!("Optional candidates reordered for this request by host-side relevance scoring; the original order remains in session history and the Python variable.");
        }
        if !dropped.is_empty() || order_applied {
            envelope["jev_notice"] = json!("Optional candidates omitted for this request; original search results remain in session history and the Python variable.");
        }
        Some(json!([{"type":"text", "text":serde_json::to_string(&envelope).ok()?}]))
    }

    /// Typed inputs for the rerank stage: `(ordinal, excerpt)` for the first
    /// `max` retained scored candidates, in envelope order. Ordinals are local
    /// to this list and map back through the retained scored positions.
    pub fn rerank_inputs(&self, max: usize, removals: &[usize]) -> Vec<(usize, String)> {
        let eligible: BTreeSet<usize> = self
            .batches
            .iter()
            .flat_map(|batch| batch.candidate_indices.iter().copied())
            .collect();
        let dropped: BTreeSet<usize> = removals.iter().copied().collect();
        self.scored
            .iter()
            .filter(|(index, _)| eligible.contains(index) && !dropped.contains(index))
            .take(max)
            .enumerate()
            .map(|(ordinal, (_, excerpt))| (ordinal, excerpt.clone()))
            .collect()
    }

    /// Truthful rerank telemetry for the correlation ledger: what was scored,
    /// what actually moved, and the declared scope.
    pub fn rerank_action_metadata(
        &self,
        removals: &[usize],
        order: Option<&[usize]>,
        scope: &str,
    ) -> BTreeMap<String, String> {
        let eligible: BTreeSet<usize> = self
            .batches
            .iter()
            .flat_map(|batch| batch.candidate_indices.iter().copied())
            .collect();
        let dropped: BTreeSet<usize> = removals.iter().copied().collect();
        let retained: Vec<usize> = self
            .scored
            .iter()
            .map(|(index, _)| *index)
            .filter(|index| eligible.contains(index) && !dropped.contains(index))
            .collect();
        let mut moved = 0usize;
        if let Some(order) = order {
            for (position, ordinal) in order.iter().enumerate() {
                if retained.get(*ordinal).copied() != retained.get(position).copied() {
                    moved += 1;
                }
            }
        }
        BTreeMap::from([
            ("rerank_retained".to_string(), retained.len().to_string()),
            ("rerank_reordered".to_string(), moved.to_string()),
            ("rerank_scope".to_string(), scope.to_string()),
            ("rerank_order_applied".to_string(), order.is_some().to_string()),
        ])
    }
}

/// Pinned/instruction-class candidate: position anchor, never scored and
/// never a line-find source. Shared with `jev_line_find`.
pub(crate) fn pinned(candidate: &Value) -> bool {
    let path = candidate["path"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    candidate["mandatory"] == true
        || path.ends_with(".md")
        || path.ends_with(".mdx")
        || path.ends_with(".mdc")
        || path.contains("agents")
        || path.contains("instruction")
        || path.contains("prompt")
        || path.contains("claude")
}

fn candidate_valid(candidate: &Value) -> bool {
    let Some(object) = candidate.as_object() else {
        return false;
    };
    object.keys().all(|key| {
        matches!(
            key.as_str(),
            "kind" | "path" | "line" | "snippet" | "mandatory"
        )
    }) && matches!(
        candidate["kind"].as_str(),
        Some("file" | "symbol" | "grep" | "reference" | "test")
    ) && candidate["path"]
        .as_str()
        .is_some_and(|path| !path.is_empty() && path.chars().count() <= 1024)
        && candidate
            .get("line")
            .is_none_or(|line| line.as_u64().is_some_and(|line| line > 0))
        && candidate.get("snippet").is_none_or(|text| {
            text.as_str()
                .is_some_and(|text| text.chars().count() <= 4096)
        })
        && candidate.get("mandatory").is_none_or(Value::is_boolean)
}

/// Recognize only an entire successful, paired search presentation, never raw
/// arbitrary tool text or mixed output. The provider-checkpoint prefix stays pinned.
pub fn prepare(
    messages: &[Value],
    query: &str,
    options: &FilteringOptions,
) -> Vec<SearchPresentation> {
    if messages.len() > 512 || query.trim().is_empty() || options.validate().is_err() {
        return Vec::new();
    }
    if messages.iter().any(|message| {
        message["content"]
            .as_array()
            .is_some_and(|blocks| blocks.len() > 64)
    }) {
        return Vec::new();
    }
    let start = messages
        .iter()
        .rposition(|message| {
            message
                .get("providerContext")
                .is_some_and(|value| !value.is_null())
        })
        .map_or(0, |index| index + 1);
    let mut presentations = Vec::new();
    let mut budget = MAX_SCORED;
    let mut scanned_bytes = 0usize;
    for (index, message) in messages.iter().enumerate().skip(start).rev() {
        if budget == 0 {
            break;
        }
        if message["role"] != "toolResult"
            || message["toolName"] != "ipython"
            || message["isError"] != false
            || message.get("textSignature").is_some()
            || ["pinned", "mandatory", "instructions", "edited"]
                .iter()
                .any(|key| {
                    message
                        .get(*key)
                        .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                        || message["details"]
                            .get(*key)
                            .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                })
        {
            continue;
        }
        let details = &message["details"];
        if details["status"] != "ok"
            || details["kernelRestarted"] == true
            || ["stderr", "backgroundOutput", "result"]
                .iter()
                .any(|key| details.get(*key).is_some_and(|v| !v.is_null() && v != ""))
            || ["diffs", "attachments", "sentAgentMessages"]
                .iter()
                .any(|key| {
                    details.get(*key).is_some_and(|v| {
                        !v.is_null() && v.as_array().is_none_or(|items| !items.is_empty())
                    })
                })
        {
            continue;
        }
        let Some(blocks) = message["content"]
            .as_array()
            .filter(|blocks| blocks.len() == 1)
        else {
            continue;
        };
        if blocks[0]["type"] != "text" || blocks[0].get("textSignature").is_some() {
            continue;
        }
        let Some(text) = blocks[0]["text"]
            .as_str()
            .filter(|text| text.len() <= MAX_PRESENTATION_BYTES)
        else {
            continue;
        };
        scanned_bytes += text.len();
        if scanned_bytes > 2 * 1024 * 1024 {
            break;
        }
        let Ok(envelope) = serde_json::from_str::<Value>(text) else {
            continue;
        };
        if envelope["schema"] != "rlm.code-search/1"
            || envelope.as_object().is_none_or(|object| object.len() != 2)
        {
            continue;
        }
        let Some(candidates) = envelope["candidates"]
            .as_array()
            .filter(|items| !items.is_empty() && items.len() <= 500)
        else {
            continue;
        };
        if !candidates.iter().all(candidate_valid) {
            continue;
        }
        let Some(id) = message["toolCallId"].as_str().filter(|id| !id.is_empty()) else {
            continue;
        };
        let calls: Vec<_> = messages
            .iter()
            .enumerate()
            .flat_map(|(at, msg)| {
                msg["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(move |call| {
                        (msg["role"] == "assistant"
                            && call["type"] == "toolCall"
                            && call["id"] == id)
                            .then_some((at, call))
                    })
            })
            .collect();
        if calls.len() != 1
            || calls[0].0 < start
            || calls[0].0 >= index
            || calls[0].1["name"] != "ipython"
            || messages
                .iter()
                .filter(|msg| msg["role"] == "toolResult" && msg["toolCallId"] == id)
                .count()
                != 1
        {
            continue;
        }
        let selected: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| !pinned(candidate))
            .take(budget)
            .map(|(index, candidate)| {
                let excerpt = format!(
                    "{} {}:{} {}",
                    candidate["kind"].as_str().unwrap_or_default(),
                    candidate["path"].as_str().unwrap_or_default(),
                    candidate["line"],
                    candidate["snippet"].as_str().unwrap_or_default()
                );
                (index, excerpt, candidate.to_string().len())
            })
            .collect();
        budget -= selected.len();
        let scored: Vec<(usize, String)> = selected
            .iter()
            .map(|(candidate_index, excerpt, _)| (*candidate_index, excerpt.clone()))
            .collect();
        let batches = selected
            .chunks(options.max_candidates)
            .map(|chunk| {
                let mut batch = PreparedRelevance::new(
                    DecisionCategory::CodeSearchRelevance,
                    query,
                    chunk.to_vec(),
                );
                batch.configure(options);
                batch
            })
            .collect();
        presentations.push(SearchPresentation {
            message_index: index,
            batches,
            fingerprint: pi_jev::snapshot::fingerprint_of(&json!([id, text, query, options])),
            scored,
            envelope,
        });
    }
    presentations
}

/// The most recent parseable code-search envelope in the request window, as a
/// line-find INPUT source only (ROOT CONTRACT v1: line matching judges
/// explicitly supplied text).
///
/// Unlike [`prepare`], this does NOT require the pristine two-key envelope:
/// the filter and rerank stages add disclosure keys to the request copy, and
/// the CURRENT (filtered+reranked) envelope is exactly the text line-find must
/// read. The paired-call discipline is also not re-checked here; the envelope
/// text itself is the explicit presentation, and this locator performs no
/// scoring, removal or reordering. Most recent wins; bounded like [`prepare`].
pub fn recent_code_search_envelope(messages: &[Value]) -> Option<(usize, Value)> {
    if messages.len() > 512
        || messages
            .iter()
            .any(|message| {
                message["content"]
                    .as_array()
                    .is_some_and(|blocks| blocks.len() > 64)
            })
    {
        return None;
    }
    let start = messages
        .iter()
        .rposition(|message| {
            message
                .get("providerContext")
                .is_some_and(|value| !value.is_null())
        })
        .map_or(0, |index| index + 1);
    for (index, message) in messages.iter().enumerate().skip(start).rev() {
        if message["role"] != "toolResult"
            || message["toolName"] != "ipython"
            || message["isError"] != false
            || message.get("textSignature").is_some()
            || ["pinned", "mandatory", "instructions", "edited"]
                .iter()
                .any(|key| {
                    message
                        .get(*key)
                        .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                        || message["details"]
                            .get(*key)
                            .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                })
        {
            continue;
        }
        let details = &message["details"];
        if details["status"] != "ok"
            || details["kernelRestarted"] == true
            || ["stderr", "backgroundOutput", "result"]
                .iter()
                .any(|key| details.get(*key).is_some_and(|v| !v.is_null() && v != ""))
            || ["diffs", "attachments", "sentAgentMessages"]
                .iter()
                .any(|key| {
                    details.get(*key).is_some_and(|v| {
                        !v.is_null() && v.as_array().is_none_or(|items| !items.is_empty())
                    })
                })
        {
            continue;
        }
        let Some(blocks) = message["content"].as_array().filter(|blocks| blocks.len() == 1) else {
            continue;
        };
        if blocks[0]["type"] != "text" || blocks[0].get("textSignature").is_some() {
            continue;
        }
        let Some(text) = blocks[0]["text"]
            .as_str()
            .filter(|text| text.len() <= MAX_PRESENTATION_BYTES)
        else {
            continue;
        };
        let Ok(envelope) = serde_json::from_str::<Value>(text) else {
            continue;
        };
        if envelope["schema"] != "rlm.code-search/1" {
            continue;
        }
        let Some(candidates) = envelope["candidates"]
            .as_array()
            .filter(|items| !items.is_empty() && items.len() <= 500)
        else {
            continue;
        };
        if !candidates.iter().all(candidate_valid) {
            continue;
        }
        return Some((index, envelope));
    }
    None
}

// Remember queued Compare observations for exact input and policy, avoiding
// repeated shadow calls. Active never reuses decisions from an earlier boundary.
// No raw source text is retained.
type Cache = BTreeMap<String, Instant>;
fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub fn already_observed(key: &str) -> bool {
    let mut entries = cache().lock().unwrap_or_else(|p| p.into_inner());
    entries.retain(|_, at| at.elapsed() < Duration::from_secs(1800));
    entries.contains_key(key)
}
pub fn remember_observation(key: String) {
    let mut entries = cache().lock().unwrap_or_else(|p| p.into_inner());
    if entries.len() >= 128 {
        if let Some(oldest) = entries
            .iter()
            .min_by_key(|(_, at)| **at)
            .map(|(key, _)| key.clone())
        {
            entries.remove(&oldest);
        }
    }
    entries.insert(key, Instant::now());
}

#[cfg(test)]
mod tests {
    use super::*;
    fn messages(count: usize) -> Vec<Value> {
        let candidates: Vec<_> = (0..count).map(|index| json!({"kind":"grep","path":format!("src/file{index}.rs"),"line":1,"snippet":"optional source ".repeat(30)})).collect();
        vec![
            json!({"role":"user","content":"Find session expiry"}),
            json!({"role":"assistant","content":[{"type":"toolCall","name":"ipython","id":"c1","arguments":{"code":"present(candidates)"}}]}),
            json!({"role":"toolResult","toolName":"ipython","toolCallId":"c1","isError":false,"details":{"status":"ok"},
                "content":[{"type":"text","text":json!({"schema":"rlm.code-search/1","candidates":candidates}).to_string()}]}),
        ]
    }
    #[test]
    fn batches_score_before_injection_without_removing_unscored_or_durable_content() {
        let original = messages(150);
        let plans = prepare(&original, "Find session expiry", &Default::default());
        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.batches.len(), 8);
        assert!(plan
            .batches
            .iter()
            .all(|batch| batch.questions().len() == 8));
        let projected = plan.project(&[0, 2, 99]).unwrap();
        let output: Value = serde_json::from_str(projected[0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(output["candidates"].as_array().unwrap().len(), 148);
        assert_eq!(output["candidates"][0]["path"], "src/file1.rs");
        assert!(output["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["path"] == "src/file99.rs"));
        assert!(plan.project(&[]).is_none());
        assert_eq!(original, messages(150));
    }
    #[test]
    fn instructions_mandatory_errors_mixed_output_and_checkpoint_prefix_remain_intact() {
        let original = messages(2);
        for key in ["isError", "textSignature"] {
            let mut input = original.clone();
            input[2][key] = json!(true);
            assert!(prepare(&input, "task", &Default::default()).is_empty());
        }
        let mut input = original.clone();
        input[2]["details"]["diffs"] = json!([{"path":"changed.rs"}]);
        assert!(prepare(&input, "task", &Default::default()).is_empty());
        let mut input = original.clone();
        input[2]["content"][0]["text"] = json!("prefix {\"schema\":\"rlm.code-search/1\"}");
        assert!(prepare(&input, "task", &Default::default()).is_empty());
        let mut input = original.clone();
        input.push(json!({"role":"user","providerContext":{"opaque":"checkpoint"}}));
        assert!(prepare(&input, "task", &Default::default()).is_empty());
        let mut input = original.clone();
        let mut data: Value =
            serde_json::from_str(input[2]["content"][0]["text"].as_str().unwrap()).unwrap();
        data["candidates"][0]["path"] = json!("AGENTS.md");
        data["candidates"][1]["mandatory"] = json!(true);
        input[2]["content"][0]["text"] = json!(data.to_string());
        let plan = prepare(&input, "task", &Default::default());
        assert!(plan[0].batches.is_empty());
        assert!(plan[0].project(&[0, 1]).is_none());
    }
    #[test]
    fn candidate_and_task_excerpts_are_redacted_and_cache_expires() {
        let plan = prepare(&messages(2), "password=hunter2-secret", &Default::default());
        assert!(!plan[0].batches[0]
            .state
            .to_string()
            .contains("hunter2-secret"));
        let key = format!("test-{}", uuid::Uuid::new_v4());
        remember_observation(key.clone());
        assert!(already_observed(&key));
        cache()
            .lock()
            .unwrap()
            .insert(key.clone(), Instant::now() - Duration::from_secs(1801));
        assert!(!already_observed(&key));
        assert!(!already_observed("unrelated"));
    }
    #[test]
    fn project_with_order_reorders_scored_only_and_keeps_pinned_anchors() {
        let mut original = messages(8);
        // Pin candidates 0 and 3 (positions stay fixed; they are never scored).
        let mut data: Value =
            serde_json::from_str(original[2]["content"][0]["text"].as_str().unwrap()).unwrap();
        data["candidates"][0]["mandatory"] = json!(true);
        data["candidates"][3]["mandatory"] = json!(true);
        original[2]["content"][0]["text"] = json!(data.to_string());
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        // Scored set excludes the pinned anchors (indices 0 and 3).
        let scored_positions: Vec<usize> = plan
            .scored
            .iter()
            .map(|(index, _)| *index)
            .collect();
        assert_eq!(scored_positions, vec![1, 2, 4, 5, 6, 7]);
        // Reorder the retained scored candidates: reverse their ordinals.
        let retained: Vec<usize> = plan.rerank_inputs(64, &[]).iter().map(|(ordinal, _)| *ordinal).collect();
        assert_eq!(retained.len(), 6);
        let order: Vec<usize> = [5usize, 4, 3, 2, 1, 0].to_vec();
        let projected = plan.project_with_order(&[], Some(&order), Some("mock"), Some("all")).unwrap();
        let output: Value = serde_json::from_str(projected[0]["text"].as_str().unwrap()).unwrap();
        let candidates = output["candidates"].as_array().unwrap();
        // Pinned anchors keep their exact positions.
        assert_eq!(candidates[0]["path"], "src/file0.rs");
        assert_eq!(candidates[3]["path"], "src/file3.rs");
        // The eligible positions now hold the reversed scored set.
        assert_eq!(candidates[1]["path"], "src/file7.rs");
        assert_eq!(candidates[2]["path"], "src/file6.rs");
        assert_eq!(candidates[4]["path"], "src/file5.rs");
        assert_eq!(candidates[5]["path"], "src/file4.rs");
        assert_eq!(candidates[6]["path"], "src/file2.rs");
        assert_eq!(candidates[7]["path"], "src/file1.rs");
        assert_eq!(output["jev_reranked"], json!(true));
        assert_eq!(output["jev_rerank_candidates"], json!(6));
        assert_eq!(output["jev_rerank_model"], json!("mock"));
        assert_eq!(output["jev_rerank_scope"], json!("all"));
        assert_eq!(original, {
            let mut fresh = messages(8);
            let mut data: Value =
                serde_json::from_str(fresh[2]["content"][0]["text"].as_str().unwrap()).unwrap();
            data["candidates"][0]["mandatory"] = json!(true);
            data["candidates"][3]["mandatory"] = json!(true);
            fresh[2]["content"][0]["text"] = json!(data.to_string());
            fresh
        });
    }

    #[test]
    fn invalid_or_identity_orders_fail_open_without_metadata() {
        let original = messages(6);
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        let retained: Vec<usize> = plan.rerank_inputs(64, &[]).iter().map(|(ordinal, _)| *ordinal).collect();
        assert_eq!(retained.len(), 6);
        // Not a permutation of 0..k and nothing dropped: nothing changes at all.
        let duplicate = [0usize, 0, 1, 2, 3, 4].to_vec();
        assert!(plan.project_with_order(&[], Some(&duplicate), None, None).is_none());
        // Identity order changes nothing: None, so no untruthful metadata.
        let identity = [0usize, 1, 2, 3, 4, 5].to_vec();
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        assert!(plan.project_with_order(&[], Some(&identity), None, None).is_none());
        // Oversized order (beyond the retained set) is ignored the same way.
        let oversized = [5usize, 4, 3, 2, 1, 0, 6].to_vec();
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        assert!(plan.project_with_order(&[], Some(&oversized), None, None).is_none());
        // With removals, an invalid order still produces the filter-only
        // projection: the failed rerank retains the original surviving order.
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        let projected = plan.project_with_order(&[2], Some(&duplicate), None, None).unwrap();
        let output: Value = serde_json::from_str(projected[0]["text"].as_str().unwrap()).unwrap();
        assert!(output.get("jev_reranked").is_none());
        assert_eq!(output["jev_omitted_candidates"], json!(1));
        let paths: Vec<&str> = output["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|candidate| candidate["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, ["src/file0.rs", "src/file1.rs", "src/file3.rs", "src/file4.rs", "src/file5.rs"]);
        // Filter+rerank compose: removals apply, then the order maps retained ordinals.
        let plan = prepare(&original, "Find session expiry", &Default::default()).remove(0);
        let inputs = plan.rerank_inputs(64, &[1, 4]);
        assert_eq!(inputs.len(), 4);
        let order = [3usize, 2, 1, 0].to_vec();
        let projected = plan.project_with_order(&[1, 4], Some(&order), None, None).unwrap();
        let output: Value = serde_json::from_str(projected[0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(output["jev_omitted_candidates"], json!(2));
        assert_eq!(output["candidates"].as_array().unwrap().len(), 4);
        let paths: Vec<&str> = output["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|candidate| candidate["path"].as_str().unwrap())
            .collect();
        // Retained scored positions are 0,2,3,5; the order [3,2,1,0] reverses
        // them, so the projected array reads file5, file3, file2, file0.
        assert_eq!(paths, ["src/file5.rs", "src/file3.rs", "src/file2.rs", "src/file0.rs"]);
        let metadata = plan.rerank_action_metadata(&[1, 4], Some(&order), "all");
        assert_eq!(metadata["rerank_retained"], "4");
        assert_eq!(metadata["rerank_reordered"], "4");
        assert_eq!(metadata["rerank_order_applied"], "true");
    }

    #[test]
    fn current_envelope_locator_sees_projected_and_pristine_envelopes() {
        // Pristine two-key envelope is found.
        let original = messages(3);
        let (index, envelope) = recent_code_search_envelope(&original).unwrap();
        assert_eq!(index, 2);
        assert_eq!(envelope["candidates"].as_array().unwrap().len(), 3);
        // A projected (filtered+reranked) request-copy envelope adds
        // disclosure keys: the locator must still see it, because line-find
        // judges the CURRENT supplied text, not the pristine presentation.
        let mut projected = original.clone();
        let mut data: Value =
            serde_json::from_str(projected[2]["content"][0]["text"].as_str().unwrap()).unwrap();
        data["jev_reranked"] = json!(true);
        data["jev_rerank_candidates"] = json!(3);
        data["jev_rerank_scope"] = json!("all");
        data["jev_rerank_notice"] = json!("Optional candidates reordered for this request");
        data["jev_notice"] = json!("Optional candidates omitted for this request");
        data["jev_omitted_candidates"] = json!(0);
        projected[2]["content"][0]["text"] = json!(data.to_string());
        let (index, envelope) = recent_code_search_envelope(&projected).unwrap();
        assert_eq!(index, 2);
        assert_eq!(envelope["candidates"][0]["path"], "src/file0.rs");
        // Non-envelope text and error results stay invisible.
        let mut noise = original.clone();
        noise[2]["content"][0]["text"] = json!("plain text");
        assert!(recent_code_search_envelope(&noise).is_none());
        let mut failed = original.clone();
        failed[2]["isError"] = json!(true);
        assert!(recent_code_search_envelope(&failed).is_none());
        // A checkpoint prefix pins the window: presentations before it are out.
        let mut pinned_window = original.clone();
        pinned_window.insert(
            3,
            json!({"role": "user", "providerContext": {"opaque": "checkpoint"}}),
        );
        // A checkpoint after the presentation ends the scan window.
        assert!(recent_code_search_envelope(&pinned_window).is_none());
    }

    #[tokio::test]
    async fn mock_decisions_filter_only_under_explicit_active_policy() {
        use pi_jev::hooks::{JevObserver, JevObserverConfig};
        use pi_jev::{JevLimits, JevMode, JevStats, JevSystemOne, MockJevTransport, SecretString};
        use std::sync::Arc;
        let temp = tempfile::tempdir().unwrap();
        let mode = JevMode::CompareAndActive;
        let client=Arc::new(JevSystemOne::new(mode,SecretString::new("synthetic"),Arc::new(MockJevTransport::scripted(vec![pi_jev::MockStep::Body(json!({"model":"mock", "answers":{
            "code_search_relevance.0":{"type":"choice","choice":"drop","probabilities":{"drop":0.99,"keep":0.01},"confidence":0.99},
            "code_search_relevance.1":{"type":"choice","choice":"keep","probabilities":{"drop":0.01,"keep":0.99},"confidence":0.99}
        }}).to_string())])),JevLimits::default(),Arc::new(JevStats::default())).unwrap());
        let observer = JevObserver::new(
            JevObserverConfig {
                mode_gate: Arc::new(move |_| (mode, pi_jev::hooks::SYSTEM_ONE_MODEL.to_string())),
                ..Default::default()
            },
            client,
            temp.path().join("records.jsonl"),
        );
        let plan = prepare(&messages(2), "task", &Default::default());
        let batch = &plan[0].batches[0];
        let payload = json!({"session_id":"search-test","turn":1,"state":batch.state});
        let disabled = observer
            .decide_prepared(
                &payload,
                "code_search",
                batch.questions(),
                &Default::default(),
            )
            .await;
        assert!(disabled.decisions.is_empty());
        let policy = pi_jev::active::ActivationPolicy {
            enabled_categories: [DecisionCategory::CodeSearchRelevance]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let enabled = observer
            .decide_prepared(&payload, "code_search", batch.questions(), &policy)
            .await;
        assert_eq!(enabled.decisions.len(), 2);
        assert!(observer.can_apply(&enabled));
        let removals = batch.removals(
            &enabled.decisions,
            enabled.request_id.as_deref().unwrap(),
            1,
        );
        assert_eq!(removals, vec![0]);
        assert!(plan[0].project(&removals).is_some());
        assert!(batch
            .removals(
                &enabled.decisions[..1],
                enabled.request_id.as_deref().unwrap(),
                1
            )
            .is_empty());
        observer.cancel_decisions("search-test");
        assert!(!observer.can_apply(&enabled));
        let mut exhausted_payload = payload.clone();
        exhausted_payload["decision_timeout_ms"] = json!(0);
        let exhausted = observer
            .decide_prepared(
                &exhausted_payload,
                "code_search",
                batch.questions(),
                &policy,
            )
            .await;
        assert!(!exhausted.dispatched);
        assert!(exhausted.decisions.is_empty());
        assert_eq!(
            exhausted.terminal_reason.as_deref(),
            Some("deadline_exhausted")
        );
        observer.shutdown();
    }
}
