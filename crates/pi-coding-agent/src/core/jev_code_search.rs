//! Explicit search candidates at the provider-context boundary. Retrieval stays local.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use pi_jev::filtering::FilteringOptions;
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};

use super::jev_retrieval::PreparedRelevance;

const MAX_PRESENTATION_BYTES: usize = 512 * 1024;
const MAX_SCORED: usize = 64;

pub struct SearchPresentation {
    pub message_index: usize,
    pub batches: Vec<PreparedRelevance>,
    pub fingerprint: String,
    envelope: Value,
}

impl SearchPresentation {
    pub fn project(&self, removals: &[usize]) -> Option<Value> {
        let candidates = self.envelope["candidates"].as_array()?;
        let eligible: BTreeSet<usize> = self
            .batches
            .iter()
            .flat_map(|batch| batch.candidate_indices.iter().copied())
            .collect();
        let dropped: BTreeSet<usize> = removals
            .iter()
            .copied()
            .filter(|index| eligible.contains(index))
            .collect();
        if dropped.is_empty() || dropped.len() == candidates.len() {
            return None;
        }
        let mut envelope = self.envelope.clone();
        envelope["candidates"] = Value::Array(
            candidates
                .iter()
                .enumerate()
                .filter(|(index, _)| !dropped.contains(index))
                .map(|(_, value)| value.clone())
                .collect(),
        );
        envelope["jev_omitted_candidates"] = json!(dropped.len());
        envelope["jev_notice"] = json!("Optional candidates omitted for this request; original search results remain in session history and the Python variable.");
        Some(json!([{"type":"text", "text":serde_json::to_string(&envelope).ok()?}]))
    }
}

fn pinned(candidate: &Value) -> bool {
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
            envelope,
        });
    }
    presentations
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
                mode_gate: Arc::new(move |_| mode),
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
