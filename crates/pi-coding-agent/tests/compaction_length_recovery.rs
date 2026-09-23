use std::sync::{Arc, Mutex};
use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
use pi_ai::types::{AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Model, TextContent, UserContent, UserMessage};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{compact_with_metrics, default_compaction_settings, default_summary_call_runner, generate_summary, CompactionPreparation, ProviderRetryPolicy, SummarySlice, SummaryUpdatePolicy};
use pi_coding_agent::core::compaction::metrics::CompactionMetrics;
use pi_coding_agent::core::compaction::utils::create_file_ops;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const VALID: &str = "## Goal\nComplete.\n## Constraints & Preferences\nNone.\n## Progress\nDone.\n## Key Decisions\nWait.\n## Next Steps\nReview.\n## Critical Context\nSaved.";
const PREFIX: &str = "## Original Request\nComplete.\n## Early Progress\nSaved.\n## Context for Suffix\nReady.";

fn response(reason: &str, raw: Option<&str>, text: &str) -> AssistantMessage {
    AssistantMessage {
        stop_reason: reason.into(), stop_reason_raw: raw.map(str::to_string),
        content: vec![ContentBlock::Text(TextContent::new(text))], ..Default::default()
    }
}

#[derive(Clone)]
struct Request { max_tokens: f64, content: Value, reasoning: Option<String> }

fn fixture(name: &str, responses: Vec<AssistantMessage>, cancel: Option<CancellationToken>) -> (Model, Arc<Mutex<Vec<Request>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    register_api_provider_simple(ApiProviderSimple {
        api: name.into(), stream: Arc::new(|_, _, _| panic!("unexpected base stream")),
        stream_simple: Arc::new(move |_, context: &Context, options| {
            let options = options.unwrap();
            assert!(context.tools.is_none(), "summary retry must never execute chat tools");
            assert_eq!(options.stream.headers.as_ref().unwrap()["x-fixture"], "preserved");
            let mut seen = seen.lock().unwrap();
            let index = seen.len();
            seen.push(Request { max_tokens: options.stream.max_tokens.unwrap(),
                content: serde_json::to_value(&context.messages[0]).unwrap()["content"].clone(),
                reasoning: options.reasoning.clone() });
            let stream = AssistantMessageEventStream::new();
            if let Some(cancel) = &cancel { cancel.cancel(); }
            let message = responses.get(index).expect("unexpected extra summary request").clone();
            stream.push(AssistantMessageEvent::Done { reason: message.stop_reason.clone(), message });
            stream
        }), compact: None, supports_compaction: None,
    }, None);
    let mut model = Model::new(name, name, name, "faux", "https://fixture.invalid");
    model.reasoning = true;
    model.context_window = 1_000_000.0;
    model.max_tokens = 64_000.0;
    (model, requests)
}

async fn run(model: &Model, signal: Option<&CancellationToken>, text: &str, reserve: f64) -> Result<SummarySlice, String> {
    let messages = vec![AgentMessage::from(UserMessage::new(UserContent::Text(text.into()), 0))];
    let retry = ProviderRetryPolicy { enabled:false, max_retries:0, base_delay_ms:0.0, max_retry_delay_ms:0.0 };
    generate_summary(&messages, model, reserve, "fixture-unused", signal, None, None, Some(&ThinkingLevel::Xhigh),
        Some(&retry), default_summary_call_runner(Some(json!({"x-fixture":"preserved"}).as_object().unwrap().clone())),
        &SummaryUpdatePolicy::default()).await
}

#[tokio::test]
async fn confirmed_length_retries_only_the_same_summary_once_with_more_output() {
    let mut truncated = response("length", Some("max_output_tokens"), "PARTIAL-MUST-NOT-BE-COMMITTED");
    truncated.usage.output = 13_107.0;
    let mut complete = response("stop", None, VALID);
    complete.usage.output = 25.0;
    let (model, requests) = fixture("summary-length-recovery", vec![truncated, complete], None);
    let result = run(&model, None, "preserve original history", 16_384.0).await.unwrap();
    assert_eq!(result.summary, VALID);
    assert_eq!(result.usage.unwrap().output, 13_132.0, "both paid summary attempts remain accounted");
    let seen = requests.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].max_tokens, 13_107.0);
    assert_eq!(seen[1].max_tokens, 26_214.0);
    assert_eq!(seen[0].content, seen[1].content, "retry uses original input, never a partial checkpoint");
    assert!(seen.iter().all(|request| request.reasoning.as_deref() == Some("xhigh")));
}

#[tokio::test]
async fn repeated_length_fails_without_accepting_partial_handoff_or_a_third_request() {
    let (model, requests) = fixture("summary-length-twice", vec![response("length", Some("max_output_tokens"), VALID); 2], None);
    let error = run(&model, None, "original history", 16_384.0).await.unwrap_err();
    assert!(error.contains("existing conversation preserved"), "{error}");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn unconfirmed_length_safety_errors_abort_and_malformed_output_do_not_get_budget_retries() {
    let mut failed = response("error", Some("max_output_tokens"), "failed");
    failed.error_message = Some("provider failed".into());
    for (index, message) in [response("length", None, VALID), response("length", Some("content_filter"), VALID),
        response("length", Some("refusal"), VALID), response("length", Some("unknown"), VALID),
        response("aborted", Some("max_output_tokens"), VALID), failed,
        response("stop", None, "missing required sections")].into_iter().enumerate() {
        let (model, requests) = fixture(&format!("summary-no-length-retry-{index}"), vec![message], None);
        assert!(run(&model, None, "original history", 16_384.0).await.is_err());
        assert_eq!(requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn cancellation_wins_over_a_confirmed_length_response() {
    let signal = CancellationToken::new();
    let (model, requests) = fixture("summary-length-cancel", vec![response("length", Some("max_output_tokens"), VALID)], Some(signal.clone()));
    assert!(run(&model, Some(&signal), "original history", 16_384.0).await.unwrap_err().contains("Aborted"));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn model_output_ceiling_prevents_a_useless_length_retry() {
    let (mut model, requests) = fixture("summary-length-ceiling", vec![response("length", Some("max_output_tokens"), VALID)], None);
    model.max_tokens = 13_107.0;
    assert!(run(&model, None, "original history", 16_384.0).await.is_err());
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn one_length_retry_is_shared_across_all_chunks_not_reset_per_chunk() {
    let (mut model, requests) = fixture("summary-length-chunks", vec![response("length", Some("max_output_tokens"), VALID),
        response("stop", None, VALID), response("length", Some("max_output_tokens"), VALID)], None);
    model.context_window = 12_000.0;
    model.max_tokens = 6_000.0;
    let error = run(&model, None, &"original history ".repeat(10_000), 1_000.0).await.unwrap_err();
    assert!(error.contains("existing conversation preserved"), "{error}");
    let seen = requests.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].max_tokens, 800.0);
    assert_eq!(seen[1].max_tokens, 1_600.0);
    assert_eq!(seen[2].max_tokens, 800.0);
    assert_eq!(seen[0].content, seen[1].content);
    assert_ne!(seen[1].content, seen[2].content);
}

#[tokio::test]
async fn split_history_and_turn_prefix_have_independent_bounded_length_recovery() {
    let seen = Arc::new(Mutex::new(std::collections::BTreeMap::<bool, Vec<(f64, Value)>>::new()));
    let captured = seen.clone();
    let api = "summary-length-split";
    register_api_provider_simple(ApiProviderSimple {
        api: api.into(), stream: Arc::new(|_, _, _| panic!("unexpected base stream")),
        stream_simple: Arc::new(move |_, context, options| {
            let content = serde_json::to_value(&context.messages[0]).unwrap()["content"].clone();
            let prefix = content.to_string().contains("PREFIX-WORK");
            let mut captured = captured.lock().unwrap();
            let requests = captured.entry(prefix).or_default();
            requests.push((options.unwrap().stream.max_tokens.unwrap(), content));
            assert!(requests.len() <= 2, "no third request for either split summary");
            let message = if requests.len() == 1 {
                response("length", Some("max_output_tokens"), "partial")
            } else {
                response("stop", None, if prefix { PREFIX } else { VALID })
            };
            let stream = AssistantMessageEventStream::new();
            stream.push(AssistantMessageEvent::Done { reason: message.stop_reason.clone(), message });
            stream
        }), compact: None, supports_compaction: None,
    }, None);
    let mut model = Model::new(api, api, api, "faux", "https://fixture.invalid");
    model.context_window = 1_000_000.0;
    model.max_tokens = 64_000.0;
    let message = |text: &str| AgentMessage::from(UserMessage::new(UserContent::Text(text.into()), 0));
    let preparation = CompactionPreparation { first_kept_entry_id:"kept".into(),
        messages_to_summarize:vec![message("HISTORY-WORK")], turn_prefix_messages:vec![message("PREFIX-WORK")],
        is_split_turn:true, tokens_before:250_001.0, retained_state_anchor: None, previous_summary:None, file_ops:create_file_ops(), settings:default_compaction_settings() };
    let metrics = CompactionMetrics::new(None, &model);
    let result = compact_with_metrics(&preparation, &model, "fixture-unused", None, None, None,
        default_summary_call_runner(None), None, None, &metrics).await.unwrap();
    assert!(result.summary.contains(VALID) && result.summary.contains(PREFIX));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    for (prefix, requests) in seen.iter() {
        assert_eq!(requests.len(), 2);
        let initial = if *prefix { 8_192.0 } else { 13_107.0 };
        assert_eq!(requests[0].0, initial);
        assert_eq!(requests[1].0, initial * 2.0);
        assert_eq!(requests[0].1, requests[1].1);
    }
}
