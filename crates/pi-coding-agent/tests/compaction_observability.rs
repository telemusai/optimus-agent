use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use pi_agent_core::performance_metrics::{
    PerformanceMetricEvent, PerformanceMetricIdScope, PerformanceMetricMeasurement as Measurement,
    PerformanceMetricOperation as Operation, PerformanceMetricOutcome as Outcome,
    PerformanceMetricRecorder,
};
use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple, SimpleStreamFunction};
use pi_ai::compaction::CompactionOptions;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Model, ProviderResponse,
    ProviderUsageObservation, SimpleStreamOptions, TextContent, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::compaction::compaction::{
    compact_with_metrics, default_compaction_settings, default_summary_call_runner,
    CompactionPreparation, ProviderRetryPolicy,
};
use pi_coding_agent::core::compaction::metrics::CompactionMetrics;
use pi_coding_agent::core::compaction::utils::create_file_ops;
use tokio::sync::{Barrier, Notify};
use tokio_util::sync::CancellationToken;

const VALID: &str = "## Goal\nComplete.\n## Constraints & Preferences\nNone.\n## Progress\nDone.\n## Key Decisions\nWait.\n## Next Steps\nReview.\n## Critical Context\nSaved.";
const PREFIX: &str =
    "## Original Request\nComplete.\n## Early Progress\nSaved.\n## Context for Suffix\nReady.";

#[derive(Default)]
struct Recorder {
    clock: AtomicU64,
    ids: AtomicU64,
    events: Mutex<Vec<PerformanceMetricEvent>>,
}
impl PerformanceMetricRecorder for Recorder {
    fn session_id(&self) -> &str {
        "isolated-compaction"
    }
    fn monotonic_now(&self) -> f64 {
        self.clock.load(Ordering::SeqCst) as f64
    }
    fn next_id(&self, scope: PerformanceMetricIdScope) -> String {
        format!("{scope:?}-{}", self.ids.fetch_add(1, Ordering::SeqCst))
    }
    fn record(&self, event: PerformanceMetricEvent) {
        self.events.lock().unwrap().push(event);
    }
    fn flush(&self) {}
    fn close(&self) {}
}

fn response(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: "stop".into(),
        ..Default::default()
    }
}
fn register(name: &str, stream: SimpleStreamFunction) -> Model {
    register_api_provider_simple(
        ApiProviderSimple {
            api: name.into(),
            stream: Arc::new(|_, _, _| panic!("unexpected base stream")),
            stream_simple: stream,
            compact: None,
            supports_compaction: None,
        },
        None,
    );
    let mut model = Model::new(name, name, name, "faux", "https://fixture.invalid");
    model.reasoning = true;
    model.context_window = 1_000_000.0;
    model.max_tokens = 32_000.0;
    model
}
fn preparation(split: bool) -> CompactionPreparation {
    let message = |text: &str| {
        AgentMessage::Message(pi_ai::types::Message::User(UserMessage::new(
            UserContent::Text(text.into()),
            0,
        )))
    };
    CompactionPreparation {
        first_kept_entry_id: "retained-turn".into(),
        messages_to_summarize: vec![message("SECRET-HISTORY-PROMPT")],
        turn_prefix_messages: if split {
            vec![message("SECRET-PREFIX-PROMPT")]
        } else {
            Vec::new()
        },
        is_split_turn: split,
        tokens_before: 250_905.0,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: default_compaction_settings(),
    }
}
fn push(stream: &AssistantMessageEventStream, response: AssistantMessage) {
    stream.push(AssistantMessageEvent::Done {
        reason: response.stop_reason.clone(),
        message: response,
    });
}
fn terminal(
    events: &[PerformanceMetricEvent],
    operation: Operation,
) -> Vec<&PerformanceMetricEvent> {
    events
        .iter()
        .filter(|event| {
            event.operation == operation
                && matches!(event.outcome, Some(Outcome::Success | Outcome::Failure | Outcome::Cancelled | Outcome::Unavailable))
        })
        .collect()
}

fn started(
    events: &[PerformanceMetricEvent],
    operation: Operation,
) -> Vec<&PerformanceMetricEvent> {
    events
        .iter()
        .filter(|event| event.operation == operation && event.outcome == Some(Outcome::Started))
        .collect()
}

#[tokio::test]
async fn text_summary_preserves_options_and_records_body_timing_without_content() {
    let recorder = Arc::new(Recorder::default());
    let clock = recorder.clone();
    let called_hooks = Arc::new(AtomicU64::new(0));
    let model = register(
        "compaction-observed-options",
        Arc::new(move |model, _, options| {
            let options = options.unwrap().clone();
            assert_eq!(options.stream.timeout_ms, Some(1_200_000.0));
            assert_eq!(
                options.stream.session_id.as_deref(),
                Some("isolated-cache-session")
            );
            assert_eq!(options.stream.service_tier, Some(Some("priority".into())));
            assert_eq!(options.stream.transport.as_deref(), Some("sse"));
            assert_eq!(options.reasoning.as_deref(), Some("xhigh"));
            assert!(options.stream.max_tokens.unwrap() <= 32_000.0);
            let model = model.clone();
            let clock = clock.clone();
            let stream = AssistantMessageEventStream::new();
            let output = stream.clone();
            tokio::spawn(async move {
                clock.clock.store(10, Ordering::SeqCst);
                assert!(options.stream.on_payload.as_ref().unwrap()(
                    serde_json::json!({"secret": "SECRET-PAYLOAD"}),
                    &model
                )
                .await
                .is_none());
                clock.clock.store(30, Ordering::SeqCst);
                options.stream.on_response.as_ref().unwrap()(
                    ProviderResponse {
                        status: 200,
                        headers: [("x-optimus-transport".into(), "sse".into())].into(),
                    },
                    &model,
                )
                .await;
                let observe = options.stream.on_stream_observation.as_ref().unwrap();
                clock.clock.store(50, Ordering::SeqCst);
                observe("raw_event");
                clock.clock.store(70, Ordering::SeqCst);
                observe("thinking");
                clock.clock.store(100, Ordering::SeqCst);
                observe("text");
                clock.clock.store(300, Ordering::SeqCst);
                observe("terminal");
                options.stream.on_usage_observation.as_ref().unwrap()(
                    ProviderUsageObservation {
                        input_tokens: Some(Some(200.0)),
                        reasoning_tokens: Some(Some(0.0)),
                        output_tokens: Some(Some(30.0)),
                        total_tokens: Some(Some(230.0)),
                        ..Default::default()
                    },
                    &model,
                )
                .await;
                clock.clock.store(320, Ordering::SeqCst);
                push(&output, response(VALID));
            });
            stream
        }),
    );
    let mut options = CompactionOptions::default();
    options.simple.stream.timeout_ms = Some(1_200_000.0);
    options.simple.stream.session_id = Some("isolated-cache-session".into());
    options.simple.stream.service_tier = Some(Some("priority".into()));
    options.simple.stream.transport = Some("sse".into());
    let hooks = called_hooks.clone();
    options.simple.stream.on_usage_observation = Some(Arc::new(move |_, _| {
        let hooks = hooks.clone();
        Box::pin(async move {
            hooks.fetch_add(1, Ordering::SeqCst);
        })
    }));
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let result = compact_with_metrics(
        &preparation(false),
        &model,
        "SECRET-API-KEY",
        None,
        None,
        Some(&ThinkingLevel::Xhigh),
        default_summary_call_runner(None),
        None,
        Some((&Context::default(), Some(&options))),
        &metrics,
    )
    .await
    .unwrap();
    assert_eq!(result.summary, VALID);
    assert_eq!(result.tokens_before, 250_905.0);
    assert_eq!(model.context_window, 1_000_000.0);
    assert_eq!(called_hooks.load(Ordering::SeqCst), 1);
    let events = recorder.events.lock().unwrap();
    let requests = terminal(&events, Operation::ProviderAttempt);
    assert_eq!(requests.len(), 1);
    let measures = requests[0].measurements.as_ref().unwrap();
    assert_eq!(
        measures[&Measurement::DispatchToResponseHeadersMs],
        Some(20.0)
    );
    assert_eq!(
        measures[&Measurement::DispatchToFirstThinkingMs],
        Some(60.0)
    );
    assert_eq!(
        measures[&Measurement::DispatchToNetworkTerminalMs],
        Some(290.0)
    );
    assert_eq!(measures[&Measurement::LocalDrainMs], Some(20.0));
    assert_eq!(measures[&Measurement::TransportWebsocket], Some(0.0));
    assert_eq!(
        requests[0].usage.as_ref().unwrap().reasoning_tokens,
        Some(0.0)
    );
    assert_eq!(
        terminal(&events, Operation::CompactionHistory)[0].outcome,
        Some(Outcome::Success)
    );
    let encoded = serde_json::to_string(&*events).unwrap();
    for secret in ["SECRET-HISTORY", "SECRET-API", "SECRET-PAYLOAD", VALID] {
        assert!(!encoded.contains(secret));
    }
}

/// B2/B5: a WebSocket send acknowledgement, real HTTP headers and a provider that only
/// labels its transport on an observation stage must stay distinguishable.
#[tokio::test]
async fn transport_edges_are_recorded_without_confusing_an_ack_with_headers() {
    let recorder = Arc::new(Recorder::default());
    let clock = recorder.clone();
    let model = register(
        "compaction-observed-transport-edges",
        Arc::new(|_, _, _| AssistantMessageEventStream::new()),
    );
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    // The wrapper phase is a history phase, so the three attempt records below are the
    // only provider-attempt terminals in this fixture.
    let phase = metrics.phase(Operation::CompactionHistory);
    let requests = phase.requests();

    // A WebSocket attempt: the provider answers the upgrade with a send acknowledgement.
    let request = requests.next();
    let mut options = SimpleStreamOptions::default();
    request.observe(&mut options);
    let response_hook = options.stream.on_response.clone().unwrap();
    let observation_hook = options.stream.on_stream_observation.clone().unwrap();
    clock.clock.store(100, Ordering::SeqCst);
    assert!(options.stream.on_payload.as_ref().unwrap()(
        serde_json::json!({"secret": "SECRET-PAYLOAD"}),
        &model
    )
    .await
    .is_none());
    clock.clock.store(150, Ordering::SeqCst);
    response_hook(
        ProviderResponse {
            status: 101,
            headers: [
                ("x-optimus-transport".into(), "websocket".into()),
                ("x-optimus-response-edge".into(), "transport_send_ack".into()),
            ]
            .into(),
        },
        &model,
    )
    .await;
    clock.clock.store(300, Ordering::SeqCst);
    observation_hook("terminal");
    request.finish(&response("acknowledged"), false);

    // An SSE attempt of the same request shape reports a real header edge.
    let request = requests.next();
    let mut options = SimpleStreamOptions::default();
    request.observe(&mut options);
    clock.clock.store(400, Ordering::SeqCst);
    assert!(options.stream.on_payload.as_ref().unwrap()(
        serde_json::json!({"secret": "SECRET-PAYLOAD"}),
        &model
    )
    .await
    .is_none());
    clock.clock.store(420, Ordering::SeqCst);
    options.stream.on_response.as_ref().unwrap()(
        ProviderResponse {
            status: 200,
            headers: [
                ("x-optimus-transport".into(), "sse".into()),
                ("x-optimus-response-edge".into(), "response_headers".into()),
            ]
            .into(),
        },
        &model,
    )
    .await;
    request.finish(&response("streamed"), false);

    // A WebSocket provider that never calls the response hook labels its transport on the
    // observation stage instead, so the attempt is still comparable with an SSE attempt.
    let request = requests.next();
    let mut options = SimpleStreamOptions::default();
    request.observe(&mut options);
    assert!(options.stream.on_payload.as_ref().unwrap()(
        serde_json::json!({"secret": "SECRET-PAYLOAD"}),
        &model
    )
    .await
    .is_none());
    options.stream.on_stream_observation.as_ref().unwrap()("transport_ws");
    options.stream.on_stream_observation.as_ref().unwrap()("terminal");
    request.finish(&response("labelled"), false);

    phase.finish(Outcome::Success);
    let events = recorder.events.lock().unwrap();
    let attempts = terminal(&events, Operation::ProviderAttempt);
    assert_eq!(attempts.len(), 3);
    let ack = attempts[0].measurements.as_ref().unwrap();
    assert_eq!(
        ack[&Measurement::DispatchToResponseHeadersMs],
        None::<f64>,
        "a send acknowledgement is not an HTTP header edge"
    );
    assert_eq!(ack[&Measurement::TransportWebsocket], Some(1.0));
    assert_eq!(ack[&Measurement::DispatchToNetworkTerminalMs], Some(200.0));
    let sse = attempts[1].measurements.as_ref().unwrap();
    assert_eq!(sse[&Measurement::DispatchToResponseHeadersMs], Some(20.0));
    assert_eq!(sse[&Measurement::TransportWebsocket], Some(0.0));
    let labelled = attempts[2].measurements.as_ref().unwrap();
    assert_eq!(
        labelled[&Measurement::DispatchToResponseHeadersMs],
        None::<f64>,
        "an unobserved header edge stays null, never zero"
    );
    assert_eq!(labelled[&Measurement::TransportWebsocket], Some(1.0));
    let encoded = serde_json::to_string(&*events).unwrap();
    assert!(!encoded.contains("SECRET-PAYLOAD"));
    assert!(!encoded.contains("acknowledged"));
}

#[tokio::test]
async fn split_failure_cancels_pending_sibling_without_cancelling_parent() {
    let barrier = Arc::new(Barrier::new(2));
    let cancelled = Arc::new(Notify::new());
    let observed_cancel = cancelled.clone();
    let model = register(
        "compaction-observed-split-failure",
        Arc::new(move |_, context, options| {
            let prefix = serde_json::to_string(context)
                .unwrap()
                .contains("SECRET-PREFIX-PROMPT");
            let signal = options.unwrap().stream.signal.clone().unwrap();
            let barrier = barrier.clone();
            let cancelled = observed_cancel.clone();
            let stream = AssistantMessageEventStream::new();
            let output = stream.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                if prefix {
                    push(&output, response("unusable summary"));
                } else {
                    signal.cancelled().await;
                    cancelled.notify_one();
                }
            });
            stream
        }),
    );
    let parent = CancellationToken::new();
    let recorder = Arc::new(Recorder::default());
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        compact_with_metrics(
            &preparation(true),
            &model,
            "unused",
            None,
            Some(&parent),
            None,
            default_summary_call_runner(None),
            None,
            None,
            &metrics,
        ),
    )
    .await
    .expect("failed split must return promptly");
    assert!(result.unwrap_err().contains("unusable handoff"));
    tokio::time::timeout(Duration::from_secs(2), cancelled.notified())
        .await
        .expect("detached sibling must observe cancellation");
    assert!(!parent.is_cancelled());
    let events = recorder.events.lock().unwrap();
    assert_eq!(
        terminal(&events, Operation::CompactionPrefix)[0].outcome,
        Some(Outcome::Failure)
    );
    assert_eq!(
        terminal(&events, Operation::CompactionHistory)[0].outcome,
        Some(Outcome::Cancelled)
    );
    assert!(terminal(&events, Operation::ProviderAttempt)
        .iter()
        .any(|event| event.outcome == Some(Outcome::Cancelled)));
}

#[tokio::test]
async fn successful_split_requests_remain_concurrent_and_separately_correlated() {
    let barrier = Arc::new(Barrier::new(2));
    let model = register(
        "compaction-observed-split-success",
        Arc::new(move |_, context, _| {
            let prefix = serde_json::to_string(context)
                .unwrap()
                .contains("SECRET-PREFIX-PROMPT");
            let barrier = barrier.clone();
            let stream = AssistantMessageEventStream::new();
            let output = stream.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                push(&output, response(if prefix { PREFIX } else { VALID }));
            });
            stream
        }),
    );
    let recorder = Arc::new(Recorder::default());
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        compact_with_metrics(
            &preparation(true),
            &model,
            "unused",
            None,
            None,
            None,
            default_summary_call_runner(None),
            None,
            None,
            &metrics,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(result.summary.contains(VALID) && result.summary.contains(PREFIX));
    let events = recorder.events.lock().unwrap();
    let requests = terminal(&events, Operation::ProviderAttempt);
    assert_eq!(requests.len(), 2);
    let first = requests[0].correlation.as_ref().unwrap();
    let second = requests[1].correlation.as_ref().unwrap();
    assert_eq!(first.action_id, second.action_id);
    assert_ne!(first.logical_request_id, second.logical_request_id);
    assert_ne!(first.provider_attempt_id, second.provider_attempt_id);
}

#[tokio::test]
async fn provider_attempt_start_rows_are_labeled_started_and_paired_one_to_one() {
    let count = Arc::new(AtomicU64::new(0));
    let called = count.clone();
    let model = register(
        "compaction-observed-start-rows",
        Arc::new(move |_, _, _| {
            let stream = AssistantMessageEventStream::new();
            let message = if called.fetch_add(1, Ordering::SeqCst) == 0 {
                AssistantMessage {
                    stop_reason: "error".into(),
                    error_message: Some("server unavailable".into()),
                    ..Default::default()
                }
            } else {
                response(VALID)
            };
            push(&stream, message);
            stream
        }),
    );
    let recorder = Arc::new(Recorder::default());
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let retry = ProviderRetryPolicy {
        enabled: true,
        max_retries: 1,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 100.0,
    };
    compact_with_metrics(
        &preparation(false),
        &model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        Some(&retry),
        None,
        &metrics,
    )
    .await
    .unwrap();
    let events = recorder.events.lock().unwrap();
    // Every provider attempt is emitted twice under the same correlation IDs:
    // an explicitly labeled `started` row (never null, never a failure) and a
    // terminal row. Terminal accounting counts terminals only.
    let starts = started(&events, Operation::ProviderAttempt);
    let terminals = terminal(&events, Operation::ProviderAttempt);
    assert_eq!(starts.len(), terminals.len());
    for (start, terminal) in starts.iter().zip(terminals.iter()) {
        let start = start.correlation.as_ref().unwrap();
        let terminal = terminal.correlation.as_ref().unwrap();
        assert_eq!(start.action_id, terminal.action_id);
        assert_eq!(start.logical_request_id, terminal.logical_request_id);
        assert_eq!(start.provider_attempt_id, terminal.provider_attempt_id);
    }
    assert_eq!(terminals[0].outcome, Some(Outcome::Failure));
    assert_eq!(terminals[1].outcome, Some(Outcome::Success));
    // Phase rows pair the same way.
    assert_eq!(started(&events, Operation::CompactionHistory).len(), 1);
    assert_eq!(terminal(&events, Operation::CompactionHistory).len(), 1);
}

#[tokio::test]
async fn retry_has_one_attempt_record_per_actual_call() {
    let count = Arc::new(AtomicU64::new(0));
    let called = count.clone();
    let model = register(
        "compaction-observed-retry",
        Arc::new(move |_, _, _| {
            let stream = AssistantMessageEventStream::new();
            let message = if called.fetch_add(1, Ordering::SeqCst) == 0 {
                AssistantMessage {
                    stop_reason: "error".into(),
                    error_message: Some("server unavailable".into()),
                    ..Default::default()
                }
            } else {
                response(VALID)
            };
            push(&stream, message);
            stream
        }),
    );
    let recorder = Arc::new(Recorder::default());
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let retry = ProviderRetryPolicy {
        enabled: true,
        max_retries: 1,
        base_delay_ms: 0.0,
        max_retry_delay_ms: 100.0,
    };
    compact_with_metrics(
        &preparation(false),
        &model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        Some(&retry),
        None,
        &metrics,
    )
    .await
    .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2);
    let events = recorder.events.lock().unwrap();
    let requests = terminal(&events, Operation::ProviderAttempt);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].outcome, Some(Outcome::Failure));
    assert_eq!(requests[1].outcome, Some(Outcome::Success));
    assert_eq!(
        requests[0].correlation.as_ref().unwrap().logical_request_id,
        requests[1].correlation.as_ref().unwrap().logical_request_id
    );
    assert_ne!(
        requests[0]
            .correlation
            .as_ref()
            .unwrap()
            .provider_attempt_id,
        requests[1]
            .correlation
            .as_ref()
            .unwrap()
            .provider_attempt_id
    );
    assert_eq!(
        requests[1].measurements.as_ref().unwrap()[&Measurement::AttemptOrdinal],
        Some(2.0)
    );
}

#[tokio::test]
async fn dropped_compaction_cancels_detached_provider_and_records_cancelled() {
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let start = started.clone();
    let cancel = cancelled.clone();
    let model = register(
        "compaction-observed-drop",
        Arc::new(move |_, _, options| {
            let signal = options.unwrap().stream.signal.clone().unwrap();
            let cancel = cancel.clone();
            start.notify_one();
            tokio::spawn(async move {
                signal.cancelled().await;
                cancel.notify_one();
            });
            AssistantMessageEventStream::new()
        }),
    );
    let recorder = Arc::new(Recorder::default());
    let metrics = CompactionMetrics::new(Some(recorder.clone()), &model);
    let parent = CancellationToken::new();
    let signal = parent.clone();
    let task = tokio::spawn(async move {
        compact_with_metrics(
            &preparation(false),
            &model,
            "unused",
            None,
            Some(&signal),
            None,
            default_summary_call_runner(None),
            None,
            None,
            &metrics,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), cancelled.notified())
        .await
        .unwrap();
    assert!(!parent.is_cancelled());
    let events = recorder.events.lock().unwrap();
    assert_eq!(
        terminal(&events, Operation::CompactionHistory)[0].outcome,
        Some(Outcome::Cancelled)
    );
    assert_eq!(
        terminal(&events, Operation::ProviderAttempt)[0].outcome,
        Some(Outcome::Cancelled)
    );
}
