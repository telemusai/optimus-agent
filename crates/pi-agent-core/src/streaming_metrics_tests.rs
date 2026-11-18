//! Offline client-stream timing tests. All event times use a synthetic clock.
use super::rlm_t16_tests::REGISTRY_TESTS;
use super::*;
use crate::performance_metrics::{AgentLoopPerformanceMetrics, PerformanceMetricIdScope};
use pi_ai::types::{OnStreamObservation, OnUsageObservation};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use std::sync::atomic::AtomicU64;

#[derive(Default)]
struct Recorder {
    clock: AtomicU64,
    ids: AtomicU64,
    records: Mutex<Vec<PerformanceMetricEvent>>,
    progress: Mutex<Vec<(String, f64, f64)>>,
}

impl Recorder {
    fn set_clock(&self, now: f64) {
        self.clock.store(now.to_bits(), Ordering::SeqCst);
    }

    fn attempts(&self) -> Vec<PerformanceMetricEvent> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.operation == PerformanceMetricOperation::ProviderAttempt)
            .cloned()
            .collect()
    }
}

impl PerformanceMetricRecorder for Recorder {
    fn session_id(&self) -> &str {
        "streaming-metrics-test"
    }
    fn monotonic_now(&self) -> f64 {
        f64::from_bits(self.clock.load(Ordering::SeqCst))
    }
    fn next_id(&self, scope: PerformanceMetricIdScope) -> String {
        format!("{scope:?}-{}", self.ids.fetch_add(1, Ordering::SeqCst))
    }
    fn record(&self, event: PerformanceMetricEvent) {
        self.records.lock().unwrap().push(event);
    }
    fn generation_progress(&self, id: &str, tokens: f64, elapsed: f64) {
        self.progress
            .lock()
            .unwrap()
            .push((id.into(), tokens, elapsed));
    }
    fn flush(&self) {}
    fn close(&self) {}
}

#[derive(Clone, Copy)]
enum DeltaKind {
    Text,
    Thinking,
    Tool,
}

fn delta_event(kind: DeltaKind, text: &str, partial: &AssistantMessage) -> AssistantMessageEvent {
    match kind {
        DeltaKind::Text => AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: text.into(),
            partial: partial.clone(),
        },
        DeltaKind::Thinking => AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: text.into(),
            partial: partial.clone(),
        },
        DeltaKind::Tool => AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: text.into(),
            partial: partial.clone(),
        },
    }
}

/// The sink releases the next provider event only after the previous one was
/// consumed, so pre-queued producer work cannot race the synthetic clock.
async fn run_attempt(
    recorder: Arc<Recorder>,
    deltas: &[(f64, DeltaKind, &str)],
    terminal_at: f64,
    stop_reason: &str,
    usage: Option<ProviderUsageObservation>,
    ordinal: u64,
) {
    let model = Model::new(
        "m",
        "M",
        "openai-completions",
        "openai",
        "https://example.test",
    );
    let mut config = AgentLoopConfig::new(model.clone());
    let mut metrics = AgentLoopPerformanceMetrics::new(recorder.clone());
    metrics.provider_attempt_number = Some(ordinal);
    config.performance_metrics = Some(metrics);
    let partial = AssistantMessage::new(
        model.api.clone(),
        model.provider.clone(),
        model.id.clone(),
        1,
    );
    let mut terminal = partial.clone();
    terminal.stop_reason = stop_reason.into();
    // Final-only content is not a streamed delta and must not extend the clock.
    terminal.content = vec![ContentBlock::Text(TextContent::new("terminal content"))];
    let mut timeline: VecDeque<_> = deltas
        .iter()
        .map(|(time, kind, text)| (*time, delta_event(*kind, text, &partial)))
        .collect();
    timeline.push_back((
        terminal_at,
        if stop_reason == pi_ai::types::STOP_REASON_ERROR {
            AssistantMessageEvent::Error {
                reason: stop_reason.into(),
                error: terminal,
            }
        } else {
            AssistantMessageEvent::Done {
                reason: stop_reason.into(),
                message: terminal,
            }
        },
    ));
    let timeline = Arc::new(Mutex::new(timeline));
    let stream = AssistantMessageEventStream::new();
    let usage_hook: Arc<Mutex<Option<OnUsageObservation>>> = Arc::new(Mutex::new(None));
    let stream_hook: Arc<Mutex<Option<OnStreamObservation>>> = Arc::new(Mutex::new(None));
    let stream_fn: StreamFn = {
        let recorder = recorder.clone();
        let stream = stream.clone();
        let usage_hook = usage_hook.clone();
        let stream_hook = stream_hook.clone();
        Arc::new(move |model, _, options| {
            let recorder = recorder.clone();
            let stream = stream.clone();
            let partial = partial.clone();
            *usage_hook.lock().unwrap() = options.stream.on_usage_observation.clone();
            *stream_hook.lock().unwrap() = options.stream.on_stream_observation.clone();
            Box::pin(async move {
                recorder.set_clock(10.0);
                if let Some(hook) = options.stream.on_payload {
                    hook(serde_json::json!({}), &model).await;
                }
                recorder.set_clock(1000.0);
                stream.push(AssistantMessageEvent::Start { partial });
                stream
            })
        })
    };
    let signal = CancellationToken::new();
    let cancel_at_end = stop_reason == pi_ai::types::STOP_REASON_ABORTED;
    let emit: AgentEventSink = {
        let recorder = recorder.clone();
        let signal = signal.clone();
        Arc::new(move |event| {
            let next = if matches!(
                event,
                AgentEvent::MessageStart { .. } | AgentEvent::MessageUpdate { .. }
            ) {
                timeline.lock().unwrap().pop_front()
            } else {
                None
            };
            let recorder = recorder.clone();
            let signal = signal.clone();
            let stream = stream.clone();
            let model = model.clone();
            let usage = usage.clone();
            let usage_hook = usage_hook.lock().unwrap().clone();
            let stream_hook = stream_hook.lock().unwrap().clone();
            Box::pin(async move {
                if let Some((time, next)) = next {
                    recorder.set_clock(time);
                    if matches!(
                        next,
                        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
                    ) {
                        if cancel_at_end {
                            signal.cancel();
                            return Ok(());
                        }
                        if let Some(hook) = stream_hook {
                            hook("terminal");
                        }
                        if let (Some(hook), Some(usage)) = (usage_hook, usage) {
                            hook(usage, &model).await;
                        }
                    }
                    stream.push(next);
                }
                Ok(())
            })
        })
    };
    stream_assistant_response(
        &mut AgentContext::default(),
        &mut Vec::new(),
        &config,
        Some(&signal),
        &emit,
        &mut AgentLoopMetricState::default(),
        Some(stream_fn),
    )
    .await
    .unwrap();
}

fn reported_usage() -> ProviderUsageObservation {
    ProviderUsageObservation {
        output_tokens: Some(Some(80.0)),
        reasoning_tokens: Some(Some(30.0)),
        reasoning_included_in_output: Some(Some(true)),
        ..Default::default()
    }
}

fn streaming_ms(event: &PerformanceMetricEvent) -> Option<f64> {
    *event
        .measurements
        .as_ref()
        .unwrap()
        .get(&PerformanceMetricMeasurement::StreamingMs)
        .unwrap()
}

#[test]
fn streaming_clock_requires_a_positive_finite_interval_without_fallbacks() {
    let mut clock = StreamingMetricState::default();
    assert_eq!(clock.elapsed_ms(), None);
    clock.observe("", Some(1.0));
    assert_eq!(clock.first_at, None);
    clock.observe("first", Some(1000.0));
    assert_eq!(
        clock.elapsed_ms(),
        None,
        "one delta never has measurable duration"
    );
    clock.observe("same instant", Some(1000.0));
    assert_eq!(clock.elapsed_ms(), None);
    clock.observe("last", Some(1250.0));
    assert_eq!(clock.elapsed_ms(), Some(250.0));
    clock.observe("", Some(9000.0));
    assert_eq!(
        clock.elapsed_ms(),
        Some(250.0),
        "empty deltas do not extend the window"
    );

    for bad in [
        None,
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(999.0),
    ] {
        let mut clock = StreamingMetricState::default();
        clock.observe("first", Some(1000.0));
        clock.observe("bad", bad);
        clock.observe("later", Some(1500.0));
        assert_eq!(
            clock.elapsed_ms(),
            None,
            "invalid clock must not restart or recover"
        );
    }
    let mut clock = StreamingMetricState::default();
    clock.observe("missing first timestamp", None);
    clock.observe("later", Some(1500.0));
    clock.observe("last", Some(1600.0));
    assert_eq!(clock.elapsed_ms(), None);
    let mut clock = StreamingMetricState::default();
    clock.observe("first", Some(-f64::MAX));
    clock.observe("last", Some(f64::MAX));
    assert_eq!(
        clock.elapsed_ms(),
        None,
        "finite endpoints can still overflow"
    );
}

#[tokio::test]
async fn streaming_text_reasoning_and_tool_only_exclude_latency_and_drain() {
    let _registry = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    for kind in [DeltaKind::Text, DeltaKind::Thinking, DeltaKind::Tool] {
        let recorder = Arc::new(Recorder::default());
        run_attempt(
            recorder.clone(),
            &[
                (1500.0, kind, ""),
                (2000.0, kind, "12345678"),
                (2500.0, kind, "abcdefgh"),
                (9000.0, kind, ""),
            ],
            12000.0,
            pi_ai::types::STOP_REASON_STOP,
            Some(reported_usage()),
            1,
        )
        .await;
        let attempts = recorder.attempts();
        assert_eq!(attempts.len(), 1);
        assert_eq!(streaming_ms(&attempts[0]), Some(500.0));
        let measurements = attempts[0].measurements.as_ref().unwrap();
        assert_eq!(
            measurements[&PerformanceMetricMeasurement::TotalMs],
            Some(11990.0)
        );
        assert_eq!(
            measurements[&PerformanceMetricMeasurement::DispatchToNetworkTerminalMs],
            Some(11990.0)
        );
        let usage = attempts[0].usage.as_ref().unwrap();
        assert_eq!(
            usage.output_tokens,
            Some(80.0),
            "never add reasoning to the provider total"
        );
        assert_eq!(usage.reasoning_tokens, Some(30.0));
        let progress = recorder.progress.lock().unwrap();
        assert_eq!(progress.len(), 2, "empty deltas do not emit estimates");
        assert_eq!((progress[0].1, progress[0].2), (2.0, 0.0));
        assert_eq!((progress[1].1, progress[1].2), (4.0, 500.0));
    }
}

#[tokio::test]
async fn streaming_missing_single_and_invalid_delta_intervals_are_unavailable() {
    let _registry = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    for times in [
        vec![],
        vec![2000.0],
        vec![2000.0, 2000.0],
        vec![2000.0, 1500.0],
        vec![f64::NAN, 2000.0],
    ] {
        let recorder = Arc::new(Recorder::default());
        let deltas: Vec<_> = times
            .into_iter()
            .map(|time| (time, DeltaKind::Text, "chunk"))
            .collect();
        run_attempt(
            recorder.clone(),
            &deltas,
            12000.0,
            pi_ai::types::STOP_REASON_STOP,
            Some(reported_usage()),
            1,
        )
        .await;
        assert_eq!(streaming_ms(&recorder.attempts()[0]), None);
        assert!(recorder
            .progress
            .lock()
            .unwrap()
            .iter()
            .all(|(_, _, ms)| *ms == 0.0));
    }
}

#[tokio::test]
async fn streaming_retry_clock_and_character_estimate_start_fresh() {
    let _registry = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let recorder = Arc::new(Recorder::default());
    run_attempt(
        recorder.clone(),
        &[
            (2000.0, DeltaKind::Thinking, "12345678"),
            (2500.0, DeltaKind::Tool, "abcdefgh"),
        ],
        12000.0,
        pi_ai::types::STOP_REASON_ERROR,
        None,
        1,
    )
    .await;
    run_attempt(
        recorder.clone(),
        &[
            (20000.0, DeltaKind::Text, "1234"),
            (20250.0, DeltaKind::Text, "abcd"),
        ],
        30000.0,
        pi_ai::types::STOP_REASON_STOP,
        Some(reported_usage()),
        2,
    )
    .await;
    let attempts = recorder.attempts();
    assert_eq!(attempts.len(), 2);
    assert_eq!(streaming_ms(&attempts[0]), Some(500.0));
    assert_eq!(streaming_ms(&attempts[1]), Some(250.0));
    assert_eq!(attempts[0].outcome, Some(PerformanceMetricOutcome::Failure));
    assert_eq!(attempts[0].usage.as_ref().unwrap().output_tokens, None);
    let progress = recorder.progress.lock().unwrap();
    assert_eq!(progress.len(), 4);
    assert_ne!(progress[0].0, progress[2].0);
    assert_eq!((progress[2].1, progress[2].2), (1.0, 0.0));
    assert_eq!((progress[3].1, progress[3].2), (2.0, 250.0));
}

#[tokio::test]
async fn streaming_cancellation_preserves_last_delta_time_without_usage() {
    let _registry = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let recorder = Arc::new(Recorder::default());
    run_attempt(
        recorder.clone(),
        &[
            (2000.0, DeltaKind::Thinking, "12345678"),
            (2500.0, DeltaKind::Tool, "abcdefgh"),
        ],
        12000.0,
        pi_ai::types::STOP_REASON_ABORTED,
        None,
        1,
    )
    .await;
    let attempts = recorder.attempts();
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].outcome,
        Some(PerformanceMetricOutcome::Cancelled)
    );
    assert_eq!(streaming_ms(&attempts[0]), Some(500.0));
    assert_eq!(attempts[0].usage.as_ref().unwrap().output_tokens, None);
    let progress = recorder.progress.lock().unwrap();
    assert_eq!(progress.len(), 2);
    assert_eq!((progress[1].1, progress[1].2), (4.0, 500.0));
}
