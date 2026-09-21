use super::*;
use pi_ai::types::{AssistantMessageEvent, ContentBlock, ToolCall};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::stream_failure::{
    record_stream_failure, StreamFailureError, StreamFailureInfo, ThrownStreamError,
};
use std::sync::atomic::AtomicUsize;

async fn assert_attempts(kind: &str, content: Vec<ContentBlock>, expected: usize) {
    assert_attempt_metrics(kind, content, expected, true, false).await;
}

async fn assert_attempt_metrics(
    kind: &str,
    content: Vec<ContentBlock>,
    expected: usize,
    retry_enabled: bool,
    repeat_failure: bool,
) {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    session.settings_manager.lock().unwrap().apply_overrides(
        serde_json::json!({"retry": {"enabled": retry_enabled, "maxRetries": 1, "baseDelayMs": 0}})
            .as_object()
            .unwrap(),
    );
    let recorder = Arc::new(RetryMetricRecorder::default());
    session
        .agent
        .set_performance_metrics(Some(AgentLoopPerformanceMetrics::new(recorder.clone())));
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    let kind = kind.to_string();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let index = count.fetch_add(1, Ordering::SeqCst);
        let kind = kind.clone();
        let content = content.clone();
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            let mut message = AssistantMessage::new(
                model.api.clone(),
                model.provider.clone(),
                model.id.clone(),
                0,
            );
            stream.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            if index == 0 || repeat_failure {
                message.content = content;
                message.stop_reason = "error".into();
                message.error_message = Some("Isolated provider interruption".into());
                let failure = StreamFailureError::new(
                    "Isolated provider interruption",
                    StreamFailureInfo {
                        kind,
                        ..Default::default()
                    },
                );
                record_stream_failure(&model, &mut message, &ThrownStreamError::Failure(&failure));
                stream.push(AssistantMessageEvent::Error {
                    reason: "error".into(),
                    error: message,
                });
            } else {
                message.content = vec![ContentBlock::Text(TextContent::new("retried"))];
                stream.push(AssistantMessageEvent::Done {
                    reason: "stop".into(),
                    message,
                });
            }
            stream.end(None);
            stream
        })
    }));
    tokio::time::timeout(
        std::time::Duration::from_secs(8),
        session.prompt("isolated retry test", None),
    )
    .await
    .expect("prompt bounded")
    .expect("prompt settled");
    tokio::time::timeout(std::time::Duration::from_secs(8), session.wait_for_idle())
        .await
        .expect("idle bounded")
        .expect("idle settled");
    let actual = attempts.load(Ordering::SeqCst);
    if expected == 1 {
        assert!(session.agent.state().messages.iter().any(|entry| matches!(entry,
            AgentMessage::Message(Message::Assistant(message)) if message.stop_reason == "error"
        )), "the interrupted response must remain visible instead of being dropped for replay");
    }
    let events = recorder.0.lock().unwrap().clone();
    let attempt_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.operation
                == pi_agent_core::performance_metrics::PerformanceMetricOperation::ProviderAttempt
        })
        .collect();
    let terminal_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.operation
                == pi_agent_core::performance_metrics::PerformanceMetricOperation::LogicalRequest
        })
        .collect();
    assert_eq!(
        attempt_events.len(),
        expected,
        "every provider attempt is measured"
    );
    assert_eq!(
        terminal_events.len(),
        1,
        "one request, including any retry, settles exactly once"
    );
    let terminal = terminal_events[0];
    let group_id = terminal
        .correlation
        .as_ref()
        .unwrap()
        .logical_request_id
        .as_ref()
        .unwrap();
    for (index, attempt) in attempt_events.iter().enumerate() {
        assert_eq!(
            attempt
                .correlation
                .as_ref()
                .unwrap()
                .logical_request_id
                .as_ref(),
            Some(group_id)
        );
        assert_eq!(
            attempt.measurements.as_ref().unwrap().get(
                &pi_agent_core::performance_metrics::PerformanceMetricMeasurement::AttemptOrdinal,
            ),
            Some(&Some((index + 1) as f64)),
            "retries retain their group and advance its ordinal",
        );
    }
    assert_eq!(
        terminal
            .measurements
            .as_ref()
            .unwrap()
            .get(&pi_agent_core::performance_metrics::PerformanceMetricMeasurement::AttemptCount,),
        Some(&Some(expected as f64)),
    );
    assert_eq!(
        terminal.outcome,
        Some(if repeat_failure || expected == 1 {
            pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure
        } else {
            pi_agent_core::performance_metrics::PerformanceMetricOutcome::Success
        }),
    );
    session.dispose_async(Some(false)).await;
    assert_eq!(actual, expected, "automatic request count");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uncertain_websocket_send_never_replays_even_before_visible_output() {
    assert_attempts("request_interrupted", vec![], 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_error_after_partial_text_never_replays() {
    assert_attempts(
        "server_error",
        vec![ContentBlock::Text(TextContent::new("already shown"))],
        1,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_error_after_tool_proposal_never_replays() {
    assert_attempts(
        "server_error",
        vec![ContentBlock::ToolCall(ToolCall::new(
            "partial",
            "never_execute",
            Default::default(),
        ))],
        1,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clean_pre_output_rate_limit_still_retries_once_then_succeeds() {
    assert_attempts("rate_limit", vec![], 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabled_retry_settles_the_initial_error_metric() {
    assert_attempt_metrics("server_error", vec![], 1, false, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhausted_retry_group_settles_one_failure_with_two_attempts() {
    assert_attempt_metrics("server_error", vec![], 2, true, true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_retry_backoff_settles_one_cancelled_request_without_replay() {
    use pi_agent_core::performance_metrics::{
        PerformanceMetricMeasurement, PerformanceMetricOperation, PerformanceMetricOutcome,
    };
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    session.settings_manager.lock().unwrap().apply_overrides(
        serde_json::json!({"retry": {"enabled": true, "maxRetries": 1, "baseDelayMs": 10000}})
            .as_object()
            .unwrap(),
    );
    let recorder = Arc::new(RetryMetricRecorder::default());
    session
        .agent
        .set_performance_metrics(Some(AgentLoopPerformanceMetrics::new(recorder.clone())));
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            let mut message = AssistantMessage::new(
                model.api.clone(),
                model.provider.clone(),
                model.id.clone(),
                0,
            );
            message.stop_reason = "error".into();
            message.error_message = Some("isolated retry cancellation fixture".into());
            let failure = StreamFailureError::new(
                "isolated retry cancellation fixture",
                StreamFailureInfo {
                    kind: "server_error".into(),
                    ..Default::default()
                },
            );
            record_stream_failure(&model, &mut message, &ThrownStreamError::Failure(&failure));
            stream.push(AssistantMessageEvent::Error {
                reason: "error".into(),
                error: message,
            });
            stream.end(None);
            stream
        })
    }));
    let prompt_session = session.clone();
    let prompt =
        tokio::spawn(async move { prompt_session.prompt("cancel pending retry", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while session.retry_abort_controller.lock().unwrap().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("retry reaches cancellable backoff");
    assert!(
        !recorder
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|event| event.operation == PerformanceMetricOperation::LogicalRequest),
        "failed attempt stays unsettled while retry is pending"
    );
    session.abort_retry();
    tokio::time::timeout(std::time::Duration::from_secs(3), prompt)
        .await
        .expect("prompt bounded")
        .expect("prompt task joined")
        .expect("prompt settled");
    tokio::time::timeout(std::time::Duration::from_secs(3), session.wait_for_idle())
        .await
        .expect("idle bounded")
        .expect("idle settled");
    let events = recorder.0.lock().unwrap().clone();
    let terminals: Vec<_> = events
        .iter()
        .filter(|event| event.operation == PerformanceMetricOperation::LogicalRequest)
        .collect();
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "cancellation must not replay the request"
    );
    assert_eq!(terminals.len(), 1);
    assert_eq!(
        terminals[0].outcome,
        Some(PerformanceMetricOutcome::Cancelled)
    );
    assert_eq!(
        terminals[0]
            .measurements
            .as_ref()
            .unwrap()
            .get(&PerformanceMetricMeasurement::AttemptCount),
        Some(&Some(1.0))
    );
    session.dispose_async(Some(false)).await;
}

/// B6: a host-owned retry group settles exactly one outer logical-request terminal, and
/// no later attempt is attributed to that finished group.
#[derive(Default)]
struct RetryMetricRecorder(Mutex<Vec<pi_agent_core::performance_metrics::PerformanceMetricEvent>>);

impl PerformanceMetricRecorder for RetryMetricRecorder {
    fn session_id(&self) -> &str {
        "retry-metric-fixture"
    }
    fn monotonic_now(&self) -> f64 {
        now_ms()
    }
    fn next_id(
        &self,
        scope: pi_agent_core::performance_metrics::PerformanceMetricIdScope,
    ) -> String {
        format!("{scope:?}-{}", uuid::Uuid::new_v4())
    }
    fn record(&self, event: pi_agent_core::performance_metrics::PerformanceMetricEvent) {
        self.0.lock().unwrap().push(event);
    }
    fn flush(&self) {}
    fn close(&self) {}
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_group_settles_one_terminal_and_later_turns_start_a_new_group() {
    use pi_agent_core::performance_metrics::PerformanceMetricEvent;
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    session
        .settings_manager
        .lock()
        .unwrap()
        .set_retry_enabled(true);
    let recorder = Arc::new(RetryMetricRecorder::default());
    session
        .agent
        .set_performance_metrics(Some(AgentLoopPerformanceMetrics::new(recorder.clone())));
    let attempts = Arc::new(AtomicUsize::new(0));
    let count = attempts.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let index = count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            let mut message = AssistantMessage::new(
                model.api.clone(),
                model.provider.clone(),
                model.id.clone(),
                0,
            );
            stream.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            if index == 0 {
                message.stop_reason = "error".into();
                message.error_message = Some("Isolated provider interruption".into());
                let failure = StreamFailureError::new(
                    "Isolated provider interruption",
                    StreamFailureInfo {
                        kind: "server_error".into(),
                        ..Default::default()
                    },
                );
                record_stream_failure(&model, &mut message, &ThrownStreamError::Failure(&failure));
                stream.push(AssistantMessageEvent::Error {
                    reason: "error".into(),
                    error: message,
                });
            } else {
                message.content = vec![ContentBlock::Text(TextContent::new("retried"))];
                stream.push(AssistantMessageEvent::Done {
                    reason: "stop".into(),
                    message,
                });
            }
            stream.end(None);
            stream
        })
    }));
    // Turn one exercises the retry group; turn two runs strictly after it finished.
    for prompt in ["retry group turn one", "later turn two"] {
        tokio::time::timeout(
            std::time::Duration::from_secs(8),
            session.prompt(prompt, None),
        )
        .await
        .expect("prompt bounded")
        .expect("prompt settled");
        tokio::time::timeout(std::time::Duration::from_secs(8), session.wait_for_idle())
            .await
            .expect("idle bounded")
            .expect("idle settled");
    }
    let events = recorder.0.lock().unwrap().clone();
    let logical_id = |event: &PerformanceMetricEvent| {
        event
            .correlation
            .as_ref()
            .and_then(|correlation| correlation.logical_request_id.clone())
    };
    let provider_attempts: Vec<_> = events
        .iter()
        .filter(|event| {
            event.operation
                == pi_agent_core::performance_metrics::PerformanceMetricOperation::ProviderAttempt
        })
        .collect();
    assert_eq!(
        provider_attempts.len(),
        3,
        "failure, successful retry, then later prompt"
    );
    let retry_id = logical_id(provider_attempts[0]).expect("initial request id");
    assert_eq!(
        logical_id(provider_attempts[1]).as_deref(),
        Some(retry_id.as_str())
    );
    let later_id = logical_id(provider_attempts[2]).expect("later request id");
    assert_ne!(retry_id, later_id, "later prompt starts a fresh group");
    for (attempt, expected_ordinal) in provider_attempts.iter().zip([1.0, 2.0, 1.0]) {
        assert_eq!(
            attempt.measurements.as_ref().unwrap().get(
                &pi_agent_core::performance_metrics::PerformanceMetricMeasurement::AttemptOrdinal,
            ),
            Some(&Some(expected_ordinal)),
        );
    }
    // Index every logical-request terminal by its group id.
    let mut terminals: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (index, event) in events.iter().enumerate() {
        if event.operation
            != pi_agent_core::performance_metrics::PerformanceMetricOperation::LogicalRequest
        {
            continue;
        }
        let Some(id) = logical_id(event) else {
            continue;
        };
        assert!(
            terminals.insert(id.clone(), index).is_none(),
            "a logical request must settle at most one terminal, but {id} settled twice"
        );
    }
    assert_eq!(
        terminals.len(),
        2,
        "the failed attempt must not prematurely settle a separate group"
    );
    for (id, expected_count) in [(&retry_id, 2.0), (&later_id, 1.0)] {
        let terminal = &events[*terminals.get(id).expect("each group has a terminal")];
        assert_eq!(
            terminal.outcome,
            Some(pi_agent_core::performance_metrics::PerformanceMetricOutcome::Success)
        );
        assert_eq!(
            terminal.measurements.as_ref().unwrap().get(
                &pi_agent_core::performance_metrics::PerformanceMetricMeasurement::AttemptCount,
            ),
            Some(&Some(expected_count)),
        );
    }
    // No provider attempt may appear after its own group terminalized.
    for (index, event) in events.iter().enumerate() {
        if event.operation
            != pi_agent_core::performance_metrics::PerformanceMetricOperation::ProviderAttempt
        {
            continue;
        }
        let Some(id) = logical_id(event) else {
            continue;
        };
        let Some(terminal) = terminals.get(&id) else {
            continue;
        };
        assert!(
            index < *terminal,
            "attempt {index} is attributed to logical request {id}, which already terminalized at {terminal}"
        );
    }
    session.dispose_async(Some(false)).await;
}

fn prepare_retry(session: &AgentSession) {
    session
        .settings_manager
        .lock()
        .unwrap()
        .set_retry_enabled(true);
    let message = AssistantMessage {
        stop_reason: "error".into(),
        error_message: Some("retry fixture failure".into()),
        ..Default::default()
    };
    session.create_retry_promise_for_agent_end(&AgentEvent::AgentEnd {
        messages: vec![AgentMessage::Message(Message::Assistant(message))],
    });
    assert!(session.is_retrying());
}

#[tokio::test]
async fn retry_waiters_share_pending_state_and_completion_never_resurrects_busy() {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    prepare_retry(&session);
    let mut first = Box::pin(session.wait_for_retry());
    let mut second = Box::pin(session.wait_for_retry());
    assert!(futures::poll!(first.as_mut()).is_pending());
    assert!(
        session.is_retrying(),
        "one waiter cannot consume shared retry state"
    );
    assert!(
        futures::poll!(second.as_mut()).is_pending(),
        "both waiters must await settlement"
    );
    session.resolve_retry();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(first, second);
    })
    .await
    .expect("both waiters settle");
    assert!(
        !session.is_retrying(),
        "completed waiters cannot reinsert ready promises"
    );
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn settled_retry_waiter_cannot_clear_the_next_retry_generation() {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    prepare_retry(&session);
    let mut old = Box::pin(session.wait_for_retry());
    assert!(futures::poll!(old.as_mut()).is_pending());
    session.resolve_retry();
    prepare_retry(&session);
    tokio::time::timeout(std::time::Duration::from_secs(2), old)
        .await
        .unwrap();
    assert!(
        session.is_retrying(),
        "old waiter cannot settle a new generation"
    );
    session.resolve_retry();
    assert!(!session.is_retrying());
    session.dispose_async(Some(false)).await;
}
