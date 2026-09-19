//! Opt-in, content-free compaction timings. Recorder failures never own liveness.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use pi_agent_core::performance_metrics::{
    elapsed_metric_ms, performance_metric_usage_from_assistant, provider_metric_usage,
    safe_record_performance_metric, PerformanceMetricComponent, PerformanceMetricCorrelation,
    PerformanceMetricEvent, PerformanceMetricIdScope, PerformanceMetricIdentity,
    PerformanceMetricMeasurement as Measurement, PerformanceMetricOperation as Operation,
    PerformanceMetricOutcome as Outcome, PerformanceMetricRecorder, PerformanceMetricUsageV1,
};
use pi_ai::types::{AssistantMessage, Model, SimpleStreamOptions};

#[derive(Clone)]
pub struct CompactionMetrics {
    recorder: Option<Arc<dyn PerformanceMetricRecorder>>,
    action_id: Option<String>,
    identity: PerformanceMetricIdentity,
}

impl CompactionMetrics {
    pub fn new(recorder: Option<Arc<dyn PerformanceMetricRecorder>>, model: &Model) -> Self {
        let action_id = safe_id(recorder.as_ref(), PerformanceMetricIdScope::LogicalRequest);
        Self {
            recorder,
            action_id,
            identity: PerformanceMetricIdentity {
                provider: Some(Some(model.provider.clone())),
                model: Some(Some(model.id.clone())),
                api: Some(Some(model.api.clone())),
                component: Some(PerformanceMetricComponent::Compaction),
            },
        }
    }

    pub fn phase(&self, operation: Operation) -> CompactionPhase {
        CompactionPhase::new(self.clone(), operation, None, None)
    }
}

fn safe_id(
    recorder: Option<&Arc<dyn PerformanceMetricRecorder>>,
    scope: PerformanceMetricIdScope,
) -> Option<String> {
    let recorder = recorder?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recorder.next_id(scope))).ok()
}

fn safe_now(recorder: Option<&Arc<dyn PerformanceMetricRecorder>>) -> Option<f64> {
    let recorder = recorder?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recorder.monotonic_now())).ok()
}

pub struct CompactionPhase {
    metrics: CompactionMetrics,
    event: PerformanceMetricEvent,
    started_at: Option<f64>,
    finished: bool,
}

impl CompactionPhase {
    fn new(
        metrics: CompactionMetrics,
        operation: Operation,
        logical_id: Option<String>,
        ordinal: Option<u64>,
    ) -> Self {
        let mut event = PerformanceMetricEvent::new(operation);
        event.identity = Some(metrics.identity.clone());
        event.correlation = Some(PerformanceMetricCorrelation {
            action_id: metrics.action_id.clone(),
            logical_request_id: logical_id.or_else(|| {
                safe_id(
                    metrics.recorder.as_ref(),
                    PerformanceMetricIdScope::LogicalRequest,
                )
            }),
            provider_attempt_id: ordinal.and_then(|_| {
                safe_id(
                    metrics.recorder.as_ref(),
                    PerformanceMetricIdScope::ProviderAttempt,
                )
            }),
            tool_call_id: None,
        });
        let started_at = safe_now(metrics.recorder.as_ref());
        if let Some(ordinal) = ordinal {
            event
                .measurements
                .get_or_insert_with(Default::default)
                .insert(Measurement::AttemptOrdinal, Some(ordinal as f64));
        }
        // The start row is explicitly labeled `started` (never null, never a
        // failure) so paired accounting cannot mistake it for a terminal; the
        // terminal event reuses the same IDs with a terminal outcome.
        event.outcome = Some(Outcome::Started);
        safe_record_performance_metric(metrics.recorder.as_ref(), event.clone());
        Self {
            metrics,
            event,
            started_at,
            finished: false,
        }
    }

    pub fn measurement(&mut self, key: Measurement, value: Option<f64>) {
        self.event
            .measurements
            .get_or_insert_with(Default::default)
            .insert(key, value);
    }

    pub fn requests(&self) -> CompactionRequestFactory {
        CompactionRequestFactory {
            metrics: self.metrics.clone(),
            logical_id: self
                .event
                .correlation
                .as_ref()
                .and_then(|c| c.logical_request_id.clone()),
            count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn finish(mut self, outcome: Outcome) {
        self.record_terminal(outcome);
    }

    pub fn finish_result<T, E>(self, result: &Result<T, E>, cancelled: bool) {
        self.finish(if cancelled {
            Outcome::Cancelled
        } else if result.is_ok() {
            Outcome::Success
        } else {
            Outcome::Failure
        });
    }

    fn record_terminal(&mut self, outcome: Outcome) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.event.outcome = Some(outcome);
        self.measurement(
            Measurement::TotalMs,
            elapsed_metric_ms(self.started_at, safe_now(self.metrics.recorder.as_ref())),
        );
        safe_record_performance_metric(self.metrics.recorder.as_ref(), self.event.clone());
    }
}

#[derive(Clone)]
pub struct CompactionRequestFactory {
    metrics: CompactionMetrics,
    logical_id: Option<String>,
    count: Arc<AtomicU64>,
}

impl CompactionRequestFactory {
    pub fn next(&self) -> CompactionRequestMetrics {
        let ordinal = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        CompactionRequestMetrics {
            phase: CompactionPhase::new(
                self.metrics.clone(),
                Operation::ProviderAttempt,
                self.logical_id.clone(),
                Some(ordinal),
            ),
            state: Arc::new(Mutex::new(RequestState::default())),
        }
    }
}

impl Drop for CompactionPhase {
    fn drop(&mut self) {
        // Dropping an in-flight future is cancellation, not a provider failure.
        self.record_terminal(Outcome::Cancelled);
    }
}

#[derive(Default)]
struct RequestState {
    dispatch: Option<f64>,
    headers: Option<f64>,
    first_raw: Option<f64>,
    thinking: Option<f64>,
    tool: Option<f64>,
    text: Option<f64>,
    terminal: Option<f64>,
    websocket: Option<f64>,
    usage: Option<PerformanceMetricUsageV1>,
}

pub struct CompactionRequestMetrics {
    phase: CompactionPhase,
    state: Arc<Mutex<RequestState>>,
}

impl CompactionRequestMetrics {
    pub fn observe(&self, options: &mut SimpleStreamOptions) {
        if self.phase.metrics.recorder.is_none() {
            return;
        }
        let state = self.state.clone();
        let recorder = self.phase.metrics.recorder.clone();
        let caller = options.stream.on_payload.clone();
        options.stream.on_payload = Some(Arc::new(move |payload, model| {
            let future = caller.as_ref().map(|caller| caller(payload, model));
            let state = state.clone();
            let recorder = recorder.clone();
            Box::pin(async move {
                let replacement = match future {
                    Some(future) => future.await,
                    None => None,
                };
                if let Ok(mut state) = state.lock() {
                    state
                        .dispatch
                        .get_or_insert_with(|| safe_now(recorder.as_ref()).unwrap_or(f64::NAN));
                }
                replacement
            })
        }));
        let state = self.state.clone();
        let recorder = self.phase.metrics.recorder.clone();
        let caller = options.stream.on_response.clone();
        options.stream.on_response = Some(Arc::new(move |response, model| {
            if let Ok(mut state) = state.lock() {
                // B2: a WebSocket send acknowledgement is not an HTTP header edge, so it
                // must not claim `dispatch_to_response_headers_ms` here either. The
                // compaction phase has no dedicated ack stage, so it records none.
                let is_send_ack = response
                    .headers
                    .get("x-optimus-response-edge")
                    .map(String::as_str)
                    == Some("transport_send_ack");
                if !is_send_ack {
                    state.headers = state.headers.or_else(|| safe_now(recorder.as_ref()));
                }
                match response
                    .headers
                    .get("x-optimus-transport")
                    .map(String::as_str)
                {
                    Some("websocket") => state.websocket = Some(1.0),
                    Some("sse") => state.websocket = Some(0.0),
                    _ => {}
                }
            }
            caller
                .as_ref()
                .map(|caller| caller(response, model))
                .unwrap_or_else(|| Box::pin(async {}))
        }));
        let state = self.state.clone();
        let recorder = self.phase.metrics.recorder.clone();
        let caller = options.stream.on_stream_observation.clone();
        options.stream.on_stream_observation = Some(Arc::new(move |stage| {
            if let Ok(mut state) = state.lock() {
                // B5: a provider whose WebSocket transport reports no response header edge
                // labels its transport on its own stage, so the attempt is comparable with
                // an SSE attempt of the same provider.
                if stage == "transport_ws" {
                    if state.websocket.is_none() {
                        state.websocket = Some(1.0);
                    }
                } else {
                    let slot = match stage {
                        "raw_event" => Some(&mut state.first_raw),
                        "thinking" => Some(&mut state.thinking),
                        "tool" => Some(&mut state.tool),
                        "text" => Some(&mut state.text),
                        "terminal" => Some(&mut state.terminal),
                        _ => None,
                    };
                    if let Some(slot) = slot {
                        *slot = slot.or_else(|| safe_now(recorder.as_ref()));
                    }
                }
            }
            if let Some(caller) = &caller {
                caller(stage);
            }
        }));
        let state = self.state.clone();
        let caller = options.stream.on_usage_observation.clone();
        options.stream.on_usage_observation = Some(Arc::new(move |observation, model| {
            if let Ok(mut state) = state.lock() {
                state.usage = Some(provider_metric_usage(&observation));
            }
            caller
                .as_ref()
                .map(|caller| caller(observation, model))
                .unwrap_or_else(|| Box::pin(async {}))
        }));
    }

    pub fn finish(mut self, message: &AssistantMessage, cancelled: bool) {
        self.capture(Some(message));
        let outcome = if cancelled || message.stop_reason == "aborted" {
            Outcome::Cancelled
        } else if message.stop_reason == "stop" && message.error_message.is_none() {
            Outcome::Success
        } else {
            Outcome::Failure
        };
        self.phase.record_terminal(outcome);
    }

    fn capture(&mut self, message: Option<&AssistantMessage>) {
        if let Ok(state) = self.state.lock() {
            for (key, at) in [
                (Measurement::DispatchToResponseHeadersMs, state.headers),
                (Measurement::DispatchToFirstRawMs, state.first_raw),
                (Measurement::DispatchToFirstThinkingMs, state.thinking),
                (Measurement::DispatchToFirstToolMs, state.tool),
                (Measurement::DispatchToFirstTextMs, state.text),
                (Measurement::DispatchToNetworkTerminalMs, state.terminal),
            ] {
                self.phase
                    .measurement(key, elapsed_metric_ms(state.dispatch, at));
            }
            self.phase.measurement(
                Measurement::LocalDrainMs,
                elapsed_metric_ms(
                    state.terminal,
                    safe_now(self.phase.metrics.recorder.as_ref()),
                ),
            );
            self.phase
                .measurement(Measurement::TransportWebsocket, state.websocket);
            self.phase.event.usage = state
                .usage
                .clone()
                .or_else(|| message.map(performance_metric_usage_from_assistant));
        }
    }
}

impl Drop for CompactionRequestMetrics {
    fn drop(&mut self) {
        if !self.phase.finished {
            self.capture(None);
        }
    }
}
