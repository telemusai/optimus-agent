//! Content-free client timings. No prompt, key, output or error text is recorded.
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use pi_agent_core::performance_metrics::{
    safe_record_performance_metric, PerformanceMetricComponent, PerformanceMetricCorrelation,
    PerformanceMetricEvent, PerformanceMetricIdentity, PerformanceMetricMeasurement as M,
    PerformanceMetricOperation as Op, PerformanceMetricOutcome as Outcome, PerformanceMetricRecorder,
};
use crate::core::performance_monitor::PerformanceMonitor;

pub(super) struct UiMetrics {
    pub recorder: Arc<dyn PerformanceMetricRecorder>,
    session_id: String,
    input: Option<Instant>,
    menus: Vec<Instant>,
    frames: u64,
    render_ms: f64,
    max_render_ms: f64,
    window: Instant,
    attachment_generation: u64,
    pending: HashMap<SubmissionKey, PendingSubmission>,
    apply_count: u64,
    apply_ms: f64,
    max_apply_ms: f64,
    queue_ms: f64,
    queue_count: u64,
    fallback_ticks: u64,
    /// A12: fallback ticks whose repaint was skipped by the idle gate.
    fallback_ticks_skipped: u64,
    /// A15: per-phase render sums for the current window.
    phase_render_ms: f64,
    phase_diff_ms: f64,
    phase_write_ms: f64,
    /// A13: deadline after which a pending ack is settled as `timeout`.
    ack_deadline: Duration,
}

/// A13: default prompt-ack deadline (10s). p95 submit-to-ack is 2.1s, so a
/// healthy ack never trips it; compaction-gated prompts (observed >10s) get an
/// explicit `timeout` outcome instead of hanging or inflating the drop path.
const DEFAULT_ACK_DEADLINE: Duration = Duration::from_secs(10);

/// A13: `OPTIMUS_UI_ACK_DEADLINE_MS` overrides the ack deadline. Clamped to
/// 100ms..=10min; `0` disables the deadline (pre-change behavior for the
/// expiry path). Read once per process.
fn ack_deadline_from_environment() -> Duration {
    static DEADLINE_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let milliseconds = *DEADLINE_MS.get_or_init(|| {
        std::env::var("OPTIMUS_UI_ACK_DEADLINE_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_ACK_DEADLINE.as_millis() as u64)
    });
    match milliseconds {
        0 => Duration::from_secs(u64::MAX / 2), // disabled: effectively never expires
        value => Duration::from_millis(value.max(100).min(10 * 60 * 1000)),
    }
}

pub(super) fn duration_event(op: Op, elapsed: Duration, outcome: Outcome) -> PerformanceMetricEvent {
    let mut event = PerformanceMetricEvent::new(op);
    event.identity = Some(PerformanceMetricIdentity { component: Some(PerformanceMetricComponent::Session), ..Default::default() });
    event.outcome = Some(outcome);
    event.measurements = Some([(M::TotalMs, Some(elapsed.as_secs_f64() * 1000.0))].into_iter().collect());
    event
}

impl UiMetrics {
    pub fn new(session_id: &str) -> Self {
        Self::with_recorder(session_id, Arc::new(PerformanceMonitor::from_environment(crate::config::get_agent_dir()).recorder(session_id.into())))
    }
    pub(super) fn with_recorder(session_id: &str, recorder: Arc<dyn PerformanceMetricRecorder>) -> Self {
        Self::with_recorder_and_ack_deadline(session_id, recorder, ack_deadline_from_environment())
    }

    /// A13: test/override constructor with an explicit ack deadline.
    pub(super) fn with_recorder_and_ack_deadline(session_id: &str, recorder: Arc<dyn PerformanceMetricRecorder>, ack_deadline: Duration) -> Self {
        Self { recorder, session_id: session_id.into(), input: None, menus: Vec::new(), frames: 0, render_ms: 0.0, max_render_ms: 0.0, window: Instant::now(), attachment_generation: 0, pending: HashMap::new(), apply_count: 0, apply_ms: 0.0, max_apply_ms: 0.0, queue_ms: 0.0, queue_count: 0, fallback_ticks: 0, fallback_ticks_skipped: 0, phase_render_ms: 0.0, phase_diff_ms: 0.0, phase_write_ms: 0.0, ack_deadline }
    }
    pub fn session(&mut self, session_id: &str) {
        if self.session_id != session_id {
            self.reset_attachment();
            self.flush_render();
            let generation = self.attachment_generation;
            *self = Self::new(session_id);
            self.attachment_generation = generation;
        }
    }
    pub fn reset_attachment(&mut self) {
        let deadline = self.ack_deadline;
        for (_, pending) in self.pending.drain() {
            // A13: an ack settling on drop never saw its reply. Classify it as
            // an ack timeout with the deadline-capped duration instead of
            // `cancelled` + full wall clock, which reported failure acks up to
            // 392s in the 15-day review (settle-on-drop artifact).
            pending.ticket.settle_timeout(deadline);
        }
        self.attachment_generation = self.attachment_generation.wrapping_add(1);
    }
    pub fn begin_submission(&mut self) -> SubmissionTicket {
        if self.pending.len() == MAX_PENDING_SUBMISSIONS {
            let oldest = self.pending.iter().min_by_key(|(_, pending)| pending.ticket.submitted)
                .map(|(key, _)| key.clone()).expect("bounded pending entry");
            if let Some(pending) = self.pending.remove(&oldest) {
                pending.ticket.settle(Outcome::Unavailable, None, None, None);
            }
        }
        let ticket = SubmissionTicket {
            key: SubmissionKey { session_id: self.session_id.clone(), attachment_generation: self.attachment_generation,
                submission_id: format!("UiInput-{}", uuid::Uuid::new_v4()) },
            submitted: Instant::now(), recorder: self.recorder.clone(), settled: Arc::new(AtomicBool::new(false)),
        };
        let mut event = ticket.event(Op::UiInputAck, Duration::ZERO, Outcome::Started);
        event.measurements.as_mut().expect("duration measurements").insert(M::UiPendingCount, Some((self.pending.len() + 1) as f64));
        safe_record_performance_metric(Some(&ticket.recorder), event);
        self.pending.insert(ticket.key.clone(), PendingSubmission { ticket: ticket.clone(), rendered: false });
        ticket
    }
    pub fn pending_count(&self) -> usize { self.pending.len() }
    pub fn reply(&mut self, ticket: &SubmissionTicket) -> bool {
        if ticket.key.session_id != self.session_id || ticket.key.attachment_generation != self.attachment_generation { return false; }
        self.pending.remove(&ticket.key).is_some()
    }
    pub fn receipt_rendered(&mut self) {
        for pending in self.pending.values_mut().filter(|pending| !pending.rendered) {
            pending.rendered = true;
            let mut event = pending.ticket.event(Op::UiInputAck, Duration::ZERO, Outcome::Started);
            event.measurements.as_mut().expect("duration measurements").insert(M::UiSubmitToReceiptRenderMs, Some(pending.ticket.submitted.elapsed().as_secs_f64() * 1000.0));
            safe_record_performance_metric(Some(&pending.ticket.recorder), event);
        }
    }
    pub fn applied(&mut self, elapsed: Duration, queue_age: Option<Duration>) {
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.apply_count += 1;
        self.apply_ms += ms;
        self.max_apply_ms = self.max_apply_ms.max(ms);
        if let Some(age) = queue_age { self.queue_ms += age.as_secs_f64() * 1000.0; self.queue_count += 1; }
    }
    pub fn fallback_tick(&mut self) { self.fallback_ticks += 1; }
    /// A12: count a fallback tick whose repaint was skipped by the idle gate.
    pub fn fallback_tick_skipped(&mut self) { self.fallback_ticks_skipped += 1; }
    /// A13: settle pending submissions whose ack deadline elapsed. Called from
    /// the host's 16ms loop; the underlying submission future keeps running and
    /// a late reply is ignored exactly like a reply after a cancelled ticket.
    pub fn expire_ack_deadlines(&mut self) {
        let deadline = self.ack_deadline;
        let expired: Vec<SubmissionKey> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.ticket.submitted.elapsed() >= deadline)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            if let Some(pending) = self.pending.remove(&key) {
                pending.ticket.settle_timeout(deadline);
            }
        }
    }
    pub fn input(&mut self, received: Instant) {
        self.input.get_or_insert(received);
    }
    pub fn menu(&mut self, started: Instant) {
        if self.menus.len() < 32 { self.menus.push(started); }
    }
    pub fn first_frame(&self, started: Instant) {
        safe_record_performance_metric(Some(&self.recorder), duration_event(Op::UiSessionOpen, started.elapsed(), Outcome::Success));
    }
    pub fn rendered(&mut self, elapsed: Duration, phases: &pi_tui::tui::RenderPhaseTimings) {
        self.frames += 1;
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.render_ms += ms;
        self.max_render_ms = self.max_render_ms.max(ms);
        // A15: cheap per-phase sums make unattributed worst frames diagnosable
        // (46/100 worst freezes had no instrumented cause in the 15-day review).
        self.phase_render_ms += phases.render_ms;
        self.phase_diff_ms += phases.diff_ms;
        self.phase_write_ms += phases.write_ms;
        if let Some(started) = self.input.take() {
            safe_record_performance_metric(Some(&self.recorder), duration_event(Op::UiInput, started.elapsed(), Outcome::Success));
        }
        for started in self.menus.drain(..) {
            safe_record_performance_metric(Some(&self.recorder), duration_event(Op::UiMenuOpen, started.elapsed(), Outcome::Success));
        }
        if self.window.elapsed() >= Duration::from_secs(1) { self.flush_render(); }
    }
    pub fn flush_render(&mut self) {
        if self.apply_count > 0 {
            let mut event = duration_event(Op::UiEventApply, Duration::from_secs_f64(self.apply_ms / 1000.0), Outcome::Success);
            let values = event.measurements.as_mut().expect("duration measurements");
            values.insert(M::UiEventCount, Some(self.apply_count as f64));
            values.insert(M::MaxMs, Some(self.max_apply_ms));
            values.insert(M::QueueMs, (self.queue_count == self.apply_count).then_some(self.queue_ms));
            values.insert(M::SerializedBytes, None);
            safe_record_performance_metric(Some(&self.recorder), event);
            self.apply_count = 0; self.apply_ms = 0.0; self.max_apply_ms = 0.0; self.queue_ms = 0.0; self.queue_count = 0;
        }
        if self.fallback_ticks > 0 || self.fallback_ticks_skipped > 0 {
            let mut event = PerformanceMetricEvent::new(Op::UiTick);
            event.identity = Some(PerformanceMetricIdentity { component: Some(PerformanceMetricComponent::Session), ..Default::default() });
            event.measurements = Some([(M::UiTickFallbackCount, Some(self.fallback_ticks as f64))].into_iter().collect());
            // A12: additive counter of gated-out fallback paints.
            event.measurements.as_mut().expect("ui tick measurements").insert(M::UiTickFallbackSkipped, Some(self.fallback_ticks_skipped as f64));
            safe_record_performance_metric(Some(&self.recorder), event);
            self.fallback_ticks = 0;
            self.fallback_ticks_skipped = 0;
        }
        if self.frames == 0 { return; }
        let mut event = duration_event(Op::UiRender, Duration::from_secs_f64(self.render_ms / 1000.0), Outcome::Success);
        let measurements = event.measurements.as_mut().expect("duration measurements");
        measurements.insert(M::FrameCount, Some(self.frames as f64));
        measurements.insert(M::MaxMs, Some(self.max_render_ms));
        // A15: per-phase sums for the window (component render, compose/diff,
        // terminal write). Zero when the host could not attribute a phase.
        measurements.insert(M::RenderMs, Some(self.phase_render_ms));
        measurements.insert(M::DiffMs, Some(self.phase_diff_ms));
        measurements.insert(M::WriteMs, Some(self.phase_write_ms));
        safe_record_performance_metric(Some(&self.recorder), event);
        self.frames = 0;
        self.render_ms = 0.0;
        self.max_render_ms = 0.0;
        self.phase_render_ms = 0.0;
        self.phase_diff_ms = 0.0;
        self.phase_write_ms = 0.0;
        self.window = Instant::now();
    }
}

const MAX_PENDING_SUBMISSIONS: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SubmissionKey {
    session_id: String,
    attachment_generation: u64,
    submission_id: String,
}

#[derive(Clone)]
pub(super) struct SubmissionTicket {
    key: SubmissionKey,
    submitted: Instant,
    recorder: Arc<dyn PerformanceMetricRecorder>,
    settled: Arc<AtomicBool>,
}
struct PendingSubmission { ticket: SubmissionTicket, rendered: bool }

impl SubmissionTicket {
    pub fn id(&self) -> &str { &self.key.submission_id }

    fn event(&self, op: Op, elapsed: Duration, outcome: Outcome) -> PerformanceMetricEvent {
        let mut event = duration_event(op, elapsed, outcome);
        event.correlation = Some(PerformanceMetricCorrelation { action_id: Some(self.key.submission_id.clone()), ..Default::default() });
        event.measurements.as_mut().expect("duration measurements").insert(M::UiAttachmentGeneration, Some(self.key.attachment_generation as f64));
        event
    }
    fn settle(&self, outcome: Outcome, task: Option<Instant>, await_start: Option<Instant>, reply: Option<Instant>) {
        if self.settled.swap(true, Ordering::SeqCst) { return; }
        let elapsed = match (await_start, reply) {
            (Some(start), Some(end)) => end.saturating_duration_since(start),
            _ => self.submitted.elapsed(),
        };
        let mut event = self.event(Op::UiInputAck, elapsed, outcome);
        let values = event.measurements.as_mut().expect("duration measurements");
        let ms = |at: Option<Instant>| at.map(|at| at.saturating_duration_since(self.submitted).as_secs_f64() * 1000.0);
        values.insert(M::UiSubmitToTaskMs, ms(task));
        values.insert(M::UiSubmitToAwaitMs, ms(await_start));
        values.insert(M::UiSubmitToReplyMs, ms(reply));
        safe_record_performance_metric(Some(&self.recorder), event);
    }

    /// A13: settle a ticket whose acknowledgement deadline elapsed (expiry
    /// sweep or attachment drop). The duration is the time to the deadline or
    /// drop event, capped at the deadline so a ticket that outlived it (for
    /// example a frozen UI loop) cannot report inflated wall clock.
    fn settle_timeout(&self, deadline: Duration) {
        if self.settled.swap(true, Ordering::SeqCst) { return; }
        let elapsed = self.submitted.elapsed().min(deadline);
        let mut event = self.event(Op::UiInputAck, elapsed, Outcome::Timeout);
        let values = event.measurements.as_mut().expect("duration measurements");
        values.insert(M::UiAckDeadlineMs, Some(deadline.as_secs_f64() * 1000.0));
        values.insert(M::UiSubmitToTaskMs, None);
        values.insert(M::UiSubmitToAwaitMs, None);
        values.insert(M::UiSubmitToReplyMs, None);
        safe_record_performance_metric(Some(&self.recorder), event);
    }
}

pub(super) async fn acknowledged<F>(ticket: SubmissionTicket, task: Instant, future: F) -> Result<(), String>
where F: std::future::Future<Output = Result<(), String>> {
    let await_start = Instant::now();
    let result = future.await;
    ticket.settle(if result.is_ok() { Outcome::Success } else { Outcome::Failure }, Some(task), Some(await_start), Some(Instant::now()));
    result
}

impl Drop for UiMetrics {
    fn drop(&mut self) { self.reset_attachment(); self.flush_render(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    #[derive(Default)]
    struct Recorder(Mutex<Vec<PerformanceMetricEvent>>);
    impl PerformanceMetricRecorder for Recorder {
        fn session_id(&self) -> &str { "test" }
        fn monotonic_now(&self) -> f64 { 0.0 }
        fn next_id(&self, _: pi_agent_core::performance_metrics::PerformanceMetricIdScope) -> String { "test".into() }
        fn record(&self, event: PerformanceMetricEvent) { self.0.lock().unwrap().push(event); }
        fn flush(&self) {}
        fn close(&self) {}
    }
    #[test]
    fn input_and_menu_only_settle_after_render_and_frames_are_aggregated() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder("test", recorder.clone());
        metrics.input(Instant::now());
        metrics.menu(Instant::now());
        assert!(recorder.0.lock().unwrap().is_empty());
        let phases = pi_tui::tui::RenderPhaseTimings { render_ms: 1.0, diff_ms: 2.0, write_ms: 0.5 };
        metrics.rendered(Duration::from_millis(4), &phases);
        metrics.rendered(Duration::from_millis(6), &phases);
        metrics.flush_render();
        let events = recorder.0.lock().unwrap();
        assert_eq!(events.iter().map(|e| e.operation).collect::<Vec<_>>(), [Op::UiInput, Op::UiMenuOpen, Op::UiRender]);
        let values = events[2].measurements.as_ref().unwrap();
        assert_eq!(values[&M::TotalMs], Some(10.0));
        assert_eq!(values[&M::MaxMs], Some(6.0));
        assert_eq!(values[&M::FrameCount], Some(2.0));
        // A15: per-phase sums aggregate across the window's frames.
        assert_eq!(values[&M::RenderMs], Some(2.0));
        assert_eq!(values[&M::DiffMs], Some(4.0));
        assert_eq!(values[&M::WriteMs], Some(1.0));
    }
    #[tokio::test]
    async fn acknowledgement_waits_for_the_matching_reply_and_omits_error_contents() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder("test", recorder.clone());
        let ticket = metrics.begin_submission();
        let id = ticket.key.submission_id.clone();
        let (reply, wait) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(acknowledged(ticket.clone(), Instant::now(), async move { wait.await.unwrap() }));
        tokio::task::yield_now().await;
        assert_eq!(recorder.0.lock().unwrap().len(), 1, "only the local start exists before reply");
        assert_eq!(metrics.pending_count(), 1);
        reply.send(Err("private prompt and credential".into())).unwrap();
        assert!(task.await.unwrap().is_err());
        assert!(metrics.reply(&ticket));
        assert!(!metrics.reply(&ticket));
        let events = recorder.0.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].operation, Op::UiInputAck);
        assert_eq!(events[1].outcome, Some(Outcome::Failure));
        assert_eq!(events[0].correlation.as_ref().unwrap().action_id.as_deref(), Some(id.as_str()));
        assert_eq!(events[1].correlation, events[0].correlation);
        let values = events[1].measurements.as_ref().unwrap();
        assert!(values[&M::UiSubmitToTaskMs].unwrap() <= values[&M::UiSubmitToAwaitMs].unwrap());
        assert!(values[&M::UiSubmitToAwaitMs].unwrap() <= values[&M::UiSubmitToReplyMs].unwrap());
        assert!(!serde_json::to_string(&*events).unwrap().contains("private"));
    }

    #[tokio::test]
    async fn ui_speed_reordered_replies_and_same_session_reset_settle_once() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder("test", recorder.clone());
        let first = metrics.begin_submission();
        let second = metrics.begin_submission();
        assert_ne!(first.key, second.key);
        metrics.receipt_rendered();
        metrics.receipt_rendered();
        let (reply, wait) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(acknowledged(first.clone(), Instant::now(), async move { wait.await.unwrap() }));
        acknowledged(second.clone(), Instant::now(), async { Ok(()) }).await.unwrap();
        assert!(metrics.reply(&second));
        assert_eq!(metrics.pending_count(), 1);
        metrics.reset_attachment();
        metrics.reset_attachment();
        let replacement = metrics.begin_submission();
        assert_eq!(replacement.key.session_id, first.key.session_id);
        assert_ne!(replacement.key.attachment_generation, first.key.attachment_generation);
        reply.send(Err("late private failure".into())).unwrap();
        assert!(task.await.unwrap().is_err());
        assert!(!metrics.reply(&first));
        assert_eq!(metrics.pending_count(), 1);
        let events = recorder.0.lock().unwrap();
        let terminals: Vec<_> = events.iter().filter(|event| event.operation == Op::UiInputAck && event.outcome != Some(Outcome::Started)).collect();
        assert_eq!(terminals.len(), 2);
        assert_eq!(terminals.iter().filter(|event| event.correlation.as_ref().unwrap().action_id == Some(first.key.submission_id.clone())).count(), 1);
        // A13: an ack that settles on an attachment reset without its reply is
        // classified as an ack timeout (duration capped at the deadline), not
        // `cancelled` with unbounded wall clock.
        assert_eq!(terminals.iter().find(|event| event.correlation.as_ref().unwrap().action_id == Some(first.key.submission_id.clone())).unwrap().outcome, Some(Outcome::Timeout));
        assert_eq!(events.iter().filter(|event| event.measurements.as_ref().is_some_and(|values| values.contains_key(&M::UiSubmitToReceiptRenderMs))).count(), 2);
        assert!(!serde_json::to_string(&*events).unwrap().contains("private"));
    }

    #[test]
    fn ui_speed_capacity_disposal_and_unknown_phase_values_are_bounded() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder("test", recorder.clone());
        let oldest = metrics.begin_submission();
        for _ in 0..MAX_PENDING_SUBMISSIONS { metrics.begin_submission(); }
        assert_eq!(metrics.pending_count(), MAX_PENDING_SUBMISSIONS);
        assert!(!metrics.reply(&oldest));
        metrics.reset_attachment();
        assert_eq!(metrics.pending_count(), 0);
        let last = metrics.begin_submission();
        drop(metrics);
        let events = recorder.0.lock().unwrap();
        let terminals: Vec<_> = events.iter().filter(|event| event.operation == Op::UiInputAck && event.outcome != Some(Outcome::Started)).collect();
        assert_eq!(terminals.len(), MAX_PENDING_SUBMISSIONS + 2);
        assert_eq!(terminals[0].outcome, Some(Outcome::Unavailable));
        assert_eq!(terminals.last().unwrap().correlation.as_ref().unwrap().action_id.as_deref(), Some(last.key.submission_id.as_str()));
        for event in terminals {
            let values = event.measurements.as_ref().unwrap();
            assert_eq!(values[&M::UiSubmitToTaskMs], None);
            assert_eq!(values[&M::UiSubmitToAwaitMs], None);
            assert_eq!(values[&M::UiSubmitToReplyMs], None);
        }
    }

    #[test]
    fn ui_speed_apply_and_fallback_counters_preserve_unknown_bytes_and_queue_age() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder("test", recorder.clone());
        metrics.applied(Duration::from_millis(2), Some(Duration::from_millis(5)));
        metrics.applied(Duration::from_millis(3), None);
        metrics.fallback_tick(); metrics.fallback_tick();
        metrics.fallback_tick_skipped();
        metrics.flush_render(); metrics.flush_render();
        let events = recorder.0.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].operation, Op::UiEventApply);
        let values = events[0].measurements.as_ref().unwrap();
        assert_eq!(values[&M::UiEventCount], Some(2.0));
        assert_eq!(values[&M::TotalMs], Some(5.0));
        assert_eq!(values[&M::MaxMs], Some(3.0));
        assert_eq!(values[&M::SerializedBytes], None);
        assert_eq!(values[&M::QueueMs], None);
        assert_eq!(events[1].operation, Op::UiTick);
        assert_eq!(events[1].measurements.as_ref().unwrap()[&M::UiTickFallbackCount], Some(2.0));
        // A12: gated-out fallback paints are counted additively.
        assert_eq!(events[1].measurements.as_ref().unwrap()[&M::UiTickFallbackSkipped], Some(1.0));
    }

    /// A13: a pending submission whose ack deadline elapsed settles as an
    /// explicit timeout with the deadline-capped duration, and a late reply is
    /// ignored exactly like a reply after a cancelled ticket.
    #[test]
    fn ack_deadline_expires_pending_submissions_as_timeout() {
        let recorder = Arc::new(Recorder::default());
        let mut metrics = UiMetrics::with_recorder_and_ack_deadline("test", recorder.clone(), Duration::from_millis(1));
        let ticket = metrics.begin_submission();
        std::thread::sleep(Duration::from_millis(5));
        metrics.expire_ack_deadlines();
        assert_eq!(metrics.pending_count(), 0, "the expired ticket leaves the pending map");
        let events = recorder.0.lock().unwrap();
        let terminal = events.iter().find(|event| event.operation == Op::UiInputAck && event.outcome == Some(Outcome::Timeout))
            .expect("an explicit timeout terminal event");
        let values = terminal.measurements.as_ref().unwrap();
        assert_eq!(values[&M::UiAckDeadlineMs], Some(1.0));
        assert_eq!(values[&M::TotalMs], Some(1.0), "duration is capped at the deadline, not the wall clock");
        drop(events);
        assert!(!metrics.reply(&ticket), "a late reply after the timeout is ignored");
    }

    /// A13: dropping the metrics (session end) with an unanswered submission
    /// reports a timeout bounded by the deadline instead of `cancelled` with
    /// full wall clock (the 392s settle-on-drop artifact).
    #[test]
    fn drop_settled_ack_is_timeout_bounded_by_deadline() {
        let recorder = Arc::new(Recorder::default());
        let deadline = Duration::from_millis(50);
        let mut metrics = UiMetrics::with_recorder_and_ack_deadline("test", recorder.clone(), deadline);
        let _ticket = metrics.begin_submission();
        drop(metrics);
        let events = recorder.0.lock().unwrap();
        let terminal = events.iter().find(|event| event.operation == Op::UiInputAck && event.outcome != Some(Outcome::Started))
            .expect("a terminal event for the dropped ticket");
        assert_eq!(terminal.outcome, Some(Outcome::Timeout));
        let values = terminal.measurements.as_ref().unwrap();
        let total = values[&M::TotalMs].expect("capped duration");
        assert!(total <= deadline.as_secs_f64() * 1000.0, "drop-settled duration must not exceed the deadline: {total}ms");
    }
}
