//! Bounded asynchronous comparison scheduler (DESIGN.md section 4).
//!
//! Fixed small concurrency, per-request deadline, payload and question caps,
//! min-interval and budget caps, a consecutive-failure circuit breaker,
//! finite retries honoring Retry-After, queue-full drops with metrics that
//! never block the caller, and cancellation cleanup on session disposal.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::sync::mpsc;

use crate::client::bundle_with_questions;
use crate::types::{DecisionOutcome, SystemOne, SystemOneRequest};

/// Scheduler limits; small fixed values keep comparison overhead bounded.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub queue_capacity: usize,
    pub concurrency: usize,
    pub request_deadline: Duration,
    pub max_payload_bytes: usize,
    pub max_questions_per_request: usize,
    pub min_interval: Duration,
    pub max_requests_per_minute: u32,
    pub max_consecutive_failures: u32,
    pub breaker_cooldown: Duration,
    // NOTE: transport retries are client-owned (`JevLimits::max_retries`);
    // the scheduler deliberately carries no retry knobs (one logical attempt
    // per job under its own deadline).
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 128,
            concurrency: 2,
            request_deadline: Duration::from_secs(10),
            max_payload_bytes: 32 * 1024,
            max_questions_per_request: 16,
            min_interval: Duration::from_millis(100),
            max_requests_per_minute: 60,
            max_consecutive_failures: 5,
            breaker_cooldown: Duration::from_secs(30),
        }
    }
}

/// Per-question metadata captured at the boundary.
#[derive(Debug, Clone)]
pub struct QuestionMeta {
    pub question_id: String,
    pub category: String,
}

/// Everything the result sink needs, captured synchronously at the boundary
/// BEFORE any shadow result returns (baselines included).
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub session_id: String,
    pub turn: u64,
    pub stage: String,
    pub state_fingerprint: String,
    pub state_schema_version: String,
    pub prompt_version: String,
    pub mode: String,
    /// Requested Jev model captured with this request's authoritative policy
    /// generation. The scheduler dispatch gate rechecks both before send and
    /// after completion, so queued Compare work cannot cross a model write.
    pub requested_model: String,
    pub policy_generation: String,
    pub questions: Vec<QuestionMeta>,
    /// question_id -> actual observed choice at the boundary (None = unknown).
    pub baselines: BTreeMap<String, Option<String>>,
    /// RFC 3339 boundary time (request start).
    pub request_start_ts: String,
}

#[derive(Debug)]
pub enum JobResult {
    Completed {
        ctx: RequestContext,
        /// Validated client outcome; `applied` is always false.
        outcome: DecisionOutcome,
        duration_ms: u64,
    },
    Failed {
        ctx: RequestContext,
        /// Stable reason code, prefixed `request_failed:`; never error text.
        reason: String,
        /// Logical scheduler attempts (one per job); transport-level retry
        /// counts are client-owned and surface inside the outcome instead.
        attempts: u32,
    },
    Dropped {
        ctx: RequestContext,
        reason: String,
    },
}

pub type ResultSink = Arc<dyn Fn(JobResult) + Send + Sync>;

#[derive(Default)]
pub struct SchedulerMetrics {
    pub enqueued: AtomicU64,
    pub dropped_queue_full: AtomicU64,
    pub dropped_circuit_open: AtomicU64,
    pub dropped_budget: AtomicU64,
    pub dropped_cancelled: AtomicU64,
    pub dropped_deadline: AtomicU64,
    pub dropped_limit: AtomicU64,
    pub dropped_oversize: AtomicU64,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
}

impl SchedulerMetrics {
    pub fn report(&self) -> serde_json::Value {
        json!({
            "enqueued": self.enqueued.load(Ordering::SeqCst),
            "dropped_queue_full": self.dropped_queue_full.load(Ordering::SeqCst),
            "dropped_circuit_open": self.dropped_circuit_open.load(Ordering::SeqCst),
            "dropped_budget": self.dropped_budget.load(Ordering::SeqCst),
            "dropped_cancelled": self.dropped_cancelled.load(Ordering::SeqCst),
            "dropped_deadline": self.dropped_deadline.load(Ordering::SeqCst),
            "dropped_limit": self.dropped_limit.load(Ordering::SeqCst),
            "dropped_oversize": self.dropped_oversize.load(Ordering::SeqCst),
            "completed": self.completed.load(Ordering::SeqCst),
            "failed": self.failed.load(Ordering::SeqCst),
        })
    }
}

struct Breaker {
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

impl Breaker {
    fn new() -> Self {
        Self {
            consecutive_failures: 0,
            open_until: None,
        }
    }
}

struct Budget {
    window_start: Instant,
    used: u32,
}

struct Job {
    request: SystemOneRequest,
    ctx: RequestContext,
    deadline: Instant,
    /// Enqueue wall-clock marker; jobs enqueued before a session's cancel
    /// time are dropped, jobs enqueued after it run (Compare can return).
    enqueued_at: Instant,
}

struct Shared {
    config: SchedulerConfig,
    system_one: Arc<dyn SystemOne>,
    sink: ResultSink,
    queue_tx: Mutex<Option<mpsc::Sender<Job>>>,
    metrics: SchedulerMetrics,
    breaker: Mutex<Breaker>,
    budget: Mutex<Budget>,
    last_dispatch: Mutex<Option<Instant>>,
    /// Session id -> cancellation instant. Only work enqueued BEFORE the
    /// instant is dropped; later requests flow, so Compare -> Off -> Compare
    /// does not black-hole a session. Bounded: oldest entries are evicted.
    cancelled_sessions: Mutex<HashMap<String, Instant>>,
    in_flight: Mutex<HashMap<(String, String), CancellationToken>>,
    closed: AtomicBool,
    dispatch_gate: Arc<dyn Fn(&RequestContext) -> bool + Send + Sync>,
    status: Arc<Mutex<HashMap<String, SessionStatus>>>,
}

#[derive(Clone, Default, serde::Serialize)]
struct SessionStatus {
    success_count: u64,
    failure_count: u64,
    dropped_comparisons: u64,
    pending: usize,
    last_success_ms: Option<u64>,
    last_latency_ms: Option<u64>,
    response_model: Option<String>,
    fallback_reason: Option<String>,
}

/// Cooperative cancellation token (no tokio-util dependency).
#[derive(Default, Debug, Clone)]
pub struct CancellationToken {
    inner: Arc<CancellationState>,
}

#[derive(Default, Debug)]
struct CancellationState {
    notify: tokio::sync::Notify,
    flag: std::sync::atomic::AtomicBool,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.inner.flag.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.flag.load(Ordering::SeqCst)
    }

    /// Resolves when cancelled.
    pub async fn cancelled(&self) {
        let notified = self.inner.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

/// Bounded comparison scheduler. One instance per observer.
pub struct JevScheduler {
    shared: Arc<Shared>,
}

impl JevScheduler {
    /// Build and spawn the worker tasks. Must be called inside a tokio
    /// runtime; the workers are fire-and-forget tasks.
    pub fn new(
        config: SchedulerConfig,
        system_one: Arc<dyn SystemOne>,
        sink: ResultSink,
    ) -> Self {
        Self::new_with_gate(config, system_one, sink, Arc::new(|_| true))
    }

    pub fn new_with_gate(
        config: SchedulerConfig,
        system_one: Arc<dyn SystemOne>,
        sink: ResultSink,
        dispatch_gate: Arc<dyn Fn(&RequestContext) -> bool + Send + Sync>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Job>(config.queue_capacity);
        let status: Arc<Mutex<HashMap<String, SessionStatus>>> = Arc::new(Mutex::new(HashMap::new()));
        let terminal_status = Arc::clone(&status);
        let sink: ResultSink = Arc::new(move |result| {
            let ctx = match &result {
                JobResult::Completed { ctx, .. } | JobResult::Failed { ctx, .. }
                    | JobResult::Dropped { ctx, .. } => ctx,
            };
            {
                let mut all = terminal_status.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(status) = all.get_mut(&ctx.session_id) {
                    status.pending = status.pending.saturating_sub(1);
                    match &result {
                        JobResult::Completed { outcome, duration_ms, .. } => {
                            status.success_count += 1;
                            status.last_success_ms = Some(crate::client::utc_now_ms());
                            status.last_latency_ms = Some(*duration_ms);
                            status.response_model = outcome.response_model.as_ref().map(|model| {
                                crate::error::sanitize_detail(model).chars()
                                    .filter(|ch| !ch.is_control()).take(120).collect()
                            });
                            status.fallback_reason = None;
                        }
                        JobResult::Failed { reason, .. } => {
                            status.failure_count += 1;
                            status.fallback_reason = Some(reason.clone());
                        }
                        JobResult::Dropped { reason, .. } => {
                            status.dropped_comparisons += 1;
                            status.fallback_reason = Some(reason.clone());
                        }
                    }
                }
            }
            sink(result);
        });
        let shared = Arc::new(Shared {
            config: config.clone(),
            system_one,
            sink,
            queue_tx: Mutex::new(Some(tx)),
            metrics: SchedulerMetrics::default(),
            breaker: Mutex::new(Breaker::new()),
            budget: Mutex::new(Budget {
                window_start: Instant::now(),
                used: 0,
            }),
            last_dispatch: Mutex::new(None),
            cancelled_sessions: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            dispatch_gate,
            status,
        });
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        for _worker_index in 0..config.concurrency.max(1) {
            let shared = Arc::clone(&shared);
            let rx = Arc::clone(&rx);
            tokio::spawn(async move {
                loop {
                    let job = {
                        let mut guard = rx.lock().await;
                        guard.recv().await
                    };
                    let Some(job) = job else { break };
                    if shared.closed.load(Ordering::SeqCst) {
                        (shared.sink)(JobResult::Dropped {
                            ctx: job.ctx,
                            reason: "cancelled".to_string(),
                        });
                        continue;
                    }
                    run_job(&shared, job).await;
                }
            });
        }
        Self { shared }
    }

    /// Enqueue one bundle. Never blocks: a full queue drops the request with
    /// a metric. Returns false when the request was dropped.
    pub fn enqueue(&self, mut request: SystemOneRequest, mut ctx: RequestContext) -> bool {
        if self.shared.closed.load(Ordering::SeqCst) {
            (self.shared.sink)(JobResult::Dropped { ctx, reason: "cancelled".into() });
            return false;
        }
        self.shared.metrics.enqueued.fetch_add(1, Ordering::SeqCst);
        {
            let mut status = self.shared.status.lock().unwrap_or_else(|p| p.into_inner());
            if status.len() >= 64 && !status.contains_key(&ctx.session_id) {
                if let Some(id) = status.iter().find(|(_, status)| status.pending == 0).map(|(id, _)| id.clone()) {
                    status.remove(&id);
                }
            }
            if status.len() < 64 || status.contains_key(&ctx.session_id) {
                status.entry(ctx.session_id.clone()).or_default().pending += 1;
            }
        }
        // Payload cap.
        let payload_bytes = serde_json::to_vec(&request.state).unwrap_or_default().len();
        if payload_bytes > self.shared.config.max_payload_bytes {
            self.shared.metrics.dropped_oversize.fetch_add(1, Ordering::SeqCst);
            (self.shared.sink)(JobResult::Dropped {
                ctx,
                reason: "state_too_large".to_string(),
            });
            return false;
        }
        // Question cap: bundle at most max_questions_per_request; the caller
        // records the remainder as skipped (never fabricated).
        let max = self.shared.config.max_questions_per_request;
        if ctx.questions.len() > max {
            // The observer normally records overflow itself (question_limit
            // skips); this is the backstop, counted for honesty.
            self.shared
                .metrics
                .dropped_limit
                .fetch_add((ctx.questions.len() - max) as u64, Ordering::SeqCst);
            ctx.questions.truncate(max);
            request.questions.retain(|id, _| ctx.questions.iter().any(|meta| &meta.question_id == id));
        }
        let deadline = Instant::now() + self.shared.config.request_deadline;
        let enqueued_at = Instant::now();
        let tx_guard = self
            .shared
            .queue_tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(tx) = tx_guard.as_ref() else {
            (self.shared.sink)(JobResult::Dropped { ctx, reason: "cancelled".into() });
            return false;
        };
        match tx.try_send(Job {
            request,
            ctx,
            deadline,
            enqueued_at,
        }) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(job)) => {
                self.shared.metrics.dropped_queue_full.fetch_add(1, Ordering::SeqCst);
                (self.shared.sink)(JobResult::Dropped {
                    ctx: job.ctx,
                    reason: "queue_full".to_string(),
                });
                false
            }
            Err(mpsc::error::TrySendError::Closed(job)) => {
                (self.shared.sink)(JobResult::Dropped { ctx: job.ctx, reason: "cancelled".into() });
                false
            },
        }
    }

    /// Drop every queued and in-flight request for a session (disposal).
    pub fn cancel_session(&self, session_id: &str) {
        {
            let mut cancelled = self
                .shared
                .cancelled_sessions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Bounded bookkeeping: cancelled ids map to their cancellation
            // instant. Oldest entries are evicted past the cap (one late
            // re-dispatch for an ancient session at worst).
            if cancelled.len() >= 1024 {
                let oldest = cancelled
                    .iter()
                    .min_by_key(|(_, at)| *at)
                    .map(|(id, _)| id.clone());
                if let Some(oldest) = oldest {
                    cancelled.remove(&oldest);
                }
            }
            cancelled.insert(session_id.to_string(), Instant::now());
        }
        let mut in_flight = self
            .shared
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let keys: Vec<_> = in_flight
            .keys()
            .filter(|(session, _)| session == session_id)
            .cloned()
            .collect();
        for key in keys {
            if let Some(token) = in_flight.remove(&key) {
                token.cancel();
            }
        }
    }

    /// Stop accepting work, cancel everything in flight, and close the queue.
    pub fn shutdown(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        self.shared
            .queue_tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let mut in_flight = self
            .shared
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (_, token) in in_flight.drain() {
            token.cancel();
        }
    }

    pub fn metrics(&self) -> serde_json::Value {
        self.shared.metrics.report()
    }

    /// Bounded per-session metadata only; no request state or credentials.
    pub fn session_status(&self, session_id: &str) -> Option<serde_json::Value> {
        let status = self.shared.status.lock().unwrap_or_else(|p| p.into_inner()).get(session_id).cloned()?;
        let in_flight = self.shared.in_flight.lock().unwrap_or_else(|p| p.into_inner())
            .keys().filter(|(session, _)| session == session_id).count();
        let mut value = serde_json::to_value(&status).ok()?;
        value["in_flight"] = json!(in_flight);
        value["queue_depth"] = json!(status.pending.saturating_sub(in_flight));
        value["queue_capacity"] = json!(self.shared.config.queue_capacity);
        value["checking"] = json!(in_flight > 0);
        value["applied"] = json!(false);
        value["llm_calls_actually_avoided"] = json!(0);
        Some(value)
    }
}

impl Drop for JevScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn run_job(shared: &Arc<Shared>, job: Job) {
    let request_id = job.ctx.request_id.clone();
    let session_id = job.ctx.session_id.clone();
    // Cancelled while queued? Only work enqueued BEFORE the cancel instant is
    // dropped; requests enqueued after the cancel flow normally, so a
    // Compare -> Off -> Compare transition never black-holes the session.
    {
        let cancelled_at = shared
            .cancelled_sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&session_id)
            .copied();
        if cancelled_at.is_some_and(|at| job.enqueued_at < at) {
            shared.metrics.dropped_cancelled.fetch_add(1, Ordering::SeqCst);
            (shared.sink)(JobResult::Dropped {
                ctx: job.ctx,
                reason: "cancelled".to_string(),
            });
            return;
        }
    }
    // Circuit breaker.
    {
        let breaker = shared.breaker.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(open_until) = breaker.open_until {
            if Instant::now() < open_until {
                shared.metrics.dropped_circuit_open.fetch_add(1, Ordering::SeqCst);
                drop(breaker);
                (shared.sink)(JobResult::Dropped {
                    ctx: job.ctx,
                    reason: "circuit_open".to_string(),
                });
                return;
            }
        }
    }
    // Budget (fixed one-minute window).
    {
        let mut budget = shared.budget.lock().unwrap_or_else(|p| p.into_inner());
        if budget.window_start.elapsed() >= Duration::from_secs(60) {
            budget.window_start = Instant::now();
            budget.used = 0;
        }
        if budget.used >= shared.config.max_requests_per_minute {
            drop(budget);
            shared.metrics.dropped_budget.fetch_add(1, Ordering::SeqCst);
            (shared.sink)(JobResult::Dropped {
                ctx: job.ctx,
                reason: "budget_exceeded".to_string(),
            });
            return;
        }
        budget.used += 1;
    }
    // Min interval between dispatches.
    let wait = {
        let last = shared
            .last_dispatch
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        last.map(|at| {
            shared
                .config
                .min_interval
                .saturating_sub(at.elapsed())
        })
        .unwrap_or(Duration::ZERO)
    };

    let token = CancellationToken::new();
    shared
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert((session_id.clone(), request_id.clone()), token.clone());

    // Register first, then recheck: cancellation before registration is in
    // the marker map; cancellation after registration reaches this token.
    let cancelled_before_start = shared
        .cancelled_sessions.lock().unwrap_or_else(|p| p.into_inner())
        .get(&session_id).is_some_and(|at| job.enqueued_at < *at);
    if cancelled_before_start || shared.closed.load(Ordering::SeqCst)
        || !(shared.dispatch_gate)(&job.ctx)
    {
        token.cancel();
    }

    if !wait.is_zero() {
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = token.cancelled() => {}
        }
    }
    *shared
        .last_dispatch
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(Instant::now());

    let total_deadline = job.deadline;
    let started = Instant::now();

    // One logical scheduler attempt per job: the SystemOne client owns
    // bounded transport retries (including Retry-After honoring) inside
    // `decide`, so the scheduler never duplicates them. The scheduler owns
    // the job deadline, the breaker and all drops.
    enum JobVerdict {
        Completed(DecisionOutcome, u64),
        Cancelled,
        Deadline,
    }
    let verdict = {
        let remaining = total_deadline.saturating_duration_since(Instant::now());
        if token.is_cancelled() || shared.closed.load(Ordering::SeqCst)
            || !(shared.dispatch_gate)(&job.ctx)
        {
            JobVerdict::Cancelled
        } else if remaining.is_zero() {
            shared.metrics.dropped_deadline.fetch_add(1, Ordering::SeqCst);
            JobVerdict::Deadline
        } else {
            let system_one = Arc::clone(&shared.system_one);
            let bundle = bundle_with_questions(
                job.ctx.session_id.clone(),
                job.ctx.turn,
                job.ctx.stage.clone(),
                job.request.state.clone(),
                job.request.model.clone(),
                job.request.questions.clone(),
            );
            let attempt_future = async move { system_one.decide(bundle).await };
            tokio::select! {
                _ = token.cancelled() => JobVerdict::Cancelled,
                waited = tokio::time::timeout(remaining, attempt_future) => match waited {
                    Ok(outcome) if !token.is_cancelled()
                        && !shared.closed.load(Ordering::SeqCst)
                        && (shared.dispatch_gate)(&job.ctx) =>
                        JobVerdict::Completed(outcome, started.elapsed().as_millis() as u64),
                    Ok(_) => JobVerdict::Cancelled,
                    Err(_elapsed) => {
                        shared.metrics.dropped_deadline.fetch_add(1, Ordering::SeqCst);
                        JobVerdict::Deadline
                    }
                },
            }
        }
    };

    shared
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(&(session_id.clone(), request_id.clone()));

    match verdict {
        JobVerdict::Completed(outcome, duration_ms) => {
            // A fully skipped outcome is a failure for breaker purposes; any
            // accepted record is a success. Reason is a stable code only.
            let failure_reason = if outcome.records.is_empty() {
                Some(
                    outcome
                        .skips
                        .first()
                        .map(|(_, kind)| kind.to_string())
                        .unwrap_or_else(|| "no_answers".to_string()),
                )
            } else {
                None
            };
            match failure_reason {
                None => {
                    shared.metrics.completed.fetch_add(1, Ordering::SeqCst);
                    {
                        let mut breaker = shared.breaker.lock().unwrap_or_else(|p| p.into_inner());
                        breaker.consecutive_failures = 0;
                        breaker.open_until = None;
                    }
                    (shared.sink)(JobResult::Completed {
                        ctx: job.ctx,
                        outcome,
                        duration_ms,
                    });
                }
                Some(kind) => {
                    shared.metrics.failed.fetch_add(1, Ordering::SeqCst);
                    bump_breaker(shared);
                    (shared.sink)(JobResult::Failed {
                        ctx: job.ctx,
                        reason: format!("request_failed:{kind}"),
                        attempts: 1,
                    });
                }
            }
        }
        JobVerdict::Cancelled => {
            shared.metrics.dropped_cancelled.fetch_add(1, Ordering::SeqCst);
            (shared.sink)(JobResult::Dropped {
                ctx: job.ctx,
                reason: "cancelled".to_string(),
            });
        }
        JobVerdict::Deadline => {
            shared.metrics.failed.fetch_add(1, Ordering::SeqCst);
            bump_breaker(shared);
            (shared.sink)(JobResult::Failed {
                ctx: job.ctx,
                reason: "request_failed:timeout".to_string(),
                attempts: 1,
            });
        }
    }
}

/// Count one job failure for the circuit breaker; open it after the
/// configured consecutive-failure threshold.
fn bump_breaker(shared: &Arc<Shared>) {
    let mut breaker = shared.breaker.lock().unwrap_or_else(|p| p.into_inner());
    breaker.consecutive_failures += 1;
    if breaker.consecutive_failures >= shared.config.max_consecutive_failures {
        breaker.open_until = Some(Instant::now() + shared.config.breaker_cooldown);
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[derive(Default)]
    struct Probe {
        calls: AtomicU64,
    }

    impl SystemOne for Probe {
        fn mode(&self) -> crate::config::JevMode { crate::config::JevMode::Compare }
        fn decide(&self, _bundle: crate::types::DecisionBundle) -> crate::types::BoxFuture<DecisionOutcome> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { DecisionOutcome::skipped_all("mock") })
        }
    }

    fn input(id: &str) -> (SystemOneRequest, RequestContext) {
        (SystemOneRequest { state: json!({}), model: "mock".into(), questions: BTreeMap::new() },
         RequestContext {
            request_id: id.into(), session_id: "session".into(), turn: 0,
            stage: "turn_start".into(), state_fingerprint: String::new(),
            state_schema_version: "test".into(), prompt_version: "test".into(),
            mode: "compare".into(), requested_model: "mock".into(),
            policy_generation: "test".into(), questions: Vec::new(),
            baselines: BTreeMap::new(), request_start_ts: String::new(),
         })
    }

    async fn until(mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !condition() { tokio::time::sleep(Duration::from_millis(2)).await; }
        }).await.expect("bounded scheduler condition");
    }

    #[tokio::test]
    async fn cloned_cancellation_is_persistent_before_and_after_waiters() {
        let token = CancellationToken::new();
        token.clone().cancel();
        assert!(token.is_cancelled());
        tokio::time::timeout(Duration::from_millis(100), token.cancelled()).await.unwrap();
        tokio::time::timeout(Duration::from_millis(100), token.clone().cancelled()).await.unwrap();
    }

    #[tokio::test]
    async fn cancel_during_rate_wait_does_not_dispatch_and_new_work_can_resume() {
        let probe = Arc::new(Probe::default());
        let scheduler = JevScheduler::new(SchedulerConfig {
            concurrency: 1, min_interval: Duration::from_millis(150), ..Default::default()
        }, probe.clone(), Arc::new(|_| {}));
        let (request, ctx) = input("first");
        scheduler.enqueue(request, ctx);
        until(|| probe.calls.load(Ordering::SeqCst) == 1).await;
        let (request, ctx) = input("cancelled");
        scheduler.enqueue(request, ctx);
        until(|| scheduler.shared.in_flight.lock().unwrap().contains_key(&("session".into(), "cancelled".into()))).await;
        scheduler.cancel_session("session");
        until(|| scheduler.shared.metrics.dropped_cancelled.load(Ordering::SeqCst) == 1).await;
        assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
        let (request, ctx) = input("resumed");
        scheduler.enqueue(request, ctx);
        until(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    }

    #[tokio::test]
    async fn current_mode_gate_refuses_already_queued_work() {
        let probe = Arc::new(Probe::default());
        let enabled = Arc::new(AtomicBool::new(true));
        let gate_enabled = Arc::clone(&enabled);
        let scheduler = JevScheduler::new_with_gate(SchedulerConfig {
            concurrency: 1, min_interval: Duration::from_millis(150), ..Default::default()
        }, probe.clone(), Arc::new(|_| {}), Arc::new(move |_| gate_enabled.load(Ordering::SeqCst)));
        let (request, ctx) = input("first");
        scheduler.enqueue(request, ctx);
        until(|| probe.calls.load(Ordering::SeqCst) == 1).await;
        let (request, ctx) = input("gated");
        scheduler.enqueue(request, ctx);
        until(|| scheduler.shared.in_flight.lock().unwrap().contains_key(&("session".into(), "gated".into()))).await;
        enabled.store(false, Ordering::SeqCst);
        until(|| scheduler.shared.metrics.dropped_cancelled.load(Ordering::SeqCst) == 1).await;
        assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropping_scheduler_releases_workers_and_client() {
        let probe = Arc::new(Probe::default());
        let client_weak = Arc::downgrade(&probe);
        let scheduler = JevScheduler::new(SchedulerConfig::default(), probe, Arc::new(|_| {}));
        let shared_weak = Arc::downgrade(&scheduler.shared);
        tokio::task::yield_now().await;
        drop(scheduler);
        until(|| shared_weak.upgrade().is_none()).await;
        assert!(client_weak.upgrade().is_none());
    }

    struct SlowProbe { calls: Arc<AtomicU64> }

    impl SystemOne for SlowProbe {
        fn mode(&self) -> crate::config::JevMode { crate::config::JevMode::Compare }
        fn decide(&self, _bundle: crate::types::DecisionBundle) -> crate::types::BoxFuture<DecisionOutcome> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(80)).await;
                DecisionOutcome::skipped_all("mock")
            })
        }
    }

    #[tokio::test]
    async fn changed_mode_or_credential_discards_inflight_completion() {
        let calls = Arc::new(AtomicU64::new(0));
        let enabled = Arc::new(AtomicBool::new(true));
        let gate = Arc::clone(&enabled);
        let scheduler = JevScheduler::new_with_gate(SchedulerConfig::default(),
            Arc::new(SlowProbe { calls: Arc::clone(&calls) }), Arc::new(|_| {}),
            Arc::new(move |_| gate.load(Ordering::SeqCst)));
        let (request, ctx) = input("inflight");
        scheduler.enqueue(request, ctx);
        until(|| calls.load(Ordering::SeqCst) == 1).await;
        enabled.store(false, Ordering::SeqCst);
        until(|| scheduler.shared.metrics.dropped_cancelled.load(Ordering::SeqCst) == 1).await;
        assert_eq!(scheduler.shared.metrics.failed.load(Ordering::SeqCst), 0);
        assert_eq!(scheduler.shared.metrics.completed.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn disposal_emits_one_terminal_for_each_queued_or_inflight_request() {
        let calls = Arc::new(AtomicU64::new(0));
        let results = Arc::new(Mutex::new(Vec::new()));
        let received = Arc::clone(&results);
        let scheduler = JevScheduler::new(SchedulerConfig { concurrency: 1, ..Default::default() },
            Arc::new(SlowProbe { calls: Arc::clone(&calls) }),
            Arc::new(move |result| {
                let id = match result {
                    JobResult::Completed { ctx, .. } | JobResult::Failed { ctx, .. }
                        | JobResult::Dropped { ctx, .. } => ctx.request_id,
                };
                received.lock().unwrap().push(id);
            }));
        for id in ["inflight", "queued-1", "queued-2"] {
            let (request, ctx) = input(id);
            scheduler.enqueue(request, ctx);
        }
        until(|| calls.load(Ordering::SeqCst) == 1).await;
        drop(scheduler);
        until(|| results.lock().unwrap().len() == 3).await;
        let mut ids = results.lock().unwrap().clone();
        ids.sort(); ids.dedup();
        assert_eq!(ids.len(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
