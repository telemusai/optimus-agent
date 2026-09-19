//! Transport-agnostic JevObserver core (DESIGN.md section 4).
//!
//! Consumes event payloads as `serde_json::Value` DTOs (the adapter in
//! pi-coding-agent builds bounded values). In Compare: capture baselines
//! synchronously at the boundary, build one bundled request per snapshot,
//! enqueue asynchronously, and write correlated records when results arrive.
//! Nothing a Compare answer says can re-enter the agent loop.
//!
//! In Active, `decide_active` is the single decision-returning path. It makes
//! one bounded call at the boundary, runs the answer through the acceptance
//! policy in `crate::active`, and returns only the decisions the policy
//! accepted. The caller owns the effect: this crate never mutates host state
//! and never applies anything itself. Every outcome, accepted or refused, is
//! recorded with the single reason that stopped it.
//!
//! In Off: a single cheap mode check and nothing else — no scheduling, no
//! client work, no records.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use uuid::Uuid;

use crate::config::JevMode;
use crate::correlate::Correlator;
use crate::evaluators::{for_boundary, EvaluatorOutput, PreparedQuestion};
use crate::scheduler::{JevScheduler, JobResult, QuestionMeta, RequestContext, SchedulerConfig};
use crate::snapshot::{
    bound_json, fingerprint_of, SnapshotStage, StateSnapshot, STATE_SCHEMA_VERSION,
    MAX_STATE_BYTES,
};
use crate::types::DecisionCategory;

/// Default SystemOne model id from the official docs.
pub const SYSTEM_ONE_MODEL: &str = "jev-latest";

pub const PROMPT_VERSION: &str = "jev-compare-prompts/1";

/// Operational bounds for the Active decision path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveSettings {
    /// Hard deadline for one Active decision call. The caller waits at most
    /// this long before behaving as if Jev were absent.
    pub deadline: std::time::Duration,
    /// Consecutive failed or timed-out calls that open the breaker.
    pub max_consecutive_failures: u32,
    /// How long the breaker stays open before another attempt is allowed.
    pub breaker_cooldown: std::time::Duration,
}

impl Default for ActiveSettings {
    fn default() -> Self {
        Self {
            deadline: std::time::Duration::from_millis(2500),
            max_consecutive_failures: 3,
            breaker_cooldown: std::time::Duration::from_secs(30),
        }
    }
}

/// Outcome of one Active decision boundary.
#[derive(Debug, Clone)]
pub struct ActiveDecideOutcome {
    pub session_id: String,
    pub turn: u64,
    pub stage: String,
    pub request_id: Option<String>,
    pub response_model: Option<String>,
    pub duration_ms: Option<u64>,
    pub state_fingerprint: String,
    /// Decisions the policy accepted, in the order the answers arrived.
    pub decisions: Vec<crate::active::ActiveDecision>,
    /// Answers that were refused, each with the one reason that stopped it.
    pub refusals: Vec<ActiveRefusal>,
    /// Set when no answer was obtained at all (transport, timeout, breaker,
    /// wrong mode). The caller must then behave as if Jev were absent.
    pub unavailable: Option<crate::active::FallbackReason>,
}

/// One refused answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveRefusal {
    pub category: DecisionCategory,
    pub reason: crate::active::FallbackReason,
}

impl ActiveRefusal {
    pub fn new(category: DecisionCategory, reason: crate::active::FallbackReason) -> Self {
        Self { category, reason }
    }
}

/// Per-session counters for the Active path. These describe boundaries, not
/// answers: `applied` counts boundaries where a request field actually changed.
#[derive(Debug, Default)]
struct ActiveCounters {
    applied: u64,
    accepted_no_effect: u64,
    refused: u64,
    unavailable: u64,
    last_reason: Option<String>,
    last_category: Option<String>,
}

/// Breaker state for the Active path.
#[derive(Debug, Default)]
struct ActiveBreaker {
    consecutive_failures: u32,
    open_until: Option<std::time::Instant>,
}

/// Result of preparing one boundary's request.
enum BundlePreparation {
    /// Nothing askable: no eligible question, or a missing session id.
    Nothing,
    /// The bounded state exceeded its cap; every eligible category is recorded
    /// as skipped.
    StateTooLarge,
    Ready(Box<PreparedBundle>),
}

/// A prepared request plus the context needed to record it.
struct PreparedBundle {
    request: crate::types::SystemOneRequest,
    ctx: RequestContext,
}

/// Observer configuration.
#[derive(Clone)]
pub struct JevObserverConfig {
    /// Cheap effective-mode check invoked once per observed event.
    pub mode_gate: Arc<dyn Fn(Option<&str>) -> JevMode + Send + Sync>,
    /// Categories to observe; empty set means all eligible by default.
    pub enabled_categories: HashSet<String>,
    pub scheduler: SchedulerConfig,
    /// Hypothetical acceptance threshold for choice/score confidence.
    pub min_confidence: f64,
    /// UI metadata notification only; never carries a recommendation.
    pub on_terminal: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Bounds for the Active decision path. Unused in Compare and Off.
    pub active: ActiveSettings,
}

impl Default for JevObserverConfig {
    fn default() -> Self {
        Self {
            mode_gate: Arc::new(|_| JevMode::Off),
            enabled_categories: HashSet::new(),
            scheduler: SchedulerConfig::default(),
            min_confidence: 0.7,
            on_terminal: None,
            active: ActiveSettings::default(),
        }
    }
}

impl std::fmt::Debug for JevObserverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevObserverConfig")
            .field("enabled_categories", &self.enabled_categories)
            .field("scheduler", &self.scheduler.request_deadline)
            .finish()
    }
}

/// Jev comparison observer. Build once per wiring; consume events through
/// `observe`. No method returns Jev output; nothing re-enters the loop.
pub struct JevObserver {
    config: JevObserverConfig,
    correlator: Arc<Correlator>,
    scheduler: JevScheduler,
    /// Retained for the Active path: one bounded, synchronous call per
    /// decision boundary. Unused by Compare, which goes through the queue.
    system_one: Arc<dyn crate::types::SystemOne>,
    active_breaker: Mutex<ActiveBreaker>,
    active_counters: Mutex<std::collections::HashMap<String, ActiveCounters>>,
    sessions: Mutex<std::collections::HashMap<String, u32>>,
    skipped_categories: Mutex<std::collections::HashMap<String, BTreeMap<String, String>>>,
}

impl JevObserver {
    pub fn new(
        config: JevObserverConfig,
        system_one: Arc<dyn crate::types::SystemOne>,
        records_path: std::path::PathBuf,
    ) -> Arc<Self> {
        let correlator = Arc::new(Correlator::new(records_path, config.min_confidence));
        let correlator_for_sink = Arc::clone(&correlator);
        let on_terminal = config.on_terminal.clone();
        let sink: crate::scheduler::ResultSink = Arc::new(move |result: JobResult| {
            correlator_for_sink.handle_result(&result);
            if let Some(notify) = &on_terminal {
                let ctx = match &result {
                    JobResult::Completed { ctx, .. } | JobResult::Failed { ctx, .. }
                        | JobResult::Dropped { ctx, .. } => ctx,
                };
                notify(&ctx.session_id);
            }
        });
        let mode_gate = Arc::clone(&config.mode_gate);
        let scheduler = JevScheduler::new_with_gate(
            config.scheduler.clone(),
            Arc::clone(&system_one),
            sink,
            Arc::new(move |session_id| mode_gate(Some(session_id)) == JevMode::Compare),
        );
        Arc::new(Self {
            config,
            correlator,
            scheduler,
            system_one,
            active_breaker: Mutex::new(ActiveBreaker::default()),
            active_counters: Mutex::new(std::collections::HashMap::new()),
            sessions: Mutex::new(std::collections::HashMap::new()),
            skipped_categories: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn correlator(&self) -> Arc<Correlator> {
        Arc::clone(&self.correlator)
    }

    pub fn scheduler_metrics(&self) -> Value {
        self.scheduler.metrics()
    }

    pub fn session_status(&self, session_id: &str) -> Option<Value> {
        let scheduler_status = self.scheduler.session_status(session_id);
        let active_status = self.active_status(session_id);
        if scheduler_status.is_none() && active_status.is_none() {
            return None;
        }
        let mut status = scheduler_status.unwrap_or_else(|| Value::Object(Default::default()));
        if status.get("skipped_categories").is_none() {
            let skipped = self.skipped_categories.lock().unwrap_or_else(|p| p.into_inner());
            status["skipped_categories"] = serde_json::to_value(skipped.get(session_id).cloned().unwrap_or_default()).ok()?;
        }
        if let Some(active) = active_status {
            status["active"] = active;
        }
        Some(status)
    }

    /// Counters for the last Active boundaries in one session. `applied` counts
    /// boundaries where a field of the outgoing request actually changed, so it
    /// is not an answer count.
    pub fn active_status(&self, session_id: &str) -> Option<Value> {
        let counters = self.active_counters.lock().unwrap_or_else(|p| p.into_inner());
        let counters = counters.get(session_id)?;
        Some(serde_json::json!({
            "applied": counters.applied,
            "accepted_no_effect": counters.accepted_no_effect,
            "refused": counters.refused,
            "unavailable": counters.unavailable,
            "last_reason": counters.last_reason,
            "last_category": counters.last_category,
        }))
    }

    /// Drop every queued/in-flight request for a session (disposal path).
    pub fn cancel_session(&self, session_id: &str) {
        self.scheduler.cancel_session(session_id);
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(session_id);
    }

    /// Stop all comparison work (observer disposal).
    pub fn shutdown(&self) {
        self.scheduler.shutdown();
    }

    /// Observe one event. Fire-and-forget; never blocks beyond the cheap
    /// mode check and one bounded extraction.
    pub fn observe(&self, event_type: &str, payload: &Value) {
        // Session disposal cancels pending work even when the mode just
        // flipped to Off; the check itself is cheap.
        if event_type == "session_shutdown" {
            let shutdown_session = payload
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if !shutdown_session.is_empty() {
                self.cancel_session(&shutdown_session);
            }
            return;
        }
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // Single cheap mode check first; in Off nothing else happens.
        if (self.config.mode_gate)(if session_id.is_empty() { None } else { Some(session_id.as_str()) })
            != JevMode::Compare
        {
            return;
        }
        match event_type {
            "turn_start" => self.observe_snapshot(event_type, payload, SnapshotStage::TurnStart),
            "tool_call" => self.observe_snapshot(event_type, payload, SnapshotStage::ToolCall),
            "model_select" => self.observe_snapshot(event_type, payload, SnapshotStage::ModelSelect),
            "agent_end" => self.observe_snapshot(event_type, payload, SnapshotStage::AgentEnd),
            // Observe-only bookkeeping: no scheduling, no records.
            "session_start" | "agent_start" | "message_end" | "tool_execution_start"
            | "tool_execution_end" | "input" => {}
            _ => {}
        }
    }

    fn next_seq(&self, session_id: &str) -> u32 {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Bounded: evict the oldest session when the map grows past 64.
        if sessions.len() > 64 {
            if let Some(first_key) = sessions.keys().next().cloned() {
                sessions.remove(&first_key);
            }
        }
        let counter = sessions.entry(session_id.to_string()).or_insert(0);
        *counter = counter.wrapping_add(1);
        *counter
    }

    /// Build the bounded request for one boundary: snapshot, baselines and
    /// typed questions. Shared by the Compare queue and the Active decision
    /// path so both ask the same thing of the same snapshot.
    ///
    /// Every refusal to ask is recorded here rather than silently dropped.
    fn prepare_bundle(&self, payload: &Value, stage: SnapshotStage, mode: &str) -> BundlePreparation {
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if session_id.is_empty() {
            return BundlePreparation::Nothing;
        }
        let turn = payload
            .get("turn")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let seq = self.next_seq(&session_id);
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        // Defensive bounding of the bridge-built state.
        let state = bound_json(
            payload.get("state").cloned().unwrap_or(Value::Null),
            0,
        );
        if serde_json::to_vec(&state).unwrap_or_default().len() > MAX_STATE_BYTES {
            // State too large after bounding: every eligible category gets an
            // explicit skip record (never fabricated, never silent).
            for evaluator in for_boundary(stage) {
                self.correlator.record_skipped_category(
                    &session_id,
                    turn,
                    stage.as_str(),
                    evaluator.category(),
                    "state_too_large",
                    PROMPT_VERSION,
                    mode,
                );
            }
            return BundlePreparation::StateTooLarge;
        }

        // Baselines captured synchronously at the boundary BEFORE any shadow
        // result returns.
        let mut baselines: BTreeMap<String, Option<String>> = BTreeMap::new();
        if stage == SnapshotStage::ToolCall {
            if let Some(tool) = payload.get("tool_name").and_then(Value::as_str) {
                baselines.insert(
                    crate::evaluators::question_id(crate::types::DecisionCategory::ToolCandidates, 0),
                    Some(tool.to_string()),
                );
            }
        }
        if stage == SnapshotStage::ModelSelect {
            if let Some(selected) = payload.get("selected_model").and_then(Value::as_str) {
                baselines.insert(
                    crate::evaluators::question_id(
                        crate::types::DecisionCategory::SubagentModelRouting,
                        0,
                    ),
                    Some(selected.to_string()),
                );
            }
        }
        if stage == SnapshotStage::AgentEnd {
            // The agent loop ended: the observed actual decision is stop.
            baselines.insert(
                crate::evaluators::question_id(
                    crate::types::DecisionCategory::ContinueStopEscalate,
                    0,
                ),
                Some("stop".to_string()),
            );
        }

        // Evaluate eligible categories at this boundary.
        let mut questions: Vec<PreparedQuestion> = Vec::new();
        let evaluator_input = StateSnapshot::new(
            stage,
            session_id.clone(),
            turn,
            seq,
            model.clone(),
            state.clone(),
            Vec::new(),
        );
        let snapshot = match evaluator_input {
            Ok(snapshot) => snapshot,
            Err(_) => return BundlePreparation::Nothing,
        };
        for evaluator in for_boundary(stage) {
            let category_id = evaluator.category().as_str().to_string();
            if !self.config.enabled_categories.is_empty()
                && !self.config.enabled_categories.contains(&category_id)
            {
                self.correlator.record_skipped_category(
                    &session_id,
                    turn,
                    stage.as_str(),
                    evaluator.category(),
                    "category_disabled",
                    PROMPT_VERSION,
                    mode,
                );
                continue;
            }
            match evaluator.evaluate(&snapshot) {
                EvaluatorOutput::Questions(mut prepared) => {
                    if let Some(skips) = self.skipped_categories.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&session_id) {
                        skips.remove(&category_id);
                    }
                    questions.append(&mut prepared);
                }
                EvaluatorOutput::Skipped(reason) => {
                    {
                        let mut skipped = self.skipped_categories.lock().unwrap_or_else(|p| p.into_inner());
                        if skipped.len() >= 64 && !skipped.contains_key(&session_id) {
                            if let Some(oldest) = skipped.keys().next().cloned() { skipped.remove(&oldest); }
                        }
                        skipped.entry(session_id.clone()).or_default().insert(category_id, reason.clone());
                    }
                    self.correlator.record_skipped_category(
                        &session_id,
                        turn,
                        stage.as_str(),
                        evaluator.category(),
                        &reason,
                        PROMPT_VERSION,
                        mode,
                    );
                }
            }
        }

        // Enforce the question cap honestly: overflow questions are recorded
        // as skipped, never dropped silently.
        let max_questions = self.config.scheduler.max_questions_per_request;
        let overflow: Vec<PreparedQuestion> =
            questions.split_off(max_questions.min(questions.len()));
        for question in overflow {
            self.correlator.record_skipped_category(
                &session_id,
                turn,
                stage.as_str(),
                DecisionCategory::parse(
                    question
                        .question_id
                        .rsplit_once('.')
                        .map(|(prefix, _)| prefix)
                        .unwrap_or_default(),
                )
                .unwrap_or(DecisionCategory::TaskClassification),
                "question_limit",
                PROMPT_VERSION,
                mode,
            );
        }
        if questions.is_empty() {
            // Every eligible category was skipped: nothing to ask.
            return BundlePreparation::Nothing;
        }

        // One bundled request per snapshot stage; no question depends on
        // another's answer (all sibling specs).
        let mut specs = BTreeMap::new();
        for question in &questions {
            specs.insert(question.question_id.clone(), question.spec.clone());
        }
        let request = crate::types::SystemOneRequest {
            state: state.clone(),
            model: SYSTEM_ONE_MODEL.to_string(),
            questions: specs,
        };
        let ctx = RequestContext {
            request_id: format!("jev-{}", Uuid::new_v4()),
            session_id,
            turn,
            stage: stage.as_str().to_string(),
            state_fingerprint: fingerprint_of(&state),
            state_schema_version: STATE_SCHEMA_VERSION.to_string(),
            prompt_version: PROMPT_VERSION.to_string(),
            mode: mode.to_string(),
            questions: questions
                .iter()
                .map(|question| QuestionMeta {
                    question_id: question.question_id.clone(),
                    category: question
                        .question_id
                        .rsplit_once('.')
                        .map(|(prefix, _)| prefix.to_string())
                        .unwrap_or_default(),
                })
                .collect(),
            baselines,
            request_start_ts: snapshot.created_at.clone(),
        };
        BundlePreparation::Ready(Box::new(PreparedBundle { request, ctx }))
    }

    /// Observe one boundary in Compare mode: capture baselines synchronously,
    /// hand the request to the bounded queue, and return. Fire-and-forget.
    fn observe_snapshot(&self, _event_type: &str, payload: &Value, stage: SnapshotStage) {
        // Baselines and context were captured synchronously at the boundary;
        // the SystemOne call happens asynchronously in the scheduler.
        if let BundlePreparation::Ready(bundle) = self.prepare_bundle(payload, stage, "compare") {
            self.correlator.track(&bundle.ctx);
            // Dropped requests are recorded by the scheduler sink.
            let _ = self.scheduler.enqueue(bundle.request, bundle.ctx);
        }
    }

    /// Decide one boundary in Active mode and return the answers an activation
    /// policy may apply. This is the only path in this crate that returns Jev
    /// output to a caller, and it is reachable only when the effective mode is
    /// `Active`.
    ///
    /// Bounded by `config.active.deadline`. A timeout, transport failure, open
    /// breaker or refused answer yields no decision and a reason; the caller
    /// must then behave exactly as if Jev were absent.
    pub async fn decide_active(
        &self,
        payload: &Value,
        stage: SnapshotStage,
        policy: &crate::active::ActivationPolicy,
    ) -> ActiveDecideOutcome {
        use crate::active::FallbackReason;

        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let turn = payload.get("turn").and_then(Value::as_u64).unwrap_or(0);
        let appliable: Vec<DecisionCategory> = for_boundary(stage)
            .into_iter()
            .map(|evaluator| evaluator.category())
            .filter(|category| policy.appliable().contains(category))
            .collect();
        let mut outcome = ActiveDecideOutcome {
            session_id: session_id.clone(),
            turn,
            stage: stage.as_str().to_string(),
            request_id: None,
            response_model: None,
            duration_ms: None,
            state_fingerprint: String::new(),
            decisions: Vec::new(),
            refusals: Vec::new(),
            unavailable: None,
        };
        if session_id.is_empty() {
            outcome.unavailable = Some(FallbackReason::ModeNotActive);
            return outcome;
        }
        // The gate is read again here: a caller cannot reach this path with a
        // non-Active effective mode.
        if (self.config.mode_gate)(Some(session_id.as_str())) != JevMode::Active {
            outcome.unavailable = Some(FallbackReason::ModeNotActive);
            return outcome;
        }
        if appliable.is_empty() {
            // Nothing at this boundary has a reversible effect, so a call
            // could not change the request even if it succeeded.
            outcome.unavailable = Some(FallbackReason::CategoryNotAppliable);
            return outcome;
        }
        if !self.active_breaker_allows() {
            for category in &appliable {
                outcome.refusals.push(ActiveRefusal::new(*category, FallbackReason::Unavailable));
            }
            outcome.unavailable = Some(FallbackReason::Unavailable);
            return outcome;
        }
        let bundle = match self.prepare_bundle(payload, stage, "active") {
            BundlePreparation::Ready(bundle) => bundle,
            BundlePreparation::StateTooLarge | BundlePreparation::Nothing => {
                outcome.unavailable = Some(FallbackReason::NoAnswer);
                return outcome;
            }
        };
        outcome.request_id = Some(bundle.ctx.request_id.clone());
        outcome.state_fingerprint = bundle.ctx.state_fingerprint.clone();
        let started = std::time::Instant::now();
        let call = self.system_one.decide(crate::client::bundle_with_questions(
            bundle.ctx.session_id.clone(),
            bundle.ctx.turn,
            bundle.ctx.stage.clone(),
            bundle.request.state.clone(),
            bundle.request.model.clone(),
            bundle.request.questions.clone(),
        ));
        let decision = match tokio::time::timeout(self.config.active.deadline, call).await {
            Ok(decision) => decision,
            Err(_) => {
                self.note_active_failure();
                outcome.duration_ms = Some(started.elapsed().as_millis() as u64);
                for category in &appliable {
                    outcome.refusals.push(ActiveRefusal::new(*category, FallbackReason::Unavailable));
                }
                outcome.unavailable = Some(FallbackReason::Unavailable);
                return outcome;
            }
        };
        outcome.duration_ms = Some(started.elapsed().as_millis() as u64);
        outcome.response_model = decision.response_model.clone();
        if decision.records.is_empty() {
            // No answer at all: transport failure, refusal to call, or an
            // unusable response. Either way nothing may be applied.
            self.note_active_failure();
            for category in &appliable {
                outcome.refusals.push(ActiveRefusal::new(*category, FallbackReason::NoAnswer));
            }
            outcome.unavailable = Some(FallbackReason::NoAnswer);
            return outcome;
        }
        self.note_active_success();
        let now = std::time::SystemTime::now();
        for record in &decision.records {
            let confidence = record.answer.confidence();
            let value = Some(record.answer.selected_value());
            if !appliable.contains(&record.category) {
                // Record-only category: answered, but this mode has no
                // reversible effect for it.
                let reason = if crate::active::DEFAULT_APPLIABLE_CATEGORIES.contains(&record.category) {
                    FallbackReason::CategoryDisabled
                } else {
                    FallbackReason::CategoryNotAppliable
                };
                outcome.refusals.push(ActiveRefusal::new(record.category, reason));
                continue;
            }
            let candidate = crate::active::AnswerCandidate {
                category: record.category,
                question_id: record.question_id.clone(),
                value,
                confidence,
                response_model: record.response_model.clone(),
                request_id: bundle.ctx.request_id.clone(),
                turn,
                decided_at: now,
            };
            match crate::active::evaluate_answer(policy, JevMode::Active, &candidate, now) {
                crate::active::Acceptance::Accepted(decision) => outcome.decisions.push(*decision),
                crate::active::Acceptance::Fallback(reason) => {
                    outcome.refusals.push(ActiveRefusal::new(record.category, reason));
                }
            }
        }
        outcome
    }

    /// Write the records for one Active boundary. `effects` maps a category id
    /// to the fields the host actually changed for that category; a category
    /// with no entry is recorded as accepted with no changes, which is not a
    /// success story and is visible as such.
    pub fn record_active(
        &self,
        outcome: &ActiveDecideOutcome,
        effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>,
    ) -> usize {
        if outcome.session_id.is_empty() {
            return 0;
        }
        let ctx = crate::correlate::ActiveRecordContext {
            request_id: outcome
                .request_id
                .clone()
                .unwrap_or_else(|| format!("active-{}", uuid::Uuid::new_v4())),
            session_id: outcome.session_id.clone(),
            turn: outcome.turn,
            stage: outcome.stage.clone(),
            response_model: outcome.response_model.clone(),
            duration_ms: outcome.duration_ms,
            state_fingerprint: outcome.state_fingerprint.clone(),
            prompt_version: PROMPT_VERSION.to_string(),
        };
        let mut rows: Vec<crate::correlate::ActiveRecordRow> = Vec::new();
        for decision in &outcome.decisions {
            let category_id = decision.category.as_str().to_string();
            rows.push(crate::correlate::ActiveRecordRow {
                category: category_id.clone(),
                question_id: decision.question_id.clone(),
                accepted: true,
                selected_value: Some(decision.value.clone()),
                confidence: Some(decision.confidence),
                fallback_reason: None,
                applied_effects: effects.get(&category_id).cloned().unwrap_or_default(),
            });
        }
        for refusal in &outcome.refusals {
            rows.push(crate::correlate::ActiveRecordRow {
                category: refusal.category.as_str().to_string(),
                question_id: format!("{}.0", refusal.category.as_str()),
                accepted: false,
                selected_value: None,
                confidence: None,
                fallback_reason: Some(refusal.reason.as_str().to_string()),
                applied_effects: Vec::new(),
            });
        }
        {
            let mut counters = self.active_counters.lock().unwrap_or_else(|p| p.into_inner());
            let entry = counters
                .entry(outcome.session_id.clone())
                .or_default();
            let applied = outcome
                .decisions
                .iter()
                .filter(|decision| effects.contains_key(decision.category.as_str()))
                .count() as u64;
            entry.applied += applied;
            entry.accepted_no_effect += outcome.decisions.len() as u64 - applied;
            entry.refused += outcome.refusals.len() as u64;
            if outcome.unavailable.is_some() {
                entry.unavailable += 1;
            }
            entry.last_reason = outcome
                .unavailable
                .map(|reason| reason.as_str().to_string())
                .or_else(|| {
                    outcome
                        .refusals
                        .first()
                        .map(|refusal| refusal.reason.as_str().to_string())
                });
            entry.last_category = outcome
                .decisions
                .first()
                .map(|decision| decision.category.as_str().to_string())
                .or_else(|| {
                    outcome
                        .refusals
                        .first()
                        .map(|refusal| refusal.category.as_str().to_string())
                });
        }
        if rows.is_empty() {
            return 0;
        }
        self.correlator.record_active_rows(&ctx, &rows)
    }

    /// True while the Active breaker permits a new call.
    fn active_breaker_allows(&self) -> bool {
        let breaker = self
            .active_breaker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match breaker.open_until {
            Some(deadline) => std::time::Instant::now() >= deadline,
            None => true,
        }
    }

    /// One failed or timed-out Active call. Repeated failures open the
    /// breaker so a dead service degrades to normal behavior instead of
    /// stalling every turn for the deadline.
    fn note_active_failure(&self) {
        let mut breaker = self
            .active_breaker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        breaker.consecutive_failures = breaker.consecutive_failures.saturating_add(1);
        if breaker.consecutive_failures >= self.config.active.max_consecutive_failures {
            breaker.open_until = Some(std::time::Instant::now() + self.config.active.breaker_cooldown);
            breaker.consecutive_failures = 0;
        }
    }

    fn note_active_success(&self) {
        let mut breaker = self
            .active_breaker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        breaker.consecutive_failures = 0;
        breaker.open_until = None;
    }

    /// Breaker state, for status surfaces and tests.
    pub fn active_breaker_open(&self) -> bool {
        !self.active_breaker_allows()
    }
}
