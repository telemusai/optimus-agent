//! Bounded System One decision and observation boundaries.
//! Compare runs asynchronously. Active and combined share one awaited result;
//! the host applies permitted effects and reports the actual changes.
//! Independent compaction uses a separate gate and cancellation axis.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

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
    pub mode: JevMode,
    pub context: Option<RequestContext>,
    pub raw: Option<crate::types::DecisionOutcome>,
    pub baseline_action: BTreeMap<String, String>,
    pub compaction_enabled: Option<bool>,
    pub terminal_reason: Option<String>,
    pub dispatched: bool,
    token: crate::scheduler::CancellationToken,
    independent: bool,
    policy_generation: String,
}

/// One refused answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRefusal {
    pub category: DecisionCategory,
    pub question_id: String,
    pub reason: crate::active::FallbackReason,
}

impl ActiveRefusal {
    pub fn new(category: DecisionCategory, question_id: String, reason: crate::active::FallbackReason) -> Self {
        Self { category, question_id, reason }
    }
}

/// Per-session counters for the Active path. These describe boundaries, not
/// answers: `applied` counts boundaries where a request field actually changed.
#[derive(Debug, Default)]
struct ActiveCounters {
    recommended: u64,
    accepted: u64,
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
    /// Independent compaction permission, also bound to credential generation.
    pub independent_gate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    pub policy_generation: Arc<dyn Fn(&str, bool) -> String + Send + Sync>,
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
            independent_gate: Arc::new(|_| false),
            policy_generation: Arc::new(|_, _| String::new()),
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

/// Shared observer and bounded decision service. The host owns every mutation.
pub struct JevObserver {
    config: JevObserverConfig,
    correlator: Arc<Correlator>,
    scheduler: JevScheduler,
    /// Retained for the Active path: one bounded, synchronous call per
    /// decision boundary. Unused by Compare, which goes through the queue.
    system_one: Arc<dyn crate::types::SystemOne>,
    active_breaker: Mutex<ActiveBreaker>,
    active_sessions: Mutex<std::collections::HashMap<(String, bool), crate::scheduler::CancellationToken>>,
    active_slots: tokio::sync::Semaphore,
    closed: AtomicBool,
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
            Arc::new(move |session_id| mode_gate(Some(session_id)).allows_compare()),
        );
        Arc::new(Self {
            config,
            correlator,
            scheduler,
            system_one,
            active_breaker: Mutex::new(ActiveBreaker::default()),
            active_sessions: Mutex::new(std::collections::HashMap::new()),
            active_slots: tokio::sync::Semaphore::new(2),
            closed: AtomicBool::new(false),
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
            "recommended": counters.recommended,
            "accepted": counters.accepted,
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
        let mut active = self.active_sessions.lock().unwrap_or_else(|p| p.into_inner());
        for independent in [false, true] {
            if let Some(token) = active.remove(&(session_id.to_string(), independent)) { token.cancel(); }
        }
        drop(active);
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(session_id);
    }

    pub fn cancel_decisions(&self, session_id: &str) {
        self.scheduler.cancel_session(session_id);
        if let Some(token) = self.active_sessions.lock().unwrap_or_else(|p| p.into_inner()).remove(&(session_id.to_string(), false)) { token.cancel(); }
    }

    /// Stop all comparison work (observer disposal).
    pub fn shutdown(&self) {
        self.closed.store(true, Ordering::SeqCst);
        for (_, token) in self.active_sessions.lock().unwrap_or_else(|p| p.into_inner()).drain() {
            token.cancel();
        }
        self.scheduler.shutdown();
    }

    /// Observe one event. Fire-and-forget; never blocks beyond the cheap
    /// mode check and one bounded extraction.
    pub fn observe(&self, event_type: &str, payload: &Value) {
        if self.closed.load(Ordering::SeqCst) { return; }
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
        if !(self.config.mode_gate)(if session_id.is_empty() { None } else { Some(session_id.as_str()) })
            .allows_compare()
        {
            return;
        }
        match event_type {
            "turn_start" => self.observe_snapshot(event_type, payload, SnapshotStage::TurnStart),
            "turn_end" => self.observe_snapshot(event_type, payload, SnapshotStage::TurnEnd),
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
                    if reason == "feature_disabled" { continue; }
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
        let mode = (self.config.mode_gate)(payload.get("session_id").and_then(Value::as_str));
        if let BundlePreparation::Ready(bundle) = self.prepare_bundle(payload, stage, mode.as_str()) {
            self.correlator.track_boundary(&bundle.ctx, action_baseline(payload), payload.get("compaction_enabled").and_then(Value::as_bool));
            // Dropped requests are recorded by the scheduler sink.
            let _ = self.scheduler.enqueue(bundle.request, bundle.ctx);
        }
    }

    /// Prepare explicit questions without truncating candidate IDs or question sets.
    fn prepare_explicit(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>, mode: JevMode) -> Option<PreparedBundle> {
        let session_id = payload.get("session_id")?.as_str()?.to_string();
        let state = payload.get("state")?.clone();
        if session_id.is_empty() || questions.is_empty()
            || questions.len() > self.config.scheduler.max_questions_per_request
            || serde_json::to_vec(&state).ok()?.len() > MAX_STATE_BYTES {
            return None;
        }
        let mut specs = BTreeMap::new();
        let mut metadata = Vec::new();
        for question in questions {
            let category = question.question_id.rsplit_once('.')?.0;
            if category != "compaction" { crate::types::DecisionCategory::parse(category)?; }
            metadata.push(QuestionMeta { question_id: question.question_id.clone(), category: category.to_string() });
            if specs.insert(question.question_id, question.spec).is_some() { return None; }
        }
        let request = crate::types::SystemOneRequest { state: state.clone(), model: SYSTEM_ONE_MODEL.to_string(), questions: specs };
        if crate::types::validate_request_shape(&request).is_err() { return None; }
        Some(PreparedBundle {
            request,
            ctx: RequestContext {
                request_id: format!("jev-{}", Uuid::new_v4()), session_id,
                turn: payload.get("turn").and_then(Value::as_u64).unwrap_or(0),
                stage: stage.to_string(), state_fingerprint: fingerprint_of(&state),
                state_schema_version: STATE_SCHEMA_VERSION.to_string(), prompt_version: PROMPT_VERSION.to_string(),
                mode: mode.as_str().to_string(), questions: metadata, baselines: BTreeMap::new(),
                request_start_ts: crate::client::utc_now_rfc3339(),
            },
        })
    }

    pub fn observe_prepared(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>) {
        let mode = (self.config.mode_gate)(payload.get("session_id").and_then(Value::as_str));
        if !mode.allows_compare() { return; }
        if let Some(bundle) = self.prepare_explicit(payload, stage, questions, mode) {
            self.correlator.track_boundary(&bundle.ctx, action_baseline(payload), payload.get("compaction_enabled").and_then(Value::as_bool));
            let _ = self.scheduler.enqueue(bundle.request, bundle.ctx);
        }
    }

    pub async fn decide_active(&self, payload: &Value, stage: SnapshotStage, policy: &crate::active::ActivationPolicy) -> ActiveDecideOutcome {
        let mode = (self.config.mode_gate)(payload.get("session_id").and_then(Value::as_str));
        let bundle = if mode.allows_active() {
            match self.prepare_bundle(payload, stage, mode.as_str()) {
                BundlePreparation::Ready(bundle) => Some(*bundle),
                _ => None,
            }
        } else { None };
        self.decide_bundle(payload, stage.as_str(), bundle, policy, mode, false).await
    }

    pub async fn decide_prepared(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>, policy: &crate::active::ActivationPolicy) -> ActiveDecideOutcome {
        let mode = (self.config.mode_gate)(payload.get("session_id").and_then(Value::as_str));
        let bundle = if mode.allows_active() { self.prepare_explicit(payload, stage, questions, mode) } else { None };
        self.decide_bundle(payload, stage, bundle, policy, mode, false).await
    }

    pub async fn decide_independent(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>) -> ActiveDecideOutcome {
        let mode = (self.config.mode_gate)(payload.get("session_id").and_then(Value::as_str));
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let bundle = if (self.config.independent_gate)(session_id) { self.prepare_explicit(payload, stage, questions, mode) } else { None };
        self.decide_bundle(payload, stage, bundle, &crate::active::ActivationPolicy::default(), mode, true).await
    }

    async fn decide_bundle(&self, payload: &Value, stage: &str, bundle: Option<PreparedBundle>, policy: &crate::active::ActivationPolicy, mode: JevMode, independent: bool) -> ActiveDecideOutcome {
        use crate::active::FallbackReason;
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("").to_string();
        let token = {
            let mut sessions = self.active_sessions.lock().unwrap_or_else(|p| p.into_inner());
            let key = (session_id.clone(), independent);
            if sessions.len() >= 64 && !sessions.contains_key(&key) {
                if let Some(key) = sessions.keys().next().cloned() {
                    if let Some(token) = sessions.remove(&key) { token.cancel(); }
                }
            }
            sessions.entry(key).or_default().clone()
        };
        let mut outcome = ActiveDecideOutcome {
            session_id, turn: payload.get("turn").and_then(Value::as_u64).unwrap_or(0),
            stage: stage.to_string(), request_id: None, response_model: None, duration_ms: None,
            state_fingerprint: String::new(), decisions: Vec::new(), refusals: Vec::new(), unavailable: None,
            mode, context: None, raw: None, baseline_action: action_baseline(payload),
            compaction_enabled: payload.get("compaction_enabled").and_then(Value::as_bool),
            terminal_reason: None, dispatched: false, token, independent,
            policy_generation: payload.get("policy_generation").and_then(Value::as_str).map(str::to_string)
                .unwrap_or_else(|| (self.config.policy_generation)(payload.get("session_id").and_then(Value::as_str).unwrap_or(""), independent)),
        };
        if !self.can_apply(&outcome) {
            outcome.unavailable = Some(FallbackReason::ModeNotActive);
            outcome.terminal_reason = Some("mode_or_generation_changed".to_string());
            return outcome;
        }
        let Some(bundle) = bundle else {
            outcome.unavailable = Some(FallbackReason::NoAnswer);
            outcome.terminal_reason = Some("no_eligible_questions".to_string());
            return outcome;
        };
        outcome.request_id = Some(bundle.ctx.request_id.clone());
        outcome.state_fingerprint = bundle.ctx.state_fingerprint.clone();
        outcome.context = Some(bundle.ctx.clone());
        if !self.active_breaker_allows() {
            outcome.unavailable = Some(FallbackReason::Unavailable);
            outcome.terminal_reason = Some("circuit_open".to_string());
            return outcome;
        }
        let Ok(_slot) = self.active_slots.try_acquire() else {
            outcome.unavailable = Some(FallbackReason::Unavailable);
            outcome.terminal_reason = Some("concurrency_limit".to_string());
            return outcome;
        };
        let started = std::time::Instant::now();
        outcome.dispatched = true;
        let call = self.system_one.decide(crate::client::bundle_with_questions(
            bundle.ctx.session_id.clone(), bundle.ctx.turn, bundle.ctx.stage.clone(),
            bundle.request.state, bundle.request.model, bundle.request.questions,
        ));
        let decision = tokio::select! {
            biased;
            _ = outcome.token.cancelled() => None,
            result = tokio::time::timeout(self.config.active.deadline, call) => match result {
                Ok(result) => Some(result),
                Err(_) => { outcome.terminal_reason = Some("timeout".to_string()); self.note_active_failure(); None }
            },
        };
        outcome.duration_ms = Some(started.elapsed().as_millis() as u64);
        if !self.can_apply(&outcome) || decision.is_none() {
            outcome.unavailable = Some(FallbackReason::Unavailable);
            if outcome.terminal_reason.is_none() { outcome.terminal_reason = Some("cancelled_or_generation_changed".to_string()); }
            return outcome;
        }
        let decision = decision.expect("checked decision");
        outcome.response_model = decision.response_model.clone();
        if decision.records.is_empty() {
            self.note_active_failure();
            outcome.unavailable = Some(FallbackReason::NoAnswer);
            outcome.terminal_reason = Some(decision.skips.first().map(|(_, reason)| *reason).unwrap_or("no_answer").to_string());
        } else { self.note_active_success(); }
        if !independent {
            let now = std::time::SystemTime::now();
            for record in &decision.records {
                let candidate = crate::active::AnswerCandidate {
                    category: record.category, question_id: record.question_id.clone(),
                    value: Some(record.answer.selected_value()), confidence: policy_confidence(record),
                    response_model: record.response_model.clone(), request_id: bundle.ctx.request_id.clone(),
                    turn: bundle.ctx.turn, decided_at: now,
                };
                let mut question_policy = policy.clone();
                if crate::active::OPTIONAL_APPLIABLE_CATEGORIES.contains(&record.category) {
                    if let Some(minimum) = payload.get("optional_min_confidence").and_then(Value::as_f64) {
                        question_policy.min_confidence = question_policy.min_confidence.max(minimum);
                    }
                    if let Some(age) = payload.get("optional_max_decision_age_ms").and_then(Value::as_u64) {
                        question_policy.max_decision_age = question_policy.max_decision_age.min(std::time::Duration::from_millis(age));
                    }
                }
                match crate::active::evaluate_answer(&question_policy, mode, &candidate, now) {
                    crate::active::Acceptance::Accepted(decision) => outcome.decisions.push(*decision),
                    crate::active::Acceptance::Fallback(reason) => outcome.refusals.push(ActiveRefusal::new(record.category, record.question_id.clone(), reason)),
                }
            }
        }
        outcome.raw = Some(decision);
        outcome
    }

    /// Re-check immediately before host mutation, including the captured credential generation.
    pub fn can_apply(&self, outcome: &ActiveDecideOutcome) -> bool {
        if outcome.session_id.is_empty() || self.closed.load(Ordering::SeqCst) || outcome.token.is_cancelled() { return false; }
        if (self.config.policy_generation)(&outcome.session_id, outcome.independent) != outcome.policy_generation { return false; }
        if outcome.independent { return (self.config.independent_gate)(&outcome.session_id); }
        outcome.mode.allows_active() && (self.config.mode_gate)(Some(&outcome.session_id)) == outcome.mode
    }

    pub fn record_active(&self, outcome: &ActiveDecideOutcome, effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>) -> usize {
        let mut actual = outcome.baseline_action.clone();
        for effect in effects.values().flatten() { actual.insert(effect.field.clone(), effect.to.clone().unwrap_or_else(|| "absent".to_string())); }
        self.record_active_with_action(outcome, effects, &actual)
    }

    pub fn record_active_with_action(&self, outcome: &ActiveDecideOutcome, effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>, actual: &BTreeMap<String, String>) -> usize {
        let still_current = self.can_apply(outcome);
        let Some(captured) = &outcome.context else { return 0; };
        if outcome.mode.allows_compare() && !outcome.independent {
            self.correlator.track_boundary(captured, outcome.baseline_action.clone(), outcome.compaction_enabled);
            if let Some(raw) = &outcome.raw {
                self.correlator.handle_result(&JobResult::Completed {
                    ctx: captured.clone(), outcome: raw.clone(), duration_ms: outcome.duration_ms.unwrap_or(0),
                });
            } else {
                self.correlator.skip_request(captured, outcome.terminal_reason.as_deref().unwrap_or("unavailable"), None);
            }
        }
        let ctx = crate::correlate::ActiveRecordContext {
            request_id: captured.request_id.clone(), session_id: captured.session_id.clone(), turn: captured.turn,
            stage: captured.stage.clone(), response_model: outcome.response_model.clone(), duration_ms: outcome.duration_ms,
            state_fingerprint: captured.state_fingerprint.clone(), prompt_version: captured.prompt_version.clone(),
            mode: captured.mode.clone(), request_start_ts: Some(captured.request_start_ts.clone()),
            attempts: outcome.raw.as_ref().map(|raw| raw.attempts).unwrap_or(u32::from(outcome.dispatched)),
            attempt_count_known: outcome.raw.is_some() || !outcome.dispatched, baselines: captured.baselines.clone(),
            baseline_action: outcome.baseline_action.clone(), compaction_enabled: outcome.compaction_enabled,
            observed_metrics: outcome.raw.as_ref().map(crate::correlate::usage_metrics).unwrap_or_default(),
        };
        let mut rows = Vec::new();
        for question in &captured.questions {
            let answer = outcome.raw.as_ref().and_then(|raw| raw.records.iter().find(|record| record.question_id == question.question_id));
            let accepted = still_current && outcome.decisions.iter().any(|decision| decision.question_id == question.question_id);
            let applied_effects = if accepted {
                effects.get(&question.question_id).or_else(|| effects.get(&question.category)).cloned().unwrap_or_default()
            } else { Vec::new() };
            let actual_action = if still_current { actual.clone() } else { outcome.baseline_action.clone() };
            let fallback = if !still_current { Some("cancelled_or_policy_changed".to_string()) } else {
                outcome.refusals.iter().find(|refusal| refusal.question_id == question.question_id)
                    .map(|refusal| refusal.reason.as_str().to_string()).or_else(|| outcome.terminal_reason.clone())
            };
            let skipped = outcome.raw.as_ref().and_then(|raw| raw.skips.iter().find(|(id, _)| id.is_empty() || id == &question.question_id)).map(|(_, reason)| reason.to_string());
            rows.push(crate::correlate::ActiveRecordRow {
                category: question.category.clone(), question_id: question.question_id.clone(), accepted,
                selected_value: answer.map(|record| record.answer.selected_value()), confidence: answer.and_then(|record| record.answer.confidence()),
                fallback_reason: if accepted { None } else { fallback.or_else(|| skipped.clone()).or_else(|| Some("no_answer".to_string())) },
                outcome: if !applied_effects.is_empty() { "applied" } else if accepted { "accepted_no_effect" } else if outcome.unavailable.is_some() { "unavailable" } else { "refused" }.to_string(),
                applied_effects, skipped_reason: skipped, actual_action,
            });
        }
        {
            let mut counters = self.active_counters.lock().unwrap_or_else(|p| p.into_inner());
            if counters.len() >= 64 && !counters.contains_key(&outcome.session_id) {
                if let Some(key) = counters.keys().next().cloned() { counters.remove(&key); }
            }
            let entry = counters.entry(outcome.session_id.clone()).or_default();
            entry.recommended += rows.iter().filter(|row| row.selected_value.is_some()).count() as u64;
            entry.accepted += rows.iter().filter(|row| row.accepted).count() as u64;
            entry.applied += rows.iter().filter(|row| !row.applied_effects.is_empty()).count() as u64;
            entry.accepted_no_effect += rows.iter().filter(|row| row.accepted && row.applied_effects.is_empty()).count() as u64;
            entry.refused += rows.iter().filter(|row| !row.accepted).count() as u64;
            entry.unavailable += u64::from(outcome.unavailable.is_some());
            entry.last_reason = outcome.terminal_reason.clone().or_else(|| rows.iter().find_map(|row| row.fallback_reason.clone()));
            entry.last_category = rows.first().map(|row| row.category.clone());
        }
        let count = self.correlator.record_active_rows(&ctx, &rows);
        if let Some(notify) = &self.config.on_terminal { notify(&outcome.session_id); }
        count
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

fn action_baseline(payload: &Value) -> BTreeMap<String, String> {
    payload.get("baseline_action").and_then(Value::as_object).map(|object| object.iter().take(16)
        .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_string()))).collect()).unwrap_or_default()
}

fn policy_confidence(record: &crate::types::DecisionRecord) -> Option<f64> {
    let confidence = record.answer.confidence()?;
    if crate::active::OPTIONAL_APPLIABLE_CATEGORIES.contains(&record.category) {
        if let crate::types::Answer::Choice { choice, probabilities, .. } = &record.answer {
            return probabilities.get(choice).copied().map(|probability| confidence.min(probability));
        }
        return None;
    }
    Some(confidence)
}
