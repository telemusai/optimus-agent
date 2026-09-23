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

pub const PROMPT_VERSION: &str = "jev-compare-prompts/2";

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
    /// Requested Jev model captured with `policy_generation` before dispatch.
    pub requested_model: String,
    pub context: Option<RequestContext>,
    pub raw: Option<crate::types::DecisionOutcome>,
    pub baseline_action: BTreeMap<String, String>,
    pub compaction_enabled: Option<bool>,
    pub terminal_reason: Option<String>,
    pub dispatched: bool,
    token: crate::scheduler::CancellationToken,
    retention_cancel: crate::scheduler::CancellationToken,
    retention_generation: u64,
    independent: bool,
    /// Decision-policy generation captured for this outcome; consumed by the
    /// host control seam (jev_bridge decide_control) for correlation.
    pub policy_generation: String,
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

/// One authoritative request snapshot. Production hosts construct all four
/// fields from ONE settings-store load before any await. `policy_generation`
/// includes the durable write identity and effective feature/compaction
/// policy; `allowed` covers mode/credential or independent-compaction gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevRequestGate {
    pub mode: JevMode,
    pub requested_model: String,
    pub policy_generation: String,
    pub allowed: bool,
}

/// Observer configuration.
#[derive(Clone)]
pub struct JevObserverConfig {
    /// Cheap effective-mode AND requested-Jev-model resolution, invoked once
    /// per observed event (ROOT-CONTRACT v9). Both values MUST come from ONE
    /// authoritative settings snapshot: the host resolves them together so a
    /// request's mode gate and its `model` field can never disagree, and the
    /// capture happens at the boundary BEFORE any await. The requested model
    /// is stamped into every prepared request (compare queue, active decide,
    /// guidance, control, search, evidence, independent compaction); a late
    /// selection change cannot mix into an already-captured payload — the
    /// durable `write_revision` carried by the policy generation invalidates
    /// such work at the apply boundary instead. The default returns
    /// `(JevMode::Off, SYSTEM_ONE_MODEL)` so unchanged hosts stay
    /// byte-equivalent.
    pub mode_gate: Arc<dyn Fn(Option<&str>) -> (JevMode, String) + Send + Sync>,
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
    /// Production-only single-load gate. When present, it is authoritative
    /// for mode, requested model, durable policy generation and permission.
    /// The legacy closures remain as a compatibility fallback for isolated
    /// embedders and fixtures.
    pub authoritative_gate:
        Option<Arc<dyn Fn(Option<&str>, bool) -> JevRequestGate + Send + Sync>>,
}

impl Default for JevObserverConfig {
    fn default() -> Self {
        Self {
            mode_gate: Arc::new(|_| (JevMode::Off, SYSTEM_ONE_MODEL.to_string())),
            enabled_categories: HashSet::new(),
            scheduler: SchedulerConfig::default(),
            min_confidence: 0.7,
            on_terminal: None,
            active: ActiveSettings::default(),
            independent_gate: Arc::new(|_| false),
            policy_generation: Arc::new(|_, _| String::new()),
            authoritative_gate: None,
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

fn resolve_request_gate(
    config: &JevObserverConfig,
    session_id: Option<&str>,
    independent: bool,
) -> JevRequestGate {
    if let Some(gate) = &config.authoritative_gate {
        return gate(session_id, independent);
    }
    let (mode, requested_model) = (config.mode_gate)(session_id);
    let id = session_id.unwrap_or("");
    let policy_generation = (config.policy_generation)(id, independent);
    let allowed = if independent {
        (config.independent_gate)(id)
    } else {
        mode.is_enabled()
    };
    JevRequestGate {
        mode,
        requested_model,
        policy_generation,
        allowed,
    }
}

fn payload_matches_gate(
    config: &JevObserverConfig,
    payload: &Value,
    gate: &JevRequestGate,
) -> bool {
    match payload.get("policy_generation").and_then(Value::as_str) {
        Some(captured) => captured == gate.policy_generation,
        None => config.authoritative_gate.is_none(),
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
        let scheduler_gate_config = config.clone();
        let scheduler = JevScheduler::new_with_gate(
            config.scheduler.clone(),
            Arc::clone(&system_one),
            sink,
            Arc::new(move |ctx| {
                let gate = resolve_request_gate(
                    &scheduler_gate_config,
                    Some(&ctx.session_id),
                    false,
                );
                gate.allowed
                    && gate.mode.allows_compare()
                    && gate.requested_model == ctx.requested_model
                    && gate.policy_generation == ctx.policy_generation
            }),
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
        let usage = crate::telemetry::session_usage(session_id);
        if scheduler_status.is_none() && active_status.is_none() && usage.is_none() {
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
        if let Some(usage) = usage {
            status["usage"] = serde_json::to_value(usage).ok()?;
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
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let Some(work) = crate::scheduler::register_session_work(session_id, None) else { return; };
        self.observe_owned(event_type, payload);
        work.finish();
    }

    fn observe_owned(&self, event_type: &str, payload: &Value) {
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
        // Single authoritative gate check first; in Off nothing else happens.
        let gate = resolve_request_gate(
            &self.config,
            if session_id.is_empty() { None } else { Some(session_id.as_str()) },
            false,
        );
        if !gate.allowed || !gate.mode.allows_compare() {
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
    fn prepare_bundle(
        &self,
        payload: &Value,
        stage: SnapshotStage,
        mode: &str,
        requested_model: &str,
        policy_generation: &str,
    ) -> BundlePreparation {
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
        // ROOT-CONTRACT v9: the request carries the requested Jev model
        // captured with the mode gate BEFORE the request is queued. The
        // scheduler sends it as-is (no late getter) and the durable
        // `write_revision` invalidates it when the selection changes
        // meanwhile — including A->B->A — at the apply boundary.
        let request = crate::types::SystemOneRequest {
            state: state.clone(),
            model: requested_model.to_string(),
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
            requested_model: requested_model.to_string(),
            policy_generation: policy_generation.to_string(),
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
        // Production resolves mode + requested model + durable generation from
        // ONE settings snapshot. A payload built from an older feature snapshot
        // is refused before queueing rather than mixed with the newer model.
        let gate = resolve_request_gate(
            &self.config,
            payload.get("session_id").and_then(Value::as_str),
            false,
        );
        if !gate.allowed || !gate.mode.allows_compare() || !payload_matches_gate(&self.config, payload, &gate) {
            return;
        }
        if let BundlePreparation::Ready(bundle) = self.prepare_bundle(
            payload,
            stage,
            gate.mode.as_str(),
            &gate.requested_model,
            &gate.policy_generation,
        ) {
            self.correlator.track_boundary(&bundle.ctx, action_baseline(payload), payload.get("compaction_enabled").and_then(Value::as_bool));
            // Dropped requests are recorded by the scheduler sink. Its dispatch
            // gate rechecks model + generation both before send and on return.
            let _ = self.scheduler.enqueue(bundle.request, bundle.ctx);
        }
    }

    /// Prepare explicit questions without truncating candidate IDs or question sets.
    fn prepare_explicit(
        &self,
        payload: &Value,
        stage: &str,
        questions: Vec<PreparedQuestion>,
        mode: JevMode,
        requested_model: &str,
        policy_generation: &str,
    ) -> Option<PreparedBundle> {
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
        // ROOT-CONTRACT v9: explicit-questions requests (guidance, control,
        // search, evidence and independent-compaction decides) carry the
        // requested Jev model captured with the SAME gate snapshot.
        let request = crate::types::SystemOneRequest { state: state.clone(), model: requested_model.to_string(), questions: specs };
        if crate::types::validate_request_shape(&request).is_err() { return None; }
        Some(PreparedBundle {
            request,
            ctx: RequestContext {
                request_id: format!("jev-{}", Uuid::new_v4()), session_id,
                turn: payload.get("turn").and_then(Value::as_u64).unwrap_or(0),
                stage: stage.to_string(), state_fingerprint: fingerprint_of(&state),
                state_schema_version: STATE_SCHEMA_VERSION.to_string(), prompt_version: PROMPT_VERSION.to_string(),
                mode: mode.as_str().to_string(), requested_model: requested_model.to_string(),
                policy_generation: policy_generation.to_string(), questions: metadata,
                baselines: BTreeMap::new(),
                request_start_ts: crate::client::utc_now_rfc3339(),
            },
        })
    }

    pub fn observe_prepared(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>) {
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let Some(work) = crate::scheduler::register_session_work(session_id, None) else { return; };
        self.observe_prepared_owned(payload, stage, questions);
        work.finish();
    }

    fn observe_prepared_owned(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>) {
        let gate = resolve_request_gate(
            &self.config,
            payload.get("session_id").and_then(Value::as_str),
            false,
        );
        if !gate.allowed || !gate.mode.allows_compare() || !payload_matches_gate(&self.config, payload, &gate) {
            return;
        }
        if let Some(bundle) = self.prepare_explicit(
            payload,
            stage,
            questions,
            gate.mode,
            &gate.requested_model,
            &gate.policy_generation,
        ) {
            self.correlator.track_boundary(&bundle.ctx, action_baseline(payload), payload.get("compaction_enabled").and_then(Value::as_bool));
            let _ = self.scheduler.enqueue(bundle.request, bundle.ctx);
        }
    }

    pub async fn decide_active(&self, payload: &Value, stage: SnapshotStage, policy: &crate::active::ActivationPolicy) -> ActiveDecideOutcome {
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let work = crate::scheduler::register_session_work(session_id, None);
        let gate = resolve_request_gate(
            &self.config,
            payload.get("session_id").and_then(Value::as_str),
            false,
        );
        let bundle = if work.is_some() && gate.allowed && gate.mode.allows_active() && payload_matches_gate(&self.config, payload, &gate) {
            match self.prepare_bundle(
                payload,
                stage,
                gate.mode.as_str(),
                &gate.requested_model,
                &gate.policy_generation,
            ) {
                BundlePreparation::Ready(bundle) => Some(*bundle),
                _ => None,
            }
        } else { None };
        self.decide_bundle(payload, stage.as_str(), bundle, policy, gate, false, work).await
    }

    pub async fn decide_prepared(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>, policy: &crate::active::ActivationPolicy) -> ActiveDecideOutcome {
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let work = crate::scheduler::register_session_work(session_id, None);
        let gate = resolve_request_gate(
            &self.config,
            payload.get("session_id").and_then(Value::as_str),
            false,
        );
        let bundle = if work.is_some() && gate.allowed && gate.mode.allows_active() && payload_matches_gate(&self.config, payload, &gate) {
            self.prepare_explicit(
                payload,
                stage,
                questions,
                gate.mode,
                &gate.requested_model,
                &gate.policy_generation,
            )
        } else { None };
        self.decide_bundle(payload, stage, bundle, policy, gate, false, work).await
    }

    pub async fn decide_independent(&self, payload: &Value, stage: &str, questions: Vec<PreparedQuestion>) -> ActiveDecideOutcome {
        let session_id = payload.get("session_id").and_then(Value::as_str).unwrap_or("");
        let work = crate::scheduler::register_session_work(session_id, None);
        let gate = resolve_request_gate(
            &self.config,
            payload.get("session_id").and_then(Value::as_str),
            true,
        );
        let bundle = if work.is_some() && gate.allowed && payload_matches_gate(&self.config, payload, &gate) {
            self.prepare_explicit(
                payload,
                stage,
                questions,
                gate.mode,
                &gate.requested_model,
                &gate.policy_generation,
            )
        } else { None };
        self.decide_bundle(payload, stage, bundle, &crate::active::ActivationPolicy::default(), gate, true, work).await
    }

    async fn decide_bundle(&self, payload: &Value, stage: &str, bundle: Option<PreparedBundle>, policy: &crate::active::ActivationPolicy, gate: JevRequestGate, independent: bool, mut work: Option<crate::scheduler::SessionWorkGuard>) -> ActiveDecideOutcome {
        let outcome = self.decide_bundle_owned(payload, stage, bundle, policy, gate, independent, &mut work).await;
        if let Some(work) = work { work.finish(); }
        outcome
    }

    async fn decide_bundle_owned(&self, payload: &Value, stage: &str, bundle: Option<PreparedBundle>, policy: &crate::active::ActivationPolicy, gate: JevRequestGate, independent: bool, work: &mut Option<crate::scheduler::SessionWorkGuard>) -> ActiveDecideOutcome {
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
        if let Some(work) = work.as_mut() { work.bind_normal_cancel(token.clone()); }
        let retention_cancel = work.as_ref().map(|work| work.token()).unwrap_or_else(|| {
            let token = crate::scheduler::CancellationToken::new();
            token.cancel();
            token
        });
        let retention_generation = work.as_ref().map(|work| work.generation).unwrap_or_default();
        let mode = gate.mode;
        let captured_generation = payload
            .get("policy_generation")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| gate.policy_generation.clone());
        let mut outcome = ActiveDecideOutcome {
            session_id, turn: payload.get("turn").and_then(Value::as_u64).unwrap_or(0),
            stage: stage.to_string(), request_id: None, response_model: None, duration_ms: None,
            state_fingerprint: String::new(), decisions: Vec::new(), refusals: Vec::new(), unavailable: None,
            mode, requested_model: gate.requested_model, context: None, raw: None,
            baseline_action: action_baseline(payload),
            compaction_enabled: payload.get("compaction_enabled").and_then(Value::as_bool),
            terminal_reason: None, dispatched: false, token, retention_cancel, retention_generation, independent,
            policy_generation: captured_generation,
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
        let deadline = payload.get("decision_timeout_ms").and_then(Value::as_u64)
            .map(std::time::Duration::from_millis).unwrap_or(self.config.active.deadline)
            .min(self.config.active.deadline);
        if deadline.is_zero() {
            outcome.unavailable = Some(FallbackReason::Unavailable);
            outcome.terminal_reason = Some("deadline_exhausted".to_string());
            return outcome;
        }
        let started = std::time::Instant::now();
        outcome.dispatched = true;
        // Clone the moved request fields: the typed acceptance borrows below
        // still need the prepared bundle intact (same request identity).
        let call = self.system_one.decide(crate::client::bundle_with_questions(
            bundle.ctx.session_id.clone(), bundle.ctx.turn, bundle.ctx.stage.clone(),
            bundle.request.state.clone(), bundle.request.model.clone(), bundle.request.questions.clone(),
        ));
        let decision = tokio::select! {
            biased;
            _ = outcome.token.cancelled() => None,
            _ = outcome.retention_cancel.cancelled() => None,
            result = tokio::time::timeout(deadline, call) => match result {
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
            let mut line_find_records: Vec<&crate::types::DecisionRecord> = Vec::new();
            for record in &decision.records {
                // ROOT CONTRACT v1 (Search): scored categories take the typed
                // acceptance path. A Noul is never refused for lacking Choice
                // confidence and never gated by a confidence threshold; the
                // legacy path below is byte-identical for all other categories.
                // ROOT-CONTRACT v6 (Evidence lane): the citation check is a
                // Choice-category advisory assessment with its own typed
                // assessor (never a Noul, never a keep/drop effect).
                if record.category == crate::types::DecisionCategory::CodeCitationCheck {
                    match assess_citation_record(record, &bundle, policy, mode, now) {
                        crate::active::Acceptance::Accepted(decision) => outcome.decisions.push(*decision),
                        crate::active::Acceptance::Fallback(reason) => outcome.refusals.push(ActiveRefusal::new(record.category, record.question_id.clone(), reason)),
                    }
                    continue;
                }
                if crate::active::SCORED_SEARCH_CATEGORIES.contains(&record.category) {
                    if record.category == crate::types::DecisionCategory::CodeLineFind {
                        line_find_records.push(record);
                        continue;
                    }
                    match assess_scored_record(record, &bundle, policy, mode, now) {
                        crate::active::Acceptance::Accepted(decision) => outcome.decisions.push(*decision),
                        crate::active::Acceptance::Fallback(reason) => outcome.refusals.push(ActiveRefusal::new(record.category, record.question_id.clone(), reason)),
                    }
                    continue;
                }
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
            if !line_find_records.is_empty() {
                // The where-Choice and existence-Noul of one line-find request
                // are accepted as ONE typed pair: a partial pair never applies.
                match assess_line_find_pair(&line_find_records, &bundle, policy, mode, now) {
                    LineFindAcceptance::Accepted(mut decisions) => outcome.decisions.append(&mut decisions),
                    LineFindAcceptance::Fallback(reason) => {
                        for record in line_find_records {
                            outcome.refusals.push(ActiveRefusal::new(record.category, record.question_id.clone(), reason));
                        }
                    }
                }
            }
        }
        outcome.raw = Some(decision);
        outcome
    }

    /// Re-check immediately before host mutation from one authoritative
    /// settings snapshot, including credential, durable write/model identity,
    /// mode and independent-compaction permission.
    pub fn can_apply(&self, outcome: &ActiveDecideOutcome) -> bool {
        if outcome.session_id.is_empty()
            || self.closed.load(Ordering::SeqCst)
            || outcome.token.is_cancelled()
            || outcome.retention_cancel.is_cancelled()
            || !crate::scheduler::session_work_is_current(&outcome.session_id, outcome.retention_generation)
        {
            return false;
        }
        let current = resolve_request_gate(
            &self.config,
            Some(&outcome.session_id),
            outcome.independent,
        );
        if !current.allowed
            || current.policy_generation != outcome.policy_generation
            || current.requested_model != outcome.requested_model
        {
            return false;
        }
        if outcome.independent {
            return true;
        }
        outcome.mode.allows_active() && current.mode == outcome.mode
    }

    pub fn record_active(&self, outcome: &ActiveDecideOutcome, effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>) -> usize {
        let mut actual = outcome.baseline_action.clone();
        for effect in effects.values().flatten() { actual.insert(effect.field.clone(), effect.to.clone().unwrap_or_else(|| "absent".to_string())); }
        self.record_active_with_action(outcome, effects, &actual)
    }

    pub fn record_active_with_action(&self, outcome: &ActiveDecideOutcome, effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>, actual: &BTreeMap<String, String>) -> usize {
        let Some(work) = crate::scheduler::register_session_work(&outcome.session_id, None) else { return 0; };
        if work.generation != outcome.retention_generation {
            work.finish();
            return 0;
        }
        let count = self.record_active_with_action_owned(outcome, effects, actual);
        work.finish();
        count
    }

    fn record_active_with_action_owned(&self, outcome: &ActiveDecideOutcome, effects: &BTreeMap<String, Vec<crate::active::AppliedEffect>>, actual: &BTreeMap<String, String>) -> usize {
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



#[cfg(test)]
mod settlement_tests {
    use super::*;
    use crate::scheduler::{request_session_retain_stop, session_retain_status, settle_session_retain_stop, begin_session_retain_generation};
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    struct Probe {
        calls: AtomicU64,
        started: tokio::sync::Notify,
        pending: bool,
        dropped: Arc<AtomicBool>,
    }
    impl crate::types::SystemOne for Probe {
        fn mode(&self) -> JevMode { JevMode::Active }
        fn decide(&self, _: crate::types::DecisionBundle) -> crate::types::BoxFuture<crate::types::DecisionOutcome> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            struct Mark(Arc<AtomicBool>);
            impl Drop for Mark { fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); } }
            let mark = Mark(self.dropped.clone());
            let pending = self.pending;
            Box::pin(async move {
                let _mark = mark;
                if pending { std::future::pending::<()>().await; }
                crate::types::DecisionOutcome::skipped_all("local_mock")
            })
        }
    }

    fn probe(pending: bool) -> Arc<Probe> {
        Arc::new(Probe { calls: AtomicU64::new(0), started: tokio::sync::Notify::new(), pending, dropped: Arc::new(AtomicBool::new(false)) })
    }

    fn config() -> JevObserverConfig {
        JevObserverConfig {
            mode_gate: Arc::new(|_| (JevMode::CompareAndActive, SYSTEM_ONE_MODEL.into())),
            independent_gate: Arc::new(|_| true),
            scheduler: SchedulerConfig { min_interval: Duration::ZERO, ..Default::default() },
            ..Default::default()
        }
    }

    fn input() -> Value {
        serde_json::json!({"session_id": format!("hooks-retain-{}", Uuid::new_v4()), "turn": 1, "state": {"task": "synthetic"}})
    }

    fn questions() -> Vec<PreparedQuestion> {
        vec![PreparedQuestion { question_id: "tool_requirement.0".into(), spec: crate::mock::choice_question("Need tools?", &[("none", None), ("read", None)]) }]
    }

    #[tokio::test]
    async fn active_and_independent_cancellation_drop_actual_call_before_acknowledgement() {
        for independent in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let client = probe(true);
            let observer = JevObserver::new(config(), client.clone(), dir.path().join("records.jsonl"));
            let input = input();
            let id = input["session_id"].as_str().unwrap().to_string();
            let captured = observer.clone();
            let task = tokio::spawn(async move {
                if independent { captured.decide_independent(&input, "local_fixture", questions()).await }
                else { captured.decide_prepared(&input, "local_fixture", questions(), &crate::active::ActivationPolicy::default()).await }
            });
            client.started.notified().await;
            assert_eq!(request_session_retain_stop(&id).pending_work, 1);
            let outcome = task.await.unwrap();
            assert!(!observer.can_apply(&outcome));
            let done = settle_session_retain_stop(&id, Duration::from_secs(1)).await;
            assert!(done.settled, "{done:?}");
            assert!(client.dropped.load(Ordering::SeqCst));
            assert_eq!(done.cancelled_work, 1);
            assert_eq!(observer.record_active(&outcome, &BTreeMap::new()), 0);
        }
    }

    #[tokio::test]
    async fn observer_replacement_and_explicit_resume_do_not_revive_old_outcomes() {
        let dir = tempfile::tempdir().unwrap();
        let client = probe(false);
        let observer = JevObserver::new(config(), client.clone(), dir.path().join("first.jsonl"));
        let input = input();
        let id = input["session_id"].as_str().unwrap();
        let old = observer.decide_prepared(&input, "fixture", questions(), &crate::active::ActivationPolicy::default()).await;
        assert!(observer.can_apply(&old));
        assert!(!session_retain_status(id).cancel_requested);
        let stopped = request_session_retain_stop(id);
        assert!(stopped.settled);
        let replacement = JevObserver::new(config(), client.clone(), dir.path().join("second.jsonl"));
        replacement.observe_prepared(&input, "fixture", questions());
        let declined = replacement.decide_prepared(&input, "fixture", questions(), &crate::active::ActivationPolicy::default()).await;
        assert!(!declined.dispatched);
        assert!(!replacement.can_apply(&declined));
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
        assert!(begin_session_retain_generation(id, stopped.generation));
        assert!(!replacement.can_apply(&old));
        assert_eq!(replacement.record_active(&old, &BTreeMap::new()), 0);
        let new = replacement.decide_prepared(&input, "fixture", questions(), &crate::active::ActivationPolicy::default()).await;
        assert!(replacement.can_apply(&new));
        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn active_record_callback_is_tracked_until_return_even_after_stop() {
        let dir = tempfile::tempdir().unwrap();
        let client = probe(false);
        let (started, ready) = tokio::sync::oneshot::channel();
        let started = Arc::new(Mutex::new(Some(started)));
        let (release, wait) = std::sync::mpsc::channel();
        let wait = Arc::new(Mutex::new(wait));
        let mut cfg = config();
        cfg.on_terminal = Some(Arc::new(move |_| {
            if let Some(started) = started.lock().unwrap().take() { let _ = started.send(()); }
            wait.lock().unwrap().recv_timeout(Duration::from_secs(5)).unwrap();
        }));
        let observer = JevObserver::new(cfg, client, dir.path().join("records.jsonl"));
        let input = input();
        let id = input["session_id"].as_str().unwrap();
        let outcome = observer.decide_prepared(&input, "fixture", questions(), &crate::active::ActivationPolicy::default()).await;
        let captured = observer.clone();
        let task = tokio::spawn(async move { captured.record_active(&outcome, &BTreeMap::new()) });
        ready.await.unwrap();
        assert_eq!(request_session_retain_stop(id).pending_work, 1);
        assert!(!settle_session_retain_stop(id, Duration::ZERO).await.settled);
        release.send(()).unwrap();
        assert!(task.await.unwrap() > 0);
        assert!(settle_session_retain_stop(id, Duration::from_secs(1)).await.settled);
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

// ROOT CONTRACT v1 (Search): typed acceptance for scored categories. This
// never weakens the legacy Choice path; a Noul is never converted into
// Choice confidence and never gated by a confidence threshold.
enum LineFindAcceptance {
    Accepted(Vec<crate::active::ActiveDecision>),
    Fallback(crate::active::FallbackReason),
}

fn assess_scored_record(
    record: &crate::types::DecisionRecord,
    bundle: &PreparedBundle,
    policy: &crate::active::ActivationPolicy,
    mode: JevMode,
    now: std::time::SystemTime,
) -> crate::active::Acceptance {
    use crate::active::{ActiveDecision, FallbackReason};
    if !mode.allows_active() {
        return crate::active::Acceptance::Fallback(FallbackReason::ModeNotActive);
    }
    if !policy.allows(record.category) {
        return crate::active::Acceptance::Fallback(FallbackReason::CategoryDisabled);
    }
    let noul = match &record.answer {
        crate::types::Answer::Noul { noul } if noul.is_finite() && (0.0..=1.0).contains(noul) => *noul,
        _ => return crate::active::Acceptance::Fallback(FallbackReason::InvalidValue),
    };
    let assessment = crate::types::RerankAssessment {
        candidate_id: record
            .question_id
            .rsplit_once('.')
            .and_then(|(_, id)| id.parse().ok())
            .unwrap_or(usize::MAX),
        noul,
        complete: true,
        correlated: true,
        fresh: true,
    };
    // The legacy confidence slot carries the raw Noul for record continuity;
    // no gate ever compares it to a Choice-confidence threshold (the typed
    // struct above carries the acceptance truth).
    crate::active::Acceptance::Accepted(Box::new(ActiveDecision {
        category: record.category,
        question_id: record.question_id.clone(),
        value: record.answer.selected_value(),
        confidence: assessment.noul,
        response_model: record.response_model.clone(),
        request_id: bundle.ctx.request_id.clone(),
        turn: bundle.ctx.turn,
        decided_at: now,
    }))
}

/// ROOT-CONTRACT v6 (Evidence lane): typed acceptance for the advisory
/// citation-relation record (Choice: supports / contradicts / unclear).
/// Mirrors the scored path's policy/mode checks. It is never a keep/drop
/// effect and never a verification: the chosen criterion string is carried
/// as the decision value and consumers label it advisory-only.
fn assess_citation_record(
    record: &crate::types::DecisionRecord,
    bundle: &PreparedBundle,
    policy: &crate::active::ActivationPolicy,
    mode: JevMode,
    now: std::time::SystemTime,
) -> crate::active::Acceptance {
    use crate::active::{ActiveDecision, FallbackReason};
    if !mode.allows_active() {
        return crate::active::Acceptance::Fallback(FallbackReason::ModeNotActive);
    }
    if !policy.allows(record.category) {
        return crate::active::Acceptance::Fallback(FallbackReason::CategoryDisabled);
    }
    let value = match &record.answer {
        crate::types::Answer::Choice { choice, confidence, .. }
            if confidence.is_finite()
                && (0.0..=1.0).contains(confidence)
                && !choice.trim().is_empty()
                && choice.trim().chars().count() <= crate::active::MAX_VALUE_CHARS =>
        {
            choice.trim().to_string()
        }
        _ => return crate::active::Acceptance::Fallback(FallbackReason::InvalidValue),
    };
    crate::active::Acceptance::Accepted(Box::new(ActiveDecision {
        category: record.category,
        question_id: record.question_id.clone(),
        value,
        confidence: record.answer.confidence().unwrap_or(0.0),
        response_model: record.response_model.clone(),
        request_id: bundle.ctx.request_id.clone(),
        turn: bundle.ctx.turn,
        decided_at: now,
    }))
}

fn assess_line_find_pair(
    records: &[&crate::types::DecisionRecord],
    bundle: &PreparedBundle,
    policy: &crate::active::ActivationPolicy,
    mode: JevMode,
    now: std::time::SystemTime,
) -> LineFindAcceptance {
    use crate::active::{ActiveDecision, FallbackReason};
    if !mode.allows_active() {
        return LineFindAcceptance::Fallback(FallbackReason::ModeNotActive);
    }
    if records.iter().any(|record| !policy.allows(record.category)) {
        return LineFindAcceptance::Fallback(FallbackReason::CategoryDisabled);
    }
    let where_record = records
        .iter()
        .find(|record| record.question_id == crate::search::WHERE_QUESTION_ID);
    let exists_record = records
        .iter()
        .find(|record| record.question_id == crate::search::EXISTS_QUESTION_ID);
    let (Some(where_record), Some(exists_record)) = (where_record, exists_record) else {
        return LineFindAcceptance::Fallback(FallbackReason::NoAnswer);
    };
    let crate::types::Answer::Choice { choice, confidence, .. } = &where_record.answer else {
        return LineFindAcceptance::Fallback(FallbackReason::InvalidValue);
    };
    let crate::types::Answer::Noul { noul } = &exists_record.answer else {
        return LineFindAcceptance::Fallback(FallbackReason::InvalidValue);
    };
    if !noul.is_finite() || !(0.0..=1.0).contains(noul) || choice.is_empty() {
        return LineFindAcceptance::Fallback(FallbackReason::InvalidValue);
    }
    let _assessment = crate::types::LineFindAssessment {
        where_line: Some(choice.clone()),
        existence_noul: Some(*noul),
        complete: true,
        correlated: true,
        fresh: true,
        windowed: false, // set by the bridge from presentation facts
        truncated: false,
    };
    LineFindAcceptance::Accepted(vec![
        ActiveDecision {
            category: where_record.category,
            question_id: where_record.question_id.clone(),
            value: where_record.answer.selected_value(),
            confidence: *confidence,
            response_model: where_record.response_model.clone(),
            request_id: bundle.ctx.request_id.clone(),
            turn: bundle.ctx.turn,
            decided_at: now,
        },
        ActiveDecision {
            category: exists_record.category,
            question_id: exists_record.question_id.clone(),
            value: exists_record.answer.selected_value(),
            // The Noul rides the legacy slot for record continuity only.
            confidence: *noul,
            response_model: exists_record.response_model.clone(),
            request_id: bundle.ctx.request_id.clone(),
            turn: bundle.ctx.turn,
            decided_at: now,
        },
    ])
}
