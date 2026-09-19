//! Transport-agnostic JevObserver core (DESIGN.md section 4).
//!
//! Consumes event payloads as `serde_json::Value` DTOs (the adapter in
//! pi-coding-agent builds bounded values). In Compare: capture baselines
//! synchronously at the boundary, build one bundled request per snapshot,
//! enqueue asynchronously, and write correlated records when results arrive.
//! There is no decision-returning API: nothing Jev produces can re-enter the
//! agent loop. In Off: a single cheap mode check and nothing else — no
//! scheduling, no client work, no records.

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
}

impl Default for JevObserverConfig {
    fn default() -> Self {
        Self {
            mode_gate: Arc::new(|_| JevMode::Off),
            enabled_categories: HashSet::new(),
            scheduler: SchedulerConfig::default(),
            min_confidence: 0.7,
            on_terminal: None,
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
            config.scheduler.clone(), system_one, sink,
            Arc::new(move |session_id| mode_gate(Some(session_id)) == JevMode::Compare),
        );
        Arc::new(Self {
            config,
            correlator,
            scheduler,
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
        let mut status = self.scheduler.session_status(session_id)?;
        let skipped = self.skipped_categories.lock().unwrap_or_else(|p| p.into_inner());
        status["skipped_categories"] = serde_json::to_value(skipped.get(session_id).cloned().unwrap_or_default()).ok()?;
        Some(status)
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

    fn observe_snapshot(&self, _event_type: &str, payload: &Value, stage: SnapshotStage) {
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if session_id.is_empty() {
            return;
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
                );
            }
            return;
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
            Err(_) => return,
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
            );
        }
        if questions.is_empty() {
            // Every eligible category was skipped: nothing to ask.
            return;
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
            mode: "compare".to_string(),
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
        // Baselines and context were captured synchronously at the boundary;
        // the SystemOne call happens asynchronously in the scheduler.
        self.correlator.track(&ctx);
        let enqueued = self.scheduler.enqueue(request, ctx);
        if !enqueued {
            // Dropped requests are recorded by the scheduler sink.
            return;
        }
    }
}
