//! Eleven legacy comparison categories plus opt-in failure and trace observations.
//! Categories 1-8 run at TurnStart, category 4 at ToolCall, 9-11 at AgentEnd,
//! and advisory routing at ModelSelect. Opt-in loop and failure observations
//! also run at TurnEnd. No evaluator grants runtime authority.
//!
//! Every evaluator produces typed questions with stable ids
//! `"<category>.<n>"` (n = 0-based sub-question) and returns an explicit
//! skipped reason when inputs are missing, insufficient or unsupported.
//! Nothing is ever fabricated.

pub mod continue_stop_escalate;
pub mod context_relevance;
pub mod complexity;
pub mod first_pass_verification;
pub mod memory_relevance;
pub mod result_sufficiency;
pub mod retry_classification;
pub mod trace_assessment;
pub mod subagent_model_routing;
pub mod subagent_requirement;
pub mod task_classification;
pub mod tool_candidates;
pub mod tool_requirement;

use std::sync::Arc;

use crate::snapshot::{SnapshotStage, StateSnapshot};
use crate::types::DecisionCategory;

/// One prepared question inside a snapshot bundle.
#[derive(Debug, Clone)]
pub struct PreparedQuestion {
    pub question_id: String,
    pub spec: crate::types::QuestionSpec,
}

/// Evaluator outcome: questions or an explicit skip. Never a fabricated
/// decision.
#[derive(Debug)]
pub enum EvaluatorOutput {
    Questions(Vec<PreparedQuestion>),
    Skipped(String),
}

/// Bounded state view over a snapshot.
pub struct StateView<'a> {
    snapshot: &'a StateSnapshot,
}

impl<'a> StateView<'a> {
    pub fn new(snapshot: &'a StateSnapshot) -> Self {
        Self { snapshot }
    }

    pub fn feature_enabled(&self, name: &str) -> bool {
        self.snapshot.state.get("features")
            .and_then(|features| features.get(name))
            .and_then(serde_json::Value::as_bool) == Some(true)
    }

    pub fn observation(&self) -> Option<crate::observation::TraceSummary> {
        let value = self.snapshot.state.get("observation")?;
        serde_json::from_value(value.clone()).ok()
    }

    pub fn str_field(&self, key: &str) -> Option<String> {
        self.snapshot
            .state
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    }

    pub fn opt_str_field(&self, key: &str) -> Option<String> {
        self.snapshot
            .state
            .get(key)
            .and_then(|v| if v.is_null() { None } else { v.as_str().map(str::to_string) })
            .filter(|v| !v.is_empty())
    }

    pub fn num_field(&self, key: &str) -> Option<f64> {
        self.snapshot.state.get(key).and_then(serde_json::Value::as_f64)
    }

    pub fn str_array(&self, key: &str) -> Vec<String> {
        self.snapshot
            .state
            .get(key)
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// All evaluators are synchronous, side-effect free and observe-only.
pub trait CategoryEvaluator: Send + Sync {
    fn category(&self) -> DecisionCategory;
    /// Boundaries this category asks questions at.
    fn boundaries(&self) -> &'static [SnapshotStage];
    fn evaluate(&self, snapshot: &StateSnapshot) -> EvaluatorOutput;
}

/// Stable question id `<category>.<n>`.
pub fn question_id(category: DecisionCategory, n: usize) -> String {
    format!("{}.{}", category.as_str(), n)
}

/// Registry of all evaluators in stable order.
pub fn all_evaluators() -> Vec<Arc<dyn CategoryEvaluator>> {
    vec![
        Arc::new(task_classification::TaskClassification),
        Arc::new(complexity::Complexity),
        Arc::new(tool_requirement::ToolRequirement),
        Arc::new(tool_candidates::ToolCandidates),
        Arc::new(subagent_requirement::SubagentRequirement),
        Arc::new(subagent_model_routing::SubagentModelRouting),
        Arc::new(context_relevance::ContextRelevance),
        Arc::new(memory_relevance::MemoryRelevance),
        Arc::new(continue_stop_escalate::ContinueStopEscalate),
        Arc::new(result_sufficiency::ResultSufficiency),
        Arc::new(first_pass_verification::FirstPassVerification),
        Arc::new(retry_classification::RetryClassification),
        Arc::new(trace_assessment::TraceAssessment),
    ]
}

/// Evaluators eligible at a snapshot stage.
pub fn for_boundary(stage: SnapshotStage) -> Vec<Arc<dyn CategoryEvaluator>> {
    all_evaluators()
        .into_iter()
        .filter(|evaluator| evaluator.boundaries().contains(&stage))
        .collect()
}
