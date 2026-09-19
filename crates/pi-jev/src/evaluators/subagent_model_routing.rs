//! Category 6: eligible subagent/model routing (ModelSelect).
//!
//! ADVISORY ONLY by explicit coordination correction: this records suitable
//! candidates strictly inside the user-approved model allowlist and explicit
//! role assignments. It never switches the primary or a child model and never
//! executes routing actions. With no user-approved allowlist the category is
//! skipped; it never invents candidates.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::observation::{valid_model_id, RoutingMetrics, MAX_ROUTING_CANDIDATES};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct SubagentModelRouting;

impl super::CategoryEvaluator for SubagentModelRouting {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::SubagentModelRouting
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::ModelSelect]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        let allowlist = view.str_array("model_allowlist");
        if allowlist.is_empty() {
            return EvaluatorOutput::Skipped("no_model_allowlist".to_string());
        }
        let mut criteria = BTreeMap::new();
        for model in &allowlist {
            if model != "none" && valid_model_id(model) {
                criteria.insert(model.clone(), None);
            }
            if criteria.len() == MAX_ROUTING_CANDIDATES {
                break;
            }
        }
        if criteria.is_empty() {
            return EvaluatorOutput::Skipped("no_eligible_models".to_string());
        }
        let candidates = criteria.keys().cloned().collect::<Vec<_>>();
        let mut metrics = BTreeMap::<String, RoutingMetrics>::new();
        if let Some(rows) = snapshot
            .state
            .get("routing_metrics")
            .and_then(serde_json::Value::as_array)
        {
            for row in rows.iter().take(crate::snapshot::MAX_ITEMS) {
                let Ok(metric) = serde_json::from_value::<RoutingMetrics>(row.clone()) else {
                    continue;
                };
                if metric.is_valid() && criteria.contains_key(&metric.model) {
                    metrics.entry(metric.model.clone()).or_insert(metric);
                }
            }
        }
        for (model, description) in &mut criteria {
            *description = Some(match metrics.get(model) {
                Some(metric) => format!(
                    "Measured local sample only: {}; observed success rate={}. Missing measurements are unknown; no savings estimate.",
                    serde_json::to_string(metric).unwrap_or_default(),
                    metric.observed_success_rate().unwrap_or_default(),
                ),
                None => "No measured latency, cost or reliability evidence; unknown, not zero.".to_string(),
            });
        }
        criteria.insert(
            "none".to_string(),
            Some("No supported suitable candidate.".to_string()),
        );
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Advisory only: which user-approved candidate is suitable for the observed subagent role? Candidates: {}. Choose none when role or suitability evidence is missing. Measured samples do not guarantee future reliability or savings. This never switches any model, effort or role and never spawns a child.",
                    candidates.join(", ")
                ),
                criteria,
            },
        }])
    }
}
