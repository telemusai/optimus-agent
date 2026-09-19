//! Category 6: eligible subagent/model routing (ModelSelect).
//!
//! ADVISORY ONLY by explicit coordination correction: this records suitable
//! candidates strictly inside the user-approved model allowlist and explicit
//! role assignments. It never switches the primary or a child model and never
//! executes routing actions. With no user-approved allowlist the category is
//! skipped; it never invents candidates.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct SubagentModelRouting;

const MAX_CANDIDATES: usize = 8;

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
        for model in allowlist.iter().take(MAX_CANDIDATES) {
            criteria.insert(model.clone(), None);
        }
        criteria.insert("none".to_string(), None);
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Advisory only: which user-approved candidate is suitable for a subagent role? Candidates: {}. This never switches any model.",
                    allowlist.join(", ")
                ),
                criteria,
            },
        }])
    }
}
