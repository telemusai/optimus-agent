//! Category 10: result sufficiency (AgentEnd).

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct ResultSufficiency;

impl super::CategoryEvaluator for ResultSufficiency {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::ResultSufficiency
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::AgentEnd]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        let Some(result_excerpt) = view.str_field("result_excerpt") else {
            return EvaluatorOutput::Skipped("no_result_observed".to_string());
        };
        let mut criteria = BTreeMap::new();
        for option in ["sufficient", "insufficient", "unknown"] {
            criteria.insert(option.to_string(), None);
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Is the bounded final result sufficient for the task? Result excerpt: {result_excerpt}"
                ),
                criteria,
            },
        }])
    }
}
