//! Category 11: first-pass verification (AgentEnd). Only ever a
//! recommendation; never proof that tests passed and never executed.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct FirstPassVerification;

impl super::CategoryEvaluator for FirstPassVerification {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::FirstPassVerification
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::AgentEnd]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        if view.str_field("result_excerpt").is_none() {
            return EvaluatorOutput::Skipped("no_result_observed".to_string());
        }
        let mut criteria = BTreeMap::new();
        for option in ["none", "rerun", "escalate", "verify"] {
            criteria.insert(option.to_string(), None);
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Recommend (never execute) a first-pass verification step. Result excerpt: {}",
                    view.str_field("result_excerpt").unwrap_or_default()
                ),
                criteria,
            },
        }])
    }
}
