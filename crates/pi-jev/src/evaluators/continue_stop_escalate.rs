//! Category 9: continue/stop/escalate (AgentEnd). The hook records the actual
//! observed decision (the agent loop ended) as the baseline.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct ContinueStopEscalate;

impl super::CategoryEvaluator for ContinueStopEscalate {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::ContinueStopEscalate
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
        for option in ["continue", "stop", "escalate"] {
            criteria.insert(option.to_string(), None);
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Given the bounded result, should the agent continue, stop, or escalate? Result excerpt: {}",
                    view.str_field("result_excerpt").unwrap_or_default()
                ),
                criteria,
            },
        }])
    }
}
