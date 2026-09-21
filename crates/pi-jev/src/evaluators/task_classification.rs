//! Category 1: task classification (TurnStart).

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct TaskClassification;

const OPTIONS: [&str; 5] = ["coding", "research", "debugging", "planning", "general"];

impl super::CategoryEvaluator for TaskClassification {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::TaskClassification
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnStart]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        let Some(task_text) = view.str_field("user_text_excerpt") else {
            return EvaluatorOutput::Skipped("no_task_text".to_string());
        };
        let mut criteria = BTreeMap::new();
        for option in OPTIONS {
            criteria.insert(option.to_string(), crate::types::EntryValue::Null);
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: Some(crate::types::EntryValue::Text(format!(
                    "Classify the current agent task. Bounded task excerpt: {task_text}"
                ))),
                criteria,
            },
        }])
    }
}
