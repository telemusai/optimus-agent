//! Category 3: tool requirement (TurnStart).

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
use std::collections::BTreeMap;

pub struct ToolRequirement;

const OPTIONS: [&str; 7] = ["none", "read", "search", "shell", "python", "delegate", "multiple"];

impl super::CategoryEvaluator for ToolRequirement {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::ToolRequirement
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
            criteria.insert(option.to_string(), None);
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "What tool support does this task need? Bounded task excerpt: {task_text}"
                ),
                criteria,
            },
        }])
    }
}
