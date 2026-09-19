//! Category 5: subagent requirement (TurnStart). Advisory only.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, NoulCriteria, QuestionSpec};

pub struct SubagentRequirement;

impl super::CategoryEvaluator for SubagentRequirement {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::SubagentRequirement
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnStart]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        let Some(task_text) = view.str_field("user_text_excerpt") else {
            return EvaluatorOutput::Skipped("no_task_text".to_string());
        };
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Noul {
                instructions: format!(
                    "Would delegating part of this task to a subagent plausibly help? Recommendation only; nothing is spawned. Bounded task excerpt: {task_text}"
                ),
                criteria: Some(NoulCriteria {
                    r#true: "A subagent could plausibly help.".to_string(),
                    r#false: "No subagent is warranted.".to_string(),
                }),
            },
        }])
    }
}
