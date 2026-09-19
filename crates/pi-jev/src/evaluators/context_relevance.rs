//! Category 7: context relevance (TurnStart). Score question over bounded
//! context stats; no raw transcript contents are ever sent.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};

pub struct ContextRelevance;

impl super::CategoryEvaluator for ContextRelevance {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::ContextRelevance
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnStart]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        if let Some(questions) = crate::filtering::candidate_questions(&snapshot.state, self.category(), "context_candidates") {
            return if questions.is_empty() {
                EvaluatorOutput::Skipped("no_eligible_candidates".to_string())
            } else {
                EvaluatorOutput::Questions(questions)
            };
        }
        let view = StateView::new(snapshot);
        let message_count = view.num_field("message_count").unwrap_or(0.0);
        if message_count <= 0.0 {
            return EvaluatorOutput::Skipped("no_context_messages".to_string());
        }
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Score {
                instructions: format!(
                    "How relevant is the current bounded conversation context to the task (message_count={message_count})? Bounded task excerpt: {}",
                    view.str_field("user_text_excerpt").unwrap_or_default()
                ),
                criteria: vec![
                    "irrelevant".to_string(),
                    "partially relevant".to_string(),
                    "highly relevant".to_string(),
                ],
            },
        }])
    }
}
