//! Category 8: memory relevance (TurnStart). Requires memory signals the
//! bridge cannot always provide; absence is an explicit skip, never a guess.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};

pub struct MemoryRelevance;

impl super::CategoryEvaluator for MemoryRelevance {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::MemoryRelevance
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnStart]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        if let Some(questions) = crate::filtering::candidate_questions(&snapshot.state, self.category(), "memory_candidates") {
            return if questions.is_empty() {
                EvaluatorOutput::Skipped("no_eligible_candidates".to_string())
            } else {
                EvaluatorOutput::Questions(questions)
            };
        }
        let view = StateView::new(snapshot);
        let Some(memory_excerpt) = view.str_field("memory_excerpt") else {
            return EvaluatorOutput::Skipped("no_memory_state".to_string());
        };
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Noul {
                instructions: Some(crate::types::EntryValue::Text(format!(
                    "Is stored memory relevant to this task? Bounded memory summary: {memory_excerpt}"
                ))),
                criteria: Some(crate::types::NoulCriteria::text(
                    "Stored memory is relevant.",
                    "Stored memory is not relevant.",
                )),
            },
        }])
    }
}
