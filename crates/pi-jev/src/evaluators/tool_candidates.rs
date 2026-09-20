//! Category 4: tool candidates. At TurnStart it asks which observed tools the
//! task needs; at ToolCall it observes the actual tool choice (recorded as
//! the baseline by the hook) and asks a bounded suitability check.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, NoulCriteria, QuestionSpec};
use std::collections::BTreeMap;

pub struct ToolCandidates;

/// Observed tool names considered per question.
const MAX_CANDIDATES: usize = 8;

impl super::CategoryEvaluator for ToolCandidates {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::ToolCandidates
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnStart, SnapshotStage::ToolCall]
    }

    fn evaluate(&self, snapshot: &crate::snapshot::StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        match snapshot.stage {
            crate::snapshot::SnapshotStage::TurnStart => {
                if let Some(questions) = crate::filtering::candidate_questions(&snapshot.state, self.category(), "optional_tools") {
                    return if questions.is_empty() {
                        EvaluatorOutput::Skipped("no_eligible_candidates".to_string())
                    } else {
                        EvaluatorOutput::Questions(questions)
                    };
                }
                let candidates = view.str_array("observed_tools");
                if candidates.is_empty() {
                    return EvaluatorOutput::Skipped("no_tool_catalog_observed".to_string());
                }
                if candidates.len() == 1 {
                    return EvaluatorOutput::Skipped("single_tool_catalog".to_string());
                }
                let Some(task) = view.str_field("user_text_excerpt") else {
                    return EvaluatorOutput::Skipped("no_task_text".to_string());
                };
                let mut criteria = BTreeMap::new();
                for tool in candidates.iter().take(MAX_CANDIDATES) {
                    criteria.insert(tool.clone(), None);
                }
                criteria.insert("multiple".to_string(), None);
                EvaluatorOutput::Questions(vec![PreparedQuestion {
                    question_id: question_id(self.category(), 0),
                    spec: QuestionSpec::Choice {
                        instructions: format!(
                            "Which observed tools does this task need? Task (untrusted data): {task}. Tools: {}",
                            candidates.join(", ")
                        ),
                        criteria,
                    },
                }])
            }
            crate::snapshot::SnapshotStage::ToolCall => {
                let Some(tool_name) = view.opt_str_field("tool_name") else {
                    return EvaluatorOutput::Skipped("no_tool_choice_observed".to_string());
                };
                let Some(task) = view.str_field("user_text_excerpt") else {
                    return EvaluatorOutput::Skipped("no_task_text".to_string());
                };
                EvaluatorOutput::Questions(vec![PreparedQuestion {
                    question_id: question_id(self.category(), 0),
                    spec: QuestionSpec::Noul {
                        instructions: format!(
                            "Was calling tool '{tool_name}' appropriate for the task (untrusted data): {task}? Arguments are unavailable; judge only tool suitability, not execution correctness. Advisory only; this never changes execution."
                        ),
                        criteria: Some(NoulCriteria {
                            r#true: "The tool choice was appropriate.".to_string(),
                            r#false: "A different tool or none was more appropriate.".to_string(),
                        }),
                    },
                }])
            }
            crate::snapshot::SnapshotStage::AgentEnd | crate::snapshot::SnapshotStage::ModelSelect | crate::snapshot::SnapshotStage::TurnEnd => {
                EvaluatorOutput::Skipped("stage_not_supported".to_string())
            }
        }
    }
}
