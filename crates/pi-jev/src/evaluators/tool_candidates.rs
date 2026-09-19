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
                let candidates = view.str_array("observed_tools");
                if candidates.is_empty() {
                    return EvaluatorOutput::Skipped("no_tool_catalog_observed".to_string());
                }
                let mut criteria = BTreeMap::new();
                for tool in candidates.iter().take(MAX_CANDIDATES) {
                    criteria.insert(tool.clone(), None);
                }
                criteria.insert("multiple".to_string(), None);
                EvaluatorOutput::Questions(vec![PreparedQuestion {
                    question_id: question_id(self.category(), 0),
                    spec: QuestionSpec::Choice {
                        instructions: format!(
                            "Which observed tools does this task need? Tools: {}",
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
                EvaluatorOutput::Questions(vec![PreparedQuestion {
                    question_id: question_id(self.category(), 0),
                    spec: QuestionSpec::Noul {
                        instructions: format!(
                            "Was calling tool '{tool_name}' an appropriate choice for this step? Advisory only; the tool is already running and this never changes execution."
                        ),
                        criteria: Some(NoulCriteria {
                            r#true: "The tool choice was appropriate.".to_string(),
                            r#false: "A different tool or none was more appropriate.".to_string(),
                        }),
                    },
                }])
            }
            crate::snapshot::SnapshotStage::AgentEnd | crate::snapshot::SnapshotStage::ModelSelect => {
                EvaluatorOutput::Skipped("stage_not_supported".to_string())
            }
        }
    }
}
