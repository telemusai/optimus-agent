//! Category 4: tool candidates. At TurnStart it asks which observed tools the
//! task needs; at ToolCall it observes the actual tool choice (recorded as
//! the baseline by the hook) and asks a bounded suitability check.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::snapshot::SnapshotStage;
use crate::types::{DecisionCategory, QuestionSpec};
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
                // ROOT skill-audit fix: the Choice carries a no-match escape
                // ("none") and describes ONLY the assessed bounded subset.
                // Reserved option names are never overwritten by a genuine
                // tool: an observed tool literally named "none"/"multiple"
                // would make the model's answer ambiguous, so it is excluded
                // from this advisory question and the exclusion is disclosed.
                const RESERVED_OUTCOMES: [&str; 2] = ["none", "multiple"];
                let reserved_count = candidates.iter()
                    .filter(|tool| RESERVED_OUTCOMES.contains(&tool.as_str()))
                    .count();
                let assessable: Vec<&str> = candidates.iter()
                    .filter(|tool| !RESERVED_OUTCOMES.contains(&tool.as_str()))
                    .map(|tool| tool.as_str())
                    .collect();
                if assessable.is_empty() {
                    return EvaluatorOutput::Skipped("no_assessable_tools".to_string());
                }
                if assessable.len() == 1 {
                    return EvaluatorOutput::Skipped("single_tool_catalog".to_string());
                }
                let Some(task) = view.str_field("user_text_excerpt") else {
                    return EvaluatorOutput::Skipped("no_task_text".to_string());
                };
                let assessed: Vec<&str> = assessable.iter().take(MAX_CANDIDATES).copied().collect();
                let unassessed = assessable.len() - assessed.len();
                let mut criteria = BTreeMap::new();
                for tool in &assessed {
                    criteria.insert(tool.to_string(), crate::types::EntryValue::Null);
                }
                criteria.insert("none".to_string(), crate::types::EntryValue::Null);
                criteria.insert("multiple".to_string(), crate::types::EntryValue::Null);
                let mut instructions = format!(
                    "Which of these observed tools does this task need? Task (untrusted data): {task}. Assessed tools (bounded subset): {}.",
                    assessed.join(", ")
                );
                if unassessed > 0 {
                    instructions.push_str(&format!(
                        " {unassessed} further observed tool name(s) are outside this bounded question; not being asked about them is not an assessment of them."
                    ));
                }
                if reserved_count > 0 {
                    instructions.push_str(&format!(
                        " {reserved_count} observed tool name(s) named like the reserved outcomes (none/multiple) are excluded from this advisory question so every option stays unambiguous; that exclusion is a naming-collision rule, not an assessment of those tools."
                    ));
                }
                instructions.push_str(
                    " Choose \"none\" when none of the assessed tools is needed for this task excerpt: that means no need for the listed tools only, not that other tools cannot exist. Choose \"multiple\" when more than one assessed tool is needed. Advisory only: this never changes tool availability and never grants execution.",
                );
                EvaluatorOutput::Questions(vec![PreparedQuestion {
                    question_id: question_id(self.category(), 0),
                    spec: QuestionSpec::Choice {
                        instructions: Some(crate::types::EntryValue::Text(instructions)),
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
                        instructions: Some(crate::types::EntryValue::Text(format!(
                            "Was calling tool '{tool_name}' appropriate for the task (untrusted data): {task}? Arguments are unavailable; judge only tool suitability, not execution correctness. Advisory only; this never changes execution."
                        ))),
                        criteria: Some(crate::types::NoulCriteria::text(
                            "The tool choice was appropriate.",
                            "A different tool or none was more appropriate.",
                        )),
                    },
                }])
            }
            crate::snapshot::SnapshotStage::AgentEnd | crate::snapshot::SnapshotStage::ModelSelect | crate::snapshot::SnapshotStage::TurnEnd => {
                EvaluatorOutput::Skipped("stage_not_supported".to_string())
            }
        }
    }
}
