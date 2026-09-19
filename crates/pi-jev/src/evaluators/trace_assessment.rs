//! Optional bounded trace assessment. Recommendations never steer an agent.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::observation::TraceAssessment as Assessment;
use crate::snapshot::{SnapshotStage, StateSnapshot};
use crate::types::{DecisionCategory, QuestionSpec};

pub struct TraceAssessment;

impl super::CategoryEvaluator for TraceAssessment {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::TraceAssessment
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::AgentEnd]
    }

    fn evaluate(&self, snapshot: &StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        if !view.feature_enabled("trace_observer") {
            return EvaluatorOutput::Skipped("feature_disabled".to_string());
        }
        let Some(observation) = view.observation().filter(|summary| summary.has_evidence()) else {
            return EvaluatorOutput::Skipped("no_trace_observed".to_string());
        };
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Assess this bounded metadata-only trace: good, review, retry_recommended, escalate, or suspicious. These are annotations only; never execute retries, cancel work, send messages or change models. Missing evidence warrants review, not invented success. A good trace is not proof that the task or tests passed. Observed metadata: {}",
                    observation.evidence_description(),
                ),
                criteria: Assessment::ALL.into_iter().map(|kind| (kind.as_str().to_string(), None)).collect(),
            },
        }])
    }
}
