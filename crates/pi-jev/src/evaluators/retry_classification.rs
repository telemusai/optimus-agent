//! Optional error classification. This never changes replay, backoff or retry limits.

use crate::evaluators::{question_id, EvaluatorOutput, PreparedQuestion, StateView};
use crate::observation::RetryFailureKind;
use crate::snapshot::{SnapshotStage, StateSnapshot};
use crate::types::{DecisionCategory, QuestionSpec};

pub struct RetryClassification;

impl super::CategoryEvaluator for RetryClassification {
    fn category(&self) -> DecisionCategory {
        DecisionCategory::RetryClassification
    }

    fn boundaries(&self) -> &'static [SnapshotStage] {
        &[SnapshotStage::TurnEnd, SnapshotStage::AgentEnd]
    }

    fn evaluate(&self, snapshot: &StateSnapshot) -> EvaluatorOutput {
        let view = StateView::new(snapshot);
        if !view.feature_enabled("retry_classification") {
            return EvaluatorOutput::Skipped("feature_disabled".to_string());
        }
        let Some(observation) = view
            .observation()
            .filter(|summary| summary.failure_kind.is_some())
        else {
            return EvaluatorOutput::Skipped("no_failure_observed".to_string());
        };
        EvaluatorOutput::Questions(vec![PreparedQuestion {
            question_id: question_id(self.category(), 0),
            spec: QuestionSpec::Choice {
                instructions: format!(
                    "Classify the observed failure, not whether to execute a retry. Use unknown when metadata is insufficient. Transient does not mean replay is safe: interrupted or partial responses remain subject to host replay protection. Do not change backoff, retry limits, permissions or credentials. Observed metadata: {}",
                    observation.evidence_description(),
                ),
                criteria: RetryFailureKind::ALL.into_iter().map(|kind| (kind.as_str().to_string(), None)).collect(),
            },
        }])
    }
}
