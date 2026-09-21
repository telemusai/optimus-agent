use std::sync::Arc;

use pi_jev::compaction::{
    decisions, estimate_tokens, prepare, truncated_result, CallAction, CompactionConfig,
    CompactionPlan, CompactionSkip, HistoryExcerpt, PairCandidate, MAX_REQUEST_BYTES,
    MAX_STATE_BYTES,
};
use pi_jev::{
    Answer, DecisionCategory, DecisionOutcome, DecisionRecord, JevLimits, JevMode, JevStats,
    JevSystemOne, MockJevTransport, SecretString, SystemOne, SystemOneRequest,
};

fn candidates(count: usize) -> Vec<PairCandidate> {
    (1..=count)
        .map(|index| PairCandidate {
            id: format!("t{index}"),
            tool: "read_file".into(),
            result_chars: 10_000,
            allow_drop_call: true,
        })
        .collect()
}
fn plan() -> CompactionPlan {
    prepare(&[], &candidates(2), &CompactionConfig::default()).unwrap()
}
fn outcome(plan: &CompactionPlan, call: f64, result: f64) -> DecisionOutcome {
    DecisionOutcome {
        records: plan
            .questions
            .keys()
            .map(|id| DecisionRecord {
                question_id: id.clone(),
                category: DecisionCategory::ContextRelevance,
                answer: Answer::Noul {
                    noul: if id.contains(".call_") { call } else { result },
                },
                response_model: Some("mock".into()),
                requested_model: "jev-latest".into(),
                applied: false,
            })
            .collect(),
        skips: vec![],
        response_model: Some("mock".into()),
        usage: Default::default(),
        applied: false,
        attempts: 1,
        server_request_id: None,
    }
}

#[test]
fn config_defaults_and_all_requested_parameters_validate() {
    let config = CompactionConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.keep_threshold, 0.5);
    assert_eq!(config.preserve_recent_messages, 6);
    assert_eq!(config.max_state_tokens, 25_000);
    assert_eq!(config.max_request_tokens, 30_000);
    assert_eq!(config.truncate_head_chars, 300);
    assert_eq!(config.minimum_reduction_ratio, 0.25);
    for config in [
        CompactionConfig {
            keep_threshold: f64::NAN,
            ..Default::default()
        },
        CompactionConfig {
            keep_threshold: 1.1,
            ..Default::default()
        },
        CompactionConfig {
            preserve_recent_messages: 1025,
            ..Default::default()
        },
        CompactionConfig {
            max_state_tokens: 0,
            ..Default::default()
        },
        CompactionConfig {
            max_request_tokens: 30_001,
            ..Default::default()
        },
        CompactionConfig {
            truncate_head_chars: 4097,
            ..Default::default()
        },
        CompactionConfig {
            minimum_reduction_ratio: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert_eq!(config.validate(), Err(CompactionSkip::InvalidConfig));
    }
}

#[test]
fn all_or_nothing_decisions_keep_truncate_or_delete() {
    let plan = plan();
    assert!(decisions(&plan, &outcome(&plan, 0.1, 0.9))
        .unwrap()
        .iter()
        .all(|d| d.action == CallAction::Keep));
    assert!(decisions(&plan, &outcome(&plan, 0.9, 0.1))
        .unwrap()
        .iter()
        .all(|d| d.action == CallAction::TruncateResult));
    assert!(decisions(&plan, &outcome(&plan, 0.1, 0.1))
        .unwrap()
        .iter()
        .all(|d| d.action == CallAction::DropCall));
    let mut candidates = candidates(1);
    candidates[0].allow_drop_call = false;
    let plan = prepare(&[], &candidates, &CompactionConfig::default()).unwrap();
    assert_eq!(
        decisions(&plan, &outcome(&plan, 0.1, 0.1)).unwrap()[0].action,
        CallAction::TruncateResult
    );
}

#[test]
fn uncertain_missing_extra_duplicate_nonfinite_and_wrong_type_refuse_entire_patch() {
    let plan = plan();
    assert_eq!(
        decisions(&plan, &outcome(&plan, 0.51, 0.1)),
        Err(CompactionSkip::UncertainAnswer)
    );
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert_eq!(
            decisions(&plan, &outcome(&plan, invalid, 0.1)),
            Err(CompactionSkip::InvalidAnswers)
        );
    }
    let good = outcome(&plan, 0.1, 0.1);
    let mut missing = good.clone();
    missing.records.pop();
    let mut extra = good.clone();
    extra.records[0].question_id = "compaction.call_t99".into();
    let mut duplicate = good.clone();
    duplicate.records[0] = duplicate.records[1].clone();
    let mut model = good.clone();
    model.response_model = None;
    let mut skipped = good.clone();
    skipped.skips.push(("".into(), "timeout"));
    for bad in [missing, extra, duplicate, model, skipped] {
        assert_eq!(decisions(&plan, &bad), Err(CompactionSkip::InvalidAnswers));
    }
}

#[test]
fn state_is_redacted_bounded_and_contains_every_candidate() {
    let history = vec![HistoryExcerpt {
        index: 0,
        role: "user",
        pinned: true,
        text: "TYPESAFE_API_KEY=sk-live-never-send-this Authorization: Basic c2VjcmV0".into(),
    }];
    let plan = prepare(&history, &candidates(8), &CompactionConfig::default()).unwrap();
    let json = serde_json::to_string(&plan.state).unwrap();
    assert!(!json.contains("never-send-this"));
    assert!(!json.contains("c2VjcmV0"));
    assert!(json.contains("[redacted]"));
    assert!(json.len() <= MAX_STATE_BYTES);
    assert_eq!(plan.state["candidates"].as_array().unwrap().len(), 8);
    assert_eq!(plan.questions.len(), 16);
    for questions in &plan.batches {
        let request = SystemOneRequest::new(plan.state.clone(), questions.clone());
        let request = serde_json::to_string(&request).unwrap();
        assert!(request.len() <= MAX_REQUEST_BYTES);
        assert!(estimate_tokens(&request) <= CompactionConfig::default().max_request_tokens);
    }
}

#[test]
fn token_batched_questions_keep_pairs_together_and_never_exceed_budget() {
    let base = plan();
    let first_pair = prepare(&[], &candidates(1), &CompactionConfig::default()).unwrap();
    let request = SystemOneRequest::new(base.state.clone(), first_pair.questions.clone());
    let tokens = estimate_tokens(&serde_json::to_string(&request).unwrap());
    let config = CompactionConfig {
        max_request_tokens: tokens,
        ..Default::default()
    };
    let plan = prepare(&[], &candidates(2), &config).unwrap();
    assert_eq!(plan.batches.len(), 2);
    assert!(plan.batches.iter().all(|batch| batch.len() == 2));
    assert_eq!(
        prepare(
            &[],
            &candidates(2),
            &CompactionConfig {
                max_state_tokens: 64,
                ..Default::default()
            }
        )
        .unwrap_err(),
        CompactionSkip::StateLimit
    );
    assert_eq!(
        prepare(
            &[],
            &candidates(2),
            &CompactionConfig {
                max_request_tokens: 128,
                ..Default::default()
            }
        )
        .unwrap_err(),
        CompactionSkip::RequestLimit
    );
}

#[test]
fn state_fitting_and_hard_input_limits_fail_safely() {
    let history: Vec<_> = (0..30)
        .map(|index| HistoryExcerpt {
            index,
            role: "assistant",
            pinned: false,
            text: "ordinary prose ".repeat(100),
        })
        .collect();
    let plan = prepare(&history, &candidates(1), &CompactionConfig::default()).unwrap();
    assert_ne!(plan.state_stage, "excerpts");
    assert_eq!(
        prepare(&[], &[], &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::NoCandidates
    );
    assert_eq!(
        prepare(&[], &candidates(9), &CompactionConfig::default()).unwrap_err(),
        CompactionSkip::InputLimit
    );
}

#[test]
fn result_truncation_is_unicode_safe_bounded_and_idempotent() {
    let original = "é界🙂".repeat(2000);
    let truncated = truncated_result(&original, 7).unwrap();
    assert!(truncated.starts_with("é界🙂é界🙂é\n"));
    assert!(truncated.contains("5993 characters omitted"));
    assert!(!truncated.contains("re-run"));
    assert!(truncated_result(&truncated, 7).is_none());
    assert!(truncated_result("short", 7).is_none());
    assert!(truncated_result(&original, 0)
        .unwrap()
        .starts_with("[Jev compaction:"));
}

#[tokio::test]
async fn prepared_questions_use_existing_mock_systemone_transport_offline() {
    let plan = plan();
    let transport = Arc::new(MockJevTransport::all_valid());
    let client = JevSystemOne::new(
        JevMode::Active,
        SecretString::new("synthetic-only"),
        transport,
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    let bundle = pi_jev::DecisionBundle {
        session_id: "fixture".into(),
        turn: 1,
        stage: "compaction".into(),
        state: plan.state.clone(),
        model: "jev-latest".into(),
        questions: plan.questions.clone(),
        question_categories: plan
            .questions
            .keys()
            .map(|id| (id.clone(), DecisionCategory::ContextRelevance))
            .collect(),
    };
    let result = client.decide(bundle).await;
    assert_eq!(result.records.len(), plan.questions.len());
    assert!(result.skips.is_empty());
    assert_eq!(result.attempts, 1);
}
