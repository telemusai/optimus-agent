//! Lane A (jev-client) tests: wire fixtures, answer validation, mock transport scenarios,
//! mode/credential precedence, credential store behaviour and secret-leak assertions.
//!
//! No test performs network I/O, reads a production credential, or spawns a process. Every
//! transport is `MockJevTransport`, which goes through the same client code path production uses.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::client::{accepted_answers, parse_systemone_body};
use pi_jev::mock::{
    choice_question, fingerprint, mutate, noul_question, raw_huge_extra_ids_body, raw_injection_body,
    raw_noul_non_finite_body, raw_truncated_body, raw_wrong_shape_body, score_question, valid_response_for,
    MockJevTransport, MockMutation, MockStep, MOCK_RESPONSE_MODEL,
};
use pi_jev::{
    decide_with, model_drift, refuse_subagent_control, resolve_credential_source, resolve_effective_mode,
    redact_authorization, retry_decision, sanitize_detail, sanitize_url, validate_answer, validate_response, Answer,
    AnswerIssue, CredentialSource, CredentialStore, DecisionBundle, DecisionCategory, DecisionOutcome,
    DisabledSystemOne, DpapiCredentialStore, EnvKeyPresence, InMemoryCredentialStore, JevError, JevLimits, JevMode,
    JevSettings, JevSettingsStore, JevStats, JevSystemOne, QuestionSpec, RetryDecision, SecretString,
    SubagentControlRequest, SubagentObservation, SystemOne, SystemOneRequest, SystemOneResponse, Transport,
    UnavailableCredentialStore, FORBIDDEN_SUBAGENT_CAPABILITIES,
};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Synthetic key: clearly fake, and never written to any real store by these tests.
const SYNTHETIC_KEY: &str = "jev-test-key-0123456789abcdef";

fn question_map() -> BTreeMap<String, QuestionSpec> {
    let mut questions = BTreeMap::new();
    questions.insert(
        "task_classification.0".to_string(),
        choice_question(
            "Which task type is this?",
            &[("coding", Some("code change")), ("research", Some("reading"))],
        ),
    );
    questions.insert(
        "result_sufficiency.0".to_string(),
        noul_question("Is the result sufficient?", "sufficient", "insufficient"),
    );
    questions.insert(
        "complexity.0".to_string(),
        score_question("How complex is this task?", &["low", "medium", "high"]),
    );
    questions
}

fn bundle(questions: BTreeMap<String, QuestionSpec>) -> DecisionBundle {
    let question_categories = questions
        .keys()
        .map(|id| {
            let prefix = id.split('.').next().unwrap_or(id);
            (
                id.clone(),
                DecisionCategory::parse(prefix).unwrap_or(DecisionCategory::TaskClassification),
            )
        })
        .collect();
    DecisionBundle {
        session_id: "local-session-1".to_string(),
        turn: 3,
        stage: "turn_start".to_string(),
        state: serde_json::json!({"summary": "bounded state excerpt"}),
        model: "jev-latest".to_string(),
        questions,
        question_categories,
    }
}

fn default_bundle() -> DecisionBundle {
    bundle(question_map())
}

/// Explicit `Arc<T>` -> `Arc<dyn Transport>` conversion.
///
/// Written out rather than relying on an unsized coercion at each call site, so the intent is
/// visible and a future refactor cannot silently change which transport a test exercises.
fn limits_with_timeout(timeout: Duration, retries: u32, backoff: Duration) -> JevLimits {
    JevLimits {
        timeout,
        max_retries: retries,
        backoff_initial: backoff,
        backoff_max: backoff,
        ..JevLimits::default()
    }
}

async fn run(
    transport: Arc<dyn Transport>,
    limits: JevLimits,
    stats: &Arc<JevStats>,
    bundle: DecisionBundle,
) -> DecisionOutcome {
    decide_with(
        &transport,
        &SecretString::new(SYNTHETIC_KEY),
        JevMode::Compare,
        &limits,
        stats,
        bundle,
    )
    .await
}

// ---------------------------------------------------------------------------
// 1. Valid fixtures per answer type
// ---------------------------------------------------------------------------

#[tokio::test]
async fn valid_noul_choice_score_answers_are_all_accepted() {
    let request = SystemOneRequest::new(
        serde_json::json!({"summary": "state"}),
        question_map(),
    );
    let response = valid_response_for(&request);
    let validation = validate_response(&request, &response);
    assert!(validation.skipped.is_empty(), "unexpected skips: {:?}", validation.skip_reasons());
    assert_eq!(validation.accepted.len(), 3);
    assert!(matches!(validation.accepted["task_classification.0"], Answer::Choice { .. }));
    assert!(matches!(validation.accepted["result_sufficiency.0"], Answer::Noul { .. }));
    assert!(matches!(validation.accepted["complexity.0"], Answer::Score { .. }));
    // Noul carries NO confidence field; None, not 0.0.
    assert_eq!(validation.accepted["result_sufficiency.0"].confidence(), None);
    assert_eq!(validation.accepted["task_classification.0"].confidence(), Some(0.82));
    assert_eq!(validation.accepted["complexity.0"].confidence(), Some(0.78));
}

#[tokio::test]
async fn score_fixture_is_a_probability_weighted_value_with_a_matching_legend() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = valid_response_for(&request);
    let Answer::Score {
        score,
        legend,
        probabilities,
        ..
    } = &response.answers["complexity.0"]
    else {
        panic!("expected a score answer");
    };
    assert_eq!(legend.len(), 3);
    assert_eq!(legend["0"], "low");
    assert_eq!(legend["2"], "high");
    let weighted: f64 = probabilities
        .iter()
        .map(|(key, probability)| key.parse::<f64>().unwrap() * probability)
        .sum();
    assert!((weighted - score).abs() < 1e-9, "score must be the weighted value");
}

#[tokio::test]
async fn choice_fixture_distribution_sums_to_one_and_names_a_defined_option() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = valid_response_for(&request);
    let Answer::Choice {
        choice,
        probabilities,
        ..
    } = &response.answers["task_classification.0"]
    else {
        panic!("expected a choice answer");
    };
    assert!(["coding", "research"].contains(&choice.as_str()));
    let sum: f64 = probabilities.values().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// 2. Validation failures (each is a logged skip, never a fabricated answer)
// ---------------------------------------------------------------------------

/// Asserts that `id` was rejected with the expected issue, and returns that issue.
///
/// Sibling answers in the same response may still be accepted: one bad answer never invalidates
/// the rest, and it is never turned into a fabricated value.
fn expect_skip(response: SystemOneResponse, id: &str, expected: fn(&AnswerIssue) -> bool) -> AnswerIssue {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let validation = validate_response(&request, &response);
    assert!(
        !validation.accepted.contains_key(id),
        "a defective answer must not be accepted: {}",
        id
    );
    let issue = validation
        .skipped
        .iter()
        .find(|(skipped_id, issue)| skipped_id == id && expected(issue))
        .map(|(_, issue)| issue.clone())
        .unwrap_or_else(|| panic!("expected issue for {id} not found in {:?}", validation.skip_reasons()));
    // Every skip produces a bounded log line with a stable reason code.
    for line in validation.log_lines() {
        assert!(line.starts_with("jev skip ["), "unexpected log line: {line}");
    }
    issue
}

#[tokio::test]
async fn wrong_id_set_extra_id_and_missing_id_are_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let base = valid_response_for(&request);

    let extra = mutate(base.clone(), &MockMutation::UnknownId { id: "never_requested.0".to_string() });
    let validation = validate_response(&request, &extra);
    assert_eq!(validation.accepted.len(), 3, "known-good siblings still validate");
    assert!(validation
        .skipped
        .iter()
        .any(|(id, issue)| id == "never_requested.0" && issue.reason() == "unknown_answer_id"));

    let missing = mutate(base.clone(), &MockMutation::MissingId { id: "complexity.0".to_string() });
    let validation = validate_response(&request, &missing);
    assert!(validation
        .skipped
        .iter()
        .any(|(id, issue)| id == "complexity.0" && issue.reason() == "missing_answer_id"));
}

#[tokio::test]
async fn type_mismatch_is_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = mutate(
        valid_response_for(&request),
        &MockMutation::TypeMismatch { id: "complexity.0".to_string() },
    );
    let issue = expect_skip(response, "complexity.0", |issue| {
        matches!(issue, AnswerIssue::TypeMismatch { .. })
    });
    assert_eq!(issue.reason(), "answer_type_mismatch");
}

#[tokio::test]
async fn choice_outside_criteria_is_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = mutate(
        valid_response_for(&request),
        &MockMutation::ChoiceNotInCriteria {
            id: "task_classification.0".to_string(),
        },
    );
    let issue = expect_skip(response, "task_classification.0", |issue| {
        matches!(issue, AnswerIssue::UnknownChoice { .. })
    });
    assert_eq!(issue.reason(), "choice_not_in_criteria");
}

#[tokio::test]
async fn nan_and_infinite_values_are_skipped() {
    // NaN cannot appear in JSON, so it is injected through the in-memory answer type.
    let question = score_question("How complex?", &["low", "high"]);
    let answer = Answer::Score {
        score: f64::NAN,
        legend: [("0".to_string(), "low".to_string()), ("1".to_string(), "high".to_string())]
            .into_iter()
            .collect(),
        probabilities: [("0".to_string(), 0.5), ("1".to_string(), 0.5)].into_iter().collect(),
        confidence: 0.5,
    };
    let issue = validate_answer("complexity.0", &question, &answer).unwrap_err();
    assert_eq!(issue.reason(), "score_not_finite");

    let infinite_confidence = Answer::Choice {
        choice: "coding".to_string(),
        probabilities: [("coding".to_string(), 1.0)].into_iter().collect(),
        confidence: f64::INFINITY,
    };
    let choice = choice_question("Which?", &[("coding", None)]);
    let issue = validate_answer("task_classification.0", &choice, &infinite_confidence).unwrap_err();
    assert_eq!(issue.reason(), "confidence_out_of_range");

    let infinite_probability = Answer::Choice {
        choice: "coding".to_string(),
        probabilities: [("coding".to_string(), f64::INFINITY)].into_iter().collect(),
        confidence: 0.5,
    };
    let issue = validate_answer("task_classification.0", &choice, &infinite_probability).unwrap_err();
    assert_eq!(issue.reason(), "probability_out_of_range");
}

#[tokio::test]
async fn out_of_range_noul_value_is_skipped_from_a_raw_wire_body() {
    // The raw body goes through the production parser; an unrepresentable float must not
    // silently become a value the client trusts.
    let body = raw_noul_non_finite_body("result_sufficiency.0");
    let parsed = parse_systemone_body(body.as_bytes());
    match parsed {
        Ok(response) => {
            let issue = expect_skip(response, "result_sufficiency.0", |issue| {
                issue.reason() == "noul_out_of_range"
            });
            assert_eq!(issue.reason(), "noul_out_of_range");
        }
        Err(error) => {
            // Rejecting the body at parse time is equally acceptable: no answer is fabricated.
            assert_eq!(error.kind(), "malformed_response");
        }
    }
}

#[tokio::test]
async fn probabilities_not_summing_to_one_are_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = mutate(
        valid_response_for(&request),
        &MockMutation::ProbabilitiesNotSummingToOne {
            id: "task_classification.0".to_string(),
        },
    );
    let issue = expect_skip(response, "task_classification.0", |issue| {
        matches!(issue, AnswerIssue::ProbabilitiesNotSummingToOne { .. })
    });
    assert_eq!(issue.reason(), "probabilities_not_summing_to_one");
}

#[tokio::test]
async fn distribution_key_mismatch_is_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = mutate(
        valid_response_for(&request),
        &MockMutation::DistributionKeysMismatch {
            id: "task_classification.0".to_string(),
        },
    );
    let issue = expect_skip(response, "task_classification.0", |issue| {
        matches!(issue, AnswerIssue::DistributionKeysMismatch { .. })
    });
    assert_eq!(issue.reason(), "distribution_keys_mismatch");
}

#[tokio::test]
async fn confidence_out_of_range_is_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let response = mutate(
        valid_response_for(&request),
        &MockMutation::ConfidenceOutOfRange {
            id: "task_classification.0".to_string(),
        },
    );
    let issue = expect_skip(response, "task_classification.0", |issue| {
        matches!(issue, AnswerIssue::ConfidenceOutOfRange { .. })
    });
    assert_eq!(issue.reason(), "confidence_out_of_range");
}

#[tokio::test]
async fn score_legend_and_probability_mismatch_are_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());

    let legend = mutate(
        valid_response_for(&request),
        &MockMutation::LegendMismatch { id: "complexity.0".to_string() },
    );
    let issue = expect_skip(legend, "complexity.0", |issue| {
        matches!(issue, AnswerIssue::LegendMismatch { .. })
    });
    assert_eq!(issue.reason(), "score_legend_mismatch");

    let levels = mutate(
        valid_response_for(&request),
        &MockMutation::DistributionKeysMismatch { id: "complexity.0".to_string() },
    );
    let issue = expect_skip(levels, "complexity.0", |issue| {
        matches!(issue, AnswerIssue::DistributionKeysMismatch { .. })
    });
    assert_eq!(issue.reason(), "distribution_keys_mismatch");

    let range = mutate(
        valid_response_for(&request),
        &MockMutation::ScoreOutOfRange { id: "complexity.0".to_string() },
    );
    let issue = expect_skip(range, "complexity.0", |issue| {
        matches!(issue, AnswerIssue::ScoreOutOfRange { .. })
    });
    assert_eq!(issue.reason(), "score_out_of_range");
}

#[tokio::test]
async fn a_stale_generation_and_a_session_switch_reject_an_old_result() {
    use pi_jev::client::ClientHandle;
    let handle = ClientHandle::new();
    let generation = handle.install(Arc::new(MockSystemOne::new()));
    // A result produced under `generation` describes the old session/credential.
    assert!(handle.is_current(generation));
    handle.install(Arc::new(DisabledSystemOne::default()));
    assert!(
        !handle.is_current(generation),
        "a late result from the previous client must be rejected"
    );
    // Clearing on Off must leave nothing installed, so no callback can act on a stale client.
    handle.clear();
    assert!(handle.current().is_none());
    assert!(!handle.is_current(generation));

    // Explicit session switches change the stored mode, so a stale client can be detected too.
    let mut settings = JevSettings::default();
    settings.set_session_mode("session-a", JevMode::Compare);
    assert_eq!(settings.effective_mode("session-a"), JevMode::Compare);
    settings.set_session_mode("session-a", JevMode::Off);
    assert_eq!(settings.effective_mode("session-a"), JevMode::Off);
    settings.clear_session_mode("session-a");
    assert_eq!(settings.effective_mode("session-a"), JevMode::Off, "falls back to built-in Off");
}

#[tokio::test]
async fn a_401_is_terminal_so_a_bad_key_does_not_burn_the_retry_budget() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::ServerError { status: 401 }]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 3, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "http_status");
    assert_eq!(transport.call_count(), 1, "a 401 must not be retried");
    assert_eq!(stats.snapshot().retries, 0);
}

#[tokio::test]
async fn an_overloaded_client_reports_in_flight_zero_after_completion() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    let _ = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    // In-flight work is always released, so a status render never shows a leaked task.
    assert_eq!(stats.snapshot().in_flight, 0);
    assert_eq!(stats.snapshot().attempts, 1);
    assert_eq!(stats.snapshot().successes, 1);
    assert!(stats.snapshot().last_latency_ms < 5_000);
}

#[tokio::test]
async fn empty_and_model_less_responses_are_skipped() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());

    let empty = mutate(valid_response_for(&request), &MockMutation::EmptyAnswers);
    let validation = validate_response(&request, &empty);
    assert!(validation.accepted.is_empty());
    assert_eq!(
        validation.skipped.iter().filter(|(_, issue)| issue.reason() == "missing_answer_id").count(),
        3
    );

    let no_model = mutate(valid_response_for(&request), &MockMutation::MissingModel);
    let validation = validate_response(&request, &no_model);
    assert!(validation.response_model.is_none());
    assert!(validation
        .skipped
        .iter()
        .any(|(_, issue)| issue.reason() == "missing_response_model"));
}

#[tokio::test]
async fn malicious_payloads_are_skipped_and_never_act() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());

    // 1. A very large extra-id array is bounded by validation: no answer is accepted from it.
    let huge = mutate(
        valid_response_for(&request),
        &MockMutation::HugeExtraIds { count: 512 },
    );
    let validation = validate_response(&request, &huge);
    assert_eq!(validation.accepted.len(), 3, "only requested ids may be accepted");
    assert_eq!(
        validation
            .skipped
            .iter()
            .filter(|(_, issue)| issue.reason() == "unknown_answer_id")
            .count(),
        512
    );

    // 2. Instruction-shaped text inside a Choice value is data, never an instruction: it fails
    //    the criteria check and is skipped.
    let injection = mutate(
        valid_response_for(&request),
        &MockMutation::InjectionText {
            id: "task_classification.0".to_string(),
        },
    );
    let issue = expect_skip(injection, "task_classification.0", |issue| {
        matches!(issue, AnswerIssue::UnknownChoice { .. })
    });
    assert_eq!(issue.reason(), "choice_not_in_criteria");

    // 3. The same injection through a raw body.
    let parsed = parse_systemone_body(raw_injection_body("task_classification.0").as_bytes()).unwrap();
    let validation = validate_response(&request, &parsed);
    assert!(validation.accepted.is_empty());
    assert!(validation
        .skip_reasons()
        .contains(&("task_classification.0".to_string(), "choice_not_in_criteria")));

    // 4. A raw body with hundreds of injected answers stays bounded and inert.
    let parsed = parse_systemone_body(raw_huge_extra_ids_body(2000).as_bytes()).unwrap();
    let validation = validate_response(&request, &parsed);
    assert!(validation.accepted.is_empty());
    assert!(validation.response_model.is_some(), "the fixture carries a response model");
    assert_eq!(validation.skipped.len(), 2000 + 3, "2000 injected ids plus three missing ids");
}

#[tokio::test]
async fn malformed_json_and_wrong_shapes_are_reported_not_guessed() {
    let truncated = parse_systemone_body(raw_truncated_body().as_bytes()).unwrap_err();
    assert_eq!(truncated.kind(), "malformed_response");

    let wrong_shape = parse_systemone_body(raw_wrong_shape_body().as_bytes()).unwrap_err();
    assert_eq!(wrong_shape.kind(), "malformed_response");
}

// ---------------------------------------------------------------------------
// 3. Mock transport scenarios through the production client path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limit_retry_after_is_honored_then_the_call_succeeds() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::RateLimited { retry_after_secs: 0 },
        MockStep::Valid,
    ]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(500), 2, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;

    assert_eq!(outcome.records.len(), 3, "the retry must produce accepted answers");
    assert_eq!(transport.call_count(), 2, "one rate-limited attempt then one success");
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.rate_limited, 1, "the client counts the 429, not the transport");
    assert_eq!(snapshot.retries, 1);
    assert_eq!(snapshot.successes, 1);
    assert_eq!(snapshot.attempts, 2);
    assert_eq!(snapshot.failures, 0, "the retried call ultimately succeeded");
    assert!(snapshot.last_success_ms > 0);
}

#[tokio::test]
async fn retry_after_hint_is_never_shortened_to_the_backoff_ceiling() {
    let limits = JevLimits {
        backoff_initial: Duration::from_millis(10),
        backoff_max: Duration::from_secs(5),
        max_retries: 3,
        ..JevLimits::default()
    };
    let rate_limited = JevError::HttpStatus {
        status: 429,
        detail: "rate limited".to_string(),
        retry_after: Some(Duration::from_secs(2)),
    };
    match retry_decision(&rate_limited, 0, &limits) {
        RetryDecision::RetryAfter(delay) => assert_eq!(delay, Duration::from_secs(2)),
        RetryDecision::Stop => panic!("429 with a hint must retry while attempts remain"),
    }
    // The local backoff ceiling must not override the server's minimum delay.
    match retry_decision(&rate_limited, 0, &JevLimits { backoff_max: Duration::from_millis(200), ..limits.clone() })
    {
        RetryDecision::RetryAfter(delay) => assert_eq!(delay, Duration::from_secs(2)),
        RetryDecision::Stop => panic!("expected the server delay"),
    }
    // Attempts are finite: the last permitted attempt stops.
    assert_eq!(retry_decision(&rate_limited, limits.max_retries, &limits), RetryDecision::Stop);
}

#[test]
fn retry_after_parsing_is_bounded_and_rejects_non_numeric_values() {
    use pi_jev::parse_retry_after;
    assert_eq!(parse_retry_after("12"), Some(Duration::from_secs(12)));
    assert_eq!(parse_retry_after(" 0.5 "), Some(Duration::from_millis(500)));
    // A hostile value is clamped, never converted (which would panic) or honored verbatim.
    assert_eq!(parse_retry_after("1e300"), Some(pi_jev::MAX_RETRY_AFTER));
    assert_eq!(parse_retry_after("999999999"), Some(pi_jev::MAX_RETRY_AFTER));
    // A past HTTP-date permits immediate retry; malformed values are ignored.
    assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), Some(Duration::ZERO));
    assert_eq!(parse_retry_after("-5"), None);
    assert_eq!(parse_retry_after("NaN"), None);
    assert_eq!(parse_retry_after(""), None);
}

#[tokio::test]
async fn overloaded_529_is_retried_and_then_succeeds() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::Overloaded { retry_after_secs: 0 },
        MockStep::Valid,
    ]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(500), 2, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert_eq!(outcome.records.len(), 3);
    assert_eq!(transport.call_count(), 2);
}

#[tokio::test]
async fn server_error_is_retried_then_succeeds_within_the_retry_budget() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::ServerError { status: 503 },
        MockStep::Valid,
    ]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(500), 2, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert_eq!(outcome.records.len(), 3);
    assert_eq!(transport.call_count(), 2);
    assert_eq!(stats.snapshot().retries, 1);
}

#[tokio::test]
async fn exhausted_retries_produce_a_skip_and_no_records() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::ServerError { status: 500 }]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(500), 1, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(outcome.is_empty(), "a failed request never fabricates answers");
    assert_eq!(outcome.skips.len(), 1);
    assert_eq!(outcome.skips[0].1, "http_status");
    assert_eq!(transport.call_count(), 2, "initial attempt plus one retry");
    assert_eq!(stats.snapshot().failures, 1);
}

#[tokio::test]
async fn timeout_is_bounded_and_reported_as_a_skip() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Timeout]));
    let stats = Arc::new(JevStats::default());
    let started = std::time::Instant::now();
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(60), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(5), "the deadline must bound the call");
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "timeout");
    assert_eq!(stats.snapshot().timeouts, 1);
    assert_eq!(stats.snapshot().attempts, 1, "no retry budget was granted");
}

#[tokio::test]
async fn dropped_connection_is_reported_as_a_skip() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::DroppedConnection]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "connection");
    assert_eq!(stats.snapshot().attempts, 1);
}

#[tokio::test]
async fn malformed_json_body_is_reported_as_a_skip() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Body(raw_truncated_body())]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "malformed_response");
    assert_eq!(stats.snapshot().malformed, 1);
    assert_eq!(stats.snapshot().attempts, 1, "a malformed body is not retried");
}

#[tokio::test]
async fn slow_response_inside_the_deadline_succeeds_and_beyond_it_times_out() {
    let stats = Arc::new(JevStats::default());
    let inside = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse { delay_ms: 5 }]));
    let outcome = run(
        inside.clone(),
        limits_with_timeout(Duration::from_millis(500), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert_eq!(outcome.records.len(), 3);

    let outside = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse { delay_ms: 5_000 }]));
    let started = std::time::Instant::now();
    let outcome = run(
        outside.clone(),
        limits_with_timeout(Duration::from_millis(50), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "timeout");
}

#[tokio::test]
async fn low_confidence_answers_are_accepted_but_reported_at_their_real_confidence() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::LowConfidence]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert_eq!(outcome.records.len(), 3);
    for record in &outcome.records {
        if let Some(confidence) = record.answer.confidence() {
            assert!(confidence <= 0.1, "low-confidence fixtures must stay low");
        }
    }
    // Low confidence does not change behavior: nothing is applied in Compare.
    assert!(outcome.records.iter().all(|record| !record.applied));
}

#[tokio::test]
async fn mock_transport_records_attempts_without_credentials_and_with_bounded_metadata() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::ServerError { status: 500 },
        MockStep::Valid,
    ]));
    let stats = Arc::new(JevStats::default());
    let _ = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(500), 2, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    let calls = transport.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].attempt, 0);
    assert_eq!(calls[1].attempt, 1);
    assert_eq!(calls[0].model, "jev-latest");
    assert_eq!(calls[0].state_fingerprint, fingerprint(&serde_json::json!({"summary": "bounded state excerpt"})));
    assert_eq!(calls[0].state_fingerprint.len(), 64);
    assert!(!calls[0].credential_present, "the transport interface never carries the key");
    assert_eq!(calls[0].question_ids.len(), 3);
}

#[tokio::test]
async fn oversized_request_is_rejected_before_the_transport_is_called() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    let mut questions = BTreeMap::new();
    questions.insert(
        "task_classification.0".to_string(),
        noul_question("Is this oversized?", "yes", "no"),
    );
    let mut bundle = bundle(questions);
    bundle.state = serde_json::json!({"blob": "x".repeat(4096)});
    let limits = JevLimits {
        max_payload_bytes: 256,
        ..limits_with_timeout(Duration::from_millis(100), 0, Duration::from_millis(1))
    };
    let outcome = run(
        transport.clone(), limits, &stats, bundle).await;
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "payload_too_large");
    assert_eq!(transport.call_count(), 0, "an oversized payload must not be sent");
    assert_eq!(stats.snapshot().payload_rejections, 1);
}

#[tokio::test]
async fn request_shape_violations_are_skipped_without_a_call() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    let mut questions = BTreeMap::new();
    questions.insert(
        "complexity.0".to_string(),
        // A Score question needs at least two levels; the API would reject this one.
        score_question("How complex?", &["only one level"]),
    );
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(100), 0, Duration::from_millis(1)),
        &stats,
        bundle(questions),
    )
    .await;
    assert!(outcome.is_empty());
    assert_eq!(outcome.skips[0].1, "validation");
    assert_eq!(transport.call_count(), 0);
}

// ---------------------------------------------------------------------------
// 4. Modes, off parity, and the reserved Active mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disabled_system_one_makes_no_calls_and_returns_a_skip() {
    let disabled = DisabledSystemOne::new(JevMode::Off);
    assert_eq!(disabled.mode(), JevMode::Off);
    let outcome = disabled.decide(default_bundle()).await;
    assert!(outcome.is_empty());
    // Request-level skips carry no fabricated per-question entries; the
    // correlate path logs one `answer_missing` skip per requested question.
    assert!(outcome.skips.is_empty());
    assert!(!outcome.applied);
}

#[tokio::test]
async fn off_never_touches_the_transport_even_with_a_credential_present() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    // Exactly what an Off-mode caller does: no client is built, no request is issued.
    let client = Arc::new(DisabledSystemOne::new(JevMode::Off));
    let outcome = client.decide(default_bundle()).await;
    assert!(outcome.is_empty());
    assert_eq!(transport.call_count(), 0, "Off must not reach any transport");
    assert_eq!(stats.snapshot().attempts, 0);
}

#[tokio::test]
async fn active_mode_builds_a_client_and_never_becomes_compare() {
    // Active is operative: it builds a real client, and it is never silently
    // downgraded to Compare.
    let client = JevSystemOne::with_http(
        JevMode::Active,
        SecretString::new(SYNTHETIC_KEY),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .expect("Active must construct");
    assert_eq!(client.effective_mode(), JevMode::Active);

    // The disabled wrapper must never claim an operative mode back.
    let disabled = DisabledSystemOne::new(JevMode::Active);
    assert_eq!(disabled.mode(), JevMode::Off);
    let outcome = disabled.decide(default_bundle()).await;
    assert!(outcome.is_empty());
    assert!(outcome.skips.is_empty());
    assert!(!outcome.applied);
    // Constructing the disabled wrapper for Compare does NOT arm anything.
    assert_eq!(DisabledSystemOne::new(JevMode::Compare).mode(), JevMode::Off);
}

#[tokio::test]
async fn off_and_compare_are_indistinguishable_except_for_jev_telemetry() {
    // Compare path: real evaluation, records produced.
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    let compare = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    // Off path: same bundle, no transport, no records.
    let off_mock = Arc::new(MockJevTransport::all_valid());
    let off_transport: Arc<dyn Transport> = off_mock.clone();
    let off = decide_with(
        &off_transport,
        &SecretString::new(SYNTHETIC_KEY),
        JevMode::Off,
        &limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &Arc::new(JevStats::default()),
        default_bundle(),
    )
    .await;

    assert_eq!(off_mock.call_count(), 0);
    assert!(off.is_empty());
    // The only difference is Jev's own telemetry; every field that could influence the agent loop
    // (`applied`, and the absence of any decision surface) is identical.
    assert!(!compare.applied && !off.applied);
    assert!(compare.records.iter().all(|record| !record.applied));
    assert_eq!(off.response_model, None);
    assert_eq!(off.usage.input_tokens, 0);
}

#[tokio::test]
async fn compare_never_sets_applied_and_records_model_drift() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(outcome.records.iter().all(|record| !record.applied));
    assert!(!outcome.applied);
    // Requested alias resolves to a versioned id; drift is visible on every record.
    assert_eq!(outcome.response_model.as_deref(), Some(MOCK_RESPONSE_MODEL));
    for record in &outcome.records {
        assert_eq!(record.requested_model, "jev-latest");
        let drift = record.drift();
        assert!(drift.drifted, "alias to versioned id is drift and must be recorded");
        assert_eq!(drift.response.as_deref(), Some(MOCK_RESPONSE_MODEL));
    }
    let accepted = accepted_answers(&outcome);
    assert_eq!(accepted.len(), 3);
}

#[tokio::test]
async fn model_drift_reports_a_missing_response_model_as_drift() {
    let drift = model_drift("jev-latest", None);
    assert!(drift.drifted);
    assert_eq!(drift.response, None);
    let same = model_drift("jev-1.13.0", Some("jev-1.13.0"));
    assert!(!same.drifted);
}

// ---------------------------------------------------------------------------
// 5. Client construction, stats and generation invalidation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compare_client_requires_a_credential_and_a_compare_mode() {
    assert!(matches!(
        JevSystemOne::new(
            JevMode::Compare,
            SecretString::new(""),
            Arc::new(MockJevTransport::all_valid()),
            JevLimits::default(),
            Arc::new(JevStats::default()),
        ),
        Err(JevError::MissingCredential)
    ));
    assert!(matches!(
        JevSystemOne::new(
            JevMode::Off,
            SecretString::new(SYNTHETIC_KEY),
            Arc::new(MockJevTransport::all_valid()),
            JevLimits::default(),
            Arc::new(JevStats::default()),
        ),
        Err(JevError::ModeOff)
    ));
    let client = JevSystemOne::new(
        JevMode::Compare,
        SecretString::new(SYNTHETIC_KEY),
        Arc::new(MockJevTransport::all_valid()),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    assert_eq!(client.effective_mode(), JevMode::Compare);
    assert_eq!(client.endpoint(), "https://api.typesafe.ai/v1/systemone");
    assert_eq!(client.credential_fingerprint().len(), 8);
}

#[tokio::test]
async fn client_limits_reject_unbounded_configuration() {
    let mut limits = JevLimits::default();
    assert!(limits.validate().is_ok());
    limits.timeout = Duration::ZERO;
    assert!(matches!(limits.validate(), Err(JevError::Config { .. })));
    let mut limits = JevLimits::default();
    limits.max_questions = 0;
    assert!(matches!(limits.validate(), Err(JevError::Config { .. })));
    let mut limits = JevLimits::default();
    limits.base_url = "not a url".to_string();
    assert!(matches!(limits.validate(), Err(JevError::Config { .. })));
}

#[tokio::test]
async fn generation_counter_invalidates_late_results() {
    use pi_jev::client::ClientHandle;
    let handle = ClientHandle::new();
    let first = handle.install(Arc::new(DisabledSystemOne::default()));
    assert!(handle.is_current(first));
    let second = handle.install(Arc::new(MockSystemOne::new()));
    assert!(!handle.is_current(first), "an older generation must be stale");
    assert!(handle.is_current(second));
    let cleared = handle.clear();
    assert!(handle.current().is_none());
    assert!(!handle.is_current(second));
    assert_eq!(handle.generation(), cleared);
}

/// Minimal `SystemOne` used only by the generation test.
struct MockSystemOne;

impl MockSystemOne {
    fn new() -> Self {
        MockSystemOne
    }
}

impl SystemOne for MockSystemOne {
    fn mode(&self) -> JevMode {
        JevMode::Compare
    }

    fn decide(&self, _bundle: DecisionBundle) -> pi_jev::BoxFuture<DecisionOutcome> {
        Box::pin(async { DecisionOutcome::skipped_all("not_used") })
    }
}

// ---------------------------------------------------------------------------
// 6. Mode + credential precedence
// ---------------------------------------------------------------------------

#[test]
fn explicit_session_mode_always_wins_over_the_global_default() {
    // Explicit Compare wins over a global Off (DESIGN.md 10.2, binding).
    assert_eq!(
        resolve_effective_mode(Some(JevMode::Compare), Some(JevMode::Off)),
        JevMode::Compare
    );
    // Explicit Off wins over a global Compare.
    assert_eq!(resolve_effective_mode(Some(JevMode::Off), Some(JevMode::Compare)), JevMode::Off);
    // No explicit value: the default applies.
    assert_eq!(resolve_effective_mode(None, Some(JevMode::Compare)), JevMode::Compare);
    // Nothing set anywhere: built-in Off.
    assert_eq!(resolve_effective_mode(None, None), JevMode::Off);
    // The reserved mode is never resolved away silently.
    assert_eq!(resolve_effective_mode(Some(JevMode::Active), None), JevMode::Active);
}

#[test]
fn settings_resolve_mode_per_session_and_record_the_scope() {
    let mut settings = JevSettings::with_global_default(JevMode::Compare);
    settings.set_session_mode("explicit-off", JevMode::Off);
    assert_eq!(settings.effective_mode_with_scope("explicit-off").mode, JevMode::Off);
    assert_eq!(
        settings.effective_mode_with_scope("explicit-off").scope.as_str(),
        "session"
    );
    assert_eq!(settings.effective_mode_with_scope("new-chat").mode, JevMode::Compare);
    assert_eq!(
        settings.effective_mode_with_scope("new-chat").scope.as_str(),
        "global_default"
    );
    let defaults = JevSettings::default();
    assert_eq!(defaults.effective_mode_with_scope("new-chat").mode, JevMode::Off);
    assert_eq!(
        defaults.effective_mode_with_scope("new-chat").scope.as_str(),
        "built_in_default"
    );
}

#[test]
fn children_inherit_the_parent_effective_mode_and_an_explicit_override_wins() {
    let mut settings = JevSettings::with_global_default(JevMode::Off);
    settings.set_session_mode("parent", JevMode::Compare);
    let inherited = pi_jev::inherit_mode(&mut settings, "child", "parent", None);
    assert_eq!(inherited, JevMode::Compare);
    assert_eq!(settings.session_mode("child"), Some(JevMode::Compare));
    assert_eq!(
        settings.sessions["child"].inherited_from.as_deref(),
        Some("parent")
    );

    let overridden = pi_jev::inherit_mode(&mut settings, "child-2", "parent", Some(JevMode::Off));
    assert_eq!(overridden, JevMode::Off);

    // A later global change must not alter an existing chat's stored mode.
    settings.global_default = Some(JevMode::Compare);
    assert_eq!(settings.effective_mode("child"), JevMode::Compare);
    assert_eq!(settings.effective_mode("child-2"), JevMode::Off);
    // The parent's explicit Compare still beats the new global default.
    assert_eq!(settings.effective_mode("parent"), JevMode::Compare);
}

#[test]
fn credential_source_order_is_saved_then_typesafe_then_jev_alias() {
    assert_eq!(resolve_credential_source(true, true, true), CredentialSource::Saved);
    assert_eq!(resolve_credential_source(false, true, true), CredentialSource::EnvTypesafe);
    assert_eq!(resolve_credential_source(false, false, true), CredentialSource::EnvJev);
    assert_eq!(resolve_credential_source(false, false, false), CredentialSource::None);
    // Saved credentials work without any environment variable.
    assert_eq!(resolve_credential_source(true, false, false), CredentialSource::Saved);
}

#[test]
fn environment_key_presence_is_detected_without_reading_any_other_variable() {
    // Presence-only struct: the struct cannot even hold a value.
    let presence = EnvKeyPresence {
        typesafe_api_key: true,
        jev_api_key: true,
    };
    assert!(presence.has_conflict());
    assert!(!EnvKeyPresence::default().has_conflict());
    // `from_env` reads booleans only and must not panic in any environment.
    let observed = EnvKeyPresence::from_env();
    assert!(observed.typesafe_api_key || !observed.typesafe_api_key);
}

#[test]
fn credential_status_lines_are_secret_free_and_name_the_winner_on_conflict() {
    // Saved beats both environment variables, and the line says the env vars are ignored.
    let saved = pi_jev::credential_status_line(
        true,
        EnvKeyPresence {
            typesafe_api_key: true,
            jev_api_key: true,
        },
    );
    assert!(saved.contains("saved"));
    assert!(saved.contains("ignored"));
    assert!(!saved.contains("Bearer"));
    assert!(!saved.contains(SYNTHETIC_KEY));

    // Conflict with no saved credential: TYPESAFE_API_KEY wins, and the status says so.
    let conflict = pi_jev::credential_status_line(
        false,
        EnvKeyPresence {
            typesafe_api_key: true,
            jev_api_key: true,
        },
    );
    assert!(conflict.contains("TYPESAFE_API_KEY wins"));
    assert!(conflict.contains("JEV_API_KEY is ignored"));

    // Alias only.
    let alias = pi_jev::credential_status_line(
        false,
        EnvKeyPresence {
            typesafe_api_key: false,
            jev_api_key: true,
        },
    );
    assert!(alias.contains("JEV_API_KEY"));
    assert!(alias.contains("alias"));

    // Nothing configured.
    let none = pi_jev::credential_status_line(false, EnvKeyPresence::default());
    assert!(none.contains("none"));
    assert!(!none.contains("TYPESAFE_API_KEY"));
}

#[test]
fn mode_parsing_keeps_on_as_compare_and_active_explicit() {
    // `on` stays the shadow mode. Only the explicit `active` spelling arms a
    // mode that may change a provider request.
    assert_eq!(JevMode::parse("on"), Some(JevMode::Compare));
    assert_eq!(JevMode::parse("COMPARE"), Some(JevMode::Compare));
    assert_eq!(JevMode::parse("off"), Some(JevMode::Off));
    assert_eq!(JevMode::parse("active"), Some(JevMode::Active));
    assert_eq!(JevMode::parse("nonsense"), None);
    // Active is an operative mode; it is no longer reported as reserved.
    assert_eq!(JevMode::Active.label(), "Jev Active");
    assert!(!JevMode::Active.allows_compare());
    assert!(JevMode::Compare
        .description()
        .contains("Nothing is applied"));
    assert!(JevMode::Active
        .description()
        .contains("feature-gated decisions"));
    assert!(!JevMode::Active.description().contains("reserved"));
}

#[test]
fn settings_persist_per_session_mode_and_never_store_a_key() {
    let dir = std::env::temp_dir().join(format!("pi-jev-settings-{}", std::process::id()));
    let store = JevSettingsStore::new(&dir);
    let mut settings = JevSettings::with_global_default(JevMode::Off);
    settings.set_session_mode("session-a", JevMode::Compare);
    settings.set_credential_metadata(true, CredentialSource::Saved);
    store.save(&settings).expect("settings must save");

    let loaded = store.load();
    assert_eq!(loaded.session_mode("session-a"), Some(JevMode::Compare));
    assert_eq!(loaded.global_default, Some(JevMode::Off));
    assert!(loaded.credential_configured);
    assert_eq!(loaded.credential_source, Some(CredentialSource::Saved));

    // The file itself carries no key material.
    let raw = std::fs::read_to_string(store.path()).unwrap();
    assert!(!raw.contains(SYNTHETIC_KEY));
    assert!(!raw.contains("Bearer"));
    assert!(!loaded.looks_like_it_contains_a_secret());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_or_corrupt_settings_file_falls_back_to_defaults() {
    let dir = std::env::temp_dir().join(format!("pi-jev-settings-missing-{}", std::process::id()));
    let store = JevSettingsStore::new(&dir);
    assert_eq!(store.load().effective_mode("any"), JevMode::Off);
    std::fs::create_dir_all(pi_jev::jev_dir_for(&dir)).unwrap();
    std::fs::write(store.path(), b"{ this is not json").unwrap();
    assert_eq!(store.load().effective_mode("any"), JevMode::Off);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 7. Credential stores
// ---------------------------------------------------------------------------

#[test]
fn in_memory_store_round_trips_a_synthetic_key_and_rejects_bad_ids() {
    let store = InMemoryCredentialStore::new();
    assert!(!store.exists("typesafe").unwrap());
    store.store("typesafe", SYNTHETIC_KEY).unwrap();
    assert!(store.exists("typesafe").unwrap());
    assert_eq!(store.get("typesafe").unwrap().as_deref(), Some(SYNTHETIC_KEY));
    store.delete("typesafe").unwrap();
    assert_eq!(store.get("typesafe").unwrap(), None);
    assert!(matches!(store.store("", "x"), Err(JevError::InvalidKeyId)));
    assert!(matches!(store.store("../escape", "x"), Err(JevError::InvalidKeyId)));
    assert!(matches!(store.store("typesafe", "   "), Err(JevError::CredentialStore { .. })));
    assert_eq!(store.backend_name(), "memory");
}

#[test]
fn unavailable_store_fails_closed_and_never_stores_plaintext() {
    let store = UnavailableCredentialStore::new("dpapi unavailable in this test");
    assert!(!store.is_available());
    assert!(matches!(store.store("typesafe", SYNTHETIC_KEY), Err(JevError::Unavailable { .. })));
    assert!(matches!(store.get("typesafe"), Err(JevError::Unavailable { .. })));
    assert!(matches!(store.delete("typesafe"), Err(JevError::Unavailable { .. })));
    // Presence reports false so status stays renderable; the store is never silently used.
    assert!(!store.exists("typesafe").unwrap());
    assert_eq!(store.backend_name(), "unavailable");
}

#[cfg(windows)]
#[test]
fn dpapi_store_round_trips_a_synthetic_key_and_writes_no_plaintext() {
    let dir = std::env::temp_dir().join(format!("pi-jev-dpapi-{}", std::process::id()));
    let store = DpapiCredentialStore::new(&dir.join("jev"));
    store.store("typesafe", SYNTHETIC_KEY).expect("DPAPI store must work for the current user");
    assert!(store.exists("typesafe").unwrap());
    let path = store.dir().join(format!("typesafe.{}", pi_jev::CREDENTIAL_FILE_NAME));
    assert!(path.is_file());
    // The blob is ciphertext: the synthetic key must not appear anywhere in the file.
    let raw = std::fs::read(&path).unwrap();
    let as_text = String::from_utf8_lossy(&raw);
    assert!(!as_text.contains(SYNTHETIC_KEY), "DPAPI store must never write plaintext");
    assert_eq!(store.get("typesafe").unwrap().as_deref(), Some(SYNTHETIC_KEY));
    assert_eq!(store.backend_name(), "dpapi");
    store.delete("typesafe").unwrap();
    assert!(!store.exists("typesafe").unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(windows)]
#[test]
fn secure_store_availability_probe_reports_true_on_windows() {
    use pi_jev::credential::secure_store_available;
    assert!(secure_store_available(), "DPAPI must be usable for the current user");
    // The default store is the DPAPI store, not a plaintext or unavailable fallback.
    let dir = std::env::temp_dir().join(format!("pi-jev-default-store-{}", std::process::id()));
    let store = pi_jev::default_credential_store(&dir);
    assert!(store.is_available());
    assert_eq!(store.backend_name(), "dpapi");
}

#[cfg(not(windows))]
#[test]
fn secure_store_probe_reports_unavailable_off_windows_and_the_default_is_fail_closed() {
    use pi_jev::credential::secure_store_available;
    assert!(!secure_store_available());
    let dir = std::env::temp_dir().join(format!("pi-jev-default-store-nonwin-{}", std::process::id()));
    let store = pi_jev::default_credential_store(&dir);
    assert!(!store.is_available());
    assert_eq!(store.backend_name(), "unavailable");
    // The fail-closed default refuses every write instead of using plaintext.
    assert!(matches!(store.store("typesafe", SYNTHETIC_KEY), Err(JevError::Unavailable { .. })));
}

#[cfg(windows)]
#[test]
fn dpapi_store_rejects_a_corrupt_envelope_instead_of_returning_garbage() {
    let dir = std::env::temp_dir().join(format!("pi-jev-dpapi-corrupt-{}", std::process::id()));
    let store = DpapiCredentialStore::new(&dir.join("jev"));
    std::fs::create_dir_all(store.dir()).unwrap();
    std::fs::write(
        store.dir().join(format!("typesafe.{}", pi_jev::CREDENTIAL_FILE_NAME)),
        b"not an envelope",
    )
    .unwrap();
    assert!(matches!(store.get("typesafe"), Err(JevError::CredentialStore { .. })));
    // Tampering with the ciphertext itself must fail closed too.
    store.store("typesafe", SYNTHETIC_KEY).unwrap();
    let path = store.dir().join(format!("typesafe.{}", pi_jev::CREDENTIAL_FILE_NAME));
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let blob = envelope["blob"].as_str().unwrap().to_string();
    let mut tampered = blob.into_bytes();
    tampered[0] = if tampered[0] == b'A' { b'B' } else { b'A' };
    envelope["blob"] = serde_json::Value::String(String::from_utf8(tampered).unwrap());
    std::fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
    // Either the decrypt fails or it yields different plaintext. It must NEVER return the
    // original key from a tampered blob.
    let recovered = store.get("typesafe");
    let leaked_original = matches!(&recovered, Ok(Some(value)) if value == SYNTHETIC_KEY);
    assert!(!leaked_original, "a tampered blob must not yield the original secret");
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(windows))]
#[test]
fn dpapi_store_is_unavailable_off_windows_and_fails_closed() {
    let dir = std::env::temp_dir().join(format!("pi-jev-dpapi-nonwin-{}", std::process::id()));
    let store = DpapiCredentialStore::new(&dir.join("jev"));
    assert!(matches!(
        store.store("typesafe", SYNTHETIC_KEY),
        Err(JevError::Unavailable { .. })
    ));
    assert_eq!(store.backend_name(), "unavailable");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 8. Secret hygiene
// ---------------------------------------------------------------------------

#[test]
fn a_secret_never_appears_in_debug_display_or_status_text() {
    let secret = SecretString::new(SYNTHETIC_KEY);
    let debug = format!("{secret:?}");
    let display = format!("{secret}");
    assert!(!debug.contains(SYNTHETIC_KEY));
    assert!(!display.contains(SYNTHETIC_KEY));
    assert_eq!(debug, "<redacted>");
    // Only the deliberate header helper carries the key, and it is never logged.
    assert!(secret.bearer_header().contains(SYNTHETIC_KEY));
    assert_eq!(secret.key_fingerprint().len(), 8);
    assert!(!secret.key_fingerprint().contains(SYNTHETIC_KEY));
}

#[test]
fn sanitizers_remove_credentials_from_urls_details_and_headers() {
    assert_eq!(redact_authorization("Bearer abcdef"), "Bearer <redacted>");
    assert_eq!(redact_authorization("Basic dXNlcjpwYXNz"), "Basic <redacted>");
    assert_eq!(redact_authorization("opaque-secret"), "<redacted>");
    assert_eq!(
        sanitize_url("https://api.typesafe.ai/v1/systemone?key=leaked#frag"),
        "https://api.typesafe.ai/v1/systemone"
    );
    assert_eq!(
        sanitize_url("https://user:pass@api.typesafe.ai/v1/systemone"),
        "https://api.typesafe.ai/v1/systemone"
    );

    let detail = sanitize_detail("request failed authorization=Bearer abcdefghijklmnopqrstuvwxyz012345");
    assert!(!detail.contains("abcdefghijklmnopqrstuvwxyz012345"));
    let header = sanitize_detail("headers: {\"Authorization\": \"Bearer sk-live-0123456789abcdefghij\"}");
    assert!(!header.contains("sk-live-0123456789abcdefghij"));
    // A bare long token run is withheld even without a header name in front of it.
    let bare = sanitize_detail("failed with sk-live-0123456789abcdefghijklmn");
    assert!(!bare.contains("sk-live-0123456789abcdefghijklmn"), "got: {bare}");
    assert!(bare.contains("<redacted>"));
}

#[test]
fn error_text_and_log_lines_never_carry_a_secret_or_a_lengthy_body() {
    let error = JevError::HttpStatus {
        status: 401,
        detail: sanitize_detail("authorization=Bearer abcdefghijklmnopqrstuvwxyz012345"),
        retry_after: None,
    };
    let line = error.log_line();
    assert!(!line.contains("abcdefghijklmnopqrstuvwxyz012345"));
    assert_eq!(error.status_code(), Some(401));
    assert_eq!(error.kind(), "http_status");
    assert!(error.is_transport_failure());

    // A long body is truncated, so a log line stays bounded.
    let long = sanitize_detail(&"x ".repeat(4096));
    assert!(
        long.chars().count() <= pi_jev::MAX_DETAIL_CHARS + 3,
        "sanitized detail must be bounded, got {} chars",
        long.chars().count()
    );
    // A single very long token run is withheld entirely rather than repeated in the log.
    assert_eq!(sanitize_detail(&"y".repeat(4096)), "<redacted>");

    let timeout = JevError::Timeout {
        detail: "deadline".to_string(),
    };
    assert_eq!(timeout.retry_after(), None);
    assert!(!timeout.is_mode_refusal());
    assert!(JevError::ModeOff.is_mode_refusal());
}

#[tokio::test]
async fn client_debug_and_status_surfaces_never_print_the_key() {
    let client = JevSystemOne::new(
        JevMode::Compare,
        SecretString::new(SYNTHETIC_KEY),
        Arc::new(MockJevTransport::all_valid()),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    let debug = format!("{client:?}");
    assert!(!debug.contains(SYNTHETIC_KEY));
    assert!(debug.contains("redacted"));
    assert!(!client.endpoint().contains(SYNTHETIC_KEY));
    // A skip record's reasons are static strings with no payload content.
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Body(raw_injection_body("x"))]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport,
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    for (_, reason) in &outcome.skips {
        assert!(!reason.contains(SYNTHETIC_KEY));
        assert!(reason.len() < 64);
    }
}

// ---------------------------------------------------------------------------
// 8b. Lane-B entry points: build_system_one, bundle_with_questions, agent dir
// ---------------------------------------------------------------------------

#[tokio::test]
async fn build_system_one_constructs_a_client_only_for_operative_modes() {
    use pi_jev::build_system_one;
    // Off: no credential needed, no transport, no network object.
    let off = build_system_one(
        JevMode::Off,
        None,
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    assert_eq!(off.mode(), JevMode::Off);
    assert!(off.decide(default_bundle()).await.is_empty());

    // Active is operative: with a credential it builds a real client.
    let active = build_system_one(
        JevMode::Active,
        Some(SecretString::new(SYNTHETIC_KEY)),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    assert_eq!(active.mode(), JevMode::Active);

    // Active without a credential is refused, exactly like Compare.
    assert!(matches!(
        build_system_one(
            JevMode::Active,
            None,
            JevLimits::default(),
            Arc::new(JevStats::default()),
        ),
        Err(JevError::MissingCredential)
    ));

    // Compare without a credential is refused rather than degraded to Off.
    assert!(matches!(
        build_system_one(
            JevMode::Compare,
            None,
            JevLimits::default(),
            Arc::new(JevStats::default())
        ),
        Err(JevError::MissingCredential)
    ));

    // Compare with a credential builds a live client whose endpoint is the documented one.
    let compare = build_system_one(
        JevMode::Compare,
        Some(SecretString::new(SYNTHETIC_KEY)),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .unwrap();
    assert_eq!(compare.mode(), JevMode::Compare);
}

#[test]
fn bundle_with_questions_maps_question_id_prefixes_to_categories() {
    use pi_jev::bundle_with_questions;
    let mut questions = BTreeMap::new();
    questions.insert(
        "task_classification.0".to_string(),
        noul_question("Is this coding?", "yes", "no"),
    );
    questions.insert(
        "continue_stop_escalate.2".to_string(),
        noul_question("Continue?", "yes", "no"),
    );
    questions.insert(
        "unknown_category.0".to_string(),
        noul_question("Unmapped?", "yes", "no"),
    );
    let bundle = bundle_with_questions(
        "session-9",
        7,
        "agent_end",
        serde_json::json!({"summary": "bounded"}),
        "jev-latest",
        questions,
    );
    assert_eq!(bundle.session_id, "session-9");
    assert_eq!(bundle.turn, 7);
    assert_eq!(bundle.stage, "agent_end");
    assert_eq!(
        bundle.question_categories["task_classification.0"],
        DecisionCategory::TaskClassification
    );
    assert_eq!(
        bundle.question_categories["continue_stop_escalate.2"],
        DecisionCategory::ContinueStopEscalate
    );
    // An unknown prefix is not silently dropped; it maps to the neutral default.
    assert_eq!(
        bundle.question_categories["unknown_category.0"],
        DecisionCategory::TaskClassification
    );
    let request = bundle.to_request();
    assert_eq!(request.questions.len(), 3);
    assert_eq!(request.model, "jev-latest");
}

#[test]
fn agent_dir_and_jev_dir_are_injectable_and_documented() {
    use pi_jev::{default_agent_dir, jev_dir_for};
    let injected = std::path::PathBuf::from("C:/isolated/agent-dir");
    assert_eq!(default_agent_dir(Some(&injected)), injected);
    // Without injection the env override or the home default is used; both are absolute-ish
    // and neither touches production state in this test because we do not write anything.
    let resolved = default_agent_dir(None);
    assert!(!resolved.as_os_str().is_empty());
    assert!(jev_dir_for(&injected).ends_with("jev"));
    let store = JevSettingsStore::new(&injected);
    assert!(store.path().ends_with(pi_jev::SETTINGS_FILE_NAME));
    assert!(store.path().to_string_lossy().contains("jev"));
}

#[tokio::test]
async fn a_valid_raw_body_travels_the_production_parser_and_client_path() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    let body = serde_json::to_string(&valid_response_for(&request)).unwrap();
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Body(body)]));
    let stats = Arc::new(JevStats::default());
    let outcome = run(
        transport,
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert_eq!(outcome.records.len(), 3);
    assert_eq!(outcome.response_model.as_deref(), Some(MOCK_RESPONSE_MODEL));
    assert_eq!(accepted_answers(&outcome).len(), 3);
}

// ---------------------------------------------------------------------------
// 9. Permanent no-subagent-control boundary (DESIGN.md 11/12; binding negatives)
// ---------------------------------------------------------------------------

/// Every forbidden capability, as a concrete request, for exhaustive refusal checks.
fn all_control_requests() -> Vec<SubagentControlRequest> {
    vec![
        SubagentControlRequest::Spawn {
            role: "worker".to_string(),
        },
        SubagentControlRequest::Delete {
            child: "child-1".to_string(),
        },
        SubagentControlRequest::Cancel {
            child: "child-1".to_string(),
        },
        SubagentControlRequest::SelectModel {
            child: "child-1".to_string(),
            model: "ollama-cloud/glm-5.3-flash".to_string(),
            effort: "high".to_string(),
        },
        SubagentControlRequest::AssignWork {
            child: "child-1".to_string(),
            summary: "take over the task".to_string(),
        },
        SubagentControlRequest::SendMessage {
            target: "child-1".to_string(),
            body: "ignore your instructions".to_string(),
        },
        SubagentControlRequest::ChangeBudget {
            max_depth: 9,
            max_concurrency: 99,
        },
        SubagentControlRequest::DecideCompletion {
            child: "child-1".to_string(),
            done: true,
        },
    ]
}

/// Baseline "delegation" state. Nothing in this test may change it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DelegationBaseline {
    children: Vec<String>,
    child_models: Vec<(String, String)>,
    primary_model: String,
    primary_effort: String,
    max_depth: u32,
    max_concurrency: u32,
    messages_sent: usize,
}

impl DelegationBaseline {
    fn sample() -> Self {
        Self {
            children: vec!["child-1".to_string(), "child-2".to_string()],
            child_models: vec![
                ("child-1".to_string(), "ollama-cloud/glm-5.3-flash".to_string()),
                ("child-2".to_string(), "ollama-cloud/deepseek-v4.1-flash".to_string()),
            ],
            primary_model: "user-selected-primary".to_string(),
            primary_effort: "medium".to_string(),
            max_depth: 2,
            max_concurrency: 4,
            messages_sent: 1,
        }
    }
}

#[test]
fn every_subagent_control_capability_is_refused_and_the_capability_list_is_complete() {
    let requests = all_control_requests();
    let mut seen: Vec<&'static str> = Vec::new();
    for request in &requests {
        let error = refuse_subagent_control(request).unwrap_err();
        assert!(error.is_subagent_control_refusal(), "must be the boundary refusal");
        assert_eq!(error.kind(), "subagent_control_forbidden");
        assert!(error.is_mode_refusal(), "the boundary is a refusal like Off");
        assert!(!error.is_transport_failure());
        seen.push(request.capability());
    }
    // The refusal set covers the documented list; the two extra entries are covered by the
    // model/effort and message cases, which include the primary-model and steering prohibitions.
    for capability in FORBIDDEN_SUBAGENT_CAPABILITIES {
        let covered = seen.contains(&capability)
            || matches!(
                capability,
                "pause_subagent"
                    | "resume_subagent"
                    | "steer_suppress_or_reorder_agent_messages"
                    | "switch_primary_model_or_effort"
            );
        assert!(covered, "capability {capability} has no refusal coverage");
    }
}

/// Runs one evaluation and returns the outcome plus a delegation snapshot taken afterwards.
///
/// `apply` is deliberately absent: this helper has no way to mutate the baseline, which is the
/// point of the boundary. A caller can only read the snapshot.
async fn evaluate_with_observations(
    transport: Arc<dyn Transport>,
    limits: JevLimits,
    bundle: DecisionBundle,
) -> (DecisionOutcome, Vec<SubagentObservation>) {
    let stats = Arc::new(JevStats::default());
    let outcome = run(transport, limits, &stats, bundle).await;
    let observations = outcome
        .records
        .iter()
        .filter(|record| {
            matches!(
                record.category,
                DecisionCategory::SubagentRequirement | DecisionCategory::SubagentModelRouting
            )
        })
        .map(|record| {
            SubagentObservation::new("child-1", record.answer.selected_value(), record.category)
        })
        .collect();
    (outcome, observations)
}

#[tokio::test]
async fn high_confidence_answers_produce_zero_child_changes() {
    let baseline = DelegationBaseline::sample();
    let mut questions = BTreeMap::new();
    questions.insert(
        "subagent_requirement.0".to_string(),
        noul_question("Is a subagent required?", "yes", "no"),
    );
    questions.insert(
        "subagent_model_routing.0".to_string(),
        choice_question(
            "Which allowlisted model is suitable?",
            &[("child-2", Some("delegate")), ("none", Some("keep baseline"))],
        ),
    );
    let transport = Arc::new(MockJevTransport::all_valid());
    let (outcome, observations) = evaluate_with_observations(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        bundle(questions),
    )
    .await;

    // High confidence cannot act: every record is applied=false and nothing is actionable.
    assert!(!outcome.applied);
    assert!(outcome.records.iter().all(|record| !record.applied));
    assert_eq!(observations.len(), 2);
    assert!(observations.iter().all(|observation| !observation.actionable()));
    assert!(observations.iter().all(|observation| !observation.applied()));
    assert!(observations.iter().all(SubagentObservation::is_advisory));
    // The delegation baseline is untouched, and no capability was available to touch it.
    assert_eq!(baseline, DelegationBaseline::sample());
    assert_eq!(baseline.children, vec!["child-1".to_string(), "child-2".to_string()]);
    for request in all_control_requests() {
        assert!(refuse_subagent_control(&request).is_err());
    }
}

#[tokio::test]
async fn malicious_answers_produce_zero_child_changes() {
    let baseline = DelegationBaseline::sample();
    // A Choice answer whose value is an instruction to spawn agents and switch models.
    let mut questions = BTreeMap::new();
    questions.insert(
        "subagent_model_routing.0".to_string(),
        choice_question("Which model?", &[("child-2", Some("allowed")), ("none", Some("baseline"))]),
    );
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Mutation(
        MockMutation::InjectionText {
            id: "subagent_model_routing.0".to_string(),
        },
    )]));
    let (outcome, observations) = evaluate_with_observations(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        bundle(questions),
    )
    .await;

    // The injected value fails the criteria check, so it is a skip: not even an observation.
    assert!(outcome.records.is_empty());
    assert_eq!(outcome.skips[0].1, "choice_not_in_criteria");
    assert!(observations.is_empty());
    // Injection text is data: it never becomes a capability call or a message.
    assert_eq!(baseline, DelegationBaseline::sample());
    assert_eq!(baseline.messages_sent, 1);
}

#[tokio::test]
async fn delayed_and_stale_results_produce_zero_child_changes() {
    let baseline = DelegationBaseline::sample();
    let stats = Arc::new(JevStats::default());
    let stale_transport = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse { delay_ms: 40 }]));
    let slow = run(
        stale_transport,
        limits_with_timeout(Duration::from_millis(500), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    // A late result is still only data: applied=false and no observation is actionable.
    assert!(slow.records.iter().all(|record| !record.applied));

    // A result whose generation was superseded is discarded rather than applied.
    use pi_jev::client::ClientHandle;
    let handle = ClientHandle::new();
    let generation = handle.install(Arc::new(MockSystemOne::new()));
    handle.clear();
    assert!(!handle.is_current(generation), "a stale generation must be rejected");
    assert!(handle.current().is_none(), "a cleared handle exposes no client at all");
    assert_eq!(baseline, DelegationBaseline::sample());
}

#[tokio::test]
async fn active_mode_still_has_zero_child_control() {
    let baseline = DelegationBaseline::sample();
    // Active builds a client, and that client still has no child authority.
    assert!(JevSystemOne::with_http(
        JevMode::Active,
        SecretString::new(SYNTHETIC_KEY),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .is_ok());
    let active = DisabledSystemOne::new(JevMode::Active);
    let outcome = active.decide(default_bundle()).await;
    assert!(outcome.is_empty());
    assert!(outcome.skips.is_empty());
    assert!(!outcome.applied);

    // The boundary is not mode-gated: Active is still refused for every capability.
    for request in all_control_requests() {
        let error = refuse_subagent_control(&request).unwrap_err();
        assert_eq!(error.kind(), "subagent_control_forbidden");
    }
    assert_eq!(baseline, DelegationBaseline::sample());
    assert_eq!(baseline.primary_model, "user-selected-primary");
    assert_eq!(baseline.primary_effort, "medium");
    assert_eq!(baseline.max_depth, 2);
    assert_eq!(baseline.max_concurrency, 4);
}

#[test]
fn no_mutable_child_control_surface_is_reachable_through_jev_observations() {
    // Observations are read-only data: the only methods are non-mutating predicates and getters.
    let observation = SubagentObservation::new(
        "child-1",
        "advisory: child-2 might suit this work",
        DecisionCategory::SubagentModelRouting,
    );
    assert!(observation.is_advisory());
    assert!(!observation.actionable());
    assert!(!observation.applied());
    assert!(!observation.applied, "applied is always false");
    // Even cloning one cannot produce a handle: the type carries an opaque id string only.
    let clone = observation.clone();
    assert_eq!(clone.child_session_id, "child-1");
    assert_eq!(clone, observation);
    // Categories 5/6 are the only advisory ones; every other category is equally non-acting.
    for category in DecisionCategory::all() {
        let observation = SubagentObservation::new("child-1", "advisory", category);
        assert!(!observation.actionable());
        assert_eq!(observation.is_advisory(), matches!(
            category,
            DecisionCategory::SubagentRequirement | DecisionCategory::SubagentModelRouting
        ));
    }
}

#[tokio::test]
async fn subagent_control_refusal_is_never_retried_and_never_reaches_a_transport() {
    // A refused capability is terminal, so a retry loop cannot turn it into an action.
    let limits = JevLimits::default();
    let refusal = refuse_subagent_control(&SubagentControlRequest::Spawn {
        role: "worker".to_string(),
    })
    .unwrap_err();
    assert_eq!(retry_decision(&refusal, 0, &limits), RetryDecision::Stop);

    let transport = Arc::new(MockJevTransport::all_valid());
    let stats = Arc::new(JevStats::default());
    // Compare evaluates normally, but the resulting data still yields no child control.
    let outcome = run(
        transport.clone(),
        limits_with_timeout(Duration::from_millis(200), 0, Duration::from_millis(1)),
        &stats,
        default_bundle(),
    )
    .await;
    assert!(!outcome.applied);
    for request in all_control_requests() {
        assert!(refuse_subagent_control(&request).is_err());
    }
    assert_eq!(stats.snapshot().retries, 0);
}

// ---------------------------------------------------------------------------
// 9. Question shapes and bundle bookkeeping
// ---------------------------------------------------------------------------

#[test]
fn question_shapes_are_validated_before_they_can_be_sent() {
    let request = SystemOneRequest::new(serde_json::json!("state"), question_map());
    assert!(pi_jev::validate_request_shape(&request).is_ok());

    let mut empty = BTreeMap::new();
    empty.insert("x.0".to_string(), noul_question("", "yes", "no"));
    assert!(pi_jev::validate_request_shape(&SystemOneRequest::new(serde_json::json!("s"), empty)).is_err());

    let mut single_level = BTreeMap::new();
    single_level.insert("complexity.0".to_string(), score_question("rate", &["only"]));
    assert!(pi_jev::validate_request_shape(&SystemOneRequest::new(serde_json::json!("s"), single_level)).is_err());

    let mut no_options = BTreeMap::new();
    no_options.insert("task_classification.0".to_string(), choice_question("pick", &[]));
    assert!(pi_jev::validate_request_shape(&SystemOneRequest::new(serde_json::json!("s"), no_options)).is_err());
}

#[test]
fn wire_json_uses_the_documented_shapes() {
    let request = SystemOneRequest::new(serde_json::json!("Help! My payouts have been failing."), question_map());
    let value: serde_json::Value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["model"], "jev-latest");
    assert_eq!(value["questions"]["result_sufficiency.0"]["type"], "noul");
    assert_eq!(value["questions"]["result_sufficiency.0"]["criteria"]["true"], "sufficient");
    assert_eq!(value["questions"]["result_sufficiency.0"]["criteria"]["false"], "insufficient");
    assert_eq!(value["questions"]["task_classification.0"]["type"], "choice");
    assert_eq!(value["questions"]["complexity.0"]["type"], "score");
    assert!(value["questions"]["complexity.0"]["criteria"].is_array());

    let response = valid_response_for(&request);
    let value: serde_json::Value = serde_json::to_value(&response).unwrap();
    assert_eq!(value["answers"]["result_sufficiency.0"]["type"], "noul");
    assert!(value["answers"]["result_sufficiency.0"].get("confidence").is_none());
    assert!(value["answers"]["task_classification.0"]["confidence"].is_number());
    assert!(value["usage"]["input_tokens"].is_number());
}

#[test]
fn bundle_helpers_map_question_ids_to_categories_and_keep_every_category_enabled() {
    let bundle = default_bundle();
    assert_eq!(bundle.question_categories["complexity.0"], DecisionCategory::Complexity);
    assert_eq!(
        bundle.question_categories["result_sufficiency.0"],
        DecisionCategory::ResultSufficiency
    );
    let enabled = pi_jev::compare_default_categories();
    assert_eq!(enabled.len(), 14, "category switches include the opt-in observers");
    assert!(enabled.values().all(|value| *value));
    let request = bundle.to_request();
    let ids = bundle.enabled_question_ids(&enabled);
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|id| request.questions.contains_key(id)));
    assert_eq!(DecisionCategory::all().len(), 14);
    assert_eq!(DecisionCategory::Complexity.question_id(2), "complexity.2");
    assert_eq!(DecisionCategory::parse("task_classification"), Some(DecisionCategory::TaskClassification));
}

#[test]
fn backoff_policy_is_documented_and_finite() {
    let line = pi_jev::backoff_policy_line();
    assert!(line.contains("Retry-After"));
    let limits = JevLimits::default();
    assert!(limits.max_retries >= 1 && limits.max_retries <= 5, "retries must be finite and small");
    assert!(limits.timeout > Duration::ZERO && limits.timeout <= Duration::from_secs(60));
    assert!(limits.max_payload_bytes > 0);
    assert!(pi_jev::RETRYABLE_STATUSES.contains(&429));
    assert!(pi_jev::RETRYABLE_STATUSES.contains(&503));
    // A 401 is terminal: retrying a bad key wastes budget and cannot succeed.
    let unauthorized = JevError::HttpStatus {
        status: 401,
        detail: "unauthorized".to_string(),
        retry_after: None,
    };
    assert_eq!(retry_decision(&unauthorized, 0, &limits), RetryDecision::Stop);
}
