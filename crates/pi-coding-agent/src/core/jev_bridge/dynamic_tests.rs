use super::*;
use pi_jev::config::JevMode;
use pi_jev::mock::{MockJevTransport, MockMutation, MockStep};
use pi_jev::{JevLimits, JevStats, JevSystemOne, SecretString, SystemOne};

fn arguments() -> Value {
    json!({"state":{"request":"Flip a coin and assess this example"},"questions":{
        "coin":{"type":"choice","instructions":"Choose a side of the coin.","criteria":{"heads":"Heads","tails":"Tails"}},
        "yes":{"type":"noul","instructions":"Does the request mention a coin?"},
        "rating":{"type":"score","instructions":{"task":"Rate clarity"},"criteria":["Unclear","Clear"]}
    },"sample":["coin"]})
}

async fn outcome(input: &Input, step: MockStep) -> DecisionOutcome {
    let transport = Arc::new(MockJevTransport::scripted(vec![step]));
    let client = JevSystemOne::new(
        JevMode::Active,
        SecretString::new("synthetic"),
        transport.clone(),
        JevLimits {
            max_retries: 0,
            ..Default::default()
        },
        Arc::new(JevStats::default()),
    )
    .unwrap();
    let result = client
        .decide(input.bundle("dynamic-output", &JevSettings::default()))
        .await;
    assert_eq!(transport.call_count(), 1);
    result
}

#[tokio::test]
async fn dynamic_returns_batched_typed_answers_sampling_and_actual_usage() {
    let input = Input::parse(arguments()).unwrap();
    let result = result_value(&input, outcome(&input, MockStep::Valid).await, 12, || 0.99).unwrap();
    assert_eq!(result["answers"]["coin"]["choice"], "heads");
    assert_eq!(result["sampled"]["coin"]["choice"], "tails");
    assert_eq!(result["sampled"]["coin"]["method"], "jev_distribution");
    assert_eq!(result["answers"]["yes"]["noul"], 0.87);
    assert!(result["answers"]["rating"]["score"].is_number());
    assert_eq!(result["usage"]["input_tokens"], 312);
    assert_eq!(result["latency_ms"], 12);
    assert_eq!(result["model"], pi_jev::mock::MOCK_RESPONSE_MODEL);
    assert!(result.get("estimated_savings").is_none());
}

#[tokio::test]
async fn dynamic_retains_low_confidence_and_unknown_usage_without_inventing_values() {
    let input = Input::parse(arguments()).unwrap();
    let result = result_value(
        &input,
        outcome(&input, MockStep::LowConfidence).await,
        0,
        || 0.5,
    )
    .unwrap();
    assert_eq!(result["answers"]["coin"]["confidence"], 0.05);
    assert_eq!(result["sampled"]["coin"]["choice"], "tails");
    assert!(result["usage"]["input_tokens"].is_null());
}

#[tokio::test]
async fn dynamic_never_samples_invalid_or_partial_results() {
    let input = Input::parse(arguments()).unwrap();
    for mutation in [
        MockMutation::EmptyAnswers,
        MockMutation::MissingId { id: "yes".into() },
        MockMutation::ProbabilitiesNotSummingToOne { id: "coin".into() },
        MockMutation::NoulOutOfRange { id: "yes".into() },
    ] {
        let result = outcome(&input, MockStep::Mutation(mutation)).await;
        assert!(result_value(&input, result, 0, || panic!(
            "must not sample invalid result"
        ))
        .is_err());
    }
}

#[test]
fn dynamic_validates_input_before_transport() {
    for sample in [
        json!(["yes"]),
        json!(["rating"]),
        json!(["missing"]),
        json!(["coin", "coin"]),
    ] {
        let mut value = arguments();
        value["sample"] = sample;
        assert!(Input::parse(value).is_err());
    }
    for questions in [
        json!({}),
        json!({"bad id":{"type":"noul","instructions":"yes?"}}),
        json!({"x":{"type":"score","instructions":"rate","criteria":["only one"]}}),
    ] {
        let mut value = arguments();
        value["questions"] = questions;
        assert!(Input::parse(value).is_err());
    }
    let mut oversized = arguments();
    oversized["state"] = json!("x".repeat(pi_jev::client::DEFAULT_MAX_PAYLOAD_BYTES));
    assert!(Input::parse(oversized).is_err());
    let mut unknown = arguments();
    unknown["model"] = json!("unconfigured-model");
    assert!(Input::parse(unknown).is_err());
}

#[test]
fn dynamic_sampling_handles_boundaries_and_zero_weights() {
    let distribution = BTreeMap::from([("a".into(), 0.0), ("b".into(), 0.25), ("c".into(), 0.75)]);
    for (draw, expected) in [(0.0, "b"), (0.249, "b"), (0.25, "c"), (0.999, "c")] {
        assert_eq!(
            sample_choice(&distribution, draw).as_deref(),
            Some(expected)
        );
    }
    for draw in [f64::NAN, f64::INFINITY, -0.1, 1.0] {
        assert!(sample_choice(&distribution, draw).is_none());
    }
}

#[test]
fn dynamic_full_jev_enables_tool_and_restores_saved_disabled_setting() {
    let mut settings = JevSettings::default();
    settings.set_session_mode("chat", JevMode::Active);
    assert!(!enabled(&settings, "chat"));
    settings.set_session_feature("chat", pi_jev::config::JevFeature::Dynamic, true);
    assert!(enabled(&settings, "chat"));
    settings.set_session_mode("chat", JevMode::Compare);
    assert!(!enabled(&settings, "chat"));
    settings.set_session_feature("chat", pi_jev::config::JevFeature::Dynamic, false);
    settings.full_jev_install();
    assert!(enabled(&settings, "chat"));
    assert!(enabled(&settings, "new-child"));
    settings.full_jev_remove();
    assert!(!enabled(&settings, "chat"));
}

#[test]
fn dynamic_tool_registration_can_enable_and_remove_without_duplicate_tools() {
    let mut extensions = vec![];
    register(&mut extensions, ExtensionRuntime::new(Default::default()));
    let extension = &extensions[0];
    assert!(extension.lock().unwrap().tools.is_empty());
    for event in [
        "session_start",
        "session_switch",
        "before_agent_start",
        "turn_start",
    ] {
        assert!(extension.lock().unwrap().handlers.contains_key(event));
    }
    assert!(sync_tools(extension, true));
    assert!(!sync_tools(extension, true));
    assert_eq!(extension.lock().unwrap().tools.len(), 1);
    let tool = extension.lock().unwrap().tools[TOOL_NAME]
        .definition
        .clone();
    assert_eq!(tool.parameters["required"], json!(["state", "questions"]));
    assert!(tool
        .description
        .contains("not a guaranteed fair distribution"));
    assert!(sync_tools(extension, false));
    assert!(extension.lock().unwrap().tools.is_empty());
}

#[tokio::test]
async fn dynamic_runner_reloads_full_jev_and_refreshes_catalog_on_the_next_prompt() {
    use crate::core::extensions::runner::{ExtensionRunner, NullModelRegistry, NullSessionManager};
    use crate::core::extensions::types::{ExtensionActions, ExtensionRuntimeState};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JevSettingsStore::new(dir.path()));
    let refreshes = Arc::new(AtomicUsize::new(0));
    let counter = refreshes.clone();
    let runtime = ExtensionRuntime::new(ExtensionRuntimeState {
        actions: Some(ExtensionActions {
            send_message: Arc::new(|_, _| {}),
            send_user_message: Arc::new(|_, _| {}),
            append_entry: Arc::new(|_, _| {}),
            set_session_name: Arc::new(|_| Box::pin(async {})),
            get_session_name: Arc::new(|| None),
            set_label: Arc::new(|_, _| {}),
            get_active_tools: Arc::new(Vec::new),
            get_all_tools: Arc::new(Vec::new),
            set_active_tools: Arc::new(|_| {}),
            refresh_tools: Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
            get_commands: Arc::new(Vec::new),
            set_model: Arc::new(|_| Box::pin(async { true })),
            get_thinking_level: Arc::new(|| pi_agent_core::types::ThinkingLevel::Off),
            set_thinking_level: Arc::new(|_| {}),
        }),
        ..Default::default()
    });
    let mut extensions = vec![];
    register_with_store(&mut extensions, runtime.clone(), store.clone());
    let runner = Arc::new(ExtensionRunner::new(
        extensions,
        runtime,
        "/synthetic".into(),
        Arc::new(NullSessionManager),
        Arc::new(NullModelRegistry),
    ));
    let mut settings = store.load();
    for (on, count) in [(false, 0), (true, 1), (true, 1), (false, 2)] {
        if on {
            settings.full_jev_install();
        } else {
            settings.full_jev_remove();
        }
        store.save(&settings).unwrap();
        // Use the real event runner, without changing environment or reading a user profile.
        runner
            .emit_before_agent_start(
                "Get Jev to flip a coin".into(),
                None,
                "base".into(),
                Default::default(),
            )
            .await;
        assert_eq!(runner.get_tool_definition(TOOL_NAME).is_some(), on);
        assert_eq!(refreshes.load(Ordering::SeqCst), count);
        settings = store.load();
    }
}

#[test]
fn dynamic_policy_generation_invalidates_off_on_and_model_aba() {
    let dir = tempfile::tempdir().unwrap();
    let store = JevSettingsStore::new(dir.path());
    let mut settings = store.load();
    settings.full_jev_install();
    store.save(&settings).unwrap();
    let original = store.load();
    let generation = super::super::decision_policy_generation(&original, "chat");
    let mut changed = original.clone();
    changed.full_jev_remove();
    store.save(&changed).unwrap();
    changed = store.load();
    changed.full_jev_install();
    store.save(&changed).unwrap();
    assert_ne!(
        generation,
        super::super::decision_policy_generation(&store.load(), "chat")
    );
    let mut model_a = store.load();
    let before = super::super::decision_policy_generation(&model_a, "chat");
    model_a.requested_model = Some("jev-other".into());
    store.save(&model_a).unwrap();
    let mut model_b = store.load();
    model_b.requested_model = original.requested_model;
    store.save(&model_b).unwrap();
    assert_ne!(
        before,
        super::super::decision_policy_generation(&store.load(), "chat")
    );
}
