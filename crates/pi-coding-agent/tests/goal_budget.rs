//! Exercise goal budget warnings through real message-end and turn-end hooks.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Model, TextContent, Usage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::goals::{GoalStatus, GOAL_CONTEXT_CUSTOM_TYPE, GOAL_STATE_CUSTOM_TYPE};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::{json, Value};

async fn fixture() -> (tempfile::TempDir, Arc<AgentSession>) {
    let root = tempfile::tempdir().unwrap();
    let cwd = root.path().join("workspace");
    let profile = root.path().join("profile");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&profile).unwrap();
    let cwd = cwd.to_string_lossy().into_owned();
    let profile = profile.to_string_lossy().into_owned();
    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        json!({"autoRefine":{"enabled":false}, "retry":{"enabled":false},
            "compaction":{"enabled":false}, "telemetryEnabled":false, "agentTracesEnabled":false})
        .as_object()
        .unwrap()
        .clone(),
    )));
    let model = Model::new(
        "goal-fixture",
        "Goal fixture",
        "openai-completions",
        "goal-fixture",
        "https://fixture.invalid",
    );
    let mut auth = AuthStorage::in_memory(Default::default(), None);
    auth.set_runtime_api_key(&model.provider, "synthetic-fixture-key");
    let mut registry = ModelRegistry::in_memory(AuthStorage::in_memory(Default::default(), None));
    registry.set_runtime_api_key(&model.provider, "synthetic-fixture-key");
    let manager = Arc::new(Mutex::new(
        SessionManager::in_memory(Some(&cwd), Some(&profile)).unwrap(),
    ));
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.clone(),
        agent_dir: Some(profile.clone()),
        auth_storage: Some(Arc::new(tokio::sync::Mutex::new(auth))),
        settings_manager: Some(settings.clone()),
        model_registry: Some(Arc::new(Mutex::new(registry))),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(DefaultResourceLoaderOptions {
            cwd,
            agent_dir: profile,
            settings_manager: Some(settings),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        }),
    })
    .await
    .unwrap();
    let session = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager: manager,
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model),
            no_tools: Some("all".into()),
            include_goals: Some(true),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .unwrap()
    .session;
    (root, session)
}

async fn run_budget_case(budget: Option<f64>, usage: [(f64, f64); 2], limited: bool) {
    let (_root, session) = fixture().await;
    session
        .handle_goal_host_request(
            "goal.create",
            Some(&json!({
                "objective":"Complete the isolated regression", "token_budget":budget,
            })),
        )
        .unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    let weak = Arc::downgrade(&session);
    session
        .agent
        .set_stream_fn(Arc::new(move |model, context, _| {
            let session = weak.upgrade().unwrap();
            let goal = session.handle_goal_host_request("goal.get", None).unwrap();
            let index = {
                let mut requests = seen.lock().unwrap();
                requests.push((context, goal));
                requests.len()
            };
            assert!(
                index <= 3,
                "budget handling must not produce a warning loop"
            );
            if index == 3 && !limited {
                session
                    .handle_goal_host_request("goal.complete", None)
                    .unwrap();
            }
            Box::pin(async move {
                // Large cache usage must not exhaust the input/output goal budget.
                let (input, output) = usage.get(index - 1).copied().unwrap_or((999.0, 999.0));
                let message = AssistantMessage {
                    model: model.id,
                    provider: model.provider,
                    api: model.api,
                    content: vec![ContentBlock::Text(TextContent::new(format!(
                        "Fixture step {index}"
                    )))],
                    stop_reason: "stop".into(),
                    timestamp: index as i64,
                    usage: Usage {
                        input,
                        output,
                        cache_read: 200_000.0,
                        total_tokens: input + output + 200_000.0,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Start {
                    partial: message.clone(),
                });
                stream.push(AssistantMessageEvent::Done {
                    reason: message.stop_reason.clone(),
                    message,
                });
                stream.end(None);
                stream
            })
        }));
    tokio::time::timeout(Duration::from_secs(10), async {
        session.prompt("Run the isolated goal", None).await.unwrap();
        session.wait_for_idle().await.unwrap();
    })
    .await
    .expect("goal run must settle");
    let goal = session.handle_goal_host_request("goal.get", None).unwrap();
    let branch = session.session_manager.lock().unwrap().get_branch(None);
    session.dispose_async(Some(false)).await;

    let total = usage
        .iter()
        .map(|(input, output)| input + output)
        .sum::<f64>();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    for (index, (_, state)) in requests.iter().enumerate() {
        let expected = usage[..index.min(2)]
            .iter()
            .map(|(i, o)| i + o)
            .sum::<f64>();
        assert_eq!(
            state.goal.as_ref().unwrap().tokens_used,
            expected,
            "message-end and turn-end must account each response exactly once"
        );
    }
    let contexts: Vec<_> = branch
        .iter()
        .filter(|row| {
            row.get("customType").and_then(Value::as_str) == Some(GOAL_CONTEXT_CUSTOM_TYPE)
        })
        .collect();
    let warnings: Vec<_> = contexts
        .iter()
        .filter(|row| row["details"]["kind"] == "budget_limit")
        .collect();
    assert_eq!(
        warnings.len(),
        usize::from(limited),
        "warn once only when the budget is actually reached"
    );
    assert!(
        !serde_json::to_string(&requests[1].0)
            .unwrap()
            .contains("has reached its token budget"),
        "the second request must continue the goal below budget"
    );
    assert!(serde_json::to_string(&requests[1].0)
        .unwrap()
        .contains("Continue working toward the active thread goal"));
    assert_eq!(goal.goal.as_ref().unwrap().tokens_used, total);
    assert_eq!(
        goal.goal.as_ref().unwrap().status,
        if limited {
            GoalStatus::BudgetLimited
        } else {
            GoalStatus::Complete
        }
    );
    assert_eq!(
        goal.remaining_tokens,
        budget.map(|value| (value - total).max(0.0))
    );
    if limited {
        assert_eq!(warnings[0]["details"]["status"], "budget_limited");
        let states: Vec<_> = branch
            .iter()
            .filter(|row| {
                row.get("customType").and_then(Value::as_str) == Some(GOAL_STATE_CUSTOM_TYPE)
                    && row["data"]["status"] == "budget_limited"
            })
            .collect();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0]["data"]["tokensUsed"].as_f64(), Some(total));
        assert!(serde_json::to_string(&requests[2].0)
            .unwrap()
            .contains("has reached its token budget"));
    } else {
        assert_eq!(contexts.len(), 2);
        assert!(contexts
            .iter()
            .all(|row| row["details"]["kind"] == "continuation"));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn below_budget_continues_without_an_exhaustion_warning() {
    run_budget_case(Some(120_000.0), [(9_985.0, 342.0), (1_015.0, 218.0)], false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_usage_does_not_exhaust_a_goal() {
    run_budget_case(Some(120_000.0), [(0.0, 0.0); 2], false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unbudgeted_goal_never_emits_an_exhaustion_warning() {
    run_budget_case(None, [(200_000.0, 1_000.0); 2], false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reaching_or_crossing_budget_warns_once_then_stops() {
    for output in [39.0, 40.0] {
        run_budget_case(Some(100.0), [(50.0, 10.0), (1.0, output)], true).await;
    }
}
