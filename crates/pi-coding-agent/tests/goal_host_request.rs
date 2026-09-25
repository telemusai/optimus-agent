//! Goal host requests must finish even when subscribers read the updated goal.
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use pi_coding_agent::core::agent_session::{AgentSession, AgentSessionEvent};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::goals::{empty_goal_state, GoalStatus, GOAL_STATE_CUSTOM_TYPE};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::{json, Value};

fn with_goal_session(
    status: GoalStatus,
    check: impl FnOnce(Arc<AgentSession>, Arc<Mutex<SessionManager>>) + Send + 'static,
) {
    // An async timeout cannot interrupt a synchronous mutex deadlock. Keep the
    // deadline outside the runtime/thread executing the actual host request.
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let root = tempfile::tempdir().unwrap();
            let cwd = root.path().join("workspace");
            let profile = root.path().join("profile");
            std::fs::create_dir_all(&cwd).unwrap();
            std::fs::create_dir_all(&profile).unwrap();
            let cwd = cwd.to_string_lossy().into_owned();
            let profile = profile.to_string_lossy().into_owned();
            let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
                json!({"autoRefine":{"enabled":false}, "telemetryEnabled":false,
                    "agentTracesEnabled":false, "retry":{"enabled":false}})
                .as_object()
                .unwrap()
                .clone(),
            )));
            let manager = Arc::new(Mutex::new(
                SessionManager::in_memory(Some(&cwd), Some(&profile)).unwrap(),
            ));
            if status != GoalStatus::Idle {
                let mut goal = empty_goal_state();
                goal.status = status;
                goal.active = status == GoalStatus::Active;
                goal.objective = Some("Previous objective".into());
                goal.goal_id = Some("previous-goal".into());
                manager
                    .lock()
                    .unwrap()
                    .append_custom_entry(
                        GOAL_STATE_CUSTOM_TYPE,
                        Some(serde_json::to_value(goal).unwrap()),
                    )
                    .unwrap();
            }
            let services = create_agent_session_services(CreateAgentSessionServicesOptions {
                cwd: cwd.clone(),
                agent_dir: Some(profile.clone()),
                auth_storage: Some(Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
                    Default::default(),
                    None,
                )))),
                settings_manager: Some(settings.clone()),
                model_registry: Some(Arc::new(Mutex::new(ModelRegistry::in_memory(
                    AuthStorage::in_memory(Default::default(), None),
                )))),
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
            let session =
                create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
                    services: Arc::new(services),
                    session_manager: manager.clone(),
                    session_start_event: None,
                    creation: AgentSessionCreationOptions {
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
            check(session.clone(), manager);
            session.dispose_async(Some(false)).await;
        });
        sender.send(()).unwrap();
    });
    receiver
        .recv_timeout(Duration::from_secs(15))
        .expect("goal host request must finish without deadlocking");
    worker.join().unwrap();
}

fn create_and_check(
    session: Arc<AgentSession>,
    manager: Arc<Mutex<SessionManager>>,
    budget: Option<f64>,
) {
    let updates = Arc::new(Mutex::new(Vec::new()));
    let observed = updates.clone();
    let weak = Arc::downgrade(&session);
    session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::GoalUpdate { goal } = event {
            let current = weak
                .upgrade()
                .unwrap()
                .handle_goal_host_request("goal.get", None)
                .unwrap();
            assert_eq!(current.goal.unwrap().goal_id, goal.goal_id);
            observed.lock().unwrap().push(goal.status);
        }
    }));
    let payload = json!({"objective":"New objective", "token_budget":budget});
    let response = session
        .handle_goal_host_request("goal.create", Some(&payload))
        .unwrap();
    let goal = response.goal.unwrap();
    assert_eq!(goal.status, GoalStatus::Active);
    assert_eq!(goal.objective, "New objective");
    assert_ne!(goal.goal_id.as_deref(), Some("previous-goal"));
    assert_eq!(goal.token_budget, budget);
    assert_eq!(response.remaining_tokens, budget);
    let entries = manager.lock().unwrap().get_branch(None);
    let persisted = entries
        .iter()
        .rev()
        .find(|entry| entry["customType"] == GOAL_STATE_CUSTOM_TYPE)
        .unwrap();
    assert_eq!(persisted["data"]["objective"], "New objective");
    assert_eq!(persisted["data"]["status"], "active");
    assert_eq!(*updates.lock().unwrap(), vec![GoalStatus::Active]);
    let completed = session
        .handle_goal_host_request("goal.complete", None)
        .unwrap();
    assert_eq!(completed.goal.unwrap().status, GoalStatus::Complete);
    assert_eq!(
        *updates.lock().unwrap(),
        vec![GoalStatus::Active, GoalStatus::Complete]
    );
}

#[test]
fn create_goal_from_idle_returns_and_publishes_budgeted_and_unbudgeted_state() {
    for budget in [None, Some(120_000.0)] {
        with_goal_session(GoalStatus::Idle, move |session, manager| {
            create_and_check(session, manager, budget)
        });
    }
}

#[test]
fn create_goal_replaces_completed_and_errored_goals_without_deadlocking() {
    for status in [GoalStatus::Complete, GoalStatus::Error] {
        with_goal_session(status, |session, manager| {
            create_and_check(session, manager, Some(120_000.0))
        });
    }
}

#[test]
fn create_goal_rejects_pending_goals_without_changing_them() {
    for status in [
        GoalStatus::Active,
        GoalStatus::Paused,
        GoalStatus::BudgetLimited,
    ] {
        with_goal_session(status, move |session, manager| {
            let before = manager.lock().unwrap().get_branch(None);
            let error = session
                .handle_goal_host_request("goal.create", Some(&json!({"objective":"Replacement"})))
                .unwrap_err();
            assert!(error.contains("cannot create a new goal"), "{error}");
            let goal = session
                .handle_goal_host_request("goal.get", None)
                .unwrap()
                .goal
                .unwrap();
            assert_eq!(goal.status, status);
            assert_eq!(goal.objective, "Previous objective");
            assert_eq!(manager.lock().unwrap().get_branch(None), before);
        });
    }
}

#[test]
fn invalid_goal_create_requests_leave_status_responsive() {
    with_goal_session(GoalStatus::Idle, |session, manager| {
        let before = manager.lock().unwrap().get_branch(None);
        for payload in [
            Value::Null,
            json!({"objective":""}),
            json!({"objective":"New", "token_budget":-1}),
            json!({"objective":"New", "token_budget":"bad"}),
        ] {
            assert!(session
                .handle_goal_host_request("goal.create", Some(&payload))
                .is_err());
            assert!(session
                .handle_goal_host_request("goal.get", None)
                .unwrap()
                .goal
                .is_none());
            assert_eq!(manager.lock().unwrap().get_branch(None), before);
        }
    });
}
