//! Offline provider captures with real session admission, persistence and tools.
use pi_ai::providers::faux::*;
use pi_ai::types::{ContentBlock, Context, Model, ToolCall};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::*;
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

static ENV: Mutex<()> = Mutex::new(());

struct Fixture {
    root: tempfile::TempDir,
    old_profile: Option<std::ffi::OsString>,
    provider: FauxProviderRegistration,
    captured: Arc<Mutex<Vec<Context>>>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("workspace")).unwrap();
        std::fs::create_dir_all(root.path().join("profile")).unwrap();
        std::fs::write(
            root.path().join("workspace/AGENTS.md"),
            "Preserve PROJECT_MODE_FIXTURE instructions.",
        )
        .unwrap();
        let old_profile = std::env::var_os("PRIME_AGENT_CODING_AGENT_DIR");
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", root.path().join("profile"));
        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("execution-mode-offline".into()),
            tokens_per_second: Some(0.0),
            ..Default::default()
        }));
        Self {
            root,
            old_profile,
            provider,
            captured: Default::default(),
        }
    }

    async fn session(&self, saved: Option<&str>, restricted: bool) -> Arc<AgentSession> {
        let cwd = self
            .root
            .path()
            .join("workspace")
            .to_string_lossy()
            .into_owned();
        let profile = self
            .root
            .path()
            .join("profile")
            .to_string_lossy()
            .into_owned();
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(json!({
            "autoRefine":{"enabled":false}, "retry":{"enabled":false}, "compaction":{"enabled":false},
            "telemetryEnabled":false, "agentTracesEnabled":false
        }).as_object().unwrap().clone())));
        let model: Model = self.provider.get_model();
        let auth = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            Default::default(),
            None,
        )));
        auth.lock()
            .await
            .set_runtime_api_key(&model.provider, "synthetic-mode-key");
        let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(Default::default(), None),
        )));
        registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, "synthetic-mode-key");
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(profile.clone()),
            auth_storage: Some(auth),
            settings_manager: Some(settings.clone()),
            model_registry: Some(registry),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir: profile.clone(),
                settings_manager: Some(settings),
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            }),
        })
        .await
        .unwrap();
        let manager = match saved {
            Some(path) => SessionManager::open(path, Some(&profile), None).unwrap(),
            None => SessionManager::create(&cwd, Some(&profile)).unwrap(),
        };
        create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::new(Mutex::new(manager)),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                allowed_tool_names: restricted.then(|| vec!["ipython".into()]),
                ..Default::default()
            },
        })
        .await
        .unwrap()
        .session
    }

    fn reply(
        &self,
        tool: Option<(&str, serde_json::Value)>,
        gate: Option<Arc<tokio::sync::Semaphore>>,
    ) {
        let captured = self.captured.clone();
        let content = tool.map(|(name, args)| {
            ContentBlock::ToolCall(ToolCall::new(
                &uuid::Uuid::new_v4().to_string(),
                name,
                args.as_object().unwrap().clone(),
            ))
        });
        self.provider
            .append_responses(vec![FauxResponseStep::Factory(Arc::new(
                move |context, _, _, _| {
                    captured.lock().unwrap().push(context.clone());
                    let gate = gate.clone();
                    let message = faux_assistant_message(
                        content
                            .clone()
                            .map(FauxAssistantContent::Block)
                            .unwrap_or_else(|| "MODE_FIXTURE_DONE".into()),
                        Some(FauxAssistantMessageOptions {
                            stop_reason: Some(
                                if content.is_some() { "toolUse" } else { "stop" }.into(),
                            ),
                            ..Default::default()
                        }),
                    );
                    Box::pin(async move {
                        if let Some(gate) = gate {
                            gate.acquire().await.unwrap().forget();
                        }
                        message
                    })
                },
            ))]);
    }

    fn assert_request(&self, index: usize, direct: bool) {
        let captured = self.captured.lock().unwrap();
        let request = &captured[index];
        let tools: Vec<_> = request
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let prompt = request.system_prompt.as_deref().unwrap();
        assert!(
            prompt.contains("PROJECT_MODE_FIXTURE"),
            "project instructions must survive"
        );
        assert_eq!(tools.contains(&"ipython"), !direct, "{tools:?}");
        assert_eq!(tools.contains(&"bash"), direct, "{tools:?}");
        assert_eq!(tools.contains(&"edit"), direct, "{tools:?}");
        assert_eq!(
            prompt.contains("Python is the orchestration language"),
            !direct
        );
        assert_eq!(prompt.contains("Execution mode: Direct tools"), direct);
        if direct {
            for stale in [
                "Pre-installed Python packages:",
                "Use Python for reading",
                "A callable `rlm`",
                "`bash(command)` starts",
            ] {
                assert!(!prompt.contains(stale), "stale instruction: {stale}");
            }
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.provider.unregister();
        match self.old_profile.take() {
            Some(value) => std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", value),
            None => std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR"),
        }
    }
}

async fn idle(session: &Arc<AgentSession>) {
    tokio::time::timeout(Duration::from_secs(30), session.wait_for_headless_idle())
        .await
        .expect("session became idle")
        .unwrap();
}

async fn turn(session: &Arc<AgentSession>, text: &str) {
    tokio::time::timeout(
        Duration::from_secs(30),
        session.prompt(text, None::<PromptOptions>),
    )
    .await
    .expect("prompt returned")
    .unwrap();
    idle(session).await;
}

#[tokio::test]
async fn mid_conversation_switch_changes_outgoing_requests_runs_direct_tools_and_resumes() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(None, false).await;
    f.reply(None, None);
    turn(&session, "Remember HISTORY_MODE_FIXTURE").await;
    f.assert_request(0, false);
    let history = session.messages();
    turn(&session, "/mode direct").await;
    assert!(session.messages().starts_with(&history));
    assert_eq!(
        f.captured.lock().unwrap().len(),
        1,
        "mode commands must never call a model"
    );
    f.reply(
        Some(("bash", json!({"command":"printf MODE_TOOL_OK"}))),
        None,
    );
    f.reply(None, None);
    turn(&session, "Run a direct command").await;
    f.assert_request(1, true);
    f.assert_request(2, true);
    let result = serde_json::to_string(&f.captured.lock().unwrap()[2].messages).unwrap();
    assert!(result.contains("MODE_TOOL_OK"));
    assert!(result.contains("HISTORY_MODE_FIXTURE"));
    let file = session.session_file().unwrap();
    session.dispose_async(Some(false)).await;
    let resumed = f.session(Some(&file), false).await;
    assert_eq!(resumed.get_active_tool_names(), ["bash", "edit"]);
    f.reply(None, None);
    turn(&resumed, "Continue after resume").await;
    f.assert_request(3, true);
    turn(&resumed, "/mode toggle").await;
    f.reply(None, None);
    turn(&resumed, "Continue in Python").await;
    f.assert_request(4, false);
    resumed.dispose_async(Some(false)).await;
    let again = f.session(Some(&file), false).await;
    assert_eq!(again.get_active_tool_names(), ["ipython"]);
    again.dispose_async(Some(false)).await;
    let fresh = f.session(None, false).await;
    assert_eq!(fresh.get_active_tool_names(), ["ipython"]);
    fresh.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn busy_switch_waits_for_current_run_then_precedes_queued_prompts() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(None, false).await;
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    f.reply(None, Some(release.clone()));
    let running = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .prompt("Wait at provider", None::<PromptOptions>)
                .await
        })
    };
    tokio::time::timeout(Duration::from_secs(10), async {
        while f.captured.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let before = session.system_prompt();
    f.reply(None, None);
    session
        .prompt(
            "Queued next prompt",
            Some(PromptOptions {
                streaming_behavior: Some("followUp".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    session
        .prompt(
            "/mode direct",
            Some(PromptOptions {
                streaming_behavior: Some("followUp".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(session.system_prompt(), before);
    assert_eq!(session.get_active_tool_names(), ["ipython"]);
    assert!(session
        .get_session_action_snapshot()
        .follow_ups
        .contains(&"/mode direct".into()));
    release.add_permits(1);
    running.await.unwrap().unwrap();
    idle(&session).await;
    f.assert_request(0, false);
    f.assert_request(1, true);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn invalid_or_restricted_mode_does_not_change_prompt_or_tools() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(None, true).await;
    let before = session.system_prompt();
    for command in ["/mode unknown", "/mode direct"] {
        let _ = session.prompt(command, None::<PromptOptions>).await;
        idle(&session).await;
        assert_eq!(session.system_prompt(), before);
        assert_eq!(session.get_active_tool_names(), ["ipython"]);
    }
    assert!(f.captured.lock().unwrap().is_empty());
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn stopped_mode_switch_preserves_queue_goal_and_stop_across_reopen() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(None, false).await;
    f.reply(None, None);
    turn(&session, "HISTORY_BEFORE_STOP").await;
    session.handle_goal_host_request("goal.create", Some(&json!({
        "objective":"Never resume this goal just to change modes"
    }))).unwrap();
    let pause = session.acquire_queued_work_pause();
    assert!(session.follow_up("pending setup", None, None, None, Some(false)).await.unwrap());
    tokio::time::timeout(Duration::from_secs(5), session.prompt(
        "QUEUED_WORK_MUST_STAY_STOPPED", Some(PromptOptions {
            streaming_behavior: Some("followUp".into()), queue_if_busy: Some(true),
            ..Default::default()
        }),
    )).await.unwrap().unwrap();
    session.abort().await.unwrap();
    pause.release();
    let queued = session.get_session_action_snapshot().follow_ups;
    assert!(queued.contains(&"QUEUED_WORK_MUST_STAY_STOPPED".into()));
    let goal = session.goal_state();
    let file = session.session_file().unwrap();
    let stop_entries = |path: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(path).unwrap().lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|entry| entry["customType"] == "prime-agent.explicit-stop")
            .collect()
    };
    let stopped = stop_entries(&file);
    assert!(stopped.last().unwrap()["data"]["generation"].is_string());
    for command in ["/mode", "/mode toggle", "/mode toggle", "/mode direct"] {
        turn(&session, command).await;
        assert!(session.is_queued_work_suspended());
        assert_eq!(session.get_session_action_snapshot().follow_ups, queued);
        assert_eq!(session.goal_state().status, goal.status);
        assert_eq!(session.goal_state().objective, goal.objective);
        assert_eq!(f.captured.lock().unwrap().len(), 1, "a setting must not wake the model");
        assert_eq!(stop_entries(&file), stopped, "a setting must not clear the durable stop");
    }
    assert_eq!(session.get_active_tool_names(), ["bash", "edit"]);
    assert!(session.system_prompt().contains("Execution mode: Direct tools"));
    tokio::time::timeout(Duration::from_secs(5), session.prompt_and_wait("/mode direct", None))
        .await.expect("stopped mode commands acknowledge completion").unwrap();
    assert!(session.is_queued_work_suspended());
    assert_eq!(session.get_session_action_snapshot().follow_ups, queued);
    assert_eq!(stop_entries(&file), stopped);
    // Two concurrent F6 presses are serialized, so neither toggle is lost.
    let (a, b) = tokio::join!(
        session.prompt("/mode toggle", None), session.prompt("/mode toggle", None)
    );
    a.unwrap(); b.unwrap();
    assert_eq!(session.get_active_tool_names(), ["bash", "edit"]);
    assert!(session.prompt("/compact", None).await.is_err());
    session.dispose_async(Some(false)).await;
    let reopened = f.session(Some(&file), false).await;
    assert!(reopened.is_queued_work_suspended());
    assert_eq!(reopened.get_active_tool_names(), ["bash", "edit"]);
    turn(&reopened, "/mode toggle").await;
    assert_eq!(reopened.get_active_tool_names(), ["ipython"]);
    assert!(reopened.is_queued_work_suspended());
    assert_eq!(f.captured.lock().unwrap().len(), 1);
    assert_eq!(stop_entries(&file), stopped);
    reopened.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn stopped_mode_switch_rejects_invalid_restricted_and_paused_changes() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    for restricted in [false, true] {
        let session = f.session(None, restricted).await;
        session.abort().await.unwrap();
        let before = session.system_prompt();
        assert!(session.prompt("/mode unknown", None).await.is_err());
        if restricted {
            assert!(session.prompt("/mode direct", None).await.is_err());
        }
        let pause = session.acquire_session_input_pause();
        assert!(session.prompt("/mode direct", None).await.unwrap_err().contains("paused"));
        pause.release();
        assert!(session.prompt("/mode direct", Some(PromptOptions {
            internal_prompt: Some(true), ..Default::default()
        })).await.is_err());
        assert_eq!(session.system_prompt(), before);
        assert_eq!(session.get_active_tool_names(), ["ipython"]);
        assert!(session.is_queued_work_suspended());
        assert!(f.captured.lock().unwrap().is_empty());
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
#[ignore = "requires PRIME_AGENT_KERNEL_PYTHON pointing to a prepared runtime"]
async fn real_python_state_survives_a_round_trip_through_direct_tools() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    assert!(std::env::var_os("PRIME_AGENT_KERNEL_PYTHON").is_some());
    let f = Fixture::new();
    let session = f.session(None, false).await;
    f.reply(
        Some((
            "ipython",
            json!({"code":"mode_switch_sentinel = 41; print(mode_switch_sentinel)"}),
        )),
        None,
    );
    f.reply(None, None);
    turn(&session, "Create a Python variable").await;
    turn(&session, "/mode direct").await;
    f.reply(
        Some(("bash", json!({"command":"printf DIRECT_BETWEEN_CELLS"}))),
        None,
    );
    f.reply(None, None);
    turn(&session, "Run a direct command").await;
    turn(&session, "/mode ipython").await;
    f.reply(
        Some(("ipython", json!({"code":"print(mode_switch_sentinel + 1)"}))),
        None,
    );
    f.reply(None, None);
    turn(&session, "Read the retained Python variable").await;
    let history = serde_json::to_value(session.messages()).unwrap();
    let results: Vec<_> = history
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "toolResult")
        .collect();
    assert_eq!(results.len(), 3);
    assert!(
        results.iter().all(|m| m["isError"] == false),
        "{results:#?}"
    );
    assert!(
        results.last().unwrap().to_string().contains("42"),
        "{results:#?}"
    );
    f.assert_request(0, false);
    f.assert_request(2, true);
    f.assert_request(4, false);
    session.dispose_async(Some(false)).await;
}
