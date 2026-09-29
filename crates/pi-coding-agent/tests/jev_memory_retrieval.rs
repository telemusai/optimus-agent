//! Synthetic memories, a local coding provider and the real Jev bridge/context hook.
use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_coding_agent::core::{
    agent_session::AgentSession,
    agent_session_services::*,
    auth_storage::{AuthStorage, AuthStorageData},
    jev_bridge,
    memory::service::MemoryService,
    model_registry::ModelRegistry,
    resource_loader::DefaultResourceLoaderOptions,
    session_manager::SessionManager,
    settings_manager::SettingsManager,
};
use pi_jev::config::{JevMode, JevSettings, JevSettingsStore};
use pi_jev::types::{Answer, QuestionSpec, SystemOneResponse, Usage};
use serde_json::{json, Value};

use std::sync::{Arc, Mutex};
use std::time::Duration;

static ENV_LOCK: Mutex<()> = Mutex::new(());
struct EnvGuard(Option<std::ffi::OsString>);
impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(old) = self.0.take() {
            std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", old);
        } else {
            std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR");
        }
        jev_bridge::invalidate_settings_cache();
        jev_bridge::debug_control_fixture_reset();
    }
}

async fn fixture(agent_dir: &str, cwd: &str) -> (Arc<AgentSession>, Arc<Mutex<Vec<Value>>>) {
    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some("memory-fixture".into()),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let model = provider.get_model();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let seen = captured.clone();
    provider.set_responses(vec![FauxResponseStep::Factory(Arc::new(
        move |context, _, _, _| {
            seen.lock()
                .unwrap()
                .push(serde_json::to_value(&context.messages).unwrap());
            Box::pin(async {
                faux_assistant_message(FauxAssistantContent::Text("DONE".into()), None)
            })
        },
    ))]);
    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        json!({
            "autoRefine":{"enabled":false},"retry":{"enabled":false},"compaction":{"enabled":false},
            "telemetryEnabled":false,"agentTracesEnabled":false,"quietStartup":true
        })
        .as_object()
        .unwrap()
        .clone(),
    )));
    let auth = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        None,
    )));
    auth.lock()
        .await
        .set_runtime_api_key(&model.provider, "synthetic-fixture-key");
    let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
    registry
        .lock()
        .unwrap()
        .set_runtime_api_key(&model.provider, "synthetic-fixture-key");
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.into(),
        agent_dir: Some(agent_dir.into()),
        auth_storage: Some(auth),
        settings_manager: Some(settings.clone()),
        model_registry: Some(registry),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(DefaultResourceLoaderOptions {
            cwd: cwd.into(),
            agent_dir: agent_dir.into(),
            no_extensions: true,
            extension_factories: vec![
                pi_coding_agent::core::extensions::builtin::memory::create_memory_extension(
                    agent_dir.into(), settings.clone(),
                ),
            ],
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        }),
    })
    .await
    .unwrap();
    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager: Arc::new(Mutex::new(
            SessionManager::in_memory(Some(cwd), Some(agent_dir)).unwrap(),
        )),
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .unwrap();
    (created.session, captured)
}

async fn seed(memory: &MemoryService) {
    memory
        .store
        .configure(&json!({"learning":false,"maxRecallEntries":1}))
        .await
        .unwrap();
    let proposal = json!({"eventId":"fixture","revision":0,"proposal":{
    "summary":"synthetic","rationale":"regression","expectedOutcome":"recall",
    "edits":[
        {"action":"create","kind":"memory","id":"literal","title":"frobnicator","content":"BASELINE_SENTINEL frobnicator literal match."},
        {"action":"create","kind":"memory","id":"semantic","title":"Widget initialization","content":"SEMANTIC_SENTINEL start the widget with the blue switch."},
        {"action":"create","kind":"memory","id":"unrelated","title":"Lunch","content":"UNRELATED_SENTINEL sandwich recipe."}
    ]}});
    memory
        .request("apply", proposal.as_object().unwrap(), None)
        .await
        .unwrap();
}

#[tokio::test]
async fn real_context_combines_recall_without_displacing_baseline_and_fails_open() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let previous = std::env::var_os("PRIME_AGENT_CODING_AGENT_DIR");
    let _env = EnvGuard(previous);
    for (mode, feature, transport, change_midflight, expect_extra) in [
        (JevMode::Active, true, "mock-control", false, true),
        (JevMode::CompareAndActive, true, "mock-control", false, true),
        (JevMode::Compare, true, "mock-control", false, false),
        (JevMode::Off, true, "mock-control", false, false),
        (JevMode::Active, false, "mock-control", false, false),
        (JevMode::Active, true, "mock-malformed", false, false),
        (JevMode::Active, true, "mock-control", true, false),
    ] {
        let root = tempfile::tempdir().unwrap();
        // Mixed case must remain usable on case-sensitive filesystems.
        let agent = root.path().join("AgentProfile");
        let cwd = root.path().join("workspace");
        std::fs::create_dir_all(&cwd).unwrap();
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", &agent);
        let mut settings = JevSettings::default();
        settings.global_default = Some(mode);
        settings.features.memory = feature;
        // Full Jev also enables this older filter: it must not remove normal recall.
        settings.features.memory_relevance = feature;
        settings.features.tool_requirement = false;
        settings.features.complexity = false;
        settings.transport = Some(transport.into());
        JevSettingsStore::new(&agent).save(&settings).unwrap();
        jev_bridge::invalidate_settings_cache();
        jev_bridge::debug_control_fixture_reset();
        let store_dir = agent.clone();
        jev_bridge::debug_control_fixture_set_respond(Some(Box::new(move |request| {
            if request
                .questions
                .keys()
                .any(|key| key.starts_with("memory_relevance."))
            {
                if change_midflight {
                    let store = JevSettingsStore::new(&store_dir);
                    let mut current = store.load();
                    current.features.memory = false;
                    store.save(&current).unwrap();
                }
                let answers = request
                    .questions
                    .iter()
                    .map(|(id, spec)| {
                        let QuestionSpec::Choice { criteria, .. } = spec else {
                            panic!("expected bounded relevance choice")
                        };
                        let index = id.rsplit_once('.').unwrap().1.parse::<usize>().unwrap();
                        let semantic = request.state["memories"][index]["content"]
                            .as_str()
                            .unwrap_or("")
                            .contains("SEMANTIC_SENTINEL");
                        let choice = if semantic { "keep" } else { "drop" };
                        (
                            id.clone(),
                            Answer::Choice {
                                choice: choice.into(),
                                confidence: 1.0,
                                probabilities: criteria
                                    .keys()
                                    .map(|key| (key.clone(), f64::from(key == choice)))
                                    .collect(),
                            },
                        )
                    })
                    .collect();
                return Some(SystemOneResponse {
                    model: request.model.clone(),
                    answers,
                    usage: Usage::default(),
                    answer_parse_skips: Vec::new(),
                    server_request_id: None,
                });
            }
            None
        })));
        let memory =
            MemoryService::new(&cwd.to_string_lossy(), &agent.to_string_lossy(), None).unwrap();
        seed(&memory).await;
        assert_eq!(memory.recall("frobnicator").ids.len(), 1);
        let (session, captured) = fixture(&agent.to_string_lossy(), &cwd.to_string_lossy()).await;
        tokio::time::timeout(Duration::from_secs(20), async {
            session.prompt("frobnicator", None).await.unwrap();
            session.wait_for_headless_idle().await.unwrap();
        })
        .await
        .expect("isolated context run timed out");
        let requests = captured.lock().unwrap();
        assert!(!requests.is_empty());
        let text = requests[0].to_string();
        assert!(text.contains("BASELINE_SENTINEL"), "{mode:?}: {text}");
        assert_eq!(
            text.contains("SEMANTIC_SENTINEL"),
            expect_extra,
            "{mode:?} {transport} changed={change_midflight}: {text}"
        );
        assert!(!text.contains("UNRELATED_SENTINEL"));
        if expect_extra {
            assert_eq!(text.matches("SEMANTIC_SENTINEL").count(), 1);
        }
        if mode == JevMode::Off || !feature {
            assert!(!jev_bridge::debug_control_fixture_calls()
                .iter()
                .any(|call| call.to_string().contains("memory_retrieval")));
        }
        drop(requests);
        session.dispose();
    }
}

#[test]
fn memory_toggle_is_independent_and_full_jev_enables_it() {
    use pi_jev::config::JevFeature;
    assert_eq!(JevFeature::parse("memory"), Some(JevFeature::Memory));
    let mut settings = JevSettings::default();
    assert!(!settings.features.memory);
    settings.features.set(JevFeature::Memory, true);
    assert!(settings.features.memory);
    assert!(!settings.features.memory_relevance);
    settings.features.set(JevFeature::Memory, false);
    settings.full_jev_install();
    assert!(settings.effective_features("fixture").memory);
    assert!(settings.effective_features("fixture").memory_relevance);
    settings.full_jev_remove();
    assert!(!settings.effective_features("fixture").memory);
}
