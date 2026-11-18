//! Fail-loud OCC-3 twin resolution for `rlm` spawns: no network, daemon,
//! provider, or live children. The fixture registers sibling providers that
//! share one model id (one OCC-3-configured, two ordinary), exactly like a
//! host whose models.json carries `dgx-glm53-occ3`, `dgx-glm53`, and
//! `ollama-cloud` entries for `glm-5.3`.
use super::*;
use pi_agent_core::types::{ShouldStopAfterTurnContext, StreamFn};
use pi_ai::types::{OnPayload, OnResponse};

type Listener = Arc<dyn Fn(AgentEvent, Option<CancellationToken>) -> BoxFuture<()> + Send + Sync>;

/// Minimal scripted agent: idle, no queue, `continue()` counted.
///
/// A plain turn is never dispatched by this suite. `prompt` rejects so that an
/// accidental dispatch is loud instead of silently completing.
struct ScriptedAgent {
    state: Mutex<AgentState>,
    continue_calls: std::sync::atomic::AtomicUsize,
    stream_fn: Mutex<StreamFn>,
}

impl ScriptedAgent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(AgentState::default()),
            continue_calls: std::sync::atomic::AtomicUsize::new(0),
            stream_fn: Mutex::new(pi_agent_core::agent::default_stream_fn()),
        })
    }
}

impl AgentHandle for ScriptedAgent {
    fn state(&self) -> AgentState {
        self.state.lock().unwrap().clone()
    }

    fn set_state(&self, state: AgentState) {
        *self.state.lock().unwrap() = state;
    }

    fn subscribe(&self, _listener: Listener) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn set_before_tool_call(&self, _hook: BeforeToolCallHook) {}
    fn set_after_tool_call(&self, _hook: AfterToolCallHook) {}
    fn set_get_continuation_messages(&self, _hook: GetContinuationMessagesHook) {}
    fn set_before_request(&self, _hook: BeforeRequestHook) {}
    fn set_should_stop_before_turn(&self, _hook: Arc<dyn Fn() -> bool + Send + Sync>) {}
    fn set_should_stop_after_turn(
        &self,
        _hook: Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxFuture<bool> + Send + Sync>,
    ) {
    }

    fn set_stream_fn(&self, stream_fn: StreamFn) {
        *self.stream_fn.lock().unwrap() = stream_fn;
    }

    fn stream_fn(&self) -> StreamFn {
        self.stream_fn.lock().unwrap().clone()
    }

    fn abort(&self) {}

    fn wait_for_idle(&self) -> BoxFuture<()> {
        Box::pin(async {})
    }

    fn prompt(&self, _messages: Vec<AgentMessage>) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Err("scripted agent does not run turns".to_string()) })
    }

    fn continue_(&self) -> BoxFuture<Result<(), AgentContinueError>> {
        self.continue_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn is_streaming(&self) -> bool {
        self.state.lock().unwrap().is_streaming
    }

    fn has_queued_messages(&self) -> bool {
        false
    }

    fn clear_all_queues(&self) {}

    fn remove_queued_messages(
        &self,
        _predicate: Arc<dyn Fn(&AgentMessage) -> bool + Send + Sync>,
    ) -> Vec<AgentMessage> {
        Vec::new()
    }

    fn follow_up(&self, _message: AgentMessage) {}
    fn set_follow_up_mode(&self, _mode: String) {}
    fn set_steering_mode(&self, _mode: String) {}

    fn set_convert_to_llm(
        &self,
        _convert: Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<Message>> + Send + Sync>,
    ) {
    }

    fn set_transform_context(
        &self,
        _transform: Arc<
            dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    ) {
    }

    fn set_get_api_key(
        &self,
        _get_api_key: Arc<dyn Fn(String) -> BoxFuture<Option<String>> + Send + Sync>,
    ) {
    }

    fn set_on_payload(&self, _hook: OnPayload) {}
    fn set_on_response(&self, _hook: OnResponse) {}
    fn set_tool_execution(&self, _mode: String) {}
    fn performance_metrics(&self) -> Option<AgentLoopPerformanceMetrics> {
        None
    }
    fn set_performance_metrics(&self, _metrics: Option<AgentLoopPerformanceMetrics>) {}
    fn signal(&self) -> Option<CancellationToken> {
        None
    }
}

fn occ3_capability() -> pi_ai::types::Occ3Capability {
    pi_ai::types::Occ3Capability {
        protocol: pi_ai::types::OCC3_PROTOCOL_V1.to_string(),
        connector: "sglang".to_string(),
        endpoint: "https://fixture.invalid/occ3".to_string(),
        api_version: pi_ai::types::OCC3_API_VERSION_V1.to_string(),
        same_model_only: true,
        enabled: true,
        validation: pi_ai::types::Occ3Validation::LiveVerified,
        fallback: Vec::new(),
        method: None,
        artifact_digest: None,
    }
}

/// One authenticated synthetic provider exposing `glm-5.3`, optionally with
/// an OCC-3 capability block.
fn register_glm_provider(
    registry: &mut crate::core::model_registry::ModelRegistry,
    provider: &str,
    occ3: bool,
) {
    registry
        .register_provider(
            provider,
            crate::core::model_registry::ProviderConfigInput {
                base_url: Some("http://127.0.0.1:9/v1".to_string()),
                api_key: Some("synthetic-key".to_string()),
                api: Some("openai-completions".to_string()),
                models: Some(vec![crate::core::model_registry::ModelDefinition {
                    id: "glm-5.3".to_string(),
                    name: Some("GLM 5.3".to_string()),
                    occ3: occ3.then(occ3_capability),
                    ..Default::default()
                }]),
                ..Default::default()
            },
        )
        .expect("register fixture provider");
}

/// The measured catalog shape: one model id under an OCC-3-configured
/// provider plus two ordinary providers.
fn twin_registry() -> crate::core::model_registry::ModelRegistry {
    let auth = crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None);
    let mut registry = crate::core::model_registry::ModelRegistry::in_memory(auth);
    register_glm_provider(&mut registry, "dgx-glm53", false);
    register_glm_provider(&mut registry, "dgx-glm53-occ3", true);
    register_glm_provider(&mut registry, "ollama-cloud", false);
    registry
}

/// A catalog with no OCC-3 entry at all: spawns keep the previous behavior.
fn plain_registry() -> crate::core::model_registry::ModelRegistry {
    let auth = crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None);
    let mut registry = crate::core::model_registry::ModelRegistry::in_memory(auth);
    register_glm_provider(&mut registry, "solo", false);
    registry
}

fn session_with_registry(
    registry: crate::core::model_registry::ModelRegistry,
) -> Arc<AgentSession> {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();
    let settings = Arc::new(Mutex::new(
        crate::core::settings_manager::SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    ));
    let manager =
        crate::core::session_manager::SessionManager::in_memory(Some(&cwd), Some(&cwd)).unwrap();
    let loader = Arc::new(crate::core::resource_loader::DefaultResourceLoader::new(
        crate::core::resource_loader::DefaultResourceLoaderOptions {
            cwd: cwd.clone(),
            agent_dir: cwd.clone(),
            settings_manager: Some(settings.clone()),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        },
    ));
    AgentSession::new(AgentSessionConfig {
        agent: ScriptedAgent::new() as Arc<dyn AgentHandle>,
        session_manager: Arc::new(Mutex::new(manager)),
        settings_manager: settings,
        service_tier_preference: None,
        cwd: cwd.clone(),
        agent_dir: Some(cwd.clone()),
        scoped_models: None,
        resource_loader: loader,
        custom_tools: None,
        model_registry: Arc::new(Mutex::new(registry)),
        initial_active_tool_names: None,
        allowed_tool_names: None,
        include_goals: Some(false),
        agent_message_controller: None,
        agent_observe_controller: None,
        include_compact_skill: Some(false),
        rlm_heartbeat_controller: None,
        mcp_manager: None,
        base_tools_override: Some(Vec::new()),
        extension_runner_ref: None,
        session_start_event: None,
        rlm_depth: Some(1),
        rlm_max_depth: Some(2),
        rlm_session_dir: None,
        rlm_parent_node_id: None,
        rlm_parent_agent: None,
        semantic_parent_session_id: None,
        semantic_spawned_by_request_id: None,
        subagent_runtime_host: None,
        autonomous: None,
        prewarm_ipython_kernel: Some(false),
        auto_refine_reviewer: None,
        serialized_refine: None,
        initial_goal: None,
    })
    .unwrap()
}

/// A session whose current (parent) model is the named fixture provider's
/// `glm-5.3`.
async fn session_on(
    registry: crate::core::model_registry::ModelRegistry,
    provider: &str,
) -> Arc<AgentSession> {
    let parent = registry.find(provider, "glm-5.3").expect("fixture model");
    let session = session_with_registry(registry);
    {
        let mut state = session.agent.state();
        state.model = parent;
        session.agent.set_state(state);
    }
    session
}

const TWIN_REFUSAL: &str = "Requested subagent model \"dgx-glm53/glm-5.3\" has no OCC-3 configuration, but an OCC-3-configured entry with the same model id exists. Children on this entry never receive the native occ3_handoff tool. Pass the exact OCC-3 selector \"dgx-glm53-occ3/glm-5.3\" instead";

#[tokio::test]
async fn inherit_from_ordinary_parent_with_occ3_twin_fails_loud() {
    let session = session_on(twin_registry(), "dgx-glm53").await;
    let error = session
        .resolve_rlm_subagent_model(None, "subagent")
        .await
        .err().expect("inheriting the ordinary twin must refuse the spawn");
    assert_eq!(error, TWIN_REFUSAL);
}

#[tokio::test]
async fn explicit_ordinary_selector_with_occ3_twin_fails_loud() {
    let session = session_on(twin_registry(), "ollama-cloud").await;
    let error = session
        .resolve_rlm_subagent_model(Some("dgx-glm53/glm-5.3"), "subagent")
        .await
        .err().expect("the ordinary twin must refuse the spawn");
    assert_eq!(error, TWIN_REFUSAL);
}

#[tokio::test]
async fn explicit_occ3_selector_resolves() {
    let session = session_on(twin_registry(), "dgx-glm53").await;
    let selection = session
        .resolve_rlm_subagent_model(Some("dgx-glm53-occ3/glm-5.3"), "subagent")
        .await
        .expect("the OCC-3 entry must resolve");
    assert_eq!(selection.model.provider, "dgx-glm53-occ3");
    assert_eq!(selection.model.id, "glm-5.3");
}

#[tokio::test]
async fn inherit_from_occ3_parent_resolves() {
    let session = session_on(twin_registry(), "dgx-glm53-occ3").await;
    let selection = session
        .resolve_rlm_subagent_model(None, "subagent")
        .await
        .expect("an OCC-3-configured parent must inherit cleanly");
    assert_eq!(selection.model.provider, "dgx-glm53-occ3");
    assert_eq!(selection.model.id, "glm-5.3");
}

#[tokio::test]
async fn unambiguous_id_without_occ3_twin_resolves_unchanged() {
    let session = session_on(plain_registry(), "solo").await;
    let selection = session
        .resolve_rlm_subagent_model(None, "subagent")
        .await
        .expect("inheritance without any OCC-3 twin keeps the previous behavior");
    assert_eq!(selection.model.provider, "solo");
    let selection = session
        .resolve_rlm_subagent_model(Some("solo/glm-5.3"), "subagent")
        .await
        .expect("an explicit ordinary selector without an OCC-3 twin keeps the previous behavior");
    assert_eq!(selection.model.provider, "solo");
    assert_eq!(selection.model.id, "glm-5.3");
}

#[tokio::test]
async fn bare_model_id_error_lists_provider_selectors_occ3_first() {
    let session = session_on(twin_registry(), "dgx-glm53").await;
    let error = session
        .resolve_rlm_subagent_model(Some("glm-5.3"), "subagent")
        .await
        .err().expect("a bare model id never matches a provider/id selector");
    assert_eq!(
        error,
        "Requested subagent model \"glm-5.3\" is unavailable, unauthenticated, or expired; the model id \"glm-5.3\" is available from multiple providers: \"dgx-glm53-occ3/glm-5.3\", \"dgx-glm53/glm-5.3\", \"ollama-cloud/glm-5.3\""
    );
}

#[tokio::test]
async fn selector_reference_miss_keeps_the_plain_unavailable_message() {
    let session = session_on(twin_registry(), "dgx-glm53").await;
    let error = session
        .resolve_rlm_subagent_model(Some("dgx-glm53/glm-9"), "subagent")
        .await
        .err().expect("an unknown selector must stay unavailable");
    assert_eq!(
        error,
        "Requested subagent model \"dgx-glm53/glm-9\" is unavailable, unauthenticated, or expired"
    );
}
