//! Search decisions must survive queued TurnStart bookkeeping and reach the provider.
//! Both providers are synthetic; session history remains unchanged.

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// These tests mutate the process-global PRIME_AGENT_CODING_AGENT_DIR env and
/// the process-global mock-control fixture scratchpad, so the whole binary
/// serializes with one test-lifetime mutex (the landed hint/control discipline).
static SEARCH_ENV_LOCK: Mutex<()> = Mutex::new(());

use pi_agent_core::types::AgentMessage;
use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{
    AssistantMessage, ContentBlock, ImageOrTextContent, Message, TextContent, ToolCall,
    ToolResultMessage, UserContent, UserMessage,
};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_jev::config::{JevFeatures, JevMode, JevSettings, JevSettingsStore};
use pi_jev::types::{Answer, SystemOneRequest, SystemOneResponse, Usage};
use serde_json::{json, Value};

const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";

struct EnvGuard {
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn new(agent_dir: &std::path::Path) -> Self {
        let previous = std::env::var_os(ENV_AGENT_DIR);
        std::env::set_var(ENV_AGENT_DIR, agent_dir);
        Self { previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var(ENV_AGENT_DIR, value),
            None => std::env::remove_var(ENV_AGENT_DIR),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider request capture (the ACTUAL outgoing provider context)
// ---------------------------------------------------------------------------

/// Serialized provider-context messages of one outgoing request: proves what
/// the provider would actually receive (annotation composition on request
/// copies).
#[derive(Clone)]
struct CapturedRequest {
    messages: Vec<Value>,
}

fn capture_factory(
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
) -> pi_ai::providers::faux::FauxResponseFactory {
    Arc::new(
        move |context: &pi_ai::types::Context,
              _options: Option<&pi_ai::types::StreamOptions>,
              _state: &pi_ai::providers::faux::FauxState,
              _model: &pi_ai::types::Model|
              -> futures::future::BoxFuture<'_, AssistantMessage> {
            let messages = context
                .messages
                .iter()
                .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
                .collect();
            captured.lock().unwrap().push(CapturedRequest { messages });
            let message = faux_assistant_message(
                FauxAssistantContent::Text("SEARCH_TERMINAL_END".to_string()),
                None,
            );
            Box::pin(async move { message })
        },
    )
}

struct Fixture {
    session: Arc<AgentSession>,
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    _dir: tempfile::TempDir,
    _guard: EnvGuard,
    _env_lock: std::sync::MutexGuard<'static, ()>,
}

fn save_search_settings(
    agent_dir: &std::path::Path,
    mode: JevMode,
    filtering: bool,
    reranking: bool,
) {
    let settings = JevSettings {
        global_default: Some(mode),
        features: JevFeatures {
            tool_requirement: false,
            complexity: false,
            code_search_relevance: true,
            code_search_filtering: filtering,
            code_search_reranking: reranking,
            skill_suggestion: false,
            ..Default::default()
        },
        transport: Some("mock-control".to_string()),
        ..Default::default()
    };
    JevSettingsStore::new(agent_dir).save(&settings).unwrap();
    pi_coding_agent::core::jev_bridge::invalidate_settings_cache();
}

fn search_responder() -> Box<dyn Fn(&SystemOneRequest) -> Option<SystemOneResponse> + Send + Sync> {
    Box::new(|request| {
        let mut answers = std::collections::BTreeMap::new();
        for id in request.questions.keys() {
            let answer = if id.starts_with("code_search_relevance.") {
                let choice = if id.ends_with(".0") { "drop" } else { "keep" };
                Answer::Choice {
                    choice: choice.to_string(),
                    probabilities: [
                        ("drop".to_string(), if choice == "drop" { 1.0 } else { 0.0 }),
                        ("keep".to_string(), if choice == "keep" { 1.0 } else { 0.0 }),
                    ]
                    .into_iter()
                    .collect(),
                    confidence: 1.0,
                }
            } else if id.starts_with("code_search_rerank.") {
                // Reverse the surviving candidates, independent of filtering.
                let ordinal: usize = id.rsplit('.').next().unwrap().parse().unwrap();
                Answer::Noul {
                    noul: (ordinal + 1) as f64 / 4.0,
                }
            } else {
                return None;
            };
            answers.insert(id.clone(), answer);
        }
        Some(SystemOneResponse {
            model: "jev-search-fixture".to_string(),
            answers,
            usage: Usage {
                input_tokens: Some(5),
                output_tokens: Some(5),
            },
            ..Default::default()
        })
    })
}

async fn build_fixture(mode: JevMode, filtering: bool, reranking: bool) -> Fixture {
    let env_lock = SEARCH_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let dir = tempfile::Builder::new()
        .prefix("jev-search-ctx-")
        .tempdir_in(std::env::temp_dir())
        .expect("scenario tempdir");
    let cwd = dir.path().join("workspace");
    let agent_dir = dir.path().join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let guard = EnvGuard::new(&agent_dir);
    pi_coding_agent::core::jev_bridge::debug_control_fixture_reset();

    save_search_settings(&agent_dir, mode, filtering, reranking);

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some("search-context-faux".to_string()),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let model = provider.get_model();

    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        json!({
            "autoRefine": {"enabled": false},
            "retry": {"enabled": false},
            "compaction": {"enabled": false},
            "telemetryEnabled": false,
            "agentTracesEnabled": false,
            "quietStartup": true,
        })
        .as_object()
        .unwrap()
        .clone(),
    )));
    let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        None,
    )));
    auth_storage
        .lock()
        .await
        .set_runtime_api_key(&model.provider, "synthetic-search-key");
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
    model_registry
        .lock()
        .expect("model registry poisoned")
        .set_runtime_api_key(&model.provider, "synthetic-search-key");

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(
            Some(&cwd.to_string_lossy()),
            Some(&agent_dir.to_string_lossy()),
        )
        .expect("in-memory session manager"),
    ));
    // Seed the SAME isolated manager before SDK session creation. The SDK's
    // real restore path installs these typed messages into Agent state; no
    // Python process or tool implementation runs in this fixture.
    {
        let mut manager = session_manager.lock().expect("session manager poisoned");
        for message in search_messages() {
            manager
                .append_message(message)
                .expect("seed history message");
        }
    }

    let loader_options = DefaultResourceLoaderOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: agent_dir.to_string_lossy().to_string(),
        no_extensions: true,
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        bundled_skills_dir: Some(None),
        ..Default::default()
    };
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: Some(agent_dir.to_string_lossy().to_string()),
        auth_storage: Some(Arc::clone(&auth_storage)),
        settings_manager: Some(Arc::clone(&settings)),
        model_registry: Some(Arc::clone(&model_registry)),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(loader_options),
    })
    .await
    .expect("services");

    let captured: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let factory = capture_factory(Arc::clone(&captured));
    provider.set_responses(
        (0..12)
            .map(|_| FauxResponseStep::Factory(Arc::clone(&factory)))
            .collect(),
    );

    let creation = AgentSessionCreationOptions {
        model: Some(model.clone()),
        prewarm_ipython_kernel: Some(false),
        telemetry_disabled: Some(true),
        ..Default::default()
    };
    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager,
        session_start_event: None,
        creation,
    })
    .await
    .expect("agent session");

    // Scripted search answers through the integrated mock-control transport.
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(search_responder()));

    Fixture {
        session: created.session,
        captured,
        _dir: dir,
        _guard: guard,
        _env_lock: env_lock,
    }
}

async fn turn(session: &Arc<AgentSession>, text: &str) {
    let prompt = session.prompt(text, None::<PromptOptions>);
    tokio::time::timeout(Duration::from_secs(120), prompt)
        .await
        .unwrap_or_else(|_| panic!("prompt never returned"))
        .unwrap_or_else(|error| panic!("prompt rejected: {error}"));
    tokio::time::timeout(Duration::from_secs(120), session.wait_for_headless_idle())
        .await
        .unwrap_or_else(|_| panic!("session never reached idle"))
        .expect("idle");
}

fn user_message(text: &str) -> AgentMessage {
    UserMessage::new(UserContent::Text(text.into()), 1).into()
}

fn ipython_call(id: &str, code: &str) -> AgentMessage {
    AssistantMessage {
        content: vec![ContentBlock::ToolCall(ToolCall::new(
            id,
            "ipython",
            json!({"code": code}).as_object().cloned().unwrap(),
        ))],
        stop_reason: "toolUse".to_string(),
        ..Default::default()
    }
    .into()
}

/// A REAL-shaped ipython tool result: the single original text block plus the
/// production `details` serialization the eligibility scans read
/// (`status: "ok"`, non-empty `stdout`, no stderr/result/background output).
fn ipython_result(id: &str, text: &str) -> AgentMessage {
    Message::ToolResult(ToolResultMessage {
        role: "toolResult".to_string(),
        tool_call_id: id.to_string(),
        tool_name: "ipython".to_string(),
        content: vec![ImageOrTextContent::Text(TextContent::new(text))],
        details: Some(json!({
            "status": "ok",
            "stdout": text,
            "kernelRestarted": false,
        })),
        is_error: false,
        timestamp: 2,
    })
    .into()
}

fn search_messages() -> Vec<AgentMessage> {
    let envelope = json!({"schema":"rlm.code-search/1", "candidates": [
        {"kind":"file", "path":"palette.py", "line":1, "snippet":"def theme(): return 'navy'"},
        {"kind":"file", "path":"retry.py", "line":1, "snippet":"def retry_delay(n): return min(2 ** n, 60)"},
        {"kind":"file", "path":"retry_test.py", "line":1, "snippet":"assert retry_delay(100) == 60"}
    ]});
    vec![
        user_message("Find retry backoff"),
        ipython_call("search-call", "present(candidates)"),
        ipython_result("search-call", &envelope.to_string()),
    ]
}

fn search_envelope(messages: &[Value]) -> Value {
    let message = messages
        .iter()
        .find(|m| m["role"] == "toolResult" && m["toolCallId"] == "search-call")
        .expect("search result");
    serde_json::from_str(message["content"][0]["text"].as_str().unwrap()).unwrap()
}

#[tokio::test]
async fn request_turn_and_outgoing_search_projection_are_independent_of_skill_suggestions() {
    use pi_coding_agent::core::extensions::types::{ExtensionEvent, TurnStartPayload};
    use pi_coding_agent::core::jev_bridge::{before_request_assessment, current_skill_hint_stamp};
    for (mode, filtering, reranking, expected) in [
        (
            JevMode::Active,
            true,
            false,
            vec!["retry.py", "retry_test.py"],
        ),
        (
            JevMode::Active,
            false,
            true,
            vec!["retry_test.py", "retry.py", "palette.py"],
        ),
        (
            JevMode::CompareAndActive,
            true,
            true,
            vec!["retry_test.py", "retry.py"],
        ),
        (
            JevMode::Compare,
            true,
            true,
            vec!["palette.py", "retry.py", "retry_test.py"],
        ),
    ] {
        let fixture = build_fixture(mode, filtering, reranking).await;
        let session = &fixture.session;
        let runner = session.extension_runner().unwrap();
        let session_id = session.session_id();
        // Deterministic version of the live race: queued events lag the awaited
        // request boundary. Neither an older event nor its eventual catch-up
        // may change the request identity while Jev is deciding.
        runner
            .emit(ExtensionEvent::TurnStart(TurnStartPayload {
                turn_index: 0.0,
                timestamp: 0.0,
            }))
            .await;
        before_request_assessment(&session_id, 1, None).await;
        assert_eq!(
            current_skill_hint_stamp(&session_id, &[]).turn,
            1,
            "request hook must advance with skill suggestions off"
        );
        runner
            .emit(ExtensionEvent::TurnStart(TurnStartPayload {
                turn_index: 0.0,
                timestamp: 0.0,
            }))
            .await;
        assert_eq!(
            current_skill_hint_stamp(&session_id, &[]).turn,
            1,
            "late events cannot rewind a request"
        );
        runner
            .emit(ExtensionEvent::TurnStart(TurnStartPayload {
                turn_index: 1.0,
                timestamp: 0.0,
            }))
            .await;
        assert_eq!(current_skill_hint_stamp(&session_id, &[]).turn, 1);
        // A new run resets the provider index to zero; max(old, new) is invalid.
        before_request_assessment(&session_id, 0, None).await;
        assert_eq!(current_skill_hint_stamp(&session_id, &[]).turn, 0);
        let original: Vec<Value> = search_messages()
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        turn(session, "Find retry backoff").await;
        let captured = fixture.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let envelope = search_envelope(&captured[0].messages);
        let paths: Vec<&str> = envelope["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["path"].as_str().unwrap())
            .collect();
        assert_eq!(
            paths, expected,
            "{mode:?}: filtering={filtering}, reranking={reranking}"
        );
        drop(captured);
        let history: Vec<Value> = session
            .agent
            .state()
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        assert_eq!(
            search_envelope(&history),
            search_envelope(&original),
            "request projection must preserve history"
        );
        // An enabled suggestion feature with an empty roster also returns early.
        let agent_dir = fixture._dir.path().join("agent");
        let store = JevSettingsStore::new(&agent_dir);
        let mut settings = store.load();
        settings.features.skill_suggestion = true;
        store.save(&settings).unwrap();
        pi_coding_agent::core::jev_bridge::invalidate_settings_cache();
        before_request_assessment(&session_id, 2, None).await;
        assert_eq!(current_skill_hint_stamp(&session_id, &[]).turn, 2);
        assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
        session.dispose_async(Some(false)).await;
        pi_coding_agent::core::jev_bridge::debug_control_fixture_reset();
    }

    let fixture = build_fixture(JevMode::Active, true, true).await;
    let agent_dir = fixture._dir.path().join("agent");
    let respond = search_responder();
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        move |request| {
            if request
                .questions
                .keys()
                .any(|id| id.starts_with("code_search_relevance."))
            {
                let store = JevSettingsStore::new(&agent_dir);
                let mut settings = store.load();
                settings.global_default = Some(JevMode::Off);
                store.save(&settings).unwrap();
            }
            respond(request)
        },
    )));
    turn(&fixture.session, "Find retry backoff").await;
    let captured = fixture.captured.lock().unwrap();
    let original: Vec<Value> = search_messages()
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect();
    assert_eq!(
        search_envelope(&captured[0].messages),
        search_envelope(&original),
        "a mode change during the decision must still refuse the projection"
    );
    drop(captured);
    fixture.session.dispose_async(Some(false)).await;
    pi_coding_agent::core::jev_bridge::debug_control_fixture_reset();
}
