//! End-to-end Jev comparison-mode tests (fake provider, no network).
//!
//! Proves through the REAL session/extension-runner paths:
//! - Off parity: a dormant observer has zero clients, records or behavior deltas.
//! - Compare parity: same prompts, tool calls, model choice and stop behavior
//!   with Jev Compare observing through the deterministic mock transport.
//! - Isolation: maximum-confidence adversarial answers cannot mutate
//!   messages, model, tools or continuation, and cannot inject prompts.
//! - Compare actually observes: correlated records exist, applied=false.
//! - Immediate Off: flipping settings stops observing and scheduling.
//!
//! The four phases run in ONE test because the mode source reads the
//! process-level agent-dir env override; parallel phases would race on it.

#![allow(clippy::all)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_ai::providers::faux::{
    faux_assistant_message, faux_tool_call, register_faux_provider, FauxAssistantContent,
    FauxProviderRegistration, FauxResponseStep, FauxAssistantMessageOptions,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{AssistantMessage, ContentBlock};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::extensions::types::ToolDefinition;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Environment guard (isolated agent dir for the whole test)
// ---------------------------------------------------------------------------

const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";
const SYNTHETIC_KEY: &str = "synthetic-jev-e2e-key";

/// The observe-only event list the Jev adapter registers (bridge parity).
const JEV_EVENT_NAMES: [&str; 11] = [
    "session_start",
    "agent_start",
    "turn_start",
    "input",
    "tool_call",
    "tool_execution_start",
    "tool_execution_end",
    "message_end",
    "agent_end",
    "model_select",
    "session_shutdown",
];

struct EnvGuard {
    previous: Option<std::ffi::OsString>,
    agent_dir: std::path::PathBuf,
}

impl EnvGuard {
    fn new(root: &std::path::Path) -> Self {
        let previous = std::env::var_os(ENV_AGENT_DIR);
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::env::set_var(ENV_AGENT_DIR, &agent_dir);
        Self { previous, agent_dir }
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

fn write_settings(agent_dir: &std::path::Path, settings: serde_json::Value) {
    let mut complete = serde_json::to_value(pi_jev::config::JevSettings::default()).unwrap();
    complete.as_object_mut().unwrap().extend(settings.as_object().unwrap().clone());
    let jev_dir = agent_dir.join("jev");
    std::fs::create_dir_all(&jev_dir).unwrap();
    std::fs::write(
        jev_dir.join("jev-settings.json"),
        serde_json::to_string_pretty(&complete).unwrap(),
    )
    .unwrap();
    pi_coding_agent::core::jev_bridge::invalidate_settings_cache();
}

fn records_path(agent_dir: &std::path::Path) -> std::path::PathBuf {
    agent_dir.join("jev").join("records.jsonl")
}

fn read_record_count(agent_dir: &std::path::Path) -> usize {
    std::fs::read_to_string(records_path(agent_dir))
        .map(|text| text.lines().count())
        .unwrap_or(0)
}

/// Poll (bounded) until the record count stops growing, so settled Compare
/// work is not mistaken for a leak.
async fn wait_records_settled(agent_dir: &std::path::Path, at_least: usize) -> usize {
    for _ in 0..200 {
        let count = read_record_count(agent_dir);
        if count >= at_least {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let again = read_record_count(agent_dir);
            if again == count {
                return again;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    read_record_count(agent_dir)
}

async fn wait_jev_session_settled(session: &AgentSession) {
    for _ in 0..300 {
        let status = pi_coding_agent::core::jev_bridge::session_status_snapshot(&session.session_id());
        let pending = status.as_ref().map(|status| {
            status.get("in_flight").and_then(Value::as_u64).unwrap_or(0)
                + status.get("queue_depth").and_then(Value::as_u64).unwrap_or(0)
        }).unwrap_or(0);
        if pending == 0 {
            // Status changes immediately before the synchronous terminal record write.
            tokio::task::yield_now().await;
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("Jev scheduler did not reach a terminal state for {}", session.session_id());
}

// ---------------------------------------------------------------------------
// Session fixture (compaction-suite pattern: real session, faux provider)
// ---------------------------------------------------------------------------

struct Fixture {
    session: Arc<AgentSession>,
    provider: FauxProviderRegistration,
    /// Keeps the scratch dir alive for the session's lifetime.
    _dir: tempfile::TempDir,
}

#[derive(Default)]
struct CreationOverrides {
    tools: Option<Vec<String>>,
    custom_tools: Option<Vec<ToolDefinition>>,
    reasoning: bool,
}

async fn build_session(provider_name: &str, overrides: CreationOverrides) -> Fixture {
    let scratch = std::env::temp_dir();
    let dir = tempfile::Builder::new()
        .prefix("jev-e2e-")
        .tempdir_in(scratch)
        .unwrap();
    let cwd = dir.path().join("workspace");
    let agent_dir = dir.path().join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let cwd_string = cwd.to_string_lossy().to_string();
    let agent_dir_string = agent_dir.to_string_lossy().to_string();

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(provider_name.to_string()),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let mut model = provider.get_model();
    model.reasoning = overrides.reasoning;

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
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(AuthStorage::in_memory(
        AuthStorageData::new(),
        None,
    ))));
    auth_storage
        .lock()
        .await
        .set_runtime_api_key(&model.provider, SYNTHETIC_KEY);
    model_registry
        .lock()
        .expect("model registry poisoned")
        .set_runtime_api_key(&model.provider, SYNTHETIC_KEY);

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(Some(&cwd_string), Some(&agent_dir_string)).unwrap(),
    ));

    let loader_options = DefaultResourceLoaderOptions {
        cwd: cwd_string.clone(),
        agent_dir: agent_dir_string.clone(),
        no_extensions: true,
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        bundled_skills_dir: Some(None),
        ..Default::default()
    };
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd_string.clone(),
        agent_dir: Some(agent_dir_string),
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

    let creation = AgentSessionCreationOptions {
        model: Some(model.clone()),
        tools: overrides.tools,
        custom_tools: overrides.custom_tools,
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

    Fixture {
        session: created.session,
        provider,
        _dir: dir,
    }
}

fn reply(index: usize) -> AssistantMessage {
    faux_assistant_message(
        FauxAssistantContent::Text(format!("JEV_E2E_REPLY_{index:02}_END")),
        None,
    )
}

fn tool_call_reply() -> AssistantMessage {
    faux_assistant_message(
        FauxAssistantContent::Blocks(vec![ContentBlock::ToolCall(faux_tool_call(
            "jev_e2e_probe",
            Map::new(),
            None,
        ))]),
        Some(FauxAssistantMessageOptions {
            stop_reason: Some("toolUse".to_string()),
            ..Default::default()
        }),
    )
}

/// The harmless custom probe tool used for tool-call parity.
fn probe_tool() -> ToolDefinition {
    ToolDefinition {
        name: "jev_e2e_probe".to_string(),
        label: "Jev E2E probe".to_string(),
        description: "Returns static text; used for tool-call parity".to_string(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
        render_shell: None,
        replay_built_in_tool_name: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(|_call_id, _args, _signal, _update, _ctx| {
            Box::pin(async {
                Ok(pi_agent_core::types::AgentToolResult::new(
                    vec![pi_agent_core::types::ContentBlock::text("PROBE_OK")],
                    json!({}),
                ))
            })
        }),
        render_call: None,
        render_result: None,
    }
}

/// One prompt/response cycle through the real session, bounded.
async fn turn(session: &Arc<AgentSession>, text: &str) {
    let prompt = session.prompt(text, None::<PromptOptions>);
    tokio::time::timeout(Duration::from_secs(120), prompt)
        .await
        .unwrap_or_else(|_| panic!("prompt never returned"))
        .unwrap_or_else(|error| panic!("prompt rejected: {error}"));
    tokio::time::timeout(Duration::from_secs(120), session.wait_for_headless_idle())
        .await
        .unwrap_or_else(|_| panic!("session never reached idle"))
        .expect("headless idle");
}

/// (role, text) pairs for parity comparisons.
fn message_shape(message: &pi_agent_core::types::AgentMessage) -> (String, String) {
    match message {
        pi_agent_core::types::AgentMessage::Message(message) => match message {
            pi_ai::types::Message::Assistant(assistant) => (
                "assistant".to_string(),
                assistant
                    .content
                    .iter()
                    .filter_map(|block| block.as_text().map(|text| text.text.clone()))
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            pi_ai::types::Message::User(user) => match &user.content {
                pi_ai::types::UserContent::Text(text) => ("user".to_string(), text.clone()),
                pi_ai::types::UserContent::Blocks(blocks) => (
                    "user".to_string(),
                    blocks
                        .iter()
                        .filter_map(|block| match block {
                            pi_ai::types::ImageOrTextContent::Text(text) => {
                                Some(text.text.clone())
                            }
                            pi_ai::types::ImageOrTextContent::Image(_) => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                ),
            },
            pi_ai::types::Message::ToolResult(result) => (
                "toolResult".to_string(),
                result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                        pi_ai::types::ImageOrTextContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            ),
        },
        pi_agent_core::types::AgentMessage::Custom(custom) => (
            format!("custom:{}", custom.role()),
            String::new(),
        ),
    }
}

fn shapes(session: &Arc<AgentSession>) -> Vec<(String, String)> {
    session.messages().iter().map(message_shape).collect()
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jev_compare_e2e_parity_and_isolation() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let root = tempfile::Builder::new()
        .prefix("jev-e2e-root-")
        .tempdir()
        .unwrap();
    let guard = EnvGuard::new(root.path());
    let agent_dir = guard.agent_dir.clone();

    // ------------------------------------------------------------------
    // Phase 1: Off session — dormant read-only adapter, zero records/client.
    // ------------------------------------------------------------------
    write_settings(&agent_dir, json!({ "global_default": "off" }));
    let off = build_session("jev-off-provider", CreationOverrides::default()).await;
    for event in [
        "session_start",
        "turn_start",
        "input",
        "tool_call",
        "tool_execution_start",
        "tool_execution_end",
        "message_end",
        "agent_end",
        "model_select",
        "session_shutdown",
    ] {
        assert!(
            off.session.has_extension_handlers(event),
            "Dormant adapter must support later activation for {event}"
        );
    }
    off.provider.set_responses(vec![
        FauxResponseStep::Message(reply(0)),
        FauxResponseStep::Message(reply(1)),
    ]);
    turn(&off.session, "e2e prompt one").await;
    turn(&off.session, "e2e prompt two").await;
    assert_eq!(off.provider.call_count(), 2, "two provider calls in Off");
    assert!(
        !records_path(&agent_dir).exists(),
        "Off must not create record files"
    );
    let off_model = off.session.model().expect("off model").id;
    let off_thinking = off.session.thinking_level();
    let off_shapes = shapes(&off.session);
    assert!(off_shapes.iter().any(|(_, text)| text == "e2e prompt one"));
    assert!(off_shapes.iter().any(|(_, text)| text == "JEV_E2E_REPLY_00_END"));
    assert_eq!(off_shapes.last().map(|(role, _)| role.as_str()), Some("assistant"));

    // ------------------------------------------------------------------
    // Phase 2: Compare with the deterministic echo mock: full parity.
    // ------------------------------------------------------------------
    write_settings(
        &agent_dir,
        json!({ "global_default": "compare", "transport": "mock" }),
    );
    let compare = build_session("jev-compare-provider", CreationOverrides::default()).await;
    for event in ["turn_start", "tool_call", "agent_end", "model_select", "input"] {
        assert!(
            compare.session.has_extension_handlers(event),
            "Compare must register the observer handler for {event}"
        );
    }
    // Mutating surfaces must stay untouched even in Compare. The provider-request
    // surface is checked here too: Active-handler presence follows the setting, so
    // a Compare-only process installs no `before_provider_request` handler at all
    // and the retry/semantic-edge behavior of a user who never enabled Active is
    // exactly the default one. Active is the only mode that adds that handler
    // (Phase 3b asserts the positive case).
    for event in [
        "context",
        "before_provider_request",
        "before_agent_start",
        "session_before_compact",
        "session_before_refine",
        "session_before_switch",
        "tool_result",
        "user_bash",
        "message_update",
    ] {
        assert!(
            !compare.session.has_extension_handlers(event),
            "Compare must NOT register a handler on mutating surface {event}"
        );
    }
    let compare_model_before = compare.session.model().expect("compare model").id;
    compare.provider.set_responses(vec![
        FauxResponseStep::Message(reply(0)),
        FauxResponseStep::Message(reply(1)),
    ]);
    turn(&compare.session, "e2e prompt one").await;
    turn(&compare.session, "e2e prompt two").await;
    assert_eq!(
        compare.provider.call_count(),
        off.provider.call_count(),
        "Compare must not add provider calls"
    );
    let compare_shapes = shapes(&compare.session);
    assert_eq!(
        off_shapes.len(),
        compare_shapes.len(),
        "message counts must match Off vs Compare: off={off_shapes:?} compare={compare_shapes:?}"
    );
    assert_eq!(
        off_shapes, compare_shapes,
        "message role/text sequence must be identical Off vs Compare"
    );
    assert_eq!(
        compare_model_before,
        compare.session.model().expect("compare model after").id,
        "Jev must not change the model"
    );
    assert_eq!(
        compare.session.thinking_level(),
        off_thinking,
        "Jev must not change effort/thinking level"
    );
    assert_eq!(off_model, compare_model_before, "both sessions run the same model id");
    // Records exist for the compare session with applied=false everywhere.
    let record_count = wait_records_settled(&agent_dir, 1).await;
    assert!(record_count >= 1, "Compare must write correlation records");
    let records_raw = std::fs::read_to_string(records_path(&agent_dir)).unwrap();
    let mut compare_records = 0usize;
    for line in records_raw.lines() {
        let record: Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["applied"], json!(false), "applied=false ALWAYS: {line}");
        assert_eq!(record["mode"], json!("compare"));
        assert_eq!(
            record["session_id"],
            json!(compare.session.session_id()),
            "records belong to the compare session"
        );
        compare_records += 1;
    }
    assert!(compare_records >= 1);

    // First-use activation must work on the EXISTING default-Off session,
    // without rebuilding its runner or restarting the process.
    off.provider.set_responses(vec![FauxResponseStep::Message(reply(7))]);
    turn(&off.session, "first use compare in the same chat").await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if pi_coding_agent::core::jev_bridge::session_status_snapshot(&off.session.session_id())
                .is_some_and(|status| status["success_count"].as_u64().unwrap_or(0) > 0) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.expect("existing Off session must begin actual mock comparisons");
    write_settings(&agent_dir, json!({ "global_default": "off", "transport": "mock" }));
    let count_before_off = read_record_count(&agent_dir);
    off.provider.set_responses(vec![FauxResponseStep::Message(reply(8))]);
    turn(&off.session, "no comparison while off").await;
    // Cancellation may record a terminal skip, but Off never starts a new call.
    let successes = pi_coding_agent::core::jev_bridge::session_status_snapshot(&off.session.session_id())
        .unwrap()["success_count"].as_u64().unwrap();
    write_settings(&agent_dir, json!({ "global_default": "compare", "transport": "mock" }));
    off.provider.set_responses(vec![FauxResponseStep::Message(reply(9))]);
    turn(&off.session, "compare again without restart").await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let status = pi_coding_agent::core::jev_bridge::session_status_snapshot(&off.session.session_id());
            if status.is_some_and(|status| status["success_count"].as_u64().unwrap_or(0) > successes) { break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.expect("Compare must rearm after Off");
    assert!(read_record_count(&agent_dir) > count_before_off);

    // A rewritten credential envelope may still resolve to the same key.
    // Mock transport resolves a fixed synthetic key and never decrypts these
    // inert fixture bytes. Changed generation must not reuse a stale gate.
    for (generation, bytes) in ["fixture-envelope-one", "fixture-envelope-generation-two"].into_iter().enumerate() {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = pi_coding_agent::core::jev_bridge::session_status_snapshot(&off.session.session_id());
                if status.is_some_and(|status| status["pending"].as_u64() == Some(0)) { break; }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }).await.expect("previous mock generation settles before rewrite");
        let accepted = || std::fs::read_to_string(records_path(&agent_dir)).unwrap_or_default()
            .lines().filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|record| record["session_id"] == json!(off.session.session_id())
                && record["selected_value"].is_string()).count();
        let before_rewrite = accepted();
        let envelope = agent_dir.join("jev").join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME));
        std::fs::write(envelope, bytes).unwrap();
        off.provider.set_responses(vec![FauxResponseStep::Message(reply(10 + generation))]);
        turn(&off.session, "comparison after same-key credential generation change").await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while accepted() <= before_rewrite {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }).await.expect("rewritten same key must allow new mock observations");
    }
    wait_jev_session_settled(&off.session).await;
    off.session.dispose_async(Some(false)).await;
    std::fs::remove_file(agent_dir.join("jev").join(format!("{}.{}",
        pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME))).unwrap();

    // ------------------------------------------------------------------
    // Phase 3: Hostile maximum-confidence answers cannot act.
    // ------------------------------------------------------------------
    write_settings(
        &agent_dir,
        json!({ "global_default": "compare", "transport": "mock-hostile" }),
    );
    let hostile = build_session("jev-hostile-provider", CreationOverrides::default()).await;
    hostile.provider.set_responses(vec![FauxResponseStep::Message(reply(0))]);
    turn(&hostile.session, "hostile prompt").await;
    let hostile_shapes = shapes(&hostile.session);
    assert!(hostile_shapes.iter().any(|(_, text)| text == "hostile prompt"));
    assert!(hostile_shapes.iter().any(|(_, text)| text == "JEV_E2E_REPLY_00_END"));
    assert_eq!(hostile.provider.call_count(), 1, "exactly one provider call");
    assert_eq!(
        hostile.session.model().expect("hostile model").id,
        off_model,
        "model unchanged under maximum-confidence adversarial answers"
    );
    assert_eq!(
        hostile.session.thinking_level(),
        off_thinking,
        "high-confidence answers must not change effort/thinking level"
    );
    assert_eq!(hostile_shapes.len(), 3, "no injected prompts or extra turns");
    wait_jev_session_settled(&hostile.session).await;
    hostile.session.dispose_async(Some(false)).await;

    // ------------------------------------------------------------------
    // Phase 3b: Active is a REAL mode with a bounded effect. It registers the
    // observe-only handlers PLUS the one provider-boundary handler, and the only
    // things that handler may change are the tool catalog (withdrawn for one
    // request whose task needs no tools) and an already-set reasoning effort (moved
    // one ladder step). Everything else — model, provider, messages, delegation —
    // is unchanged, and the session's own thinking level is never touched.
    // ------------------------------------------------------------------
    write_settings(
        &agent_dir,
        json!({ "global_default": "active", "transport": "mock" }),
    );
    let active = build_session("jev-active-provider", CreationOverrides::default()).await;
    for event in JEV_EVENT_NAMES {
        assert!(
            active.session.has_extension_handlers(event),
            "Active keeps the observer handler for {event}"
        );
    }
    assert!(
        active.session.has_extension_handlers("before_provider_request"),
        "Active installs the one provider-boundary handler"
    );
    // Every other mutating surface stays unregistered in Active as well.
    for event in [
        "context",
        "before_agent_start",
        "session_before_compact",
        "session_before_refine",
        "session_before_switch",
        "tool_result",
        "user_bash",
        "message_update",
    ] {
        assert!(
            !active.session.has_extension_handlers(event),
            "Active must NOT register a handler on {event}"
        );
    }
    active.provider.set_responses(vec![FauxResponseStep::Message(reply(4))]);
    turn(&active.session, "active-mode prompt").await;
    assert_eq!(active.provider.call_count(), 1, "active-mode turn ran normally");
    assert_eq!(
        active.session.thinking_level(),
        off_thinking,
        "Active never changes the SESSION thinking level; only a request-body hint may move"
    );
    assert_eq!(
        active.session.model().expect("active model").id,
        off_model,
        "Active must not change the model"
    );
    assert_eq!(
        shapes(&active.session).len(),
        3,
        "no injected messages under Active"
    );
    // Whatever Active recorded is bounded to the two appliable categories and to
    // effects on the three provider-body keys it owns. A record can therefore never
    // carry a model, provider, permission, budget, depth or concurrency change.
    let mut active_records = 0usize;
    for line in std::fs::read_to_string(records_path(&agent_dir)).unwrap_or_default().lines() {
        let record: Value = serde_json::from_str(line).unwrap();
        if record["mode"] != json!("active") {
            continue;
        }
        active_records += 1;
        let category = record["category"].as_str().unwrap_or("");
        if record["applied"] == json!(true) {
            assert!(["tool_requirement", "complexity"].contains(&category),
                "default Active may only apply legacy categories: {line}");
        } else {
            assert!(record["acceptance"].is_string() || record["skipped_reason"].is_string(),
                "record-only recommendations must state their outcome: {line}");
        }
        assert_eq!(
            record["session_id"],
            json!(active.session.session_id()),
            "records belong to the Active session"
        );
        for effect in record["applied_effects"].as_array().into_iter().flatten() {
            let field = effect["field"].as_str().unwrap_or("");
            assert!(
                ["tools", "tool_choice", "reasoning_effort", "reasoning.effort"].contains(&field),
                "Active may only change a provider-body tool/effort key: {line}"
            );
        }
    }
    // An accepted answer is allowed to exist here (that is what Active means); the
    // point of this phase is that its effect stays inside the request body.
    let _ = active_records;
    wait_jev_session_settled(&active.session).await;
    active.session.dispose_async(Some(false)).await;

    // ------------------------------------------------------------------
    // Phase 3c: Delayed/stale answers. The turn completes BEFORE Jev's
    // answers arrive; the answers can only become applied=false records and
    // must not re-drive, extend or alter the agent loop or delegation.
    // ------------------------------------------------------------------
    write_settings(
        &agent_dir,
        json!({ "global_default": "compare", "transport": "mock-delayed" }),
    );
    let stale = build_session("jev-stale-provider", CreationOverrides::default()).await;
    stale.provider.set_responses(vec![FauxResponseStep::Message(reply(5))]);
    turn(&stale.session, "stale-result prompt").await;
    let stale_shapes_after_turn = shapes(&stale.session);
    assert_eq!(stale_shapes_after_turn.len(), 3, "no injected messages while Jev is pending");
    assert_eq!(stale.provider.call_count(), 1, "exactly one provider call before answers arrive");
    assert_eq!(
        stale.session.thinking_level(),
        off_thinking,
        "delayed answers must not change effort"
    );
    assert_eq!(
        stale.session.model().expect("stale model").id,
        off_model,
        "delayed answers must not change the model"
    );
    // Answers land asynchronously, as records only.
    let stale_records_before = read_record_count(&agent_dir);
    let settled_count = wait_records_settled(&agent_dir, stale_records_before + 1).await;
    assert!(
        settled_count > stale_records_before,
        "delayed answers must eventually become records"
    );
    let stale_records = std::fs::read_to_string(records_path(&agent_dir)).unwrap();
    for line in stale_records.lines().skip(stale_records_before) {
        let record: Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["applied"], json!(false), "stale results are records only: {line}");
        assert_eq!(record["mode"], json!("compare"));
    }
    // The stale answers must not have re-driven or altered the agent loop.
    assert_eq!(shapes(&stale.session), stale_shapes_after_turn, "stale answers injected nothing");
    assert_eq!(
        stale.provider.call_count(),
        1,
        "stale answers must not trigger extra provider calls"
    );
    stale.session.wait_for_idle().await.expect("stale session idle");
    wait_jev_session_settled(&stale.session).await;
    stale.session.dispose_async(Some(false)).await;

    // ------------------------------------------------------------------
    // Phase 3d: Malformed (protocol-violating) answers. Validation rejects
    // them; they land as logged skips and cannot influence the agent loop.
    // ------------------------------------------------------------------
    write_settings(
        &agent_dir,
        json!({ "global_default": "compare", "transport": "mock-malformed" }),
    );
    let malformed = build_session("jev-malformed-provider", CreationOverrides::default()).await;
    malformed.provider.set_responses(vec![FauxResponseStep::Message(reply(6))]);
    turn(&malformed.session, "malformed-answer prompt").await;
    let malformed_shapes = shapes(&malformed.session);
    assert_eq!(malformed_shapes.len(), 3, "malformed answers injected nothing");
    assert_eq!(malformed.provider.call_count(), 1, "exactly one provider call");
    assert_eq!(
        malformed.session.thinking_level(),
        off_thinking,
        "malformed answers must not change effort"
    );
    assert_eq!(
        malformed.session.model().expect("malformed model").id,
        off_model,
        "malformed answers must not change the model"
    );
    let malformed_records_before = read_record_count(&agent_dir);
    let malformed_settled = wait_records_settled(&agent_dir, malformed_records_before + 1).await;
    assert!(
        malformed_settled > malformed_records_before,
        "malformed answers must produce logged skips"
    );
    let malformed_records = std::fs::read_to_string(records_path(&agent_dir)).unwrap();
    let saw_failed_skip = malformed_records
        .lines()
        .skip(malformed_records_before)
        .any(|line| {
            let record: Value = serde_json::from_str(line).unwrap();
            record["applied"] == json!(false)
                && record["skipped_reason"]
                    .as_str()
                    .map(|reason| reason.starts_with("request_failed"))
                    .unwrap_or(false)
        });
    assert!(saw_failed_skip, "malformed answers land as request_failed skips");
    assert_eq!(shapes(&malformed.session), malformed_shapes, "failures injected nothing");
    assert_eq!(malformed.provider.call_count(), 1, "failures must not trigger provider calls");
    wait_jev_session_settled(&malformed.session).await;
    malformed.session.dispose_async(Some(false)).await;

    // ------------------------------------------------------------------
    // Phase 4: Tool-call parity with a custom probe tool.
    // ------------------------------------------------------------------
    let tool_overrides = || CreationOverrides {
        tools: Some(vec!["jev_e2e_probe".to_string()]),
        custom_tools: Some(vec![probe_tool()]),
        ..Default::default()
    };
    let tool_steps = || {
        vec![
            FauxResponseStep::Message(tool_call_reply()),
            FauxResponseStep::Message(reply(2)),
        ]
    };
    write_settings(&agent_dir, json!({ "global_default": "off" }));
    let off_tool = build_session("jev-tool-off-provider", tool_overrides()).await;
    off_tool.provider.set_responses(tool_steps());
    turn(&off_tool.session, "run the probe tool").await;
    let off_tool_shapes = shapes(&off_tool.session);
    assert_eq!(off_tool.provider.call_count(), 2, "tool turn: two provider calls");
    assert!(
        off_tool_shapes.iter().any(|(role, text)| role == "toolResult" && text.contains("PROBE_OK")),
        "the probe tool must run in Off: {off_tool_shapes:?}"
    );

    write_settings(
        &agent_dir,
        json!({ "global_default": "compare", "transport": "mock" }),
    );
    let compare_tool = build_session("jev-tool-compare-provider", tool_overrides()).await;
    compare_tool.provider.set_responses(tool_steps());
    turn(&compare_tool.session, "run the probe tool").await;
    let compare_tool_shapes = shapes(&compare_tool.session);
    assert_eq!(
        compare_tool.provider.call_count(),
        off_tool.provider.call_count(),
        "Compare must not change provider-call structure"
    );
    assert_eq!(
        off_tool_shapes, compare_tool_shapes,
        "tool-call parity: Compare must not block, filter or alter the tool call"
    );
    assert!(
        compare_tool_shapes
            .iter()
            .any(|(role, text)| role == "toolResult" && text.contains("PROBE_OK")),
        "the probe tool must run in Compare"
    );
    wait_jev_session_settled(&compare_tool.session).await;
    compare_tool.session.dispose_async(Some(false)).await;

    // ------------------------------------------------------------------
    // Phase 5: Immediate Off on a Compare-registered session.
    // ------------------------------------------------------------------
    // Let phase 4's Compare work settle before measuring.
    let settled_phase4 = wait_records_settled(&agent_dir, 1).await;
    let _ = settled_phase4;
    write_settings(&agent_dir, json!({ "global_default": "off" }));
    let before_count = read_record_count(&agent_dir);
    compare.provider.set_responses(vec![FauxResponseStep::Message(reply(3))]);
    turn(&compare.session, "turn while off").await;
    let after_count = read_record_count(&agent_dir);
    assert_eq!(
        before_count, after_count,
        "Off must stop recording immediately (no new records)"
    );
    assert_eq!(compare.provider.call_count(), 3, "the turn still ran normally");
    wait_jev_session_settled(&compare.session).await;
    compare.session.dispose_async(Some(false)).await;

    // Independent execution/compaction axes through real session hooks and native projections.
    for mode in ["off", "compare", "active", "compare-active"] {
        for compaction in [false, true] {
            write_settings(&agent_dir, json!({"global_default":mode, "transport":"mock",
                "compaction_enabled":compaction, "features":{"context_relevance":true}}));
            let axis = build_session(&format!("jev-axis-{mode}-{compaction}"), CreationOverrides { reasoning: true, ..Default::default() }).await;
            axis.provider.set_responses(vec![FauxResponseStep::Message(reply(20))]);
            turn(&axis.session, "synthetic independent-axis task").await;
            wait_jev_session_settled(&axis.session).await;
            wait_records_settled(&agent_dir, 1).await;
            let runner = axis.session.extension_runner().expect("native runner");
            let ctx = runner.create_context();
            let before_request = read_record_count(&agent_dir);
            let body = json!({"model":"faux", "tools":[{"type":"function","function":{"name":"search"}}],
                "tool_choice":"auto", "reasoning":{"effort":"low", "summary":"auto"}, "messages":[{"role":"user","content":"unchanged"}]});
            let outgoing = runner.emit_before_provider_request(body.clone()).await;
            if ["active", "compare-active"].contains(&mode) {
                assert_eq!(outgoing["reasoning"]["effort"],json!("medium"), "supported Responses effort at actual provider hook");
                assert_eq!(outgoing["reasoning"]["summary"],json!("auto"));
                assert_eq!(outgoing["tools"],body["tools"], "mock delegate recommendation keeps tools");
                assert_eq!(outgoing["messages"],body["messages"]);
                assert_eq!(outgoing["model"],body["model"]);
            } else { assert_eq!(outgoing,body); }
            let new_rows: Vec<Value> = std::fs::read_to_string(records_path(&agent_dir)).unwrap_or_default()
                .lines().skip(before_request).filter_map(|line|serde_json::from_str(line).ok()).collect();
            if mode == "compare-active" {
                let answered: Vec<_> = new_rows.iter().filter(|row|row["selected_value"].is_string()).collect();
                let ids: std::collections::BTreeSet<_> = answered.iter().filter_map(|row|row["request_id"].as_str()).collect();
                assert_eq!(ids.len(),1,"one shared logical request at provider boundary");
                let shadow:Vec<_>=answered.iter().filter(|row|row["schema_version"]=="jev.compare/1").collect();
                let active:Vec<_>=answered.iter().filter(|row|row["schema_version"]=="jev.active/1").collect();
                assert!(!shadow.is_empty()); assert_eq!(shadow.len(),active.len());
                assert!(shadow.iter().all(|row|row["applied"]==false));
                assert!(active.iter().all(|row|row["baseline_action"]["tools"]=="count:1"));
                assert!(active.iter().all(|row|row["baseline_action"]["reasoning.effort"]=="low"));
                assert!(active.iter().all(|row|row["compaction_enabled"]==compaction));
            }
            let mut old_assistant = serde_json::to_value(faux_assistant_message(
                FauxAssistantContent::Blocks(vec![ContentBlock::ToolCall(faux_tool_call("search",Map::new(),None))]),None)).unwrap();
            old_assistant["content"][0]["id"]=json!("old-call");
            let mut projection=vec![json!({"role":"user","content":"old task","timestamp":0}),old_assistant,
                json!({"role":"toolResult","toolCallId":"old-call","toolName":"search","isError":false,
                    "content":[{"type":"text","text":"bounded old result ".repeat(1000)}],"timestamp":0})];
            for index in 0..8 { projection.push(json!({"role":"user","content":format!("recent pinned {index}"),"timestamp":0})); }
            let stored_before = shapes(&axis.session);
            let filtered=pi_coding_agent::core::jev_bridge::filter_context_candidates(ctx.clone(),projection.clone()).await;
            assert_eq!(filtered,projection,"subthreshold optional recommendations preserve context");
            let before_compaction=read_record_count(&agent_dir);
            let compacted=pi_coding_agent::core::jev_compaction::compact_context(ctx,filtered,None).await;
            assert_eq!(compacted,projection,"mock keep recommendation leaves projection intact");
            let compaction_rows:Vec<Value>=std::fs::read_to_string(records_path(&agent_dir)).unwrap_or_default().lines()
                .skip(before_compaction).filter_map(|line|serde_json::from_str::<Value>(line).ok())
                .filter(|row|row["schema_version"]=="jev.compaction/1").collect();
            assert_eq!(!compaction_rows.is_empty(),compaction,"independent compaction gate in {mode}");
            assert_eq!(shapes(&axis.session),stored_before,"projections never change history");
            assert_eq!(axis.provider.call_count(),1);
            wait_jev_session_settled(&axis.session).await;
            axis.session.dispose_async(Some(false)).await;
        }
    }

    // ------------------------------------------------------------------
    // Phase 6: Jev-owned files only under the jev dir.
    // ------------------------------------------------------------------
    let jev_dir = agent_dir.join("jev");
    let entries: Vec<String> = std::fs::read_dir(&jev_dir)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.file_name().to_string_lossy().to_string()))
        .collect();
    assert!(
        entries
            .iter()
            .all(|name| name == "jev-settings.json" || name.starts_with("records.jsonl")),
        "only jev-owned files may exist under the jev dir: {entries:?}"
    );

    // Drop ends the guard and restores the env.
    drop(guard);
}
