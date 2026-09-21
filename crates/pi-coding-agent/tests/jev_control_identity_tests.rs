//! NATIVE CONTROL held-identity regressions (root contract v10 CONTROL item).
//!
//! Proves through the REAL session, the REAL authoritative settings store
//! (independent instances, same agent dir — no shared in-memory cache tricks,
//! no sleep-based freshness), the REAL native dispatch (bridge decide_control /
//! decide_independent through the real observer and the v9 single-load
//! authoritative gate) and the REAL effect gates (durable ControlBook ledger,
//! epoch correlation, budget consume; compaction apply_context):
//!
//! - POSITIVE UNCHANGED control: when nothing changes during the decide await,
//!   an agent-end decision APPLIES: corrective feedback continuation queued
//!   through the existing followUp surface and exactly ONE feedback budget
//!   unit spent DURABLY (fresh book instance recovers feedback==1 via the
//!   production path).
//! - Authoritative save mid-await (values byte-identical, write_revision
//!   advances): the in-flight decision is REFUSED at the fresh apply gate —
//!   no continuation, no spend.
//! - Full-jev Off->On cycle mid-await (overlay values return to identical
//!   bytes, activation revision moves): refused.
//! - Requested-model A->B->A mid-await (v9 setter path, serialized model
//!   bytes return to identical, write_revision moves): refused. The wire
//!   carried the captured model identity, proving the refusal is the durable
//!   ABA identity, not the model-equality check.
//! - No full profile: no CONTROL dispatch at all (no control question ids
//!   reach the transport) and NO ledger record for the session (H-CONTROL-1).
//! - Independent compaction: with decisions Off (mode Off) and compaction
//!   enabled, the independent compaction decide STILL dispatches through the
//!   real independent lane and APPLIES (a large old tool result is truncated);
//!   no control dispatch, no ledger record.
//!
//! UNEXECUTED DRAFT: jev-glm-native-audit; Sol integrates and runs.

#![allow(clippy::all)]

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};
use std::time::Duration;

/// These tests mutate the process-global PRIME_AGENT_CODING_AGENT_DIR env and
/// share the process-global ControlBook::global() ledger directory, so the
/// whole binary serializes with one test-lifetime mutex (the landed hint-wiring
/// discipline): the lock is acquired BEFORE every process-global mutation.
static CONTROL_IDENTITY_ENV_LOCK: Mutex<()> = Mutex::new(());

use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{AssistantMessage, ToolResultMessage, UserContent, UserMessage};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::jev_control::ControlBook;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_jev::compaction::TRUNCATION_MARKER;
use pi_jev::config::{JevMode, JevSettings, JevSettingsStore};
use pi_jev::control::ControlBudgetSnapshot;
use pi_jev::types::{
    Answer, QuestionSpec, SystemOneRequest, SystemOneResponse, Usage,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// Environment isolation + ONE process-lifetime agent dir
// ---------------------------------------------------------------------------

const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";

/// ControlBook::global() binds its ledger path from get_agent_dir() at FIRST
/// use in the process, so every test in this binary MUST share one agent dir.
/// Sessions are keyed by unique session ids, so per-session budget/epoch state
/// stays isolated; ledger assertions filter by session id.
static SHARED_AGENT_DIR: OnceLock<(tempfile::TempDir, std::path::PathBuf)> = OnceLock::new();

fn shared_agent_dir() -> &'static std::path::Path {
    let entry = SHARED_AGENT_DIR.get_or_init(|| {
        let dir = tempfile::Builder::new()
            .prefix("jev-control-identity-")
            .tempdir_in(std::env::temp_dir())
            .expect("shared agent tempdir");
        let path = std::fs::canonicalize(dir.path())
            .unwrap_or_else(|_| dir.path().to_path_buf());
        (dir, path)
    });
    entry.1.as_path()
}

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

/// Acquired before every process-global environment/fixture mutation and held
/// until the session, workspace, and environment have all been torn down.
struct TestScope {
    _guard: EnvGuard,
    _env_lock: std::sync::MutexGuard<'static, ()>,
}

fn begin_test_scope() -> TestScope {
    let env_lock = CONTROL_IDENTITY_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let guard = EnvGuard::new(shared_agent_dir());
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    pi_coding_agent::core::jev_bridge::debug_control_fixture_reset();
    TestScope {
        _guard: guard,
        _env_lock: env_lock,
    }
}

// ---------------------------------------------------------------------------
// Provider request capture (the REAL outgoing provider context)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CapturedRequest {
    /// Serialized provider-context messages: proves what the provider would
    /// actually receive (continuations, control feedback, annotations).
    messages: Vec<serde_json::Value>,
}

fn capture_factory(
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    steps: Arc<Mutex<Vec<AssistantMessage>>>,
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
                .map(|message| serde_json::to_value(message).unwrap_or(serde_json::Value::Null))
                .collect();
            captured.lock().unwrap().push(CapturedRequest { messages });
            let message = {
                let mut steps = steps.lock().unwrap();
                if steps.is_empty() {
                    faux_assistant_message(
                        pi_ai::providers::faux::FauxAssistantContent::Text(
                            "CONTROL_IDENTITY_TERMINAL_END".to_string(),
                        ),
                        None,
                    )
                } else {
                    steps.remove(0)
                }
            };
            Box::pin(async move { message })
        },
    )
}

// ---------------------------------------------------------------------------
// Settings through the AUTHORITATIVE save path (durable write revision moves)
// ---------------------------------------------------------------------------

fn save_control_settings(
    agent_dir: &std::path::Path,
    mode: JevMode,
    install_full: bool,
    compaction_enabled: bool,
    transport: &str,
    requested_model: Option<&str>,
) {
    let mut settings = JevSettings::default();
    settings.global_default = Some(mode);
    settings.compaction_enabled = compaction_enabled;
    settings.transport = Some(transport.to_string());
    if let Some(model) = requested_model {
        settings
            .set_requested_model(model)
            .expect("valid requested model id");
    }
    if install_full {
        settings.full_jev_install();
    }
    JevSettingsStore::new(agent_dir)
        .save(&settings)
        .expect("authoritative settings save");
    pi_coding_agent::core::jev_bridge::invalidate_settings_cache();
}

/// One independent store instance targeting the same agent dir (never the
/// process cache, never a sleep): the mid-await writes go through the REAL
/// authoritative save path, exactly like a second process would.
fn fresh_store() -> JevSettingsStore {
    JevSettingsStore::new(shared_agent_dir())
}

// ---------------------------------------------------------------------------
// Fixture (modeled on the landed hint-wiring fixture; ONE shared agent dir)
// ---------------------------------------------------------------------------

struct Fixture {
    session: Arc<AgentSession>,
    session_id: String,
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    _workspace: tempfile::TempDir,
    _scope: TestScope,
}

async fn build_fixture(steps: Vec<AssistantMessage>, scope: TestScope) -> Fixture {
    let agent_dir = shared_agent_dir().to_path_buf();
    let workspace = tempfile::Builder::new()
        .prefix("jev-control-identity-ws-")
        .tempdir_in(std::env::temp_dir())
        .expect("workspace tempdir");
    let cwd = workspace.path().to_path_buf();
    std::fs::create_dir_all(&cwd).expect("workspace dir");

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some("control-identity-faux".to_string()),
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
        .set_runtime_api_key(&model.provider, "synthetic-control-identity-key");
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
    model_registry
        .lock()
        .expect("model registry poisoned")
        .set_runtime_api_key(&model.provider, "synthetic-control-identity-key");

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(Some(&cwd.to_string_lossy()), Some(&agent_dir.to_string_lossy()))
            .expect("in-memory session manager"),
    ));

    let loader_options = DefaultResourceLoaderOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: agent_dir.to_string_lossy().to_string(),
        no_extensions: true,
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
    let steps: Arc<Mutex<Vec<AssistantMessage>>> = Arc::new(Mutex::new(steps));
    let factory = capture_factory(Arc::clone(&captured), Arc::clone(&steps));
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
    let session_id = created.session.session_id();

    Fixture {
        session: created.session,
        session_id,
        captured,
        _workspace: workspace,
        _scope: scope,
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

fn terminal_step(text: &str) -> AssistantMessage {
    faux_assistant_message(
        pi_ai::providers::faux::FauxAssistantContent::Text(text.to_string()),
        None,
    )
}

// ---------------------------------------------------------------------------
// Control answers + mid-await write hooks (REAL authoritative saves)
// ---------------------------------------------------------------------------

const CONTROL_QUESTION_PREFIXES: [&str; 4] = [
    "result_sufficiency.",
    "first_pass_verification.",
    "continue_stop_escalate.",
    "retry_classification.",
];

fn is_control_request(request: &SystemOneRequest) -> bool {
    request.state.get("_jev_fixture_lane").and_then(serde_json::Value::as_str)
        == Some("explicit_control")
        && !request.questions.is_empty()
        && request.questions.keys().all(|id| {
            CONTROL_QUESTION_PREFIXES
                .iter()
                .any(|prefix| id.starts_with(prefix))
        })
}

/// Which agent-end control round this is for the session (0-based), counted
/// from the fixture's own capture — no ordering assumptions beyond the calls
/// the transport actually saw.
fn control_round_ordinal() -> usize {
    pi_coding_agent::core::jev_bridge::debug_control_fixture_calls()
        .iter()
        .filter(|call| {
            call["fixture_lane"] == json!("explicit_control")
                && call["question_ids"].as_array().is_some_and(|ids| {
                    !ids.is_empty()
                        && ids.iter().all(|id| {
                            id.as_str().is_some_and(|id| {
                                CONTROL_QUESTION_PREFIXES
                                    .iter()
                                    .any(|prefix| id.starts_with(prefix))
                            })
                        })
                })
        })
        .count()
        .saturating_sub(1)
}

/// A Choice answer concentrating mass on `chosen` with high confidence; the
/// distribution covers every criteria key (the documented argmax rule).
fn choice_answer(spec: &QuestionSpec, chosen: &str, confidence: f64) -> Option<Answer> {
    let criteria = match spec {
        QuestionSpec::Choice { criteria, .. } => criteria,
        _ => return None,
    };
    let mut probabilities = std::collections::BTreeMap::new();
    for key in criteria.keys() {
        probabilities.insert(key.clone(), 0.0_f64);
    }
    let remainder = if criteria.len() > 1 {
        (1.0 - confidence) / (criteria.len() as f64 - 1.0)
    } else {
        0.0
    };
    for (key, value) in probabilities.iter_mut() {
        *value = if key == chosen { confidence } else { remainder };
    }
    if !probabilities.contains_key(chosen) {
        return None;
    }
    Some(Answer::Choice {
        choice: chosen.to_string(),
        probabilities,
        confidence,
    })
}

/// Answer one agent-end control round: round 0 asks for the corrective
/// feedback outcome (insufficient), later rounds clear (sufficient).
fn control_round_response(request: &SystemOneRequest, round: usize) -> SystemOneResponse {
    let mut answers = std::collections::BTreeMap::new();
    let (sufficiency, assessment) = if round == 0 {
        ("insufficient", "partial")
    } else {
        ("sufficient", "complete")
    };
    for (id, spec) in &request.questions {
        let answer = if id.starts_with("result_sufficiency.0") {
            choice_answer(spec, sufficiency, 0.95)
        } else if id.starts_with("result_sufficiency.") {
            choice_answer(spec, assessment, 0.95)
        } else if id.starts_with("first_pass_verification.") {
            choice_answer(spec, "none", 0.95)
        } else if id.starts_with("continue_stop_escalate.") {
            choice_answer(spec, "continue", 0.95)
        } else if id.starts_with("retry_classification.") {
            choice_answer(spec, "unknown", 0.95)
        } else {
            None
        };
        if let Some(answer) = answer {
            answers.insert(id.clone(), answer);
        }
    }
    SystemOneResponse {
        model: "jev-mock-control/1".to_string(),
        answers,
        usage: Usage {
            input_tokens: Some(5),
            output_tokens: Some(5),
        },
        ..SystemOneResponse::default()
    }
}

// ---------------------------------------------------------------------------
// Ledger + capture assertions
// ---------------------------------------------------------------------------

fn ledger_path() -> std::path::PathBuf {
    shared_agent_dir()
        .join("jev")
        .join("control-budgets.jsonl")
}

fn ledger_records_for(session_id: &str) -> Vec<serde_json::Value> {
    let Ok(text) = std::fs::read_to_string(ledger_path()) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|record| record["session"] == json!(session_id))
        .collect()
}

fn control_calls() -> Vec<serde_json::Value> {
    pi_coding_agent::core::jev_bridge::debug_control_fixture_calls()
        .into_iter()
        .filter(|call| {
            call["fixture_lane"] == json!("explicit_control")
                && call["question_ids"].as_array().is_some_and(|ids| {
                    !ids.is_empty()
                        && ids.iter().all(|id| {
                            id.as_str().is_some_and(|id| {
                                CONTROL_QUESTION_PREFIXES
                                    .iter()
                                    .any(|prefix| id.starts_with(prefix))
                            })
                        })
                })
        })
        .collect()
}

fn compaction_calls() -> Vec<serde_json::Value> {
    pi_coding_agent::core::jev_bridge::debug_control_fixture_calls()
        .into_iter()
        .filter(|call| {
            call["question_ids"].as_array().is_some_and(|ids| {
                ids.iter()
                    .any(|id| id.as_str().is_some_and(|id| id.starts_with("compaction.")))
            })
        })
        .collect()
}


fn compaction_response(request: &SystemOneRequest) -> Option<SystemOneResponse> {
    let mut answers = std::collections::BTreeMap::new();
    for (id, spec) in &request.questions {
        if !id.starts_with("compaction.") {
            return None;
        }
        if matches!(spec, QuestionSpec::Noul { .. }) {
            answers.insert(id.clone(), Answer::Noul { noul: 0.1 });
        }
    }
    Some(SystemOneResponse {
        model: "jev-mock-control/1".to_string(),
        answers,
        usage: Usage {
            input_tokens: Some(5),
            output_tokens: Some(5),
        },
        ..SystemOneResponse::default()
    })
}

/// Compaction-eligible native transcript: one large, old source-read result
/// followed by enough recent assistant prose to preserve a tail.
fn eligible_compaction_values() -> Vec<serde_json::Value> {
    let big = "old source output\n".repeat(1200);
    let mut messages: Vec<pi_agent_core::types::AgentMessage> = vec![
        UserMessage::new(
            UserContent::Text("Keep exact constraints.".to_string()),
            1,
        )
        .into(),
        AssistantMessage {
            content: vec![pi_ai::types::ContentBlock::ToolCall(pi_ai::types::ToolCall::new(
                "call-1",
                "ipython",
                json!({"code": "from pathlib import Path\nprint(Path(\"src/parser.rs\").read_text())"})
                    .as_object()
                    .cloned()
                    .unwrap(),
            ))],
            stop_reason: "toolUse".to_string().into(),
            ..Default::default()
        }
        .into(),
        ToolResultMessage::new(
            "call-1",
            "ipython",
            vec![pi_ai::types::ImageOrTextContent::Text(
                pi_ai::types::TextContent::new(big),
            )],
            false,
            2,
        )
        .into(),
    ];
    for index in 0..7 {
        messages.push(
            AssistantMessage {
                content: vec![pi_ai::types::ContentBlock::Text(
                    pi_ai::types::TextContent::new(format!("Recent reasoning {index}")),
                )],
                ..Default::default()
            }
            .into(),
        );
    }
    messages
        .iter()
        .map(|message| serde_json::to_value(message).expect("serializable message"))
        .collect()
}

fn feedback_remaining_via_fresh_book(session_id: &str) -> (u8, bool) {
    // PRODUCTION recovery path: a fresh ControlBook instance on the same
    // durable dir; no reload helper, no in-memory state.
    let snapshot: ControlBudgetSnapshot = ControlBook::new(shared_agent_dir()).snapshot(session_id);
    (snapshot.feedback_remaining, snapshot.available)
}

fn captured_texts_contain(fixture: &Fixture, index: usize, needle: &str) -> bool {
    fixture
        .captured
        .lock()
        .unwrap()
        .get(index)
        .map(|request| {
            request
                .messages
                .iter()
                .any(|message| message.to_string().contains(needle))
        })
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// POSITIVE UNCHANGED CONTROL: nothing changes during the decide await, so
/// the agent-end decision APPLIES: the corrective feedback continuation is
/// queued through the existing followUp surface (the next outgoing provider
/// request carries the fixed CONTROL_FEEDBACK_CUSTOM_TYPE message) and ONE
/// feedback budget unit is spent DURABLY (fresh book recovers feedback==1).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn control_decision_applies_when_nothing_changes() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::CompareAndActive,
        /* install_full */ true,
        /* compaction */ false,
        "mock-control",
        None,
    );
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            if !is_control_request(request) {
                return None; // shadow batteries: per-type valid fallback
            }
            Some(control_round_response(request, control_round_ordinal()))
        },
    )));
    let fixture = build_fixture(
        vec![
            terminal_step("partial answer for the task"),
            terminal_step("completed answer for the task"),
        ],
        scope,
    )
    .await;

    turn(&fixture.session, "make retries configurable").await;

    let calls = control_calls();
    assert!(
        calls.len() >= 2,
        "both agent-end rounds decided (got {})",
        calls.len()
    );

    let captured = fixture.captured.lock().unwrap().clone();
    assert!(
        captured.len() >= 2,
        "the corrective feedback continuation ran (captured {} provider requests)",
        captured.len()
    );
    assert!(
        captured_texts_contain(
            &fixture,
            1,
            "not yet complete (assessment: insufficient)"
        ),
        "the outgoing continuation carries the fixed result-gap template"
    );

    let (remaining, available) = feedback_remaining_via_fresh_book(&fixture.session_id);
    assert!(available, "durable accounting is trusted");
    assert_eq!(
        remaining, 1,
        "exactly ONE feedback unit spent durably (maxima 2)"
    );

    let records = ledger_records_for(&fixture.session_id);
    assert!(
        records
            .iter()
            .any(|record| record["feedback"] == json!(1)),
        "the durable ledger records the spend for this session"
    );
    let status = pi_coding_agent::core::jev_bridge::session_status_snapshot(&fixture.session_id)
        .expect("worker status includes the terminal outcome");
    assert_eq!(
        status["controlTerminal"]["verification_state"],
        json!("not_applicable")
    );
    assert_eq!(status["controlTerminal"]["attention_required"], json!(false));
    assert!(status["controlTerminal"]["terminal_annotation"].is_null());
    assert!(status["controlTerminal"]["pause"].is_null());
}

/// Authoritative save mid-await: the values return byte-identical (only the
/// durable write revision advances). The in-flight decision is REFUSED at
/// the fresh apply gate: no continuation, no spend.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authoritative_save_during_await_refuses_in_flight_control() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::CompareAndActive,
        true,
        false,
        "mock-control",
        None,
    );
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            if !is_control_request(request) {
                return None;
            }
            if control_round_ordinal() == 0 {
                // MID-AWAIT authoritative write through an INDEPENDENT store
                // instance: no cache invalidation, no sleep. The serialized
                // values are unchanged; only write_revision moves.
                let store = fresh_store();
                let settings = store.load();
                store
                    .save(&settings)
                    .expect("authoritative identical-values save");
            }
            Some(control_round_response(request, control_round_ordinal()))
        },
    )));
    let fixture = build_fixture(vec![terminal_step("partial answer for the task")], scope).await;

    turn(&fixture.session, "make retries configurable").await;

    let calls = control_calls();
    assert!(
        !calls.is_empty(),
        "the control decide DID dispatch; the refusal is at the apply gate"
    );

    let captured = fixture.captured.lock().unwrap().len();
    assert_eq!(
        captured, 1,
        "no continuation: the refused decision queued no feedback"
    );
    assert!(
        !captured_texts_contain(&fixture, 0, "not yet complete (assessment: insufficient)"),
        "no control feedback in the only request"
    );

    let (remaining, available) = feedback_remaining_via_fresh_book(&fixture.session_id);
    assert!(available, "the epoch accounting stays trusted");
    assert_eq!(remaining, 2, "no budget unit spent");
    let records = ledger_records_for(&fixture.session_id);
    assert!(
        records.iter().all(|record| record["feedback"] == json!(2)),
        "ledger shows the activation but no spend"
    );
}

/// Full-jev Off->On cycle mid-await: the overlay returns to identical
/// serialized values but the activation revision (and write_revision) moved.
/// Refused: no continuation, no spend; the overlay is active again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_jev_off_on_cycle_during_await_refuses_in_flight_control() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::CompareAndActive,
        true,
        false,
        "mock-control",
        None,
    );
    let stamp_before = fresh_store().load().full_jev_stamp();
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            if !is_control_request(request) {
                return None;
            }
            if control_round_ordinal() == 0 {
                let store = fresh_store();
                let mut off = store.load();
                assert!(off.full_jev_remove(), "overlay was active");
                store.save(&off).expect("authoritative off save");
                let mut on = store.load();
                assert!(on.full_jev_install(), "reactivation with a NEW revision");
                store.save(&on).expect("authoritative on save");
            }
            Some(control_round_response(request, control_round_ordinal()))
        },
    )));
    let fixture = build_fixture(vec![terminal_step("partial answer for the task")], scope).await;

    turn(&fixture.session, "make retries configurable").await;

    let stamp_after = fresh_store().load().full_jev_stamp();
    assert_ne!(stamp_before, stamp_after, "the ABA cycle moved the stamp");
    assert!(
        fresh_store().load().full_jev_active(),
        "the overlay is active again after the cycle"
    );

    let calls = control_calls();
    assert!(!calls.is_empty(), "the decide dispatched");
    assert_eq!(
        fixture.captured.lock().unwrap().len(),
        1,
        "no continuation for the refused decision"
    );
    let (remaining, available) = feedback_remaining_via_fresh_book(&fixture.session_id);
    assert!(available);
    assert_eq!(remaining, 2, "no spend across the Off window");
}

/// Requested-model A->B->A mid-await (v9 setter path): the serialized model
/// selection returns to identical bytes while the durable write revision
/// moves; the wire carried the CAPTURED model identity, so the refusal is the
/// durable ABA identity, not the model-equality check.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requested_model_aba_identical_bytes_refuses_in_flight_control() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::CompareAndActive,
        true,
        false,
        "mock-control",
        Some("jev-model-a"),
    );
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            if !is_control_request(request) {
                return None;
            }
            if control_round_ordinal() == 0 {
                let store = fresh_store();
                let mut to_b = store.load();
                assert!(to_b.set_requested_model("jev-model-b").expect("valid id"));
                store.save(&to_b).expect("authoritative model B save");
                let mut back_to_a = store.load();
                assert!(
                    back_to_a
                        .set_requested_model("jev-model-a")
                        .expect("valid id")
                );
                store.save(&back_to_a).expect("authoritative model A restore");
            }
            Some(control_round_response(request, control_round_ordinal()))
        },
    )));
    let fixture = build_fixture(vec![terminal_step("partial answer for the task")], scope).await;

    turn(&fixture.session, "make retries configurable").await;

    let calls = control_calls();
    assert!(!calls.is_empty(), "the decide dispatched");
    let first_model = calls[0]["model"].as_str().unwrap_or_default().to_string();
    assert_eq!(
        first_model, "jev-model-a",
        "the wire carried the captured requested-model identity"
    );
    let restored = fresh_store().load().requested_model;
    assert_eq!(
        restored.as_deref(),
        Some("jev-model-a"),
        "the durable model selection returned to identical bytes"
    );
    assert_eq!(
        fixture.captured.lock().unwrap().len(),
        1,
        "no continuation for the refused ABA decision"
    );
    let (remaining, available) = feedback_remaining_via_fresh_book(&fixture.session_id);
    assert!(available);
    assert_eq!(remaining, 2, "model writes never spend control budget");
}

/// H-CONTROL-1 boundary: without the full profile, NO control dispatch
/// happens at all and NO ledger record exists for the session — even with a
/// mode that would otherwise allow native dispatches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_full_profile_no_control_dispatch_and_no_ledger() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::CompareAndActive,
        /* install_full */ false,
        false,
        "mock-control",
        None,
    );
    let fixture = build_fixture(vec![terminal_step("plain answer")], scope).await;

    turn(&fixture.session, "plain task with no full profile").await;

    assert!(
        control_calls().is_empty(),
        "no control question ids reached the transport without the full profile"
    );
    assert!(
        ledger_records_for(&fixture.session_id).is_empty(),
        "no control ledger record for the session (no epoch mint, no spend)"
    );
}

/// INDEPENDENT COMPACTION stays independent: with decisions Off (mode Off)
/// and compaction enabled, the independent compaction decide STILL dispatches
/// through the real independent lane (fresh single-load gate, requested-model
/// stamp) and APPLIES: a large old tool result is truncated by the real
/// apply path. No control dispatch, no ledger record.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_compaction_decides_while_control_decisions_off() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        /* mode */ JevMode::Off,
        /* install_full */ false,
        /* compaction */ true,
        "mock-control",
        None,
    );
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            compaction_response(request)
        },
    )));
    let fixture = build_fixture(vec![], scope).await;
    // Register the session with the native bridge. Decisions remain Off and
    // this ordinary baseline turn cannot create CONTROL ledger state.
    turn(&fixture.session, "prime independent compaction context").await;

    let values = eligible_compaction_values();
    let runner = fixture
        .session
        .extension_runner()
        .expect("native extension runner");
    let ctx = runner.create_context();
    let compacted =
        pi_coding_agent::core::jev_compaction::compact_context(ctx, values, None).await;

    let calls = compaction_calls();
    assert!(
        !calls.is_empty(),
        "the independent compaction decide dispatched with decisions Off; status={:?}; calls={:?}",
        pi_coding_agent::core::jev_compaction::session_status(&fixture.session_id),
        pi_coding_agent::core::jev_bridge::debug_control_fixture_calls()
    );
    assert_eq!(
        calls[0]["model"].as_str().unwrap_or_default(),
        "jev-latest",
        "the independent lane stamps the requested-model default identity"
    );
    assert!(
        control_calls().is_empty(),
        "control decisions stayed Off the whole time"
    );
    assert!(
        ledger_records_for(&fixture.session_id).is_empty(),
        "no control ledger record while the profile is off"
    );

    let truncated = compacted.iter().any(|message| {
        message["role"] == json!("toolResult")
            && message["content"].as_array().is_some_and(|blocks| {
                blocks.iter().any(|block| {
                    block["text"]
                        .as_str()
                        .is_some_and(|text| text.contains(TRUNCATION_MARKER))
                })
            })
    });
    assert!(
        truncated,
        "the real independent compaction APPLIED: the big old result was truncated"
    );
}

/// Independent compaction uses the same durable identity discipline as
/// CONTROL. Every otherwise-valid decide below dispatches, mutates the real
/// settings store while the transport is awaited, and is refused without
/// changing the caller's context. The three cases cover identical-value save,
/// full-profile Off->On->Off ABA, and requested-model A->B->A ABA.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_compaction_refuses_all_held_identity_changes() {
    let scope = begin_test_scope();
    save_control_settings(
        shared_agent_dir(),
        JevMode::Off,
        false,
        true,
        "mock-control",
        None,
    );
    let phase = Arc::new(AtomicUsize::new(0));
    let callback_phase = Arc::clone(&phase);
    pi_coding_agent::core::jev_bridge::debug_control_fixture_set_respond(Some(Box::new(
        move |request: &SystemOneRequest| -> Option<SystemOneResponse> {
            if !request.questions.keys().all(|id| id.starts_with("compaction.")) {
                return None;
            }
            match callback_phase.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let store = fresh_store();
                    let settings = store.load();
                    store.save(&settings).expect("identical authoritative save");
                }
                1 => {
                    let store = fresh_store();
                    let mut on = store.load();
                    assert!(on.full_jev_install(), "profile started off");
                    store.save(&on).expect("authoritative full-profile on save");
                    let mut off = store.load();
                    assert!(off.full_jev_remove(), "profile is on inside await");
                    store.save(&off).expect("authoritative full-profile off restore");
                }
                2 => {
                    let store = fresh_store();
                    let mut to_b = store.load();
                    assert!(to_b.set_requested_model("jev-model-b").expect("valid B"));
                    store.save(&to_b).expect("authoritative model B save");
                    let mut back_to_a = store.load();
                    assert!(back_to_a.set_requested_model("jev-model-a").expect("valid A"));
                    store.save(&back_to_a).expect("authoritative model A restore");
                }
                other => panic!("unexpected extra independent decide {other}"),
            }
            compaction_response(request)
        },
    )));
    let fixture = build_fixture(vec![], scope).await;
    turn(&fixture.session, "prime held independent compaction context").await;
    let runner = fixture
        .session
        .extension_runner()
        .expect("native extension runner");
    let original = eligible_compaction_values();

    for case in 0..3 {
        if case == 2 {
            let store = fresh_store();
            let mut model_a = store.load();
            assert!(model_a.set_requested_model("jev-model-a").expect("valid A"));
            store.save(&model_a).expect("capture model A before decide");
        }
        let ctx = runner.create_context();
        let result = pi_coding_agent::core::jev_compaction::compact_context(
            ctx,
            original.clone(),
            None,
        )
        .await;
        assert_eq!(
            result, original,
            "held identity case {case} must leave the original context byte-for-byte unchanged"
        );
    }

    let calls = compaction_calls();
    assert_eq!(calls.len(), 3, "all three otherwise-valid decides dispatched");
    assert_eq!(phase.load(Ordering::SeqCst), 3);
    assert_eq!(calls[0]["model"], json!("jev-latest"));
    assert_eq!(calls[1]["model"], json!("jev-latest"));
    assert_eq!(
        calls[2]["model"],
        json!("jev-model-a"),
        "model ABA request carried its captured A identity"
    );
    let restored = fresh_store().load();
    assert!(!restored.full_jev_active(), "full profile returned to Off bytes");
    assert_eq!(restored.requested_model.as_deref(), Some("jev-model-a"));
    assert!(control_calls().is_empty(), "CONTROL stayed Off");
    assert!(
        ledger_records_for(&fixture.session_id).is_empty(),
        "independent refusals never create CONTROL ledger state"
    );
}
