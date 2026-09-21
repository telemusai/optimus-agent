//! NATIVE skill-hint wiring capture (root wiring contract v7 + blockers 1-3).
//!
//! Proves through the REAL session, REAL bridge dispatch, REAL observer decide
//! (debug-only identity-keyed fixture transport) and the REAL primary provider
//! request capture (faux provider Factory context):
//! - POSITIVE delivery on a SINGLE primary request: a hint assessed at the
//!   awaited pre-context `before_request` boundary is served to the VERY
//!   request whose boundary assessed it (live get_system_prompt hook).
//! - Inner tool-continuation turns: every provider request of the run carries
//!   exactly one current hint (one assessment per loop turn).
//! - Task switch: a new task never inherits the previous task's hint.
//! - External Off->On ABA with no consumer visit during Off: the old hint is
//!   rejected at the next request (durable settings write revision moved
//!   through the authoritative save path; nothing restamped after await).
//! - Feature-off and mode-off removal: no hint renders, book cleared.
//! - Verified-state-on-the-wire: the transport observes EXACTLY the adapter's
//!   verified `PreparedGuidance.state` (user_text_excerpt + disclosure keys).

#![allow(clippy::all)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// P1 (hint-recovery review): these tests mutate the process-global
/// PRIME_AGENT_CODING_AGENT_DIR env AND a global HINT_FIXTURE_LAST_SUGGESTION
/// wire-capture cell. Cargo runs tests in parallel by default, so a single
/// test-lifetime mutex serializes the whole binary; the guard is held by the
/// Fixture and released when the fixture drops. Zero production changes.
static HINT_WIRING_ENV_LOCK: Mutex<()> = Mutex::new(());

use pi_ai::providers::faux::{
    faux_assistant_message, faux_tool_call, register_faux_provider, FauxAssistantContent,
    FauxAssistantMessageOptions, FauxProviderRegistration, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{AssistantMessage, ContentBlock};
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
use serde_json::{json, Map};

// ---------------------------------------------------------------------------
// Environment isolation (one process-wide agent dir for the whole scenario)
// ---------------------------------------------------------------------------

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
// Provider request capture (the REAL primary-provider request context)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CapturedRequest {
    system_prompt: Option<String>,
    user_texts: Vec<String>,
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
            let user_texts: Vec<String> = context
                .messages
                .iter()
                .filter_map(|message| match message {
                    pi_ai::types::Message::User(user) => match &user.content {
                        pi_ai::types::UserContent::Text(text) => Some(text.clone()),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            captured.lock().unwrap().push(CapturedRequest {
                system_prompt: context.system_prompt.clone(),
                user_texts,
            });
            let mut guard = steps.lock().unwrap();
            let message = if guard.is_empty() {
                text_reply(999)
            } else {
                guard.remove(0)
            };
            Box::pin(async move { message })
        },
    )
}

fn text_reply(index: usize) -> AssistantMessage {
    faux_assistant_message(
        FauxAssistantContent::Text(format!("HINT_WIRING_REPLY_{index:02}_END")),
        None,
    )
}

fn tool_call_reply(index: usize) -> AssistantMessage {
    faux_assistant_message(
        FauxAssistantContent::Blocks(vec![ContentBlock::ToolCall(faux_tool_call(
            "jev_hint_probe",
            Map::new(),
            None,
        ))]),
        Some(FauxAssistantMessageOptions {
            stop_reason: Some("toolUse".to_string()),
            ..Default::default()
        }),
    )
    .with_index(index)
}

trait WithIndex {
    fn with_index(self, index: usize) -> AssistantMessage;
}

impl WithIndex for AssistantMessage {
    fn with_index(self, _index: usize) -> AssistantMessage {
        self
    }
}

/// The probe tool: returns static text after a short bounded wait.
fn probe_tool() -> pi_coding_agent::core::extensions::types::ToolDefinition {
    use pi_coding_agent::core::extensions::types::ToolDefinition;
    ToolDefinition {
        name: "jev_hint_probe".to_string(),
        label: "Jev hint wiring probe".to_string(),
        description: "Returns static text; used to sequence the hint assessment".to_string(),
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
                tokio::time::sleep(Duration::from_millis(120)).await;
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

// ---------------------------------------------------------------------------
// Settings through the AUTHORITATIVE save path (durable write revision moves)
// ---------------------------------------------------------------------------

fn save_settings(
    agent_dir: &std::path::Path,
    mode: JevMode,
    skill_suggestion: bool,
    transport: &str,
) {
    let mut settings = JevSettings::default();
    settings.global_default = Some(mode);
    // ONLY the skill-suggestion gate: the JevFeatures defaults enable
    // tool_requirement/complexity, whose shadow batteries would add unrelated
    // decide traffic on top of the identity-keyed suggestion decides.
    settings.features = JevFeatures {
        tool_requirement: false,
        complexity: false,
        skill_suggestion,
        ..JevFeatures::default()
    };
    settings.transport = Some(transport.to_string());
    JevSettingsStore::new(agent_dir)
        .save(&settings)
        .expect("authoritative settings save");
    pi_coding_agent::core::jev_bridge::invalidate_settings_cache();
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct Fixture {
    session: Arc<AgentSession>,
    provider: FauxProviderRegistration,
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
    steps: Arc<Mutex<Vec<AssistantMessage>>>,
    _dir: tempfile::TempDir,
    _guard: EnvGuard,
    _env_lock: std::sync::MutexGuard<'static, ()>,
}

async fn build_fixture(provider_name: &str) -> Fixture {
    build_fixture_with(provider_name, FixtureOptions::default()).await
}

/// Builder options for `build_fixture_with`. Defaults reproduce the original
/// `build_fixture` behavior exactly, so existing tests keep their semantics.
struct FixtureOptions {
    /// `Some((mode, skill_suggestion))` saves settings through the
    /// authoritative path BEFORE session creation. `None` keeps the never-Jev
    /// baseline: no settings file exists and the built-in mode resolution is
    /// `JevMode::Off`, so no suggestion decide runs at the first request
    /// boundary.
    settings: Option<(JevMode, bool)>,
    /// Project skills written under `<cwd>/.prime/agent/skills`.
    skills: Vec<(&'static str, &'static str)>,
    /// Disable every normal skill root for the hermetic no-roster case. This
    /// maps to the existing loader option and does not change production
    /// loading outside that fixture.
    no_skills: bool,
    /// In-process extension factories passed through the loader options.
    /// They load even with `no_extensions: true`.
    extension_factories: Vec<pi_coding_agent::core::extensions::types::ExtensionFactory>,
}

impl Default for FixtureOptions {
    fn default() -> Self {
        Self {
            settings: Some((JevMode::CompareAndActive, true)),
            skills: vec![
                (
                    "alpha-skill",
                    "documented procedure for the alpha retry policy",
                ),
                (
                    "beta-skill",
                    "documented procedure for the beta cleanup flow",
                ),
            ],
            no_skills: false,
            extension_factories: Vec::new(),
        }
    }
}

async fn build_fixture_with(provider_name: &str, options: FixtureOptions) -> Fixture {
    let scratch = std::env::temp_dir();
    let dir = tempfile::Builder::new()
        .prefix("jev-hint-wiring-")
        .tempdir_in(scratch)
        .unwrap();
    let cwd = dir.path().join("workspace");
    let agent_dir = dir.path().join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&agent_dir).unwrap();
    // P1: acquire the test-lifetime lock BEFORE every process-global env or
    // capture mutation. The Fixture owns this guard until teardown; its
    // EnvGuard field drops first, so restoring the env is protected too.
    let env_lock = HINT_WIRING_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_reset();
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let guard = EnvGuard::new(&agent_dir);

    // Real project skills (the actual loaded roster the prompt renders).
    let skills_dir = cwd.join(".prime").join("agent").join("skills");
    for (name, description) in options.skills.iter().copied() {
        let skill_dir = skills_dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: {description}\n---\n\nUse the {name} procedure.\n"
            ),
        )
        .unwrap();
    }

    // Authoritative settings BEFORE session creation (write revision 1).
    // `settings: None` keeps the never-Jev baseline (no file, built-in Off).
    if let Some((mode, skill_suggestion)) = options.settings {
        save_settings(&agent_dir, mode, skill_suggestion, "mock-hint");
    }

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(provider_name.to_string()),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let mut model = provider.get_model();

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
        .set_runtime_api_key(&model.provider, "synthetic-hint-wiring-key");
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
    model_registry
        .lock()
        .expect("model registry poisoned")
        .set_runtime_api_key(&model.provider, "synthetic-hint-wiring-key");

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(
            Some(&cwd.to_string_lossy()),
            Some(&agent_dir.to_string_lossy()),
        )
        .unwrap(),
    ));

    let loader_options = DefaultResourceLoaderOptions {
        cwd: cwd.to_string_lossy().to_string(),
        agent_dir: agent_dir.to_string_lossy().to_string(),
        no_extensions: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        no_skills: options.no_skills,
        bundled_skills_dir: Some(None),
        extension_factories: options.extension_factories,
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
    let steps: Arc<Mutex<Vec<AssistantMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let factory_future = {
        let captured = Arc::clone(&captured);
        let steps = Arc::clone(&steps);
        capture_factory(captured, steps)
    };
    // The faux registration consumes ONE list item per provider request;
    // every item shares the same capture factory (which re-arms the scripted
    // reply from the shared steps queue).
    provider.set_responses(
        (0..12)
            .map(|_| FauxResponseStep::Factory(Arc::clone(&factory_future)))
            .collect(),
    );

    let creation = AgentSessionCreationOptions {
        model: Some(model.clone()),
        custom_tools: Some(vec![probe_tool()]),
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
        captured,
        steps,
        _dir: dir,
        _guard: guard,
        _env_lock: env_lock,
    }
}

fn push_steps(fixture: &Fixture, steps: Vec<AssistantMessage>) {
    fixture.steps.lock().unwrap().extend(steps);
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

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// The hint block is a BRIDGE-OWNED fixed wrapper (never user-supplied text),
/// so structural extraction is the compliant owned-slot check.
fn hint_blocks(prompt: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut rest = prompt;
    while let Some(start) = rest.find("<jev_skill_hint>") {
        let after_start = &rest[start..];
        let end = after_start
            .find("</jev_skill_hint>")
            .map(|end| start + end + "</jev_skill_hint>".len())
            .unwrap_or(rest.len());
        blocks.push(&rest[start..end]);
        rest = &rest[end..];
    }
    blocks
}

fn host_hint_blocks(prompt: &str) -> Vec<&str> {
    hint_blocks(prompt)
        .into_iter()
        .filter(|block| {
            block.contains("Assessment-only advisory hint from the independent Jev observer")
        })
        .collect()
}

fn roster_parity(base: &str, hinted: &str) {
    let blocks = hint_blocks(hinted);
    assert_eq!(blocks.len(), 1, "exactly one hint block");
    let block_text = blocks[0];
    // Locate the block in the hinted prompt and remove it together with the
    // separator the formatter owns (a leading blank-line pair).
    let position = hinted.find(block_text).unwrap();
    let mut stripped = String::new();
    stripped.push_str(&hinted[..position]);
    stripped.push_str(&hinted[position + block_text.len()..]);
    // The formatter prepends "\n\n"; strip exactly one owned separator pair.
    let stripped = stripped.strip_suffix('\n').unwrap_or(&stripped);
    let stripped = stripped.strip_suffix('\n').unwrap_or(stripped);
    assert_eq!(
        base.trim_end(),
        stripped.trim_end(),
        "roster and surrounding prompt must stay byte-identical except the owned hint block"
    );
}

/// The wire state the fixture transport observed for the last skill-suggestion
/// decide must equal the adapter's VERIFIED `PreparedGuidance.state` bytes:
/// the disclosed excerpt plus the exact disclosure key set.
fn assert_wire_state_is_verified_state(wire_state: &serde_json::Value, task_text: &str) {
    let obj = wire_state
        .as_object()
        .expect("suggestion wire state is an object");
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "assessment",
            "hint_authority",
            "skill_catalog_count",
            "skill_catalog_skipped_overlong_ids",
            "skill_catalog_truncated",
            "user_text_excerpt",
        ],
        "the wire state carries exactly the verified adapter disclosure keys"
    );
    assert_eq!(
        obj["user_text_excerpt"],
        json!(task_text),
        "the assessed excerpt is the disclosed task text (short inputs are not truncated)"
    );
    assert_eq!(obj["assessment"], json!("skill_suggestion"));
    assert_eq!(obj["hint_authority"], json!("none"));
    // Exact count/truncation/skip parity is asserted below against the
    // session's actual model-visible roster through the production view.
    let count = obj["skill_catalog_count"].as_u64().expect("count");
    assert!(count >= 2, "at least the fixture's two project skills");
}

/// Hermetic disclosure for the ACTUAL session-loaded, model-visible roster.
/// The bridge debug accessor delegates to the production visibility, safe-id
/// and truncation rule over the roster cached at the consumer seam. This is
/// intentionally not a second ambient loader with merely similar defaults.
fn actual_roster_disclosure(session_id: &str) -> (usize, bool, usize) {
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_roster_disclosure(session_id)
        .expect("session pushed its actual model-visible skill roster")
}

fn last_suggestion_wire() -> Option<(serde_json::Value, Vec<String>)> {
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_last_suggestion()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_primary_request_positive_delivery() {
    let fixture = build_fixture("jev-hint-wiring-single").await;
    let session = &fixture.session;
    let session_id = session.session_id();

    // ONE primary request: no tool call, one scripted reply.
    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;

    let captured = fixture.captured.lock().unwrap();
    assert_eq!(
        captured.len(),
        1,
        "a single-response task makes exactly one provider request"
    );
    let prompt = captured[0].system_prompt.clone().expect("system prompt");
    let blocks = hint_blocks(&prompt);
    assert_eq!(
        blocks.len(),
        1,
        "POSITIVE delivery: the primary request itself carries the assessed hint"
    );
    assert!(
        prompt.contains("alpha-skill"),
        "the fixture roster renders in the prompt"
    );
    assert!(
        blocks[0].contains("Most relevant loaded skill:"),
        "the hint names the ranked skill"
    );
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some(),
        "the assessment stored the advisory hint"
    );
    drop(captured);

    // The wire bytes are the VERIFIED adapter state (identity-keyed fixture).
    let (wire_state, wire_ids) =
        last_suggestion_wire().expect("the suggestion decide reached the transport");
    assert_wire_state_is_verified_state(&wire_state, "please follow the alpha retry policy task");
    let (actual_count, actual_truncated, actual_skipped) = actual_roster_disclosure(&session_id);
    assert_eq!(
        wire_state["skill_catalog_count"],
        json!(actual_count),
        "catalog count derives from the actual session-loaded visible roster"
    );
    assert_eq!(
        wire_state["skill_catalog_truncated"],
        json!(actual_truncated),
        "truncation disclosure uses the production roster view"
    );
    assert_eq!(
        wire_state["skill_catalog_skipped_overlong_ids"],
        json!(actual_skipped),
        "skip disclosure uses the production visibility and safe-id rule over the actual session roster"
    );
    assert_eq!(
        wire_ids,
        vec![
            "skill_suggestion.0",
            "skill_suggestion.1",
            "skill_suggestion.2",
            "skill_suggestion.3"
        ],
        "the adapter's rank + three gate questions went out together"
    );

    // Roster byte parity against a no-hint request of the SAME session.
    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "the second task adds one request");
    let base_prompt = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        hint_blocks(&base_prompt).len(),
        0,
        "the beta task ranks none"
    );
    roster_parity(&base_prompt, &prompt);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none(),
        "the beta assessment cleared the stored hint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inner_tool_turns_carry_the_assessed_hint() {
    let fixture = build_fixture("jev-hint-wiring-inner").await;
    let session = &fixture.session;
    let session_id = session.session_id();

    push_steps(&fixture, vec![tool_call_reply(1), text_reply(2)]);
    turn(session, "please follow the alpha retry policy task").await;

    let captured = fixture.captured.lock().unwrap();
    assert_eq!(
        captured.len(),
        2,
        "the tool-continuation pair made exactly two provider requests"
    );
    let prompt_first = captured[0].system_prompt.clone().expect("system prompt");
    let prompt_second = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        hint_blocks(&prompt_first).len(),
        1,
        "the FIRST request carries the hint assessed at its own boundary"
    );
    assert_eq!(
        hint_blocks(&prompt_second).len(),
        1,
        "the inner continuation carries its own per-turn assessment"
    );
    assert!(prompt_first.contains("alpha-skill"));
    // The two assessments agree, so the non-block prompt bytes are identical.
    assert_eq!(
        prompt_first, prompt_second,
        "same task, same roster, same hint bytes"
    );
    drop(captured);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some(),
        "the last per-turn assessment stored the hint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn aba_off_on_then_new_tasks_reassess() {
    let fixture = build_fixture("jev-hint-wiring-aba").await;
    let session = &fixture.session;
    let session_id = session.session_id();

    // Task A with a tool continuation (hint on both requests).
    push_steps(&fixture, vec![tool_call_reply(1), text_reply(2)]);
    turn(session, "please follow the alpha retry policy task").await;

    // External Off->On with NO consumer visit during Off: authoritative saves
    // only (durable write revision moves); nothing restamps after await.
    {
        let agent_dir = std::env::var_os(ENV_AGENT_DIR).unwrap();
        let agent_dir = std::path::PathBuf::from(agent_dir);
        save_settings(&agent_dir, JevMode::Off, true, "mock-hint");
        save_settings(&agent_dir, JevMode::CompareAndActive, true, "mock-hint");
    }

    // Task B (beta): the previous hint must not reach the new task.
    push_steps(&fixture, vec![tool_call_reply(3), text_reply(4)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    let total = captured.len();
    assert!(total >= 4, "task B produced its own requests");
    for request in &captured[total - 2..] {
        let prompt = request.system_prompt.clone().expect("system prompt");
        assert_eq!(
            hint_blocks(&prompt).len(),
            0,
            "the previous task's hint never reaches the new task's requests"
        );
    }
    drop(captured);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none(),
        "task B's NoHint decide cleared the stored hint"
    );

    // Task C (alpha again): a fresh assessment installs again.
    push_steps(&fixture, vec![tool_call_reply(5), text_reply(6)]);
    turn(session, "please follow the alpha retry policy again").await;
    let captured = fixture.captured.lock().unwrap();
    let total = captured.len();
    for request in &captured[total - 2..] {
        let prompt = request.system_prompt.clone().expect("system prompt");
        assert_eq!(
            hint_blocks(&prompt).len(),
            1,
            "a fresh post-ABA assessment installs a new hint on each request"
        );
    }
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_off_and_mode_off_clear_and_suppress() {
    let fixture = build_fixture("jev-hint-wiring-off").await;
    let session = &fixture.session;
    let session_id = session.session_id();

    // First install a real owned hint, then change to Compare-only. Compare's
    // assessment path returns without installing; the awaited consumer must
    // still refresh CURRENT gates before this next provider context is built.
    push_steps(&fixture, vec![text_reply(0)]);
    turn(session, "please follow the alpha retry policy task").await;
    let captured = fixture.captured.lock().unwrap();
    let active_prompt = captured
        .last()
        .and_then(|request| request.system_prompt.clone())
        .unwrap();
    assert_eq!(
        host_hint_blocks(&active_prompt).len(),
        1,
        "active request owns one hint"
    );
    drop(captured);
    {
        let agent_dir = std::path::PathBuf::from(std::env::var_os(ENV_AGENT_DIR).unwrap());
        save_settings(&agent_dir, JevMode::Compare, true, "mock-hint");
    }
    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "compare-only stale-hint transition").await;
    let captured = fixture.captured.lock().unwrap();
    let compare_prompt = captured
        .last()
        .and_then(|request| request.system_prompt.clone())
        .unwrap();
    assert_eq!(
        host_hint_blocks(&compare_prompt).len(),
        0,
        "Compare early return cannot leak the previously owned hint"
    );
    drop(captured);

    // Feature off: no decide at all; consumption gate clears the book.
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_reset();
    {
        let agent_dir = std::env::var_os(ENV_AGENT_DIR).unwrap();
        let agent_dir = std::path::PathBuf::from(agent_dir);
        save_settings(&agent_dir, JevMode::CompareAndActive, false, "mock-hint");
    }
    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "feature off task").await;
    let captured = fixture.captured.lock().unwrap();
    let d_prompt = captured
        .last()
        .and_then(|request| request.system_prompt.clone())
        .expect("system prompt");
    assert_eq!(hint_blocks(&d_prompt).len(), 0, "feature off: no hint");
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
    assert!(
        last_suggestion_wire().is_none(),
        "feature off: no suggestion decide reached the transport"
    );

    // Mode off: the Off path clears the book, nothing renders.
    {
        let agent_dir = std::env::var_os(ENV_AGENT_DIR).unwrap();
        let agent_dir = std::path::PathBuf::from(agent_dir);
        save_settings(&agent_dir, JevMode::Off, true, "mock-hint");
    }
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_reset();
    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "mode off task").await;
    let captured = fixture.captured.lock().unwrap();
    let e_prompt = captured
        .last()
        .and_then(|request| request.system_prompt.clone())
        .expect("system prompt");
    assert_eq!(hint_blocks(&e_prompt).len(), 0, "mode off: no hint");
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
    assert!(
        last_suggestion_wire().is_none(),
        "mode off: no suggestion decide reached the transport"
    );
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn override_composition_weaves_the_hint_into_the_embedded_base() {
    let fixture = build_fixture_with(
        "jev-hint-wiring-weave",
        FixtureOptions {
            extension_factories: vec![weave_factory()],
            ..FixtureOptions::default()
        },
    )
    .await;
    let session = &fixture.session;
    let session_id = session.session_id();

    // ALPHA: hint assessed and woven INTO the embedded base occurrence.
    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "one provider request");
    let prompt = captured[0].system_prompt.clone().expect("system prompt");
    let blocks = hint_blocks(&prompt);
    let host_blocks = host_hint_blocks(&prompt);
    assert_eq!(
        blocks.len(),
        2,
        "one user literal plus one host-owned hint block"
    );
    assert_eq!(host_blocks.len(), 1, "at most one HOST-added hint block");
    assert!(
        prompt.contains(USER_LITERAL_HINT_BLOCK),
        "literal marker-shaped extension text survives the active weave"
    );
    assert!(
        prompt.contains(WEAVE_SUFFIX_MARKER),
        "the extension suffix survives the weave"
    );
    assert!(
        prompt.rfind(host_blocks[0]).expect("host block present")
            < prompt.rfind(WEAVE_SUFFIX_MARKER).expect("suffix present"),
        "the owned block rides inside the base occurrence, before the suffix"
    );
    drop(captured);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some(),
        "the assessment stored the advisory hint"
    );

    // BETA: NoHint decide; the commit seam re-applies the composition, and the
    // owned block must be gone while the suffix stays.
    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "the beta task adds one request");
    let base_prompt = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        host_hint_blocks(&base_prompt).len(),
        0,
        "no host hint on beta"
    );
    assert_eq!(
        hint_blocks(&base_prompt).len(),
        1,
        "the user literal block remains"
    );
    assert!(base_prompt.contains(USER_LITERAL_HINT_BLOCK));
    assert!(
        base_prompt.contains(WEAVE_SUFFIX_MARKER),
        "the suffix survives the clean turn"
    );
    let block = host_hint_blocks(&prompt)[0];
    let block_at = prompt.find(block).expect("owned hint block");
    let owned_separator_at = prompt[..block_at]
        .strip_suffix("\n\n")
        .map(|prefix| prefix.len())
        .expect("owned hint separator");
    let stripped = format!(
        "{}{}",
        &prompt[..owned_separator_at],
        &prompt[block_at + block.len()..]
    );
    assert_eq!(
        stripped, base_prompt,
        "embedded-base weave changes only the owned hint block and separator"
    );
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_override_composition_fails_open_and_is_never_clobbered() {
    let fixture = build_fixture_with(
        "jev-hint-wiring-override",
        FixtureOptions {
            extension_factories: vec![override_factory()],
            ..FixtureOptions::default()
        },
    )
    .await;
    let session = &fixture.session;
    let session_id = session.session_id();

    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "one provider request");
    let prompt = captured[0].system_prompt.clone().expect("system prompt");
    assert_eq!(
        prompt, OVERRIDE_PROMPT,
        "FAIL-OPEN: an override that does not embed the base wins unchanged"
    );
    assert_eq!(
        hint_blocks(&prompt).len(),
        1,
        "the user literal marker remains"
    );
    assert_eq!(
        host_hint_blocks(&prompt).len(),
        0,
        "no HOST hint is woven into an unknown override"
    );
    drop(captured);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some(),
        "the assessment still stored the advisory hint (host-projected, never rendered)"
    );

    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "the beta task adds one request");
    let base_prompt = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        base_prompt, OVERRIDE_PROMPT,
        "the override survives the clean turn byte-identically"
    );
    assert_eq!(
        hint_blocks(&base_prompt).len(),
        1,
        "literal marker survives NoHint"
    );
    assert_eq!(host_hint_blocks(&base_prompt).len(), 0);
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());

    // Off is another early-refusal path. The every-request renderer must not
    // treat marker-shaped external text as ownership evidence.
    {
        let agent_dir = std::path::PathBuf::from(std::env::var_os(ENV_AGENT_DIR).unwrap());
        save_settings(&agent_dir, JevMode::Off, true, "mock-hint");
    }
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_reset();
    push_steps(&fixture, vec![text_reply(3)]);
    turn(session, "mode off must preserve the literal override").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 3);
    let off_prompt = captured[2].system_prompt.clone().expect("system prompt");
    assert_eq!(
        off_prompt, OVERRIDE_PROMPT,
        "Off preserves unknown override bytes"
    );
    assert_eq!(hint_blocks(&off_prompt).len(), 1);
    assert_eq!(host_hint_blocks(&off_prompt).len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn never_jev_baseline_prompt_is_byte_stable_through_first_activation() {
    let fixture = build_fixture_with(
        "jev-hint-wiring-baseline",
        FixtureOptions {
            settings: None,
            ..FixtureOptions::default()
        },
    )
    .await;
    let session = &fixture.session;
    let session_id = session.session_id();

    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "one provider request");
    let baseline_prompt = captured[0].system_prompt.clone().expect("system prompt");
    assert_eq!(hint_blocks(&baseline_prompt).len(), 0, "never-Jev: no hint");
    drop(captured);
    assert!(
        last_suggestion_wire().is_none(),
        "never-Jev: no decide reached the transport"
    );
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());

    // FIRST activation through the authoritative save path. No consumer visit
    // happened while the built-in Off was active.
    {
        let agent_dir = std::env::var_os(ENV_AGENT_DIR).unwrap();
        let agent_dir = std::path::PathBuf::from(agent_dir);
        save_settings(&agent_dir, JevMode::CompareAndActive, true, "mock-hint");
    }

    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "the activated task adds one request");
    let after = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(hint_blocks(&after).len(), 0, "the beta task ranks none");
    assert_eq!(
        after, baseline_prompt,
        "the first activation is prompt-invisible for the request that follows"
    );
    drop(captured);
    assert!(
        last_suggestion_wire().is_some(),
        "the activation made the decide run on the wire"
    );
    let (wire_state, _) = last_suggestion_wire().expect("decide observed");
    assert_eq!(
        wire_state["user_text_excerpt"],
        json!("please follow the beta cleanup task now"),
        "the decide assessed the CURRENT task text"
    );
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compare_only_mode_never_installs_even_when_the_transport_would_answer() {
    let fixture = build_fixture_with(
        "jev-hint-wiring-compare",
        FixtureOptions {
            settings: Some((JevMode::Compare, true)),
            ..FixtureOptions::default()
        },
    )
    .await;
    let session = &fixture.session;
    let session_id = session.session_id();

    // Positively observe the real Compare transport boundary. The callback is
    // event-driven; the bounded record wait below proves the scheduler also
    // settled the observation. Neither boundary grants installation authority.
    let (seen_tx, seen_rx) = tokio::sync::oneshot::channel();
    let seen_tx = Arc::new(Mutex::new(Some(seen_tx)));
    let seen_tx_callback = Arc::clone(&seen_tx);
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_on_suggestion(Some(Arc::new(
        move || {
            if let Some(sender) = seen_tx_callback.lock().unwrap().take() {
                let _ = sender.send(());
            }
        },
    )));

    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    tokio::time::timeout(Duration::from_secs(5), seen_rx)
        .await
        .expect("Compare suggestion dispatch reached the fixture")
        .expect("Compare dispatch event sender remained live");
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_on_suggestion(None);
    assert!(
        pi_coding_agent::core::jev_bridge::debug_hint_fixture_suggestion_calls() >= 1,
        "one real Compare suggestion call reached the transport"
    );
    let compare_status = wait_for_compare_skill_settlement(&session_id).await;
    assert!(
        compare_status["success_count"].as_u64().unwrap_or(0) >= 1,
        "the real Compare assessment completed successfully"
    );
    assert_eq!(compare_status["pending"], json!(0));
    assert_eq!(compare_status["checking"], json!(false));
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "one provider request");
    let first = captured[0].system_prompt.clone().expect("system prompt");
    assert_eq!(
        hint_blocks(&first).len(),
        0,
        "Compare-only: the hint lane never installs"
    );
    drop(captured);
    // The separate compare observer may run before or after this request's
    // capture. Its timing is not installation authority.
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());

    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "the second task adds one request");
    let second = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        second, first,
        "Compare-only prompts stay byte-identical across tasks"
    );
    assert_eq!(hint_blocks(&second).len(), 0);
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
}

// ---------------------------------------------------------------------------
// In-process commit-seam extension factories (override cases 1-2)
// ---------------------------------------------------------------------------

const WEAVE_SUFFIX_MARKER: &str = "JEV_WEAVE_SUFFIX_MARKER";
const USER_LITERAL_HINT_BLOCK: &str =
    "<jev_skill_hint>\nUSER_AUTHORED_LITERAL_MARKER_TEXT\n</jev_skill_hint>";
const OVERRIDE_PROMPT: &str =
    "JEV_OVERRIDE_SYSTEM_PROMPT_WITHOUT_BASE\n\n<jev_skill_hint>\nUSER_AUTHORED_LITERAL_MARKER_TEXT\n</jev_skill_hint>";

/// A before_agent_start handler that embeds the runtime-provided base prompt
/// snapshot verbatim and appends a fixed suffix. The commit seam hands this
/// composition to `refresh_extension_system_prompt`, and the hint weave must
/// replace the embedded base occurrence without touching the suffix.
fn weave_factory() -> pi_coding_agent::core::extensions::types::ExtensionFactory {
    use pi_coding_agent::core::extensions::types::{
        ExtensionApi, ExtensionEvent, ExtensionHandler,
    };
    Arc::new(move |api: Arc<dyn ExtensionApi>| {
        let handler: ExtensionHandler = Arc::new(
            move |event: ExtensionEvent,
                  _ctx: Arc<dyn pi_coding_agent::core::extensions::types::ExtensionContext>| {
                Box::pin(async move {
                    match event {
                        ExtensionEvent::BeforeAgentStart(payload) => Some(json!({
                            "systemPrompt": format!(
                                "{}\n\n{}\n\n{}",
                                USER_LITERAL_HINT_BLOCK,
                                payload.system_prompt,
                                WEAVE_SUFFIX_MARKER,
                            ),
                        })),
                        _ => None,
                    }
                })
            },
        );
        api.on("before_agent_start", handler);
        Box::pin(async move { Ok(()) })
    })
}

/// A before_agent_start handler that returns a fixed composition which does
/// NOT embed the canonical base. `refresh_extension_system_prompt` keeps it
/// unchanged, and the request weave must fail open (no hint, no clobber).
fn override_factory() -> pi_coding_agent::core::extensions::types::ExtensionFactory {
    use pi_coding_agent::core::extensions::types::{
        ExtensionApi, ExtensionEvent, ExtensionHandler,
    };
    Arc::new(move |api: Arc<dyn ExtensionApi>| {
        let handler: ExtensionHandler = Arc::new(
            move |event: ExtensionEvent,
                  _ctx: Arc<dyn pi_coding_agent::core::extensions::types::ExtensionContext>| {
                Box::pin(async move {
                    match event {
                        ExtensionEvent::BeforeAgentStart(_) => {
                            Some(json!({ "systemPrompt": OVERRIDE_PROMPT }))
                        }
                        _ => None,
                    }
                })
            },
        );
        api.on("before_agent_start", handler);
        Box::pin(async move { Ok(()) })
    })
}

// ---------------------------------------------------------------------------
// Missing v10 hint acceptance: no-roster, cancellation, bounded state records
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_actual_roster_never_assesses_and_never_installs() {
    let fixture = build_fixture_with(
        "jev-hint-wiring-noroster",
        FixtureOptions {
            skills: Vec::new(),
            no_skills: true,
            ..FixtureOptions::default()
        },
    )
    .await;
    let session = &fixture.session;
    let session_id = session.session_id();

    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    assert_eq!(
        actual_roster_disclosure(&session_id),
        (0, false, 0),
        "the actual session-loaded production-visible roster is empty"
    );
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "one provider request");
    let prompt = captured[0].system_prompt.clone().expect("system prompt");
    assert_eq!(hint_blocks(&prompt).len(), 0, "no roster: no hint anywhere");
    drop(captured);
    assert!(
        last_suggestion_wire().is_none(),
        "no roster: the decide never ran (empty-roster early return)"
    );
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abort_during_assessment_stores_nothing_and_the_next_task_recovers() {
    let fixture = build_fixture("jev-hint-wiring-cancel").await;
    let session_id = fixture.session.session_id();

    // The fixture's capped timer-backed hold keeps the decide pending after
    // the event callback. The callback schedules the real session abort and
    // stores its JoinHandle so recovery cannot race a tardy abort.
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_set_hold_ms(2_000);
    let abort_slot = Arc::new(Mutex::new(None));
    let abort_slot_for_callback = Arc::clone(&abort_slot);
    let abort_target = Arc::clone(&fixture.session);
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_on_suggestion(Some(Arc::new(
        move || {
            let target = Arc::clone(&abort_target);
            let handle = tokio::spawn(async move { target.abort().await });
            *abort_slot_for_callback.lock().unwrap() = Some(handle);
        },
    )));

    push_steps(&fixture, vec![text_reply(1)]);
    let prompt_session = Arc::clone(&fixture.session);
    let prompt_task = tokio::spawn(async move {
        prompt_session
            .prompt(
                "please follow the alpha retry policy task",
                None::<PromptOptions>,
            )
            .await
    });
    let prompt_join = tokio::time::timeout(Duration::from_secs(30), prompt_task)
        .await
        .expect("the aborted turn returned promptly");
    let _prompt_result = prompt_join.expect("the aborted prompt task did not panic");
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_on_suggestion(None);
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_set_hold_ms(0);
    assert!(
        pi_coding_agent::core::jev_bridge::debug_hint_fixture_suggestion_calls() >= 1,
        "the decide reached the transport before the abort (the window was real)"
    );
    let abort_handle = abort_slot
        .lock()
        .unwrap()
        .take()
        .expect("the suggestion callback scheduled the abort");
    let _abort_result = tokio::time::timeout(Duration::from_secs(30), abort_handle)
        .await
        .expect("the abort task returned promptly")
        .expect("the abort task did not panic");

    tokio::time::timeout(
        Duration::from_secs(30),
        fixture.session.wait_for_headless_idle(),
    )
    .await
    .expect("the session settles after the abort")
    .expect("idle");
    let captured = fixture.captured.lock().unwrap();
    for request in captured.iter() {
        let prompt = request.system_prompt.clone().expect("system prompt");
        assert_eq!(
            hint_blocks(&prompt).len(),
            0,
            "no request of the aborted turn carries a hint"
        );
    }
    let before_recovery = captured.len();
    drop(captured);
    assert!(
        pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none(),
        "the cancelled assessment stored no hint"
    );

    // Drop any scripted reply the cancelled turn never consumed. With the
    // abort task joined and the session idle, it cannot contaminate recovery.
    fixture.steps.lock().unwrap().clear();
    pi_coding_agent::core::jev_bridge::debug_hint_fixture_reset();
    push_steps(&fixture, vec![text_reply(2)]);
    turn(
        &fixture.session,
        "please follow the alpha retry policy again",
    )
    .await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(
        captured.len(),
        before_recovery + 1,
        "recovery adds exactly one fresh primary-provider request"
    );
    let prompt = captured
        .last()
        .unwrap()
        .system_prompt
        .clone()
        .expect("system prompt");
    assert_eq!(
        hint_blocks(&prompt).len(),
        1,
        "the fresh recovery request receives one fresh hint"
    );
    drop(captured);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some());
}

fn skill_lane_records() -> Vec<pi_jev::correlate::CorrelationRecord> {
    let agent_dir = std::env::var_os(ENV_AGENT_DIR).unwrap();
    let path = std::path::PathBuf::from(agent_dir)
        .join("jev")
        .join("records.jsonl");
    pi_jev::correlate::read_records(&path, 8 * 1024 * 1024)
        .into_iter()
        .filter(|record| {
            record.category == "skill_suggestion" && record.stage == "skill_suggestion"
        })
        .collect()
}

async fn wait_for_compare_skill_settlement(session_id: &str) -> serde_json::Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(status) =
                pi_coding_agent::core::jev_bridge::debug_hint_fixture_observer_status(session_id)
            {
                let settled = status["success_count"].as_u64().unwrap_or(0) >= 1
                    && status["pending"].as_u64() == Some(0)
                    && status["checking"].as_bool() == Some(false);
                if settled {
                    return status;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the real Compare observation reached bounded scheduler settlement")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn skill_lane_records_are_bounded_state_events_and_render_is_observed_separately() {
    let fixture = build_fixture("jev-hint-wiring-records").await;
    let session = &fixture.session;
    let session_id = session.session_id();

    push_steps(&fixture, vec![text_reply(1)]);
    turn(session, "please follow the alpha retry policy task").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let rendered = captured[0].system_prompt.clone().expect("system prompt");
    assert_eq!(
        hint_blocks(&rendered).len(),
        1,
        "actual provider capture, not the record, proves prompt projection"
    );
    drop(captured);

    let records = skill_lane_records();
    assert_eq!(
        records.len(),
        1,
        "one bounded state event for the decided hint"
    );
    assert_eq!(records[0].category, "skill_suggestion");
    assert_eq!(records[0].stage, "skill_suggestion");
    assert_eq!(
        records[0].skipped_reason.as_deref(),
        Some("hint_state_updated")
    );
    assert!(!records[0].applied, "the row grants no execution authority");
    assert_eq!(records[0].attempt_count_known, None);
    assert_eq!(records[0].mode, "compare-active");
    assert_eq!(records[0].prompt_version, pi_jev::hooks::PROMPT_VERSION);
    assert!(
        records[0].selected_value.is_none(),
        "no task or roster text is retained"
    );
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_some());

    push_steps(&fixture, vec![text_reply(2)]);
    turn(session, "please follow the beta cleanup task now").await;
    let captured = fixture.captured.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let rendered = captured[1].system_prompt.clone().expect("system prompt");
    assert_eq!(
        hint_blocks(&rendered).len(),
        0,
        "NoHint renders no owned block"
    );
    drop(captured);

    let records = skill_lane_records();
    assert_eq!(
        records.len(),
        2,
        "the NoHint state update adds one bounded row"
    );
    assert_eq!(
        records[1].skipped_reason.as_deref(),
        Some("no_hint_state_updated")
    );
    assert!(!records[1].applied);
    assert_eq!(records[1].attempt_count_known, None);
    assert!(pi_coding_agent::core::jev_bridge::current_skill_hint(&session_id).is_none());
}
