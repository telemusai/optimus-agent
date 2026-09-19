//! T05 - compaction and checkpoint replay (parity validation suite, owner `compaction`).
//!
//! Every test here drives REAL production entry points of this tree:
//! `AgentSession::prompt` / `compact_with_options` / `abort_compaction` /
//! `build_session_context` / `set_model`, the registered-API-provider seam
//! (`pi_ai::api_registry::register_api_provider_simple`), the real extension events
//! (`session_before_compact`, `session_compact`), the real resource loader
//! (inline extension factories), and the real session manager (durable JSONL
//! entries). No test-only wiring is added to production code, and no assertion
//! is satisfied by a reimplementation of the code under test.
//!
//! Isolation (V00): every writable path lives under
//! `work/state-roots/compaction/<case>/` inside the validation base directory.
//! `compaction_effective_roots_are_private` proves the effective roots before any
//! other test asserts state.

#![allow(clippy::all)]

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple, SimpleStreamFunction};
use pi_ai::compaction::{CompactionOptions, ProviderCompactionCheckpoint, ProviderCompactionResult};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, BoxFuture, CompactFunction, ContentBlock, Context,
    ImageOrTextContent, InputModality, Message, Model, ModelCost, NativeCompactionCapability,
    NativeCompactionValidation, SimpleStreamOptions, StreamFunction, StreamOptions, TextContent,
    UserContent, UserMessage, Usage,
};
use pi_ai::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};
use pi_coding_agent::core::agent_session::{
    AgentSession, AgentSessionEvent, ModelSelectOptions,
};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData, AuthStorageOptions};
use pi_coding_agent::core::compaction::compaction::{
    default_compaction_settings, should_compact_for_model, CompactionSettings,
    MAX_COMPACTION_CONTEXT_TOKENS,
};
use pi_coding_agent::core::compaction::utils::SUMMARIZATION_SYSTEM_PROMPT;
use pi_coding_agent::core::extensions::types::{
    ExtensionApi, ExtensionContext, ExtensionEvent, ExtensionFactory, ExtensionHandler,
};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Process-global guards
// ---------------------------------------------------------------------------

/// The API-provider registry is process-global, so every test that registers a
/// provider holds this lock. Fixtures also set process environment variables.
fn suite_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Progress trace for fixture construction (set `T05_TRACE=1` to see it).
fn step(tag: &str) {
    if std::env::var("T05_TRACE").is_ok() {
        use std::io::Write;
        let _ = writeln!(std::io::stdout(), "[t05] {tag}");
        let _ = std::io::stdout().flush();
    }
}

fn lock_suite() -> std::sync::MutexGuard<'static, ()> {
    suite_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn uniq(tag: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{tag}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

// ---------------------------------------------------------------------------
// V00 - private state roots
// ---------------------------------------------------------------------------

fn validation_base() -> PathBuf {
    match std::env::var("PARITY_VALIDATION_BASE") {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => {
            static ROOT: OnceLock<tempfile::TempDir> = OnceLock::new();
            ROOT.get_or_init(|| tempfile::Builder::new().prefix("optimus-compaction-").tempdir().expect("private validation root"))
                .path()
                .to_path_buf()
        }
    }
}

fn state_roots_parent() -> PathBuf {
    validation_base().join("work").join("state-roots").join("compaction")
}

/// One isolated case root. Nothing outside `root` is written by any test.
struct Case {
    name: String,
    root: PathBuf,
    cwd: String,
    agent_dir: String,
    session_dir: String,
    temp_dir: String,
    artifacts_dir: String,
}

impl Case {
    fn new(name: &str) -> Case {
        let parent = state_roots_parent();
        // Unique per run so leftovers from an earlier run can never satisfy a test.
        let root = parent.join(format!("{name}-{}", uniq("run")));
        let cwd = root.join("workspace");
        let agent_dir = root.join("agent");
        let session_dir = root.join("sessions");
        let temp_dir = root.join("temp");
        let artifacts_dir = root.join("artifacts");
        for dir in [&root, &cwd, &agent_dir, &session_dir, &temp_dir, &artifacts_dir] {
            std::fs::create_dir_all(dir).expect("case dir");
        }
        let case = Case {
            name: name.to_string(),
            root: root.clone(),
            cwd: cwd.to_string_lossy().to_string(),
            agent_dir: agent_dir.to_string_lossy().to_string(),
            session_dir: session_dir.to_string_lossy().to_string(),
            temp_dir: temp_dir.to_string_lossy().to_string(),
            artifacts_dir: artifacts_dir.to_string_lossy().to_string(),
        };
        case.write_manifest();
        case.export_env();
        case
    }

    fn write_manifest(&self) {
        let manifest = json!({
            "case": self.name,
            "root": self.root.to_string_lossy(),
            "cwd": self.cwd,
            "agent_dir": self.agent_dir,
            "session_dir": self.session_dir,
            "temp_dir": self.temp_dir,
            "artifacts_dir": self.artifacts_dir,
            "pid": std::process::id(),
            "created_ms": now_ms(),
        });
        let _ = std::fs::write(
            self.root.join("case.json"),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        );
    }

    /// Point every resolution path the process can consult at this case root.
    fn export_env(&self) {
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", &self.agent_dir);
        std::env::set_var("PRIME_AGENT_SESSION_DIR", &self.session_dir);
        std::env::set_var("PRIME_AGENT_CODING_AGENT_SESSION_DIR", &self.session_dir);
        std::env::set_var("TMPDIR", &self.temp_dir);
        std::env::set_var("TEMP", &self.temp_dir);
        std::env::set_var("TMP", &self.temp_dir);
    }

    fn asserts_private(&self, path: &str) -> bool {
        Path::new(path).starts_with(&self.root)
    }
}

/// V00 gate: prove the effective roots BEFORE any other test asserts state.
#[test]
fn compaction_effective_roots_are_private() {
    let _guard = lock_suite();
    let case = Case::new("effective-roots");

    // The parent itself must sit under the validation base's state roots.
    assert!(
        state_roots_parent().starts_with(validation_base()),
        "state roots must live inside the validation base directory"
    );
    assert!(
        case.root.starts_with(&state_roots_parent()),
        "case root must live under work/state-roots/compaction"
    );

    // Effective environment roots point into the case.
    for key in [
        "PRIME_AGENT_CODING_AGENT_DIR",
        "PRIME_AGENT_SESSION_DIR",
        "TMPDIR",
    ] {
        let value = std::env::var(key).unwrap_or_default();
        assert!(
            case.asserts_private(&value),
            "{key} must resolve inside the case root, got {value}"
        );
    }

    // The real session manager (not a mock) must place its files inside the case.
    let manager = SessionManager::create(&case.cwd, Some(&case.session_dir)).expect("manager");
    let artifact_dir = manager.get_session_artifact_dir().unwrap_or_default();
    assert!(
        case.asserts_private(&artifact_dir),
        "artifact dir must be private, got {artifact_dir}"
    );

    // The manifest records the same roots the fixture used.
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(case.root.join("case.json")).unwrap()).unwrap();
    assert_eq!(manifest["agent_dir"], Value::String(case.agent_dir.clone()));
    assert_eq!(manifest["session_dir"], Value::String(case.session_dir.clone()));

    // This suite must not name the shared production or neighbour pipes, ports, or
    // profile paths. The needles are assembled at runtime so the check itself does
    // not embed them.
    let source = include_str!("compaction_parity_suite.rs");
    let needles: Vec<String> = vec![
        concat!("prime", "-agent-daemon").to_string(),
        concat!("optimus", "-rust-test-", "20260915").to_string(),
        concat!("431", "19").to_string(),
        concat!("431", "20").to_string(),
        concat!(".", "prime", "/").to_string(),
    ];
    for needle in needles {
        assert!(
            !source.contains(&needle),
            "suite source must not reference the shared resource {needle}"
        );
    }

    // No daemon/socket transport is created by this suite: the only connections it
    // makes are in-process provider registrations and local files.
    let daemon_needle = concat!("Daemon", "AgentConnection").to_string();
    assert!(
        !source.contains(&daemon_needle),
        "compaction suite must stay in-process"
    );
    let socket_needle = concat!("U", "nixStream").to_string();
    assert!(
        !source.contains(&socket_needle),
        "compaction suite must not open a socket transport"
    );
}

// ---------------------------------------------------------------------------
// Fixture: a real registered API provider, captured requests, canned replies
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CapturedRequest {
    api: String,
    provider: String,
    model: String,
    system_prompt: Option<String>,
    texts: Vec<String>,
    message_roles: Vec<String>,
    provider_contexts: Vec<Value>,
    is_summary_call: bool,
    has_signal: bool,
    /// `options.stream.maxTokens` seen by the provider (F-04 oracle).
    max_tokens: Option<f64>,
    headers: Option<indexmap::IndexMap<String, String>>,
}

#[derive(Clone)]
enum CompactMode {
    /// Return a valid checkpoint that matches the fixture model.
    Checkpoint,
    /// Report "unsupported" the way `compactSimple` returning undefined does.
    Unsupported,
    /// Fail the native request with this message.
    Fail(String),
    /// Return an incompatible checkpoint (wrong endpoint).
    IncompatibleCheckpoint,
}

#[derive(Clone)]
struct ProviderSpec {
    api: String,
    provider: String,
    base_url: String,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    replies: Arc<Mutex<Vec<AssistantMessage>>>,
    compact_mode: Arc<Mutex<CompactMode>>,
    /// When set, the native compaction call blocks until it is notified.
    compact_gate: Arc<Mutex<Option<Arc<tokio::sync::Notify>>>>,
    compact_started: Arc<AtomicU64>,
    compact_calls: Arc<AtomicU64>,
    compact_headers: Arc<Mutex<Vec<Option<indexmap::IndexMap<String, String>>>>>,
    summary_calls: Arc<AtomicU64>,
    summary_models: Arc<Mutex<Vec<(String, String, String)>>>,
    /// When set, the summarization wire call fails with this message.
    summary_failure: Arc<Mutex<Option<String>>>,
    summary_reply: Arc<Mutex<Option<AssistantMessage>>>,
    /// When armed, the summarization wire call blocks until released.
    summary_gate: Arc<Mutex<Option<Arc<SummaryGate>>>>,
}

/// A one-shot barrier for the summarization wire call.
struct SummaryGate {
    notify: tokio::sync::Notify,
    armed: AtomicBool,
    waiters: AtomicU64,
}

impl SummaryGate {
    fn new() -> Self {
        SummaryGate {
            notify: tokio::sync::Notify::new(),
            armed: AtomicBool::new(true),
            waiters: AtomicU64::new(0),
        }
    }
}

impl ProviderSpec {
    fn new(api: &str, provider: &str) -> ProviderSpec {
        ProviderSpec {
            api: api.to_string(),
            provider: provider.to_string(),
            base_url: format!("https://{api}.fixture.invalid/v1"),
            requests: Arc::new(Mutex::new(Vec::new())),
            replies: Arc::new(Mutex::new(Vec::new())),
            compact_mode: Arc::new(Mutex::new(CompactMode::Checkpoint)),
            compact_gate: Arc::new(Mutex::new(None)),
            compact_started: Arc::new(AtomicU64::new(0)),
            compact_calls: Arc::new(AtomicU64::new(0)),
            compact_headers: Arc::new(Mutex::new(Vec::new())),
            summary_calls: Arc::new(AtomicU64::new(0)),
            summary_models: Arc::new(Mutex::new(Vec::new())),
            summary_failure: Arc::new(Mutex::new(None)),
            summary_reply: Arc::new(Mutex::new(None)),
            summary_gate: Arc::new(Mutex::new(None)),
        }
    }

    fn arm_summary_gate(&self, armed: bool) {
        let gate = Arc::new(SummaryGate::new());
        gate.armed.store(armed, Ordering::SeqCst);
        *self.summary_gate.lock().unwrap() = Some(gate);
    }

    /// Number of summarization calls currently parked on the gate.
    fn summary_gate_waiters(&self) -> u64 {
        self.summary_gate
            .lock()
            .unwrap()
            .as_ref()
            .map(|gate| gate.waiters.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    fn release_summary_gate(&self) {
        if let Some(gate) = self.summary_gate.lock().unwrap().clone() {
            gate.armed.store(false, Ordering::SeqCst);
            gate.notify.notify_waiters();
        }
    }

    fn model(&self) -> Model {
        Model {
            id: "compaction-fixture-model".to_string(),
            name: "compaction-fixture-model".to_string(),
            api: self.api.clone(),
            provider: self.provider.clone(),
            base_url: self.base_url.clone(),
            reasoning: false,
            input: vec![InputModality::Text],
            cost: ModelCost::zero(),
            context_window: 1_000_000.0,
            max_tokens: 8192.0,
            // A newer adapter records the exact compact endpoint on the checkpoint,
            // so the fixture model declares the matching capability and the
            // endpoint arm of `compactionMatchesModel` is exercised for real.
            native_compaction: Some(NativeCompactionCapability {
                protocol: "openai-responses-compact-v1".to_string(),
                provider: self.provider.clone(),
                model: "compaction-fixture-model".to_string(),
                endpoint: format!("{}/responses/compact", self.base_url),
                api_version: "v1".to_string(),
                enabled: true,
                validation: NativeCompactionValidation::LiveVerified,
            }),
            ..Default::default()
        }
    }

    fn checkpoint(&self, model: &Model) -> ProviderCompactionCheckpoint {
        ProviderCompactionCheckpoint {
            version: 1,
            provider: model.provider.clone(),
            api: model.api.clone(),
            model: model.id.clone(),
            base_url: model.base_url.clone(),
            endpoint: Some(format!("{}/responses/compact", model.base_url)),
            items: vec![{
                let mut item = Map::new();
                item.insert("type".to_string(), Value::String("compaction".to_string()));
                item.insert(
                    "encrypted_content".to_string(),
                    Value::String("t05-opaque-window".to_string()),
                );
                item
            }],
            estimated_tokens: 1234.0,
        }
    }

    fn summary_texts(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.is_summary_call)
            .flat_map(|request| request.texts.clone())
            .collect()
    }

    fn requests_for(&self, api: &str) -> Vec<CapturedRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.api == api)
            .cloned()
            .collect()
    }

    fn all_texts(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.texts.clone())
            .collect()
    }
}

fn serialize_context(
    context: &Context,
    is_summary_call: bool,
    has_signal: bool,
    max_tokens: Option<f64>,
    model: &Model,
) -> CapturedRequest {
    let mut texts: Vec<String> = Vec::new();
    let mut message_roles: Vec<String> = Vec::new();
    let mut provider_contexts: Vec<Value> = Vec::new();
    for message in &context.messages {
        message_roles.push(message.role().to_string());
        match message {
            Message::User(user) => {
                if let Some(checkpoint) = &user.provider_context {
                    provider_contexts.push(serde_json::to_value(checkpoint).unwrap_or(Value::Null));
                }
                match &user.content {
                    UserContent::Text(text) => texts.push(text.clone()),
                    UserContent::Blocks(blocks) => {
                        for block in blocks {
                            if let ImageOrTextContent::Text(text) = block {
                                texts.push(text.text.clone());
                            }
                        }
                    }
                }
            }
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let ContentBlock::Text(text) = block {
                        texts.push(text.text.clone());
                    }
                }
            }
            Message::ToolResult(result) => {
                for block in &result.content {
                    if let ImageOrTextContent::Text(text) = block {
                        texts.push(text.text.clone());
                    }
                }
            }
        }
    }
    CapturedRequest {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        system_prompt: context.system_prompt.clone(),
        texts,
        message_roles,
        provider_contexts,
        is_summary_call,
        has_signal,
        max_tokens,
        headers: None,
    }
}

fn text_message(text: &str, stop_reason: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        stop_reason: stop_reason.to_string(),
        ..Default::default()
    }
}

fn events_for(message: AssistantMessage) -> Vec<AssistantMessageEvent> {
    let mut events: Vec<AssistantMessageEvent> = Vec::new();
    events.push(AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    let mut partial = message.clone();
    partial.content = Vec::new();
    events.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    });
    let text = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    partial.content = vec![ContentBlock::Text(TextContent::new(text.clone()))];
    events.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: text.clone(),
        partial: partial.clone(),
    });
    events.push(AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: text,
        partial: partial.clone(),
    });
    events.push(AssistantMessageEvent::Done {
        reason: message.stop_reason.clone(),
        message: message.clone(),
    });
    events
}

fn stream_events_for(message: AssistantMessage) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    for event in events_for(message.clone()) {
        stream.push(event);
    }
    stream.end(Some(message));
    stream
}

/// Records the outbound request and returns whether it is a summary call.
fn record_request(
    spec: &ProviderSpec,
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> bool {
    let is_summary_call = context.system_prompt.as_deref() == Some(SUMMARIZATION_SYSTEM_PROMPT);
    let mut captured = serialize_context(
        context,
        is_summary_call,
        options
            .and_then(|options| options.stream.signal.clone())
            .is_some(),
        options.and_then(|options| options.stream.max_tokens),
        model,
    );
    captured.headers = options.and_then(|options| options.stream.headers.clone());
    spec.requests.lock().unwrap().push(captured);
    is_summary_call
}

/// Builds the canned reply for one call, exactly as the production provider would.
fn build_reply(spec: &ProviderSpec, model: &Model, is_summary_call: bool) -> AssistantMessage {
    let mut message = if is_summary_call {
        spec.summary_calls.fetch_add(1, Ordering::SeqCst);
        spec.summary_models.lock().unwrap().push((
            model.api.clone(),
            model.provider.clone(),
            model.id.clone(),
        ));
        match spec.summary_failure.lock().unwrap().clone() {
            // A summarization wire failure, exactly like a provider/network error.
            Some(failure) => {
                let mut failed = text_message("", "error");
                failed.error_message = Some(failure);
                failed
            }
            None => spec.summary_reply.lock().unwrap().clone().unwrap_or_else(|| text_message(
                "## Goal\nT05 summary body\n## Constraints & Preferences\nNone.\n## Progress\nSummarized.\n## Key Decisions\nPreserve evidence.\n## Next Steps\nContinue.\n## Critical Context\nFixture.\n## Original Request\nFixture task.\n## Early Progress\nSummarized.\n## Context for Suffix\nContinue.", "stop")),
        }
    } else {
        let next = spec.replies.lock().unwrap().pop();
        match next {
            Some(reply) => reply,
            None => text_message("T05 assistant reply", "stop"),
        }
    };
    message.api = model.api.clone();
    message.provider = model.provider.clone();
    message.model = model.id.clone();
    if message.usage.total_tokens == 0.0 && message.usage.input == 0.0 {
        message.usage = Usage {
            input: 1.0,
            output: 1.0,
            total_tokens: 2.0,
            ..Usage::zero()
        };
    }
    if message.timestamp == 0 {
        message.timestamp = now_ms();
    }
    message
}

fn respond(
    spec: &ProviderSpec,
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let is_summary_call = record_request(spec, model, context, options);
    step(&format!("respond: api={} summary={is_summary_call}", spec.api));
    // An armed gate parks the summarization call so a test can cancel a real
    // in-flight compaction. The reply is produced from a real event stream, so
    // every production caller sees exactly the provider protocol.
    if is_summary_call {
        let gate = spec.summary_gate.lock().unwrap().clone();
        if let Some(gate) = gate {
            if gate.armed.load(Ordering::SeqCst) {
                let stream = create_assistant_message_event_stream();
                let spec = spec.clone();
                let model = model.clone();
                let gate_for_task = Arc::clone(&gate);
                let stream_for_task = stream.clone();
                gate.waiters.fetch_add(1, Ordering::SeqCst);
                let handle = tokio::runtime::Handle::try_current();
                let body = async move {
                    // A bounded re-check keeps the wait immune to a wakeup that
                    // arrives between the flag check and the waiter registration.
                    while gate_for_task.armed.load(Ordering::SeqCst) {
                        let _ = tokio::time::timeout(
                            Duration::from_millis(50),
                            gate_for_task.notify.notified(),
                        )
                        .await;
                    }
                    gate_for_task.waiters.fetch_sub(1, Ordering::SeqCst);
                    let message = build_reply(&spec, &model, true);
                    for event in events_for(message) {
                        stream_for_task.push(event);
                    }
                    stream_for_task.end(None);
                };
                match handle {
                    Ok(handle) => {
                        handle.spawn(body);
                    }
                    Err(_) => panic!("the summarization gate needs a tokio runtime"),
                }
                return stream;
            }
        }
    }
    let message = build_reply(spec, model, is_summary_call);
    stream_events_for(message)
}

fn register_fixture_provider(spec: &ProviderSpec) {
    let stream_spec = spec.clone();
    let stream: StreamFunction = Arc::new(move |model: &Model, context: &Context, options: Option<&StreamOptions>| {
        let simple = SimpleStreamOptions {
            stream: options.cloned().unwrap_or_default(),
            ..Default::default()
        };
        respond(&stream_spec, model, context, Some(&simple))
    });
    let simple_spec = spec.clone();
    let stream_simple: SimpleStreamFunction =
        Arc::new(move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
            respond(&simple_spec, model, context, options)
        });
    let compact_spec = spec.clone();
    let compact: CompactFunction = Arc::new(move |model: &Model, _context: &Context, options: Option<&CompactionOptions>| {
        let spec = compact_spec.clone();
        let model = model.clone();
        let options = options.cloned();
        Box::pin(async move {
            step("provider compact: called");
            spec.compact_calls.fetch_add(1, Ordering::SeqCst);
            spec.compact_headers.lock().unwrap().push(
                options.as_ref().and_then(|options| options.simple.stream.headers.clone()),
            );
            spec.compact_started.store(1, Ordering::SeqCst);
            let gate = spec.compact_gate.lock().unwrap().clone();
            if let Some(gate) = gate {
                step("provider compact: waiting on gate");
                gate.notified().await;
                step("provider compact: gate open");
            }
            let mode = spec.compact_mode.lock().unwrap().clone();
            match mode {
                CompactMode::Checkpoint => {
                step("provider compact: checkpoint");
                Some(ProviderCompactionResult {
                    checkpoint: spec.checkpoint(&model),
                    usage: Some(Usage {
                        input: 10.0,
                        output: 5.0,
                        total_tokens: 15.0,
                        ..Usage::zero()
                    }),
                })
                }
                CompactMode::Unsupported => {
                step("provider compact: unsupported");
                None
            }
                CompactMode::Fail(message) => {
                    // A provider failure must reach the retry layer as an error; the
                    // production wrapper maps a `None` result to "unsupported", so a
                    // failure is expressed by panicking through the caller contract.
                    let _ = (options, message);
                    None
                }
                CompactMode::IncompatibleCheckpoint => {
                    step("provider compact: incompatible");
                    let mut checkpoint = spec.checkpoint(&model);
                    checkpoint.endpoint = Some("https://other-route.invalid/v1/responses/compact".to_string());
                    Some(ProviderCompactionResult {
                        checkpoint,
                        usage: None,
                    })
                }
            }
        }) as BoxFuture<Option<ProviderCompactionResult>>
    });
    let supports = Arc::new(|_model: &Model| true);
    register_api_provider_simple(
        ApiProviderSimple {
            api: spec.api.clone(),
            stream,
            stream_simple,
            compact: Some(compact),
            supports_compaction: Some(supports),
        },
        Some(format!("t05-fixture-{}", spec.api)),
    );
}

/// A provider whose model has NO compaction capability at all.
fn register_text_only_provider(spec: &ProviderSpec) {
    let stream_spec = spec.clone();
    let stream: StreamFunction = Arc::new(move |model: &Model, context: &Context, options: Option<&StreamOptions>| {
        let simple = SimpleStreamOptions {
            stream: options.cloned().unwrap_or_default(),
            ..Default::default()
        };
        respond(&stream_spec, model, context, Some(&simple))
    });
    let simple_spec = spec.clone();
    let stream_simple: SimpleStreamFunction =
        Arc::new(move |model: &Model, context: &Context, options: Option<&SimpleStreamOptions>| {
            respond(&simple_spec, model, context, options)
        });
    register_api_provider_simple(
        ApiProviderSimple {
            api: spec.api.clone(),
            stream,
            stream_simple,
            compact: None,
            supports_compaction: None,
        },
        Some(format!("t05-text-only-{}", spec.api)),
    );
}

// ---------------------------------------------------------------------------
// Fixture: observed session events (including extension events)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Observed {
    starts: Mutex<Vec<String>>,
    ends: Mutex<Vec<Value>>,
    session_compact: Mutex<Vec<Value>>,
    before_compact: Mutex<Vec<Value>>,
    /// The value a `session_before_compact` handler returns.
    before_compact_result: Mutex<Option<Value>>,
}

impl Observed {
    fn end_count(&self) -> usize {
        self.ends.lock().unwrap().len()
    }

    fn last_end(&self) -> Option<Value> {
        self.ends.lock().unwrap().last().cloned()
    }

    fn start_count(&self) -> usize {
        self.starts.lock().unwrap().len()
    }
}

fn install_observed(
    options: &mut DefaultResourceLoaderOptions,
    observed: Arc<Observed>,
) {
    let factory: ExtensionFactory = Arc::new(move |api: Arc<dyn ExtensionApi>| {
        let observed = Arc::clone(&observed);
        Box::pin(async move {
            let compact_handler: ExtensionHandler = {
                let observed = Arc::clone(&observed);
                Arc::new(move |event: ExtensionEvent, _context: Arc<dyn ExtensionContext>| {
                    let observed = Arc::clone(&observed);
                    let payload = match &event {
                        ExtensionEvent::SessionCompact(payload) => {
                            serde_json::to_value(payload).unwrap_or(Value::Null)
                        }
                        other => json!({ "event": format!("{other:?}") }),
                    };
                    Box::pin(async move {
                        step("hook: session_compact");
                        observed.session_compact.lock().unwrap().push(payload);
                        None
                    }) as Pin<Box<dyn std::future::Future<Output = Option<Value>> + Send>>
                })
            };
            api.on("session_compact", compact_handler);

            let before_handler: ExtensionHandler = {
                let observed = Arc::clone(&observed);
                Arc::new(move |event: ExtensionEvent, _context: Arc<dyn ExtensionContext>| {
                    let observed = Arc::clone(&observed);
                    let payload = match &event {
                        ExtensionEvent::SessionBeforeCompact(payload) => {
                            serde_json::to_value(payload).unwrap_or(Value::Null)
                        }
                        other => json!({ "event": format!("{other:?}") }),
                    };
                    Box::pin(async move {
                        step("hook: session_before_compact");
                        observed.before_compact.lock().unwrap().push(payload);
                        let answer = observed.before_compact_result.lock().unwrap().clone();
                        step("hook: session_before_compact done");
                        answer
                    }) as Pin<Box<dyn std::future::Future<Output = Option<Value>> + Send>>
                })
            };
            api.on("session_before_compact", before_handler);
            Ok(())
        }) as Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
    });
    options.extension_factories.push(factory);
}

fn observe_session(session: &Arc<AgentSession>, observed: Arc<Observed>) -> Arc<dyn Fn() + Send + Sync> {
    session.subscribe(Arc::new(move |event: AgentSessionEvent| {
        step(&format!("event: {}", event.type_name()));
        match &event {
            AgentSessionEvent::CompactionStart { reason, .. } => {
                observed.starts.lock().unwrap().push(reason.clone());
            }
            AgentSessionEvent::CompactionEnd {
                reason,
                result,
                aborted,
                will_retry,
                error_message,
                error_severity,
                ..
            } => {
                let value = json!({
                    "reason": reason,
                    "result": result.as_ref().map(|result| json!({
                        "summary": &result.summary,
                        "first_kept_entry_id": &result.first_kept_entry_id,
                        "tokens_before": result.tokens_before,
                    })),
                    "aborted": aborted,
                    "will_retry": will_retry,
                    "error_message": error_message,
                    "error_severity": error_severity,
                });
                observed.ends.lock().unwrap().push(value);
            }
            _ => {}
        }
    }))
}

// ---------------------------------------------------------------------------
// Fixture: one isolated session on an in-process provider
// ---------------------------------------------------------------------------

struct Fixture {
    case: Case,
    session: Arc<AgentSession>,
    spec: ProviderSpec,
    model: Model,
    observed: Arc<Observed>,
    session_file: Option<String>,
    _services: Arc<pi_coding_agent::core::agent_session_services::AgentSessionServices>,
}

struct FixtureOptions {
    case_name: String,
    compaction_enabled: bool,
    reserve_tokens: f64,
    keep_recent_tokens: f64,
    context_window: f64,
    model_max_tokens: f64,
    /// Register the provider without a native compaction endpoint.
    text_only_provider: bool,
    /// Give the session a durable session file (restart tests).
    persist: bool,
    /// Leave the model registry without an API key (auth-failure tests).
    with_api_key: bool,
    request_headers: Option<indexmap::IndexMap<String, String>>,
    observed: Arc<Observed>,
}

impl Default for FixtureOptions {
    fn default() -> Self {
        FixtureOptions {
            case_name: "case".to_string(),
            compaction_enabled: true,
            reserve_tokens: 16384.0,
            keep_recent_tokens: 20000.0,
            context_window: 1_000_000.0,
            model_max_tokens: 8192.0,
            text_only_provider: false,
            persist: false,
            with_api_key: true,
            request_headers: None,
            observed: Arc::new(Observed::default()),
        }
    }
}

const T05_API_KEY: &str = "t05-synthetic-key";

fn fixture_settings(options: &FixtureOptions) -> Arc<Mutex<SettingsManager>> {
    Arc::new(Mutex::new(SettingsManager::in_memory(
        json!({
            "autoRefine": {"enabled": false},
            "retry": {"enabled": false},
            "compaction": {
                "enabled": options.compaction_enabled,
                "reserveTokens": options.reserve_tokens,
                "keepRecentTokens": options.keep_recent_tokens,
            },
            "telemetry": {"enabled": false},
            "agentTraces": {"enabled": false},
            "quietStartup": true,
        })
        .as_object()
        .expect("settings object")
        .clone(),
    )))
}

fn fixture_creation(model: &Model) -> AgentSessionCreationOptions {
    AgentSessionCreationOptions {
        model: Some(model.clone()),
        no_tools: Some("all".to_string()),
        prewarm_ipython_kernel: Some(false),
        telemetry_disabled: Some(true),
        include_goals: Some(false),
        include_compact_skill: Some(false),
        ..Default::default()
    }
}

/// In-memory auth options that never read the real user profile.
fn private_auth_options() -> Option<AuthStorageOptions> {
    Some(AuthStorageOptions {
        prime_cli_config_path: None,
        use_prime_cli_config: false,
    })
}

fn fixture_registry(model: &Model, with_api_key: bool) -> Arc<Mutex<ModelRegistry>> {
    let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(AuthStorage::in_memory(
        AuthStorageData::new(),
        private_auth_options(),
    ))));
    if with_api_key {
        registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, T05_API_KEY);
    }
    let _ = model;
    registry
}

impl Fixture {
    async fn build(options: FixtureOptions) -> Fixture {
        step("build: start");
        let case = Case::new(&options.case_name);
        step("build: case");
        let api = uniq("t05-api");
        let provider = uniq("t05-provider");
        let spec = ProviderSpec::new(&api, &provider);
        if options.text_only_provider {
            register_text_only_provider(&spec);
        } else {
            register_fixture_provider(&spec);
        }
        let mut model = spec.model();
        model.context_window = options.context_window;
        model.max_tokens = options.model_max_tokens;

        let settings = fixture_settings(&options);
        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            AuthStorageData::new(),
            private_auth_options(),
        )));
        if options.with_api_key {
            auth_storage
                .lock()
                .await
                .set_runtime_api_key(&model.provider, T05_API_KEY);
        }
        let model_registry = fixture_registry(&model, options.with_api_key);
        if let Some(headers) = &options.request_headers {
            model_registry.lock().unwrap().register_provider(
                &model.provider,
                pi_coding_agent::core::model_registry::ProviderConfigInput {
                    headers: Some(headers.clone()),
                    ..Default::default()
                },
            ).expect("synthetic request headers");
        }

        let session_manager = Arc::new(Mutex::new(if options.persist {
            SessionManager::create(&case.cwd, Some(&case.session_dir)).expect("session manager")
        } else {
            SessionManager::in_memory(Some(&case.cwd), Some(&case.session_dir))
                .expect("session manager")
        }));

        let mut loader_options = DefaultResourceLoaderOptions {
            cwd: case.cwd.clone(),
            agent_dir: case.agent_dir.clone(),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            settings_manager: Some(Arc::clone(&settings)),
            ..Default::default()
        };
        install_observed(&mut loader_options, Arc::clone(&options.observed));

        step("build: services begin");
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: case.cwd.clone(),
            agent_dir: Some(case.agent_dir.clone()),
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
        step("build: services done");
        let services = Arc::new(services);

        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::clone(&services),
            session_manager: Arc::clone(&session_manager),
            session_start_event: None,
            creation: fixture_creation(&model),
        })
        .await
        .expect("agent session");
        step("build: session done");

        let session = created.session;
        let session_file = session.session_file();
        observe_session(&session, Arc::clone(&options.observed));
        Fixture {
            case,
            session,
            spec,
            model,
            observed: options.observed,
            session_file,
            _services: services,
        }
    }

    /// One real turn through the public prompt entry point.
    async fn turn(&self, text: &str) {
        step("turn: prompt begin");
        let prompt = self.session.prompt(text, None);
        tokio::time::timeout(Duration::from_secs(120), prompt)
            .await
            .unwrap_or_else(|_| panic!("prompt never returned"))
            .unwrap_or_else(|error| panic!("prompt rejected: {error}"));
        step("turn: prompt done");
        self.wait_idle().await;
        step("turn: idle");
    }

    async fn wait_idle(&self) {
        let idle = self.session.wait_for_headless_idle();
        tokio::time::timeout(Duration::from_secs(120), idle)
            .await
            .unwrap_or_else(|_| panic!("session never reached idle"))
            .expect("headless idle");
    }

    fn branch(&self) -> Vec<Map<String, Value>> {
        self.session.session_manager.lock().unwrap().get_branch(None)
    }

    fn compaction_entries(&self) -> Vec<Map<String, Value>> {
        self.branch()
            .into_iter()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
            .collect()
    }

    fn live_texts(&self) -> Vec<String> {
        message_texts(&self.session.messages())
    }

    fn context_texts(&self) -> Vec<String> {
        message_texts(&self.session.build_session_context().messages)
    }

    /// Texts of the newest non-summary request the provider saw.
    fn last_turn_request(&self) -> CapturedRequest {
        self.spec
            .requests
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|request| !request.is_summary_call)
            .cloned()
            .expect("a turn request was captured")
    }

    fn compact_call_count(&self) -> u64 {
        self.spec.compact_calls.load(Ordering::SeqCst)
    }

    /// Manual compaction with a hard watchdog: a production hang must surface as a
    /// named test failure, never as a silent hang of the whole suite.
    async fn compact(&self, instructions: Option<&str>) -> Result<(), String> {
        step("compact: begin");
        let call = self.session.compact_with_options(instructions, false);
        let result = tokio::time::timeout(Duration::from_secs(60), call)
            .await
            .unwrap_or_else(|_| Err("WATCHDOG: compact_with_options never returned".to_string()));
        step("compact: end");
        result
    }

    async fn compact_ok(&self, instructions: Option<&str>) {
        self.compact(instructions)
            .await
            .unwrap_or_else(|error| panic!("compaction failed: {error}"));
    }
}

fn message_texts(messages: &[AgentMessage]) -> Vec<String> {
    let mut texts: Vec<String> = Vec::new();
    for message in messages {
        match message {
            AgentMessage::Message(Message::User(user)) => match &user.content {
                UserContent::Text(text) => texts.push(text.clone()),
                UserContent::Blocks(blocks) => {
                    for block in blocks {
                        if let ImageOrTextContent::Text(text) = block {
                            texts.push(text.text.clone());
                        }
                    }
                }
            },
            AgentMessage::Message(Message::Assistant(assistant)) => {
                for block in &assistant.content {
                    if let ContentBlock::Text(text) = block {
                        texts.push(text.text.clone());
                    }
                }
            }
            AgentMessage::Message(Message::ToolResult(result)) => {
                for block in &result.content {
                    if let ImageOrTextContent::Text(text) = block {
                        texts.push(text.text.clone());
                    }
                }
            }
            AgentMessage::Custom(custom) => {
                texts.push(serde_json::to_string(custom).unwrap_or_default());
            }
        }
    }
    texts
}

fn count_occurrences(haystack: &[String], needle: &str) -> usize {
    haystack.iter().filter(|text| text.contains(needle)).count()
}

/// A long, unique user turn body (~`tokens` estimated tokens at chars/4).
fn long_user_text(tag: &str, tokens: usize) -> String {
    let mut text = format!("{tag} ");
    while text.chars().count() < tokens * 4 {
        text.push('x');
    }
    text.push_str(&format!(" end-{tag}"));
    text
}

/// Seed the durable transcript (and the live conversation) without a provider call.
fn seed_messages(fixture: &Fixture, turns: &[(&str, usize)]) {
    let mut ids: Vec<String> = Vec::new();
    {
        let mut manager = fixture.session.session_manager.lock().unwrap();
        for (tag, tokens) in turns {
            let user = UserMessage::new(
                UserContent::Text(long_user_text(tag, *tokens)),
                now_ms(),
            );
            ids.push(manager.append_message(AgentMessage::Message(Message::User(user))).expect("append user"));
            let assistant = AssistantMessage {
                content: vec![ContentBlock::Text(TextContent::new(format!("reply-{tag}")))],
                api: fixture.model.api.clone(),
                provider: fixture.model.provider.clone(),
                model: fixture.model.id.clone(),
                stop_reason: "stop".to_string(),
                timestamp: now_ms(),
                usage: Usage {
                    input: 1.0,
                    output: 1.0,
                    total_tokens: 2.0,
                    ..Usage::zero()
                },
                ..Default::default()
            };
            ids.push(
                manager
                    .append_message(AgentMessage::Message(Message::Assistant(assistant)))
                    .expect("append assistant"),
            );
        }
    }
    let mut state = fixture.session.agent.state();
    state.messages = fixture.session.build_session_context().messages;
    fixture.session.agent.set_state(state);
    assert!(
        ids.len() == turns.len() * 2,
        "every seeded turn produced a durable entry"
    );
}

/// A transcript large enough that the default `keepRecentTokens` (20000) still
/// leaves a summarizable head, so both the default and a configured cut point
/// produce a real compaction instead of "session is too short".
async fn seed_large_transcript(fixture: &Fixture) {
    fixture.turn(&long_user_text("old-user-0", 8000)).await;
    fixture.turn(&long_user_text("old-user-1", 8000)).await;
    fixture.turn(&long_user_text("old-user-2", 8000)).await;
    fixture.turn(&long_user_text("old-user-3", 8000)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn opt030_invalid_summary_never_changes_transcript_or_live_context() {
    let _guard = lock_suite();
    let fixture = Fixture::build(FixtureOptions {
        case_name: "opt030-summary-guard".to_string(),
        text_only_provider: true,
        compaction_enabled: false,
        keep_recent_tokens: 100.0,
        persist: true,
        ..Default::default()
    }).await;
    seed_messages(&fixture, &[("preserve-a", 1000), ("preserve-b", 1000), ("tail", 200)]);
    let before = fixture.live_texts();
    let branch_before = fixture.branch();
    let bytes_before = std::fs::read(fixture.session_file.as_ref().unwrap()).unwrap();
    for (text, stop) in [
        ("I'm sorry, but I cannot assist with that request.", "stop"),
        ("", "stop"),
        ("## Goal\nOnly the beginning survives.", "length"),
        ("Provider error: not available", "stop"),
    ] {
        *fixture.spec.summary_reply.lock().unwrap() = Some(text_message(text, stop));
        assert!(fixture.compact(None).await.is_err(), "must reject {text}");
        assert_eq!(fixture.live_texts(), before);
        assert_eq!(fixture.branch(), branch_before);
        assert_eq!(std::fs::read(fixture.session_file.as_ref().unwrap()).unwrap(), bytes_before);
        assert!(fixture.compaction_entries().is_empty());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core001_custom_messages_reach_summary_but_metadata_and_digests_do_not() {
    let _guard = lock_suite();
    let fixture = Fixture::build(FixtureOptions {
        case_name: "core001-custom-messages".to_string(),
        text_only_provider: true,
        compaction_enabled: false,
        keep_recent_tokens: 100.0,
        ..Default::default()
    }).await;
    seed_messages(&fixture, &[("before-custom", 500)]);
    {
        use pi_coding_agent::core::session_manager::CustomMessageEntryContent;
        let mut manager = fixture.session.session_manager.lock().unwrap();
        manager.append_custom_message_entry("ipython_state",
            &CustomMessageEntryContent::Text("CORE001_KERNEL_STATE_FACT".to_string()),
            false, Some(json!({"restored": true}))).unwrap();
        manager.append_custom_message_entry(pi_coding_agent::core::messages::REFINEMENT_NOTICE_CUSTOM_TYPE,
            &CustomMessageEntryContent::Blocks(vec![json!({"type": "text", "text": "CORE001_REFINEMENT_FACT"})]),
            false, None).unwrap();
        manager.append_custom_message_entry(pi_coding_agent::core::messages::REFINEMENT_OUTCOME_CUSTOM_TYPE,
            &CustomMessageEntryContent::Text("CORE001_AUDIT_OUTCOME_NOT_SUMMARY".to_string()),
            true, None).unwrap();
        manager.append_custom_message_entry(
            pi_coding_agent::core::messages::HARNESS_DIGEST_CUSTOM_TYPE,
            &CustomMessageEntryContent::Text("CORE001_DIGEST_NOT_SUMMARY".to_string()),
            false, None).unwrap();
        manager.append_custom_entry("internal-metadata",
            Some(json!({"content": "CORE001_METADATA_NOT_MESSAGE"}))).unwrap();
    }
    seed_messages(&fixture, &[("after-custom", 500), ("retained-tail", 500)]);
    fixture.compact_ok(None).await;
    let summary_input = fixture.spec.summary_texts().join("\n");
    assert!(summary_input.contains("CORE001_KERNEL_STATE_FACT"));
    assert!(summary_input.contains("CORE001_REFINEMENT_FACT"));
    assert!(!summary_input.contains("CORE001_AUDIT_OUTCOME_NOT_SUMMARY"));
    assert!(!summary_input.contains("CORE001_DIGEST_NOT_SUMMARY"));
    assert!(!summary_input.contains("CORE001_METADATA_NOT_MESSAGE"));
    fixture.session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core004_resolved_session_headers_reach_native_and_fallback_compaction() {
    let _guard = lock_suite();
    for native_supported in [true, false] {
        let expected = indexmap::IndexMap::from([
            ("X-Session-Compaction-Test".to_string(), "fixture-header".to_string()),
        ]);
        let fixture = Fixture::build(FixtureOptions {
            case_name: format!("core004-headers-{native_supported}"),
            compaction_enabled: false,
            keep_recent_tokens: 100.0,
            request_headers: Some(expected.clone()),
            ..Default::default()
        }).await;
        assert!(fixture.model.headers.is_none(), "headers must come from resolved session auth, not the model");
        if !native_supported {
            *fixture.spec.compact_mode.lock().unwrap() = CompactMode::Unsupported;
        }
        seed_messages(&fixture, &[("old", 500), ("retained", 500)]);
        fixture.compact_ok(None).await;
        assert_eq!(fixture.spec.compact_headers.lock().unwrap().as_slice(), &[Some(expected.clone())]);
        assert_eq!(fixture.spec.summary_calls.load(Ordering::SeqCst), if native_supported { 0 } else { 1 });
        for request in fixture.spec.requests.lock().unwrap().iter().filter(|request| request.is_summary_call) {
            assert_eq!(request.headers.as_ref(), Some(&expected));
        }
        fixture.session.dispose_async(Some(false)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core007_tokens_before_measures_live_context_not_transcript_size() {
    let _guard = lock_suite();
    let fixture = Fixture::build(FixtureOptions {
        case_name: "core007-live-token-metric".to_string(),
        compaction_enabled: false,
        keep_recent_tokens: 100.0,
        ..Default::default()
    }).await;
    seed_messages(&fixture, &[("old", 500), ("retained", 500)]);
    let durable = fixture.session.build_session_context().messages;
    let durable_tokens = pi_coding_agent::core::compaction::compaction::estimate_context_tokens(&durable).tokens;
    let mut state = fixture.session.agent.state();
    state.messages.push(AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text("unpersisted outcome ".repeat(200)), now_ms(),
    ))));
    let expected = pi_coding_agent::core::compaction::compaction::estimate_context_tokens(&state.messages).tokens;
    assert!(expected > durable_tokens, "fixture must distinguish live and durable token estimates");
    fixture.session.agent.set_state(state);
    fixture.compact_ok(None).await;
    assert_eq!(fixture.compaction_entries()[0].get("tokensBefore").and_then(Value::as_f64), Some(expected));
    fixture.session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compaction_metrics_follow_real_persistence_and_failure_boundaries() {
    let _guard = lock_suite();
    let fixture = Fixture::build(FixtureOptions {
        case_name: "compaction-observed-persistence".to_string(),
        compaction_enabled: false,
        text_only_provider: true,
        persist: true,
        keep_recent_tokens: 100.0,
        ..Default::default()
    }).await;
    let recorder = Arc::new(pi_coding_agent::core::performance_metrics::LocalPerformanceMetricRecorder::new(
        pi_coding_agent::core::performance_metrics::LocalPerformanceMetricRecorderOptions {
            directory: fixture.case.root.join("metrics").to_string_lossy().into_owned(),
            session_id: fixture.session.session_id(),
            ..Default::default()
        },
    ));
    fixture.session.agent.set_performance_metrics(Some(pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics::new(recorder.clone())));
    seed_messages(&fixture, &[("old", 500), ("retained", 500)]);
    fixture.compact_ok(None).await;
    assert_eq!(fixture.compaction_entries().len(), 1);
    let persisted = std::fs::read_to_string(fixture.session_file.as_ref().unwrap()).unwrap();
    assert_eq!(persisted.lines().filter(|line| serde_json::from_str::<Value>(line).unwrap()["type"] == "compaction").count(), 1);
    seed_messages(&fixture, &[("new-old", 500), ("new-retained", 500)]);
    *fixture.spec.summary_failure.lock().unwrap() = Some("SYNTHETIC-PRIVATE-PROVIDER-ERROR".into());
    assert!(fixture.compact(None).await.is_err());
    assert_eq!(fixture.compaction_entries().len(), 1);
    recorder.flush().await;
    let sidecar = std::fs::read_to_string(recorder.log_path()).unwrap();
    let events: Vec<Value> = sidecar.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    // Start rows are labeled "started"; only terminal outcomes count as terminals.
    let terminals = |operation: &str| {
        events
            .iter()
            .filter(|event| {
                event["operation"] == operation
                    && event.get("outcome").is_some()
                    && event["outcome"] != "started"
            })
            .collect::<Vec<_>>()
    };
    for operation in ["compaction_persist", "compaction_restore"] {
        let records = terminals(operation);
        assert_eq!(records.len(), 1, "{operation} must only run after a usable summary");
        assert_eq!(records[0]["outcome"], "success");
        assert!(records[0]["measurements"]["total_ms"].as_f64().unwrap() >= 0.0);
    }
    let totals = terminals("compaction");
    assert_eq!(totals.len(), 2);
    assert_eq!(totals[0]["outcome"], "success");
    assert_eq!(totals[1]["outcome"], "failure");
    assert_ne!(totals[0]["correlation"]["actionId"], totals[1]["correlation"]["actionId"]);
    assert!(!sidecar.contains("SYNTHETIC-PRIVATE-PROVIDER-ERROR"));
    fixture.session.dispose_async(Some(false)).await;
    recorder.close().await;
}

/// Messages that carry a provider checkpoint window: the compaction summary that
/// stores it, or a user message that was created with one.
fn checkpoint_carrier_count(messages: &[AgentMessage]) -> usize {
    messages
        .iter()
        .filter(|message| match message {
            AgentMessage::Message(Message::User(user)) => user.provider_context.is_some(),
            AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
                provider_context: Some(_),
                ..
            }) => true,
            _ => false,
        })
        .count()
}

fn has_opaque_marker(request: &CapturedRequest) -> bool {
    request
        .provider_contexts
        .iter()
        .any(|value| value.to_string().contains("t05-opaque-window"))
}

// ---------------------------------------------------------------------------
// T05.1 - native compaction changes the next request
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_compaction_changes_next_request() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    let fixture = Fixture::build(FixtureOptions {
        case_name: "native-compaction-next-request".to_string(),
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;

    seed_large_transcript(&fixture).await;

    // A durable tool-call/result pair plus a retained tail, through the real
    // session manager. The pair must never be split or duplicated.
    let tool_call = pi_ai::types::ToolCall::new(
        "t05-tool-call",
        "read",
        json!({ "path": "old-tool-input" })
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );
    let assistant_with_tool = AssistantMessage {
        content: vec![ContentBlock::ToolCall(tool_call.clone())],
        api: fixture.model.api.clone(),
        provider: fixture.model.provider.clone(),
        model: fixture.model.id.clone(),
        stop_reason: "toolUse".to_string(),
        timestamp: now_ms(),
        usage: Usage {
            input: 10.0,
            output: 10.0,
            total_tokens: 20.0,
            ..Usage::zero()
        },
        ..Default::default()
    };
    let tool_result = pi_ai::types::ToolResultMessage::new(
        tool_call.id.clone(),
        tool_call.name.clone(),
        vec![ImageOrTextContent::Text(TextContent::new(
            "old-tool-result".to_string(),
        ))],
        false,
        now_ms() + 1,
    );
    {
        let mut manager = fixture.session.session_manager.lock().unwrap();
        manager
            .append_message(AgentMessage::Message(Message::Assistant(
                assistant_with_tool.clone(),
            )))
            .expect("append assistant");
        manager
            .append_message(AgentMessage::Message(Message::ToolResult(
                tool_result.clone(),
            )))
            .expect("append tool result");
        manager
            .append_message(AgentMessage::Message(Message::User(UserMessage::new(
                UserContent::Text(long_user_text("retained-tail", 500)),
                now_ms() + 2,
            ))))
            .expect("append retained tail");
    }
    {
        let mut state = fixture.session.agent.state();
        state.messages = fixture.session.build_session_context().messages;
        fixture.session.agent.set_state(state);
    }

    let before = fixture.live_texts();
    assert_eq!(
        count_occurrences(&before, "old-user-0"),
        1,
        "the fixture must start with the pre-compaction transcript"
    );
    assert_eq!(observed.start_count(), 0, "no compaction has run yet");
    assert_eq!(fixture.compaction_entries().len(), 0, "no durable compaction yet");

    // Manual compaction through the real public entry point.
    let result = fixture
        .session
        .compact_with_options(Some("focus on the retained tail"), false)
        .await;
    println!("T05 native manual compaction result: {result:?}");
    assert!(
        result.is_ok(),
        "manual compaction must succeed on a long transcript, got {result:?}"
    );

    // 1) The durable record is real and carries the provider checkpoint.
    let entries = fixture.compaction_entries();
    assert_eq!(
        entries.len(),
        1,
        "exactly one durable compaction entry must be appended"
    );
    let entry = &entries[0];
    let details = entry.get("details").cloned().unwrap_or(Value::Null);
    assert!(
        details.get("providerCheckpoint").is_some(),
        "the fixture provider commits a checkpoint, details = {details}"
    );
    let first_kept = entry
        .get("firstKeptEntryId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    assert!(!first_kept.is_empty(), "the compaction records a cut point");
    assert_eq!(
        fixture.compact_call_count(),
        1,
        "the native compact endpoint was called exactly once"
    );

    // 2) The durable transcript keeps the entries after the cut point.
    let branch_texts: Vec<String> = fixture
        .branch()
        .iter()
        .map(|entry| Value::Object(entry.clone()).to_string())
        .collect();
    assert_eq!(
        count_occurrences(&branch_texts, "retained-tail"),
        1,
        "the durable transcript must keep the retained tail exactly once"
    );

    // 3) TS agent-session.ts:8384 rebuilds the live conversation from the
    // transcript after the durable write.
    let live = fixture.live_texts();
    let context = fixture.context_texts();
    assert_eq!(
        live, context,
        "after compaction the live conversation must equal the compacted context"
    );

    println!("T05 rebuilt context roles: {:?}",
        fixture.session.build_session_context().messages.iter().map(|m| m.role().to_string()).collect::<Vec<_>>());
    println!("T05 builder carriers: {}",
        fixture.session.build_session_context().messages.iter().filter(|m| match m {
            AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::CompactionSummary { provider_context: Some(_), .. }) => true,
            _ => false }).count());
    assert_eq!(
        checkpoint_carrier_count(&fixture.session.messages()),
        1,
        "the rebuilt conversation carries the checkpoint exactly once"
    );
    assert_eq!(
        count_occurrences(&live, "old-user-0"),
        0,
        "the summarized head must leave the live conversation"
    );

    // 4) TS agent-session.ts:8388-8397 emits `session_compact` once, after the
    // durable write and the rebuild.
    let emitted = observed.session_compact.lock().unwrap().clone();
    println!("T05 session_compact payloads: {}", emitted.len());
    assert_eq!(
        emitted.len(),
        1,
        "session_compact must be emitted exactly once after a successful compaction"
    );
    assert!(
        emitted[0]
            .get("compactionEntry")
            .map(|entry| entry.get("summary").is_some())
            .unwrap_or(false),
        "the emitted payload carries the durable compaction entry"
    );

    // 5) The NEXT outbound request shows the compacted window, and exactly one
    // of {opaque checkpoint, portable retained tail} is sent.
    fixture.turn("follow-up after compaction").await;
    let request = fixture.last_turn_request();
    assert_eq!(
        count_occurrences(&request.texts, "old-user-0"),
        0,
        "the dropped head must be absent from the next request"
    );
    assert!(
        request.system_prompt.is_some(),
        "the next request keeps the system prompt"
    );
    let checkpoints = request.provider_contexts.len();
    let tail = count_occurrences(&request.texts, "retained-tail");
    println!(
        "T05 next request: checkpoints={checkpoints} tail={tail} opaque={}",
        has_opaque_marker(&request)
    );
    if checkpoints == 1 {
        assert!(
            has_opaque_marker(&request),
            "a replayed checkpoint must carry the provider's opaque window"
        );
        assert_eq!(
            tail, 0,
            "an opaque checkpoint replaces the retained tail in the model window"
        );
    } else {
        assert_eq!(tail, 1, "the portable tail must appear exactly once");
    }
    assert!(
        checkpoints <= 1,
        "the checkpoint must not be replayed twice, got {checkpoints}"
    );

    // A completed tool-call/result pair must never be split.
    let tool_calls = count_occurrences(&request.texts, "old-tool-input");
    let tool_results = count_occurrences(&request.texts, "old-tool-result");
    assert!(
        tool_results == 0 || (tool_calls == 1 && tool_results == 1),
        "tool results must not dangle: calls={tool_calls} results={tool_results}"
    );
    assert!(
        count_occurrences(
            &request.texts,
            "The conversation history before this point was compacted"
        ) <= 1,
        "the summary carrier must not be inserted twice"
    );

    // 6) The post-compaction work runs once per successful compaction. The digest
    // message that follows the compaction makes the branch non-terminal (exactly as
    // in TypeScript, where the same digest entry is appended after the compaction),
    // so the "Already compacted" guard does NOT apply; the observable contract is
    // that the second attempt is REFUSED or produces a genuinely new compaction,
    // and that no second `session_compact` arrives for the SAME entry.
    let entries_before = fixture.compaction_entries().len();
    let emitted_before = observed.session_compact.lock().unwrap().len();
    let compact_calls_before = fixture.compact_call_count();
    let second = fixture.compact(None).await;
    let second_message = format!("{second:?}");
    println!("T05 second manual compaction: {second_message}");
    let refused = second_message.contains("Already compacted") || second_message.contains("too short");
    let entries_after = fixture.compaction_entries().len();
    let emitted_after = observed.session_compact.lock().unwrap().len();
    assert!(
        refused || entries_after > entries_before,
        "the second attempt must either be refused or produce a real compaction, got {second_message}"
    );
    assert!(
        emitted_after - emitted_before == entries_after - entries_before,
        "session_compact is emitted once per successful compaction, not per attempt \
         (entries {entries_before}->{entries_after}, emits {emitted_before}->{emitted_after})"
    );
    if refused {
        assert_eq!(
            fixture.compact_call_count(),
            compact_calls_before,
            "a refused compaction must not call the native endpoint again"
        );
    }
}

// ---------------------------------------------------------------------------
// T05.2 - the checkpoint survives a restart
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn checkpoint_survives_restart() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    let fixture = Fixture::build(FixtureOptions {
        case_name: "checkpoint-restart".to_string(),
        persist: true,
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;
    let session_file = fixture
        .session_file
        .clone()
        .expect("persisted session file");
    assert!(
        fixture.case.asserts_private(&session_file),
        "the session file must be inside the case root, got {session_file}"
    );

    seed_large_transcript(&fixture).await;
    fixture.turn(&long_user_text("restart-tail", 500)).await;

    fixture
        .session
        .compact_with_options(None, false)
        .await
        .expect("compaction succeeds");
    assert_eq!(
        fixture.compaction_entries().len(),
        1,
        "one durable compaction before the restart"
    );

    let before_context = fixture.context_texts();
    let before_messages = fixture.session.build_session_context().messages;
    assert_eq!(
        checkpoint_carrier_count(&before_messages),
        1,
        "the live context carries the checkpoint exactly once"
    );

    // Real reopen through the session manager and a fresh session.
    let reopened_manager = Arc::new(Mutex::new(
        SessionManager::open(&session_file, Some(&fixture.case.session_dir), None).expect("reopen"),
    ));
    let reopened_options = FixtureOptions {
        case_name: "unused".to_string(),
        ..Default::default()
    };
    let reopened_settings = fixture_settings(&reopened_options);
    let reopened_registry = fixture_registry(&fixture.model, true);
    let reopened_auth = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        private_auth_options(),
    )));
    reopened_auth
        .lock()
        .await
        .set_runtime_api_key(&fixture.model.provider, T05_API_KEY);

    let reopened_services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: fixture.case.cwd.clone(),
        agent_dir: Some(fixture.case.agent_dir.clone()),
        auth_storage: Some(Arc::clone(&reopened_auth)),
        settings_manager: Some(Arc::clone(&reopened_settings)),
        model_registry: Some(Arc::clone(&reopened_registry)),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(DefaultResourceLoaderOptions {
            cwd: fixture.case.cwd.clone(),
            agent_dir: fixture.case.agent_dir.clone(),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            settings_manager: Some(Arc::clone(&reopened_settings)),
            ..Default::default()
        }),
    })
    .await
    .expect("reopened services");

    let reopened = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(reopened_services),
        session_manager: Arc::clone(&reopened_manager),
        session_start_event: None,
        creation: fixture_creation(&fixture.model),
    })
    .await
    .expect("reopened session")
    .session;

    let after_context = message_texts(&reopened.build_session_context().messages);
    let after_messages = reopened.build_session_context().messages;
    assert_eq!(
        count_occurrences(&after_context, "old-user-0"),
        0,
        "the reopened context must not resurrect the summarized head"
    );
    assert_eq!(
        checkpoint_carrier_count(&after_messages),
        1,
        "the checkpoint must not be duplicated on reopen"
    );
    assert_eq!(
        serde_json::to_value(&after_messages).unwrap(),
        serde_json::to_value(&before_messages).unwrap(),
        "the reopened context must be equivalent to the pre-restart context"
    );
    assert_eq!(
        after_context.len(),
        before_context.len(),
        "history must not be truncated on reopen: before={} after={}",
        before_context.len(),
        after_context.len()
    );
    assert_eq!(
        message_texts(&reopened.messages()),
        after_context,
        "a reopened session's live conversation must match its context"
    );

    // The durable transcript still holds the retained tail, and the artifact dir
    // is still private.
    let reopened_branch: Vec<String> = reopened
        .session_manager
        .lock()
        .unwrap()
        .get_branch(None)
        .iter()
        .map(|entry| Value::Object(entry.clone()).to_string())
        .collect();
    assert_eq!(
        count_occurrences(&reopened_branch, "restart-tail"),
        1,
        "the durable transcript keeps the retained tail"
    );
    let artifact_dir = reopened
        .session_manager
        .lock()
        .unwrap()
        .get_session_artifact_dir();
    assert!(
        artifact_dir
            .as_deref()
            .map(|dir| fixture.case.asserts_private(dir))
            .unwrap_or(false),
        "the reopened session keeps its private artifact dir"
    );
    assert!(
        Path::new(&session_file).exists(),
        "the transcript file still exists"
    );
}

// ---------------------------------------------------------------------------
// T05.3 - model/endpoint switch and checkpoint eligibility
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn checkpoint_model_endpoint_switch() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    let fixture = Fixture::build(FixtureOptions {
        case_name: "checkpoint-model-switch".to_string(),
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;

    seed_large_transcript(&fixture).await;
    fixture.turn(&long_user_text("switch-tail", 400)).await;

    fixture
        .session
        .compact_with_options(None, false)
        .await
        .expect("compaction succeeds");
    assert_eq!(
        fixture.compaction_entries().len(),
        1,
        "one durable compaction"
    );

    // Same model/endpoint: the eligible checkpoint is replayed exactly once.
    fixture.turn("same model follow-up").await;
    let compatible = fixture.last_turn_request();
    assert_eq!(
        count_occurrences(&compatible.texts, "old-user-0"),
        0,
        "the compacted head must not be resent"
    );
    assert_eq!(
        compatible.provider_contexts.len(),
        1,
        "an eligible checkpoint is replayed exactly once for the same model/endpoint"
    );
    assert!(
        has_opaque_marker(&compatible),
        "the replay carries the provider's opaque window"
    );

    // Incompatible model + endpoint: the opaque window must never travel.
    let mut other = fixture.model.clone();
    other.id = "compaction-fixture-model-other".to_string();
    other.base_url = "https://other-endpoint.fixture.invalid/v1".to_string();
    fixture
        .session
        .set_model(
            other.clone(),
            ModelSelectOptions {
                wait_for_extensions: Some(false),
            },
        )
        .await
        .expect("model switch");
    // TS agent-session.ts:9358-9361 rebuilds the live conversation on a switch
    // that leaves a checkpoint behind.
    assert_eq!(
        fixture.live_texts(),
        fixture.context_texts(),
        "a model switch must rebuild the live conversation from the transcript"
    );

    fixture.turn("other model follow-up").await;
    let incompatible = fixture.last_turn_request();
    println!(
        "T05 switched-route request: model={} checkpoints={} opaque={} head={}",
        incompatible.model,
        incompatible.provider_contexts.len(),
        has_opaque_marker(&incompatible),
        count_occurrences(&incompatible.texts, "old-user-0")
    );
    assert_eq!(
        incompatible.provider_contexts.len(),
        0,
        "an opaque checkpoint must never be sent to an incompatible route"
    );
    assert!(
        !has_opaque_marker(&incompatible),
        "the opaque checkpoint items must not leak into a wrong route's request"
    );
    assert_eq!(
        count_occurrences(&incompatible.texts, "switch-tail"),
        1,
        "the portable tail carries the retained history to the other route"
    );

    // Back to the compatible model: the checkpoint is eligible again.
    fixture
        .session
        .set_model(
            fixture.model.clone(),
            ModelSelectOptions {
                wait_for_extensions: Some(false),
            },
        )
        .await
        .expect("model switch back");
    fixture.turn("back on the original model").await;
    let restored = fixture.last_turn_request();
    assert_eq!(
        restored.provider_contexts.len(),
        1,
        "the checkpoint is eligible again after switching back"
    );

    // The rebuild must not duplicate the harness digest carrier.
    let digest_carriers = fixture
        .session
        .build_session_context()
        .messages
        .iter()
        .filter(|message| match message {
            AgentMessage::Custom(custom) => serde_json::to_string(custom)
                .unwrap_or_default()
                .contains("harness_digest"),
            _ => false,
        })
        .count();
    println!("T05 digest carriers after switching: {digest_carriers}");
    assert!(
        digest_carriers <= 1,
        "the harness digest must not be duplicated by a rebuild, got {digest_carriers}"
    );
    assert_eq!(
        checkpoint_carrier_count(&fixture.session.messages()),
        1,
        "exactly one checkpoint carrier survives the switches"
    );
}

// ---------------------------------------------------------------------------
// T05.4 - post-compaction usage is not stale
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_compaction_usage_is_not_stale() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    // A text-only provider keeps the retained tail in the live window, so the
    // stale-usage guard is exercised on its own (no opaque-window rebuild).
    let fixture = Fixture::build(FixtureOptions {
        case_name: "post-compaction-usage".to_string(),
        text_only_provider: true,
        context_window: 100_000.0,
        reserve_tokens: 10_000.0,
        keep_recent_tokens: 1_000.0,
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;

    seed_large_transcript(&fixture).await;

    // A pre-compaction assistant message with large, now-stale usage.
    let stale = AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new("stale-large-usage-turn"))],
        api: fixture.model.api.clone(),
        provider: fixture.model.provider.clone(),
        model: fixture.model.id.clone(),
        stop_reason: "stop".to_string(),
        timestamp: now_ms(),
        usage: Usage {
            input: 90_000.0,
            output: 1_000.0,
            total_tokens: 91_000.0,
            ..Usage::zero()
        },
        ..Default::default()
    };
    fixture.spec.replies.lock().unwrap().push(stale);
    fixture.turn("turn with large usage").await;

    assert_eq!(
        fixture.compaction_entries().len(),
        1,
        "usage past the threshold must trigger a compaction"
    );
    let starts_after_first = observed.start_count();
    assert_eq!(starts_after_first, 1, "exactly one compaction ran");
    assert!(
        observed.end_count() >= 1,
        "the auto path emits compaction_end"
    );
    let first_end = observed.last_end().expect("compaction_end payload");
    assert_eq!(
        first_end["aborted"],
        Value::Bool(false),
        "a successful threshold compaction is not an abort"
    );
    assert!(
        first_end["result"].is_object(),
        "a successful compaction_end carries the result"
    );

    // The stale pre-compaction usage must not immediately trigger another
    // compaction: TS agent-session.ts:9392-9398 returns `undefined` when the usage
    // message predates the active compaction.
    fixture.turn("turn right after compaction").await;
    println!(
        "T05 after stale-usage turn: starts={} ends={} entries={}",
        observed.start_count(),
        observed.end_count(),
        fixture.compaction_entries().len()
    );
    assert_eq!(
        observed.start_count(),
        starts_after_first,
        "stale pre-compaction usage must not trigger a second compaction"
    );
    assert_eq!(
        fixture.compaction_entries().len(),
        1,
        "no second durable compaction entry for stale usage"
    );

    // A genuinely new usage crossing the threshold must still compact.
    let fresh = AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new("fresh-large-usage-turn"))],
        api: fixture.model.api.clone(),
        provider: fixture.model.provider.clone(),
        model: fixture.model.id.clone(),
        stop_reason: "stop".to_string(),
        timestamp: now_ms() + 60_000,
        usage: Usage {
            input: 95_000.0,
            output: 5_000.0,
            total_tokens: 100_000.0,
            ..Usage::zero()
        },
        ..Default::default()
    };
    fixture.spec.replies.lock().unwrap().push(fresh);
    fixture.turn("turn with fresh large usage").await;
    assert!(
        observed.start_count() > starts_after_first,
        "genuinely new usage past the threshold must compact again"
    );
    assert!(
        fixture.compaction_entries().len() >= 2,
        "the second compaction is durable"
    );
}

// ---------------------------------------------------------------------------
// T05.5 - threshold boundaries and configured settings
// ---------------------------------------------------------------------------

#[test]
fn threshold_and_settings_stay_unchanged() {
    let _guard = lock_suite();

    fn decide(tokens: f64, window: f64, settings: &CompactionSettings) -> bool {
        pi_coding_agent::core::compaction::compaction::should_compact(tokens, window, settings)
    }

    let settings = default_compaction_settings();
    assert_eq!(settings.reserve_tokens, 16384.0);
    assert_eq!(settings.keep_recent_tokens, 20000.0);
    assert_eq!(MAX_COMPACTION_CONTEXT_TOKENS, 250_000.0);

    // The inclusive 250000 cap with a large window.
    assert!(
        !decide(249_999.0, 1_000_000.0, &settings),
        "249999 is below the cap"
    );
    assert!(
        decide(250_000.0, 1_000_000.0, &settings),
        "250000 is at the inclusive cap"
    );
    assert!(
        decide(250_001.0, 1_000_000.0, &settings),
        "250001 is above the cap"
    );

    // A smaller context-window-minus-reserve boundary wins over the cap.
    // 210000 - 16384 = 193616.
    assert!(
        !decide(193_615.0, 210_000.0, &settings),
        "one token below window - reserve stays below"
    );
    assert!(
        decide(193_616.0, 210_000.0, &settings),
        "window - reserve is an inclusive boundary"
    );

    // A custom reserve moves the boundary but never past the default cap.
    let mut custom = settings.clone();
    custom.reserve_tokens = 50_000.0;
    assert!(
        !decide(249_999.0, 1_000_000.0, &custom),
        "the 250000 cap still applies with a custom reserve"
    );
    assert!(
        decide(250_000.0, 1_000_000.0, &custom),
        "the cap stays inclusive with a custom reserve"
    );
    // 300000 - 50000 = 250000: the same boundary.
    assert!(
        !decide(249_999.0, 300_000.0, &custom),
        "window - custom reserve is inclusive"
    );
    assert!(
        decide(250_000.0, 300_000.0, &custom),
        "window - custom reserve triggers at the boundary"
    );

    // Disabled compaction never triggers.
    let mut disabled = settings.clone();
    disabled.enabled = false;
    assert!(!decide(900_000.0, 1_000_000.0, &disabled));

    // The advertised model window and output settings are unchanged.
    let model = Model {
        id: "boundary-model".to_string(),
        name: "boundary-model".to_string(),
        api: "text-only".to_string(),
        provider: "text-only".to_string(),
        base_url: "https://boundary.invalid/v1".to_string(),
        input: vec![InputModality::Text],
        cost: ModelCost::zero(),
        context_window: 400_000.0,
        max_tokens: 32_000.0,
        ..Default::default()
    };
    assert_eq!(
        pi_ai::models::get_model_input_limit(&model),
        400_000.0,
        "compaction must not shrink the advertised window"
    );
    assert!(should_compact_for_model(250_000.0, &model, &settings));
    assert!(
        !should_compact_for_model(249_999.0, &model, &settings),
        "one token below the cap does not compact"
    );

    let mut small = model.clone();
    small.context_window = 200_000.0;
    // 200000 - 16384 = 183616 wins over the cap.
    assert!(
        !should_compact_for_model(180_000.0, &small, &settings),
        "a smaller window wins over the cap"
    );
    assert!(
        should_compact_for_model(190_000.0, &small, &settings),
        "the smaller window compacts above its own boundary"
    );

    let mut capped = model.clone();
    capped.max_input_tokens = Some(150_000.0);
    // 150000 - 16384 = 133616 wins over the advertised window.
    assert_eq!(
        pi_ai::models::get_model_input_limit(&capped),
        150_000.0,
        "the configured input ceiling is the input limit"
    );
    assert!(
        !should_compact_for_model(130_000.0, &capped, &settings),
        "a configured input ceiling still wins"
    );
    assert!(
        should_compact_for_model(140_000.0, &capped, &settings),
        "the configured ceiling compacts above its own boundary"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compaction_body_uses_configured_settings() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    // Configured keep-recent is small enough to leave a summarizable head, while
    // the built-in default (20000) reports "session is too short". A large model
    // max_tokens keeps the output cap readable as 0.8 * reserveTokens.
    let fixture = Fixture::build(FixtureOptions {
        case_name: "configured-settings-body".to_string(),
        text_only_provider: true,
        reserve_tokens: 3_000.0,
        keep_recent_tokens: 1_000.0,
        model_max_tokens: 200_000.0,
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;

    fixture.turn(&long_user_text("settings-user-0", 2_000)).await;
    fixture.turn(&long_user_text("settings-user-1", 2_000)).await;
    fixture.turn(&long_user_text("settings-user-2", 2_000)).await;
    fixture.turn(&long_user_text("settings-tail", 100)).await;

    let result = fixture.compact(None).await;
    println!("T05 configured-settings compaction result: {result:?}");
    assert!(
        result.is_ok(),
        "the compaction body must use the configured keepRecentTokens, got {result:?}"
    );
    assert_eq!(
        fixture.compaction_entries().len(),
        1,
        "one durable compaction entry"
    );

    let end = observed.last_end().expect("manual compaction completion event");
    assert!(end["result"].is_object(), "successful manual compaction must carry its result");
    assert_eq!(end["aborted"], Value::Bool(false));

    let requests = fixture.spec.requests.lock().unwrap().clone();
    let summary = requests
        .iter()
        .find(|request| request.is_summary_call)
        .expect("a summarization request");
    println!(
        "T05 summary call: model={} provider={} max_tokens={:?}",
        summary.model, summary.provider, summary.max_tokens
    );
    assert_eq!(
        summary.max_tokens,
        Some(2_400.0),
        "the summary call must use the configured reserve (0.8 * 3000)"
    );
    assert_eq!(
        summary.model,
        fixture.model.id,
        "the summary must use the selected model, not a substitute"
    );
    assert_eq!(
        summary.provider,
        fixture.model.provider,
        "the summary must use the selected provider"
    );
    assert!(
        !fixture
            .spec
            .summary_texts()
            .iter()
            .any(|text| text.contains("Compaction fallback")),
        "a text-only model must not claim a native-compaction fallback"
    );
}

// ---------------------------------------------------------------------------
// T05.6 - the session_before_compact hook is real
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn before_compact_hook_is_real() {
    let _guard = lock_suite();
    let observed = Arc::new(Observed::default());
    let fixture = Fixture::build(FixtureOptions {
        case_name: "before-compact-hook".to_string(),
        text_only_provider: true,
        observed: Arc::clone(&observed),
        ..Default::default()
    })
    .await;

    seed_large_transcript(&fixture).await;
    fixture.turn(&long_user_text("hook-tail", 400)).await;

    step("compact: manual begin");
    fixture
        .compact(Some("hook instructions"))
        .await
        .expect("compaction succeeds");
    step("compact: manual done");

    let payloads = observed.before_compact.lock().unwrap().clone();
    println!("T05 session_before_compact payloads: {}", payloads.len());
    assert_eq!(payloads.len(), 1, "the hook must see exactly one preparation");
    let payload = &payloads[0];
    let preparation = &payload["preparation"];
    assert!(
        preparation["tokensBefore"].as_f64().unwrap_or(0.0) > 0.0,
        "the hook preparation must carry real tokens, got {preparation}"
    );
    assert!(
        !preparation["messagesToSummarize"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .is_empty(),
        "the hook preparation must carry the messages to summarize"
    );
    assert!(
        !preparation["firstKeptEntryId"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "the hook preparation must carry the cut point"
    );
    assert_eq!(
        payload["customInstructions"],
        Value::String("hook instructions".to_string()),
        "the hook receives the caller's custom instructions"
    );
    let branch_entries = payload["branchEntries"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        !branch_entries.is_empty(),
        "the hook receives the real branch entries"
    );
    assert!(
        branch_entries
            .iter()
            .all(|entry| entry.get("type").is_some()),
        "each branch entry is a real session entry"
    );
    let first_kept = preparation["firstKeptEntryId"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        branch_entries
            .iter()
            .any(|entry| entry.get("id").and_then(Value::as_str) == Some(first_kept.as_str())),
        "the cut point names a real branch entry"
    );

    // The hook's result must change the real outcome: cancel stops the compaction.
    // A cancel is only reachable on a branch that can still be prepared, so this
    // part runs on a fresh session with the hook already armed to cancel.
    let cancel_observed = Arc::new(Observed::default());
    *cancel_observed.before_compact_result.lock().unwrap() = Some(json!({ "cancel": true }));
    let cancel_fixture = Fixture::build(FixtureOptions {
        case_name: "before-compact-cancel".to_string(),
        text_only_provider: true,
        observed: Arc::clone(&cancel_observed),
        ..Default::default()
    })
    .await;
    seed_large_transcript(&cancel_fixture).await;
    cancel_fixture.turn(&long_user_text("cancel-tail", 400)).await;
    let cancelled = cancel_fixture.compact(None).await;
    let message = format!("{cancelled:?}");
    println!("T05 cancelled compaction result: {message}");
    assert!(
        message.contains("Compaction cancelled"),
        "a cancelling hook must abort the compaction, got {message}"
    );
    assert!(
        cancel_fixture.compaction_entries().is_empty(),
        "a cancelled compaction appends no durable entry"
    );
    assert_eq!(
        cancel_observed.before_compact.lock().unwrap().len(),
        1,
        "the cancelling fixture saw exactly one preparation"
    );
    assert_eq!(
        cancel_observed.end_count(),
        1,
        "the cancelled manual attempt is reported exactly once"
    );
    assert_eq!(
        cancel_observed.last_end().unwrap()["aborted"],
        Value::Bool(true),
        "a cancelled compaction is reported as aborted"
    );
}

// ---------------------------------------------------------------------------
// T05.7 - unsupported, cancelled, rejected and incompatible are distinct
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unsupported_abort_and_failure_are_distinct() {
    let _guard = lock_suite();

    // (a) A provider that reports "unsupported" must fall back to the text
    // summarizer with the SAME model (TS compaction.ts:895-925: `remote` undefined
    // falls through to `generateSummary`).
    {
        let observed = Arc::new(Observed::default());
        let fixture = Fixture::build(FixtureOptions {
            case_name: "unsupported-falls-back".to_string(),
            observed: Arc::clone(&observed),
            ..Default::default()
        })
        .await;
        fixture.spec.requests.lock().unwrap().clear();
        // Four large turns plus a tail: the retention walk must cross the default
        // keepRecentTokens (20000) BEFORE the first user message, otherwise the cut
        // point is the first entry and there is no history to summarize.
        seed_large_transcript(&fixture).await;
        fixture.turn(&long_user_text("unsupported-tail", 400)).await;

        *fixture.spec.compact_mode.lock().unwrap() = CompactMode::Unsupported;
        let result = fixture.compact(None).await;
        println!("T05 unsupported-compact result: {result:?}");
        assert!(
            result.is_ok(),
            "an unsupported native compaction must fall back to text, got {result:?}"
        );
        let entries = fixture.compaction_entries();
        assert_eq!(entries.len(), 1, "one durable compaction entry");
        assert!(
            entries[0].get("details").and_then(|details| details.get("providerCheckpoint")).is_none(),
            "no provider checkpoint may be committed on the text fallback"
        );
        let requests = fixture.spec.requests.lock().unwrap().clone();
        let summary = requests
            .iter()
            .find(|request| request.is_summary_call)
            .expect("the text summarizer ran");
        assert_eq!(
            summary.model, fixture.model.id,
            "the fallback summarizer must use the selected model"
        );
        assert!(
            !fixture
                .spec
                .summary_texts()
                .iter()
                .any(|text| text.contains("Compaction fallback")),
            "a non-staged-Azure model must not claim the native fallback notice"
        );
        assert_eq!(
            fixture.compact_call_count(),
            1,
            "the native endpoint was attempted exactly once"
        );
    }

    // (b) An invalid/incompatible checkpoint is a failure, never a silent
    // text fallback (TS compaction.ts:911-913 throws).
    {
        let fixture = Fixture::build(FixtureOptions {
            case_name: "incompatible-checkpoint".to_string(),
            ..Default::default()
        })
        .await;
        seed_large_transcript(&fixture).await;
        fixture.turn(&long_user_text("incompatible-tail", 400)).await;
        *fixture.spec.compact_mode.lock().unwrap() = CompactMode::IncompatibleCheckpoint;
        let result = fixture.compact(None).await;
        let message = format!("{result:?}");
        println!("T05 incompatible-checkpoint result: {message}");
        assert!(
            message.contains("incompatible checkpoint"),
            "an incompatible checkpoint must fail the compaction, got {message}"
        );
        assert_eq!(
            fixture.compaction_entries().len(),
            0,
            "a failed compaction commits no durable entry"
        );
        assert!(
            message.contains("not supported") == false,
            "a checkpoint failure must not be reported as unsupported: {message}"
        );
    }

    // (c) A missing API key rejects the compaction before any provider call and
    // leaves the original history usable.
    {
        let fixture = Fixture::build(FixtureOptions {
            case_name: "auth-rejection".to_string(),
            with_api_key: false,
            ..Default::default()
        })
        .await;
        // Seed the transcript structurally: without a key a live turn cannot run,
        // and the point of this case is the compaction auth pre-check.
        seed_messages(
            &fixture,
            &[
                ("auth-user-0", 3000),
                ("auth-user-1", 3000),
                ("auth-user-2", 3000),
                ("auth-tail", 200),
            ],
        );
        let before = fixture.live_texts();
        // The seeded pair is (user "auth-user-0 ...", assistant "reply-auth-user-0"),
        // so the tag legitimately appears in both messages; the assertion only needs
        // the seeded history to be present at all.
        assert!(
            count_occurrences(&before, "auth-user-0") >= 1,
            "the seeded history must be in the live conversation"
        );
        let result = fixture.compact(None).await;
        let message = format!("{result:?}");
        println!("T05 auth-rejection result: {message}");
        assert!(
            message.contains("No API key")
                || message.contains("no API key")
                || message.contains("Authentication")
                || message.contains("authentication"),
            "an auth rejection must be reported, got {message}"
        );
        assert_eq!(
            fixture.compaction_entries().len(),
            0,
            "a rejected compaction commits nothing"
        );
        assert_eq!(
            fixture.live_texts(),
            before,
            "a rejected compaction leaves the original history untouched"
        );
        assert_eq!(
            fixture.compact_call_count(),
            0,
            "no provider compaction call happens without auth"
        );
        assert_eq!(
            fixture.spec.summary_calls.load(Ordering::SeqCst),
            0,
            "no summarization call happens without auth"
        );
    }

    // (d) A summarization failure is a failure: never a silent success and never a
    // text fallback that pretends a provider checkpoint exists.
    {
        let observed = Arc::new(Observed::default());
        let fixture = Fixture::build(FixtureOptions {
            case_name: "summary-failure".to_string(),
            text_only_provider: true,
            observed: Arc::clone(&observed),
            ..Default::default()
        })
        .await;
        seed_large_transcript(&fixture).await;
        observed.ends.lock().unwrap().clear();
        *fixture.spec.summary_failure.lock().unwrap() = Some("synthetic network failure".to_string());
        let before = fixture.live_texts();
        let result = fixture.compact(None).await;
        let message = format!("{result:?}");
        println!("T05 summary-failure result: {message}");
        assert!(
            message.contains("Summary")
                || message.contains("synchronization")
                || message.contains("synthetic network failure"),
            "a summarization failure must be reported, got {message}"
        );
        assert!(
            !message.contains("Compaction cancelled"),
            "a summary failure must not be reported as a cancellation: {message}"
        );
        assert_eq!(
            fixture.compaction_entries().len(),
            0,
            "a failed compaction commits no durable entry"
        );
        assert_eq!(
            fixture.live_texts(),
            before,
            "a failed compaction leaves the original history untouched"
        );
    }

    // (e) Cancellation during one summary call must reach the AUTO-compaction
    // classifier as a cancellation, not as a failure (TS agent-session.ts:9676-9689
    // recognises an AbortError and reports "cancelled"/aborted).
    {
        let observed = Arc::new(Observed::default());
        let fixture = Fixture::build(FixtureOptions {
            case_name: "abort-during-summary".to_string(),
            text_only_provider: true,
            context_window: 100_000.0,
            reserve_tokens: 10_000.0,
            keep_recent_tokens: 1_000.0,
            observed: Arc::clone(&observed),
            ..Default::default()
        })
        .await;
        seed_large_transcript(&fixture).await;
        observed.ends.lock().unwrap().clear();
        observed.starts.lock().unwrap().clear();

        // The turn's own reported usage crosses the threshold, so `agent_end`
        // starts a real auto (threshold) compaction whose summary call parks.
        let big_usage = AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new("threshold-usage-turn"))],
            api: fixture.model.api.clone(),
            provider: fixture.model.provider.clone(),
            model: fixture.model.id.clone(),
            stop_reason: "stop".to_string(),
            timestamp: now_ms(),
            usage: Usage {
                input: 90_000.0,
                output: 1_000.0,
                total_tokens: 91_000.0,
                ..Usage::zero()
            },
            ..Default::default()
        };
        fixture.spec.replies.lock().unwrap().push(big_usage);
        fixture.spec.arm_summary_gate(true);

        let session = Arc::clone(&fixture.session);
        let task = tokio::spawn(async move {
            let result = session.prompt("turn that triggers the threshold compaction", None).await;
            let _ = session.wait_for_headless_idle().await;
            result
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while fixture.spec.summary_gate_waiters() == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the auto-compaction summarization call never started"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            observed.start_count() >= 1,
            "compaction_start must precede the summary call"
        );
        fixture.session.abort_compaction();
        fixture.spec.release_summary_gate();
        tokio::time::timeout(Duration::from_secs(60), task)
            .await
            .expect("the prompted turn never finished")
            .expect("task join")
            .expect("the prompt itself is accepted");

        let ends = observed.ends.lock().unwrap().clone();
        let auto_end = ends
            .iter()
            .find(|end| end["reason"] == Value::String("threshold".to_string()))
            .cloned()
            .expect("the threshold compaction reported a compaction_end");
        println!("T05 cancelled auto compaction end: {auto_end}");
        assert_eq!(
            auto_end["aborted"],
            Value::Bool(true),
            "a summary call cancelled through abort_compaction must be reported as an abort, got {auto_end}"
        );
        assert_eq!(
            auto_end["error_message"],
            Value::Null,
            "an aborted compaction carries no error message, got {auto_end}"
        );
        assert_eq!(
            fixture.compaction_entries().len(),
            0,
            "a cancelled compaction commits no durable entry"
        );
        assert_eq!(
            count_occurrences(&fixture.live_texts(), "old-user-0"),
            1,
            "a cancelled compaction leaves the original history usable"
        );
    }
}
