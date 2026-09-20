//! Native System One adapter. Compare observes without changing execution.
//! Active and combined modes share bounded provider/retrieval decisions with
//! request-local transformations. Compaction has a separate opt-in gate.
//! Credentials, cancellation and captured policy generations remain local.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use pi_jev::config::{JevMode, JevSettings};
use pi_jev::hooks::JevObserver;
use pi_jev::mock::MockJevTransport;
use serde_json::{json, Value};

use crate::config::get_agent_dir;
use crate::core::extensions::types::SharedExtension;
use crate::core::extensions::types::{Extension, ExtensionContext, ExtensionEvent, ExtensionHandler};
use crate::core::memory::search::MemoryHit;

/// Path of the internal observer extension (stable, easy to spot in logs).
pub const JEV_OBSERVER_PATH: &str = "<jev-observer-internal>";

fn live_bridges() -> &'static Mutex<Vec<Weak<JevBridgeCore>>> {
    static BRIDGES: OnceLock<Mutex<Vec<Weak<JevBridgeCore>>>> = OnceLock::new();
    BRIDGES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Worker-local read-only telemetry. None means no observed request in this
/// process, not a fabricated healthy/zero snapshot. Daemon callers must use
/// the session worker's result rather than the supervisor's empty registry.
pub fn session_status_snapshot(session_id: &str) -> Option<Value> {
    let cores: Vec<_> = live_bridges().lock().unwrap_or_else(|p| p.into_inner())
        .iter().filter_map(Weak::upgrade).collect();
    let mut result = None;
    for core in cores {
        if let Some(build) = core.observer.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            if let Some(mut status) = build.observer.session_status(session_id) {
                status["active_breaker_open"] = json!(build.observer.active_breaker_open());
                result = Some(status);
                break;
            }
        }
    }
    if let Some(compaction) = crate::core::jev_compaction::session_status(session_id) {
        let status = result.get_or_insert_with(|| json!({}));
        status["compaction"] = compaction;
    }
    result
}

/// Metadata-only footer text. This function never creates a client or task.
pub fn footer_status_text(session_id: &str) -> String {
    let settings = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
    let mode = settings.effective_mode(session_id);
    let (color, label) = if !mode.is_enabled() {
        ("error", "Jev Off")
    } else {
        let present = pi_jev::config::jev_dir_for(get_agent_dir()).join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME)).is_file()
            || pi_jev::config::EnvKeyPresence::from_env() != pi_jev::config::EnvKeyPresence::default();
        let status = session_status_snapshot(session_id);
        if !present && !(cfg!(debug_assertions) && settings.transport.as_deref().is_some_and(|value| value.starts_with("mock"))) {
            ("warning", "Jev unavailable")
        } else if status.as_ref().is_some_and(|value| value["in_flight"].as_u64().unwrap_or(0) > 0) {
            ("warning", "Jev checking")
        } else if status.as_ref().is_some_and(|value| value["fallback_reason"].as_str().is_some_and(|reason| !reason.is_empty())) {
            ("warning", "Jev fallback")
        } else if status.as_ref().is_some_and(|value| value["active"]["last_reason"].as_str().is_some_and(|reason| !reason.is_empty())) {
            ("warning", "Jev fallback")
        } else { ("accent", mode.label()) }
    };
    crate::modes::interactive::theme::theme::theme().fg(color, &format!("\u{25cf} {label}"))
}

/// The one event the Active handler subscribes to. Registering it makes the
/// provider path dispatch one bounded extension call per request; in Off and
/// Compare the handler returns the untouched payload immediately.
pub const JEV_ACTIVE_EVENT: &str = "before_provider_request";

/// Settings-file cache TTL: one cheap stat/read per interval per process.
const SETTINGS_TTL: Duration = Duration::from_millis(250);

/// Events the observer extension subscribes to. Every handler here is
/// observe-only and always returns `None`; no session_before_* event and no
/// mutating surface is touched. `input` is bookkeeping-only (task text
/// capture) and returns `None`, which the input pipeline maps to "Continue"
/// with the original text.
///
/// Active mode adds one separate handler on `before_provider_request`
/// (see [`JEV_ACTIVE_EVENT`]). It is registered in addition to these, and it
/// is the only Jev handler in the process that may return a modified value.
pub const JEV_EVENTS: [&str; 12] = [
    "session_start",
    "agent_start",
    "turn_start",
    "turn_end",
    "input",
    "tool_call",
    "tool_execution_start",
    "tool_execution_end",
    "message_end",
    "agent_end",
    "model_select",
    "session_shutdown",
];

// ---------------------------------------------------------------------------
// Settings cache (cheap mode checks)
// ---------------------------------------------------------------------------

struct CachedSettings {
    loaded: Instant,
    settings: JevSettings,
}

fn settings_cache() -> &'static Mutex<Option<CachedSettings>> {
    static CACHE: OnceLock<Mutex<Option<CachedSettings>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Load Jev settings with a tiny TTL cache. Off-mode cost: one cached read
/// per event; no client, no network, no tasks.
fn load_settings_cached() -> JevSettings {
    {
        let cache = settings_cache().lock().unwrap_or_else(|p| p.into_inner());
        if let Some(cached) = cache.as_ref() {
            if cached.loaded.elapsed() < SETTINGS_TTL {
                return cached.settings.clone();
            }
        }
    }
    let agent_dir = get_agent_dir();
    let settings = pi_jev::config::JevSettingsStore::new(&agent_dir).load();
    let mut cache = settings_cache().lock().unwrap_or_else(|p| p.into_inner());
    *cache = Some(CachedSettings {
        loaded: Instant::now(),
        settings: settings.clone(),
    });
    settings
}

/// Invalidate the settings cache; the /jev UI lane can call this after
/// writing settings so a mode change is visible immediately. It also re-syncs
/// Active-handler presence, so `/jev active` takes effect without a restart.
pub fn invalidate_settings_cache() {
    *settings_cache().lock().unwrap_or_else(|p| p.into_inner()) = None;
    sync_active_handlers();
}

/// True when some session or the global default is set to Active.
fn active_mode_requested() -> bool {
    let settings = load_settings_cached();
    settings.global_default.is_some_and(JevMode::allows_active)
        || settings
            .sessions
            .values()
            .any(|session| session.mode.is_some_and(JevMode::allows_active))
}

/// Add or remove the Active provider-request handler on every live bridge.
///
/// The handler is the only Jev surface that can change a provider request, and
/// the runner treats "a `before_provider_request` handler exists" as "the
/// request body may change", which also decides whether a retry may reuse the
/// previous turn's semantic edges. Registering it while every session is Off or
/// Compare would change retry behavior for a user who never enabled Jev, so
/// handler presence follows the setting instead of the process.
fn sync_active_handlers() {
    let cores: Vec<_> = live_bridges()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    let wanted = active_mode_requested();
    for core in cores {
        core.set_active_handler(&core, wanted);
    }
}

/// A dormant adapter must exist even at default-Off startup so `/jev compare`
/// can take effect in the same session. Client and tasks remain lazy.
pub fn maybe_register_jev_observer(extensions: &mut Vec<SharedExtension>) {
    let settings = load_settings_cached();
    let core = Arc::new(JevBridgeCore::new(settings));
    {
        let mut bridges = live_bridges().lock().unwrap_or_else(|p| p.into_inner());
        bridges.retain(|core| core.strong_count() > 0);
        bridges.push(Arc::downgrade(&core));
    }
    let extension = build_observer_extension(Arc::clone(&core));
    core.attach_extension(&extension);
    core.set_active_handler(&core, active_mode_requested());
    extensions.push(extension);
}

// ---------------------------------------------------------------------------
// Bridge core
// ---------------------------------------------------------------------------

/// Bounded per-session bookkeeping. Only a capped task-text excerpt is kept
/// (for the next TurnStart snapshot); it never reaches records or logs.
#[derive(Clone, Default)]
struct SessionBook {
    last_task_excerpt: Option<String>,
    observed_tools: Vec<String>,
    turn: u64,
    trace: pi_jev::observation::TraceObserver,
}

struct JevBridgeCore {
    /// Cached observer keyed by the credential fingerprint it was built
    /// with; a credential rotation (or transport change) rebuilds it.
    observer: Mutex<Option<ObserverBuild>>,
    sessions: Mutex<HashMap<String, SessionBook>>,
    run_metrics: crate::core::jev_run_metrics::JevRunMetrics,
    /// Weak handle to this core's registered extension, so Active-handler
    /// presence can follow the setting at runtime.
    extension: Mutex<Weak<Mutex<Extension>>>,
}

struct ObserverBuild {
    /// Cheap change probe matched BEFORE any DPAPI/client work on the hot path.
    cheap_stamp: String,
    observer: Arc<JevObserver>,
}

const MAX_TRACKED_SESSIONS: usize = 64;
const MAX_OBSERVED_TOOLS: usize = 16;

impl JevBridgeCore {
    fn new(settings: JevSettings) -> Self {
        let _ = settings;
        Self {
            observer: Mutex::new(None),
            sessions: Mutex::new(HashMap::new()),
            run_metrics: crate::core::jev_run_metrics::JevRunMetrics::new(std::path::PathBuf::from(get_agent_dir())),
            extension: Mutex::new(Weak::new()),
        }
    }

    /// Remember the registered extension so Active-handler presence can be
    /// added or removed later. Called once, at registration.
    fn attach_extension(&self, extension: &SharedExtension) {
        *self.extension.lock().unwrap_or_else(|p| p.into_inner()) = Arc::downgrade(extension);
    }

    /// Make Active-handler presence match `wanted`. This is the only place the
    /// provider-request handler is installed, and it is installed for every
    /// registered bridge rather than for one session: an extension handler list
    /// is process-wide, while modes are per session. Presence is therefore the
    /// conservative union ("some session is Active"), and the handler itself
    /// re-checks the effective mode of the session it is called for.
    fn set_active_handler(&self, core: &Arc<JevBridgeCore>, wanted: bool) {
        let Some(extension) = self
            .extension
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .upgrade()
        else {
            return;
        };
        let mut guard = extension.lock().unwrap_or_else(|p| p.into_inner());
        let handlers = guard
            .handlers
            .entry(JEV_ACTIVE_EVENT.to_string())
            .or_default();
        if wanted {
            if handlers.is_empty() {
                handlers.push(make_active_handler(Arc::clone(core)));
            }
        } else {
            handlers.clear();
        }
    }

    fn remember_task_text(&self, session_id: &str, text: &str) {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if sessions.len() > MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) {
            if let Some(oldest) = sessions.keys().next().cloned() {
                sessions.remove(&oldest);
            }
        }
        let book = sessions.entry(session_id.to_string()).or_default();
        // Redacted while the excerpt is built: this text is later sent to
        // SystemOne as `user_text_excerpt`.
        book.last_task_excerpt = Some(pi_jev::redact::bounded_excerpt(text, 400));
    }

    fn task_excerpt(&self, session_id: &str) -> Option<String> {
        let sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions
            .get(session_id)
            .and_then(|book| book.last_task_excerpt.clone())
    }

    fn note_tool_call(&self, session_id: &str, tool_name: &str) {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let book = sessions.entry(session_id.to_string()).or_default();
        if !book.observed_tools.iter().any(|name| name == tool_name) {
            book.observed_tools.push(tool_name.to_string());
            book.observed_tools.truncate(MAX_OBSERVED_TOOLS);
        }
    }

    fn observed_tools(&self, session_id: &str) -> Vec<String> {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(session_id)
            .map(|book| book.observed_tools.clone())
            .unwrap_or_default()
    }

    fn forget_session(&self, session_id: &str) {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(session_id);
    }

    fn note_turn(&self, session_id: &str, turn: u64) {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner())
            .entry(session_id.to_string()).or_default().turn = turn;
    }

    fn turn(&self, session_id: &str) -> u64 {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner())
            .get(session_id).map(|book| book.turn).unwrap_or(0)
    }

    fn note_observation(&self, session_id: &str, event: &ExtensionEvent) {
        use pi_jev::observation::{ObservedStopReason, TraceEvent};
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let book = sessions.entry(session_id.to_string()).or_default();
        match event {
            ExtensionEvent::TurnStart(_) => book.trace.record(TraceEvent::TurnStarted),
            ExtensionEvent::ToolExecutionEnd(payload) => book.trace.record(TraceEvent::ToolEnded { is_error: payload.is_error }),
            ExtensionEvent::MessageEnd(payload) if payload.message.get("role").and_then(Value::as_str) == Some("assistant") => {
                let stop = payload.message.get("stopReason").and_then(Value::as_str).unwrap_or("");
                let kind = observed_failure_kind(&payload.message);
                book.trace.record(TraceEvent::AssistantEnded { stop_reason: ObservedStopReason::from_stop_reason(stop), failure_kind: kind });
            }
            _ => {}
        }
    }

    fn observation(&self, session_id: &str) -> pi_jev::observation::TraceSummary {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner()).get(session_id)
            .map(|book| book.trace.summary()).unwrap_or_default()
    }

    async fn observe_optional(&self, session_id: &str, event: &ExtensionEvent,
        ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>, handler_event: &'static str) {
        let settings = load_settings_cached();
        let features = settings.effective_features(session_id);
        let stage = match event {
            ExtensionEvent::TurnEnd(_) if features.loop_control || features.retry_classification => pi_jev::snapshot::SnapshotStage::TurnEnd,
            ExtensionEvent::AgentEnd(_) if features.result_sufficiency || features.loop_control || features.verification
                || features.retry_classification || features.trace_observer => pi_jev::snapshot::SnapshotStage::AgentEnd,
            _ => return,
        };
        // Only explicitly enabled observational categories run in Active-only mode.
        let Some(core) = bridge_for_session(session_id) else { return; };
        let Some((_, mut payload)) = bridge_event(&core, event, ctx, session_id, handler_event) else { return; };
        payload["policy_generation"] = json!(decision_policy_generation(&settings,session_id));
        payload["state"]["features"] = json!(features);
        let Some(observer) = self.observer(session_id, ctx.ui()) else { return; };
        let state = payload.get("state").cloned().unwrap_or(Value::Null);
        let Ok(snapshot) = pi_jev::snapshot::StateSnapshot::new(stage, session_id, self.turn(session_id), 0, None, state, Vec::new()) else { return; };
        let mut questions = Vec::new();
        for evaluator in pi_jev::evaluators::for_boundary(stage) {
            let enabled = match evaluator.category().as_str() {
                "result_sufficiency" => features.result_sufficiency,
                "continue_stop_escalate" => features.loop_control,
                "first_pass_verification" => features.verification,
                "retry_classification" => features.retry_classification,
                "trace_assessment" => features.trace_observer,
                _ => false,
            };
            if enabled {
                if let pi_jev::evaluators::EvaluatorOutput::Questions(mut prepared) = evaluator.evaluate(&snapshot) { questions.append(&mut prepared); }
            }
        }
        let policy = pi_jev::active::ActivationPolicy { enabled_categories: Default::default(), ..Default::default() };
        let outcome = observer.decide_prepared(&payload, stage.as_str(), questions, &policy).await;
        observer.record_active(&outcome, &BTreeMap::new());
    }

    /// Off-mode immediate effect: cancel queued/in-flight comparison work
    /// for the session and drop its bounded bookkeeping. The observer slot
    /// itself is kept (rebuilt lazily on the next Compare event).
    fn drop_session_work(&self, session_id: &str) {
        if let Some(build) = self
            .observer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            build.observer.cancel_session(session_id);
        }
        self.forget_session(session_id);
    }

    fn drop_decision_work(&self, session_id: &str) {
        if let Some(build) = self.observer.lock().unwrap_or_else(|p| p.into_inner()).as_ref() { build.observer.cancel_decisions(session_id); }
        if load_settings_cached().effective_compaction_enabled(session_id) {
            if let Some(book) = self.sessions.lock().unwrap_or_else(|p| p.into_inner()).get_mut(session_id) { *book = SessionBook::default(); }
        } else { self.forget_session(session_id); }
    }

    fn own_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if sessions.len() >= MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) { return; }
        sessions.entry(session_id.to_string()).or_default();
    }

    /// Effective mode for one session (cheap cached settings read).
    fn effective_mode(&self, session_id: Option<&str>) -> JevMode {
        load_settings_cached().effective_mode(session_id.unwrap_or(""))
    }

    /// Lazily construct the observer on the first Compare-mode event, and
    /// rebuild it when the effective credential changes (rotation). In Off
    /// nothing is ever constructed.
    fn observer(&self, session_id: &str, ui: Arc<dyn crate::core::extensions::types::ExtensionUiContext>) -> Option<Arc<JevObserver>> {
        let settings = load_settings_cached();
        if !settings.effective_mode(session_id).is_enabled() && !settings.effective_compaction_enabled(session_id) {
            return None;
        }
        // Cheap change probe BEFORE any credential read or client build: the
        // transport selector, the credential envelope's stamp and the env
        // presence booleans. No DPAPI decrypt and no reqwest client on the
        // hot path when nothing changed.
        let cheap = cheap_credential_stamp(&settings);
        if let Some(cached) = self
            .observer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            if cached.cheap_stamp == cheap {
                return Some(Arc::clone(&cached.observer));
            }
        }
        // A changed envelope generation must rebuild even if it decrypts to
        // the same key: the dispatch gate captures the generation, not only
        // key identity. Reusing that observer would reject all later work.
        let (transport, credential, _fingerprint) = build_transport(&settings);
        let effective = settings.effective_mode(session_id);
        // The transport is data-only. Independent compaction can use it while decisions are Off.
        let client_mode = if effective.is_enabled() { effective } else { JevMode::Compare };
        let system_one: Arc<dyn pi_jev::types::SystemOne> = match
            pi_jev::client::JevSystemOne::new(
                client_mode,
                credential,
                transport,
                pi_jev::client::JevLimits::default(),
                Arc::new(pi_jev::client::JevStats::default()),
            ) {
            Ok(client) => Arc::new(client),
            // Construction can only fail on an unusable credential here;
            // comparison requests then fail closed to logged skips and no
            // network object is used.
            Err(_) => Arc::new(pi_jev::client::DisabledSystemOne::new(effective)),
        };
        // The observer re-checks the effective mode itself; the gate reads the
        // same cached settings the handlers use (one cheap read per event).
        let observed_stamp = cheap.clone();
        let independent_stamp = cheap.clone();
        let footer_session = session_id.to_string();
        let config = pi_jev::hooks::JevObserverConfig {
            mode_gate: Arc::new(move |session_id: Option<&str>| {
                // Dispatch and completion use current settings, not the event
                // cache: a queued request cannot outlive Off/key rotation.
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                if cheap_credential_stamp(&current) != observed_stamp {
                    return JevMode::Off;
                }
                current.effective_mode(session_id.unwrap_or(""))
            }),
            policy_generation: Arc::new(|session_id, independent| {
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                if independent {
                    compaction_policy_generation(&current, session_id)
                } else {
                    decision_policy_generation(&current, session_id)
                }
            }),
            independent_gate: Arc::new(move |session_id| {
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                cheap_credential_stamp(&current) == independent_stamp && current.effective_compaction_enabled(session_id)
            }),
            on_terminal: Some(Arc::new(move |session_id| {
                if session_id == footer_session {
                    ui.set_status("jev".into(), Some(footer_status_text(session_id)));
                }
            })),
            ..pi_jev::hooks::JevObserverConfig::default()
        };
        let records_path = std::path::Path::new(&get_agent_dir())
            .join("jev")
            .join("records.jsonl");
        let observer = JevObserver::new(config, system_one, records_path);
        let build = ObserverBuild {
            cheap_stamp: cheap,
            observer: Arc::clone(&observer),
        };
        let previous = self.observer.lock().unwrap_or_else(|p| p.into_inner()).replace(build);
        if let Some(previous) = previous {
            previous.observer.shutdown();
        }
        Some(observer)
    }
}


/// Cheap change probe for the observer cache: the transport selector, the
/// credential envelope's metadata stamp and the env presence booleans. Reads
/// no secret and performs no DPAPI work; a changed stamp triggers the full
/// transport build (whose fingerprint includes the decrypted key's hash).
fn cheap_credential_stamp(settings: &JevSettings) -> String {
    credential_stamp_at(settings, &std::path::PathBuf::from(get_agent_dir()))
}

fn credential_stamp_at(settings: &JevSettings, agent_dir: &std::path::Path) -> String {
    let transport_selector = settings.transport.clone().unwrap_or_default();
    let envelope = std::fs::metadata(
        pi_jev::config::jev_dir_for(agent_dir).join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME)),
    )
    .ok()
    .map(|meta| {
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        (modified, meta.len())
    })
    .unwrap_or((0, 0));
    // Only hashes leave this scope; changing an already-present env key must
    // invalidate the old client just like replacing the saved envelope.
    let env_fingerprint = |name: &str| std::env::var(name).ok()
        .map(|value| pi_jev::credential::SecretString::new(value).key_fingerprint())
        .unwrap_or_default();
    format!(
        "cheap:{transport_selector}:{modified}:{len}:{typesafe}:{jev}",
        modified = envelope.0,
        len = envelope.1,
        typesafe = env_fingerprint(pi_jev::config::ENV_TYPESAFE_API_KEY),
        jev = env_fingerprint(pi_jev::config::ENV_JEV_API_KEY),
    )
}

/// Binding credential order (DESIGN.md 3.1): saved credential first, then
/// env `TYPESAFE_API_KEY`, then alias `JEV_API_KEY`. The value is wrapped in
/// a SecretString and never logged, stored or copied elsewhere.
fn load_credential() -> Option<pi_jev::credential::SecretString> {
    if let Some(saved) = load_saved_credential() {
        return Some(saved);
    }
    for env_name in ["TYPESAFE_API_KEY", "JEV_API_KEY"] {
        if let Ok(value) = std::env::var(env_name) {
            let trimmed = value.trim().to_string();
            if !trimmed.is_empty() {
                return Some(pi_jev::credential::SecretString::new(trimmed));
            }
        }
    }
    None
}

/// Selects the transport per settings. Mock selection is a development/test
/// aid and only honored in debug builds; production release builds always
/// use the real endpoint path (saved credential, then env fallback).
fn build_transport(
    settings: &JevSettings,
) -> (
    Arc<dyn pi_jev::types::Transport>,
    pi_jev::credential::SecretString,
    String,
) {
    let synthetic_credential =
        || pi_jev::credential::SecretString::new("jev-mock-credential-not-a-real-key");
    // An UNKNOWN transport value falls through to the real endpoint below
    // (fail toward honesty, never toward a silent mock); a warning-level note
    // is unnecessary because /jev status shows the effective source.
    if cfg!(debug_assertions) {
        match settings.transport.as_deref() {
            Some("mock") => {
                return (
                    Arc::new(MockJevTransport::all_valid()),
                    synthetic_credential(),
                    "mock".to_string(),
                );
            }
            // Development-only hostile variant: valid, maximum-confidence
            // answers chosen adversarially. Proves nothing Jev says can act.
            Some("mock-hostile") => {
                return (
                    Arc::new(HostileTestTransport),
                    synthetic_credential(),
                    "mock-hostile".to_string(),
                );
            }
            // Test-only: answers arrive AFTER the turn boundary (stale) and
            // can only ever become applied=false records.
            Some("mock-delayed") => {
                return (
                    Arc::new(MockJevTransport::scripted(vec![
                        pi_jev::mock::MockStep::SlowResponse { delay_ms: 600 },
                    ])),
                    synthetic_credential(),
                    "mock-delayed".to_string(),
                );
            }
            // Test-only: protocol-violating (malformed) answers. The client
            // validation rejects them; they land as logged skips and cannot
            // influence the agent loop.
            Some("mock-malformed") => {
                return (
                    Arc::new(MockJevTransport::scripted(vec![
                        pi_jev::mock::MockStep::Body(TRUNCATED_MALFORMED_BODY.to_string()),
                    ])),
                    synthetic_credential(),
                    "mock-malformed".to_string(),
                );
            }
            _ => {}
        }
    }
    // The real endpoint needs a credential; without one, comparison requests
    // fail closed to logged skips.
    match load_credential() {
        Some(secret) => {
            let fingerprint = format!("saved:{}", secret.key_fingerprint());
            match pi_jev::client::JevHttpTransport::new(
                pi_jev::client::JevLimits::default(),
                secret.clone(),
            ) {
                Ok(transport) => (Arc::new(transport), secret, fingerprint),
                Err(_) => (
                    Arc::new(NoCredentialTransport),
                    synthetic_credential(),
                    "unavailable".to_string(),
                ),
            }
        }
        None => (
            Arc::new(NoCredentialTransport),
            synthetic_credential(),
            "no_credential".to_string(),
        ),
    }
}

/// Development/test-only transport: valid-shaped, maximum-confidence
/// adversarial answers (mock-hostile). Proves nothing Jev says can act.
/// Selected only via the settings `transport` override; never by default.
struct HostileTestTransport;

impl pi_jev::types::Transport for HostileTestTransport {
    fn post(
        &self,
        request: &pi_jev::types::SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>>
    {
        let response = hostile_response_for(request);
        Box::pin(async move { Ok(response) })
    }
}

/// Deterministic truncated payload for the mock-malformed test transport.
const TRUNCATED_MALFORMED_BODY: &str =
    r#"{"model":"jev-latest","answers":{"task_classification.0":{"#;

/// Loads the saved credential through the agent-dir store. The value is
/// wrapped and never logged, stored or copied elsewhere; `None` when absent
/// or the store is unavailable (fail closed).
fn load_saved_credential() -> Option<pi_jev::credential::SecretString> {
    let store = pi_jev::credential::default_credential_store(get_agent_dir());
    match store.get(pi_jev::config::DEFAULT_KEY_ID) {
        Ok(Some(key)) => Some(pi_jev::credential::SecretString::new(key)),
        _ => None,
    }
}

/// Transport used when no credential is configured: fails closed to a logged
/// skip, never a network call.
struct NoCredentialTransport;

impl pi_jev::types::Transport for NoCredentialTransport {
    fn post(
        &self,
        _request: &pi_jev::types::SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>>
    {
        Box::pin(async {
            // The store is fine; there is simply no credential yet, so the
            // skip reason must say exactly that.
            Err(pi_jev::error::JevError::MissingCredential)
        })
    }
}

// ---------------------------------------------------------------------------
// Internal extension construction
// ---------------------------------------------------------------------------

/// Build the programmatic internal observer Extension. Handlers mirror the
/// runner's `ExtensionHandler` signature exactly.
fn build_observer_extension(core: Arc<JevBridgeCore>) -> SharedExtension {
    let mut handlers: HashMap<String, Vec<ExtensionHandler>> = HashMap::new();
    for event_type in JEV_EVENTS {
        let handler = make_handler(Arc::clone(&core), event_type);
        handlers
            .entry(event_type.to_string())
            .or_default()
            .push(handler);
    }
    // The Active handler ([`JEV_ACTIVE_EVENT`]) is NOT registered here.
    // [`sync_active_handlers`] installs it when some session is actually Active,
    // because its presence alone changes retry behavior for the whole process.
    // The caller stores the extension on the core via
    // [`JevBridgeCore::attach_extension`].
    Arc::new(Mutex::new(Extension {
        path: JEV_OBSERVER_PATH.to_string(),
        resolved_path: JEV_OBSERVER_PATH.to_string(),
        source_info: crate::core::source_info::create_synthetic_source_info(
            JEV_OBSERVER_PATH,
            &crate::core::source_info::SyntheticSourceInfoOptions {
                source: "internal".to_string(),
                ..Default::default()
            },
        ),
        handlers,
        tools: HashMap::new(),
        message_renderers: HashMap::new(),
        commands: indexmap::IndexMap::new(),
        flags: HashMap::new(),
        shortcuts: HashMap::new(),
    }))
}

/// Build one observe-only handler. Off cost: one cheap mode check, then
/// return `None` (the runner never sees a result from Jev).
fn make_handler(
    core: Arc<JevBridgeCore>,
    handler_event: &'static str,
) -> ExtensionHandler {
    Arc::new(
        move |event: ExtensionEvent,
              ctx: Arc<dyn crate::core::extensions::types::ExtensionContext>| {
            let core = Arc::clone(&core);
            Box::pin(async move {
                let session_id = ctx.session_manager().get_session_id();
                let settings = load_settings_cached();
                let mode = settings.effective_mode(&session_id);
                let _ = core.run_metrics.observe(&session_id, &event, mode, settings.effective_compaction_enabled(&session_id), settings.effective_features(&session_id));
                if handler_event == "session_shutdown" {
                    crate::core::jev_compaction::clear_session_status(&session_id);
                    core.drop_session_work(&session_id);
                    return None::<Value>;
                }
                if mode.is_enabled() || load_settings_cached().effective_compaction_enabled(&session_id) { core.own_session(&session_id); }
                if !mode.is_enabled() {
                    // Off takes effect immediately: cancel/forget this
                    // session's queued and in-flight comparison work so no
                    // late result can surface after the mode changed.
                    core.drop_decision_work(&session_id);
                    if handler_event == "session_start" {
                        ctx.ui().set_status("jev".into(), Some(footer_status_text(&session_id)));
                    }
                    return None::<Value>;
                }
                // Bounded per-session bookkeeping (never leaves the
                // process except as a capped excerpt inside a snapshot).
                //
                // This runs in Active too: the Active decision at the provider
                // boundary asks the same questions about the same state, so it
                // needs the same task excerpt and observed tool names.
                match &event {
                    ExtensionEvent::Input(payload) => {
                        core.remember_task_text(&session_id, &payload.text);
                    }
                    ExtensionEvent::ToolCall(tool_call) => {
                        core.note_tool_call(&session_id, tool_call.tool_name());
                    }
                    ExtensionEvent::TurnStart(payload) => {
                        core.note_turn(&session_id, payload.turn_index as u64);
                    }
                    ExtensionEvent::SessionShutdown(_) => {
                        core.forget_session(&session_id);
                    }
                    _ => {}
                }
                core.note_observation(&session_id, &event);
                // The shared actionable bundle replaces the duplicate shadow turn-start call.
                if mode.allows_active() && handler_event == "turn_start" { return None::<Value>; }
                if !mode.allows_compare() {
                    core.observe_optional(&session_id, &event, &ctx, handler_event).await;
                    return None::<Value>;
                }
                let Some(observer) = core.observer(&session_id, ctx.ui()) else {
                    return None::<Value>;
                };
                if let Some((event_type, payload)) =
                    bridge_event(&core, &event, &ctx, &session_id, handler_event)
                {
                    observer.observe(&event_type, &payload);
                    ctx.ui().set_status("jev".into(), Some(footer_status_text(&session_id)));
                }
                None::<Value>
            })
        },
    )
}

/// Build the one Active-mode handler.
///
/// Registered in addition to the observe-only handlers, and only reached when
/// the effective mode allows Active. In Off and Compare it
/// returns the untouched payload, so the provider path is unchanged.
fn make_active_handler(core: Arc<JevBridgeCore>) -> ExtensionHandler {
    Arc::new(
        move |event: ExtensionEvent,
              ctx: Arc<dyn crate::core::extensions::types::ExtensionContext>| {
            let core = Arc::clone(&core);
            Box::pin(async move {
                let ExtensionEvent::BeforeProviderRequest(payload) = event else {
                    return None::<Value>;
                };
                let session_id = ctx.session_manager().get_session_id();
                if !core.effective_mode(Some(&session_id)).allows_active() {
                    return None::<Value>;
                }
                let Some(observer) = core.observer(&session_id, ctx.ui()) else {
                    return None::<Value>;
                };
                let mut params = payload.payload;
                // One decision boundary per provider request. The snapshot is
                // built from what the adapter already tracks plus the tool
                // catalog actually advertised in this request.
                let settings = load_settings_cached();
                let mut state = active_request_state(&core, &ctx, &session_id, &params, &settings);
                let features = settings.effective_features(&session_id);
                let tool_plan = features.tool_candidates.then(|| crate::core::jev_active::prepare_tool_pruning(
                    &params, core.task_excerpt(&session_id).as_deref().unwrap_or(""), &settings.filtering));
                if let Some(plan) = &tool_plan { state["state"]["optional_tools"] = plan.state["optional_tools"].clone(); }
                state["optional_min_confidence"] = json!(settings.filtering.min_confidence);
                state["optional_max_decision_age_ms"] = json!(settings.filtering.max_decision_age_ms);
                let policy = activation_policy(features);
                let call = observer.decide_active(&state, pi_jev::snapshot::SnapshotStage::TurnStart, &policy);
                let outcome = if let Some(signal) = ctx.signal() {
                    tokio::select! { biased;
                        _ = signal.cancelled() => { observer.cancel_decisions(&session_id); return None::<Value>; },
                        outcome = call => outcome,
                    }
                } else { call.await };
                let mut effects: BTreeMap<String, Vec<pi_jev::active::AppliedEffect>> =
                    BTreeMap::new();
                for decision in &outcome.decisions {
                    if !observer.can_apply(&outcome) || decision.turn != core.turn(&session_id)
                        || !decision.is_fresh(std::time::SystemTime::now(), &policy) { break; }
                    let before = request_action(&params);
                    let changes = crate::core::jev_active::apply_model_decision(
                        &mut params,
                        decision.category.as_str(),
                        &decision.value,
                        ctx.model().as_ref(),
                    );
                    if changes.is_empty() {
                        continue;
                    }
                    effects.insert(
                        decision.category.as_str().to_string(),
                        changes
                            .iter()
                            .map(|change| {
                                pi_jev::active::AppliedEffect::new(
                                    change.key.clone(),
                                    before.get(&change.key).cloned(),
                                    request_action(&params).get(&change.key).cloned(),
                                )
                            })
                            .collect(),
                    );
                }
                if let Some(plan) = &tool_plan {
                    if observer.can_apply(&outcome) {
                        let changes = plan.apply(&mut params, &outcome.decisions,
                            outcome.request_id.as_deref().unwrap_or(""), core.turn(&session_id));
                        let mut dropped: Vec<_> = outcome.decisions.iter().filter(|decision|
                            decision.category == pi_jev::types::DecisionCategory::ToolCandidates && decision.value == "drop").collect();
                        dropped.sort_by_key(|decision| decision.question_id.rsplit_once('.').and_then(|(_, suffix)| suffix.parse::<usize>().ok()));
                        for (decision, change) in dropped.into_iter().zip(changes) {
                            effects.insert(decision.question_id.clone(), vec![pi_jev::active::AppliedEffect::new(
                                "optional_tool", change.from, Some("omitted_for_request".to_string()))]);
                        }
                    }
                }
                if !observer.can_apply(&outcome) { effects.clear(); }
                observer.record_active_with_action(&outcome, &effects, &request_action(&params));
                ctx.ui()
                    .set_status("jev".into(), Some(footer_status_text(&session_id)));
                if effects.is_empty() {
                    // Nothing applied: hand the payload back untouched rather
                    // than claim a change the request never had.
                    None::<Value>
                } else {
                    Some(params)
                }
            })
        },
    )
}

fn decision_policy_generation(settings: &JevSettings, session_id: &str) -> String {
    json!([settings.effective_features(session_id), settings.filtering]).to_string()
}

pub fn compaction_policy_generation(settings: &JevSettings, session_id: &str) -> String {
    json!([settings.effective_compaction_enabled(session_id), settings.compaction]).to_string()
}

pub fn record_compaction_skip(ctx: Arc<dyn ExtensionContext>, reason: &str, stats: Value) {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_compaction_enabled(&session_id) { return; }
    let capture = pi_jev::scheduler::RequestContext {
        request_id:format!("jev-{}",uuid::Uuid::new_v4()), session_id:session_id.clone(),
        turn:bridge_for_session(&session_id).map(|core| core.turn(&session_id)).unwrap_or(0),
        stage:"compaction".to_string(),state_fingerprint:String::new(),
        state_schema_version:pi_jev::snapshot::STATE_SCHEMA_VERSION.to_string(),
        prompt_version:pi_jev::hooks::PROMPT_VERSION.to_string(), mode:settings.effective_mode(&session_id).as_str().to_string(),
        questions:Vec::new(),baselines:BTreeMap::new(),request_start_ts:pi_jev::client::utc_now_rfc3339(),
    };
    let records=std::path::PathBuf::from(get_agent_dir()).join("jev").join("records.jsonl");
    pi_jev::correlate::Correlator::new(records,0.7).record_compaction(&capture,None,0,true,None,&stats,Some(reason));
}

pub struct CompactionDecision {
    pub outcome: pi_jev::types::DecisionOutcome,
    observer: Arc<JevObserver>,
    boundary: pi_jev::hooks::ActiveDecideOutcome,
    signal: Option<tokio_util::sync::CancellationToken>,
}

impl CompactionDecision {
    pub fn can_apply(&self) -> bool {
        self.signal.as_ref().is_none_or(|signal| !signal.is_cancelled()) && self.observer.can_apply(&self.boundary)
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.boundary.terminal_reason.as_deref().or_else(|| self.outcome.skips.first().map(|(_, reason)| *reason))
    }

    pub fn record_compaction(&self, mut stats: Value, fallback: Option<&str>) {
        let Some(ctx) = &self.boundary.context else { return; };
        let reason = if self.can_apply() { self.fallback_reason().or(fallback) } else { Some("cancelled_or_policy_changed") };
        if self.outcome.usage.input_tokens > 0 { stats["jev_input_tokens"] = json!(self.outcome.usage.input_tokens); }
        if self.outcome.usage.output_tokens > 0 { stats["jev_output_tokens"] = json!(self.outcome.usage.output_tokens); }
        let attempts = self.boundary.raw.as_ref().map(|raw|raw.attempts).unwrap_or(u32::from(self.boundary.dispatched));
        let known = self.boundary.raw.is_some() || !self.boundary.dispatched;
        self.observer.correlator().record_compaction(ctx, self.outcome.response_model.as_deref(), attempts, known,
            self.boundary.duration_ms, &stats, reason);
    }
}

pub async fn decide_compaction(ctx: Arc<dyn ExtensionContext>, mut bundle: pi_jev::types::DecisionBundle,
    signal: Option<tokio_util::sync::CancellationToken>) -> Option<CompactionDecision> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_compaction_enabled(&session_id) || signal.as_ref().is_some_and(|signal| signal.is_cancelled()) { return None; }
    let core = bridge_for_session(&session_id)?;
    let observer = core.observer(&session_id, ctx.ui())?;
    let captured_generation = bundle.state.as_object_mut().and_then(|state| state.remove("_jev_policy_generation"))
        .and_then(|value| value.as_str().map(str::to_string))?;
    let payload = json!({"session_id":session_id, "turn":core.turn(&session_id), "state":bundle.state,
        "compaction_enabled":true, "policy_generation":captured_generation});
    let questions = bundle.questions.into_iter().map(|(question_id, spec)| pi_jev::evaluators::PreparedQuestion { question_id, spec }).collect();
    let call = observer.decide_independent(&payload, "compaction", questions);
    let boundary = if let Some(signal) = &signal {
        tokio::select! { biased; _ = signal.cancelled() => return None, result = call => result }
    } else { call.await };
    let outcome = boundary.raw.clone().unwrap_or_else(|| pi_jev::types::DecisionOutcome::skipped_all("unavailable"));
    Some(CompactionDecision { outcome, observer, boundary, signal })
}

async fn relevance_decision(ctx: &Arc<dyn ExtensionContext>, prepared: &crate::core::jev_retrieval::PreparedRelevance, settings: &JevSettings)
    -> Option<(Arc<JevObserver>, pi_jev::hooks::ActiveDecideOutcome, Vec<usize>)> {
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let core = bridge_for_session(&session_id)?;
    let observer = core.observer(&session_id, ctx.ui())?;
    let payload = json!({"session_id":session_id, "turn":core.turn(&session_id), "state":prepared.state,
        "baseline_action":prepared.action_metadata(&[]),
        "compaction_enabled":settings.effective_compaction_enabled(&session_id),
        "policy_generation":decision_policy_generation(settings, &session_id)});
    let questions = prepared.questions();
    if questions.is_empty() { return None; }
    if !mode.allows_active() {
        if mode.allows_compare() { observer.observe_prepared(&payload, "retrieval", questions); }
        return None;
    }
    let mut policy = activation_policy(settings.effective_features(&session_id));
    policy.min_confidence = settings.filtering.min_confidence;
    policy.max_decision_age = Duration::from_millis(settings.filtering.max_decision_age_ms);
    let call = observer.decide_prepared(&payload, "retrieval", questions, &policy);
    let outcome = if let Some(signal) = ctx.signal() {
        tokio::select! { biased;
            _ = signal.cancelled() => { observer.cancel_decisions(&session_id); return None; },
            outcome = call => outcome,
        }
    } else { call.await };
    let removals = if observer.can_apply(&outcome) && core.turn(&session_id) == outcome.turn {
        prepared.removals(&outcome.decisions, outcome.request_id.as_deref().unwrap_or(""), outcome.turn)
    } else { Vec::new() };
    Some((observer, outcome, removals))
}

fn relevance_effects(prepared: &crate::core::jev_retrieval::PreparedRelevance, removals: &[usize]) -> BTreeMap<String, Vec<pi_jev::active::AppliedEffect>> {
    prepared.candidate_indices.iter().enumerate().filter(|(_, index)| removals.contains(index)).map(|(ordinal, _)| {
        let id = format!("{}.{}", prepared.category.as_str(), ordinal);
        (id.clone(), vec![pi_jev::active::AppliedEffect::new(id, Some("included".to_string()), Some("omitted_for_request".to_string()))])
    }).collect()
}

pub async fn filter_memory_candidates(ctx: Arc<dyn ExtensionContext>, query: &str, hits: Vec<MemoryHit>) -> Vec<MemoryHit> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_mode(&session_id).is_enabled() || !settings.effective_features(&session_id).memory_relevance { return hits; }
    let mut prepared = crate::core::jev_retrieval::prepare_memory(&hits, query);
    prepared.configure(&settings.filtering);
    let Some((observer, outcome, mut removals)) = relevance_decision(&ctx, &prepared, &settings).await else { return hits; };
    if !observer.can_apply(&outcome) { removals.clear(); }
    let effects = relevance_effects(&prepared, &removals);
    observer.record_active_with_action(&outcome, &effects, &prepared.action_metadata(&removals));
    crate::core::jev_retrieval::apply_memory(hits, &removals)
}

pub async fn filter_context_candidates(ctx: Arc<dyn ExtensionContext>, messages: Vec<Value>) -> Vec<Value> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    let messages = filter_code_search_candidates(&ctx, messages, &settings).await;
    if !settings.effective_mode(&session_id).is_enabled() || !settings.effective_features(&session_id).context_relevance { return messages; }
    let mut prepared = crate::core::jev_retrieval::prepare_context(&messages);
    prepared.configure(&settings.filtering);
    let Some((observer, outcome, mut removals)) = relevance_decision(&ctx, &prepared, &settings).await else { return messages; };
    if !observer.can_apply(&outcome) { removals.clear(); }
    let effects = relevance_effects(&prepared, &removals);
    observer.record_active_with_action(&outcome, &effects, &prepared.action_metadata(&removals));
    crate::core::jev_retrieval::apply_context(messages, &removals)
}

async fn filter_code_search_candidates(ctx: &Arc<dyn ExtensionContext>, mut messages: Vec<Value>, settings: &JevSettings) -> Vec<Value> {
    use futures::StreamExt;
    use crate::core::jev_code_search;
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let features = settings.effective_features(&session_id);
    if !mode.is_enabled() || !features.code_search_relevance { return messages; }
    if ctx.signal().is_some_and(|signal| signal.is_cancelled()) { return messages; }
    let Some(core) = bridge_for_session(&session_id) else { return messages; };
    let Some(observer) = core.observer(&session_id, ctx.ui()) else { return messages; };
    let query = crate::core::jev_retrieval::query_from_messages(&messages);
    let query = if query.is_empty() { core.task_excerpt(&session_id).unwrap_or_default() } else { query };
    let presentations = jev_code_search::prepare(&messages, &query, &settings.filtering);
    let generation = decision_policy_generation(settings, &session_id);
    let stamp = cheap_credential_stamp(settings);
    let started = tokio::time::Instant::now();
    for presentation in presentations {
        let cache_key = pi_jev::snapshot::fingerprint_of(&json!([session_id, core.turn(&session_id), mode.as_str(), generation, stamp, presentation.fingerprint]));
        if !mode.allows_active() && jev_code_search::already_observed(&cache_key) {
            continue;
        }
        let payloads: Vec<_> = presentation.batches.iter().map(|batch| json!({
            "session_id":session_id, "turn":core.turn(&session_id), "state":batch.state,
            "baseline_action":batch.action_metadata(&[]), "policy_generation":generation,
            "compaction_enabled":settings.effective_compaction_enabled(&session_id)
        })).collect();
        if !mode.allows_active() {
            for (batch, payload) in presentation.batches.iter().zip(&payloads) {
                observer.observe_prepared(payload, "code_search", batch.questions());
            }
            jev_code_search::remember_observation(cache_key);
            continue;
        }
        let policy = pi_jev::active::ActivationPolicy {
            enabled_categories: if features.code_search_filtering { [pi_jev::types::DecisionCategory::CodeSearchRelevance].into_iter().collect() } else { Default::default() },
            min_confidence: settings.filtering.min_confidence,
            max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
        };
        let signal = ctx.signal();
        let calls: Vec<_> = payloads.into_iter().enumerate().map(|(index, payload)| {
            let observer = observer.clone();
            let policy = policy.clone();
            let signal = signal.clone();
            let questions = presentation.batches[index].questions();
            async move {
                let remaining = Duration::from_millis(2500).saturating_sub(started.elapsed());
                let mut payload = payload;
                payload["decision_timeout_ms"] = json!(if signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
                    0
                } else { remaining.as_millis() as u64 });
                let call = observer.decide_prepared(&payload, "code_search", questions, &policy);
                tokio::pin!(call);
                let outcome = if let Some(signal) = signal {
                    tokio::select! { biased;
                        // Register the request token before cancelling it, including
                        // when cancellation races the first poll of this future.
                        result = &mut call => result,
                        _ = signal.cancelled() => {
                            observer.cancel_decisions(payload["session_id"].as_str().unwrap_or_default());
                            call.await
                        }
                    }
                } else { call.await };
                (index, outcome)
            }
        }).collect();
        let outcomes: Vec<_> = futures::stream::iter(calls).buffer_unordered(2).collect().await;
        let complete = outcomes.iter().all(|(_, outcome)|
            observer.can_apply(outcome) && outcome.turn == core.turn(&session_id) && outcome.unavailable.is_none()
                && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty()))
            && signal.as_ref().is_none_or(|signal| !signal.is_cancelled());
        let removals: Vec<_> = if complete && features.code_search_filtering {
            outcomes.iter().flat_map(|(index, outcome)| {
                let batch = &presentation.batches[*index];
                batch.removals(&outcome.decisions, outcome.request_id.as_deref().unwrap_or(""), outcome.turn)
            }).collect()
        } else { Vec::new() };
        let projection = presentation.project(&removals);
        let applied = if projection.is_some() { removals } else { Vec::new() };
        for (index, outcome) in &outcomes {
            let batch = &presentation.batches[*index];
            observer.record_active_with_action(outcome, &relevance_effects(batch, &applied), &batch.action_metadata(&applied));
        }
        if complete {
            if let Some(content) = projection { messages[presentation.message_index]["content"] = content; }
        }
    }
    messages
}

fn observed_failure_kind(message: &Value) -> Option<pi_jev::observation::RetryFailureKind> {
    use pi_jev::observation::RetryFailureKind;
    for diagnostic in message.get("diagnostics")?.as_array()?.iter().take(32) {
        match diagnostic.get("type").and_then(Value::as_str) {
            Some("agent_lifecycle_failure") => return Some(RetryFailureKind::Fatal),
            Some("provider_stream_failure") => {
                let kind = diagnostic.get("details").and_then(|details| details.get("kind")).and_then(Value::as_str).unwrap_or("");
                return Some(RetryFailureKind::from_provider_kind(kind));
            }
            _ => {}
        }
    }
    None
}

fn activation_policy(features: pi_jev::config::JevFeatures) -> pi_jev::active::ActivationPolicy {
    use pi_jev::types::DecisionCategory;
    let enabled_categories = [
        (features.tool_requirement, DecisionCategory::ToolRequirement),
        (features.complexity, DecisionCategory::Complexity),
        (features.tool_candidates, DecisionCategory::ToolCandidates),
        (features.context_relevance, DecisionCategory::ContextRelevance),
        (features.memory_relevance, DecisionCategory::MemoryRelevance),
    ].into_iter().filter_map(|(enabled, category)| enabled.then_some(category)).collect();
    pi_jev::active::ActivationPolicy { enabled_categories, ..Default::default() }
}

fn request_action(params: &Value) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    if let Some(tools) = params.get("tools").and_then(Value::as_array) { result.insert("tools".to_string(), format!("count:{}", tools.len())); }
    if let Some((key, effort)) = crate::core::jev_active::reasoning_effort(params) {
        result.insert(key.to_string(), effort.to_string());
    }
    if params.get("tool_choice").is_some() { result.insert("tool_choice".to_string(), "present".to_string()); }
    result
}

fn bridge_for_session(session_id: &str) -> Option<Arc<JevBridgeCore>> {
    let cores: Vec<_> = live_bridges().lock().unwrap_or_else(|p| p.into_inner()).iter().filter_map(Weak::upgrade).collect();
    cores.iter().find(|core| core.sessions.lock().unwrap_or_else(|p| p.into_inner()).contains_key(session_id)).cloned()
}

/// Bounded state for one Active decision boundary at the provider edge.
///
/// Reuses the same field names the Compare path sends at `turn_start`, so both
/// modes ask the same questions of the same snapshot. The tool catalog is the
/// one this request actually advertises, which is what the tool-candidate and
/// tool-requirement evaluators need; the observed-tool history is the fallback.
fn active_request_state(
    core: &Arc<JevBridgeCore>,
    ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>,
    session_id: &str,
    params: &Value,
    settings: &JevSettings,
) -> Value {
    let advertised = advertised_tool_names(params);
    let observed = if advertised.is_empty() {
        core.observed_tools(session_id)
    } else {
        advertised
    };
    let model_id = ctx.model().map(|model| model.id.clone());
    json!({
        "session_id": session_id,
        "turn": core.turn(session_id),
        "model": model_id,
        "baseline_action": request_action(params),
        "compaction_enabled": settings.effective_compaction_enabled(session_id),
        "policy_generation": decision_policy_generation(settings, session_id),
        "state": {
            "features": settings.effective_features(session_id),
            "observation": core.observation(session_id),
            "user_text_excerpt": core.task_excerpt(session_id),
            "observed_tools": observed,
            "message_count": ctx.session_manager().get_entry_count(),
            "model": model_id,
            "model_allowlist": Vec::<String>::new(),
            "cwd_name": std::path::Path::new(&ctx.cwd())
                .file_name()
                .map(|name| name.to_string_lossy().to_string()),
        },
    })
}

/// Tool names advertised in one provider request body, in order, bounded.
///
/// Only `function.name` entries are read. A tool schema is public by
/// definition: it is already on its way to the model provider.
fn advertised_tool_names(params: &Value) -> Vec<String> {
    const MAX_ADVERTISED_TOOLS: usize = 16;
    let Some(tools) = params.get(crate::core::jev_active::TOOLS_KEY).and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for tool in tools {
        let name = tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| tool.get("name").and_then(Value::as_str));
        if let Some(name) = name {
            if !name.is_empty() && !names.iter().any(|seen| seen == name) {
                names.push(name.to_string());
            }
        }
        if names.len() >= MAX_ADVERTISED_TOOLS {
            break;
        }
    }
    names
}

/// Convert an `ExtensionEvent` into the bounded `(event_type, payload)` DTO
/// the transport-agnostic observer consumes. Everything here is bounded:
/// message text is excerpt-capped, tool args excerpted, no full transcripts.
fn bridge_event(
    core: &Arc<JevBridgeCore>,
    event: &ExtensionEvent,
    ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>,
    session_id: &str,
    handler_event: &'static str,
) -> Option<(String, Value)> {
    let _ = handler_event;
    let model_id = ctx.model().map(|model| model.id.clone());
    let message_count = ctx.session_manager().get_entry_count();
    let turn = core.turn(session_id);
    // No user-approved allowlist source exists in this build; category 6
    // records an explicit `no_model_allowlist` skip instead of inventing
    // candidates. The selected model id is still observed for the record.
    let allowlist: Vec<String> = Vec::new();
    let mut result = match event {
        ExtensionEvent::SessionStart(_) => Some((
            "session_start".to_string(),
            json!({ "session_id": session_id, "reason": "startup" }),
        )),
        ExtensionEvent::AgentStart => Some((
            "agent_start".to_string(),
            json!({ "session_id": session_id }),
        )),
        ExtensionEvent::TurnStart(payload) => Some((
            "turn_start".to_string(),
            json!({
                "session_id": session_id,
                "turn": payload.turn_index as u64,
                "model": model_id,
                "state": {
                    "user_text_excerpt": core.task_excerpt(session_id),
                    "observed_tools": core.observed_tools(session_id),
                    "message_count": message_count,
                    "model": model_id,
                    "model_allowlist": allowlist,
                    "cwd_name": std::path::Path::new(&ctx.cwd())
                        .file_name()
                        .map(|name| name.to_string_lossy().to_string()),
                },
            }),
        )),
        ExtensionEvent::Input(payload) => Some((
            "input".to_string(),
            json!({
                "session_id": session_id,
                "turn": turn,
                "state": {
                    "user_text_excerpt": pi_jev::redact::bounded_excerpt(&payload.text, 400),
                    "message_count": message_count,
                    "model": model_id,
                    "model_allowlist": allowlist,
                },
            }),
        )),
        ExtensionEvent::ToolCall(tool_call) => {
            let tool_name = tool_call.tool_name().to_string();
            let tool_call_id = tool_call.tool_call_id().to_string();
            let mut state = tool_call_observation(&tool_name, &tool_call_id, &allowlist);
            state["user_text_excerpt"] = json!(core.task_excerpt(session_id));
            Some((
                "tool_call".to_string(),
                json!({
                    "session_id": session_id,
                    "turn": turn,
                    "tool_name": tool_name,
                    "model": model_id,
                    "state": state,
                }),
            ))
        }
        ExtensionEvent::ToolExecutionStart(payload) => Some((
            "tool_execution_start".to_string(),
            json!({
                "session_id": session_id,
                "tool_name": pi_jev::snapshot::truncate_text(&payload.tool_name, 120).0,
            }),
        )),
        ExtensionEvent::ToolExecutionEnd(payload) => Some((
            "tool_execution_end".to_string(),
            json!({
                "session_id": session_id,
                "tool_name": pi_jev::snapshot::truncate_text(&payload.tool_name, 120).0,
                "is_error": payload.is_error,
            }),
        )),
        ExtensionEvent::MessageEnd(payload) => Some((
            "message_end".to_string(),
            json!({
                "session_id": session_id,
                "role": payload.message.get("role").cloned().unwrap_or(Value::Null),
            }),
        )),
        ExtensionEvent::ModelSelect(payload) => Some((
            "model_select".to_string(),
            json!({
                "session_id": session_id,
                "selected_model": payload.model.id.clone(),
                "model": payload.model.id.clone(),
                "state": {
                    "model_allowlist": allowlist,
                    "selected_model": payload.model.id,
                    "previous_model": payload
                        .previous_model
                        .as_ref()
                        .map(|model| model.id.clone()),
                },
            }),
        )),
        ExtensionEvent::TurnEnd(payload) => Some(("turn_end".to_string(), json!({
            "session_id": session_id, "turn": payload.turn_index as u64, "model": model_id,
            "state": { "result_excerpt": bounded_assistant_excerpt(&payload.message),
                "user_text_excerpt": core.task_excerpt(session_id), "model_allowlist": allowlist },
        }))),
        ExtensionEvent::AgentEnd(payload) => {
            // Bounded result summary: stop reason + excerpt of the last
            // assistant text; never full transcripts.
            let summary = summarize_agent_end(&payload.messages);
            Some((
                "agent_end".to_string(),
                json!({
                    "session_id": session_id,
                    "turn": turn,
                    "model": model_id,
                    "state": {
                        "result_excerpt": summary.result_excerpt,
                        "user_text_excerpt": core.task_excerpt(session_id),
                        "stop_reason": summary.stop_reason,
                        "message_count": message_count,
                        "model_allowlist": allowlist,
                    },
                }),
            ))
        }
        ExtensionEvent::SessionShutdown(payload) => Some((
            "session_shutdown".to_string(),
            json!({
                "session_id": session_id,
                "reason": payload.reason,
            }),
        )),
        _ => None,
    };
    if let Some((_, payload)) = result.as_mut() {
        let settings = load_settings_cached();
        payload["compaction_enabled"] = json!(settings.effective_compaction_enabled(session_id));
        if let Some(state) = payload.get_mut("state").and_then(Value::as_object_mut) {
            state.insert("features".to_string(), json!(settings.effective_features(session_id)));
            state.insert("observation".to_string(), json!(core.observation(session_id)));
        }
    }
    result
}

/// Observation state for one tool call.
///
/// Tool ARGUMENTS are never observed. Callers pass credentials, connection
/// strings and file bodies through tool input, and the tool-choice evaluators
/// only need tool identity. Serializing `input()` here - even truncated right
/// afterwards - would copy a potentially multi-megabyte payload and could
/// disclose a credential that sits in its first characters.
fn tool_call_observation(tool_name: &str, tool_call_id: &str, allowlist: &[String]) -> Value {
    json!({
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "args_omitted": true,
        "model_allowlist": allowlist,
    })
}

struct AgentEndSummary {
    result_excerpt: Option<String>,
    stop_reason: Option<String>,
}

/// Bounded, redacted summary of the final assistant message.
///
/// The excerpt is accumulated character by character and stops at the redaction
/// scan window, so a multi-megabyte final message is never cloned or joined in
/// full just to keep 400 characters. Redaction happens before the value reaches
/// the snapshot state.
fn summarize_agent_end(messages: &[Value]) -> AgentEndSummary {
    let last_assistant = messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"));
    let Some(message) = last_assistant else {
        return AgentEndSummary {
            result_excerpt: None,
            stop_reason: None,
        };
    };
    AgentEndSummary {
        result_excerpt: Some(bounded_assistant_excerpt(message)),
        stop_reason: message
            .get("stopReason")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// Incremental excerpt of one assistant message's text blocks.
fn bounded_assistant_excerpt(message: &Value) -> String {
    let mut collected = String::new();
    let mut budget = pi_jev::redact::MAX_SCAN_CHARS;
    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        for block in blocks {
            if budget == 0 {
                break;
            }
            let Some(text) = block.get("text").and_then(Value::as_str) else {
                continue;
            };
            if !collected.is_empty() && budget > 1 {
                collected.push(' ');
                budget -= 1;
            }
            for character in text.chars() {
                if budget == 0 {
                    break;
                }
                collected.push(character);
                budget -= 1;
            }
        }
    }
    pi_jev::redact::bounded_excerpt(&collected, 400)
}

/// High-confidence valid answers that pick the most "act now"-looking option
/// in every question (e.g. escalate, insufficient). These can only land in
/// records: no code path consumes them.
fn hostile_response_for(request: &pi_jev::types::SystemOneRequest) -> pi_jev::types::SystemOneResponse {
    let mut answers = std::collections::BTreeMap::new();
    for (id, spec) in &request.questions {
        let answer = match spec {
            pi_jev::types::QuestionSpec::Noul { .. } => pi_jev::types::Answer::Noul { noul: 1.0 },
            pi_jev::types::QuestionSpec::Choice { criteria, .. } => {
                let choice = criteria
                    .keys()
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "none".to_string());
                let mut probabilities = std::collections::BTreeMap::new();
                for key in criteria.keys() {
                    probabilities.insert(key.clone(), 0.0);
                }
                probabilities.insert(choice.clone(), 1.0);
                pi_jev::types::Answer::Choice {
                    choice,
                    probabilities,
                    confidence: 1.0,
                }
            }
            pi_jev::types::QuestionSpec::Score { criteria, .. } => {
                let mut probabilities = std::collections::BTreeMap::new();
                for index in 0..criteria.len() {
                    probabilities.insert(index.to_string(), 0.0);
                }
                if let Some(last) = criteria.len().checked_sub(1) {
                    probabilities.insert(last.to_string(), 1.0);
                }
                let weighted: f64 = probabilities
                    .iter()
                    .map(|(k, v)| k.parse::<usize>().map(|i| v * i as f64).unwrap_or(0.0))
                    .sum();
                let mut legend = std::collections::BTreeMap::new();
                for (index, level) in criteria.iter().enumerate() {
                    legend.insert(index.to_string(), level.clone());
                }
                pi_jev::types::Answer::Score {
                    score: weighted,
                    legend,
                    probabilities,
                    confidence: 1.0,
                }
            }
        };
        answers.insert(id.clone(), answer);
    }
    pi_jev::types::SystemOneResponse {
        model: "jev-mock-hostile/1".to_string(),
        answers,
        usage: pi_jev::types::Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    }
}

// ---------------------------------------------------------------------------
// Negative-capability unit tests (no subagent control, ever)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod no_subagent_control_tests {
    use super::*;

    #[test]
    fn credential_stamp_tracks_the_actual_named_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let settings = JevSettings::default();
        let before = credential_stamp_at(&settings, dir.path());
        let jev_dir = dir.path().join("jev");
        std::fs::create_dir_all(&jev_dir).unwrap();
        let envelope = jev_dir.join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME));
        std::fs::write(&envelope, b"synthetic-encrypted-envelope").unwrap();
        let stored = credential_stamp_at(&settings, dir.path());
        assert_ne!(before, stored);
        std::fs::write(&envelope, b"different-synthetic-encrypted-envelope").unwrap();
        assert_ne!(stored, credential_stamp_at(&settings, dir.path()));
        std::fs::remove_file(envelope).unwrap();
        assert_eq!(before, credential_stamp_at(&settings, dir.path()));
        assert!(!stored.contains("synthetic"));
    }

    /// The internal observer extension must expose NO callable capability
    /// surface into the runtime: zero tools, commands, shortcuts, flags and
    /// message renderers. Handlers on observe-only events only.
    #[test]
    fn observer_extension_has_no_mutable_capability_surface() {
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        let extension = build_observer_extension(core);
        let extension = extension.lock().unwrap();
        assert!(extension.tools.is_empty(), "Jev must register no tools");
        assert!(extension.commands.is_empty(), "Jev must register no commands");
        assert!(extension.shortcuts.is_empty(), "Jev must register no shortcuts");
        assert!(extension.flags.is_empty(), "Jev must register no flags");
        assert!(
            extension.message_renderers.is_empty(),
            "Jev must register no message renderers"
        );
        let mut mutating = 0usize;
        for event_type in extension.handlers.keys() {
            assert!(
                !event_type.starts_with("session_before_")
                    && !matches!(
                        event_type.as_str(),
                        "context"
                            | "before_provider_request"
                            | "before_agent_start"
                            | "tool_result"
                            | "user_bash"
                            | "message_update"
                    ),
                "Jev must never subscribe to mutating surface {event_type}"
            );
            let _ = &mut mutating;
        }
        // Every registered event is one of the documented observe-only set.
        for event_type in JEV_EVENTS {
            assert!(
                extension.handlers.contains_key(event_type),
                "expected handler for {event_type}"
            );
        }
    }

    /// The documented event list itself must stay free of mutating surfaces.
    #[test]
    fn jev_event_list_excludes_all_mutating_surfaces() {
        for event_type in JEV_EVENTS {
            assert!(
                !event_type.starts_with("session_before_"),
                "session_before_* is decision-returning: {event_type}"
            );
            assert!(
                !matches!(
                    event_type,
                    "context"
                        | "before_provider_request"
                        | "before_agent_start"
                        | "tool_result"
                        | "user_bash"
                        | "message_update"
                ),
                "mutating surface {event_type} must never be observed"
            );
        }
    }

    /// Active is operative: it wants the observer, keeps the exact per-session
    /// value (never coerced to Compare or Off), and is not a Compare mode.
    #[test]
    fn active_settings_select_active_and_never_compare() {
        let mut settings = JevSettings::default();
        settings.global_default = Some(JevMode::Active);
        settings.sessions.insert(
            "any-session".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Active),
                inherited_from: None,
                ..Default::default()
            },
        );
        assert!(settings.wants_observer());
        assert_eq!(settings.effective_mode("any-session"), JevMode::Active);
        assert!(!JevMode::Active.allows_compare());
        // A per-session override always wins over the global default, in both
        // directions, so an Active session never silently becomes Compare.
        settings.sessions.insert(
            "any-session".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Off),
                inherited_from: None,
                ..Default::default()
            },
        );
        assert_eq!(settings.effective_mode("any-session"), JevMode::Off);
    }

    /// The provider-request handler exists only while Active is requested.
    ///
    /// Its presence alone tells the runner that a request body may change,
    /// which also gates retry reuse of semantic edges. A process where nobody
    /// enabled Active must therefore keep the handler absent.
    #[test]
    fn active_handler_presence_follows_the_setting() {
        fn active_handlers(extension: &SharedExtension) -> usize {
            let guard = extension.lock().unwrap();
            guard
                .handlers
                .get(JEV_ACTIVE_EVENT)
                .map(|handlers| handlers.len())
                .unwrap_or(0)
        }

        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        // A default-Off build registers no provider-request handler at all.
        let extension = build_observer_extension(Arc::clone(&core));
        assert_eq!(active_handlers(&extension), 0);
        core.attach_extension(&extension);
        // Installing is idempotent; removing leaves an empty entry.
        core.set_active_handler(&core, true);
        assert_eq!(active_handlers(&extension), 1);
        core.set_active_handler(&core, true);
        assert_eq!(active_handlers(&extension), 1);
        core.set_active_handler(&core, false);
        assert_eq!(active_handlers(&extension), 0);
    }
}


#[cfg(test)]
mod observation_redaction_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn diagnostic_observation_uses_native_shape_without_error_text() {
        use pi_jev::observation::RetryFailureKind;
        let message=json!({"role":"assistant","stopReason":"error","errorMessage":"secret raw provider error",
            "diagnostics":[{"type":"provider_stream_failure","details":{"kind":"rate_limit","body":"do not copy"}}]});
        assert_eq!(observed_failure_kind(&message),Some(RetryFailureKind::RateLimited));
        let lifecycle=json!({"diagnostics":[{"type":"agent_lifecycle_failure","details":{"kind":"do not infer transient"}}]});
        assert_eq!(observed_failure_kind(&lifecycle),Some(RetryFailureKind::Fatal));
        assert_eq!(observed_failure_kind(&json!({"errorKind":"rate_limit"})),None);
        let mut many=vec![json!({"type":"other"});32];
        many.push(json!({"type":"provider_stream_failure","details":{"kind":"rate_limit"}}));
        assert_eq!(observed_failure_kind(&json!({"diagnostics":many})),None);
    }

    #[test]
    fn session_ownership_never_falls_back_to_an_unrelated_bridge() {
        let first=Arc::new(JevBridgeCore::new(JevSettings::default()));
        let second=Arc::new(JevBridgeCore::new(JevSettings::default()));
        first.own_session("ownership-first-unique");
        second.own_session("ownership-second-unique");
        {
            let mut bridges=live_bridges().lock().unwrap();
            bridges.push(Arc::downgrade(&first)); bridges.push(Arc::downgrade(&second));
        }
        assert!(Arc::ptr_eq(&bridge_for_session("ownership-first-unique").unwrap(),&first));
        assert!(Arc::ptr_eq(&bridge_for_session("ownership-second-unique").unwrap(),&second));
        first.forget_session("ownership-first-unique");
        assert!(bridge_for_session("ownership-first-unique").is_none());
        assert!(bridge_for_session("never-registered-unique").is_none());
    }

    /// Fixtures used across these tests: values that must never reach a
    /// SystemOne request built from a shadow observation.
    const SECRETS: [&str; 6] = [
        "sk-live-abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "hunter2-the-password",
        "dXNlcjpwYXNzd29yZA==",
        "AKIAIOSFODNN7EXAMPLE",
        "MIIEowIBAAKCAQEAprivatekeymaterial",
    ];

    fn assert_no_secret(payload: &Value) {
        let text = serde_json::to_string(payload).unwrap();
        for secret in SECRETS {
            assert!(!text.contains(secret), "{secret} reached the payload: {text}");
        }
    }

    #[test]
    fn tool_arguments_are_never_observed_or_serialized() {
        let state = tool_call_observation("bash", "call-1", &[]);
        assert_eq!(state["args_omitted"], json!(true));
        assert!(state.get("args_excerpt").is_none(), "{state}");
        // Source guard: serializing tool input anywhere in the PRODUCTION
        // bridge would reintroduce the raw-argument capture this repair
        // removed. The scan stops at the first `#[cfg(test)]` module, because
        // test code must be able to name the field it forbids. Tokens are
        // split so the guard cannot match its own source text.
        let source = include_str!("jev_bridge.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        let raw_args_field = concat!("args", "_exc", "erpt");
        let raw_input_call = concat!("tool_call", ".in", "put()");
        assert!(
            !production.contains(raw_args_field) && !production.contains(raw_input_call),
            "the bridge serializes raw tool arguments again"
        );
    }

    #[test]
    fn secret_bearing_task_text_is_redacted_before_the_payload() {
        let raw = "deploy with TYPESAFE_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz \
                  and Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz";
        let state = json!({ "user_text_excerpt": pi_jev::redact::bounded_excerpt(raw, 400) });
        assert_no_secret(&state);
        assert!(serde_json::to_string(&state).unwrap().contains(pi_jev::redact::REDACTED));
    }

    #[test]
    fn secret_bearing_final_message_is_redacted_in_the_end_of_turn_state() {
        let messages = vec![json!({
            "role": "assistant",
            "stopReason": "end_turn",
            "content": [
                { "type": "text", "text": "wrote the config" },
                { "type": "text", "text": "password=hunter2-the-password" },
                { "type": "text", "text": "token AKIAIOSFODNN7EXAMPLE ok" },
            ],
        })];
        let summary = summarize_agent_end(&messages);
        let excerpt = summary.result_excerpt.unwrap_or_default();
        for secret in SECRETS {
            assert!(!excerpt.contains(secret), "{secret} survived: {excerpt}");
        }
        assert!(excerpt.contains("wrote the config"), "{excerpt}");
    }

    #[test]
    fn a_multi_megabyte_final_message_is_bounded_without_a_full_copy() {
        // 4 MiB assistant message ending in a credential: the excerpt must stay
        // bounded and must not disclose the credential.
        let big = format!(
            "{}{}",
            "x".repeat(4 * 1024 * 1024),
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAprivatekeymaterial"
        );
        let messages = vec![json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": big }],
        })];
        let summary = summarize_agent_end(&messages);
        let excerpt = summary.result_excerpt.unwrap_or_default();
        assert!(excerpt.chars().count() <= 400);
        for secret in SECRETS {
            assert!(!excerpt.contains(secret), "{secret} survived: {excerpt}");
        }
    }

    #[test]
    fn observation_state_never_carries_a_raw_allowlist_or_transcript() {
        let allowlist: Vec<String> = Vec::new();
        let state = tool_call_observation("read_file", "call-2", &allowlist);
        assert_eq!(state["model_allowlist"], json!([]));
        assert_eq!(state["tool_name"], json!("read_file"));
    }
}

#[cfg(test)]
mod provider_action_tests {
    use super::*;
    #[test]
    fn telemetry_tracks_the_responses_field_that_actually_changes() {
        let mut params=json!({"reasoning":{"effort":"low","summary":"auto"}});
        let before=request_action(&params);
        let changes=crate::core::jev_active::apply_decision(&mut params,"complexity","high");
        let after=request_action(&params);
        assert_eq!(before[&changes[0].key],"low");
        assert_eq!(after[&changes[0].key],"medium");
        assert!(!after.contains_key("reasoning_effort"));
    }
}
