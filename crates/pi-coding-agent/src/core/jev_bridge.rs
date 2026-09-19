//! Jev comparison adapter for pi-coding-agent (lane B).
//!
//! Builds a programmatic internal `Extension` (see
//! `core/extensions/types.rs::Extension`) whose handlers observe agent events
//! and feed the transport-agnostic `pi_jev::hooks::JevObserver`. Registration
//! retains a dormant adapter so first-use Compare works without restart. Handlers
//! never return Jev output (always `None`), so nothing Jev produces can
//! re-enter the agent loop, and Off costs exactly one cheap mode check.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use pi_jev::config::{JevMode, JevSettings};
use pi_jev::hooks::JevObserver;
use pi_jev::mock::MockJevTransport;
use serde_json::{json, Value};

use crate::config::get_agent_dir;
use crate::core::extensions::types::SharedExtension;
use crate::core::extensions::types::{Extension, ExtensionEvent, ExtensionHandler};

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
    for core in cores {
        if let Some(build) = core.observer.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            if let Some(status) = build.observer.session_status(session_id) {
                return Some(status);
            }
        }
    }
    None
}

/// Metadata-only footer text. This function never creates a client or task.
pub fn footer_status_text(session_id: &str) -> String {
    let settings = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
    let mode = settings.effective_mode(session_id);
    let (color, label) = if mode != JevMode::Compare {
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
        } else { ("accent", "Jev Compare") }
    };
    crate::modes::interactive::theme::theme::theme().fg(color, &format!("\u{25cf} {label}"))
}

/// Settings-file cache TTL: one cheap stat/read per interval per process.
const SETTINGS_TTL: Duration = Duration::from_millis(250);

/// Events the observer extension subscribes to. Every handler is
/// observe-only and always returns `None`; no session_before_* event and no
/// mutating surface (`context`, `before_provider_request`) is touched.
/// `input` is bookkeeping-only (task text capture) and returns `None`, which
/// the input pipeline maps to "Continue" with the original text.
pub const JEV_EVENTS: [&str; 11] = [
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
/// writing settings so a mode change is visible immediately.
pub fn invalidate_settings_cache() {
    *settings_cache().lock().unwrap_or_else(|p| p.into_inner()) = None;
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
    let extension = build_observer_extension(core);
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
}

struct JevBridgeCore {
    /// Cached observer keyed by the credential fingerprint it was built
    /// with; a credential rotation (or transport change) rebuilds it.
    observer: Mutex<Option<ObserverBuild>>,
    sessions: Mutex<HashMap<String, SessionBook>>,
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
        book.last_task_excerpt = Some(pi_jev::snapshot::truncate_text(text, 400).0);
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

    /// Effective mode for one session (cheap cached settings read).
    fn effective_mode(&self, session_id: Option<&str>) -> JevMode {
        load_settings_cached().effective_mode(session_id.unwrap_or(""))
    }

    /// Lazily construct the observer on the first Compare-mode event, and
    /// rebuild it when the effective credential changes (rotation). In Off
    /// nothing is ever constructed.
    fn observer(&self, session_id: &str, ui: Arc<dyn crate::core::extensions::types::ExtensionUiContext>) -> Option<Arc<JevObserver>> {
        let settings = load_settings_cached();
        if settings.effective_mode(session_id) != JevMode::Compare {
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
        let system_one: Arc<dyn pi_jev::types::SystemOne> = match
            pi_jev::client::JevSystemOne::new(
                JevMode::Compare,
                credential,
                transport,
                pi_jev::client::JevLimits::default(),
                Arc::new(pi_jev::client::JevStats::default()),
            ) {
            Ok(client) => Arc::new(client),
            // Construction can only fail on an unusable credential here;
            // comparison requests then fail closed to logged skips and no
            // network object is used.
            Err(_) => Arc::new(pi_jev::client::DisabledSystemOne::new(JevMode::Compare)),
        };
        // The observer re-checks the effective mode itself; the gate reads the
        // same cached settings the handlers use (one cheap read per event).
        let observed_stamp = cheap.clone();
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
                if core.effective_mode(Some(&session_id)) != JevMode::Compare {
                    // Off takes effect immediately: cancel/forget this
                    // session's queued and in-flight comparison work so no
                    // late result can surface after the mode changed.
                    core.drop_session_work(&session_id);
                    if handler_event == "session_start" {
                        ctx.ui().set_status("jev".into(), Some(footer_status_text(&session_id)));
                    }
                    return None::<Value>;
                }
                let Some(observer) = core.observer(&session_id, ctx.ui()) else {
                    return None::<Value>;
                };
                // Bounded per-session bookkeeping (never leaves the
                // process except as a capped excerpt inside a snapshot).
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
    match event {
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
                    "user_text_excerpt": pi_jev::snapshot::truncate_text(&payload.text, 400).0,
                    "message_count": message_count,
                    "model": model_id,
                    "model_allowlist": allowlist,
                },
            }),
        )),
        ExtensionEvent::ToolCall(tool_call) => {
            let tool_name = tool_call.tool_name().to_string();
            let tool_call_id = tool_call.tool_call_id().to_string();
            // Bounded args excerpt (never the full input).
            let args_excerpt = pi_jev::snapshot::truncate_text(&tool_call.input().to_string(), 400).0;
            Some((
                "tool_call".to_string(),
                json!({
                    "session_id": session_id,
                    "turn": turn,
                    "tool_name": tool_name,
                    "model": model_id,
                    "state": {
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "args_excerpt": args_excerpt,
                        "model_allowlist": allowlist,
                    },
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
    }
}

struct AgentEndSummary {
    result_excerpt: Option<String>,
    stop_reason: Option<String>,
}

/// Bounded summary of the final assistant message (no full transcript copy).
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
    let text = message
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    AgentEndSummary {
        result_excerpt: Some(pi_jev::snapshot::truncate_text(&text, 400).0),
        stop_reason: message
            .get("stopReason")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
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

    /// Future-Active settings can never register anything: the resolver
    /// fails Active closed to Off, so `wants_observer` is false.
    #[test]
    fn active_settings_register_nothing() {
        let mut settings = JevSettings::default();
        settings.global_default = Some(JevMode::Active);
        settings.sessions.insert(
            "any-session".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Active),
                inherited_from: None,
            },
        );
        assert!(!settings.wants_observer());
        // The reserved mode stays representable for honest status; it can
        // never register (Compare-only gate) and the client refuses it.
        assert_eq!(settings.effective_mode("any-session"), JevMode::Active);
    }
}
