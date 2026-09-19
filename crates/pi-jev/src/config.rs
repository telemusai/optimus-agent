//! Mode + settings + credential-source resolution (DESIGN.md sections 0, 3.1, 10.2, 11).
//!
//! Precedence (binding, DESIGN.md 10.2):
//!   resolve_effective_mode(explicit_session, inherited_or_global_default)
//!       = explicit_session.or(global_default).unwrap_or(Off)
//! Explicit per-session Off/Compare ALWAYS wins. A global default of Off must NOT defeat an
//! explicit per-session Compare. API-key presence never influences any mode.
//!
//! Credential source order (binding, DESIGN.md 3.1): saved > TYPESAFE_API_KEY > JEV_API_KEY.
//! Key values are never written to the settings file; only key-presence metadata is.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Authoritative mode control (DESIGN.md sections 0/3.1). `/jev` is the only writer;
/// key presence never sets a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevMode {
    Off,
    Compare,
    /// Reserved. Selecting it does NOT enable anything; a later reviewed activation policy
    /// is required before any recommendation can act.
    Active,
}

impl JevMode {
    /// Stable wire/config name.
    pub fn as_str(self) -> &'static str {
        match self {
            JevMode::Off => "off",
            JevMode::Compare => "compare",
            JevMode::Active => "active",
        }
    }

    /// Parses a user-supplied mode name. `/jev on` means Compare for this release.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "false" | "no" | "disabled" => Some(JevMode::Off),
            "compare" | "comparison" | "on" | "true" | "yes" | "shadow" => Some(JevMode::Compare),
            "active" | "enable" => Some(JevMode::Active),
            _ => None,
        }
    }

    /// True when shadow evaluation may run.
    pub fn allows_compare(self) -> bool {
        matches!(self, JevMode::Compare)
    }

    /// True when the UI must explain that this option is reserved and disabled.
    pub fn is_reserved(self) -> bool {
        matches!(self, JevMode::Active)
    }

    /// Short label for the footer. Text accompanies the colour so colour is never the only cue.
    pub fn label(self) -> &'static str {
        match self {
            JevMode::Off => "Jev Off",
            JevMode::Compare => "Jev Compare",
            JevMode::Active => "Jev On (reserved)",
        }
    }

    /// One-line explanation used by the menu and by help output.
    pub fn description(self) -> &'static str {
        match self {
            JevMode::Off => "Jev is off: no shadow calls, no network, no overhead beyond this check.",
            JevMode::Compare => {
                "Jev Compare: recommendations are recorded for comparison only. Nothing Jev returns is applied; your model, tools, context and stopping behavior do not change."
            }
            JevMode::Active => {
                "Jev Active is reserved and disabled: acting on recommendations requires a separately reviewed activation policy. Selecting this does not enable anything."
            }
        }
    }
}

impl fmt::Display for JevMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Saved-credential key id used by both the client lane and the UI lane.
pub const DEFAULT_KEY_ID: &str = "typesafe";

/// Primary environment variable name.
pub const ENV_TYPESAFE_API_KEY: &str = "TYPESAFE_API_KEY";

/// Documented alias, consulted only when the primary variable is absent.
pub const ENV_JEV_API_KEY: &str = "JEV_API_KEY";

/// Agent-dir override used by the repo for isolated state.
pub const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";

/// Built-in default agent directory (mirrors the repo's `.prime/agent`).
pub const DEFAULT_AGENT_DIR_NAME: &str = ".prime/agent";

/// Settings file name inside the agent dir.
pub const SETTINGS_FILE_NAME: &str = "jev-settings.json";

/// Schema version of the persisted settings file.
pub const SETTINGS_SCHEMA_VERSION: u32 = 1;

/// Where the effective credential came from. Never carries the secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialSource {
    /// A key from the credential store. Wins over every environment variable.
    Saved,
    /// `TYPESAFE_API_KEY`.
    EnvTypesafe,
    /// `JEV_API_KEY` (alias; used only when `TYPESAFE_API_KEY` is absent).
    EnvJev,
    /// Nothing configured.
    None,
}

impl CredentialSource {
    /// Stable name for status output.
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialSource::Saved => "saved",
            CredentialSource::EnvTypesafe => ENV_TYPESAFE_API_KEY,
            CredentialSource::EnvJev => ENV_JEV_API_KEY,
            CredentialSource::None => "none",
        }
    }

    /// True when a key of this source exists.
    pub fn is_configured(self) -> bool {
        !matches!(self, CredentialSource::None)
    }
}

/// Which scope supplied the effective mode, for the UI "this chat vs new chats" line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeScope {
    /// An explicit value chosen for this session.
    Session,
    /// The global default for new sessions.
    GlobalDefault,
    /// The built-in default (Off).
    BuiltIn,
}

impl ModeScope {
    pub fn as_str(self) -> &'static str {
        match self {
            ModeScope::Session => "session",
            ModeScope::GlobalDefault => "global_default",
            ModeScope::BuiltIn => "built_in_default",
        }
    }
}

/// Result of mode resolution, including which scope decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeResolution {
    pub mode: JevMode,
    pub scope: ModeScope,
}

/// BINDING precedence (DESIGN.md 10.2): explicit session > inherited/global default > Off.
///
/// An explicit per-session `Off` wins over a global `Compare`, and an explicit per-session
/// `Compare` wins over a global `Off`. Credentials are not an input here at all.
pub fn resolve_effective_mode(
    explicit_session: Option<JevMode>,
    inherited_or_global_default: Option<JevMode>,
) -> JevMode {
    resolve_mode(explicit_session, inherited_or_global_default).mode
}

/// Same resolution with the deciding scope reported.
pub fn resolve_mode(
    explicit_session: Option<JevMode>,
    inherited_or_global_default: Option<JevMode>,
) -> ModeResolution {
    match explicit_session {
        Some(mode) => ModeResolution {
            mode,
            scope: ModeScope::Session,
        },
        None => match inherited_or_global_default {
            Some(mode) => ModeResolution {
                mode,
                scope: ModeScope::GlobalDefault,
            },
            None => ModeResolution {
                mode: JevMode::Off,
                scope: ModeScope::BuiltIn,
            },
        },
    }
}

/// BINDING credential order (DESIGN.md 3.1): saved > TYPESAFE_API_KEY > JEV_API_KEY.
pub fn resolve_credential_source(
    saved_present: bool,
    env_typesafe_present: bool,
    env_jev_present: bool,
) -> CredentialSource {
    if saved_present {
        CredentialSource::Saved
    } else if env_typesafe_present {
        CredentialSource::EnvTypesafe
    } else if env_jev_present {
        CredentialSource::EnvJev
    } else {
        CredentialSource::None
    }
}

/// Resolves the effective credential source from a saved-presence flag and env presence.
///
/// Convenience wrapper so callers do not re-implement the documented order.
pub fn resolve_credential_source_from_presence(
    saved_present: bool,
    env: EnvKeyPresence,
) -> CredentialSource {
    resolve_credential_source(saved_present, env.typesafe_api_key, env.jev_api_key)
}

/// One-line credential status for `/jev status`. Never contains the secret.
///
/// When both environment variables are present and no saved credential exists, the line says
/// explicitly that `TYPESAFE_API_KEY` wins and the alias is ignored. When a saved credential
/// exists, the line says the environment variables are ignored entirely.
pub fn credential_status_line(saved_present: bool, env: EnvKeyPresence) -> String {
    let source = resolve_credential_source_from_presence(saved_present, env);
    let mut line = match source {
        CredentialSource::Saved => "credential: saved".to_string(),
        CredentialSource::EnvTypesafe => {
            format!("credential: {ENV_TYPESAFE_API_KEY}")
        }
        CredentialSource::EnvJev => {
            format!("credential: {ENV_JEV_API_KEY} (alias; {ENV_TYPESAFE_API_KEY} absent)")
        }
        CredentialSource::None => "credential: none (set /jev to add one)".to_string(),
    };
    if env.has_conflict() && source != CredentialSource::Saved {
        line.push_str(&format!(
            "; both environment variables are set, {ENV_TYPESAFE_API_KEY} wins and {ENV_JEV_API_KEY} is ignored"
        ));
    }
    if env.has_conflict() && source == CredentialSource::Saved {
        line.push_str(&format!(
            "; environment variables are ignored while a saved credential exists ({ENV_TYPESAFE_API_KEY}, {ENV_JEV_API_KEY})"
        ));
    }
    line
}

/// Which environment variables are present. Booleans only; values are never read here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvKeyPresence {
    pub typesafe_api_key: bool,
    pub jev_api_key: bool,
}

impl EnvKeyPresence {
    /// Reads key PRESENCE from the process environment. Values are never copied or logged.
    pub fn from_env() -> Self {
        Self {
            typesafe_api_key: env_var_non_empty(ENV_TYPESAFE_API_KEY),
            jev_api_key: env_var_non_empty(ENV_JEV_API_KEY),
        }
    }

    /// True when the primary variable is present and the alias is also present.
    /// Status says so explicitly, because the primary wins.
    pub fn has_conflict(&self) -> bool {
        self.typesafe_api_key && self.jev_api_key
    }
}

fn env_var_non_empty(name: &str) -> bool {
    std::env::var(name).map(|value| !value.trim().is_empty()).unwrap_or(false)
}

/// Per-session mode override. Absent means "inherit / use the global default".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedSessionMode {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<JevMode>,
    /// Opaque parent session id when the value was inherited at child creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<String>,
}

/// Persisted settings. Contains NO secret material; only presence metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevSettings {
    pub schema_version: u32,
    /// Global default for new sessions. `None` means built-in Off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_default: Option<JevMode>,
    /// Per-session explicit overrides, keyed by opaque session id.
    #[serde(default)]
    pub sessions: std::collections::BTreeMap<String, PersistedSessionMode>,
    /// Advisory metadata only: whether a saved credential existed at last write.
    #[serde(default)]
    pub credential_configured: bool,
    /// Advisory metadata only: which source was effective at last write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_source: Option<CredentialSource>,
    /// Whether the first-use disclosure has been shown.
    #[serde(default)]
    pub disclosure_shown: bool,
    /// Development/test-only transport selector among the crate's built-in
    /// deterministic mock transports ("mock", "mock-hostile", "mock-delayed",
    /// "mock-malformed"). `None` (production default) uses the saved
    /// credential against the real endpoint; a mock never reads a credential.
    /// This selects a transport ONLY; it never changes a mode and never
    /// grants any capability (DESIGN.md 11/12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
}

impl Default for JevSettings {
    fn default() -> Self {
        Self {
            schema_version: SETTINGS_SCHEMA_VERSION,
            global_default: None,
            sessions: std::collections::BTreeMap::new(),
            credential_configured: false,
            credential_source: None,
            disclosure_shown: false,
            transport: None,
        }
    }
}

impl JevSettings {
    /// True when at least one Compare configuration survives mode resolution.
    ///
    /// Registration gate for the observer extension: an Active-only or Off
    /// configuration registers nothing (fail closed; DESIGN.md 10/12).
    pub fn wants_observer(&self) -> bool {
        if self.global_default == Some(JevMode::Compare) {
            return true;
        }
        self.sessions
            .values()
            .any(|entry| entry.mode == Some(JevMode::Compare))
    }

    /// Settings with an explicit global default.
    pub fn with_global_default(mode: JevMode) -> Self {
        Self {
            global_default: Some(mode),
            ..Self::default()
        }
    }

    /// Explicit per-session override, if any.
    pub fn session_mode(&self, session_id: &str) -> Option<JevMode> {
        self.sessions.get(session_id).and_then(|entry| entry.mode)
    }

    /// Sets an explicit per-session override.
    pub fn set_session_mode(&mut self, session_id: &str, mode: JevMode) {
        self.sessions
            .entry(session_id.to_string())
            .or_default()
            .mode = Some(mode);
    }

    /// Clears an explicit per-session override, so the session falls back to the default.
    pub fn clear_session_mode(&mut self, session_id: &str) {
        if let Some(entry) = self.sessions.get_mut(session_id) {
            entry.mode = None;
        }
    }

    /// Effective mode for a session. See `resolve_effective_mode`.
    pub fn effective_mode(&self, session_id: &str) -> JevMode {
        resolve_effective_mode(self.session_mode(session_id), self.global_default)
    }

    /// Effective mode with the deciding scope.
    pub fn effective_mode_with_scope(&self, session_id: &str) -> ModeResolution {
        resolve_mode(self.session_mode(session_id), self.global_default)
    }

    /// Records key presence metadata. Never stores a key value.
    pub fn set_credential_metadata(&mut self, configured: bool, source: CredentialSource) {
        self.credential_configured = configured;
        self.credential_source = Some(source);
    }

    /// True when the serialized form looks like it carries credential material.
    ///
    /// `JevSettings` has no field able to hold a key; this guard exists so a later field
    /// cannot silently introduce one, and it is asserted by the settings tests.
    pub fn looks_like_it_contains_a_secret(&self) -> bool {
        let Ok(serialized) = serde_json::to_string(self) else {
            return false;
        };
        let lower = serialized.to_ascii_lowercase();
        if lower.contains("bearer ") {
            return true;
        }
        const FORBIDDEN_KEYS: [&str; 7] = [
            "\"api_key\"",
            "\"apikey\"",
            "\"api-key\"",
            "\"secret\"",
            "\"token\"",
            "\"password\"",
            "\"authorization\"",
        ];
        if FORBIDDEN_KEYS.iter().any(|key| lower.contains(key)) {
            return true;
        }
        // Session ids stay opaque and id-shaped; anything else is treated as secret material.
        self.sessions
            .keys()
            .chain(self.sessions.values().filter_map(|entry| entry.inherited_from.as_ref()))
            .any(|id| !is_opaque_local_id(id))
    }
}

/// Accepts only opaque local ids (session ids): bounded, id-shaped, no whitespace.
fn is_opaque_local_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Child inheritance: the parent's EFFECTIVE mode is snapshotted at creation time.
///
/// An explicit child override wins and is recorded; otherwise the child stores the parent's
/// effective mode as its own explicit value, so a later global change cannot silently alter
/// an existing chat. (DESIGN.md section 0: existing chats must not silently change.)
pub fn inherit_mode(
    settings: &mut JevSettings,
    child_session_id: &str,
    parent_session_id: &str,
    explicit_child_override: Option<JevMode>,
) -> JevMode {
    let inherited = explicit_child_override.unwrap_or_else(|| settings.effective_mode(parent_session_id));
    let entry = settings.sessions.entry(child_session_id.to_string()).or_default();
    entry.mode = Some(inherited);
    entry.inherited_from = Some(parent_session_id.to_string());
    inherited
}

/// Agent dir for Jev state: the injected value wins, otherwise the documented env override,
/// otherwise `$HOME/.prime/agent`.
pub fn default_agent_dir(injected: Option<&Path>) -> PathBuf {
    if let Some(dir) = injected {
        return dir.to_path_buf();
    }
    if let Ok(value) = std::env::var(ENV_AGENT_DIR) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(DEFAULT_AGENT_DIR_NAME)
}

/// Jev state directory inside the agent dir.
pub fn jev_dir_for(agent_dir: impl AsRef<Path>) -> PathBuf {
    agent_dir.as_ref().join("jev")
}

/// Settings store bound to one agent dir. `agent_dir` is injected to keep it testable and to
/// keep every test's state inside its own temporary directory.
#[derive(Debug, Clone)]
pub struct JevSettingsStore {
    path: PathBuf,
}

impl JevSettingsStore {
    /// Store for an explicit agent dir.
    pub fn new(agent_dir: impl AsRef<Path>) -> Self {
        Self {
            path: jev_dir_for(agent_dir).join(SETTINGS_FILE_NAME),
        }
    }

    /// Store for an explicit settings file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Settings file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Loads settings. A missing file yields defaults; a corrupt file yields defaults and
    /// never panics (a broken settings file must not block startup).
    pub fn load(&self) -> JevSettings {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice::<JevSettings>(&bytes).unwrap_or_default(),
            Err(_) => JevSettings::default(),
        }
    }

    /// Saves settings. Refuses to write a file that appears to contain a secret.
    pub fn save(&self, settings: &JevSettings) -> Result<(), crate::error::JevError> {
        if settings.looks_like_it_contains_a_secret() {
            return Err(crate::error::JevError::CredentialStore {
                detail: "refusing to persist settings that appear to contain a secret".to_string(),
            });
        }
        let Some(parent) = self.path.parent() else {
            return Err(crate::error::JevError::config("settings path has no parent directory"));
        };
        std::fs::create_dir_all(parent).map_err(|error| {
            crate::error::JevError::config(format!("cannot create settings dir: {}", error.kind()))
        })?;
        let serialized = serde_json::to_vec_pretty(settings).map_err(|_| {
            crate::error::JevError::config("cannot serialize settings")
        })?;
        let temp = self.path.with_extension("json.tmp");
        std::fs::write(&temp, serialized).map_err(|error| {
            crate::error::JevError::config(format!("cannot write settings: {}", error.kind()))
        })?;
        std::fs::rename(&temp, &self.path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            crate::error::JevError::config(format!("cannot replace settings: {}", error.kind()))
        })
    }
}

/// Convenience: shared settings store for an agent dir.
pub fn settings_store(agent_dir: impl AsRef<Path>) -> Arc<JevSettingsStore> {
    Arc::new(JevSettingsStore::new(agent_dir))
}
