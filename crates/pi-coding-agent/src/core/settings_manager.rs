//! Port of packages/coding-agent/src/core/settings-manager.ts
//!
//! `Settings` is modelled as a JSON object (`serde_json::Map<String, Value>`)
//! rather than a struct: the TypeScript manager iterates keys dynamically, does
//! spread merges, and must preserve keys written by other tools. Unknown keys,
//! absent-vs-null, and key order therefore stay observable exactly as in the
//! reference. Typed getters/setters below are the port of the typed interface.
//!
//! `CONFIG_DIR_NAME` and `getAgentDir()` come from config.ts (slice ca-root);
//! the minimal local definitions here keep the dependency explicit until
//! `pi_coding_agent::config` lands.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::utils::atomic_file::{realpath_if_present_sync, write_file_atomic_sync, WriteFileAtomicOptions};
use crate::utils::store_lock::{lock_store_sync, read_store};

const RECENT_MODELS_LIMIT: usize = 20;
pub const DEFAULT_IDLE_EVICTION_MINUTES: f64 = 90.0;

/// `CONFIG_DIR_NAME` from config.ts (`package.json` `piConfig.configDir`).
/// Private: config.ts belongs to another slice (ca-root).
const CONFIG_DIR_NAME: &str = ".prime/agent";

/// `interface Settings` - keys are written verbatim, so this is a JSON object.
pub type Settings = Map<String, Value>;

pub type SummaryUpdatePolicySetting = String;
const SUMMARY_UPDATE_POLICY_OFF: &str = "off";
const SUMMARY_UPDATE_POLICY_CONSOLIDATE_REPEATED_V1: &str = "consolidate-repeated-v1";

pub type ModelToolOutputPolicySetting = String;
const MODEL_TOOL_OUTPUT_POLICY_OFF: &str = "off";
const MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1: &str = "repeated-large-text-v1";

pub type MermaidRenderingMode = String;
const MERMAID_RENDERING_MODE_OFF: &str = "off";
const MERMAID_RENDERING_MODE_FINAL: &str = "final";
const MERMAID_RENDERING_MODE_STREAMING: &str = "streaming";

pub type TransportSetting = pi_ai::types::Transport;
pub type ServiceTier = pi_ai::types::ServiceTier;

// ---------------------------------------------------------------------------
// Settings sub-objects
// ---------------------------------------------------------------------------

/// `interface CompactionSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_recent_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_callable: Option<bool>,
    /// Behavior-changing iterative summary prompt. Default: off until semantic quality is proven.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_update_policy: Option<SummaryUpdatePolicySetting>,
}

/// `interface BranchSummarySettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummarySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_prompt: Option<bool>,
}

/// `interface AutoRefineSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoRefineSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_interval: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cooldown_ms: Option<f64>,
}

/// Resolved `getAutoRefineSettings()` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedAutoRefineSettings {
    pub enabled: bool,
    pub turn_interval: f64,
    pub compact: bool,
    pub cooldown_ms: f64,
}

/// `interface ProviderRetrySettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderRetrySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<f64>,
}

/// `interface RetrySettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_delay_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRetrySettings>,
}

/// Resolved `getRetrySettings()` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedRetrySettings {
    pub enabled: bool,
    pub max_retries: f64,
    pub base_delay_ms: f64,
}

/// Resolved `getProviderRetrySettings()` record. `timeoutMs` stays optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedProviderRetrySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<f64>,
    pub max_retry_delay_ms: f64,
}

/// `interface TerminalSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_images: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_on_shrink: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_terminal_progress: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fullscreen: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fullscreen_mouse: Option<bool>,
}

/// `interface ImageSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_resize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_images: Option<bool>,
}

/// `interface MarkdownSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkdownSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_block_indent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mermaid: Option<MermaidRenderingMode>,
}

/// `interface BundledSkillsSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundledSkillsSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websearch: Option<bool>,
}

/// Resolved `getBundledSkills()` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedBundledSkills {
    pub websearch: bool,
}

/// `interface WarningSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WarningSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_extra_usage: Option<bool>,
}

/// `interface AgentTracesSettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTracesSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// `interface TelemetrySettings`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetrySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice_shown: Option<bool>,
}

/// `getCompactionSettings()` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: f64,
    pub keep_recent_tokens: f64,
    pub summary_update_policy: SummaryUpdatePolicySetting,
}

/// `getBranchSummarySettings()` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedBranchSummarySettings {
    pub reserve_tokens: f64,
    pub skip_prompt: bool,
}

/// `PackageSource` - string form loads all resources; object form filters them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PackageSource {
    Source(String),
    Filtered(FilteredPackageSource),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FilteredPackageSource {
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub themes: Option<Vec<String>>,
}

/// `McpServerConfig` - user-declared MCP servers (built-ins live in the ai/mcp catalog).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpServerConfig {
    Http(HttpMcpServerConfig),
    Stdio(StdioMcpServerConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HttpMcpServerConfig {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    /// Env var holding a static bearer token (skips OAuth).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_token_env_var: Option<String>,
    /// Use the generic OAuth login flow for this server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth: Option<bool>,
    /// Force-disable even when credentials exist.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_timeout_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StdioMcpServerConfig {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Environment variables resolved from the kernel environment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_timeout_ms: Option<f64>,
}

/// `number | "off"` for the global idle-eviction policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdleEvictionMinutes {
    Minutes(f64),
    Off,
}

pub type TreeFilterMode = String;
pub const TREE_FILTER_MODES: [&str; 5] = ["default", "no-tools", "user-only", "labeled-only", "all"];

/// `JSON.stringify` of a JS number keeps integers integral (`3`, not `3.0`).
fn json_number(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        Value::Number(serde_json::Number::from(value as i64))
    } else {
        serde_json::Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn is_plain_object(value: &Value) -> bool {
    matches!(value, Value::Object(_))
}

/// `JSON.stringify(settings, null, 2)`.
fn stringify_settings(settings: &Settings) -> String {
    serde_json::to_string_pretty(&Value::Object(settings.clone())).unwrap_or_else(|_| "{}".to_string())
}

/// The throw `SettingsManager.migrateSettings(JSON.parse(content))` raises for a
/// JSON document that is not an object: `"queueMode" in 5` is a TypeError
/// (settings-manager.ts:411-412, 428), so `tryLoadFromStorage` records
/// "failed to parse" instead of silently using defaults
/// (settings-manager.ts:419-423).
const NON_OBJECT_SETTINGS_ERROR: &str = "Cannot use 'in' operator to search for 'queueMode' in settings: JSON.parse result is not an object";

/// `JSON.parse` widened to the object shape the settings file must have.
///
/// A valid JSON value of another shape (`null`, `5`, `"x"`, `[]`) is an error
/// here, matching the `migrateSettings` TypeError it raises in the reference.
fn parse_settings(content: &str) -> Result<Settings, SettingsErrorValue> {
    let value: Value = serde_json::from_str(content).map_err(|error| SettingsErrorValue {
        message: error.to_string(),
    })?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(SettingsErrorValue::new(NON_OBJECT_SETTINGS_ERROR)),
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Rust stand-in for the TypeScript `Error` carried in `SettingsError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsErrorValue {
    pub message: String,
}

impl SettingsErrorValue {
    pub fn new(message: impl Into<String>) -> Self {
        SettingsErrorValue {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SettingsErrorValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SettingsErrorValue {}

pub type SettingsScope = String;
const SETTINGS_SCOPE_GLOBAL: &str = "global";
const SETTINGS_SCOPE_PROJECT: &str = "project";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
    pub scope: SettingsScope,
    pub error: SettingsErrorValue,
}

// ---------------------------------------------------------------------------
// Deep merge
// ---------------------------------------------------------------------------

/// Deep merge settings: project/overrides take precedence, nested objects merge recursively.
fn deep_merge_settings(base: &Settings, overrides: &Settings) -> Settings {
    let mut result = base.clone();

    for (key, override_value) in overrides {
        let base_value = base.get(key);
        // TypeScript skips `undefined` overrides; JSON input never carries them,
        // so every key present here (including an explicit `null`) is stored.
        if is_plain_object(override_value) && base_value.map(is_plain_object).unwrap_or(false) {
            // Shallow spread of the nested object, matching `{ ...baseValue, ...overrideValue }`.
            let mut merged = base_value.unwrap().as_object().cloned().unwrap_or_default();
            for (nested_key, nested_value) in override_value.as_object().unwrap() {
                merged.insert(nested_key.clone(), nested_value.clone());
            }
            result.insert(key.clone(), Value::Object(merged));
        } else {
            result.insert(key.clone(), override_value.clone());
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// `interface SettingsStorage`.
pub trait SettingsStorage: Send + Sync {
    /// `withLock(scope, fn)` - `fn` receives the current file content and returns
    /// the next content, or `None` to leave the file untouched.
    ///
    /// The reference `withLock` throws when the callback, the read or the write
    /// throws (settings-manager.ts:281-313, inside a `try` whose `finally`
    /// releases the lock). That throw has no Rust equivalent for a `FnMut`
    /// callback, so it is reported instead: the failure is parked with
    /// [`SettingsStorage::record_failure`] (what the reference `enqueueWrite`
    /// `.catch` turns into a settings error, settings-manager.ts:583-585) and
    /// `false` is returned.
    ///
    /// `true` means the callback ran and its content (if any) reached the file.
    fn with_lock(&self, scope: &str, update: &mut dyn FnMut(Option<&str>) -> Option<String>) -> bool;

    /// Records a failure raised while no `SettingsManager` could observe it.
    ///
    /// The reference has no analogue because its `withLock` throws synchronously
    /// into the caller (settings-manager.ts:289); here the failure is parked on
    /// this storage and replayed by the next
    /// [`SettingsManager::drain_errors`] for that scope.
    fn record_failure(&self, scope: &str, error: SettingsErrorValue);

    /// Takes the parked failures of `scope`, in the order they happened.
    fn take_failures(&self, scope: &str) -> Vec<SettingsErrorValue>;
}

/// `Failed to acquire settings lock for ${path}: ${error}` as a settings error.
fn lock_error(path: &str, error: &std::io::Error) -> SettingsErrorValue {
    SettingsErrorValue::new(format!("Failed to acquire settings lock for {path}: {error}"))
}

/// `FileSettingsStorage` - global settings at `<agentDir>/settings.json`,
/// project settings at `<cwd>/<CONFIG_DIR_NAME>/settings.json`.
pub struct FileSettingsStorage {
    global_settings_path: String,
    project_settings_path: String,
    failures: Mutex<Vec<SettingsError>>,
}

impl FileSettingsStorage {
    pub fn new(cwd: &str, agent_dir: &str) -> Self {
        FileSettingsStorage {
            global_settings_path: join_path(agent_dir, "settings.json"),
            project_settings_path: join_path(&join_path(cwd, CONFIG_DIR_NAME), "settings.json"),
            failures: Mutex::new(Vec::new()),
        }
    }

    /// Records a lock/read/write failure raised while no `SettingsManager`
    /// could observe it; the reference throws into `withLock`'s caller instead
    /// (settings-manager.ts:289) and the caller records it
    /// (settings-manager.ts:583-585).
    fn record_failure(&self, scope: &str, error: SettingsErrorValue) {
        self.failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(SettingsError {
                scope: scope.to_string(),
                error,
            });
    }

    fn take_failures(&self, scope: &str) -> Vec<SettingsErrorValue> {
        let mut parked = self
            .failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut taken: Vec<SettingsErrorValue> = Vec::new();
        parked.retain(|entry| {
            if entry.scope == scope {
                taken.push(entry.error.clone());
                false
            } else {
                true
            }
        });
        taken
    }

    fn path_for(&self, scope: &str) -> &str {
        if scope == SETTINGS_SCOPE_GLOBAL {
            &self.global_settings_path
        } else {
            &self.project_settings_path
        }
    }

    fn acquire_lock_sync_with_retry(path: &str) -> Result<std::fs::File, SettingsErrorValue> {
        lock_store_sync(path).map_err(|error| lock_error(path, &error))
    }

    fn read_current(&self, scope: &str, path: &str) -> Result<Option<String>, ()> {
        read_store(path).map_err(|error| {
            self.record_failure(scope, SettingsErrorValue::new(format!("Failed to read settings: {error}")));
        })
    }
}

impl SettingsStorage for FileSettingsStorage {
    fn record_failure(&self, scope: &str, error: SettingsErrorValue) {
        FileSettingsStorage::record_failure(self, scope, error);
    }

    fn take_failures(&self, scope: &str) -> Vec<SettingsErrorValue> {
        FileSettingsStorage::take_failures(self, scope)
    }

    fn with_lock(&self, scope: &str, update: &mut dyn FnMut(Option<&str>) -> Option<String>) -> bool {
        let path = realpath_if_present_sync(self.path_for(scope))
            .unwrap_or_else(|_| self.path_for(scope).to_string());
        let dir = parent_dir(&path);

        let mut release: Option<std::fs::File> = None;
        let file_exists = Path::new(&path).exists();
        if file_exists {
            match FileSettingsStorage::acquire_lock_sync_with_retry(&path) {
                Ok(guard) => release = Some(guard),
                Err(error) => {
                    // `proper-lockfile` throws out of `withLock`
                    // (settings-manager.ts:289); the caller turns that into a
                    // settings error (settings-manager.ts:419-423) and the file
                    // is left untouched.
                    self.record_failure(scope, error);
                    return false;
                }
            }
        }
        let Ok(current) = self.read_current(scope, &path) else {
            return false;
        };
        let mut next = update(current.as_deref());
        if next.is_some() {
            if !Path::new(&dir).exists() {
                let _ = std::fs::create_dir_all(&dir);
            }
            if release.is_none() {
                match FileSettingsStorage::acquire_lock_sync_with_retry(&path) {
                    Ok(guard) => release = Some(guard),
                    Err(error) => {
                        self.record_failure(scope, error);
                        return false;
                    }
                }
                // The first-write read ran unlocked; a racing first writer may have landed since.
                let Ok(current) = self.read_current(scope, &path) else {
                    return false;
                };
                if current.is_some() {
                    next = update(current.as_deref());
                }
            }
            if let Some(next_value) = next {
                // The TypeScript `writeFileAtomicSync` throws; the throw is what
                // `enqueueWrite`'s `.catch` turns into a settings error.
                if let Err(error) = write_file_atomic_sync(
                    &path,
                    &next_value,
                    WriteFileAtomicOptions {
                        mode: Some(0o600),
                        ..Default::default()
                    },
                ) {
                    self.record_failure(scope, SettingsErrorValue::new(error.to_string()));
                    return false;
                }
            }
        }
        drop(release);
        true
    }
}

/// `InMemorySettingsStorage`.
#[derive(Default)]
pub struct InMemorySettingsStorage {
    global: Mutex<Option<String>>,
    project: Mutex<Option<String>>,
    failures: Mutex<Vec<SettingsError>>,
}

impl InMemorySettingsStorage {
    pub fn new() -> Self {
        Self::default()
    }

}

impl SettingsStorage for InMemorySettingsStorage {
    fn record_failure(&self, scope: &str, error: SettingsErrorValue) {
        self.failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(SettingsError {
                scope: scope.to_string(),
                error,
            });
    }

    fn take_failures(&self, scope: &str) -> Vec<SettingsErrorValue> {
        let mut parked = self
            .failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut taken: Vec<SettingsErrorValue> = Vec::new();
        parked.retain(|entry| {
            if entry.scope == scope {
                taken.push(entry.error.clone());
                false
            } else {
                true
            }
        });
        taken
    }

    fn with_lock(&self, scope: &str, update: &mut dyn FnMut(Option<&str>) -> Option<String>) -> bool {
        let slot = if scope == SETTINGS_SCOPE_GLOBAL {
            &self.global
        } else {
            &self.project
        };
        let current = slot.lock().ok().and_then(|guard| guard.clone());
        let next = update(current.as_deref());
        if let Some(next) = next {
            if let Ok(mut guard) = slot.lock() {
                *guard = Some(next);
            }
        }
        true
    }
}

/// `path.dirname`.
fn parent_dir(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `path.join` for the two-part joins used by settings-manager.
fn join_path(base: &str, leaf: &str) -> String {
    let sep = std::path::MAIN_SEPARATOR;
    if base.is_empty() {
        return leaf.to_string();
    }
    if base.ends_with(sep) {
        format!("{base}{leaf}")
    } else {
        format!("{base}{sep}{leaf}")
    }
}

/// `getAgentDir()` from config.ts (slice ca-root) - private until that slice lands.
///
/// `const envDir = process.env[ENV_AGENT_DIR];` reads only the piConfig-derived
/// var (config.ts:523-528), so `PI_CODING_AGENT_DIR` is not a fallback.
fn get_agent_dir() -> String {
    let env_dir = std::env::var("PRIME_AGENT_CODING_AGENT_DIR").unwrap_or_default();
    if !env_dir.is_empty() {
        return expand_tilde_path(&env_dir);
    }
    let home = dirs::home_dir().unwrap_or_default();
    join_path(&home.to_string_lossy(), CONFIG_DIR_NAME)
}

/// `expandTildePath(path)` from config.ts.
fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        let home = dirs::home_dir().unwrap_or_default();
        return join_path(&home.to_string_lossy(), rest);
    }
    path.to_string()
}

// ---------------------------------------------------------------------------
// SettingsManager
// ---------------------------------------------------------------------------

/// Field-key constants: the strings written by `markModified` are the
/// TypeScript interface keys and are observable in the settings file.
mod keys {
    pub const ONBOARDING_SHOWN: &str = "onboardingShown";
    pub const ONBOARDING_COMPLETED: &str = "onboardingCompleted";
    pub const DEFAULT_PROVIDER: &str = "defaultProvider";
    pub const DEFAULT_MODEL: &str = "defaultModel";
    pub const RECENT_MODELS: &str = "recentModels";
    pub const DEFAULT_THINKING_LEVEL: &str = "defaultThinkingLevel";
    pub const DEFAULT_SERVICE_TIER: &str = "defaultServiceTier";
    pub const RLM_MAX_DEPTH: &str = "rlmMaxDepth";
    pub const IDLE_EVICTION_MINUTES: &str = "idleEvictionMinutes";
    pub const TRANSPORT: &str = "transport";
    pub const STEERING_MODE: &str = "steeringMode";
    pub const FOLLOW_UP_MODE: &str = "followUpMode";
    pub const THEME: &str = "theme";
    pub const COMPACTION: &str = "compaction";
    pub const MODEL_TOOL_OUTPUT_POLICY: &str = "modelToolOutputPolicy";
    pub const AUTO_REFINE: &str = "autoRefine";
    pub const AGENT_TRACES: &str = "agentTraces";
    pub const TELEMETRY: &str = "telemetry";
    pub const BRANCH_SUMMARY: &str = "branchSummary";
    pub const RETRY: &str = "retry";
    pub const HIDE_THINKING_BLOCK: &str = "hideThinkingBlock";
    pub const SHELL_PATH: &str = "shellPath";
    pub const QUIET_STARTUP: &str = "quietStartup";
    pub const SHELL_COMMAND_PREFIX: &str = "shellCommandPrefix";
    pub const NPM_COMMAND: &str = "npmCommand";
    pub const MCP_SERVERS: &str = "mcpServers";
    pub const PACKAGES: &str = "packages";
    pub const EXTENSIONS: &str = "extensions";
    pub const SKILLS: &str = "skills";
    pub const PROMPTS: &str = "prompts";
    pub const THEMES: &str = "themes";
    pub const ENABLE_SKILL_COMMANDS: &str = "enableSkillCommands";
    pub const BUNDLED_SKILLS: &str = "bundledSkills";
    pub const ENABLE_BUILTIN_SKILLS: &str = "enableBuiltinSkills";
    pub const TERMINAL: &str = "terminal";
    pub const IMAGES: &str = "images";
    pub const ENABLED_MODELS: &str = "enabledModels";
    pub const TREE_FILTER_MODE: &str = "treeFilterMode";
    pub const THINKING_BUDGETS: &str = "thinkingBudgets";
    pub const EDITOR_PADDING_X: &str = "editorPaddingX";
    pub const AUTOCOMPLETE_MAX_VISIBLE: &str = "autocompleteMaxVisible";
    pub const SHOW_HARDWARE_CURSOR: &str = "showHardwareCursor";
    pub const MARKDOWN: &str = "markdown";
    pub const WARNINGS: &str = "warnings";
    pub const SESSION_DIR: &str = "sessionDir";
    pub const ENABLED: &str = "enabled";
    pub const RESERVE_TOKENS: &str = "reserveTokens";
    pub const KEEP_RECENT_TOKENS: &str = "keepRecentTokens";
    pub const AGENT_CALLABLE: &str = "agentCallable";
    pub const SUMMARY_UPDATE_POLICY: &str = "summaryUpdatePolicy";
    pub const SKIP_PROMPT: &str = "skipPrompt";
    pub const NOTICE_SHOWN: &str = "noticeShown";
    pub const MAX_RETRIES: &str = "maxRetries";
    pub const BASE_DELAY_MS: &str = "baseDelayMs";
    pub const PROVIDER: &str = "provider";
    pub const TIMEOUT_MS: &str = "timeoutMs";
    pub const MAX_RETRY_DELAY_MS: &str = "maxRetryDelayMs";
    pub const SHOW_IMAGES: &str = "showImages";
    pub const CLEAR_ON_SHRINK: &str = "clearOnShrink";
    pub const SHOW_TERMINAL_PROGRESS: &str = "showTerminalProgress";
    pub const FULLSCREEN: &str = "fullscreen";
    pub const FULLSCREEN_MOUSE: &str = "fullscreenMouse";
    pub const AUTO_RESIZE: &str = "autoResize";
    pub const BLOCK_IMAGES: &str = "blockImages";
    pub const CODE_BLOCK_INDENT: &str = "codeBlockIndent";
    pub const MERMAID: &str = "mermaid";
    pub const WEBSEARCH: &str = "websearch";
    pub const TURN_INTERVAL: &str = "turnInterval";
    pub const COMPACT: &str = "compact";
    pub const COOLDOWN_MS: &str = "cooldownMs";
}

type NestedModifiedFields = BTreeMap<String, BTreeSet<String>>;

struct WriteTask {
    scope: SettingsScope,
    task: Box<dyn FnOnce() + Send + Sync>,
}

pub struct SettingsManager {
    storage: Arc<dyn SettingsStorage>,
    global_settings: Settings,
    project_settings: Settings,
    settings: Settings,
    runtime_overrides: Settings,
    /// Track global fields modified during session.
    modified_fields: BTreeSet<String>,
    /// Track global nested field modifications.
    modified_nested_fields: NestedModifiedFields,
    /// Track project fields modified during session.
    modified_project_fields: BTreeSet<String>,
    /// Track project nested field modifications.
    modified_project_nested_fields: NestedModifiedFields,
    /// Track if global settings file had parse errors.
    global_settings_load_error: Option<SettingsErrorValue>,
    /// Track if project settings file had parse errors.
    project_settings_load_error: Option<SettingsErrorValue>,
    /// Pending writes. TypeScript chains promises eagerly; this port keeps the
    /// same order and runs each task here at enqueue time, like the
    /// `.then()` continuation the reference enqueues.
    write_queue: Vec<WriteTask>,
    errors: Vec<SettingsError>,
}

impl SettingsManager {
    fn new(
        storage: Arc<dyn SettingsStorage>,
        initial_global: Settings,
        initial_project: Settings,
        global_load_error: Option<SettingsErrorValue>,
        project_load_error: Option<SettingsErrorValue>,
        initial_errors: Vec<SettingsError>,
    ) -> Self {
        let settings = deep_merge_settings(&initial_global, &initial_project);
        SettingsManager {
            storage,
            global_settings: initial_global,
            project_settings: initial_project,
            settings,
            runtime_overrides: Settings::new(),
            modified_fields: BTreeSet::new(),
            modified_nested_fields: NestedModifiedFields::new(),
            modified_project_fields: BTreeSet::new(),
            modified_project_nested_fields: NestedModifiedFields::new(),
            global_settings_load_error: global_load_error,
            project_settings_load_error: project_load_error,
            write_queue: Vec::new(),
            errors: initial_errors,
        }
    }

    /// Create a SettingsManager that loads from files.
    /// `agentDir` defaults to `getAgentDir()`.
    pub fn create(cwd: &str, agent_dir: Option<&str>) -> Self {
        let agent_dir = agent_dir
            .map(|dir| dir.to_string())
            .unwrap_or_else(get_agent_dir);
        let storage = Arc::new(FileSettingsStorage::new(cwd, &agent_dir));
        SettingsManager::from_storage(storage)
    }

    /// Create a SettingsManager from an arbitrary storage backend.
    pub fn from_storage(storage: Arc<dyn SettingsStorage>) -> Self {
        let global_load = SettingsManager::try_load_from_storage(storage.as_ref(), SETTINGS_SCOPE_GLOBAL);
        let project_load = SettingsManager::try_load_from_storage(storage.as_ref(), SETTINGS_SCOPE_PROJECT);
        let mut initial_errors: Vec<SettingsError> = Vec::new();
        if let Some(error) = global_load.error.clone() {
            initial_errors.push(SettingsError {
                scope: SETTINGS_SCOPE_GLOBAL.to_string(),
                error,
            });
        }
        if let Some(error) = project_load.error.clone() {
            initial_errors.push(SettingsError {
                scope: SETTINGS_SCOPE_PROJECT.to_string(),
                error,
            });
        }

        SettingsManager::new(
            storage,
            global_load.settings,
            project_load.settings,
            global_load.error,
            project_load.error,
            initial_errors,
        )
    }

    /// Create an in-memory SettingsManager (no file I/O).
    pub fn in_memory(settings: Settings) -> Self {
        let storage = Arc::new(InMemorySettingsStorage::new());
        let initial_settings = SettingsManager::migrate_settings(settings);
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| {
            Some(stringify_settings(&initial_settings))
        });
        SettingsManager::from_storage(storage)
    }

    fn load_from_storage(storage: &dyn SettingsStorage, scope: &str) -> Result<Settings, SettingsErrorValue> {
        let mut content: Option<String> = None;
        let completed = storage.with_lock(scope, &mut |current| {
            content = current.map(|value| value.to_string());
            None
        });

        // A failed lock or read surfaces as the error of the scope:
        // `loadFromStorage` throws there (settings-manager.ts:289) and
        // `tryLoadFromStorage` catches it (settings-manager.ts:419-423).
        if !completed {
            return Err(storage
                .take_failures(scope)
                .into_iter()
                .next()
                .unwrap_or_else(|| SettingsErrorValue::new("Failed to read settings")));
        }

        let Some(content) = content else {
            return Ok(Settings::new());
        };
        if content.is_empty() {
            return Ok(Settings::new());
        }
        let settings = parse_settings(&content)?;
        Ok(SettingsManager::migrate_settings(settings))
    }

    fn try_load_from_storage(
        storage: &dyn SettingsStorage,
        scope: &str,
    ) -> LoadResult {
        match SettingsManager::load_from_storage(storage, scope) {
            Ok(settings) => LoadResult {
                settings,
                error: None,
            },
            Err(error) => LoadResult {
                settings: Settings::new(),
                error: Some(error),
            },
        }
    }

    /// Migrate old settings format to new format.
    fn migrate_settings(mut settings: Settings) -> Settings {
        if settings.contains_key("queueMode") && !settings.contains_key("steeringMode") {
            if let Some(value) = settings.remove("queueMode") {
                settings.insert(keys::STEERING_MODE.to_string(), value);
            }
        }
        if !settings.contains_key(keys::TRANSPORT) {
            if let Some(Value::Bool(websockets)) = settings.get("websockets").cloned() {
                settings.insert(
                    keys::TRANSPORT.to_string(),
                    Value::String(if websockets { "websocket" } else { "sse" }.to_string()),
                );
                settings.remove("websockets");
            }
        }
        if is_plain_object(settings.get(keys::SKILLS).unwrap_or(&Value::Null)) {
            let skills_settings = settings
                .get(keys::SKILLS)
                .and_then(|value| value.as_object())
                .cloned()
                .unwrap_or_default();
            // `skillsSettings.enableSkillCommands !== undefined`: JSON has no
            // `undefined`, so any present key (including `null`) migrates.
            if let Some(enable) = skills_settings.get("enableSkillCommands").cloned() {
                if !settings.contains_key(keys::ENABLE_SKILL_COMMANDS) {
                    settings.insert(keys::ENABLE_SKILL_COMMANDS.to_string(), enable);
                }
            }
            match skills_settings.get("customDirectories") {
                Some(Value::Array(directories)) if !directories.is_empty() => {
                    settings.insert(keys::SKILLS.to_string(), Value::Array(directories.clone()));
                }
                _ => {
                    settings.remove(keys::SKILLS);
                }
            }
        }
        if is_plain_object(settings.get(keys::RETRY).unwrap_or(&Value::Null)) {
            let mut retry_settings = settings
                .get(keys::RETRY)
                .and_then(|value| value.as_object())
                .cloned()
                .unwrap_or_default();
            let provider_settings = match retry_settings.get(keys::PROVIDER) {
                Some(Value::Object(provider)) => Some(provider.clone()),
                _ => None,
            };
            let has_provider_delay = provider_settings
                .as_ref()
                .and_then(|provider| provider.get(keys::MAX_RETRY_DELAY_MS))
                .map(|value| !value.is_null())
                .unwrap_or(false);
            if let Some(max_delay) = retry_settings.get("maxDelayMs").and_then(value_to_f64) {
                if !has_provider_delay {
                    let mut provider = provider_settings.unwrap_or_default();
                    provider.insert(keys::MAX_RETRY_DELAY_MS.to_string(), json_number(max_delay));
                    retry_settings.insert(keys::PROVIDER.to_string(), Value::Object(provider));
                }
            }
            retry_settings.remove("maxDelayMs");
            settings.insert(keys::RETRY.to_string(), Value::Object(retry_settings));
        }

        match settings.get(keys::TELEMETRY).cloned() {
            Some(Value::Bool(enabled)) => {
                let mut telemetry = Map::new();
                telemetry.insert(keys::ENABLED.to_string(), Value::Bool(enabled));
                settings.insert(keys::TELEMETRY.to_string(), Value::Object(telemetry));
            }
            Some(value) if !is_plain_object(&value) => {
                settings.remove(keys::TELEMETRY);
            }
            _ => {}
        }

        if let Some(value) = settings.get(keys::MARKDOWN).cloned() {
            if !is_plain_object(&value) {
                settings.remove(keys::MARKDOWN);
            }
        }

        settings
    }
}

struct LoadResult {
    settings: Settings,
    error: Option<SettingsErrorValue>,
}

impl SettingsManager {
    pub fn get_global_settings(&self) -> Settings {
        self.global_settings.clone()
    }

    pub fn get_project_settings(&self) -> Settings {
        self.project_settings.clone()
    }

    /// `getSettings()` - the merged global + project view (plus runtime overrides).
    pub async fn reload(&mut self) {
        self.reload_sync();
    }

    pub(crate) fn reload_sync(&mut self) {
        self.flush_sync();
        let global_load = SettingsManager::try_load_from_storage(self.storage.as_ref(), SETTINGS_SCOPE_GLOBAL);
        match global_load.error {
            None => {
                self.global_settings = global_load.settings;
                self.global_settings_load_error = None;
            }
            Some(error) => {
                self.global_settings_load_error = Some(error.clone());
                self.record_error(SETTINGS_SCOPE_GLOBAL, error);
            }
        }

        self.modified_fields.clear();
        self.modified_nested_fields.clear();
        self.modified_project_fields.clear();
        self.modified_project_nested_fields.clear();

        let project_load = SettingsManager::try_load_from_storage(self.storage.as_ref(), SETTINGS_SCOPE_PROJECT);
        match project_load.error {
            None => {
                self.project_settings = project_load.settings;
                self.project_settings_load_error = None;
            }
            Some(error) => {
                self.project_settings_load_error = Some(error.clone());
                self.record_error(SETTINGS_SCOPE_PROJECT, error);
            }
        }

        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
    }

    /// Apply additional overrides on top of current settings.
    pub fn apply_overrides(&mut self, overrides: &Settings) {
        self.runtime_overrides = deep_merge_settings(&self.runtime_overrides, overrides);
        self.settings = deep_merge_settings(&self.settings, overrides);
    }

    /// Mark a global field as modified during this session.
    fn mark_modified(&mut self, field: &str, nested_key: Option<&str>) {
        self.modified_fields.insert(field.to_string());
        if let Some(nested_key) = nested_key {
            self.modified_nested_fields
                .entry(field.to_string())
                .or_default()
                .insert(nested_key.to_string());
        }
    }

    /// Mark a project field as modified during this session.
    fn mark_project_modified(&mut self, field: &str, nested_key: Option<&str>) {
        self.modified_project_fields.insert(field.to_string());
        if let Some(nested_key) = nested_key {
            self.modified_project_nested_fields
                .entry(field.to_string())
                .or_default()
                .insert(nested_key.to_string());
        }
    }

    fn record_error(&mut self, scope: &str, error: SettingsErrorValue) {
        self.errors.push(SettingsError {
            scope: scope.to_string(),
            error,
        });
    }

    fn clear_modified_scope(&mut self, scope: &str) {
        if scope == SETTINGS_SCOPE_GLOBAL {
            self.modified_fields.clear();
            self.modified_nested_fields.clear();
            return;
        }

        self.modified_project_fields.clear();
        self.modified_project_nested_fields.clear();
    }

    fn enqueue_write(&mut self, scope: &str, task: Box<dyn FnOnce() + Send + Sync>) {
        self.write_queue.push(WriteTask {
            scope: scope.to_string(),
            task,
        });
    }

    fn clone_modified_nested_fields(source: &NestedModifiedFields) -> NestedModifiedFields {
        source.clone()
    }

    fn persist_scoped_settings(
        storage: &Arc<dyn SettingsStorage>,
        scope: &str,
        snapshot_settings: &Settings,
        modified_fields: &BTreeSet<String>,
        modified_nested_fields: &NestedModifiedFields,
    ) {
        let failure_sink: Arc<dyn SettingsStorage> = Arc::clone(storage);
        storage.with_lock(scope, &mut |current| {
            let current_file_settings = match current {
                Some(current) if !current.is_empty() => {
                    match parse_settings(current) {
                        Ok(parsed) => SettingsManager::migrate_settings(parsed),
                        Err(error) => {
                            // settings-manager.ts:602-605 runs
                            // `SettingsManager.migrateSettings(JSON.parse(current))`
                            // inside the `withLock` callback, so the parse throw
                            // aborts the write and the corrupt file is preserved
                            // for repair; the throw is recorded by the
                            // `enqueueWrite` `.catch` (settings-manager.ts:583-585).
                            failure_sink.record_failure(scope, error);
                            return None;
                        }
                    }
                }
                _ => Settings::new(),
            };
            let mut merged_settings: Settings = current_file_settings.clone();
            for field in modified_fields {
                let Some(value) = snapshot_settings.get(field).cloned() else {
                    // JSON.stringify omits fields assigned undefined.
                    merged_settings.remove(field);
                    continue;
                };
                let nested_modified = modified_nested_fields.get(field);
                if let (Some(nested_modified), Value::Object(in_memory_nested)) = (nested_modified, &value) {
                    let mut merged_nested = match current_file_settings.get(field) {
                        Some(Value::Object(base)) => base.clone(),
                        _ => Map::new(),
                    };
                    for nested_key in nested_modified {
                        match in_memory_nested.get(nested_key) {
                            Some(value) => {
                                merged_nested.insert(nested_key.clone(), value.clone());
                            }
                            None => {
                                merged_nested.remove(nested_key);
                            }
                        }
                    }
                    merged_settings.insert(field.clone(), Value::Object(merged_nested));
                } else {
                    merged_settings.insert(field.clone(), value);
                }
            }

            Some(stringify_settings(&merged_settings))
        });
    }

    fn save(&mut self) {
        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);

        if let Some(load_error) = self.global_settings_load_error.clone() {
            self.record_error(
                SETTINGS_SCOPE_GLOBAL,
                SettingsErrorValue::new(format!(
                    "Global settings not saved: settings file failed to parse: {}",
                    load_error.message
                )),
            );
            return;
        }

        let snapshot_global_settings = self.global_settings.clone();
        let modified_fields = self.modified_fields.clone();
        let modified_nested_fields = SettingsManager::clone_modified_nested_fields(&self.modified_nested_fields);
        let storage = Arc::clone(&self.storage);

        self.enqueue_write(
            SETTINGS_SCOPE_GLOBAL,
            Box::new(move || {
                SettingsManager::persist_scoped_settings(
                    &storage,
                    SETTINGS_SCOPE_GLOBAL,
                    &snapshot_global_settings,
                    &modified_fields,
                    &modified_nested_fields,
                );
            }),
        );
        // settings-manager.ts:577-585 chains the write at microtask time
        // (`this.writeQueue = this.writeQueue.then(() => { task(); ... })`), so
        // the file is written when `save()` runs and does not wait for an
        // explicit `flush()`.
        self.flush_sync();
    }

    fn save_project_settings(&mut self, settings: Settings) {
        self.project_settings = settings;
        self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);

        if let Some(load_error) = self.project_settings_load_error.clone() {
            self.record_error(
                SETTINGS_SCOPE_PROJECT,
                SettingsErrorValue::new(format!(
                    "Project settings not saved: settings file failed to parse: {}",
                    load_error.message
                )),
            );
            return;
        }

        let snapshot_project_settings = self.project_settings.clone();
        let modified_fields = self.modified_project_fields.clone();
        let modified_nested_fields =
            SettingsManager::clone_modified_nested_fields(&self.modified_project_nested_fields);
        let storage = Arc::clone(&self.storage);
        self.enqueue_write(
            SETTINGS_SCOPE_PROJECT,
            Box::new(move || {
                SettingsManager::persist_scoped_settings(
                    &storage,
                    SETTINGS_SCOPE_PROJECT,
                    &snapshot_project_settings,
                    &modified_fields,
                    &modified_nested_fields,
                );
            }),
        );
        // settings-manager.ts:649-669 saves project settings through the same
        // eager `enqueueWrite` chain, so the write runs here too.
        self.flush_sync();
    }

    /// Runs the queued writes in order. A failing write is recorded as a
    /// settings error instead of propagating, like the promise `.catch`.
    pub async fn flush(&mut self) {
        self.flush_sync();
    }

    pub(crate) fn flush_sync(&mut self) {
        let queue = std::mem::take(&mut self.write_queue);
        for WriteTask { scope, task } in queue {
            self.run_write_task(&scope, task);
        }
    }

    /// Runs one queued write: it clears the modified tracking of its scope on
    /// success and records a settings error instead of propagating on failure,
    /// like the reference `enqueueWrite` `.then(...).catch(...)` pair
    /// (settings-manager.ts:578-585) whose `.catch` calls
    /// `this.recordError(scope, error)`.
    ///
    /// The reference chains the task at microtask time
    /// (`this.writeQueue = this.writeQueue.then(() => { task(); ... })`,
    /// settings-manager.ts:578-581), so a `save()`-triggered write reaches the
    /// file without any explicit `flush()` call. This port is synchronous:
    /// running the task here is that microtask continuation.
    fn run_write_task(&mut self, scope: &str, task: Box<dyn FnOnce() + Send + Sync>) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task()));
        match result {
            Ok(()) => {
                // A storage failure is the reference `.catch(recordError)`
                // (settings-manager.ts:583-585); the modified tracking is kept
                // because the `.then` that clears it only runs after `task()`
                // returned without throwing (settings-manager.ts:579-581).
                let failures = self.storage.take_failures(scope);
                if failures.is_empty() {
                    self.clear_modified_scope(scope);
                } else {
                    for error in failures {
                        self.record_error(scope, error);
                    }
                }
            }
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|value| value.to_string()))
                    .unwrap_or_else(|| "settings write failed".to_string());
                self.record_error(scope, SettingsErrorValue::new(message));
            }
        }
    }

    /// `drainErrors(scope?)` (settings-manager.ts:675-684). It also replays the
    /// failures parked on the storage by [`SettingsStorage::record_failure`],
    /// which is where this port keeps the errors the reference records on `this`
    /// from its `.catch` (settings-manager.ts:583-585).
    pub fn drain_errors(&mut self, scope: Option<&str>) -> Vec<SettingsError> {
        let Some(scope) = scope else {
            let mut drained: Vec<SettingsError> = Vec::new();
            for error in self.storage.take_failures(SETTINGS_SCOPE_GLOBAL) {
                drained.push(SettingsError {
                    scope: SETTINGS_SCOPE_GLOBAL.to_string(),
                    error,
                });
            }
            drained.extend(self.errors.drain(..));
            for error in self.storage.take_failures(SETTINGS_SCOPE_PROJECT) {
                drained.push(SettingsError {
                    scope: SETTINGS_SCOPE_PROJECT.to_string(),
                    error,
                });
            }
            return drained;
        };
        let mut drained: Vec<SettingsError> = self
            .storage
            .take_failures(scope)
            .into_iter()
            .map(|error| SettingsError {
                scope: scope.to_string(),
                error,
            })
            .collect();
        let (matching, rest): (Vec<SettingsError>, Vec<SettingsError>) =
            self.errors.drain(..).partition(|entry| entry.scope == scope);
        drained.extend(matching);
        self.errors = rest;
        drained
    }
}

/// `if (!map[key]) map[key] = {}` then return the object for in-place writes.
fn ensure_object<'a>(map: &'a mut Settings, key: &str) -> &'a mut Map<String, Value> {
    if !is_plain_object(map.get(key).unwrap_or(&Value::Null)) {
        map.insert(key.to_string(), Value::Object(Map::new()));
    }
    map.get_mut(key)
        .and_then(|value| value.as_object_mut())
        .expect("object just inserted")
}

fn nested_bool(settings: &Settings, key: &str, nested: &str) -> Option<bool> {
    settings
        .get(key)
        .and_then(|value| value.as_object())
        .and_then(|object| object.get(nested))
        .and_then(|value| value.as_bool())
}

fn nested_f64(settings: &Settings, key: &str, nested: &str) -> Option<f64> {
    settings
        .get(key)
        .and_then(|value| value.as_object())
        .and_then(|object| object.get(nested))
        .and_then(value_to_f64)
}

fn nested_string(settings: &Settings, key: &str, nested: &str) -> Option<String> {
    settings
        .get(key)
        .and_then(|value| value.as_object())
        .and_then(|object| object.get(nested))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn top_string(settings: &Settings, key: &str) -> Option<String> {
    settings.get(key).and_then(|value| value.as_str()).map(|value| value.to_string())
}

fn top_bool(settings: &Settings, key: &str) -> Option<bool> {
    settings.get(key).and_then(|value| value.as_bool())
}

fn top_f64(settings: &Settings, key: &str) -> Option<f64> {
    settings.get(key).and_then(value_to_f64)
}

fn top_string_array(settings: &Settings, key: &str) -> Vec<String> {
    settings
        .get(key)
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|value| value.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn env_flag(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

impl SettingsManager {
    // -- onboarding -------------------------------------------------------

    pub fn get_onboarding_shown(&self) -> bool {
        top_bool(&self.settings, keys::ONBOARDING_SHOWN)
            .or_else(|| top_bool(&self.settings, keys::ONBOARDING_COMPLETED))
            .unwrap_or(false)
    }

    pub fn set_onboarding_shown(&mut self, shown: bool) {
        self.global_settings
            .insert(keys::ONBOARDING_SHOWN.to_string(), Value::Bool(shown));
        self.mark_modified(keys::ONBOARDING_SHOWN, None);
        self.save();
    }

    // -- session dir / defaults -------------------------------------------

    pub fn get_session_dir(&self) -> Option<String> {
        let session_dir = top_string(&self.settings, keys::SESSION_DIR)?;
        if session_dir.is_empty() {
            return Some(session_dir);
        }
        if session_dir == "~" {
            return Some(
                dirs::home_dir()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if let Some(rest) = session_dir.strip_prefix("~/") {
            let home = dirs::home_dir().unwrap_or_default();
            return Some(join_path(&home.to_string_lossy(), rest));
        }
        Some(session_dir)
    }

    pub fn get_default_provider(&self) -> Option<String> {
        top_string(&self.settings, keys::DEFAULT_PROVIDER)
    }

    pub fn get_default_model(&self) -> Option<String> {
        top_string(&self.settings, keys::DEFAULT_MODEL)
    }

    pub fn set_default_provider(&mut self, provider: &str) {
        self.global_settings.insert(
            keys::DEFAULT_PROVIDER.to_string(),
            Value::String(provider.to_string()),
        );
        self.mark_modified(keys::DEFAULT_PROVIDER, None);
        self.save();
    }

    pub fn set_default_model(&mut self, model_id: &str) {
        self.global_settings.insert(
            keys::DEFAULT_MODEL.to_string(),
            Value::String(model_id.to_string()),
        );
        self.mark_modified(keys::DEFAULT_MODEL, None);
        self.save();
    }

    pub fn set_default_model_and_provider(&mut self, provider: &str, model_id: &str) {
        self.global_settings.insert(
            keys::DEFAULT_PROVIDER.to_string(),
            Value::String(provider.to_string()),
        );
        self.global_settings.insert(
            keys::DEFAULT_MODEL.to_string(),
            Value::String(model_id.to_string()),
        );
        self.mark_modified(keys::DEFAULT_PROVIDER, None);
        self.mark_modified(keys::DEFAULT_MODEL, None);
        self.record_model_use_internal(provider, model_id);
        self.mark_modified(keys::RECENT_MODELS, None);
        self.save();
    }

    pub fn get_recent_models(&self) -> Vec<String> {
        top_string_array(&self.settings, keys::RECENT_MODELS)
    }

    fn record_model_use_internal(&mut self, provider: &str, model_id: &str) {
        let key = format!("{provider}/{model_id}");
        let mut next: Vec<String> = vec![key.clone()];
        next.extend(self.get_recent_models().into_iter().filter(|item| *item != key));
        next.truncate(RECENT_MODELS_LIMIT);
        self.global_settings.insert(
            keys::RECENT_MODELS.to_string(),
            Value::Array(next.into_iter().map(Value::String).collect()),
        );
    }

    // -- queue modes / theme / thinking -----------------------------------

    pub fn get_steering_mode(&self) -> String {
        top_string(&self.settings, keys::STEERING_MODE)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "one-at-a-time".to_string())
    }

    pub fn set_steering_mode(&mut self, mode: &str) {
        self.global_settings
            .insert(keys::STEERING_MODE.to_string(), Value::String(mode.to_string()));
        self.mark_modified(keys::STEERING_MODE, None);
        self.save();
    }

    pub fn get_follow_up_mode(&self) -> String {
        top_string(&self.settings, keys::FOLLOW_UP_MODE)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "one-at-a-time".to_string())
    }

    pub fn set_follow_up_mode(&mut self, mode: &str) {
        self.global_settings
            .insert(keys::FOLLOW_UP_MODE.to_string(), Value::String(mode.to_string()));
        self.mark_modified(keys::FOLLOW_UP_MODE, None);
        self.save();
    }

    pub fn get_theme(&self) -> Option<String> {
        top_string(&self.settings, keys::THEME)
    }

    pub fn set_theme(&mut self, theme: &str) {
        self.global_settings
            .insert(keys::THEME.to_string(), Value::String(theme.to_string()));
        self.mark_modified(keys::THEME, None);
        self.save();
    }

    pub fn get_default_thinking_level(&self) -> Option<String> {
        top_string(&self.settings, keys::DEFAULT_THINKING_LEVEL)
    }

    pub fn set_default_thinking_level(&mut self, level: &str) {
        self.global_settings.insert(
            keys::DEFAULT_THINKING_LEVEL.to_string(),
            Value::String(level.to_string()),
        );
        self.mark_modified(keys::DEFAULT_THINKING_LEVEL, None);
        self.save();
    }

    /// `this.settings.defaultServiceTier ?? "default"` (settings-manager.ts:791):
    /// `??` only falls back on `null`/`undefined`, so `""` is returned as-is.
    pub fn get_default_service_tier(&self) -> String {
        top_string(&self.settings, keys::DEFAULT_SERVICE_TIER)
            .unwrap_or_else(|| "default".to_string())
    }

    /// `setDefaultServiceTier(serviceTier: ServiceTier)`: `None` is an
    /// `undefined` argument (key dropped by JSON.stringify), `Some(None)` is an
    /// explicit `null`, `Some(Some(tier))` is the tier string.
    pub fn set_default_service_tier(&mut self, service_tier: Option<Option<String>>) {
        match service_tier {
            None => {
                self.global_settings.remove(keys::DEFAULT_SERVICE_TIER);
            }
            Some(None) => {
                self.global_settings
                    .insert(keys::DEFAULT_SERVICE_TIER.to_string(), Value::Null);
            }
            Some(Some(tier)) => {
                self.global_settings
                    .insert(keys::DEFAULT_SERVICE_TIER.to_string(), Value::String(tier));
            }
        }
        self.mark_modified(keys::DEFAULT_SERVICE_TIER, None);
        self.save();
    }

    // -- rlm max depth / idle eviction / transport -------------------------

    pub fn get_rlm_max_depth(&self) -> Option<f64> {
        top_f64(&self.global_settings, keys::RLM_MAX_DEPTH)
    }

    pub fn set_rlm_max_depth(&mut self, max_depth: f64) {
        self.global_settings.insert(
            keys::RLM_MAX_DEPTH.to_string(),
            json_number(max_depth),
        );
        self.mark_modified(keys::RLM_MAX_DEPTH, None);
        self.save();
    }

    pub fn get_idle_eviction_minutes(&self) -> IdleEvictionMinutes {
        let value = self.global_settings.get(keys::IDLE_EVICTION_MINUTES);
        if let Some(Value::String(text)) = value {
            if text == "off" || text == "none" {
                return IdleEvictionMinutes::Off;
            }
        }
        match value.and_then(value_to_f64) {
            Some(number) if number.is_finite() && number > 0.0 => IdleEvictionMinutes::Minutes(number),
            _ => IdleEvictionMinutes::Minutes(DEFAULT_IDLE_EVICTION_MINUTES),
        }
    }

    pub fn set_idle_eviction_minutes(&mut self, value: IdleEvictionMinutes) {
        match value {
            IdleEvictionMinutes::Off => {
                self.global_settings.insert(
                    keys::IDLE_EVICTION_MINUTES.to_string(),
                    Value::String("off".to_string()),
                );
            }
            IdleEvictionMinutes::Minutes(minutes) => {
                if !minutes.is_finite() || minutes <= 0.0 {
                    panic!("Idle eviction minutes must be a positive number or off");
                }
                self.global_settings.insert(
                    keys::IDLE_EVICTION_MINUTES.to_string(),
                    json_number(minutes),
                );
            }
        }
        self.mark_modified(keys::IDLE_EVICTION_MINUTES, None);
        self.save();
    }

    /// `this.settings.transport ?? "auto"` (settings-manager.ts:826):
    /// `??` only falls back on `null`/`undefined`, so `""` is returned as-is.
    pub fn get_transport(&self) -> TransportSetting {
        top_string(&self.settings, keys::TRANSPORT)
            .unwrap_or_else(|| pi_ai::types::TRANSPORT_AUTO.to_string())
    }

    pub fn set_transport(&mut self, transport: TransportSetting) {
        self.global_settings
            .insert(keys::TRANSPORT.to_string(), Value::String(transport));
        self.mark_modified(keys::TRANSPORT, None);
        self.save();
    }
}

impl SettingsManager {
    // -- compaction --------------------------------------------------------

    pub fn get_compaction_enabled(&self) -> bool {
        nested_bool(&self.settings, keys::COMPACTION, keys::ENABLED).unwrap_or(true)
    }

    pub fn set_compaction_enabled(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::COMPACTION)
            .insert(keys::ENABLED.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::COMPACTION, Some(keys::ENABLED));
        self.save();
    }

    pub fn get_agent_traces_enabled(&self) -> bool {
        nested_bool(&self.settings, keys::AGENT_TRACES, keys::ENABLED).unwrap_or(false)
    }

    pub fn set_agent_traces_enabled(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::AGENT_TRACES)
            .insert(keys::ENABLED.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::AGENT_TRACES, Some(keys::ENABLED));
        self.save();
    }

    /// Telemetry is AND-ed across global, project and runtime scopes so any
    /// scope can opt out and none can opt back in.
    pub fn get_telemetry_enabled(&self) -> bool {
        let global_enabled = nested_bool(&self.global_settings, keys::TELEMETRY, keys::ENABLED).unwrap_or(true);
        let project_enabled =
            nested_bool(&self.project_settings, keys::TELEMETRY, keys::ENABLED).unwrap_or(true);
        let runtime_enabled =
            nested_bool(&self.runtime_overrides, keys::TELEMETRY, keys::ENABLED).unwrap_or(true);
        global_enabled && project_enabled && runtime_enabled
    }

    fn get_or_create_global_telemetry_settings(&mut self) -> &mut Map<String, Value> {
        if !is_plain_object(
            self.global_settings
                .get(keys::TELEMETRY)
                .unwrap_or(&Value::Null),
        ) {
            self.global_settings
                .insert(keys::TELEMETRY.to_string(), Value::Object(Map::new()));
        }
        self.global_settings
            .get_mut(keys::TELEMETRY)
            .and_then(|value| value.as_object_mut())
            .expect("telemetry object just inserted")
    }

    pub fn set_telemetry_enabled(&mut self, enabled: bool) {
        self.get_or_create_global_telemetry_settings()
            .insert(keys::ENABLED.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::TELEMETRY, Some(keys::ENABLED));
        self.save();
    }

    pub fn get_telemetry_notice_shown(&self) -> bool {
        nested_bool(&self.runtime_overrides, keys::TELEMETRY, keys::NOTICE_SHOWN)
            .or_else(|| nested_bool(&self.global_settings, keys::TELEMETRY, keys::NOTICE_SHOWN))
            .unwrap_or(false)
    }

    pub fn set_telemetry_notice_shown(&mut self, shown: bool) {
        self.get_or_create_global_telemetry_settings()
            .insert(keys::NOTICE_SHOWN.to_string(), Value::Bool(shown));
        self.mark_modified(keys::TELEMETRY, Some(keys::NOTICE_SHOWN));
        self.save();
    }

    // -- compaction details ------------------------------------------------

    pub fn get_compaction_reserve_tokens(&self) -> f64 {
        nested_f64(&self.settings, keys::COMPACTION, keys::RESERVE_TOKENS).unwrap_or(16384.0)
    }

    pub fn get_compaction_keep_recent_tokens(&self) -> f64 {
        nested_f64(&self.settings, keys::COMPACTION, keys::KEEP_RECENT_TOKENS).unwrap_or(20000.0)
    }

    pub fn get_compaction_agent_callable(&self) -> bool {
        nested_bool(&self.settings, keys::COMPACTION, keys::AGENT_CALLABLE).unwrap_or(true)
    }

    pub fn get_summary_update_policy(&self) -> SummaryUpdatePolicySetting {
        let configured = nested_string(&self.settings, keys::COMPACTION, keys::SUMMARY_UPDATE_POLICY);
        match configured.as_deref() {
            Some(SUMMARY_UPDATE_POLICY_CONSOLIDATE_REPEATED_V1) => {
                return SUMMARY_UPDATE_POLICY_CONSOLIDATE_REPEATED_V1.to_string()
            }
            Some(SUMMARY_UPDATE_POLICY_OFF) => return SUMMARY_UPDATE_POLICY_OFF.to_string(),
            _ => {}
        }
        if env_flag("PRIME_AGENT_SUMMARY_UPDATE_POLICY").as_deref()
            == Some(SUMMARY_UPDATE_POLICY_CONSOLIDATE_REPEATED_V1)
        {
            return SUMMARY_UPDATE_POLICY_CONSOLIDATE_REPEATED_V1.to_string();
        }
        SUMMARY_UPDATE_POLICY_OFF.to_string()
    }

    pub fn get_model_tool_output_policy(&self) -> ModelToolOutputPolicySetting {
        let configured = top_string(&self.settings, keys::MODEL_TOOL_OUTPUT_POLICY);
        match configured.as_deref() {
            Some(MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1) => {
                return MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1.to_string()
            }
            Some(MODEL_TOOL_OUTPUT_POLICY_OFF) => return MODEL_TOOL_OUTPUT_POLICY_OFF.to_string(),
            _ => {}
        }
        if env_flag("PRIME_AGENT_MODEL_TOOL_OUTPUT_POLICY").as_deref()
            == Some(MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1)
        {
            return MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1.to_string();
        }
        MODEL_TOOL_OUTPUT_POLICY_OFF.to_string()
    }

    pub fn get_compaction_settings(&self) -> ResolvedCompactionSettings {
        ResolvedCompactionSettings {
            enabled: self.get_compaction_enabled(),
            reserve_tokens: self.get_compaction_reserve_tokens(),
            keep_recent_tokens: self.get_compaction_keep_recent_tokens(),
            summary_update_policy: self.get_summary_update_policy(),
        }
    }

    pub fn get_auto_refine_settings(&self) -> ResolvedAutoRefineSettings {
        let turn_interval = nested_f64(&self.settings, keys::AUTO_REFINE, keys::TURN_INTERVAL);
        let cooldown_ms = nested_f64(&self.settings, keys::AUTO_REFINE, keys::COOLDOWN_MS);
        ResolvedAutoRefineSettings {
            enabled: nested_bool(&self.settings, keys::AUTO_REFINE, keys::ENABLED).unwrap_or(true),
            turn_interval: match turn_interval {
                Some(value) if value.is_finite() => value.max(1.0),
                _ => 25.0,
            },
            compact: nested_bool(&self.settings, keys::AUTO_REFINE, keys::COMPACT).unwrap_or(true),
            cooldown_ms: match cooldown_ms {
                Some(value) if value.is_finite() => value.max(0.0),
                _ => 20.0 * 60_000.0,
            },
        }
    }

    pub fn get_branch_summary_settings(&self) -> ResolvedBranchSummarySettings {
        ResolvedBranchSummarySettings {
            reserve_tokens: nested_f64(&self.settings, keys::BRANCH_SUMMARY, keys::RESERVE_TOKENS)
                .unwrap_or(16384.0),
            skip_prompt: nested_bool(&self.settings, keys::BRANCH_SUMMARY, keys::SKIP_PROMPT).unwrap_or(false),
        }
    }

    pub fn get_branch_summary_skip_prompt(&self) -> bool {
        nested_bool(&self.settings, keys::BRANCH_SUMMARY, keys::SKIP_PROMPT).unwrap_or(false)
    }

    // -- retry -------------------------------------------------------------

    pub fn get_retry_enabled(&self) -> bool {
        nested_bool(&self.settings, keys::RETRY, keys::ENABLED).unwrap_or(true)
    }

    pub fn set_retry_enabled(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::RETRY)
            .insert(keys::ENABLED.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::RETRY, Some(keys::ENABLED));
        self.save();
    }

    pub fn get_retry_settings(&self) -> ResolvedRetrySettings {
        ResolvedRetrySettings {
            enabled: self.get_retry_enabled(),
            max_retries: nested_f64(&self.settings, keys::RETRY, keys::MAX_RETRIES).unwrap_or(3.0),
            base_delay_ms: nested_f64(&self.settings, keys::RETRY, keys::BASE_DELAY_MS).unwrap_or(2000.0),
        }
    }

    pub fn get_provider_retry_settings(&self) -> ResolvedProviderRetrySettings {
        ResolvedProviderRetrySettings {
            timeout_ms: self
                .settings
                .get(keys::RETRY)
                .and_then(Value::as_object)
                .and_then(|retry| retry.get(keys::PROVIDER))
                .and_then(Value::as_object)
                .and_then(|provider| provider.get(keys::TIMEOUT_MS))
                .and_then(value_to_f64),
            max_retry_delay_ms: self
                .settings
                .get(keys::RETRY)
                .and_then(|value| value.as_object())
                .and_then(|retry| retry.get(keys::PROVIDER))
                .and_then(|value| value.as_object())
                .and_then(|provider| provider.get(keys::MAX_RETRY_DELAY_MS))
                .and_then(value_to_f64)
                .unwrap_or(60000.0),
        }
    }
}

impl SettingsManager {
    // -- terminal / shell --------------------------------------------------

    pub fn get_hide_thinking_block(&self) -> bool {
        top_bool(&self.settings, keys::HIDE_THINKING_BLOCK).unwrap_or(false)
    }

    pub fn set_hide_thinking_block(&mut self, hide: bool) {
        self.global_settings
            .insert(keys::HIDE_THINKING_BLOCK.to_string(), Value::Bool(hide));
        self.mark_modified(keys::HIDE_THINKING_BLOCK, None);
        self.save();
    }

    pub fn get_shell_path(&self) -> Option<String> {
        top_string(&self.settings, keys::SHELL_PATH)
    }

    pub fn set_shell_path(&mut self, path: Option<&str>) {
        match path {
            Some(path) => {
                self.global_settings
                    .insert(keys::SHELL_PATH.to_string(), Value::String(path.to_string()));
            }
            None => {
                self.global_settings.remove(keys::SHELL_PATH);
            }
        }
        self.mark_modified(keys::SHELL_PATH, None);
        self.save();
    }

    pub fn get_quiet_startup(&self) -> bool {
        top_bool(&self.settings, keys::QUIET_STARTUP).unwrap_or(false)
    }

    pub fn set_quiet_startup(&mut self, quiet: bool) {
        self.global_settings
            .insert(keys::QUIET_STARTUP.to_string(), Value::Bool(quiet));
        self.mark_modified(keys::QUIET_STARTUP, None);
        self.save();
    }

    pub fn get_shell_command_prefix(&self) -> Option<String> {
        top_string(&self.settings, keys::SHELL_COMMAND_PREFIX)
    }

    pub fn set_shell_command_prefix(&mut self, prefix: Option<&str>) {
        match prefix {
            Some(prefix) => {
                self.global_settings.insert(
                    keys::SHELL_COMMAND_PREFIX.to_string(),
                    Value::String(prefix.to_string()),
                );
            }
            None => {
                self.global_settings.remove(keys::SHELL_COMMAND_PREFIX);
            }
        }
        self.mark_modified(keys::SHELL_COMMAND_PREFIX, None);
        self.save();
    }

    pub fn get_npm_command(&self) -> Option<Vec<String>> {
        match self.settings.get(keys::NPM_COMMAND) {
            Some(value) if value.is_array() => Some(top_string_array(&self.settings, keys::NPM_COMMAND)),
            _ => None,
        }
    }

    pub fn set_npm_command(&mut self, command: Option<Vec<String>>) {
        match command {
            Some(command) => {
                self.global_settings.insert(
                    keys::NPM_COMMAND.to_string(),
                    Value::Array(command.into_iter().map(Value::String).collect()),
                );
            }
            None => {
                self.global_settings.remove(keys::NPM_COMMAND);
            }
        }
        self.mark_modified(keys::NPM_COMMAND, None);
        self.save();
    }

    // -- resource paths ----------------------------------------------------

    pub fn get_packages(&self) -> Vec<Value> {
        match self.settings.get(keys::PACKAGES) {
            Some(Value::Array(items)) => items.clone(),
            _ => Vec::new(),
        }
    }

    pub fn set_packages(&mut self, packages: Vec<Value>) {
        self.global_settings
            .insert(keys::PACKAGES.to_string(), Value::Array(packages));
        self.mark_modified(keys::PACKAGES, None);
        self.save();
    }

    pub fn set_project_packages(&mut self, packages: Vec<Value>) {
        let mut project_settings = self.project_settings.clone();
        project_settings.insert(keys::PACKAGES.to_string(), Value::Array(packages));
        self.mark_project_modified(keys::PACKAGES, None);
        self.save_project_settings(project_settings);
    }

    pub fn get_extension_paths(&self) -> Vec<String> {
        top_string_array(&self.settings, keys::EXTENSIONS)
    }

    pub fn set_extension_paths(&mut self, paths: Vec<String>) {
        self.global_settings.insert(
            keys::EXTENSIONS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_modified(keys::EXTENSIONS, None);
        self.save();
    }

    pub fn set_project_extension_paths(&mut self, paths: Vec<String>) {
        let mut project_settings = self.project_settings.clone();
        project_settings.insert(
            keys::EXTENSIONS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_project_modified(keys::EXTENSIONS, None);
        self.save_project_settings(project_settings);
    }

    pub fn get_skill_paths(&self) -> Vec<String> {
        top_string_array(&self.settings, keys::SKILLS)
    }

    pub fn set_skill_paths(&mut self, paths: Vec<String>) {
        self.global_settings.insert(
            keys::SKILLS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_modified(keys::SKILLS, None);
        self.save();
    }

    pub fn set_project_skill_paths(&mut self, paths: Vec<String>) {
        let mut project_settings = self.project_settings.clone();
        project_settings.insert(
            keys::SKILLS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_project_modified(keys::SKILLS, None);
        self.save_project_settings(project_settings);
    }

    pub fn get_prompt_template_paths(&self) -> Vec<String> {
        top_string_array(&self.settings, keys::PROMPTS)
    }

    pub fn set_prompt_template_paths(&mut self, paths: Vec<String>) {
        self.global_settings.insert(
            keys::PROMPTS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_modified(keys::PROMPTS, None);
        self.save();
    }

    pub fn set_project_prompt_template_paths(&mut self, paths: Vec<String>) {
        let mut project_settings = self.project_settings.clone();
        project_settings.insert(
            keys::PROMPTS.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_project_modified(keys::PROMPTS, None);
        self.save_project_settings(project_settings);
    }

    pub fn get_theme_paths(&self) -> Vec<String> {
        top_string_array(&self.settings, keys::THEMES)
    }

    pub fn set_theme_paths(&mut self, paths: Vec<String>) {
        self.global_settings.insert(
            keys::THEMES.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_modified(keys::THEMES, None);
        self.save();
    }

    pub fn set_project_theme_paths(&mut self, paths: Vec<String>) {
        let mut project_settings = self.project_settings.clone();
        project_settings.insert(
            keys::THEMES.to_string(),
            Value::Array(paths.into_iter().map(Value::String).collect()),
        );
        self.mark_project_modified(keys::THEMES, None);
        self.save_project_settings(project_settings);
    }

    // -- skills ------------------------------------------------------------

    pub fn get_enable_skill_commands(&self) -> bool {
        top_bool(&self.settings, keys::ENABLE_SKILL_COMMANDS).unwrap_or(true)
    }

    pub fn set_enable_skill_commands(&mut self, enabled: bool) {
        self.global_settings
            .insert(keys::ENABLE_SKILL_COMMANDS.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::ENABLE_SKILL_COMMANDS, None);
        self.save();
    }

    pub fn get_bundled_skills(&self) -> ResolvedBundledSkills {
        ResolvedBundledSkills {
            websearch: nested_bool(&self.settings, keys::BUNDLED_SKILLS, keys::WEBSEARCH).unwrap_or(true),
        }
    }

    pub fn get_bundled_websearch_enabled(&self) -> bool {
        self.get_bundled_skills().websearch
    }

    pub fn get_enable_builtin_skills(&self) -> bool {
        top_bool(&self.settings, keys::ENABLE_BUILTIN_SKILLS).unwrap_or(true)
    }

    pub fn set_enable_builtin_skills(&mut self, enabled: bool) {
        self.global_settings
            .insert(keys::ENABLE_BUILTIN_SKILLS.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::ENABLE_BUILTIN_SKILLS, None);
        self.save();
    }

    pub fn get_thinking_budgets(&self) -> Option<Value> {
        self.settings.get(keys::THINKING_BUDGETS).cloned()
    }
}

impl SettingsManager {
    // -- terminal / images -------------------------------------------------

    pub fn get_show_images(&self) -> bool {
        nested_bool(&self.settings, keys::TERMINAL, keys::SHOW_IMAGES).unwrap_or(true)
    }

    pub fn set_show_images(&mut self, show: bool) {
        ensure_object(&mut self.global_settings, keys::TERMINAL)
            .insert(keys::SHOW_IMAGES.to_string(), Value::Bool(show));
        self.mark_modified(keys::TERMINAL, Some(keys::SHOW_IMAGES));
        self.save();
    }

    pub fn get_clear_on_shrink(&self) -> bool {
        if let Some(value) = nested_bool(&self.settings, keys::TERMINAL, keys::CLEAR_ON_SHRINK) {
            return value;
        }
        env_flag("PI_CLEAR_ON_SHRINK").as_deref() == Some("1")
    }

    pub fn set_clear_on_shrink(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::TERMINAL)
            .insert(keys::CLEAR_ON_SHRINK.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::TERMINAL, Some(keys::CLEAR_ON_SHRINK));
        self.save();
    }

    pub fn get_fullscreen(&self) -> bool {
        if let Some(value) = env_flag("PI_FULLSCREEN") {
            return value == "1";
        }
        nested_bool(&self.settings, keys::TERMINAL, keys::FULLSCREEN).unwrap_or(true)
    }

    pub fn set_fullscreen(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::TERMINAL)
            .insert(keys::FULLSCREEN.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::TERMINAL, Some(keys::FULLSCREEN));
        self.save();
    }

    pub fn get_fullscreen_mouse(&self) -> bool {
        nested_bool(&self.settings, keys::TERMINAL, keys::FULLSCREEN_MOUSE).unwrap_or(true)
    }

    pub fn set_fullscreen_mouse(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::TERMINAL)
            .insert(keys::FULLSCREEN_MOUSE.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::TERMINAL, Some(keys::FULLSCREEN_MOUSE));
        self.save();
    }

    pub fn get_show_terminal_progress(&self) -> bool {
        nested_bool(&self.settings, keys::TERMINAL, keys::SHOW_TERMINAL_PROGRESS).unwrap_or(false)
    }

    pub fn set_show_terminal_progress(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::TERMINAL)
            .insert(keys::SHOW_TERMINAL_PROGRESS.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::TERMINAL, Some(keys::SHOW_TERMINAL_PROGRESS));
        self.save();
    }

    pub fn get_image_auto_resize(&self) -> bool {
        nested_bool(&self.settings, keys::IMAGES, keys::AUTO_RESIZE).unwrap_or(true)
    }

    pub fn set_image_auto_resize(&mut self, enabled: bool) {
        ensure_object(&mut self.global_settings, keys::IMAGES)
            .insert(keys::AUTO_RESIZE.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::IMAGES, Some(keys::AUTO_RESIZE));
        self.save();
    }

    pub fn get_block_images(&self) -> bool {
        nested_bool(&self.settings, keys::IMAGES, keys::BLOCK_IMAGES).unwrap_or(false)
    }

    pub fn set_block_images(&mut self, blocked: bool) {
        ensure_object(&mut self.global_settings, keys::IMAGES)
            .insert(keys::BLOCK_IMAGES.to_string(), Value::Bool(blocked));
        self.mark_modified(keys::IMAGES, Some(keys::BLOCK_IMAGES));
        self.save();
    }

    // -- models ------------------------------------------------------------

    pub fn get_enabled_models(&self) -> Option<Vec<String>> {
        match self.settings.get(keys::ENABLED_MODELS) {
            Some(Value::Array(_)) => Some(top_string_array(&self.settings, keys::ENABLED_MODELS)),
            _ => None,
        }
    }

    pub fn set_enabled_models(&mut self, patterns: Option<Vec<String>>) {
        match patterns {
            Some(patterns) => {
                self.global_settings.insert(
                    keys::ENABLED_MODELS.to_string(),
                    Value::Array(patterns.into_iter().map(Value::String).collect()),
                );
            }
            None => {
                self.global_settings.remove(keys::ENABLED_MODELS);
            }
        }
        self.mark_modified(keys::ENABLED_MODELS, None);
        self.save();
    }

    /// MCP execution is intentionally restricted to user/global settings.
    pub fn get_global_mcp_servers(&self) -> Option<Map<String, Value>> {
        self.global_settings
            .get(keys::MCP_SERVERS)
            .and_then(|value| value.as_object())
            .cloned()
    }

    pub fn set_global_mcp_server(&mut self, name: &str, config: Value, force: bool) {
        let existing = self
            .global_settings
            .get(keys::MCP_SERVERS)
            .and_then(|value| value.as_object())
            .and_then(|servers| servers.get(name))
            .is_some();
        if existing && !force {
            panic!("MCP server \"{name}\" already exists. Use --force to replace it.");
        }
        let mut servers = self.get_global_mcp_servers().unwrap_or_default();
        servers.insert(name.to_string(), config);
        self.global_settings
            .insert(keys::MCP_SERVERS.to_string(), Value::Object(servers));
        self.mark_modified(keys::MCP_SERVERS, Some(name));
        self.save();
    }

    pub fn remove_global_mcp_server(&mut self, name: &str) -> bool {
        let mut servers = self.get_global_mcp_servers().unwrap_or_default();
        if !servers.contains_key(name) {
            return false;
        }
        servers.remove(name);
        self.global_settings
            .insert(keys::MCP_SERVERS.to_string(), Value::Object(servers));
        self.mark_modified(keys::MCP_SERVERS, Some(name));
        self.save();
        true
    }

    // -- tree filter / cursor / padding ------------------------------------

    pub fn get_tree_filter_mode(&self) -> TreeFilterMode {
        let mode = top_string(&self.settings, keys::TREE_FILTER_MODE);
        match mode {
            Some(mode) if TREE_FILTER_MODES.contains(&mode.as_str()) => mode,
            _ => "user-only".to_string(),
        }
    }

    pub fn set_tree_filter_mode(&mut self, mode: &str) {
        self.global_settings.insert(
            keys::TREE_FILTER_MODE.to_string(),
            Value::String(mode.to_string()),
        );
        self.mark_modified(keys::TREE_FILTER_MODE, None);
        self.save();
    }

    pub fn get_show_hardware_cursor(&self) -> bool {
        top_bool(&self.settings, keys::SHOW_HARDWARE_CURSOR)
            .unwrap_or_else(|| env_flag("PI_HARDWARE_CURSOR").as_deref() == Some("1"))
    }

    pub fn set_show_hardware_cursor(&mut self, enabled: bool) {
        self.global_settings
            .insert(keys::SHOW_HARDWARE_CURSOR.to_string(), Value::Bool(enabled));
        self.mark_modified(keys::SHOW_HARDWARE_CURSOR, None);
        self.save();
    }

    pub fn get_editor_padding_x(&self) -> f64 {
        top_f64(&self.settings, keys::EDITOR_PADDING_X).unwrap_or(0.0)
    }

    pub fn set_editor_padding_x(&mut self, padding: f64) {
        let clamped = padding.floor().max(0.0).min(3.0);
        self.global_settings
            .insert(keys::EDITOR_PADDING_X.to_string(), json_number(clamped));
        self.mark_modified(keys::EDITOR_PADDING_X, None);
        self.save();
    }

    pub fn get_autocomplete_max_visible(&self) -> f64 {
        top_f64(&self.settings, keys::AUTOCOMPLETE_MAX_VISIBLE).unwrap_or(5.0)
    }

    pub fn set_autocomplete_max_visible(&mut self, max_visible: f64) {
        let clamped = max_visible.floor().max(3.0).min(20.0);
        self.global_settings.insert(
            keys::AUTOCOMPLETE_MAX_VISIBLE.to_string(),
            json_number(clamped),
        );
        self.mark_modified(keys::AUTOCOMPLETE_MAX_VISIBLE, None);
        self.save();
    }

    // -- markdown / warnings -----------------------------------------------

    pub fn get_code_block_indent(&self) -> String {
        nested_string(&self.settings, keys::MARKDOWN, keys::CODE_BLOCK_INDENT)
            .unwrap_or_else(|| "  ".to_string())
    }

    pub fn get_mermaid_rendering_mode(&self) -> MermaidRenderingMode {
        match nested_string(&self.settings, keys::MARKDOWN, keys::MERMAID).as_deref() {
            Some(MERMAID_RENDERING_MODE_OFF) => MERMAID_RENDERING_MODE_OFF.to_string(),
            Some(MERMAID_RENDERING_MODE_FINAL) => MERMAID_RENDERING_MODE_FINAL.to_string(),
            _ => MERMAID_RENDERING_MODE_STREAMING.to_string(),
        }
    }

    pub fn set_mermaid_rendering_mode(&mut self, mode: &str) {
        ensure_object(&mut self.global_settings, keys::MARKDOWN)
            .insert(keys::MERMAID.to_string(), Value::String(mode.to_string()));
        self.mark_modified(keys::MARKDOWN, Some(keys::MERMAID));
        self.save();
    }

    pub fn get_warnings(&self) -> WarningSettings {
        match self.settings.get(keys::WARNINGS) {
            Some(Value::Object(object)) => {
                serde_json::from_value(Value::Object(object.clone())).unwrap_or_default()
            }
            _ => WarningSettings::default(),
        }
    }

    pub fn set_warnings(&mut self, warnings: WarningSettings) {
        let value = serde_json::to_value(&warnings).unwrap_or_else(|_| Value::Object(Map::new()));
        self.global_settings.insert(keys::WARNINGS.to_string(), value);
        self.mark_modified(keys::WARNINGS, None);
        self.save();
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use std::path::PathBuf;

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new() -> Self {
            let path = std::env::temp_dir().join(format!("prime-settings-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            TempDir { path }
        }

        pub fn child(&self, name: &str) -> String {
            let path = self.path.join(name);
            std::fs::create_dir_all(&path).unwrap();
            path.to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }

    pub fn read_json(path: &str) -> Settings {
        parse_settings(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    pub fn settings_from(json: &str) -> Settings {
        parse_settings(json).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn deep_merge_merges_nested_objects_and_overrides_scalars() {
        let base = settings_from(r#"{"theme":"dark","compaction":{"enabled":true,"reserveTokens":1}}"#);
        let overrides = settings_from(r#"{"theme":"light","compaction":{"reserveTokens":2}}"#);
        let merged = deep_merge_settings(&base, &overrides);
        assert_eq!(merged["theme"], Value::String("light".to_string()));
        assert_eq!(merged["compaction"]["enabled"], Value::Bool(true));
        assert_eq!(merged["compaction"]["reserveTokens"], json_number(2.0));
    }

    #[test]
    fn migration_maps_legacy_keys() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"queueMode":"all","websockets":true,"skills":{"enableSkillCommands":false,"customDirectories":["/a"]}}"#,
        ));
        assert_eq!(migrated["steeringMode"], Value::String("all".to_string()));
        assert!(!migrated.contains_key("queueMode"));
        assert_eq!(migrated["transport"], Value::String("websocket".to_string()));
        assert!(!migrated.contains_key("websockets"));
        assert_eq!(migrated["enableSkillCommands"], Value::Bool(false));
        assert_eq!(migrated["skills"], serde_json::json!(["/a"]));
    }

    #[test]
    fn migration_moves_retry_max_delay_into_provider() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"retry":{"enabled":true,"maxDelayMs":1234}}"#,
        ));
        assert_eq!(migrated["retry"]["provider"]["maxRetryDelayMs"], json_number(1234.0));
        assert!(migrated["retry"].get("maxDelayMs").is_none());
    }

    #[test]
    fn migration_drops_non_object_telemetry_and_markdown() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"telemetry":true,"markdown":"nope"}"#,
        ));
        assert_eq!(migrated["telemetry"]["enabled"], Value::Bool(true));
        assert!(!migrated.contains_key("markdown"));
    }

    #[test]
    fn unreadable_settings_bytes_are_not_replaced() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let path = join_path(&agent_dir, "settings.json");
        std::fs::write(&path, r#"{"theme":"dark"}"#).unwrap();
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let invalid_utf8 = [0xff, 0xfe, 0x80];
        std::fs::write(&path, invalid_utf8).unwrap();
        manager.set_theme("light");
        assert_eq!(std::fs::read(&path).unwrap(), invalid_utf8);
        assert!(!manager.drain_errors(Some(SETTINGS_SCOPE_GLOBAL)).is_empty());
        let mut reloaded = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert!(!reloaded.drain_errors(Some(SETTINGS_SCOPE_GLOBAL)).is_empty());
        reloaded.set_theme("other");
        assert_eq!(std::fs::read(&path).unwrap(), invalid_utf8);
    }

    #[test]
    fn first_write_recheck_read_failure_preserves_bytes() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let path = join_path(&agent_dir, "settings.json");
        let storage = FileSettingsStorage::new(&project_dir, &agent_dir);
        let mut calls = 0;
        assert!(!storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| {
            calls += 1;
            std::fs::write(&path, [0xff]).unwrap();
            Some("{}".to_string())
        }));
        assert_eq!(calls, 1);
        assert_eq!(std::fs::read(&path).unwrap(), [0xff]);
        assert_eq!(storage.take_failures(SETTINGS_SCOPE_GLOBAL).len(), 1);
    }

    #[test]
    fn settings_lock_file_remains_reusable_after_release() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        manager.set_theme("dark");
        assert!(Path::new(&format!("{agent_dir}/settings.json.lock")).is_file());
        let mut reloaded = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert_eq!(reloaded.get_theme().as_deref(), Some("dark"));
        reloaded.set_theme("light");
        assert!(reloaded.drain_errors(None).is_empty());
    }

    #[test]
    fn file_storage_rereads_under_the_late_first_write_lock() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let storage = FileSettingsStorage::new(&project_dir, &agent_dir);
        let settings_path = join_path(&agent_dir, "settings.json");
        assert!(!Path::new(&settings_path).exists());

        let mut seen: Vec<Option<String>> = Vec::new();
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |current| {
            seen.push(current.map(|value| value.to_string()));
            if current.is_none() {
                // A rival first writer lands between the unlocked read and the lock.
                std::fs::write(&settings_path, r#"{"theme":"rival"}"#).unwrap();
                return Some(r#"{"mine":true}"#.to_string());
            }
            let mut parsed = parse_settings(current.unwrap()).unwrap();
            parsed.insert("mine".to_string(), Value::Bool(true));
            Some(stringify_settings(&parsed))
        });

        assert_eq!(seen, vec![None, Some(r#"{"theme":"rival"}"#.to_string())]);
        let saved = read_json(&settings_path);
        assert_eq!(saved["theme"], Value::String("rival".to_string()));
        assert_eq!(saved["mine"], Value::Bool(true));
    }

    #[tokio::test]
    async fn preserves_externally_added_settings_on_unrelated_write() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"theme":"dark","defaultModel":"claude-sonnet"}"#).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let mut current = read_json(&settings_path);
        current.insert(
            "enabledModels".to_string(),
            serde_json::json!(["claude-opus-4-5", "gpt-5.2-codex"]),
        );
        std::fs::write(&settings_path, stringify_settings(&current)).unwrap();

        manager.set_default_thinking_level("high");
        manager.flush().await;

        let saved = read_json(&settings_path);
        assert_eq!(
            saved["enabledModels"],
            serde_json::json!(["claude-opus-4-5", "gpt-5.2-codex"])
        );
        assert_eq!(saved["defaultThinkingLevel"], Value::String("high".to_string()));
        assert_eq!(saved["theme"], Value::String("dark".to_string()));
        assert_eq!(saved["defaultModel"], Value::String("claude-sonnet".to_string()));
    }

    #[tokio::test]
    async fn preserves_file_changes_to_packages_when_changing_theme() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"theme":"dark","packages":["npm:pi-mcp-adapter"]}"#).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert_eq!(
            manager.get_packages(),
            vec![Value::String("npm:pi-mcp-adapter".to_string())]
        );

        let mut current = read_json(&settings_path);
        current.insert("packages".to_string(), Value::Array(Vec::new()));
        std::fs::write(&settings_path, stringify_settings(&current)).unwrap();

        manager.set_theme("light");
        manager.flush().await;

        let saved = read_json(&settings_path);
        assert_eq!(saved["packages"], serde_json::json!([]));
        assert_eq!(saved["theme"], Value::String("light".to_string()));
    }

    #[tokio::test]
    async fn in_memory_changes_override_file_changes_for_the_same_key() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"theme":"dark"}"#).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let mut current = read_json(&settings_path);
        current.insert("theme".to_string(), Value::String("nord".to_string()));
        std::fs::write(&settings_path, stringify_settings(&current)).unwrap();

        manager.set_theme("light");
        manager.flush().await;

        assert_eq!(read_json(&settings_path)["theme"], Value::String("light".to_string()));
    }

    #[tokio::test]
    async fn project_write_creates_the_config_dir_but_reading_does_not() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let project_config_dir = join_path(&project_dir, CONFIG_DIR_NAME);
        std::fs::write(join_path(&agent_dir, "settings.json"), r#"{"theme":"dark"}"#).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert!(!Path::new(&project_config_dir).exists());
        assert_eq!(manager.get_theme(), Some("dark".to_string()));

        manager.set_project_packages(vec![serde_json::json!({"source": "npm:test-pkg"})]);
        manager.flush().await;
        assert!(Path::new(&project_config_dir).exists());
        assert!(Path::new(&join_path(&project_config_dir, "settings.json")).exists());
    }

    #[tokio::test]
    async fn project_changes_keep_external_project_fields() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let project_settings_path = join_path(&join_path(&project_dir, CONFIG_DIR_NAME), "settings.json");
        std::fs::create_dir_all(Path::new(&project_settings_path).parent().unwrap()).unwrap();
        std::fs::write(
            &project_settings_path,
            r#"{"extensions":["./old-extension.ts"],"prompts":["./old-prompt.md"]}"#,
        )
        .unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let mut current = read_json(&project_settings_path);
        current.insert("prompts".to_string(), serde_json::json!(["./new-prompt.md"]));
        std::fs::write(&project_settings_path, stringify_settings(&current)).unwrap();

        manager.set_project_extension_paths(vec!["./updated-extension.ts".to_string()]);
        manager.flush().await;

        let saved = read_json(&project_settings_path);
        assert_eq!(saved["prompts"], serde_json::json!(["./new-prompt.md"]));
        assert_eq!(saved["extensions"], serde_json::json!(["./updated-extension.ts"]));
    }

    #[tokio::test]
    async fn drains_load_errors_and_reports_save_errors_per_scope() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, "{ invalid global json").unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert_eq!(manager.drain_errors(None).len(), 1);

        manager.set_rlm_max_depth(3.0);
        manager.flush().await;

        let errors = manager.drain_errors(None);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].scope, SETTINGS_SCOPE_GLOBAL);
        assert!(errors[0]
            .error
            .message
            .contains("Global settings not saved: settings file failed to parse:"));
        assert_eq!(std::fs::read_to_string(&settings_path).unwrap(), "{ invalid global json");
    }

    #[test]
    fn drains_only_the_requested_scope() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let project_settings_path = join_path(&join_path(&project_dir, CONFIG_DIR_NAME), "settings.json");
        std::fs::create_dir_all(Path::new(&project_settings_path).parent().unwrap()).unwrap();
        std::fs::write(join_path(&agent_dir, "settings.json"), "{ invalid global json").unwrap();
        std::fs::write(&project_settings_path, "{ invalid project json").unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let global: Vec<String> = manager
            .drain_errors(Some(SETTINGS_SCOPE_GLOBAL))
            .into_iter()
            .map(|entry| entry.scope)
            .collect();
        assert_eq!(global, vec![SETTINGS_SCOPE_GLOBAL.to_string()]);
        let rest: Vec<String> = manager
            .drain_errors(None)
            .into_iter()
            .map(|entry| entry.scope)
            .collect();
        assert_eq!(rest, vec![SETTINGS_SCOPE_PROJECT.to_string()]);
    }

    #[test]
    fn onboarding_shown_defaults_to_false_and_reads_the_legacy_field() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        assert!(!manager.get_onboarding_shown());
        let legacy = SettingsManager::in_memory(settings_from(r#"{"onboardingCompleted":true}"#));
        assert!(legacy.get_onboarding_shown());
    }

    #[tokio::test]
    async fn onboarding_shown_persists_globally() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        manager.set_onboarding_shown(true);
        manager.flush().await;
        assert_eq!(
            read_json(&join_path(&agent_dir, "settings.json"))["onboardingShown"],
            Value::Bool(true)
        );
    }

    /// settings-manager.ts:577-585 chains each persisted write at microtask
    /// time, so a setter that calls `save()` reaches the file without any
    /// explicit `flush()`. The whole file must be on disk after the setter.
    #[test]
    fn save_writes_global_settings_without_an_explicit_flush() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));

        manager.set_onboarding_shown(true);
        manager.set_default_model_and_provider("anthropic", "claude-sonnet-4-5");

        let persisted = std::fs::read_to_string(&settings_path)
            .expect("save() must persist the settings file before the process exits");
        let saved = parse_settings(&persisted).expect("persisted settings stay parseable");
        assert_eq!(saved["onboardingShown"], Value::Bool(true));
        assert_eq!(saved["defaultProvider"], Value::String("anthropic".to_string()));
        assert_eq!(saved["defaultModel"], Value::String("claude-sonnet-4-5".to_string()));
    }

    /// settings-manager.ts:649-669 saves project settings through the same
    /// eager `enqueueWrite` chain.
    #[test]
    fn project_save_writes_settings_without_an_explicit_flush() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let project_settings_path = join_path(&join_path(&project_dir, CONFIG_DIR_NAME), "settings.json");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));

        manager.set_project_packages(vec![serde_json::json!({"source": "npm:eager-pkg"})]);

        let saved = read_json(&project_settings_path);
        assert_eq!(saved["packages"], serde_json::json!([{"source": "npm:eager-pkg"}]));
    }

    /// settings-manager.ts:602-605 parses the file inside the `withLock`
    /// callback, so a corrupt file aborts the write and survives for repair
    /// (the throw is recorded by the `.catch` at settings-manager.ts:583-585).
    #[test]
    fn corrupt_settings_file_is_preserved_and_reported_instead_of_replaced() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"theme":"dark"}"#).unwrap();
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert!(manager.drain_errors(None).is_empty());

        // An external writer (or a partial write) corrupts the file afterwards.
        std::fs::write(&settings_path, r#"{"theme":"dark","other":"kept"#).unwrap();
        manager.set_rlm_max_depth(3.0);

        assert_eq!(
            std::fs::read_to_string(&settings_path).unwrap(),
            r#"{"theme":"dark","other":"kept"#,
            "a corrupt settings file must be preserved for repair"
        );
        let errors = manager.drain_errors(None);
        assert!(
            errors.iter().any(|entry| entry.scope == SETTINGS_SCOPE_GLOBAL),
            "the aborted write must be reported for the global scope: {errors:?}"
        );
    }

    /// settings-manager.ts:411-424 records a load error for a settings file
    /// whose JSON is not an object (`"queueMode" in 5` is a TypeError), and
    /// settings-manager.ts:261-278 rethrows a lock failure instead of hanging.
    #[test]
    fn non_object_settings_json_records_a_load_error() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        std::fs::write(join_path(&agent_dir, "settings.json"), "5").unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let errors = manager.drain_errors(Some(SETTINGS_SCOPE_GLOBAL));
        assert_eq!(errors.len(), 1, "expected one recorded global load error");
        assert_eq!(manager.get_theme(), None);
    }

    /// settings-manager.ts:267-268 rethrows the lock error; the load path
    /// catches it and keeps running with defaults plus a recorded error
    /// (settings-manager.ts:419-423). The lock directory left behind by a
    /// killed process must therefore never panic the process.
    #[test]
    fn a_stale_settings_lock_records_an_error_instead_of_panicking() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"theme":"dark"}"#).unwrap();
        std::fs::create_dir_all(format!("{settings_path}.lock")).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        let errors = manager.drain_errors(Some(SETTINGS_SCOPE_GLOBAL));
        assert_eq!(errors.len(), 1, "expected the lock failure to be recorded");
        assert!(
            errors[0].error.message.contains("Failed to acquire settings lock"),
            "unexpected message: {}",
            errors[0].error.message
        );
    }

    /// settings-manager.ts:790-792 and 825-827 use `??`, which keeps an empty
    /// string; only an absent (or `null`) value falls back to the default.
    #[test]
    fn empty_string_service_tier_and_transport_are_returned_as_is() {
        let manager = SettingsManager::in_memory(settings_from(
            r#"{"defaultServiceTier":"","transport":""}"#,
        ));
        assert_eq!(manager.get_default_service_tier(), "");
        assert_eq!(manager.get_transport(), "");

        let absent = SettingsManager::in_memory(settings_from(r#"{"defaultServiceTier":null}"#));
        assert_eq!(absent.get_default_service_tier(), "default");
        assert_eq!(absent.get_transport(), pi_ai::types::TRANSPORT_AUTO);
    }

    #[test]
    fn auto_refine_defaults_and_fallbacks_match_the_reference() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        let settings = manager.get_auto_refine_settings();
        assert!(settings.enabled);
        assert_eq!(settings.turn_interval, 25.0);
        assert!(settings.compact);
        assert_eq!(settings.cooldown_ms, 1_200_000.0);

        let opt_out = SettingsManager::in_memory(settings_from(r#"{"autoRefine":{"enabled":false}}"#));
        assert!(!opt_out.get_auto_refine_settings().enabled);

        let bad = SettingsManager::in_memory(settings_from(r#"{"autoRefine":{"turnInterval":"x","cooldownMs":null}}"#));
        let bad_settings = bad.get_auto_refine_settings();
        assert_eq!(bad_settings.turn_interval, 25.0);
        assert_eq!(bad_settings.cooldown_ms, 1_200_000.0);

        let valid = SettingsManager::in_memory(settings_from(r#"{"autoRefine":{"turnInterval":0,"cooldownMs":-5}}"#));
        let valid_settings = valid.get_auto_refine_settings();
        assert_eq!(valid_settings.turn_interval, 1.0);
        assert_eq!(valid_settings.cooldown_ms, 0.0);
    }

    #[tokio::test]
    async fn recent_models_record_dedupe_and_cap() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));

        manager.set_default_model_and_provider("anthropic", "claude-sonnet");
        manager.set_default_model_and_provider("openai", "gpt-5");
        manager.set_default_model_and_provider("anthropic", "claude-sonnet");
        manager.flush().await;
        assert_eq!(
            manager.get_recent_models(),
            vec!["anthropic/claude-sonnet".to_string(), "openai/gpt-5".to_string()]
        );

        for index in 0..25 {
            manager.set_default_model_and_provider("p", &format!("m{index}"));
        }
        manager.flush().await;
        assert_eq!(manager.get_recent_models().len(), RECENT_MODELS_LIMIT);
        assert_eq!(manager.get_recent_models()[0], "p/m24".to_string());
    }

    #[tokio::test]
    async fn mermaid_mode_survives_a_non_object_markdown_value() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, r#"{"markdown":"nope"}"#).unwrap();

        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert_eq!(manager.get_mermaid_rendering_mode(), MERMAID_RENDERING_MODE_STREAMING);
        manager.set_mermaid_rendering_mode(MERMAID_RENDERING_MODE_FINAL);
        manager.flush().await;
        assert_eq!(read_json(&settings_path)["markdown"]["mermaid"], Value::String("final".to_string()));
    }

    #[test]
    fn session_dir_expands_tilde_and_project_overrides_global() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        assert_eq!(manager.get_session_dir(), None);

        let global = SettingsManager::in_memory(settings_from(r#"{"sessionDir":"/tmp/s"}"#));
        assert_eq!(global.get_session_dir(), Some("/tmp/s".to_string()));

        let tilde = SettingsManager::in_memory(settings_from(r#"{"sessionDir":"~/s"}"#));
        let home = dirs::home_dir().unwrap().to_string_lossy().into_owned();
        assert_eq!(tilde.get_session_dir(), Some(join_path(&home, "s")));

        let mut project = Settings::new();
        project.insert("sessionDir".to_string(), Value::String("/tmp/project".to_string()));
        let storage = Arc::new(InMemorySettingsStorage::new());
        let initial = SettingsManager::migrate_settings(project);
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |_| Some(stringify_settings(&initial)));
        let manager = SettingsManager::from_storage(storage);
        assert_eq!(manager.get_session_dir(), Some("/tmp/project".to_string()));
    }

    #[test]
    fn global_mcp_servers_ignore_project_settings() {
        let storage = Arc::new(InMemorySettingsStorage::new());
        let manager = SettingsManager::from_storage(storage);
        assert_eq!(manager.get_global_mcp_servers(), None);

        let mut project = Settings::new();
        project.insert(
            "mcpServers".to_string(),
            serde_json::json!({"p": {"type": "http", "url": "https://example.com"}}),
        );
        let storage = Arc::new(InMemorySettingsStorage::new());
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |_| {
            Some(stringify_settings(&SettingsManager::migrate_settings(project.clone())))
        });
        let manager = SettingsManager::from_storage(storage);
        assert_eq!(manager.get_global_mcp_servers(), None);
    }

    #[tokio::test]
    async fn idle_eviction_defaults_to_90_and_treats_none_as_off() {
        let manager = SettingsManager::in_memory(settings_from(r#"{"idleEvictionMinutes":"none"}"#));
        assert_eq!(manager.get_idle_eviction_minutes(), IdleEvictionMinutes::Off);
        let zero = SettingsManager::in_memory(settings_from(r#"{"idleEvictionMinutes":0}"#));
        assert_eq!(
            zero.get_idle_eviction_minutes(),
            IdleEvictionMinutes::Minutes(DEFAULT_IDLE_EVICTION_MINUTES)
        );

        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        manager.set_idle_eviction_minutes(IdleEvictionMinutes::Minutes(15.0));
        manager.flush().await;
        assert_eq!(
            read_json(&join_path(&agent_dir, "settings.json"))["idleEvictionMinutes"],
            json_number(15.0)
        );
    }

    #[test]
    fn telemetry_is_and_ed_across_scopes() {
        let mut global = Settings::new();
        global.insert("telemetry".to_string(), serde_json::json!({"enabled": false}));
        let mut project = Settings::new();
        project.insert("telemetry".to_string(), serde_json::json!({"enabled": true}));
        let storage = Arc::new(InMemorySettingsStorage::new());
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| {
            Some(stringify_settings(&SettingsManager::migrate_settings(global.clone())))
        });
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |_| {
            Some(stringify_settings(&SettingsManager::migrate_settings(project.clone())))
        });
        let mut manager = SettingsManager::from_storage(storage);
        assert!(!manager.get_telemetry_enabled());

        let mut overrides = Settings::new();
        overrides.insert("telemetry".to_string(), serde_json::json!({"noticeShown": true}));
        manager.apply_overrides(&overrides);
        assert!(manager.get_telemetry_notice_shown());
        assert!(!manager.get_telemetry_enabled());
    }

    #[test]
    fn model_context_policies_default_to_off_and_read_the_env() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        assert_eq!(manager.get_summary_update_policy(), SUMMARY_UPDATE_POLICY_OFF);
        assert_eq!(manager.get_model_tool_output_policy(), MODEL_TOOL_OUTPUT_POLICY_OFF);

        let configured = SettingsManager::in_memory(settings_from(
            r#"{"modelToolOutputPolicy":"repeated-large-text-v1","compaction":{"summaryUpdatePolicy":"off"}}"#,
        ));
        assert_eq!(
            configured.get_model_tool_output_policy(),
            MODEL_TOOL_OUTPUT_POLICY_REPEATED_LARGE_TEXT_V1
        );
        assert_eq!(configured.get_summary_update_policy(), SUMMARY_UPDATE_POLICY_OFF);
    }

    #[test]
    fn clamped_setters_and_clamping_ranges() {
        let mut manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        manager.set_editor_padding_x(9.7);
        assert_eq!(manager.get_editor_padding_x(), 3.0);
        manager.set_editor_padding_x(-4.0);
        assert_eq!(manager.get_editor_padding_x(), 0.0);
        manager.set_autocomplete_max_visible(1.0);
        assert_eq!(manager.get_autocomplete_max_visible(), 3.0);
        manager.set_autocomplete_max_visible(100.0);
        assert_eq!(manager.get_autocomplete_max_visible(), 20.0);
    }

    #[test]
    fn retry_and_provider_retry_defaults() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        let retry = manager.get_retry_settings();
        assert!(retry.enabled);
        assert_eq!(retry.max_retries, 3.0);
        assert_eq!(retry.base_delay_ms, 2000.0);
        let provider = manager.get_provider_retry_settings();
        assert_eq!(provider.timeout_ms, None);
        assert_eq!(provider.max_retry_delay_ms, 60000.0);
        let configured = SettingsManager::in_memory(settings_from(
            r#"{"retry":{"timeoutMs":1,"provider":{"timeoutMs":2400,"maxRetryDelayMs":9000}}}"#,
        ));
        assert_eq!(configured.get_provider_retry_settings().timeout_ms, Some(2400.0));
        assert_eq!(configured.get_provider_retry_settings().max_retry_delay_ms, 9000.0);
    }

    #[test]
    fn compaction_and_branch_summary_defaults() {
        let manager = SettingsManager::in_memory(settings_from(r#"{}"#));
        let compaction = manager.get_compaction_settings();
        assert!(compaction.enabled);
        assert_eq!(compaction.reserve_tokens, 16384.0);
        assert_eq!(compaction.keep_recent_tokens, 20000.0);
        assert!(manager.get_compaction_agent_callable());
        let branch = manager.get_branch_summary_settings();
        assert_eq!(branch.reserve_tokens, 16384.0);
        assert!(!branch.skip_prompt);
    }

    #[test]
    fn tree_filter_mode_falls_back_for_invalid_values() {
        let manager = SettingsManager::in_memory(settings_from(r#"{"treeFilterMode":"bogus"}"#));
        assert_eq!(manager.get_tree_filter_mode(), "user-only");
        let valid = SettingsManager::in_memory(settings_from(r#"{"treeFilterMode":"all"}"#));
        assert_eq!(valid.get_tree_filter_mode(), "all");
    }
}

#[cfg(test)]
mod extra_tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn storage_backends_agree_on_scope_isolation() {
        let storage = Arc::new(InMemorySettingsStorage::new());
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| Some("g".to_string()));
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |_| Some("p".to_string()));
        let mut seen = Vec::new();
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |current| {
            seen.push(current.map(|value| value.to_string()));
            None
        });
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |current| {
            seen.push(current.map(|value| value.to_string()));
            None
        });
        assert_eq!(seen, vec![Some("g".to_string()), Some("p".to_string())]);
    }

    #[test]
    fn global_mcp_server_set_and_remove_round_trip() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        manager.set_global_mcp_server(
            "linear",
            serde_json::json!({"type": "http", "url": "https://mcp.example"}),
            false,
        );
        let servers = manager.get_global_mcp_servers().unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers["linear"]["url"], Value::String("https://mcp.example".to_string()));
        assert!(manager.remove_global_mcp_server("linear"));
        assert!(!manager.remove_global_mcp_server("linear"));
        assert!(manager.get_global_mcp_servers().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "already exists. Use --force to replace it.")]
    fn global_mcp_server_duplicate_without_force_panics() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        manager.set_global_mcp_server("linear", serde_json::json!({"type": "http", "url": "a"}), false);
        manager.set_global_mcp_server("linear", serde_json::json!({"type": "http", "url": "b"}), false);
    }

    #[test]
    #[should_panic(expected = "Idle eviction minutes must be a positive number or off")]
    fn idle_eviction_rejects_non_positive_minutes() {
        let mut manager = SettingsManager::in_memory(Settings::new());
        manager.set_idle_eviction_minutes(IdleEvictionMinutes::Minutes(0.0));
    }

    #[tokio::test]
    async fn load_from_storage_treats_missing_and_blank_files_as_empty() {
        let temp = TempDir::new();
        let agent_dir = temp.child("agent");
        let project_dir = temp.child("project");
        let mut manager = SettingsManager::create(&project_dir, Some(&agent_dir));
        assert!(manager.get_global_settings().is_empty());
        assert!(manager.get_project_settings().is_empty());

        let settings_path = join_path(&agent_dir, "settings.json");
        std::fs::write(&settings_path, "").unwrap();
        manager.reload().await;
        assert!(manager.get_global_settings().is_empty());
        assert!(manager.drain_errors(None).is_empty());
    }

    #[test]
    fn migrate_settings_drops_skills_object_without_custom_directories() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"skills":{"enableSkillCommands":true,"customDirectories":[]}}"#,
        ));
        assert!(!migrated.contains_key("skills"));
        assert_eq!(migrated["enableSkillCommands"], Value::Bool(true));
    }

    #[test]
    fn migrate_settings_keeps_existing_transport_and_provider_delay() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"transport":"sse","websockets":true,"retry":{"maxDelayMs":10,"provider":{"maxRetryDelayMs":20}}}"#,
        ));
        assert_eq!(migrated["transport"], Value::String("sse".to_string()));
        assert_eq!(migrated["retry"]["provider"]["maxRetryDelayMs"], json_number(20.0));
        assert!(migrated["retry"].get("maxDelayMs").is_none());
    }

    #[test]
    fn migrate_settings_keeps_object_telemetry_and_markdown() {
        let migrated = SettingsManager::migrate_settings(settings_from(
            r#"{"telemetry":{"enabled":false},"markdown":{"mermaid":"off"}}"#,
        ));
        assert_eq!(migrated["telemetry"]["enabled"], Value::Bool(false));
        assert_eq!(migrated["markdown"]["mermaid"], Value::String("off".to_string()));
    }

    #[test]
    fn warnings_round_trip_through_the_settings_object() {
        let mut manager = SettingsManager::in_memory(Settings::new());
        assert_eq!(manager.get_warnings(), WarningSettings::default());
        manager.set_warnings(WarningSettings {
            anthropic_extra_usage: Some(false),
        });
        assert_eq!(
            manager.get_warnings(),
            WarningSettings {
                anthropic_extra_usage: Some(false)
            }
        );
    }

    #[test]
    fn npm_command_and_enabled_models_round_trip_with_absence() {
        let mut manager = SettingsManager::in_memory(Settings::new());
        assert_eq!(manager.get_npm_command(), None);
        assert_eq!(manager.get_enabled_models(), None);
        manager.set_npm_command(Some(vec!["npm".to_string()]));
        manager.set_enabled_models(Some(vec!["claude-*".to_string()]));
        assert_eq!(manager.get_npm_command(), Some(vec!["npm".to_string()]));
        assert_eq!(manager.get_enabled_models(), Some(vec!["claude-*".to_string()]));
        manager.set_npm_command(None);
        manager.set_enabled_models(None);
        assert_eq!(manager.get_npm_command(), None);
        assert_eq!(manager.get_enabled_models(), None);
    }

    #[tokio::test]
    async fn service_tier_distinguishes_absent_from_null() {
        let mut manager = SettingsManager::in_memory(Settings::new());
        manager.set_default_service_tier(Some(Some("priority".to_string())));
        assert_eq!(
            manager.get_global_settings()["defaultServiceTier"],
            Value::String("priority".to_string())
        );
        manager.set_default_service_tier(Some(None));
        assert_eq!(manager.get_global_settings()["defaultServiceTier"], Value::Null);
        manager.flush().await;
        manager.reload().await;
        assert_eq!(manager.get_global_settings()["defaultServiceTier"], Value::Null);
        manager.set_default_service_tier(None);
        manager.flush().await;
        manager.reload().await;
        assert!(!manager.get_global_settings().contains_key("defaultServiceTier"));
    }

    #[test]
    fn shell_settings_keep_absence_distinguishable() {
        let mut manager = SettingsManager::in_memory(Settings::new());
        assert_eq!(manager.get_shell_path(), None);
        assert_eq!(manager.get_shell_command_prefix(), None);
        manager.set_shell_path(Some("/bin/zsh"));
        manager.set_shell_command_prefix(Some("shopt -s expand_aliases"));
        assert_eq!(manager.get_shell_path(), Some("/bin/zsh".to_string()));
        assert_eq!(
            manager.get_shell_command_prefix(),
            Some("shopt -s expand_aliases".to_string())
        );
        manager.set_shell_path(None);
        assert_eq!(manager.get_shell_path(), None);
    }

    #[test]
    fn telemetry_opt_out_survives_a_runtime_override() {
        let storage = Arc::new(InMemorySettingsStorage::new());
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| {
            Some(r#"{"telemetry":{"enabled":false}}"#.to_string())
        });
        let mut manager = SettingsManager::from_storage(storage);
        assert!(!manager.get_telemetry_enabled());
        manager.apply_overrides(&settings_from(r#"{"telemetry":{"enabled":true}}"#));
        assert!(!manager.get_telemetry_enabled());
    }

    #[test]
    fn project_settings_can_disable_globally_enabled_telemetry() {
        let storage = Arc::new(InMemorySettingsStorage::new());
        storage.with_lock(SETTINGS_SCOPE_GLOBAL, &mut |_| {
            Some(r#"{"telemetry":{"enabled":true}}"#.to_string())
        });
        storage.with_lock(SETTINGS_SCOPE_PROJECT, &mut |_| {
            Some(r#"{"telemetry":{"enabled":false}}"#.to_string())
        });
        let manager = SettingsManager::from_storage(storage);
        assert!(!manager.get_telemetry_enabled());
    }
}
