//! Port of packages/coding-agent/src/core/auth-storage.ts
//!
//! Credential storage for API keys and OAuth tokens. Handles loading, saving,
//! and refreshing credentials from auth.json, with a file lock so concurrent
//! instances cannot race on a token refresh.
//!

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use indexmap::IndexMap;
use pi_ai::types::BoxFuture;
use pi_ai::utils::oauth::types::{
    OAuthCredentials, OAuthProviderInterface,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) use pi_ai::mcp::catalog::register_builtin_mcp_oauth_providers;
use pi_ai::utils::oauth::get_oauth_api_key;
pub(crate) use pi_ai::utils::oauth::{
    get_oauth_provider, get_oauth_providers, register_oauth_provider, reset_oauth_providers,
};

use crate::core::prime_inference_auth::{
    clear_prime_cli_credentials, get_prime_cli_config_path, load_prime_cli_config,
    save_prime_cli_api_key, save_prime_cli_team_selection, PrimeCliConfig, PrimeTeam,
    PRIME_INFERENCE_PROVIDER_ID,
};
pub(crate) use crate::core::resolve_config_value::{
    resolve_config_value, resolve_config_value_or_throw, resolve_config_value_uncached,
};
use crate::utils::atomic_file::{realpath_if_present_sync, write_file_atomic_sync, WriteFileAtomicOptions};

// ---------------------------------------------------------------------------
// Environment-key lookup helpers.
// ---------------------------------------------------------------------------

fn api_key_env_vars(provider: &str) -> Option<Vec<&'static str>> {
    if provider == "github-copilot" {
        return Some(vec!["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]);
    }
    // ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY
    if provider == "anthropic" {
        return Some(vec!["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]);
    }
    let env_var = match provider {
        "openai" => "OPENAI_API_KEY",
        "azure-openai-responses" => "AZURE_OPENAI_API_KEY",
        "prime-inference" => "PRIME_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "google" => "GEMINI_API_KEY",
        "google-vertex" => "GOOGLE_CLOUD_API_KEY",
        "groq" => "GROQ_API_KEY",
        "cerebras" => "CEREBRAS_API_KEY",
        "xai" => "XAI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "vercel-ai-gateway" => "AI_GATEWAY_API_KEY",
        "zai" => "ZAI_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "minimax" => "MINIMAX_API_KEY",
        "minimax-cn" => "MINIMAX_CN_API_KEY",
        "moonshotai" => "MOONSHOT_API_KEY",
        "moonshotai-cn" => "MOONSHOT_API_KEY",
        "huggingface" => "HF_TOKEN",
        "fireworks" => "FIREWORKS_API_KEY",
        "opencode" => "OPENCODE_API_KEY",
        "opencode-go" => "OPENCODE_API_KEY",
        "kimi-coding" => "KIMI_API_KEY",
        "cloudflare-workers-ai" => "CLOUDFLARE_API_KEY",
        "cloudflare-ai-gateway" => "CLOUDFLARE_API_KEY",
        "xiaomi" => "XIAOMI_API_KEY",
        "xiaomi-token-plan-cn" => "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        "xiaomi-token-plan-ams" => "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        "xiaomi-token-plan-sgp" => "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
        _ => return None,
    };
    Some(vec![env_var])
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// `findEnvKeys(provider)` - only actual API key variables.
pub(crate) fn find_env_keys(provider: &str) -> Option<Vec<String>> {
    let env_vars = api_key_env_vars(provider)?;
    let found: Vec<String> = env_vars
        .iter()
        .filter(|name| env_value(name).is_some())
        .map(|name| name.to_string())
        .collect();
    if found.is_empty() {
        None
    } else {
        Some(found)
    }
}

/// Every provider's API-key environment variables (`getApiKeyEnvVars`,
/// packages/ai/src/env-api-keys.ts:92-135), plus the ambient-credential
/// variables the environment candidate falls back to for `amazon-bedrock`
/// (`env-api-keys.ts:182-199`) and `google-vertex` (`env-api-keys.ts:167-180`).
///
/// `AuthStorage.hasAuth` accepts the environment candidate
/// (`auth-storage.ts:757-759`, `auth-storage.ts:449-465`), so a test that asserts
/// "no auth configured" must clear the ambient variables for every provider -
/// the same setup the TypeScript suite performs per-test with
/// `delete process.env.<KEY>` (e.g. `model-registry.test.ts:1193-1194`).
pub(crate) fn ambient_auth_env_var_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    for provider in pi_ai::models::get_providers() {
        if let Some(vars) = api_key_env_vars(provider.as_str()) {
            names.extend(vars);
        }
    }
    names.extend([
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]);
    names.sort_unstable();
    names.dedup();
    names
}

fn has_vertex_adc_credentials() -> bool {
    match std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        Ok(path) if !path.is_empty() => Path::new(&path).exists(),
        _ => dirs::home_dir()
            .map(|home| {
                home.join(".config")
                    .join("gcloud")
                    .join("application_default_credentials.json")
                    .exists()
            })
            .unwrap_or(false),
    }
}

/// `getEnvApiKey(provider)`.
pub(crate) fn get_env_api_key(provider: &str) -> Option<String> {
    if let Some(keys) = find_env_keys(provider) {
        if let Some(key) = keys.first() {
            if let Some(value) = env_value(key) {
                return Some(value);
            }
        }
    }

    if provider == "google-vertex" {
        let has_credentials = has_vertex_adc_credentials();
        let has_project = env_value("GOOGLE_CLOUD_PROJECT").is_some() || env_value("GCLOUD_PROJECT").is_some();
        let has_location = env_value("GOOGLE_CLOUD_LOCATION").is_some();
        if has_credentials && has_project && has_location {
            return Some("<authenticated>".to_string());
        }
    }

    if provider == "amazon-bedrock" {
        let access_key = env_value("AWS_ACCESS_KEY_ID");
        let secret_key = env_value("AWS_SECRET_ACCESS_KEY");
        if env_value("AWS_PROFILE").is_some()
            || (access_key.is_some() && secret_key.is_some())
            || env_value("AWS_BEARER_TOKEN_BEDROCK").is_some()
            || env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
            || env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
            || env_value("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
        {
            return Some("<authenticated>".to_string());
        }
    }

    None
}

/// `getPrimeTeamId()` from env-api-keys.ts.
pub(crate) fn get_prime_team_id() -> Option<String> {
    if let Ok(value) = std::env::var("PRIME_TEAM_ID") {
        let trimmed = value.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    let home = dirs::home_dir()?;
    let config_path = home.join(".prime").join("config.json");
    if !config_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(config_path).ok()?;
    let parsed: Value = serde_json::from_str(&content).ok()?;
    let team_id = parsed.as_object()?.get("team_id")?.as_str()?.trim().to_string();
    if team_id.is_empty() {
        None
    } else {
        Some(team_id)
    }
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimeTeamCredential {
    #[serde(rename = "teamId")]
    pub team_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(rename = "createdAt", skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// `primeTeam?: PrimeTeamCredential | null` - absent, null and a value are all
/// distinguishable, so the field is `Option<Option<..>>`.
pub type OptionalPrimeTeam = Option<Option<PrimeTeamCredential>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthCredential {
    #[serde(rename = "api_key")]
    ApiKey {
        key: String,
        #[serde(
            rename = "primeTeam",
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_optional_prime_team"
        )]
        prime_team: OptionalPrimeTeam,
    },
    #[serde(rename = "oauth")]
    OAuth {
        #[serde(flatten)]
        credentials: OAuthCredentials,
    },
}

fn deserialize_optional_prime_team<'de, D>(deserializer: D) -> Result<OptionalPrimeTeam, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(Some(None)),
        other => serde_json::from_value(other)
            .map(|team| Some(Some(team)))
            .map_err(serde::de::Error::custom),
    }
}

pub type AuthStorageData = IndexMap<String, AuthCredential>;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthStatus {
    pub configured: bool,
    pub source: Option<String>,
    pub label: Option<String>,
}

pub type ActiveAuthStatusSource = &'static str;

pub const AUTH_SOURCE_STORED: ActiveAuthStatusSource = "stored";
pub const AUTH_SOURCE_RUNTIME: ActiveAuthStatusSource = "runtime";
pub const AUTH_SOURCE_ENVIRONMENT: ActiveAuthStatusSource = "environment";
pub const AUTH_SOURCE_PRIME_CLI: ActiveAuthStatusSource = "prime_cli";
pub const AUTH_SOURCE_FALLBACK: ActiveAuthStatusSource = "fallback";
pub const AUTH_SOURCE_MODELS_JSON_KEY: ActiveAuthStatusSource = "models_json_key";
pub const AUTH_SOURCE_MODELS_JSON_COMMAND: ActiveAuthStatusSource = "models_json_command";
pub const AUTH_SOURCE_STALE: ActiveAuthStatusSource = "stale";

#[derive(Debug, Clone, Default)]
pub struct AuthStorageOptions {
    pub prime_cli_config_path: Option<String>,
    pub use_prime_cli_config: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockResult<T> {
    pub result: T,
    pub next: Option<String>,
}

impl<T> LockResult<T> {
    pub fn new(result: T) -> Self {
        Self { result, next: None }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthSourceToken {
    pub provider: String,
    pub source: String,
    #[serde(rename = "identityFingerprint")]
    pub identity_fingerprint: String,
    #[serde(rename = "valueFingerprint")]
    pub value_fingerprint: String,
}

#[derive(Clone, Default)]
struct AuthSourceCandidate {
    source: ActiveAuthStatusSource,
    configured: bool,
    label: Option<String>,
    identity_fingerprint: String,
    value_fingerprint: Option<String>,
    resolve_value_fingerprint:
        Option<Arc<dyn Fn() -> Option<String> + Send + Sync>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthApiKeyResult {
    pub api_key: Option<String>,
    pub source_token: Option<AuthSourceToken>,
}

/// `interface AuthStorageBackend`.
///
/// The lock callback returns the `next` content to persist (the TypeScript
/// `LockResult.next`); a callback that only reads returns `Ok(None)`.
pub trait AuthStorageBackend: Send + Sync {
    fn with_lock(
        &self,
        f: &mut dyn FnMut(Option<String>) -> Result<Option<String>, String>,
    ) -> Result<(), String>;
    fn with_lock_async(&self, f: LockFn) -> BoxFuture<Result<(), String>>;
}

pub type LockFn = Box<
    dyn FnOnce(Option<String>) -> BoxFuture<Result<Option<String>, String>> + Send,
>;

/// `join(getAgentDir(), "auth.json")` (auth-storage.ts:110, imported from
/// `../config.js` at auth-storage.ts:21). `getAgentDir()` in
/// `packages/coding-agent/src/config.ts` honours the `ENV_AGENT_DIR` override
/// (`config.rs:get_agent_dir` is that port), so the hardcoded `~/.prime/agent`
/// path is replaced by the real owner and the env override is honoured.
fn auth_path_default() -> String {
    Path::new(&crate::config::get_agent_dir())
        .join("auth.json")
        .to_string_lossy()
        .to_string()
}

/// `class FileAuthStorageBackend`.
pub struct FileAuthStorageBackend {
    auth_path: String,
}

impl FileAuthStorageBackend {
    pub fn new(auth_path: Option<String>) -> Self {
        let auth_path = auth_path.unwrap_or_else(auth_path_default);
        // proper-lockfile resolves symlinks before locking. Both profiles must
        // lock the same credential file when sharing rotating OAuth tokens.
        let auth_path = realpath_if_present_sync(&auth_path).unwrap_or(auth_path);
        Self { auth_path }
    }

    fn ensure_parent_dir(&self) -> std::io::Result<()> {
        if let Some(dir) = Path::new(&self.auth_path).parent() {
            if !dir.exists() {
                create_dir_mode(dir, 0o700)?;
            }
        }
        Ok(())
    }

    fn ensure_file_exists(&self) -> std::io::Result<()> {
        use std::io::Write;
        let mut open = std::fs::OpenOptions::new();
        open.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open.mode(0o600);
        }
        match open.open(&self.auth_path) {
            Ok(mut file) => {
                file.write_all(b"{}")?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&self.auth_path, std::fs::Permissions::from_mode(0o600))?;
                }
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Steal a lock whose holder died: `proper-lockfile` marks a lock stale once
    /// its mtime is older than `stale` and removes it before retrying, which is
    /// what un-BRICKS auth writes after a crashed instance. The TS relies on that
    /// takeover in both paths - `lockfile.lockSync` (auth-storage.ts:152-157,
    /// proper-lockfile default `stale` 10000 ms) and `lockfile.lock` with
    /// `stale: 30000` (auth-storage.ts:217-230). Returns true when a stale lock
    /// was removed and the caller may retry immediately.
    fn steal_stale_lock(lock_path: &str, stale_ms: u128) -> bool {
        let Ok(metadata) = std::fs::metadata(lock_path) else {
            return false;
        };
        let Ok(modified) = metadata.modified() else {
            return false;
        };
        let Ok(age) = modified.elapsed() else {
            return false;
        };
        if age.as_millis() < stale_ms {
            return false;
        }
        std::fs::remove_dir(lock_path).is_ok()
    }

    /// `proper-lockfile` parity: a `<path>.lock` directory holds the lock.
    ///
    /// `stale` is the sync default `proper-lockfile` applies when the TS passes no
    /// `stale` option (auth-storage.ts:152-157). The TS `onCompromised` callback
    /// (auth-storage.ts:154-156) has no Rust equivalent here: it only reports that
    /// the lock was lost mid-write, and this port holds the lock for the whole
    /// callback, so the callback cannot fire.
    fn acquire_lock_sync_with_retry(&self) -> Result<(), String> {
        const STALE_MS: u128 = 10_000;
        let max_attempts = 10;
        let delay_ms = 20u64;
        let lock_path = format!("{}.lock", self.auth_path);
        let mut last_error: Option<String> = None;
        for attempt in 1..=max_attempts {
            match std::fs::create_dir(&lock_path) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    let locked = error.kind() == std::io::ErrorKind::AlreadyExists;
                    if !locked || attempt == max_attempts {
                        return Err(error.to_string());
                    }
                    last_error = Some(error.to_string());
                    if Self::steal_stale_lock(&lock_path, STALE_MS) {
                        continue;
                    }
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
            }
        }
        Err(last_error.unwrap_or_else(|| "Failed to acquire auth storage lock".to_string()))
    }

    fn release_lock_sync(&self) {
        let _ = std::fs::remove_dir(format!("{}.lock", self.auth_path));
    }
}

fn create_dir_mode(path: &Path, mode: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true).mode(mode);
        builder.create(path)
    }
    #[cfg(not(unix))]
    {
        let _ = mode;
        std::fs::create_dir_all(path)
    }
}

fn write_auth_file(path: &str, content: &str) -> Result<(), String> {
    let real_path = realpath_if_present_sync(path).unwrap_or_else(|_| path.to_string());
    write_file_atomic_sync(
        &real_path,
        content,
        WriteFileAtomicOptions {
            mode: Some(0o600),
            fsync: false,
            fsync_dir: false,
            before_rename: None,
        },
    )
    .map_err(|error| error.to_string())
}

impl AuthStorageBackend for FileAuthStorageBackend {
    fn with_lock(
        &self,
        f: &mut dyn FnMut(Option<String>) -> Result<Option<String>, String>,
    ) -> Result<(), String> {
        self.ensure_parent_dir().map_err(|error| error.to_string())?;
        self.ensure_file_exists().map_err(|error| error.to_string())?;

        self.acquire_lock_sync_with_retry()?;
        let outcome = (|| -> Result<(), String> {
            let current = if Path::new(&self.auth_path).exists() {
                Some(std::fs::read_to_string(&self.auth_path).map_err(|error| error.to_string())?)
            } else {
                None
            };
            let next = f(current)?;
            if let Some(next) = next {
                write_auth_file(&self.auth_path, &next)?;
            }
            Ok(())
        })();
        self.release_lock_sync();
        outcome
    }

    fn with_lock_async(&self, f: LockFn) -> BoxFuture<Result<(), String>> {
        let auth_path = self.auth_path.clone();
        Box::pin(async move {
            if let Some(parent) = Path::new(&auth_path).parent() {
                if !parent.exists() {
                    create_dir_mode(parent, 0o700).map_err(|error| error.to_string())?;
                }
            }
            let mut open = std::fs::OpenOptions::new();
            open.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                open.mode(0o600);
            }
            if let Ok(mut file) = open.open(&auth_path) {
                use std::io::Write;
                let _ = file.write_all(b"{}");
            }

            // proper-lockfile options from the TS: retries 10, factor 2,
            // minTimeout 100ms, maxTimeout 10000ms, `stale: 30000`
            // (auth-storage.ts:217-230). A crashed holder leaves the lock
            // directory behind, so a stale lock must be stolen or auth writes stay
            // bricked until someone deletes it by hand.
            const STALE_MS: u128 = 30_000;
            let lock_path = format!("{}.lock", auth_path);
            let mut delay = Duration::from_millis(100);
            let mut acquired = false;
            for attempt in 0..=10 {
                match std::fs::create_dir(&lock_path) {
                    Ok(()) => {
                        acquired = true;
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        if attempt == 10 {
                            return Err(error.to_string());
                        }
                        if FileAuthStorageBackend::steal_stale_lock(&lock_path, STALE_MS) {
                            continue;
                        }
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(Duration::from_millis(10_000));
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            if !acquired {
                return Err("Failed to acquire auth storage lock".to_string());
            }

            let outcome = (|| async {
                let current = if Path::new(&auth_path).exists() {
                    Some(std::fs::read_to_string(&auth_path).map_err(|error| error.to_string())?)
                } else {
                    None
                };
                let next = f(current).await?;
                if let Some(next) = next {
                    write_auth_file(&auth_path, &next)?;
                }
                Ok(())
            })()
            .await;
            let _ = std::fs::remove_dir(&lock_path);
            outcome
        })
    }
}

/// `class InMemoryAuthStorageBackend`.
///
/// The value lives behind an `Arc` so `withLockAsync` can write it back after its
/// await, the way `this.value` is reachable inside the TypeScript async method.
#[derive(Default)]
pub struct InMemoryAuthStorageBackend {
    value: Arc<Mutex<Option<String>>>,
}

impl InMemoryAuthStorageBackend {
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }
}

impl AuthStorageBackend for InMemoryAuthStorageBackend {
    fn with_lock(
        &self,
        f: &mut dyn FnMut(Option<String>) -> Result<Option<String>, String>,
    ) -> Result<(), String> {
        let mut guard = self.value.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let next = f(guard.clone())?;
        if let Some(next) = next {
            *guard = Some(next);
        }
        Ok(())
    }

    fn with_lock_async(&self, f: LockFn) -> BoxFuture<Result<(), String>> {
        let value = self.value.clone();
        let current = value
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        Box::pin(async move {
            let next = f(current).await?;
            if let Some(next) = next {
                let mut guard = value.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                *guard = Some(next);
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// AuthStorage
// ---------------------------------------------------------------------------

/// Credential storage backed by a JSON file.
pub struct AuthStorage {
    storage: Box<dyn AuthStorageBackend>,
    options: AuthStorageOptions,
    data: AuthStorageData,
    runtime_overrides: IndexMap<String, String>,
    stale_auth_sources: HashMap<String, Vec<AuthSourceToken>>,
    fallback_resolver: Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>,
    load_error: Option<String>,
    errors: Vec<String>,
}

impl std::fmt::Debug for AuthSourceCandidate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AuthSourceCandidate")
            .field("source", &self.source)
            .field("configured", &self.configured)
            .field("label", &self.label)
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("value_fingerprint", &self.value_fingerprint)
            .finish()
    }
}

impl AuthStorage {
    fn new(storage: Box<dyn AuthStorageBackend>, options: AuthStorageOptions) -> Self {
        let mut storage = Self {
            storage,
            options,
            data: IndexMap::new(),
            runtime_overrides: IndexMap::new(),
            stale_auth_sources: HashMap::new(),
            fallback_resolver: None,
            load_error: None,
            errors: Vec::new(),
        };
        storage.reload();
        storage
    }

    pub fn create(auth_path: Option<String>, options: Option<AuthStorageOptions>) -> Self {
        let auth_options = options.unwrap_or_else(|| AuthStorageOptions {
            prime_cli_config_path: None,
            use_prime_cli_config: auth_path.is_none(),
        });
        Self::new(
            Box::new(FileAuthStorageBackend::new(auth_path)),
            auth_options,
        )
    }

    pub fn from_storage(storage: Box<dyn AuthStorageBackend>, options: Option<AuthStorageOptions>) -> Self {
        Self::new(storage, options.unwrap_or_default())
    }

    pub fn in_memory(data: AuthStorageData, options: Option<AuthStorageOptions>) -> Self {
        let storage = InMemoryAuthStorageBackend::new();
        let serialized = serde_json::to_string_pretty(&data).unwrap_or_else(|_| "{}".to_string());
        let _ = storage.with_lock(&mut |_current| Ok(Some(serialized.clone())));
        Self::from_storage(Box::new(storage), options)
    }

    /// Set a runtime API key override (not persisted to disk).
    /// Used for CLI --api-key flag.
    pub fn set_runtime_api_key(&mut self, provider: &str, api_key: &str) {
        self.clear_stale_auth_source(provider, AUTH_SOURCE_RUNTIME);
        self.runtime_overrides
            .insert(provider.to_string(), api_key.to_string());
    }

    /// Remove a runtime API key override.
    pub fn remove_runtime_api_key(&mut self, provider: &str) {
        self.clear_stale_auth_source(provider, AUTH_SOURCE_RUNTIME);
        self.runtime_overrides.shift_remove(provider);
    }

    /// Set a fallback resolver for API keys not found in auth.json or env vars.
    /// Used for custom provider keys from models.json.
    pub fn set_fallback_resolver(
        &mut self,
        resolver: Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
    ) {
        self.fallback_resolver = Some(resolver);
    }

    fn record_error(&mut self, error: String) {
        self.errors.push(error);
    }

    fn fingerprint_auth_source(&self, source: ActiveAuthStatusSource, material: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        hasher.update(b"\0");
        hasher.update(material.as_bytes());
        format!("{}:{:x}", source, hasher.finalize())
    }

    fn create_auth_source_candidate(
        &self,
        source: ActiveAuthStatusSource,
        configured: bool,
        identity_material: &str,
        value_material: Option<&str>,
        label: Option<&str>,
        resolve_value_material: Option<Arc<dyn Fn() -> Option<String> + Send + Sync>>,
    ) -> AuthSourceCandidate {
        let identity_fingerprint =
            self.fingerprint_auth_source(source, &format!("identity:{}", identity_material));
        let value_fingerprint = value_material.map(|value| {
            self.fingerprint_auth_source(source, &format!("value:{}\0{}", identity_material, value))
        });
        let identity_material = identity_material.to_string();
        let source_owned = source;
        let resolve = resolve_value_material.map(|resolve| {
            let identity_material = identity_material.clone();
            Arc::new(move || {
                let value_material = resolve()?;
                Some(fingerprint_auth_source_free(
                    source_owned,
                    &format!("value:{}\0{}", identity_material, value_material),
                ))
            }) as Arc<dyn Fn() -> Option<String> + Send + Sync>
        });
        AuthSourceCandidate {
            source,
            configured,
            label: label.map(|value| value.to_string()),
            identity_fingerprint,
            value_fingerprint,
            resolve_value_fingerprint: resolve,
        }
    }

    fn get_stored_credential_value_material(
        &self,
        provider_id: &str,
        credential: &AuthCredential,
    ) -> Option<String> {
        match credential {
            AuthCredential::ApiKey { key, .. } => {
                if key.starts_with('!') {
                    let resolved = resolve_config_value_uncached(key);
                    resolved.map(|value| format!("api_key:command:{}\0{}", key, value))
                } else {
                    Some(format!(
                        "api_key:{}\0{}",
                        key,
                        resolve_config_value(key).unwrap_or_default()
                    ))
                }
            }
            AuthCredential::OAuth { credentials } => {
                let provider = get_oauth_provider(provider_id);
                let api_key = provider
                    .map(|provider| (provider.get_api_key)(credentials))
                    .unwrap_or_else(|| credentials.access.clone());
                Some(format!(
                    "oauth:{}\0{}\0{}",
                    api_key, credentials.refresh, credentials.expires
                ))
            }
        }
    }

    fn get_runtime_auth_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        let api_key = self.runtime_overrides.get(provider)?;
        let mut candidate = self.create_auth_source_candidate(
            AUTH_SOURCE_RUNTIME,
            false,
            provider,
            Some(api_key),
            None,
            None,
        );
        candidate.label = Some("--api-key".to_string());
        Some(candidate)
    }

    fn get_prime_cli_auth_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        let api_key = self.get_prime_cli_api_key(provider)?;
        let mut candidate = self.create_auth_source_candidate(
            AUTH_SOURCE_PRIME_CLI,
            false,
            provider,
            Some(&api_key),
            None,
            None,
        );
        candidate.label = Some("Prime CLI".to_string());
        Some(candidate)
    }

    fn get_stored_auth_candidate(
        &self,
        provider: &str,
        resolve_command_value: bool,
        resolved_command_value: Option<&str>,
    ) -> Option<AuthSourceCandidate> {
        let credential = self.data.get(provider)?;
        let is_command_api_key = matches!(credential, AuthCredential::ApiKey { key, .. } if key.starts_with('!'));
        let identity_material = match credential {
            AuthCredential::ApiKey { key, .. } if is_command_api_key => {
                format!("api_key:command:{}", key)
            }
            other => format!("{}:{}", provider, credential_type(other)),
        };
        let command_value_material = match (is_command_api_key, resolved_command_value) {
            (true, Some(value)) => match credential {
                AuthCredential::ApiKey { key, .. } => Some(format!("api_key:command:{}\0{}", key, value)),
                _ => None,
            },
            _ => None,
        };
        let value_material = if let Some(material) = command_value_material {
            Some(material)
        } else if is_command_api_key && !resolve_command_value {
            None
        } else {
            self.get_stored_credential_value_material(provider, credential)
        };
        let resolve = if is_command_api_key {
            let credential = credential.clone();
            let provider = provider.to_string();
            let value_material_fn: Arc<dyn Fn() -> Option<String> + Send + Sync> = Arc::new(move || {
                stored_value_material_for(&provider, &credential)
            });
            Some(value_material_fn)
        } else {
            None
        };
        Some(self.create_auth_source_candidate(
            AUTH_SOURCE_STORED,
            true,
            &identity_material,
            value_material.as_deref(),
            None,
            resolve,
        ))
    }

    fn get_environment_auth_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        let env_keys = find_env_keys(provider);
        let env_key = env_keys.as_ref().and_then(|keys| keys.first().cloned());
        let api_key = get_env_api_key(provider)?;
        let label = env_key.clone().unwrap_or_else(|| "ambient credentials".to_string());
        let identity_material = env_key
            .clone()
            .unwrap_or_else(|| self.get_ambient_environment_identity_material(provider));
        Some(self.create_auth_source_candidate(
            AUTH_SOURCE_ENVIRONMENT,
            false,
            &identity_material,
            Some(&format!("{}\0{}", identity_material, api_key)),
            Some(&label),
            None,
        ))
    }

    fn get_ambient_environment_identity_material(&self, provider: &str) -> String {
        if provider == "amazon-bedrock" {
            if let Some(profile) = env_value("AWS_PROFILE") {
                return format!("amazon-bedrock:profile:{}", profile);
            }
            if let Some(access_key) = env_value("AWS_ACCESS_KEY_ID") {
                return format!(
                    "amazon-bedrock:access-key:{}:{}:{}",
                    access_key,
                    env_value("AWS_SECRET_ACCESS_KEY").unwrap_or_default(),
                    env_value("AWS_SESSION_TOKEN").unwrap_or_default()
                );
            }
            if let Some(token) = env_value("AWS_BEARER_TOKEN_BEDROCK") {
                return format!("amazon-bedrock:bearer:{}", token);
            }
            if let Some(uri) = env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
                return format!("amazon-bedrock:ecs-relative:{}", uri);
            }
            if let Some(uri) = env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI") {
                return format!("amazon-bedrock:ecs-full:{}", uri);
            }
            if let Some(file) = env_value("AWS_WEB_IDENTITY_TOKEN_FILE") {
                return format!("amazon-bedrock:web-identity:{}", file);
            }
        }
        if provider == "google-vertex" {
            let project = env_value("GOOGLE_CLOUD_PROJECT")
                .or_else(|| env_value("GCLOUD_PROJECT"))
                .unwrap_or_default();
            let location = env_value("GOOGLE_CLOUD_LOCATION").unwrap_or_default();
            let credentials_path =
                env_value("GOOGLE_APPLICATION_CREDENTIALS").unwrap_or_else(|| "application-default".to_string());
            return format!("google-vertex:{}:{}:{}", project, location, credentials_path);
        }
        provider.to_string()
    }

    fn get_fallback_auth_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        let api_key = self.fallback_resolver.as_ref().and_then(|resolver| resolver(provider))?;
        Some(self.create_auth_source_candidate(
            AUTH_SOURCE_FALLBACK,
            false,
            provider,
            Some(&api_key),
            Some("custom provider config"),
            None,
        ))
    }

    fn get_auth_source_candidates(
        &self,
        provider: &str,
        include_fallback: bool,
    ) -> Vec<AuthSourceCandidate> {
        let fallback_candidate = if include_fallback {
            self.get_fallback_auth_candidate(provider)
        } else {
            None
        };
        let candidates = if provider == PRIME_INFERENCE_PROVIDER_ID {
            vec![
                self.get_runtime_auth_candidate(provider),
                self.get_environment_auth_candidate(provider),
                self.get_prime_cli_auth_candidate(provider),
                self.get_stored_auth_candidate(provider, false, None),
                fallback_candidate,
            ]
        } else {
            vec![
                self.get_runtime_auth_candidate(provider),
                self.get_stored_auth_candidate(provider, false, None),
                self.get_environment_auth_candidate(provider),
                fallback_candidate,
            ]
        };
        candidates.into_iter().flatten().collect()
    }

    fn is_auth_source_stale(&self, provider: &str, candidate: &AuthSourceCandidate) -> bool {
        let matching_stale = self.get_matching_stale_auth_sources(provider, candidate);
        if matching_stale.is_empty() {
            return false;
        }
        let value_fingerprint = candidate
            .value_fingerprint
            .clone()
            .or_else(|| candidate.resolve_value_fingerprint.as_ref().and_then(|resolve| resolve()));
        match value_fingerprint {
            Some(value_fingerprint) => matching_stale
                .iter()
                .any(|token| token.value_fingerprint == value_fingerprint),
            None => false,
        }
    }

    fn get_matching_stale_auth_sources(
        &self,
        provider: &str,
        candidate: &AuthSourceCandidate,
    ) -> Vec<AuthSourceToken> {
        let Some(stale) = self.stale_auth_sources.get(provider) else {
            return Vec::new();
        };
        stale
            .iter()
            .filter(|token| {
                token.source == candidate.source
                    && token.identity_fingerprint == candidate.identity_fingerprint
            })
            .cloned()
            .collect()
    }

    fn get_available_auth_candidate(
        &self,
        provider: &str,
        include_fallback: bool,
    ) -> (Option<AuthSourceCandidate>, bool) {
        let mut has_stale_candidate = false;
        for candidate in self.get_auth_source_candidates(provider, include_fallback) {
            if self.is_auth_source_stale(provider, &candidate) {
                has_stale_candidate = true;
                continue;
            }
            return (Some(candidate), has_stale_candidate);
        }
        (None, has_stale_candidate)
    }

    fn to_auth_status(&self, candidate: &AuthSourceCandidate) -> AuthStatus {
        AuthStatus {
            configured: candidate.configured,
            source: Some(candidate.source.to_string()),
            label: candidate.label.clone(),
        }
    }

    fn get_auth_status_from_candidates(&self, provider: &str) -> AuthStatus {
        let (candidate, has_stale_candidate) = self.get_available_auth_candidate(provider, true);
        if let Some(candidate) = candidate {
            return self.to_auth_status(&candidate);
        }
        if has_stale_candidate {
            return AuthStatus {
                configured: false,
                source: Some(AUTH_SOURCE_STALE.to_string()),
                label: Some("expired".to_string()),
            };
        }
        AuthStatus {
            configured: false,
            source: None,
            label: None,
        }
    }

    pub fn mark_auth_stale(&mut self, provider: &str) -> bool {
        match self.get_current_auth_source_token(provider) {
            Some(token) => self.mark_auth_source_stale(&token),
            None => false,
        }
    }

    fn get_auth_source_token_for_candidate(
        &self,
        provider: &str,
        candidate: &AuthSourceCandidate,
    ) -> Option<AuthSourceToken> {
        let value_fingerprint = candidate
            .value_fingerprint
            .clone()
            .or_else(|| candidate.resolve_value_fingerprint.as_ref().and_then(|resolve| resolve()))?;
        Some(AuthSourceToken {
            provider: provider.to_string(),
            source: candidate.source.to_string(),
            identity_fingerprint: candidate.identity_fingerprint.clone(),
            value_fingerprint,
        })
    }

    pub fn get_current_auth_source_token(&self, provider: &str) -> Option<AuthSourceToken> {
        let (candidate, _) = self.get_available_auth_candidate(provider, true);
        let candidate = candidate?;
        self.get_auth_source_token_for_candidate(provider, &candidate)
    }

    pub fn mark_auth_source_stale(&mut self, token: &AuthSourceToken) -> bool {
        if token.provider.is_empty() {
            return false;
        }
        let mut stale = self
            .stale_auth_sources
            .get(&token.provider)
            .cloned()
            .unwrap_or_default();
        if !stale.iter().any(|existing| {
            existing.source == token.source
                && existing.identity_fingerprint == token.identity_fingerprint
                && existing.value_fingerprint == token.value_fingerprint
        }) {
            stale.push(token.clone());
        }
        self.stale_auth_sources.insert(token.provider.clone(), stale);
        true
    }

    /// Forget every stale marking for a provider (explicit user re-selection).
    pub fn clear_auth_stale(&mut self, provider: &str) {
        self.stale_auth_sources.remove(provider);
    }

    fn clear_stale_auth_source(&mut self, provider: &str, source: ActiveAuthStatusSource) {
        let stale = self.stale_auth_sources.get(provider).cloned();
        let Some(stale) = stale else {
            return;
        };
        let next: Vec<AuthSourceToken> = stale
            .into_iter()
            .filter(|token| token.source != source)
            .collect();
        if next.is_empty() {
            self.stale_auth_sources.remove(provider);
        } else {
            self.stale_auth_sources.insert(provider.to_string(), next);
        }
    }

    /// `parseStorageData` (`packages/coding-agent/src/core/auth-storage.ts:649-654`).
    ///
    /// The TypeScript does `JSON.parse(content) as AuthStorageData` - a bare cast
    /// with no per-entry validation, so one non-conforming entry cannot cost us the
    /// other providers. That entry shape is produced by the TS itself:
    /// `packages/ai/src/utils/oauth/anthropic.ts:375` persists
    /// `refresh: data.refresh_token`, and `JSON.stringify` drops the key when the
    /// refresh response omits `refresh_token`. Rust cannot cast, so parse per entry
    /// and keep every entry that does conform; the entries that do not are reported
    /// so `reload` can set `load_error` and block writes, the way the TS `reload`
    /// sets `loadError` when the load throws (auth-storage.ts:659-672) and
    /// `persistProviderChange` returns early on it (auth-storage.ts:674-677).
    fn parse_storage_data_with_errors(
        content: Option<&str>,
    ) -> (AuthStorageData, Vec<String>) {
        let Some(content) = content else {
            return (IndexMap::new(), Vec::new());
        };
        // A malformed *document* is the TS `JSON.parse` throw: nothing loads.
        let value: Value = match serde_json::from_str(content) {
            Ok(value) => value,
            Err(error) => return (IndexMap::new(), vec![error.to_string()]),
        };
        let Some(entries) = value.as_object() else {
            return (
                IndexMap::new(),
                vec!["auth storage root is not a JSON object".to_string()],
            );
        };
        let mut data = IndexMap::new();
        let mut errors = Vec::new();
        for (provider, entry) in entries {
            match Self::deserialize_stored_credential(entry) {
                Ok(credential) => {
                    data.insert(provider.clone(), credential);
                }
                Err(error) => errors.push(format!(
                    "auth.json entry {:?} could not be read and is left untouched on disk: {}",
                    provider, error
                )),
            }
        }
        (data, errors)
    }

    /// One stored entry, with the tolerance the TypeScript has for free.
    ///
    /// `JSON.parse(content) as AuthStorageData` (auth-storage.ts:653) checks
    /// nothing, and `anthropic.ts:375` writes `refresh: data.refresh_token`,
    /// which `JSON.stringify` drops when the refresh response has no
    /// `refresh_token`. Rust's `OAuthCredentials` (`pi-ai
    /// utils/oauth/types.rs:15-21`) requires `refresh`/`access`/`expires`, so an
    /// absent (or `null`) field is filled with its type default instead of
    /// failing the entry - the same effect as the `#[serde(default)]` the
    /// finding asks for, kept here because `types.rs` is out of this slice.
    /// Anything else that fails to deserialize is a real error for the caller.
    fn deserialize_stored_credential(entry: &Value) -> Result<AuthCredential, String> {
        if let Ok(credential) = serde_json::from_value::<AuthCredential>(entry.clone()) {
            return Ok(credential);
        }
        let is_oauth = entry
            .get("type")
            .and_then(Value::as_str)
            .map(|tag| tag == "oauth")
            .unwrap_or(false);
        if is_oauth {
            if let Some(object) = entry.as_object() {
                let mut filled = object.clone();
                for (field, default) in [
                    ("refresh", Value::String(String::new())),
                    ("access", Value::String(String::new())),
                    ("expires", Value::from(0.0)),
                ] {
                    if filled.get(field).map(Value::is_null).unwrap_or(true) {
                        filled.insert(field.to_string(), default);
                    }
                }
                if let Ok(credential) = serde_json::from_value::<AuthCredential>(Value::Object(filled))
                {
                    return Ok(credential);
                }
            }
        }
        serde_json::from_value::<AuthCredential>(entry.clone()).map_err(|error| error.to_string())
    }

    /// Entry-loss-free load, for the read paths that do not decide about writes.
    fn parse_storage_data(content: Option<&str>) -> AuthStorageData {
        Self::parse_storage_data_with_errors(content).0
    }

    /// Adopt `content` as the in-memory data, the way `this.data = currentData`
    /// does in the TS refresh path (auth-storage.ts:824, :852).
    ///
    /// `this.loadError = null` (auth-storage.ts:825, :853) is only reachable in
    /// TS when `JSON.parse` succeeded, which for the Rust per-entry load means
    /// every entry parsed; otherwise the error stays and keeps writes blocked.
    fn adopt_loaded_data(&mut self, content: Option<&str>) {
        let (data, errors) = Self::parse_storage_data_with_errors(content);
        self.data = data;
        match errors.first() {
            None => self.load_error = None,
            Some(first) => {
                self.load_error = Some(first.clone());
                for error in errors {
                    self.record_error(error);
                }
            }
        }
    }

    /// Rewrite `content` with one provider entry replaced (`Some`) or deleted
    /// (`None`) while every other entry is copied through verbatim.
    ///
    /// This is the Rust equivalent of `const merged: AuthStorageData = { ...currentData }`
    /// (auth-storage.ts:682, :688 and :848-854): an entry this build cannot
    /// deserialize still survives the rewrite, because the TS cast keeps it too.
    fn rewrite_storage_entry(
        content: Option<&str>,
        provider: &str,
        credential: Option<&AuthCredential>,
    ) -> Result<String, String> {
        let mut entries: serde_json::Map<String, Value> = match content {
            Some(content) => match serde_json::from_str::<Value>(content) {
                Ok(Value::Object(entries)) => entries,
                // An unparseable document is the TS `JSON.parse` throw: the
                // caller must not overwrite it with a partial map.
                Ok(_) => return Err("auth storage root is not a JSON object".to_string()),
                Err(error) => return Err(error.to_string()),
            },
            None => serde_json::Map::new(),
        };
        match credential {
            Some(credential) => {
                let value = serde_json::to_value(credential).map_err(|error| error.to_string())?;
                entries.insert(provider.to_string(), value);
            }
            None => {
                entries.remove(provider);
            }
        }
        serde_json::to_string_pretty(&Value::Object(entries)).map_err(|error| error.to_string())
    }

    /// Reload credentials from storage.
    pub fn reload(&mut self) {
        let mut captured: Option<String> = None;
        let mut outcome: Result<(), String> = Ok(());
        {
            let captured_ref = &mut captured;
            outcome = self.storage.with_lock(&mut |current| {
                *captured_ref = current;
                Ok(None)
            });
        }
        match outcome {
            // `reload` sets `this.loadError` when the load throws
            // (auth-storage.ts:668-671) and every later `persistProviderChange`
            // returns early while it is set (auth-storage.ts:674-677), so the
            // file is never rewritten from a partial read. Never clear it on a
            // load that could not read every entry.
            Ok(()) => self.adopt_loaded_data(captured.as_deref()),
            Err(error) => {
                self.load_error = Some(error.clone());
                self.record_error(error);
            }
        }
    }

    fn persist_provider_change(&mut self, provider: &str, credential: Option<&AuthCredential>) {
        if self.load_error.is_some() {
            return;
        }

        let provider_owned = provider.to_string();
        let credential_owned = credential.cloned();
        // `const currentData = this.parseStorageData(current); const merged = { ...currentData }`
        // (auth-storage.ts:681-688): the rewrite is a copy of whatever the file
        // holds plus this one provider, so entries this build cannot deserialize
        // are not dropped. Entries that do not deserialize leave `load_error`
        // set, which returns before this point.
        let outcome = self.storage.with_lock(&mut |current| {
            Self::rewrite_storage_entry(
                current.as_deref(),
                &provider_owned,
                credential_owned.as_ref(),
            )
            .map(Some)
        });
        if let Err(error) = outcome {
            self.record_error(error);
        }
    }

    /// Get credential for a provider.
    pub fn get(&self, provider: &str) -> Option<AuthCredential> {
        self.data.get(provider).cloned()
    }

    /// Set credential for a provider.
    pub fn set(&mut self, provider: &str, credential: AuthCredential) {
        self.clear_stale_auth_source(provider, AUTH_SOURCE_STORED);
        self.data.insert(provider.to_string(), credential.clone());
        self.persist_provider_change(provider, Some(&credential));
    }

    /// Remove credential for a provider.
    pub fn remove(&mut self, provider: &str) {
        self.clear_stale_auth_source(provider, AUTH_SOURCE_STORED);
        self.data.shift_remove(provider);
        self.persist_provider_change(provider, None);
    }

    /// Remove a provider's credential with the disk write verified: returns an
    /// error on any load or write failure instead of recording it, so callers can
    /// refuse to proceed while the credential may still exist on disk.
    /// Disk-authoritative and idempotent - in-memory state is only updated after
    /// the write succeeds.
    pub fn remove_verified(&mut self, provider: &str) -> Result<(), String> {
        let provider_owned = provider.to_string();
        self.storage.with_lock(&mut |current| {
            // `removeVerified` (auth-storage.ts) merges the raw parsed document,
            // so it also must not drop entries it could not deserialize.
            let has_provider = match current.as_deref() {
                Some(current) => serde_json::from_str::<Value>(current)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .map(|entries| entries.contains_key(&provider_owned))
                    .unwrap_or(false),
                None => false,
            };
            if !has_provider {
                return Ok(None);
            }
            Self::rewrite_storage_entry(current.as_deref(), &provider_owned, None).map(Some)
        })?;
        self.data.shift_remove(provider);
        // Post-success only: a failed removal must not make a stale-marked credential selectable again.
        self.clear_stale_auth_source(provider, AUTH_SOURCE_STORED);
        Ok(())
    }

    /// List all providers with credentials.
    pub fn list(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Check if credentials exist for a provider in auth.json.
    pub fn has(&self, provider: &str) -> bool {
        self.data.contains_key(provider)
    }

    /// Check if any form of auth is configured for a provider.
    pub fn has_auth(&self, provider: &str) -> bool {
        self.get_available_auth_candidate(provider, true).0.is_some()
    }

    /// Return auth status without exposing credential values or refreshing tokens.
    pub fn get_auth_status(&self, provider: &str) -> AuthStatus {
        self.get_auth_status_from_candidates(provider)
    }

    /// Get all credentials (for passing to getOAuthApiKey).
    pub fn get_all(&self) -> AuthStorageData {
        self.data.clone()
    }

    pub fn drain_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.errors)
    }

    /// Login to an OAuth provider.
    pub async fn login(
        &mut self,
        provider_id: &str,
        callbacks: pi_ai::utils::oauth::types::OAuthLoginCallbacks,
    ) -> Result<(), String> {
        let provider = get_oauth_provider(provider_id)
            .ok_or_else(|| format!("Unknown OAuth provider: {}", provider_id))?;
        let credentials = (provider.login)(callbacks).await?;
        self.set(
            provider_id,
            AuthCredential::OAuth { credentials },
        );
        Ok(())
    }

    /// Logout from a provider.
    pub fn logout(&mut self, provider: &str) -> Result<(), String> {
        if provider == PRIME_INFERENCE_PROVIDER_ID && self.is_prime_cli_config_enabled() {
            let config_path = self.get_enabled_prime_cli_config_path()?;
            clear_prime_cli_credentials(Some(&config_path)).map_err(|error| {
                let message = error.to_string();
                self.errors.push(message.clone());
                message
            })?;
            self.clear_stale_auth_source(provider, AUTH_SOURCE_PRIME_CLI);
        }
        self.remove(provider);
        Ok(())
    }
}

fn fingerprint_auth_source_free(source: ActiveAuthStatusSource, material: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\0");
    hasher.update(material.as_bytes());
    format!("{}:{:x}", source, hasher.finalize())
}

fn credential_type(credential: &AuthCredential) -> &'static str {
    match credential {
        AuthCredential::ApiKey { .. } => "api_key",
        AuthCredential::OAuth { .. } => "oauth",
    }
}

fn stored_value_material_for(provider_id: &str, credential: &AuthCredential) -> Option<String> {
    match credential {
        AuthCredential::ApiKey { key, .. } => {
            if key.starts_with('!') {
                let resolved = resolve_config_value_uncached(key);
                resolved.map(|value| format!("api_key:command:{}\0{}", key, value))
            } else {
                Some(format!(
                    "api_key:{}\0{}",
                    key,
                    resolve_config_value(key).unwrap_or_default()
                ))
            }
        }
        AuthCredential::OAuth { credentials } => {
            let provider = get_oauth_provider(provider_id);
            let api_key = provider
                .map(|provider| (provider.get_api_key)(credentials))
                .unwrap_or_else(|| credentials.access.clone());
            Some(format!(
                "oauth:{}\0{}\0{}",
                api_key, credentials.refresh, credentials.expires
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Header resolution keeps the auth-specific empty-value filtering. Command
// resolution uses the shared hidden, bounded helper above.
// ---------------------------------------------------------------------------

/// `resolveHeaders(headers)`.
pub(crate) fn resolve_headers(headers: Option<&IndexMap<String, String>>) -> Option<IndexMap<String, String>> {
    let headers = headers?;
    let mut resolved = IndexMap::new();
    for (key, value) in headers {
        if let Some(resolved_value) = resolve_config_value(value) {
            if !resolved_value.is_empty() {
                resolved.insert(key.clone(), resolved_value);
            }
        }
    }
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

/// `resolveHeadersOrThrow(headers, description)`.
pub(crate) fn resolve_headers_or_throw(
    headers: Option<&IndexMap<String, String>>,
    description: &str,
) -> Result<Option<IndexMap<String, String>>, String> {
    let Some(headers) = headers else {
        return Ok(None);
    };
    let mut resolved = IndexMap::new();
    for (key, value) in headers {
        resolved.insert(
            key.clone(),
            resolve_config_value_or_throw(value, &format!("{} header \"{}\"", description, key))?,
        );
    }
    if resolved.is_empty() {
        Ok(None)
    } else {
        Ok(Some(resolved))
    }
}

// ---------------------------------------------------------------------------
// OAuth refresh + API key resolution
// ---------------------------------------------------------------------------

impl AuthStorage {
    /// Refresh OAuth token with backend locking to prevent race conditions.
    async fn refresh_oauth_token_with_lock(
        &mut self,
        provider_id: &str,
    ) -> Result<Option<(String, OAuthCredentials)>, String> {
        let Some(provider) = get_oauth_provider(provider_id) else {
            return Ok(None);
        };

        let provider_id_owned = provider_id.to_string();
        let loaded: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let refresh_result: Arc<Mutex<Option<(String, OAuthCredentials)>>> =
            Arc::new(Mutex::new(None));
        let merged_after_refresh: Arc<Mutex<Option<AuthStorageData>>> = Arc::new(Mutex::new(None));

        // `await this.storage.withLockAsync(async (current) => { ... })`
        // (auth-storage.ts:822-855): the TS holds the file lock across the network
        // refresh AND the write, so two instances cannot both refresh the same
        // token. The refresh must therefore happen inside this callback, not
        // between two separate `withLock` calls.
        {
            let loaded_ref = Arc::clone(&loaded);
            let refresh_result_ref = Arc::clone(&refresh_result);
            let merged_ref = Arc::clone(&merged_after_refresh);
            let provider_id_for_lock = provider_id_owned.clone();
            self.storage
                .with_lock_async(Box::new(move |current| {
                    Box::pin(async move {
                        if let Ok(mut slot) = loaded_ref.lock() {
                            *slot = current.clone();
                        }
                        let current_data = AuthStorage::parse_storage_data(current.as_deref());
                        let credential = current_data.get(&provider_id_for_lock).cloned();
                        let Some(credentials) = (match credential {
                            Some(AuthCredential::OAuth { credentials }) => Some(credentials),
                            _ => None,
                        }) else {
                            return Ok(None);
                        };

                        if now_millis() < credentials.expires as i64 {
                            if let Ok(mut slot) = refresh_result_ref.lock() {
                                *slot =
                                    Some(((provider.get_api_key)(&credentials), credentials));
                            }
                            return Ok(None);
                        }

                        let mut oauth_creds: HashMap<String, OAuthCredentials> = HashMap::new();
                        for (key, value) in AuthStorage::parse_storage_data(current.as_deref()) {
                            if let AuthCredential::OAuth { credentials } = value {
                                oauth_creds.insert(key, credentials);
                            }
                        }

                        let Some(refreshed) =
                            get_oauth_api_key(&provider_id_for_lock, &oauth_creds).await?
                        else {
                            return Ok(None);
                        };
                        let api_key = refreshed.api_key;
                        let new_credentials = refreshed.new_credentials;

                        // `merged = { ...currentData, [providerId]: { type: "oauth",
                        // ...refreshed.newCredentials } }` (auth-storage.ts:848-854):
                        // copy every stored entry through verbatim and replace only
                        // this provider, so entries this build cannot deserialize
                        // survive the refresh write.
                        let mut merged = current_data.clone();
                        let credential = AuthCredential::OAuth {
                            credentials: new_credentials.clone(),
                        };
                        merged.insert(provider_id_for_lock.clone(), credential.clone());
                        let next = AuthStorage::rewrite_storage_entry(
                            current.as_deref(),
                            &provider_id_for_lock,
                            Some(&credential),
                        )?;
                        if let Ok(mut slot) = merged_ref.lock() {
                            *slot = Some(merged);
                        }
                        if let Ok(mut slot) = refresh_result_ref.lock() {
                            *slot = Some((api_key, new_credentials));
                        }
                        Ok(Some(next))
                    })
                }))
                .await?;
        }

        // `this.data = currentData; this.loadError = null;` (auth-storage.ts:824-825)
        // ran inside the lock; apply the same state to this storage now that the
        // callback cannot borrow `self`.
        let captured = loaded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        self.adopt_loaded_data(captured.as_deref());

        let result = refresh_result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if let Some(merged) = merged_after_refresh
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            // `this.data = merged;` (auth-storage.ts:852).
            self.data = merged;
        }
        Ok(result)
    }

    /// Get API key for a provider with its auth source token.
    ///
    /// Priority:
    /// 1. Runtime override (CLI --api-key)
    /// 2. Prime Inference: environment variable, Prime CLI config, auth.json
    /// 3. Other providers: auth.json, environment variable
    /// 4. Fallback resolver (models.json custom providers)
    pub async fn get_api_key_with_source_token(
        &mut self,
        provider_id: &str,
        include_fallback: bool,
    ) -> Result<AuthApiKeyResult, String> {
        // Runtime overrides take precedence over stored credentials and environment keys.
        let runtime_candidate = self.get_runtime_auth_candidate(provider_id);
        let runtime_key = self.runtime_overrides.get(provider_id).cloned();
        if let (Some(runtime_key), Some(runtime_candidate)) = (runtime_key, runtime_candidate) {
            if !self.is_auth_source_stale(provider_id, &runtime_candidate) {
                return Ok(AuthApiKeyResult {
                    api_key: Some(runtime_key),
                    source_token: self.get_auth_source_token_for_candidate(provider_id, &runtime_candidate),
                });
            }
        }

        let env_candidate = self.get_environment_auth_candidate(provider_id);
        let env_key = get_env_api_key(provider_id);
        if provider_id == PRIME_INFERENCE_PROVIDER_ID {
            if let (Some(env_key), Some(env_candidate)) = (env_key.clone(), env_candidate.clone()) {
                if !self.is_auth_source_stale(provider_id, &env_candidate) {
                    return Ok(AuthApiKeyResult {
                        api_key: Some(env_key),
                        source_token: self.get_auth_source_token_for_candidate(provider_id, &env_candidate),
                    });
                }
            }

            let prime_cli_candidate = self.get_prime_cli_auth_candidate(provider_id);
            let prime_cli_key = self.get_prime_cli_api_key(provider_id);
            if let (Some(prime_cli_key), Some(prime_cli_candidate)) = (prime_cli_key, prime_cli_candidate) {
                if !self.is_auth_source_stale(provider_id, &prime_cli_candidate) {
                    return Ok(AuthApiKeyResult {
                        api_key: Some(prime_cli_key),
                        source_token: self
                            .get_auth_source_token_for_candidate(provider_id, &prime_cli_candidate),
                    });
                }
            }
        }

        let credential = self.data.get(provider_id).cloned();

        if let Some(AuthCredential::ApiKey { key, .. }) = &credential {
            let stored_candidate = self.get_stored_auth_candidate(provider_id, false, None);
            if let Some(stored_candidate) = stored_candidate {
                if !self.is_auth_source_stale(provider_id, &stored_candidate) {
                    let has_stale_record =
                        !self.get_matching_stale_auth_sources(provider_id, &stored_candidate).is_empty();
                    let api_key = if key.starts_with('!') && has_stale_record {
                        resolve_config_value_uncached(key)
                    } else {
                        resolve_config_value(key)
                    };
                    let source_token = match &api_key {
                        None => None,
                        Some(api_key) => {
                            let candidate = if key.starts_with('!') {
                                self.get_stored_auth_candidate(provider_id, false, Some(api_key))
                                    .unwrap_or_else(|| stored_candidate.clone())
                            } else {
                                stored_candidate.clone()
                            };
                            self.get_auth_source_token_for_candidate(provider_id, &candidate)
                        }
                    };
                    return Ok(AuthApiKeyResult {
                        api_key,
                        source_token,
                    });
                }
            }
        }

        if let Some(AuthCredential::OAuth { credentials }) = &credential {
            let stored_candidate = self.get_stored_auth_candidate(provider_id, false, None);
            if let Some(stored_candidate) = stored_candidate {
                if !self.is_auth_source_stale(provider_id, &stored_candidate) {
                    let Some(provider) = get_oauth_provider(provider_id) else {
                        return Ok(AuthApiKeyResult::default());
                    };
                    // Lock refreshes so concurrent instances cannot race on the credential file.
                    let needs_refresh = now_millis() >= credentials.expires as i64;

                    if needs_refresh {
                        match self.refresh_oauth_token_with_lock(provider_id).await {
                            Ok(Some((api_key, _new_credentials))) => {
                                let refreshed_candidate =
                                    self.get_stored_auth_candidate(provider_id, false, None);
                                return Ok(AuthApiKeyResult {
                                    api_key: Some(api_key),
                                    source_token: refreshed_candidate.as_ref().and_then(|candidate| {
                                        self.get_auth_source_token_for_candidate(provider_id, candidate)
                                    }),
                                });
                            }
                            Ok(None) => {}
                            Err(error) => {
                                self.record_error(error);
                                // A peer may have refreshed successfully; reload before treating this refresh as failed.
                                self.reload();
                                let updated_cred = self.data.get(provider_id).cloned();
                                if let Some(AuthCredential::OAuth { credentials }) = &updated_cred {
                                    if now_millis() < credentials.expires as i64 {
                                        let updated_candidate =
                                            self.get_stored_auth_candidate(provider_id, false, None);
                                        return Ok(AuthApiKeyResult {
                                            api_key: Some((provider.get_api_key)(credentials)),
                                            source_token: updated_candidate.as_ref().and_then(|candidate| {
                                                self.get_auth_source_token_for_candidate(
                                                    provider_id,
                                                    candidate,
                                                )
                                            }),
                                        });
                                    }
                                }
                                // Preserve credentials for a later /login retry while discovery skips this provider.
                                return Ok(AuthApiKeyResult::default());
                            }
                        }
                    } else {
                        return Ok(AuthApiKeyResult {
                            api_key: Some((provider.get_api_key)(credentials)),
                            source_token: self
                                .get_auth_source_token_for_candidate(provider_id, &stored_candidate),
                        });
                    }
                }
            }
        }
        // Stored auth wins over environment variables for non-Prime-Inference providers.
        if provider_id != PRIME_INFERENCE_PROVIDER_ID {
            if let (Some(env_key), Some(env_candidate)) = (env_key, env_candidate) {
                if !self.is_auth_source_stale(provider_id, &env_candidate) {
                    return Ok(AuthApiKeyResult {
                        api_key: Some(env_key),
                        source_token: self.get_auth_source_token_for_candidate(provider_id, &env_candidate),
                    });
                }
            }
        }
        if include_fallback {
            let fallback_candidate = self.get_fallback_auth_candidate(provider_id);
            if let Some(fallback_candidate) = fallback_candidate {
                if !self.is_auth_source_stale(provider_id, &fallback_candidate) {
                    return Ok(AuthApiKeyResult {
                        api_key: self
                            .fallback_resolver
                            .as_ref()
                            .and_then(|resolver| resolver(provider_id)),
                        source_token: self
                            .get_auth_source_token_for_candidate(provider_id, &fallback_candidate),
                    });
                }
            }
        }

        Ok(AuthApiKeyResult::default())
    }

    pub async fn get_api_key(
        &mut self,
        provider_id: &str,
        include_fallback: bool,
    ) -> Result<Option<String>, String> {
        let result = self
            .get_api_key_with_source_token(provider_id, include_fallback)
            .await?;
        Ok(result.api_key)
    }

    /// Get all registered OAuth providers.
    pub fn get_oauth_providers(&self) -> Vec<OAuthProviderInterface> {
        get_oauth_providers()
    }

    pub fn set_prime_inference_team_selection(&mut self, team: Option<PrimeTeam>) -> Result<(), String> {
        if self.is_prime_cli_config_enabled() {
            let config_path = self.get_enabled_prime_cli_config_path()?;
            save_prime_cli_team_selection(team.as_ref(), Some(&config_path)).map_err(|error| {
                let message = error.to_string();
                self.errors.push(message.clone());
                message
            })?;
            return Ok(());
        }

        let credential = self.data.get(PRIME_INFERENCE_PROVIDER_ID).cloned();
        let Some(AuthCredential::ApiKey { key, prime_team }) = credential else {
            return Ok(());
        };
        self.set(
            PRIME_INFERENCE_PROVIDER_ID,
            AuthCredential::ApiKey {
                key,
                prime_team: Some(team.map(|team| to_prime_team_credential(&team))),
            },
        );
        let _ = prime_team;
        Ok(())
    }

    pub fn set_prime_inference_api_key(&mut self, api_key: &str) -> Result<(), String> {
        if self.is_prime_cli_config_enabled() {
            let config_path = self.get_enabled_prime_cli_config_path()?;
            let config = load_prime_cli_config(Some(&config_path));
            let existing_credential = self.data.get(PRIME_INFERENCE_PROVIDER_ID).cloned();
            let legacy_prime_team = match existing_credential {
                Some(AuthCredential::ApiKey { prime_team, .. }) => prime_team,
                _ => None,
            };
            let outcome = if config.api_key.as_deref() != Some(api_key) {
                save_prime_cli_api_key(api_key, Some(&config_path)).map(|_| ())
            } else if !config.team_id_from_env
                && (legacy_prime_team == Some(None)
                    || (config.team_id.is_none() && legacy_prime_team.is_some()))
            {
                let legacy_team = legacy_prime_team.flatten().map(|credential| to_prime_team(&credential));
                save_prime_cli_team_selection(legacy_team.as_ref(), Some(&config_path)).map(|_| ())
            } else {
                Ok(())
            };
            outcome.map_err(|error| {
                let message = error.to_string();
                self.errors.push(message.clone());
                message
            })?;
            self.clear_stale_auth_source(PRIME_INFERENCE_PROVIDER_ID, AUTH_SOURCE_PRIME_CLI);
            if self.data.contains_key(PRIME_INFERENCE_PROVIDER_ID) {
                self.remove(PRIME_INFERENCE_PROVIDER_ID);
            }
            return Ok(());
        }

        let existing_credential = self.data.get(PRIME_INFERENCE_PROVIDER_ID).cloned();
        let existing_prime_team = match existing_credential {
            Some(AuthCredential::ApiKey { prime_team, .. }) => prime_team,
            _ => None,
        };
        self.set(
            PRIME_INFERENCE_PROVIDER_ID,
            AuthCredential::ApiKey {
                key: api_key.to_string(),
                prime_team: existing_prime_team,
            },
        );
        Ok(())
    }

    pub fn get_prime_inference_team_selection(&self) -> Option<Option<PrimeTeamCredential>> {
        if std::env::var("PRIME_TEAM_ID").ok().is_some_and(|value| !value.trim().is_empty()) {
            return None;
        }
        let mut config: Option<PrimeCliConfig> = None;
        if self.is_prime_cli_config_enabled() {
            config = self.get_prime_cli_config(PRIME_INFERENCE_PROVIDER_ID);
            if config
                .as_ref()
                .map(|config| config.team_id_from_env)
                .unwrap_or(false)
            {
                return None;
            }
        }

        let credential = self.data.get(PRIME_INFERENCE_PROVIDER_ID).cloned();
        let auth_source = self.get_auth_status(PRIME_INFERENCE_PROVIDER_ID).source;
        // Runtime and environment keys do not override the user's selected team.
        // A stale CLI key must not erase the selected team used to validate cached model access.
        let config_api_key = config.as_ref().and_then(|config| config.api_key.clone());
        if auth_source.as_deref() == Some(AUTH_SOURCE_PRIME_CLI)
            || (auth_source.as_deref() == Some(AUTH_SOURCE_STALE) && config_api_key.is_some())
        {
            if let Some(AuthCredential::ApiKey { prime_team, .. }) = &credential {
                if *prime_team == Some(None) {
                    return Some(None);
                }
            }
            if let Some(config) = &config {
                if let Some(team_id) = &config.team_id {
                    return Some(Some(PrimeTeamCredential {
                        team_id: team_id.clone(),
                        name: config
                            .team_name
                            .clone()
                            .unwrap_or_else(|| "Prime CLI team".to_string()),
                        slug: None,
                        role: config.team_role.clone(),
                        created_at: None,
                    }));
                }
            }
            if let Some(AuthCredential::ApiKey { prime_team, .. }) = &credential {
                if let Some(team) = prime_team {
                    return Some(team.clone());
                }
            }
            return Some(None);
        }
        if let Some(AuthCredential::ApiKey { prime_team, .. }) = &credential {
            if prime_team.is_some() {
                return prime_team.clone();
            }
        }
        let config_api_key_missing = config
            .as_ref()
            .map(|config| config.api_key.is_none())
            .unwrap_or(true);
        if config_api_key_missing {
            if let Some(config) = &config {
                if let Some(team_id) = &config.team_id {
                    return Some(Some(PrimeTeamCredential {
                        team_id: team_id.clone(),
                        name: config
                            .team_name
                            .clone()
                            .unwrap_or_else(|| "Prime CLI team".to_string()),
                        slug: None,
                        role: config.team_role.clone(),
                        created_at: None,
                    }));
                }
            }
        }
        None
    }

    pub fn get_provider_headers(&self, provider_id: &str) -> Option<IndexMap<String, String>> {
        if provider_id != PRIME_INFERENCE_PROVIDER_ID {
            return None;
        }

        if let Some(team_id) = std::env::var("PRIME_TEAM_ID").ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty()) {
            return Some(IndexMap::from([("X-Prime-Team-ID".to_string(), team_id)]));
        }

        let prime_cli_config = self.get_prime_cli_config(provider_id);
        if prime_cli_config
            .as_ref()
            .map(|config| config.team_id_from_env)
            .unwrap_or(false)
        {
            return prime_cli_config.and_then(|config| config.team_id).map(|team_id| {
                let mut headers = IndexMap::new();
                headers.insert("X-Prime-Team-ID".to_string(), team_id);
                headers
            });
        }

        let team_id = self
            .get_prime_inference_team_selection()
            .flatten()
            .map(|team| team.team_id);
        team_id.map(|team_id| {
            let mut headers = IndexMap::new();
            headers.insert("X-Prime-Team-ID".to_string(), team_id);
            headers
        })
    }

    pub fn get_prime_cli_config_path(&self) -> Option<String> {
        if !self.is_prime_cli_config_enabled() {
            return None;
        }
        Some(get_prime_cli_config_path(
            self.options.prime_cli_config_path.as_deref(),
        ))
    }

    fn get_prime_cli_config(&self, provider_id: &str) -> Option<PrimeCliConfig> {
        if provider_id != PRIME_INFERENCE_PROVIDER_ID {
            return None;
        }
        if !self.is_prime_cli_config_enabled() {
            return None;
        }
        Some(load_prime_cli_config(self.options.prime_cli_config_path.as_deref()))
    }

    fn get_prime_cli_api_key(&self, provider_id: &str) -> Option<String> {
        self.get_prime_cli_config(provider_id)?.api_key
    }

    fn get_enabled_prime_cli_config_path(&self) -> Result<String, String> {
        self.get_prime_cli_config_path()
            .ok_or_else(|| "Prime CLI config is not enabled".to_string())
    }

    fn is_prime_cli_config_enabled(&self) -> bool {
        self.options.use_prime_cli_config || self.options.prime_cli_config_path.is_some()
    }
}

/// `PrimeTeamCredential` and `prime-inference-auth.ts` `PrimeTeam` are the same
/// TypeScript shape; the two Rust structs are this port's split of that seam.
fn to_prime_team(team: &PrimeTeamCredential) -> PrimeTeam {
    PrimeTeam {
        team_id: team.team_id.clone(),
        name: team.name.clone(),
        slug: team.slug.clone(),
        role: team.role.clone(),
        created_at: team.created_at.clone(),
    }
}

fn to_prime_team_credential(team: &PrimeTeam) -> PrimeTeamCredential {
    PrimeTeamCredential {
        team_id: team.team_id.clone(),
        name: team.name.clone(),
        slug: team.slug.clone(),
        role: team.role.clone(),
        created_at: team.created_at.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::utils::oauth::types::{
        OAuthLoginCallbacks, OAuthProviderInfo, OAuthSelectPrompt,
    };
    use serde_json::json;
    use pi_ai::utils::oauth::{get_oauth_provider_info_list, unregister_oauth_provider};

    /// Age a lock *directory* so the stale-lock takeover can be exercised.
    fn set_directory_mtime_to_now_minus(path: &str, seconds: u64) {
        let mut options = std::fs::OpenOptions::new();
        #[cfg(unix)]
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.write(true);
            // FILE_FLAG_BACKUP_SEMANTICS: required to open a directory handle.
            options.custom_flags(0x0200_0000);
        }
        let file = options.open(path).expect("open lock directory");
        file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(seconds))
            .expect("age the lock directory");
    }

    fn memory(data: Value) -> AuthStorage {
        let parsed: AuthStorageData = serde_json::from_value(data).unwrap();
        AuthStorage::in_memory(
            parsed,
            Some(AuthStorageOptions {
                prime_cli_config_path: None,
                use_prime_cli_config: false,
            }),
        )
    }

    #[test]
    fn api_key_credential_round_trips_with_absent_null_and_value_team() {
        let storage = memory(json!({
            "plain": {"type": "api_key", "key": "k"},
            "withNull": {"type": "api_key", "key": "k", "primeTeam": null},
            "withTeam": {"type": "api_key", "key": "k", "primeTeam": {"teamId": "t1", "name": "One"}}
        }));

        let plain = storage.get("plain").unwrap();
        match plain {
            AuthCredential::ApiKey { prime_team, .. } => assert_eq!(prime_team, None),
            _ => panic!("expected api key"),
        }
        let with_null = storage.get("withNull").unwrap();
        match with_null {
            AuthCredential::ApiKey { prime_team, .. } => assert_eq!(prime_team, Some(None)),
            _ => panic!("expected api key"),
        }
        let with_team = storage.get("withTeam").unwrap();
        match with_team {
            AuthCredential::ApiKey { prime_team, .. } => {
                let team = prime_team.flatten().unwrap();
                assert_eq!(team.team_id, "t1");
                assert_eq!(team.name, "One");
                assert!(team.slug.is_none());
            }
            _ => panic!("expected api key"),
        }

        // Absent stays absent, null stays null when serialized again.
        let value = serde_json::to_value(storage.get("plain").unwrap()).unwrap();
        assert!(value.get("primeTeam").is_none());
        let value = serde_json::to_value(storage.get("withNull").unwrap()).unwrap();
        assert_eq!(value.get("primeTeam"), Some(&Value::Null));
    }

    #[test]
    fn prime_team_survives_key_overrides_in_isolated_process() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "core::auth_storage::tests::prime_team_override_fixture", "--ignored"])
            .env("PRIME_API_KEY", "synthetic-environment-key")
            .env_remove("PRIME_TEAM_ID")
            .status().unwrap();
        assert!(status.success());
    }

    #[test]
    #[ignore = "invoked in an isolated process with synthetic credentials"]
    fn prime_team_override_fixture() {
        let mut storage = memory(json!({"prime-inference": {
            "type":"api_key", "key":"stored-key", "primeTeam":{"teamId":"team-1","name":"One"}
        }}));
        storage.set_runtime_api_key("prime-inference", "synthetic-runtime-key");
        assert_eq!(storage.get_prime_inference_team_selection().flatten().unwrap().team_id, "team-1");
        assert_eq!(storage.get_provider_headers("prime-inference").unwrap()["X-Prime-Team-ID"], "team-1");
        storage.remove_runtime_api_key("prime-inference");
        assert_eq!(storage.get_auth_status("prime-inference").source.as_deref(), Some(AUTH_SOURCE_ENVIRONMENT));
        assert_eq!(storage.get_provider_headers("prime-inference").unwrap()["X-Prime-Team-ID"], "team-1");
        let mut personal = memory(json!({"prime-inference":{"type":"api_key","key":"stored-key","primeTeam":null}}));
        personal.set_runtime_api_key("prime-inference", "synthetic-runtime-key");
        assert_eq!(personal.get_prime_inference_team_selection(), Some(None));
        assert!(personal.get_provider_headers("prime-inference").is_none());
        std::env::set_var("PRIME_TEAM_ID", "explicit-team");
        assert_eq!(storage.get_prime_inference_team_selection(), None);
        assert_eq!(storage.get_provider_headers("prime-inference").unwrap()["X-Prime-Team-ID"], "explicit-team");
    }

    #[test]
    fn oauth_credentials_keep_extra_keys() {
        let storage = memory(json!({
            "anthropic": {"type": "oauth", "refresh": "r", "access": "a", "expires": 1, "accountId": "x"}
        }));
        match storage.get("anthropic").unwrap() {
            AuthCredential::OAuth { credentials } => {
                assert_eq!(credentials.refresh, "r");
                assert_eq!(credentials.access, "a");
                assert_eq!(
                    credentials.extra.get("accountId").and_then(Value::as_str),
                    Some("x")
                );
            }
            _ => panic!("expected oauth"),
        }
    }

    #[test]
    fn stored_credentials_report_status_and_has_auth() {
        let storage = memory(json!({"openai": {"type": "api_key", "key": "k"}}));
        assert!(storage.has("openai"));
        assert!(storage.has_auth("openai"));
        assert_eq!(storage.list(), vec!["openai".to_string()]);
        let status = storage.get_auth_status("openai");
        assert!(status.configured);
        assert_eq!(status.source.as_deref(), Some(AUTH_SOURCE_STORED));
        assert!(!storage.has_auth("unknown-provider"));
        assert_eq!(storage.get_auth_status("unknown-provider"), AuthStatus::default());
    }

    #[test]
    fn runtime_override_takes_precedence_and_can_be_removed() {
        let mut storage = memory(json!({"openai": {"type": "api_key", "key": "stored"}}));
        storage.set_runtime_api_key("openai", "runtime-key");
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.get_api_key_with_source_token("openai", true))
            .unwrap();
        assert_eq!(result.api_key.as_deref(), Some("runtime-key"));
        assert_eq!(
            result.source_token.map(|token| token.source),
            Some(AUTH_SOURCE_RUNTIME.to_string())
        );

        storage.remove_runtime_api_key("openai");
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.get_api_key_with_source_token("openai", true))
            .unwrap();
        assert_eq!(result.api_key.as_deref(), Some("stored"));
    }

    /// Restores the process environment when the test ends, including on panic.
    struct EnvRestore(Vec<(String, Option<String>)>);

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                match value {
                    Some(value) => std::env::set_var(&name, value),
                    None => std::env::remove_var(&name),
                }
            }
        }
    }

    /// Remove every ambient API-key variable for `provider`, so the only
    /// credential left is the one in this storage instance.
    ///
    /// `AuthStorage.hasAuth` accepts the environment candidate
    /// (`auth-storage.ts:757-759` -> `getAvailableAuthCandidate` ->
    /// `getEnvironmentAuthCandidate`, `auth-storage.ts:449-465` ->
    /// `getEnvApiKey`), and the candidate order is runtime, stored, environment,
    /// fallback (`auth-storage.ts:521-526`). So on a host where `OPENAI_API_KEY`
    /// is set, marking the *stored* credential stale leaves the environment
    /// credential selectable and the TS reports `environment`, exactly like the
    /// Rust port does. The TypeScript suite handles this the same way the tests
    /// below do: `auth-storage.test.ts:191-207` sets and restores `AWS_PROFILE`,
    /// and `model-registry.test.ts:1193-1194` deletes `OPENAI_API_KEY` before
    /// asserting that a provider has no auth.
    fn clear_ambient_provider_env(provider: &str) -> EnvRestore {
        let names: Vec<String> = api_key_env_vars(provider)
            .unwrap_or_default()
            .into_iter()
            .filter(|name| std::env::var_os(name).is_some())
            .map(|name| name.to_string())
            .collect();
        let saved: Vec<(String, Option<String>)> = names
            .iter()
            .map(|name| (name.clone(), std::env::var(name).ok()))
            .collect();
        for name in &names {
            std::env::remove_var(name);
        }
        EnvRestore(saved)
    }

    #[test]
    fn stale_marking_hides_a_source_until_cleared() {
        // Only the stored credential exists for this test, so the stale marking
        // must hide it and expose the stale status.
        let _env = clear_ambient_provider_env("openai");
        let mut storage = memory(json!({"openai": {"type": "api_key", "key": "k"}}));
        assert!(storage.mark_auth_stale("openai"));
        let status = storage.get_auth_status("openai");
        assert!(!status.configured);
        assert_eq!(status.source.as_deref(), Some(AUTH_SOURCE_STALE));
        assert_eq!(status.label.as_deref(), Some("expired"));
        assert!(!storage.has_auth("openai"));

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.get_api_key_with_source_token("openai", true))
            .unwrap();
        assert!(result.api_key.is_none());

        storage.clear_auth_stale("openai");
        assert!(storage.has_auth("openai"));
    }

    #[test]
    fn setting_and_removing_credentials_updates_status() {
        let mut storage = memory(json!({}));
        storage.set(
            "anthropic",
            AuthCredential::ApiKey {
                key: "k".to_string(),
                prime_team: None,
            },
        );
        assert!(storage.has("anthropic"));
        storage.remove("anthropic");
        assert!(!storage.has("anthropic"));
        assert!(storage.drain_errors().is_empty());
    }

    #[test]
    fn resolve_config_value_prefers_env_then_literal() {
        std::env::set_var("PRIME_AGENT_PORT_TEST_KEY", "from-env");
        assert_eq!(
            resolve_config_value("PRIME_AGENT_PORT_TEST_KEY").as_deref(),
            Some("from-env")
        );
        assert_eq!(
            resolve_config_value("not-a-set-variable-name").as_deref(),
            Some("not-a-set-variable-name")
        );
        std::env::set_var("PRIME_AGENT_PORT_TEST_EMPTY", "");
        assert_eq!(resolve_config_value("PRIME_AGENT_PORT_TEST_EMPTY"), None);
        std::env::remove_var("PRIME_AGENT_PORT_TEST_EMPTY");
        std::env::remove_var("PRIME_AGENT_PORT_TEST_KEY");
    }

    #[test]
    fn resolve_config_value_or_throw_reports_the_description() {
        assert_eq!(
            resolve_config_value_or_throw("literal-value", "API key").unwrap(),
            "literal-value"
        );
        let error = resolve_config_value_or_throw("!exit 1", "API key for provider \"x\"").unwrap_err();
        assert_eq!(
            error,
            "Failed to resolve API key for provider \"x\" from shell command: exit 1"
        );
    }

    #[cfg(windows)]
    fn fake_powershell_credential_helper(script: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credential-helper.ps1");
        std::fs::write(&path, format!("$ErrorActionPreference = 'Stop'\n{script}\n")).unwrap();
        let path = path.to_string_lossy().replace('\\', "/");
        let config = format!(
            "!powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"{path}\""
        );
        (dir, config)
    }

    #[cfg(windows)]
    #[test]
    fn auth_credential_helper_powershell_descendant_has_no_console_window() {
        let (_dir, config) = fake_powershell_credential_helper(
            r#"Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class CredentialConsoleProbe { [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); }'
if ([CredentialConsoleProbe]::GetConsoleWindow() -ne [IntPtr]::Zero) {
    throw 'Credential helper unexpectedly has a console window'
}
Write-Output 'isolated-test-key'"#,
        );
        assert_eq!(
            resolve_config_value_or_throw(&config, "isolated test credential").unwrap(),
            "isolated-test-key"
        );
    }

    #[cfg(windows)]
    #[test]
    fn auth_credential_helper_drains_output_larger_than_the_pipe_buffer() {
        let (_dir, config) = fake_powershell_credential_helper("[Console]::Out.Write(('x' * 1048576))");
        let value = resolve_config_value_uncached(&config).expect("large helper output must not deadlock");
        assert_eq!(value.len(), 1_048_576);
        assert!(value.bytes().all(|byte| byte == b'x'));
    }

    #[cfg(windows)]
    #[test]
    fn auth_credential_helper_shares_cache_but_uncached_refresh_still_runs() {
        let (_dir, config) = fake_powershell_credential_helper(
            r#"$counterPath = Join-Path $PSScriptRoot 'calls.txt'
$count = 0
if (Test-Path -LiteralPath $counterPath) { $count = [int](Get-Content -LiteralPath $counterPath -Raw) }
$count += 1
[IO.File]::WriteAllText($counterPath, [string]$count)
Write-Output ('isolated-test-key-' + $count)"#,
        );
        assert_eq!(resolve_config_value(&config).as_deref(), Some("isolated-test-key-1"));
        assert_eq!(resolve_config_value(&config).as_deref(), Some("isolated-test-key-1"));
        assert_eq!(
            crate::core::resolve_config_value::resolve_config_value(&config).as_deref(),
            Some("isolated-test-key-1")
        );
        assert_eq!(resolve_config_value_uncached(&config).as_deref(), Some("isolated-test-key-2"));
        assert_eq!(
            resolve_config_value_or_throw(&config, "isolated test credential").unwrap(),
            "isolated-test-key-3"
        );
    }

    #[cfg(windows)]
    #[test]
    fn auth_credential_helper_nonzero_exit_does_not_accept_stdout_as_a_key() {
        let (_dir, config) = fake_powershell_credential_helper("Write-Output 'not-a-valid-key'\nexit 9");
        assert_eq!(resolve_config_value_uncached(&config), None);
        let error = resolve_config_value_or_throw(&config, "isolated test credential").unwrap_err();
        assert!(error.starts_with("Failed to resolve isolated test credential from shell command:"));
    }

    #[test]
    fn resolve_headers_drops_empty_values() {
        std::env::set_var("PRIME_AGENT_PORT_TEST_HEADER", "");
        let mut headers = IndexMap::new();
        headers.insert("a".to_string(), "literal".to_string());
        headers.insert("b".to_string(), "PRIME_AGENT_PORT_TEST_HEADER".to_string());
        let resolved = resolve_headers(Some(&headers)).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.get("a").map(String::as_str), Some("literal"));
        assert!(resolve_headers(None).is_none());
        std::env::remove_var("PRIME_AGENT_PORT_TEST_HEADER");
    }

    #[test]
    fn fallback_resolver_is_used_last() {
        let mut storage = memory(json!({}));
        storage.set_fallback_resolver(Arc::new(|provider: &str| {
            if provider == "custom" {
                Some("fallback-key".to_string())
            } else {
                None
            }
        }));
        assert!(storage.has_auth("custom"));
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.get_api_key_with_source_token("custom", true))
            .unwrap();
        assert_eq!(result.api_key.as_deref(), Some("fallback-key"));
        assert_eq!(
            result.source_token.map(|token| token.source),
            Some(AUTH_SOURCE_FALLBACK.to_string())
        );

        // includeFallback = false skips the resolver.
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.get_api_key_with_source_token("custom", false))
            .unwrap();
        assert!(result.api_key.is_none());
    }

    #[test]
    fn file_backend_persists_and_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = AuthStorage::create(Some(path_string.clone()), None);
        storage.set(
            "openai",
            AuthCredential::ApiKey {
                key: "persisted".to_string(),
                prime_team: None,
            },
        );
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("persisted"));

        let reloaded = AuthStorage::create(Some(path_string), None);
        match reloaded.get("openai").unwrap() {
            AuthCredential::ApiKey { key, .. } => assert_eq!(key, "persisted"),
            _ => panic!("expected api key"),
        }
    }

    #[test]
    fn remove_verified_deletes_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = AuthStorage::create(Some(path_string.clone()), None);
        storage.set(
            "openai",
            AuthCredential::ApiKey {
                key: "persisted".to_string(),
                prime_team: None,
            },
        );
        storage.remove_verified("openai").unwrap();
        assert!(!storage.has("openai"));
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(!written.contains("persisted"));
        // Idempotent.
        storage.remove_verified("openai").unwrap();
    }

    #[test]
    fn oauth_registry_registers_and_resets() {
        reset_oauth_providers();
        assert!(get_oauth_provider("openai-codex").is_some());
        let provider = OAuthProviderInterface {
            id: "test-provider".to_string(),
            name: "Test".to_string(),
            login: Arc::new(|_callbacks: OAuthLoginCallbacks| {
                Box::pin(async { Ok(OAuthCredentials::default()) })
                    as pi_ai::types::BoxFuture<Result<OAuthCredentials, String>>
            }),
            uses_callback_server: None,
            refresh_token: Arc::new(|credentials: OAuthCredentials| {
                Box::pin(async move { Ok(credentials) })
                    as pi_ai::types::BoxFuture<Result<OAuthCredentials, String>>
            }),
            get_api_key: Arc::new(|credentials: &OAuthCredentials| credentials.access.clone()),
            modify_models: None,
        };
        register_oauth_provider(provider);
        assert!(get_oauth_provider_info_list().contains(&OAuthProviderInfo {
            id: "test-provider".to_string(),
            name: "Test".to_string(),
            available: true,
        }));
        assert_eq!(get_oauth_provider("test-provider").unwrap().name, "Test");
        unregister_oauth_provider("test-provider");
        assert!(get_oauth_provider("test-provider").is_none());
        register_builtin_mcp_oauth_providers();
        assert!(get_oauth_provider("mcp:linear").is_some());
        assert!(get_oauth_provider("openai-codex").is_some());
    }

    #[test]
    fn login_reports_unknown_provider() {
        reset_oauth_providers();
        let mut storage = memory(json!({}));
        let error = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(storage.login("nope", OAuthLoginCallbacks::default()))
            .unwrap_err();
        assert_eq!(error, "Unknown OAuth provider: nope");
    }

    #[test]
    fn env_keys_match_the_typescript_table() {
        assert_eq!(
            api_key_env_vars("github-copilot"),
            Some(vec![
                "COPILOT_GITHUB_TOKEN",
                "GH_TOKEN",
                "GITHUB_TOKEN"
            ])
        );
        assert_eq!(
            api_key_env_vars("anthropic"),
            Some(vec![
                "ANTHROPIC_OAUTH_TOKEN",
                "ANTHROPIC_API_KEY"
            ])
        );
        assert!(api_key_env_vars("openai-codex").is_none());
    }

    #[test]
    fn login_callbacks_type_is_usable() {
        let callbacks = OAuthLoginCallbacks {
            on_select: Some(Arc::new(|_prompt: OAuthSelectPrompt| {
                Box::pin(async { Some("id".to_string()) })
                    as pi_ai::types::BoxFuture<Option<String>>
            })),
            ..Default::default()
        };
        assert!(callbacks.on_select.is_some());
    }

    /// OAUTH-3 / C3-01 (`packages/coding-agent/src/core/auth-storage.ts:649-654`):
    /// TS loads auth.json with `JSON.parse(content) as AuthStorageData`, a bare
    /// cast that keeps every entry, and `packages/ai/src/utils/oauth/anthropic.ts:375`
    /// itself persists `refresh: data.refresh_token`, which `JSON.stringify` drops
    /// when a refresh response omits `refresh_token`. So an oauth entry without
    /// `refresh` must not cost us the other providers.
    #[test]
    fn an_oauth_entry_without_refresh_does_not_cost_the_other_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let path_string = path.to_string_lossy().to_string();
        let original = concat!(
            "{\n",
            "  \"openai\": {\"type\": \"api_key\", \"key\": \"sk-good\"},\n",
            "  \"anthropic\": {\"type\": \"oauth\", \"access\": \"at\", \"expires\": 123}\n",
            "}"
        );
        std::fs::write(&path, original).unwrap();

        let mut storage = AuthStorage::create(Some(path_string.clone()), None);
        assert!(storage.load_error.is_none(), "{:?}", storage.load_error);
        assert!(storage.drain_errors().is_empty());

        match storage.get("openai") {
            Some(AuthCredential::ApiKey { key, .. }) => assert_eq!(key, "sk-good"),
            other => panic!("api_key entry lost on load: {:?}", other),
        }
        match storage.get("anthropic") {
            Some(AuthCredential::OAuth { credentials }) => {
                assert_eq!(credentials.access, "at");
                assert_eq!(credentials.expires, 123.0);
                assert_eq!(credentials.refresh, "");
            }
            other => panic!("oauth entry lost on load: {:?}", other),
        }

        // A later save of a different provider keeps both entries on disk.
        storage.set(
            "github-copilot",
            AuthCredential::ApiKey {
                key: "gh".to_string(),
                prime_team: None,
            },
        );
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["openai"]["key"], Value::from("sk-good"));
        assert_eq!(written["anthropic"]["access"], Value::from("at"));
        assert_eq!(written["github-copilot"]["key"], Value::from("gh"));
    }

    /// C3-01: `reload` must set `loadError` on a bad load
    /// (auth-storage.ts:659-672) and `persistProviderChange` must return early
    /// while it stands (auth-storage.ts:674-677), so a malformed parse can never
    /// lead to a rewrite of auth.json without the entries it could not read.
    #[test]
    fn an_unreadable_entry_blocks_writes_instead_of_erasing_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let path_string = path.to_string_lossy().to_string();
        let original = concat!(
            "{\n",
            "  \"openai\": {\"type\": \"api_key\", \"key\": \"sk-good\"},\n",
            "  \"broken\": {\"type\": \"api_key\"}\n",
            "}"
        );
        std::fs::write(&path, original).unwrap();

        let mut storage = AuthStorage::create(Some(path_string.clone()), None);

        // The well-formed api_key entry survives the load.
        match storage.get("openai") {
            Some(AuthCredential::ApiKey { key, .. }) => assert_eq!(key, "sk-good"),
            other => panic!("api_key entry lost on load: {:?}", other),
        }

        // A load error is recorded and blocks writes.
        let errors = storage.drain_errors();
        assert!(
            errors.iter().any(|error| error.contains("broken")),
            "no load error recorded for the unreadable entry: {:?}",
            errors
        );
        assert!(storage.load_error.is_some());
        storage.set(
            "anthropic",
            AuthCredential::ApiKey {
                key: "replacement".to_string(),
                prime_team: None,
            },
        );
        storage.remove("openai");

        // Writes are blocked, so the unreadable entry cannot be erased on disk.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

    /// C3-02: a crashed instance leaves `<auth>.lock` behind; `proper-lockfile`
    /// steals it once it is older than `stale` (auth-storage.ts:217-230), and the
    /// sync path uses the same takeover (auth-storage.ts:152-157), so auth writes
    /// must not stay bricked until someone deletes the lock by hand.
    #[test]
    fn a_stale_lock_from_a_crashed_instance_is_stolen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = AuthStorage::create(Some(path_string.clone()), None);
        storage.set(
            "openai",
            AuthCredential::ApiKey {
                key: "first".to_string(),
                prime_team: None,
            },
        );

        // Simulate the corpse of a crashed writer: the lock is 60s old.
        let lock_path = format!("{}.lock", path_string);
        std::fs::create_dir(&lock_path).unwrap();
        set_directory_mtime_to_now_minus(&lock_path, 60);

        storage.set(
            "github-copilot",
            AuthCredential::ApiKey {
                key: "second".to_string(),
                prime_team: None,
            },
        );

        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["openai"]["key"], Value::from("first"));
        assert_eq!(written["github-copilot"]["key"], Value::from("second"));
        assert!(!Path::new(&lock_path).exists());
    }
}
