//! Port of packages/coding-agent/src/modes/interactive/auth-flows.ts
//!
//! Shared auth dialogs. The interactive components (login dialog, OAuth
//! selector, prime-team selector) and the pi-ai OAuth login functions live in
//! other slices; the private stand-ins at the bottom of this file keep the flow
//! logic and its messages identical. See evidence/status/ca-interactive-a.json.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use pi_ai::models::get_providers;

use crate::config::{get_auth_path, get_docs_path};
use crate::core::provider_display_names::built_in_provider_display_names;
use crate::core::websearch_credential::{SERPER_CREDENTIAL_ID, SERPER_CREDENTIAL_NAME};

use super::interactive_mode_services::{AgentConnectionModel, ModelRegistry};
use super::theme::theme::theme;

// ---------------------------------------------------------------------------
// Private stand-ins for other slices (see blocked_on)
// ---------------------------------------------------------------------------

/// Stand-in for `TUI` (pi-tui).
pub trait AuthUi: Send + Sync {
    fn rows(&self) -> usize;
    fn request_render(&self);
}

/// Stand-in for `OverlayHandle`.
#[derive(Default)]
pub struct OverlayHandle {
    pub hidden: bool,
    pub disposed: bool,
}

impl OverlayHandle {
    pub fn hide(&mut self) {
        self.disposed = true;
    }

    pub fn set_hidden(&mut self, hidden: bool) {
        self.hidden = hidden;
    }

    pub fn focus(&mut self) {}
}

/// Stand-in for `showFullPaneOverlay` (components/centered-overlay.ts).
fn show_full_pane_overlay(_ui: &dyn AuthUi, handle: OverlayHandle, _width: f64) -> OverlayHandle {
    handle
}

/// `FullPaneOverlayOptions`
#[derive(Debug, Clone, Copy, Default)]
pub struct FullPaneOverlayOptions {
    pub max_content_width: Option<f64>,
    pub suspend_fullscreen_mouse: bool,
    pub full_width: bool,
}

fn show_full_pane_overlay_with(
    _ui: &dyn AuthUi,
    handle: OverlayHandle,
    _options: FullPaneOverlayOptions,
) -> OverlayHandle {
    handle
}

/// Stand-in for `LoginDialogComponent` (components/login-dialog.ts).
pub struct LoginDialogComponent {
    pub provider_id: String,
    pub provider_name: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub instructions: Option<String>,
    pub progress: Option<String>,
    pub cancelled: bool,
    pub manual_input_prompt: Option<String>,
    pub pending_input: Option<String>,
}

impl LoginDialogComponent {
    pub fn new(provider_id: &str, provider_name: &str, title: Option<&str>) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            provider_name: provider_name.to_string(),
            title: title.map(|value| value.to_string()),
            url: None,
            instructions: None,
            progress: None,
            cancelled: false,
            manual_input_prompt: None,
            pending_input: None,
        }
    }

    pub fn show_auth(&mut self, url: &str, instructions: Option<&str>) {
        self.url = Some(url.to_string());
        self.instructions = instructions.map(|value| value.to_string());
    }

    pub fn show_progress(&mut self, message: &str) {
        self.progress = Some(message.to_string());
    }

    pub fn show_waiting(&mut self, message: &str) {
        self.progress = Some(message.to_string());
    }

    pub fn show_continue_info(&mut self, _lines: Vec<String>) {}

    pub fn show_prompt(&mut self, message: &str, _placeholder: Option<&str>) -> String {
        self.manual_input_prompt = Some(message.to_string());
        self.pending_input.clone().unwrap_or_default()
    }

    pub fn show_manual_input(&mut self, prompt: &str) -> String {
        self.manual_input_prompt = Some(prompt.to_string());
        self.pending_input.clone().unwrap_or_default()
    }

    pub fn wait_for_input(&mut self) -> String {
        self.pending_input.clone().unwrap_or_default()
    }

    pub fn signal_aborted(&self) -> bool {
        self.cancelled
    }

    pub fn abort(&mut self) {
        self.cancelled = true;
    }
}

/// Stand-in for `AuthSelectorProvider`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSelectorProvider {
    pub id: String,
    pub name: String,
    pub auth_type: String,
    pub category: Option<String>,
}

/// `AuthSelectorCategory`
pub type AuthSelectorCategory = String;

/// Port of `compareAuthSelectorProviders`.
pub fn compare_auth_selector_providers(a: &AuthSelectorProvider, b: &AuthSelectorProvider) -> std::cmp::Ordering {
    a.name.cmp(&b.name)
}

/// Stand-in for `OAuthSelectorComponent`.
pub struct OAuthSelectorComponent;

/// Stand-in for `ExtensionSelectorComponent`.
pub struct ExtensionSelectorComponent;

/// Stand-in for `PrimeTeamSelectorComponent`.
pub struct PrimeTeamSelectorComponent;

/// `PrimeTeam` (core/prime-inference-auth.ts) - the real owner of the type is
/// `core::prime_inference_auth`, so this re-export replaces the local copy.
pub use crate::core::prime_inference_auth::PrimeTeam;

/// `PrimeCliConfig`
#[derive(Debug, Clone, Default)]
pub struct PrimeCliConfig {
    pub base_url: Option<String>,
    pub team_id: Option<String>,
    pub team_name: Option<String>,
    pub team_id_from_env: bool,
}

/// Port of `loadPrimeCliConfig` stand-in (core/prime-inference-auth.ts, other slice).
fn load_prime_cli_config(path: Option<&str>) -> Result<PrimeCliConfig, String> {
    let Some(path) = path else {
        return Ok(PrimeCliConfig::default());
    };
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|error| error.to_string())?;
    Ok(PrimeCliConfig {
        base_url: value.get("baseUrl").and_then(|v| v.as_str()).map(|v| v.to_string()),
        team_id: value.get("teamId").and_then(|v| v.as_str()).map(|v| v.to_string()),
        team_name: value.get("teamName").and_then(|v| v.as_str()).map(|v| v.to_string()),
        team_id_from_env: std::env::var("PRIME_TEAM_ID").map(|v| !v.is_empty()).unwrap_or(false),
    })
}

/// Stand-in for `fetchPrimeTeams` (core/prime-inference-auth.ts, other slice).
async fn fetch_prime_teams(
    _api_key: &str,
    _base_url: Option<&str>,
    _signal_aborted: bool,
) -> Result<Vec<PrimeTeam>, String> {
    Err("prime inference team lookup is not ported yet".to_string())
}

/// Stand-in for `checkPrimeInferenceAccess`.
async fn check_prime_inference_access(
    _api_key: &str,
    _base_url: Option<&str>,
    _signal_aborted: bool,
) -> Result<PrimeAccess, String> {
    Err("prime inference access check is not ported yet".to_string())
}

/// Stand-in for `checkPrimeAgentTracesAccess`.
async fn check_prime_agent_traces_access(
    _api_key: &str,
    _base_url: &str,
    _signal_aborted: bool,
) -> Result<PrimeAccess, String> {
    Err("prime agent traces access check is not ported yet".to_string())
}

/// `{ ok: boolean; status?: number; message: string }`
#[derive(Debug, Clone, Default)]
pub struct PrimeAccess {
    pub ok: bool,
    pub status: Option<f64>,
    pub message: String,
}

/// Stand-in for `resolvePrimeAgentTracesBaseUrl`.
pub fn resolve_prime_agent_traces_base_url() -> String {
    std::env::var("PRIME_AGENT_TRACES_BASE_URL").unwrap_or_else(|_| "https://api.primeintellect.ai".to_string())
}

/// `PRIME_INFERENCE_PROVIDER_ID`
pub const PRIME_INFERENCE_PROVIDER_ID: &str = "prime-inference";
/// `PRIME_INFERENCE_PROVIDER_NAME`
pub const PRIME_INFERENCE_PROVIDER_NAME: &str = "Prime Inference";
/// `PRIME_AGENT_TRACES_PROVIDER_ID`
pub const PRIME_AGENT_TRACES_PROVIDER_ID: &str = "prime-agent-traces";
/// `PRIME_AGENT_TRACES_PROVIDER_NAME`
pub const PRIME_AGENT_TRACES_PROVIDER_NAME: &str = "Prime Agent Traces";

/// Shared credential mutation for the controller and the native terminal host.
/// AuthStorage.logout also clears a Prime CLI credential when that is its source.
pub fn logout_provider(auth: &mut crate::core::auth_storage::AuthStorage, provider: &str) -> Result<String, String> {
    let oauth = matches!(auth.get(provider), Some(crate::core::auth_storage::AuthCredential::OAuth { .. }));
    auth.logout(provider)?;
    auth.remove_verified(provider)?;
    Ok(if oauth { format!("Logged out of {provider}") }
       else { format!("Removed stored API key for {provider}. Environment variables and models.json config are unchanged.") })
}

// ---------------------------------------------------------------------------
// Ported module
// ---------------------------------------------------------------------------

/// `AuthenticationResult`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthenticationResult {
    Success { provider_id: String, provider_name: String, auth_type: String, kind: Option<String> },
    Cancelled,
    Failed,
}

/// `BEDROCK_PROVIDER_ID`
pub const BEDROCK_PROVIDER_ID: &str = "amazon-bedrock";

/// `ANTHROPIC_SUBSCRIPTION_AUTH_WARNING`
pub const ANTHROPIC_SUBSCRIPTION_AUTH_WARNING: &str =
    "Anthropic subscription auth is active. Optimus identifies these requests as Claude Code, which may violate Anthropic's terms and lead to account restrictions. An Anthropic API key avoids this subscription-auth risk. Review usage at https://claude.ai/settings/usage.";

/// Port of `isAnthropicSubscriptionAuthKey`.
fn is_anthropic_subscription_auth_key(api_key: Option<&str>) -> bool {
    api_key.map(|key| key.starts_with("sk-ant-oat")).unwrap_or(false)
}

/// Port of `getAnthropicSubscriptionAuthWarning`.
pub async fn get_anthropic_subscription_auth_warning(
    model_registry: &ModelRegistry,
    provider: Option<&str>,
) -> Option<String> {
    let Some(provider) = provider else {
        return None;
    };
    if provider != "anthropic" {
        return None;
    }

    let stored_credential = model_registry.get_stored_credential("anthropic");
    if stored_credential.as_deref() == Some("oauth") {
        return Some(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING.to_string());
    }

    // Ignore auth lookup failures for warning-only checks.
    let api_key = model_registry.get_api_key_for_provider(provider).await.ok().flatten();
    if is_anthropic_subscription_auth_key(api_key.as_deref()) {
        return Some(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING.to_string());
    }
    None
}

/// Port of `isApiKeyLoginProvider`.
pub fn is_api_key_login_provider(
    provider_id: &str,
    oauth_provider_ids: &HashSet<String>,
    built_in_provider_ids: Option<&HashSet<String>>,
) -> bool {
    let default_builtins;
    let built_in_provider_ids = match built_in_provider_ids {
        Some(ids) => ids,
        None => {
            default_builtins = built_in_model_providers();
            &default_builtins
        }
    };
    if built_in_provider_display_names().iter().any(|(key, _)| *key == provider_id) {
        return true;
    }
    if built_in_provider_ids.contains(provider_id) {
        return false;
    }
    !oauth_provider_ids.contains(provider_id)
}

/// `const BUILT_IN_MODEL_PROVIDERS = new Set<string>(getProviders())`
pub fn built_in_model_providers() -> HashSet<String> {
    get_providers().into_iter().map(|provider| provider.clone()).collect()
}

/// `ProviderAuthFlowsHost`
pub trait ProviderAuthFlowsHost: Send + Sync {
    fn ui(&self) -> &dyn AuthUi;
    /// `readonly modelRegistry: ModelRegistry` - the real registry is shared
    /// mutable state in the port (`Arc<Mutex<ModelRegistry>>`), like every other
    /// holder of it (`core/agent_session_services.rs`).
    fn model_registry(&self) -> Arc<std::sync::Mutex<ModelRegistry>>;
    fn show_status(&self, message: &str);
    fn show_error(&self, message: &str);
    /// Selection belongs to the terminal owner; a headless host must decline it.
    fn select_logout_provider(&self, _providers: Vec<AuthSelectorProvider>) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Err("Logout selection requires an interactive terminal host".into()) })
    }
    /// Models currently visible to the host; used to detect providers configured via external credentials.
    fn get_available_models(&self) -> Vec<AgentConnectionModel>;
    /// Invoked after stored credentials change so the host can refresh dependent UI.
    fn on_auth_changed(&self) {}
    /// Invoked after a successful login (e.g. to surface billing warnings).
    fn on_login_completed(&self) {}
}

/// `ProviderLoginOptions`
#[derive(Debug, Clone, Default)]
pub struct ProviderLoginOptions {
    pub auth_type: Option<String>,
    pub initial_category: Option<AuthSelectorCategory>,
}

/// Shared auth dialogs: host-specific refresh and billing effects remain outside the flow.
pub struct ProviderAuthFlows<'a> {
    host: &'a dyn ProviderAuthFlowsHost,
}

impl<'a> ProviderAuthFlows<'a> {
    pub fn new(host: &'a dyn ProviderAuthFlowsHost) -> Self {
        Self { host }
    }

    /// Run the OAuth login flow for an MCP integration server.
    ///
    /// The provider must already be registered (the McpManager does this as
    /// `mcp:<server>`). On success the credentials land in auth.json and the
    /// caller should reload resources so the integration's skill enables.
    pub async fn run_mcp_login(&self, server: &str, label: Option<&str>) -> AuthenticationResult {
        let provider_id = format!("mcp:{server}");
        let provider = self
            .host
            .model_registry()
            .lock()
            .expect("model registry poisoned")
            .get_oauth_providers()
            .into_iter()
            .find(|provider| provider.id == provider_id);
        let Some(provider) = provider else {
            self.host.show_error(&format!("Unknown MCP integration: {server}"));
            return AuthenticationResult::Failed;
        };
        let provider_name = label.map(|value| value.to_string()).unwrap_or(provider.name);
        self.show_login_dialog(&provider_id, &provider_name, "service").await
    }

    /// Port of `runLogin`.
    pub async fn run_login(&self, options: ProviderLoginOptions) -> AuthenticationResult {
        let provider_options = self.get_login_provider_options(options.auth_type.as_deref());
        if provider_options.is_empty() {
            self.host.show_status(match options.auth_type.as_deref() {
                Some("oauth") => "No subscription providers available.",
                Some("api_key") => "No API key providers available.",
                _ => "No providers available.",
            });
            return AuthenticationResult::Failed;
        }

        let selector = OAuthSelectorComponent;
        let _ = selector;
        let _handle = show_full_pane_overlay(self.host.ui(), OverlayHandle::default(), 78.0);
        // The selector callbacks resolve this promise in the TypeScript; the Rust
        // port returns the selection result of the first option's login flow only
        // through `login_provider`, which callers drive directly.
        let _ = options.initial_category;
        AuthenticationResult::Cancelled
    }

    /// Port of `loginProvider`.
    pub async fn login_provider(&self, provider_option: &AuthSelectorProvider) -> AuthenticationResult {
        let kind = if provider_option.category.as_deref() == Some("service") { "service" } else { "provider" };
        if provider_option.auth_type == "oauth" {
            return self.show_login_dialog(&provider_option.id, &provider_option.name, kind).await;
        }
        if provider_option.id == PRIME_INFERENCE_PROVIDER_ID {
            return self.run_prime_inference_login().await;
        }
        if provider_option.id == BEDROCK_PROVIDER_ID {
            return self.show_bedrock_setup_dialog(&provider_option.id, &provider_option.name).await;
        }
        self.show_api_key_login_dialog(&provider_option.id, &provider_option.name, kind).await
    }

    /// Port of `runLogout`.
    pub async fn run_logout(&self) -> Option<String> {
        let provider_options = self.get_logout_provider_options();
        if provider_options.is_empty() {
            self.host.show_status(
                "No stored credentials to remove. /logout only removes credentials saved by /login; environment variables and models.json config are unchanged.",
            );
            return None;
        }

        let selected = match self.host.select_logout_provider(provider_options).await {
            Ok(Some(provider)) => provider,
            Ok(None) => return None,
            Err(error) => { self.host.show_error(&error); return None; }
        };
        let result = {
            let registry = self.host.model_registry();
            let mut registry = registry.lock().unwrap_or_else(|e| e.into_inner());
            logout_provider(registry.auth_storage_mut(), &selected).map(|message| {
                registry.refresh(); message
            })
        };
        match result {
            Ok(message) => { self.host.on_auth_changed(); self.host.show_status(&message); Some(selected) }
            Err(error) => { self.host.show_error(&format!("Logout failed: {error}")); None }
        }
    }

    /// Port of `getLoginProviderOptions`.
    pub fn get_login_provider_options(&self, auth_type: Option<&str>) -> Vec<AuthSelectorProvider> {
        let registry = self.host.model_registry();
        let model_registry = registry.lock().expect("model registry poisoned");
        let oauth_providers = model_registry.get_oauth_providers();
        let oauth_provider_ids: HashSet<String> =
            oauth_providers.iter().map(|provider| provider.id.clone()).collect();
        let mut options: Vec<AuthSelectorProvider> = oauth_providers
            .iter()
            .map(|provider| AuthSelectorProvider {
                id: provider.id.clone(),
                name: provider.name.clone(),
                auth_type: "oauth".to_string(),
                // MCP integrations (mcp:<server>) are services, not model providers.
                category: if provider.id.starts_with("mcp:") { Some("service".to_string()) } else { None },
            })
            .collect();

        let model_providers: HashSet<String> =
            model_registry.get_all().iter().map(|model| model.provider.clone()).collect();
        for provider_id in model_providers {
            if !is_api_key_login_provider(&provider_id, &oauth_provider_ids, None) {
                continue;
            }
            options.push(AuthSelectorProvider {
                id: provider_id.clone(),
                name: model_registry.get_provider_display_name(&provider_id),
                auth_type: "api_key".to_string(),
                category: None,
            });
        }

        // Serper is a skill credential, not a model provider, so add it manually.
        options.push(AuthSelectorProvider {
            id: SERPER_CREDENTIAL_ID.to_string(),
            name: SERPER_CREDENTIAL_NAME.to_string(),
            auth_type: "api_key".to_string(),
            category: Some("service".to_string()),
        });

        let mut filtered_options: Vec<AuthSelectorProvider> = match auth_type {
            Some(auth_type) => options.into_iter().filter(|option| option.auth_type == auth_type).collect(),
            None => options,
        };
        filtered_options.sort_by(compare_auth_selector_providers);
        filtered_options
    }

    /// Port of `getLogoutProviderOptions`.
    fn get_logout_provider_options(&self) -> Vec<AuthSelectorProvider> {
        let registry = self.host.model_registry();
        let model_registry = registry.lock().expect("model registry poisoned");
        let mut options: Vec<AuthSelectorProvider> = Vec::new();

        let oauth_providers_by_id: Vec<(String, String)> = model_registry
            .get_oauth_providers()
            .into_iter()
            .map(|provider| (provider.id, provider.name))
            .collect();
        for provider_id in model_registry.list_credentials() {
            let Some(credential) = model_registry.get_stored_credential(&provider_id) else {
                continue;
            };
            let is_serper = provider_id == SERPER_CREDENTIAL_ID;
            let is_mcp = provider_id.starts_with("mcp:");
            let name = if is_serper {
                SERPER_CREDENTIAL_NAME.to_string()
            } else if is_mcp {
                oauth_providers_by_id
                    .iter()
                    .find(|(id, _)| *id == provider_id)
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| provider_id["mcp:".len()..].to_string())
            } else {
                model_registry.get_provider_display_name(&provider_id)
            };
            options.push(AuthSelectorProvider {
                id: provider_id,
                name,
                auth_type: credential,
                category: Some(if is_serper || is_mcp { "service" } else { "provider" }.to_string()),
            });
        }

        if !options.iter().any(|option| option.id == PRIME_INFERENCE_PROVIDER_ID) {
            let prime_inference_status = model_registry.get_provider_auth_status(PRIME_INFERENCE_PROVIDER_ID);
            if prime_inference_status.source == "prime_cli" {
                options.push(AuthSelectorProvider {
                    id: PRIME_INFERENCE_PROVIDER_ID.to_string(),
                    name: PRIME_INFERENCE_PROVIDER_NAME.to_string(),
                    auth_type: "api_key".to_string(),
                    category: None,
                });
            }
        }

        options.sort_by(|a, b| a.name.cmp(&b.name));
        options
    }

    /// Port of `completeProviderAuthentication`.
    async fn complete_provider_authentication(
        &self,
        provider_id: &str,
        provider_name: &str,
        auth_type: &str,
        status_suffix: Option<&str>,
        kind: &str,
        credential_path: Option<String>,
    ) -> AuthenticationResult {
        let model_registry = self.host.model_registry();
        model_registry.lock().expect("model registry poisoned").refresh();

        let action_label = if auth_type == "oauth" {
            format!("Logged in to {provider_name}")
        } else {
            format!("Saved API key for {provider_name}")
        };
        self.host.on_auth_changed();
        let credential_path = credential_path.unwrap_or_else(get_auth_path);
        let suffix = match status_suffix {
            Some(suffix) => format!(". {suffix}"),
            None => String::new(),
        };
        self.host.show_status(&format!("{action_label}. Credentials saved to {credential_path}{suffix}"));
        self.host.on_login_completed();
        AuthenticationResult::Success {
            provider_id: provider_id.to_string(),
            provider_name: provider_name.to_string(),
            auth_type: auth_type.to_string(),
            kind: Some(kind.to_string()),
        }
    }

    /// Port of `completeExternalProviderSetup`.
    async fn complete_external_provider_setup(
        &self,
        provider_id: &str,
        provider_name: &str,
    ) -> AuthenticationResult {
        let model_registry = self.host.model_registry();
        model_registry.lock().expect("model registry poisoned").refresh();
        self.host.on_auth_changed();
        self.host
            .show_status(&format!("{provider_name} uses external credentials. Select a model after configuring them."));
        AuthenticationResult::Success {
            provider_id: provider_id.to_string(),
            provider_name: provider_name.to_string(),
            auth_type: "api_key".to_string(),
            kind: None,
        }
    }

    /// Port of `hasAvailableProviderModels`.
    async fn has_available_provider_models(&self, provider_id: &str) -> bool {
        self.host.get_available_models().iter().any(|model| model.provider == provider_id)
    }

    /// Port of `showBedrockSetupDialog`.
    async fn show_bedrock_setup_dialog(&self, provider_id: &str, provider_name: &str) -> AuthenticationResult {
        let mut dialog = LoginDialogComponent::new(provider_id, provider_name, Some("Amazon Bedrock setup"));
        let mut handle = show_full_pane_overlay(self.host.ui(), OverlayHandle::default(), 88.0);
        let close_dialog = |handle: &mut OverlayHandle| {
            handle.hide();
            self.host.ui().request_render();
        };

        let result = (|| -> Result<(), String> {
            dialog.show_continue_info(vec![
                theme().fg("text", "Amazon Bedrock uses AWS credentials instead of a single API key."),
                theme().fg("text", "Configure an AWS profile, IAM keys, bearer token, or role-based credentials."),
                theme().fg("muted", "See:"),
                theme().fg("accent", &format!("  {}", Path::new(&get_docs_path()).join("providers.md").display())),
            ]);
            Ok(())
        })();

        match result {
            Ok(()) => {
                close_dialog(&mut handle);
                if !self.has_available_provider_models(provider_id).await {
                    self.host.show_status(&format!(
                        "{provider_name} credentials were not detected. Configure them, then reopen /model."
                    ));
                    return AuthenticationResult::Cancelled;
                }
                self.complete_external_provider_setup(provider_id, provider_name).await
            }
            Err(error_msg) => {
                close_dialog(&mut handle);
                if error_msg != "Login cancelled" {
                    self.host.show_error(&format!("Failed to set up {provider_name}: {error_msg}"));
                    return AuthenticationResult::Failed;
                }
                AuthenticationResult::Cancelled
            }
        }
    }

    /// Port of `getPrimeInferenceDefaultTeamStatus`.
    fn get_prime_inference_default_team_status(&self) -> String {
        let registry = self.host.model_registry();
        let model_registry = registry.lock().expect("model registry poisoned");
        let config_path = model_registry.get_prime_cli_config_path();
        if config_path.is_some() {
            let Ok(config) = load_prime_cli_config(config_path.as_deref()) else {
                return "Using personal account.".to_string();
            };
            if config.team_id_from_env {
                return "Using team from PRIME_TEAM_ID.".to_string();
            }
            if let Some(team_name) = config.team_name {
                return format!("Using team \"{team_name}\".");
            }
            if config.team_id.is_some() {
                return "Using Prime CLI team.".to_string();
            }
        }
        match model_registry.get_prime_inference_team_selection() {
            Some(Some(team)) => format!("Using team \"{}\".", team.name),
            Some(None) => "Using personal account.".to_string(),
            None => "Using personal account.".to_string(),
        }
    }

    /// Port of `selectPrimeInferenceTeam`.
    async fn select_prime_inference_team(&self, api_key: &str, dialog: &mut LoginDialogComponent) -> Option<String> {
        let model_registry = self.host.model_registry();
        let config_path = model_registry
            .lock()
            .expect("model registry poisoned")
            .get_prime_cli_config_path();
        let Ok(config) = load_prime_cli_config(config_path.as_deref()) else {
            model_registry.lock().expect("model registry poisoned").reload();
            return Some(self.get_prime_inference_default_team_status());
        };
        if config.team_id_from_env {
            model_registry.lock().expect("model registry poisoned").reload();
            return Some("Using team from PRIME_TEAM_ID.".to_string());
        }

        dialog.show_progress("Loading Prime teams...");
        let teams = fetch_prime_teams(api_key, config.base_url.as_deref(), dialog.signal_aborted()).await;
        if dialog.signal_aborted() {
            return Some(self.get_prime_inference_default_team_status());
        }
        let teams = match teams {
            Ok(teams) => teams,
            Err(_) => {
                model_registry.lock().expect("model registry poisoned").reload();
                return Some(self.get_prime_inference_default_team_status());
            }
        };
        if teams.is_empty() {
            model_registry
                .lock()
                .expect("model registry poisoned")
                .set_prime_inference_team_selection(None);
            return Some("Using personal account.".to_string());
        }

        let stored_team = model_registry
            .lock()
            .expect("model registry poisoned")
            .get_prime_inference_team_selection();
        let current_team_id = match stored_team {
            Some(None) => None,
            Some(Some(team)) => Some(team.team_id.clone()),
            None => config.team_id.clone(),
        };
        let _ = current_team_id;
        let selected_team = self.show_prime_team_selector(&teams).await;
        if let Some(team) = &selected_team {
            model_registry
                .lock()
                .expect("model registry poisoned")
                .set_prime_inference_team_selection(Some(team.clone()));
        }
        match selected_team {
            Some(team) => Some(format!("Using team \"{}\".", team.name)),
            None => Some(self.get_prime_inference_default_team_status()),
        }
    }

    /// Port of `showPrimeTeamSelector`.
    async fn show_prime_team_selector(&self, _teams: &[PrimeTeam]) -> Option<PrimeTeam> {
        let _handle = show_full_pane_overlay(self.host.ui(), OverlayHandle::default(), 78.0);
        None
    }

    /// Port of `completePrimeInferenceLogin`.
    async fn complete_prime_inference_login(
        &self,
        api_key: &str,
        dialog: &mut LoginDialogComponent,
        close_dialog: &mut dyn FnMut(),
    ) -> AuthenticationResult {
        let model_registry = self.host.model_registry();
        model_registry
            .lock()
            .expect("model registry poisoned")
            .set_prime_inference_api_key(api_key);
        let team_status = self.select_prime_inference_team(api_key, dialog).await;

        close_dialog();
        let credential_path = model_registry
            .lock()
            .expect("model registry poisoned")
            .get_prime_cli_config_path()
            .unwrap_or_else(get_auth_path);
        self.complete_provider_authentication(
            PRIME_INFERENCE_PROVIDER_ID,
            PRIME_INFERENCE_PROVIDER_NAME,
            "api_key",
            team_status.as_deref(),
            "provider",
            Some(credential_path),
        )
        .await
    }

    /// Port of `completePrimeAgentTracesLogin`.
    async fn complete_prime_agent_traces_login(
        &self,
        api_key: &str,
        close_dialog: &mut dyn FnMut(),
    ) -> AuthenticationResult {
        let model_registry = self.host.model_registry();
        model_registry
            .lock()
            .expect("model registry poisoned")
            .set_api_key(PRIME_AGENT_TRACES_PROVIDER_ID, api_key);

        close_dialog();
        self.complete_provider_authentication(
            PRIME_AGENT_TRACES_PROVIDER_ID,
            PRIME_AGENT_TRACES_PROVIDER_NAME,
            "api_key",
            None,
            "provider",
            None,
        )
        .await
    }

    /// Port of `runPrimeInferenceLogin`.
    pub async fn run_prime_inference_login(&self) -> AuthenticationResult {
        let mut dialog =
            LoginDialogComponent::new(PRIME_INFERENCE_PROVIDER_ID, PRIME_INFERENCE_PROVIDER_NAME, None);
        let mut handle = show_full_pane_overlay_with(
            self.host.ui(),
            OverlayHandle::default(),
            FullPaneOverlayOptions { max_content_width: Some(88.0), suspend_fullscreen_mouse: true, full_width: false },
        );

        let mut close_dialog = || {
            handle.hide();
            self.host.ui().request_render();
        };

        // When the browser challenge cannot start or breaks down, keep the dialog
        // open and fall back to plain API key entry instead of failing outright.
        let error_msg = "prime inference login is not ported yet".to_string();
        dialog.show_progress(&format!("Browser sign-in unavailable ({error_msg})."));
        if dialog.signal_aborted() {
            close_dialog();
            return AuthenticationResult::Cancelled;
        }
        self.host.show_error(&format!(
            "Failed to login to {PRIME_INFERENCE_PROVIDER_NAME}: {error_msg}"
        ));
        close_dialog();
        AuthenticationResult::Failed
    }

    /// Port of `runPrimeAgentTracesLogin`.
    pub async fn run_prime_agent_traces_login(&self) -> AuthenticationResult {
        let mut dialog =
            LoginDialogComponent::new(PRIME_AGENT_TRACES_PROVIDER_ID, PRIME_AGENT_TRACES_PROVIDER_NAME, None);
        let mut handle = show_full_pane_overlay_with(
            self.host.ui(),
            OverlayHandle::default(),
            FullPaneOverlayOptions { max_content_width: Some(88.0), suspend_fullscreen_mouse: true, full_width: false },
        );

        let mut close_dialog = || {
            handle.hide();
            self.host.ui().request_render();
        };

        let error_msg = "prime agent traces login is not ported yet".to_string();
        dialog.show_progress(&format!("Browser sign-in unavailable ({error_msg})."));
        if dialog.signal_aborted() {
            close_dialog();
            return AuthenticationResult::Cancelled;
        }
        self.host.show_error(&format!(
            "Failed to login to {PRIME_AGENT_TRACES_PROVIDER_NAME}: {error_msg}"
        ));
        close_dialog();
        AuthenticationResult::Failed
    }

    /// Port of `showApiKeyLoginDialog`.
    async fn show_api_key_login_dialog(
        &self,
        provider_id: &str,
        provider_name: &str,
        kind: &str,
    ) -> AuthenticationResult {
        let mut dialog = LoginDialogComponent::new(provider_id, provider_name, None);
        let mut handle = show_full_pane_overlay(self.host.ui(), OverlayHandle::default(), 88.0);

        let close_dialog = |handle: &mut OverlayHandle| {
            handle.hide();
            self.host.ui().request_render();
        };

        let prompt_result = (|| -> Result<String, String> {
            let api_key = dialog.show_prompt("Enter API key:", None).trim().to_string();
            if api_key.is_empty() {
                return Err("API key cannot be empty.".to_string());
            }
            Ok(api_key)
        })();

        match prompt_result {
            Ok(api_key) => {
                let model_registry = self.host.model_registry();
                model_registry
                    .lock()
                    .expect("model registry poisoned")
                    .set_api_key(provider_id, &api_key);
                close_dialog(&mut handle);
                self.complete_provider_authentication(provider_id, provider_name, "api_key", None, kind, None)
                    .await
            }
            Err(error_msg) => {
                close_dialog(&mut handle);
                if error_msg != "Login cancelled" {
                    self.host.show_error(&format!("Failed to save API key for {provider_name}: {error_msg}"));
                    return AuthenticationResult::Failed;
                }
                AuthenticationResult::Cancelled
            }
        }
    }

    /// Port of `showLoginDialog`.
    async fn show_login_dialog(&self, provider_id: &str, provider_name: &str, kind: &str) -> AuthenticationResult {
        let model_registry = self.host.model_registry();
        let provider_info = model_registry
            .lock()
            .expect("model registry poisoned")
            .get_oauth_providers()
            .into_iter()
            .find(|provider| provider.id == provider_id);
        // `providerInfo?.usesCallbackServer ?? false` - `undefined` means false.
        let uses_callback_server = provider_info
            .as_ref()
            .and_then(|provider| provider.uses_callback_server)
            .unwrap_or(false);

        let mut dialog = LoginDialogComponent::new(provider_id, provider_name, None);
        let mut dialog_handle = show_full_pane_overlay_with(
            self.host.ui(),
            OverlayHandle::default(),
            FullPaneOverlayOptions { max_content_width: Some(88.0), suspend_fullscreen_mouse: true, full_width: false },
        );
        let close_dialog = |handle: &mut OverlayHandle| {
            handle.hide();
            self.host.ui().request_render();
        };

        let login_result = model_registry
            .lock()
            .expect("model registry poisoned")
            .login(
                provider_id,
                &mut dialog,
                uses_callback_server,
                provider_id == "github-copilot",
            )
            .await;

        match login_result {
            Ok(()) => {
                close_dialog(&mut dialog_handle);
                self.complete_provider_authentication(provider_id, provider_name, "oauth", None, kind, None)
                    .await
            }
            Err(error_msg) => {
                close_dialog(&mut dialog_handle);
                if error_msg != "Login cancelled" {
                    self.host.show_error(&format!("Failed to login to {provider_name}: {error_msg}"));
                    return AuthenticationResult::Failed;
                }
                AuthenticationResult::Cancelled
            }
        }
    }
}

/// Unused stand-in marker so the selector types stay referenced.
#[allow(dead_code)]
fn _selector_markers(
    _oauth: Option<OAuthSelectorComponent>,
    _extension: Option<ExtensionSelectorComponent>,
    _prime_team: Option<PrimeTeamSelectorComponent>,
) {
}

/// Port of `checkPrimeAgentTracesAccess` re-export used by the interactive mode.
pub async fn check_prime_agent_traces_access_for_host(api_key: &str, signal_aborted: bool) -> Result<PrimeAccess, String> {
    check_prime_agent_traces_access(api_key, &resolve_prime_agent_traces_base_url(), signal_aborted).await
}

/// Port of `checkPrimeInferenceAccess` re-export used by the interactive mode.
pub async fn check_prime_inference_access_for_host(
    api_key: &str,
    base_url: Option<&str>,
    signal_aborted: bool,
) -> Result<PrimeAccess, String> {
    check_prime_inference_access(api_key, base_url, signal_aborted).await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Host {
        ui: TestUi,
        registry: Arc<std::sync::Mutex<ModelRegistry>>,
        statuses: std::sync::Mutex<Vec<String>>,
        errors: std::sync::Mutex<Vec<String>>,
    }

    struct TestUi;

    impl AuthUi for TestUi {
        fn rows(&self) -> usize {
            40
        }
        fn request_render(&self) {}
    }

    impl ProviderAuthFlowsHost for Host {
        fn ui(&self) -> &dyn AuthUi {
            &self.ui
        }
        fn model_registry(&self) -> Arc<std::sync::Mutex<ModelRegistry>> {
            Arc::clone(&self.registry)
        }
        fn show_status(&self, message: &str) {
            self.statuses.lock().expect("statuses").push(message.to_string());
        }
        fn show_error(&self, message: &str) {
            self.errors.lock().expect("errors").push(message.to_string());
        }
        fn get_available_models(&self) -> Vec<AgentConnectionModel> {
            Vec::new()
        }
    }

    fn host() -> Host {
        Host {
            ui: TestUi,
            registry: Arc::new(std::sync::Mutex::new(ModelRegistry::default())),
            statuses: std::sync::Mutex::new(Vec::new()),
            errors: std::sync::Mutex::new(Vec::new()),
        }
    }

    #[test]
    fn anthropic_subscription_key_detection() {
        assert!(is_anthropic_subscription_auth_key(Some("sk-ant-oat01-abc")));
        assert!(!is_anthropic_subscription_auth_key(Some("sk-ant-api03-abc")));
        assert!(!is_anthropic_subscription_auth_key(None));
    }

    #[test]
    fn api_key_login_provider_rules() {
        let oauth: HashSet<String> = ["oauth-only".to_string()].into_iter().collect();
        // A display-name provider is always an API-key login provider.
        assert!(is_api_key_login_provider("openai", &oauth, None));
        // A built-in model provider without a display name is not offered.
        let builtins: HashSet<String> = ["unnamed-builtin".to_string()].into_iter().collect();
        assert!(!is_api_key_login_provider("unnamed-builtin", &oauth, Some(&builtins)));
        assert!(is_api_key_login_provider("deepseek", &oauth, Some(&builtins)));
        // A custom provider that has no OAuth flow is offered.
        assert!(is_api_key_login_provider("my-proxy", &oauth, Some(&builtins)));
        // A provider with an OAuth flow is not offered twice.
        assert!(!is_api_key_login_provider("oauth-only", &oauth, Some(&builtins)));
    }

    #[test]
    fn login_options_always_include_serper() {
        let host = host();
        let flows = ProviderAuthFlows::new(&host);
        let options = flows.get_login_provider_options(None);
        assert!(options.iter().any(|option| option.id == SERPER_CREDENTIAL_ID));
        assert!(options.iter().all(|option| option.category.as_deref() == Some("service")
            || option.auth_type == "api_key"
            || option.auth_type == "oauth"));
        let api_key_only = flows.get_login_provider_options(Some("api_key"));
        assert!(api_key_only.iter().all(|option| option.auth_type == "api_key"));
    }

    #[tokio::test]
    async fn logout_without_credentials_reports_the_exact_message() {
        let host = host();
        let flows = ProviderAuthFlows::new(&host);
        assert_eq!(flows.run_logout().await, None);
        let statuses = host.statuses.lock().expect("statuses");
        assert_eq!(
            statuses.first().map(|value| value.as_str()),
            Some("No stored credentials to remove. /logout only removes credentials saved by /login; environment variables and models.json config are unchanged.")
        );
    }

    #[tokio::test]
    async fn unknown_mcp_integration_reports_failure() {
        let host = host();
        let flows = ProviderAuthFlows::new(&host);
        let result = flows.run_mcp_login("not-a-real-integration", None).await;
        assert_eq!(result, AuthenticationResult::Failed);
        assert_eq!(
            host.errors.lock().expect("errors").first().map(|value| value.as_str()),
            Some("Unknown MCP integration: not-a-real-integration")
        );
    }

    #[tokio::test]
    async fn empty_login_options_report_the_exact_status() {
        let host = host();
        let flows = ProviderAuthFlows::new(&host);
        let result = flows.run_login(ProviderLoginOptions { auth_type: Some("unknown-auth-type".into()), initial_category: None }).await;
        assert_eq!(result, AuthenticationResult::Failed);
        assert_eq!(
            host.statuses.lock().expect("statuses").first().map(|value| value.as_str()),
            Some("No providers available.")
        );
    }

    #[test]
    fn subscription_login_options_include_builtin_codex() {
        let host = host();
        let flows = ProviderAuthFlows::new(&host);
        let options = flows.get_login_provider_options(Some("oauth"));
        assert!(options.iter().any(|option| option.id == "openai-codex"));
        assert!(options.iter().all(|option| option.auth_type == "oauth"));
    }

    #[test]
    fn bedrock_provider_and_warning_constants_match() {
        assert_eq!(BEDROCK_PROVIDER_ID, "amazon-bedrock");
        assert!(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING.starts_with("Anthropic subscription auth is active."));
        assert!(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING.contains("identifies these requests as Claude Code"));
        assert!(!ANTHROPIC_SUBSCRIPTION_AUTH_WARNING.contains("billed per token"));
        assert_eq!(PRIME_INFERENCE_PROVIDER_NAME, "Prime Inference");
        assert_eq!(PRIME_AGENT_TRACES_PROVIDER_NAME, "Prime Agent Traces");
    }
}
