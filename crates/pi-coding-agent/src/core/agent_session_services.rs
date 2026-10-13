//! Port of packages/coding-agent/src/core/agent-session-services.ts

use std::sync::{Arc, Mutex, OnceLock, Weak};

use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::{BoxFuture, Model, ServiceTier, SimpleStreamOptions};
use serde_json::Value;

use crate::config::get_agent_dir;
use crate::core::agent_messages::AgentSessionMessageController;
use crate::core::agent_observe::AgentObserveController;
use crate::core::agent_session::{ResourceLoader, SubagentRuntimeHost};
use crate::core::agent_session_config::AgentExecutionMode;
use crate::core::agent_traces::{
    install_agent_trace_upload_for_session_manager, AgentTraceUploadInstallOptions,
};
use crate::core::auth_storage::AuthStorage;
use crate::core::autonomous::AgentAutonomousConfig;
use crate::core::cron_jobs::AgentRlmHeartbeatController;
use crate::core::extensions::builtin::herdr_agent_state::create_herdr_agent_state_extension;
use crate::core::extensions::builtin::memory::create_memory_extension;
use crate::core::extensions::builtin::telegram::create_telegram_extension;
use crate::core::extensions::types::{
    ExtensionFactory, ExtensionRuntime, ProviderConfig as ExtensionProviderConfig,
};
use crate::core::mcp::mcp_manager::{McpManager as McpManagerImpl, McpManagerOptions};
use crate::core::model_registry::{
    ModelDefinition, ModelRegistry, ProviderConfigInput, ProviderOAuthInput,
};
use crate::core::resource_loader::{DefaultResourceLoader, DefaultResourceLoaderOptions};
use crate::core::semantic_edges::semantic_edge_ledger_path;
use crate::core::session_manager::SessionManager;
use crate::core::settings_manager::{McpServerConfig, SettingsManager};

/// `interface AgentSessionRuntimeDiagnostic`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AgentSessionRuntimeDiagnostic {
    #[serde(rename = "type")]
    pub type_: String,
    pub message: String,
}

pub const DIAGNOSTIC_INFO: &str = "info";
pub const DIAGNOSTIC_ERROR: &str = "error";

/// `CreateAgentSessionServicesOptions`.
pub struct CreateAgentSessionServicesOptions {
    pub cwd: String,
    pub agent_dir: Option<String>,
    pub auth_storage: Option<Arc<tokio::sync::Mutex<AuthStorage>>>,
    pub settings_manager: Option<Arc<Mutex<SettingsManager>>>,
    pub model_registry: Option<Arc<Mutex<ModelRegistry>>>,
    /// `extensionFlagValues?: Map<string, boolean | string>`.
    pub extension_flag_values: Option<indexmap::IndexMap<String, Value>>,
    pub resource_loader_options: Option<DefaultResourceLoaderOptions>,
    /// Skip the built-in Herdr reporter for these services.
    pub no_builtin_herdr_reporter: Option<bool>,
    pub telemetry_disabled: Option<bool>,
}

/// `interface AgentSessionCreationOptions`.
#[derive(Clone, Default)]
pub struct AgentSessionCreationOptions {
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
    pub service_tier: Option<ServiceTier>,
    pub scoped_models: Option<Vec<crate::core::agent_session::ScopedModel>>,
    pub tools: Option<Vec<String>>,
    /// `noTools?: "all" | "builtin"`.
    pub no_tools: Option<String>,
    pub custom_tools: Option<Vec<crate::core::extensions::types::ToolDefinition>>,
    pub initial_active_tool_names: Option<Vec<String>>,
    pub allowed_tool_names: Option<Vec<String>>,
    pub include_goals: Option<bool>,
    pub include_compact_skill: Option<bool>,
    pub agent_message_controller: Option<Arc<dyn AgentSessionMessageController>>,
    pub agent_observe_controller: Option<Arc<dyn AgentObserveController>>,
    pub rlm_depth: Option<i64>,
    pub rlm_max_depth: Option<i64>,
    pub rlm_session_dir: Option<String>,
    pub rlm_parent_node_id: Option<String>,
    pub rlm_parent_agent: Option<String>,
    pub semantic_parent_session_id: Option<String>,
    pub semantic_spawned_by_request_id: Option<String>,
    pub subagent_runtime_host: Option<Arc<dyn SubagentRuntimeHost>>,
    pub rlm_heartbeat_controller: Option<Arc<dyn AgentRlmHeartbeatController>>,
    pub prewarm_ipython_kernel: Option<bool>,
    pub autonomous: Option<AgentAutonomousConfig>,
    pub serialized_refine: Option<bool>,
    pub execution_mode: Option<AgentExecutionMode>,
    pub telemetry_disabled: Option<bool>,
    pub initial_goal: Option<crate::core::agent_session::InitialGoal>,
}

/// `interface CreateAgentSessionFromServicesOptions extends AgentSessionCreationOptions`.
pub struct CreateAgentSessionFromServicesOptions {
    pub services: Arc<AgentSessionServices>,
    pub session_manager: Arc<Mutex<SessionManager>>,
    pub session_start_event: Option<Value>,
    pub creation: AgentSessionCreationOptions,
}

/// `interface AgentSessionServices`.
pub struct AgentSessionServices {
    pub cwd: String,
    pub agent_dir: String,
    pub auth_storage: Arc<tokio::sync::Mutex<AuthStorage>>,
    pub settings_manager: Arc<Mutex<SettingsManager>>,
    pub model_registry: Arc<Mutex<ModelRegistry>>,
    pub resource_loader: Arc<DefaultResourceLoader>,
    pub mcp_manager: Arc<Mutex<McpManagerImpl>>,
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// `applyExtensionFlagValues(resourceLoader, extensionFlagValues)`.
pub fn apply_extension_flag_values(
    resource_loader: &DefaultResourceLoader,
    extension_flag_values: Option<&indexmap::IndexMap<String, Value>>,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    if extension_flag_values.is_none() {
        return Vec::new();
    }
    let extensions_result = resource_loader.get_extensions();
    let mut registered_flags: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
    for extension in &extensions_result.extensions {
        let extension = extension.lock().expect("extension poisoned");
        for (name, flag) in &extension.flags {
            registered_flags.insert(name.clone(), flag.flag_type.clone());
        }
    }
    apply_extension_flag_values_from_flags(
        &registered_flags,
        &extensions_result.runtime,
        extension_flag_values,
    )
}

/// The pure loop of `applyExtensionFlagValues`, split out so the flag table and
/// runtime can be supplied without a loader.
fn apply_extension_flag_values_from_flags(
    registered_flags: &indexmap::IndexMap<String, String>,
    runtime: &ExtensionRuntime,
    extension_flag_values: Option<&indexmap::IndexMap<String, Value>>,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    let Some(extension_flag_values) = extension_flag_values else {
        return Vec::new();
    };

    let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = Vec::new();
    let mut unknown_flags: Vec<String> = Vec::new();
    for (name, value) in extension_flag_values {
        let flag_type = match registered_flags.get(name) {
            Some(flag_type) => flag_type.clone(),
            None => {
                unknown_flags.push(name.clone());
                continue;
            }
        };
        if flag_type == "boolean" {
            runtime.flag_values_set(name, Value::Bool(true));
            continue;
        }
        if value.is_string() {
            runtime.flag_values_set(name, value.clone());
            continue;
        }
        diagnostics.push(AgentSessionRuntimeDiagnostic {
            type_: DIAGNOSTIC_ERROR.to_string(),
            message: format!("Extension flag \"--{name}\" requires a value"),
        });
    }

    if !unknown_flags.is_empty() {
        diagnostics.push(AgentSessionRuntimeDiagnostic {
            type_: DIAGNOSTIC_ERROR.to_string(),
            message: format!(
                "Unknown option{}: {}",
                if unknown_flags.len() == 1 { "" } else { "s" },
                unknown_flags
                    .iter()
                    .map(|name| format!("--{name}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        });
    }

    diagnostics
}

/// `providerConfig` from `extensions/types.ts` converted to the registry's
/// `ProviderConfigInput` (the same object in the TypeScript).
pub(crate) fn provider_config_input(config: &ExtensionProviderConfig) -> ProviderConfigInput {
    ProviderConfigInput {
        name: config.name.clone(),
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        api: config.api.clone(),
        stream_simple: config.stream_simple.as_ref().map(|stream_simple| {
            let stream_simple = Arc::clone(stream_simple);
            Arc::new(
                move |model: &Model,
                      context: &pi_ai::types::Context,
                      options: Option<&SimpleStreamOptions>| {
                    stream_simple(model.clone(), context.clone(), options.cloned())
                },
            ) as pi_ai::api_registry::ApiStreamSimpleFunction
        }),
        headers: config.headers.as_ref().map(|headers| {
            headers
                .iter()
                .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_string())))
                .collect()
        }),
        auth_header: config.auth_header,
        oauth: config.oauth.as_ref().map(|oauth| ProviderOAuthInput {
            name: oauth.name.clone(),
            login: Arc::clone(&oauth.login),
            uses_callback_server: None,
            refresh_token: Arc::clone(&oauth.refresh_token),
            get_api_key: {
                let get_api_key = Arc::clone(&oauth.get_api_key);
                Arc::new(move |credentials: &pi_ai::utils::oauth::types::OAuthCredentials| {
                    get_api_key(credentials.clone())
                })
            },
            modify_models: None,
        }),
        models: config
            .models
            .as_ref()
            .map(|models| models.iter().map(model_definition).collect()),
    }
}

/// `ProviderModelConfig` -> `ModelDefinition`.
fn model_definition(model: &crate::core::extensions::types::ProviderModelConfig) -> ModelDefinition {
    ModelDefinition {
        id: model.id.clone(),
        name: Some(model.name.clone()),
        api: model.api.clone(),
        base_url: model.base_url.clone(),
        reasoning: Some(model.reasoning),
        thinking_level_map: model.thinking_level_map.clone(),
        input: Some(model.input.clone()),
        cost: Some(pi_ai::types::ModelCost {
            input: model.cost.input,
            output: model.cost.output,
            cache_read: model.cost.cache_read,
            cache_write: model.cost.cache_write,
        }),
        context_window: Some(model.context_window),
        max_input_tokens: model.max_input_tokens,
        max_tokens: Some(model.max_tokens),
        default_max_tokens: model.default_max_tokens,
        native_compaction: serde_json::from_value(model.native_compaction.clone().unwrap_or(Value::Null)).ok(),
        headers: None,
        compat: serde_json::from_value(model.compat.clone().unwrap_or(Value::Null)).ok(),
    }
}

/// `createAgentSessionServices(options)`.
pub async fn create_agent_session_services(
    mut options: CreateAgentSessionServicesOptions,
) -> Result<AgentSessionServices, String> {
    let cwd = options.cwd.clone();
    let agent_dir = options.agent_dir.clone().unwrap_or_else(get_agent_dir);
    let auth_storage = options.auth_storage.clone().unwrap_or_else(|| {
        Arc::new(tokio::sync::Mutex::new(AuthStorage::create(
            Some(join_path(&agent_dir, "auth.json")),
            None,
        )))
    });
    let settings_manager = options
        .settings_manager
        .clone()
        .unwrap_or_else(|| Arc::new(Mutex::new(SettingsManager::create(&cwd, Some(&agent_dir)))));
    // A1 (audit round 3, GLM-AGENT-CORE.md) - resolved in main_entry.rs.
    //
    // TS builds `authStorage` once and hands that SAME instance to the registry, so
    // the CLI runtime override is visible to request auth:
    //   agent-session-services.ts:150
    //     `const authStorage = options.authStorage ?? AuthStorage.create(join(agentDir, "auth.json"));`
    //   agent-session-services.ts:152
    //     `const modelRegistry = options.modelRegistry ?? ModelRegistry.create(authStorage, join(agentDir, "models.json"));`
    //   model-registry.ts:525 `readonly authStorage: AuthStorage,`
    //   main.ts:882 `authStorage.setRuntimeApiKey(effectiveSessionModel.provider, config.apiKey);`
    //
    // The call below gives the registry its own `AuthStorage` because the Rust
    // `ModelRegistry` owns `auth_storage: AuthStorage` by value (model_registry.rs:1146)
    // and has no constructor taking a shared handle. Instead of sharing the instance,
    // main_entry.rs applies the CLI override to BOTH stores through
    // `ModelRegistry::set_runtime_api_key` (model_registry.rs), which delegates to
    // `AuthStorage::set_runtime_api_key` (auth_storage.rs:709) on the instance that
    // `get_api_key_and_headers` resolves request auth from. Keep the two writes in
    // main_entry.rs together; this constructor has no access to the override.

    let model_registry = options.model_registry.clone().unwrap_or_else(|| {
        Arc::new(Mutex::new(ModelRegistry::create(
            AuthStorage::create(Some(join_path(&agent_dir, "auth.json")), None),
            Some(join_path(&agent_dir, "models.json")),
        )))
    });

    // MCP integrations: registers OAuth providers and gates the built-in
    // integration skills by whether the user is logged in (enable-by-login).
    let mcp_manager = Arc::new(Mutex::new(McpManagerImpl::new(McpManagerOptions {
        auth_storage: Arc::clone(&auth_storage),
        get_user_servers: {
            let settings_manager = Arc::clone(&settings_manager);
            Some(Arc::new(move || {
                let settings = settings_manager.lock().expect("settings manager poisoned");
                settings.get_global_mcp_servers().map(|servers| {
                    let mut parsed: indexmap::IndexMap<String, McpServerConfig> =
                        indexmap::IndexMap::new();
                    for (name, value) in servers {
                        if let Ok(config) = serde_json::from_value::<McpServerConfig>(value) {
                            parsed.insert(name, config);
                        }
                    }
                    parsed
                })
            })
                as Arc<
                    dyn Fn() -> Option<indexmap::IndexMap<String, McpServerConfig>> + Send + Sync,
                >)
        },
        begin_login: None,
    })));
    // refresh() resets the OAuth registry to built-ins; re-add user MCP providers too.
    {
        let mcp_manager = Arc::clone(&mcp_manager);
        let mut registry = model_registry.lock().expect("model registry poisoned");
        registry.set_on_oauth_providers_reset(Arc::new(move || {
            mcp_manager
                .lock()
                .expect("mcp manager poisoned")
                .register_user_providers();
        }));
    }

    let resource_loader_options = options.resource_loader_options.take().unwrap_or_default();
    let user_extension_factories = resource_loader_options.extension_factories.clone();
    // The built-in Herdr reporter defers to Herdr's own file-based integration
    // when the loader actually loaded it; two reporters would race on the same
    // pane. Deferral is late-bound to the loader's loaded paths (inline
    // factories run after file extensions load), so a file that exists but is
    // disabled or never discovered does not silence the built-in.
    // noExtensions is a full opt-out: it disables the built-in reporter too,
    // not just discovered extension files.
    let skip_herdr_reporter =
        options.no_builtin_herdr_reporter.unwrap_or(false) || resource_loader_options.no_extensions;
    let loader_slot: Arc<OnceLock<Weak<DefaultResourceLoader>>> = Arc::new(OnceLock::new());
    let mut builtin_extension_factories: Vec<ExtensionFactory> = Vec::new();
    if !skip_herdr_reporter {
        let loader_slot = Arc::clone(&loader_slot);
        builtin_extension_factories.push(create_herdr_agent_state_extension(Arc::new(move || {
            loader_slot
                .get()
                .and_then(Weak::upgrade)
                .map(|loader| loader.get_loaded_extension_paths())
                .unwrap_or_default()
        })));
    }
    if !resource_loader_options.no_extensions {
        builtin_extension_factories.push(create_telegram_extension(agent_dir.clone()));
        builtin_extension_factories.push(create_memory_extension(
            agent_dir.clone(),
            Arc::clone(&settings_manager),
        ));
    }

    let mut extension_factories = builtin_extension_factories;
    extension_factories.extend(user_extension_factories);
    let resource_loader = Arc::new(DefaultResourceLoader::new(DefaultResourceLoaderOptions {
        cwd: cwd.clone(),
        agent_dir: agent_dir.clone(),
        settings_manager: Some(Arc::clone(&settings_manager)),
        extension_factories,
        extra_builtin_skill_overrides: Some({
            let mcp_manager = Arc::clone(&mcp_manager);
            Arc::new(move || {
                mcp_manager
                    .lock()
                    .expect("mcp manager poisoned")
                    .get_disabled_builtin_skill_overrides()
            })
        }),
        ..resource_loader_options
    }));
    let _ = loader_slot.set(Arc::downgrade(&resource_loader));
    resource_loader.reload().await;

    let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = Vec::new();
    if !options.telemetry_disabled.unwrap_or(false)
        && is_telemetry_enabled(&settings_manager)
        && !settings_manager
            .lock()
            .expect("settings manager poisoned")
            .get_telemetry_notice_shown()
    {
        diagnostics.push(AgentSessionRuntimeDiagnostic {
            type_: DIAGNOSTIC_INFO.to_string(),
            message: "Prime Agent sends pseudonymous usage and performance metrics without prompts, responses, tool content, file paths, or repository data. Disable this with telemetry.enabled=false, PRIME_AGENT_TELEMETRY=0, DO_NOT_TRACK=1, or offline mode.".to_string(),
        });
        settings_manager
            .lock()
            .expect("settings manager poisoned")
            .set_telemetry_notice_shown(true);
    }

    let extensions_result = resource_loader.get_extensions();
    for registration in extensions_result.runtime.take_pending_provider_registrations() {
        let config = provider_config_input(&registration.config);
        match model_registry
            .lock()
            .expect("model registry poisoned")
            .register_provider(&registration.name, config)
        {
            Ok(()) => {}
            Err(error) => diagnostics.push(AgentSessionRuntimeDiagnostic {
                type_: DIAGNOSTIC_ERROR.to_string(),
                message: format!(
                    "Extension \"{}\" error: {}",
                    registration.extension_path, error
                ),
            }),
        }
    }
    diagnostics.extend(apply_extension_flag_values(
        &resource_loader,
        options.extension_flag_values.as_ref(),
    ));

    Ok(AgentSessionServices {
        cwd,
        agent_dir,
        auth_storage,
        settings_manager,
        model_registry,
        resource_loader,
        mcp_manager,
        diagnostics,
    })
}

/// `isTelemetryEnabled(settingsManager)` from `core/telemetry.ts`.
///
/// That module belongs to another slice and has not landed; the settings-driven
/// half of the check is reproduced here and the environment half is read from
/// the same variables the TypeScript reads.
pub fn is_telemetry_enabled(settings_manager: &Arc<Mutex<SettingsManager>>) -> bool {
    if env_disables_telemetry() {
        return false;
    }
    settings_manager
        .lock()
        .expect("settings manager poisoned")
        .get_telemetry_enabled()
}

/// `PRIME_AGENT_TELEMETRY=0` / `DO_NOT_TRACK=1`.
fn env_disables_telemetry() -> bool {
    if let Ok(value) = std::env::var("PRIME_AGENT_TELEMETRY") {
        if value == "0" || value.eq_ignore_ascii_case("false") {
            return true;
        }
    }
    std::env::var("DO_NOT_TRACK")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// `createAgentSessionFromServices(options)`.
pub async fn create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> Result<crate::core::sdk::CreateAgentSessionResult, String> {
    let artifact_dir = options
        .session_manager
        .lock()
        .expect("session manager poisoned")
        .get_session_artifact_dir();
    // `installAgentTraceUpload(options.sessionManager, {...})` -
    // agent-session-services.ts:234 passes the session manager as argument 1, so
    // `sessionManager.onPersist(controller.schedule)` is registered
    // (agent-traces.ts:1169) and every transcript persist schedules an upload.
    // The options-only `install_agent_trace_upload` registers no listener, which
    // left automatic trace upload dead, so this call passes the manager held by
    // `options.session_manager` - the same value handed to `createAgentSession`
    // below (agent-session-services.ts:250).
    let session_manager = Arc::clone(&options.session_manager);
    // The persist listener runs while the persist caller still holds the session
    // manager's guard: `notify_persist_listeners` is called from `rewrite_file`
    // (session_manager.rs:3436) and `persist` (session_manager.rs:3606) inside
    // `flush_now`/`append_*`, which hold the guard. `Mutex` is not reentrant, so a
    // `get_session_file` closure that always locked the manager would self-deadlock
    // the very persist that triggers the schedule - a hang the TypeScript cannot
    // have, because there `sessionManager.getSessionFile()` (agent-traces.ts:1088)
    // is a plain object read. Cache the path instead: re-read it live under
    // `try_lock` whenever the guard is free, and fall back to the last known path
    // while a persist holds it (the manager cannot switch files mid-persist).
    let cached_session_file = Arc::new(Mutex::new(
        session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file(),
    ));
    // `notify_persist_listeners` hands the listener the manager's current session
    // file (session_manager.rs:3439-3450), so this listener keeps the cache exact at
    // the moment of every persist without touching the manager's guard. It is
    // registered BEFORE the install below: listeners run in registration order
    // (session_manager.rs:3447), so the cache is already current when
    // `schedule()` (agent-traces.ts:1081) reads the path. The returned unsubscribe is
    // deliberately dropped, like the TypeScript discarding `onPersist`'s return
    // (agent-traces.ts:1169), so the listener lives as long as the manager.
    let _session_file_cache_listener = {
        let cached = Arc::clone(&cached_session_file);
        session_manager
            .lock()
            .expect("session manager poisoned")
            .on_persist(Box::new(move |session_file: &str| {
                *cached.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                    Some(session_file.to_string());
            }))
    };
    let trace_upload_options = AgentTraceUploadInstallOptions {
        auth_storage: Arc::clone(&options.services.auth_storage),
        base_url: None,
        config_path: None,
        fetch_fn: None,
        request_timeout_ms: None,
        semantic_edges_ledger_path: semantic_edge_ledger_path(
            options.creation.rlm_session_dir.as_deref(),
            artifact_dir.as_deref(),
        ),
        agent_traces_enabled: {
            let settings = Arc::clone(&options.services.settings_manager);
            Arc::new(move || settings.lock().expect("settings manager poisoned").get_agent_traces_enabled())
        },
        reload_settings: {
            let settings = Arc::clone(&options.services.settings_manager);
            Arc::new(move || {
                let settings = Arc::clone(&settings);
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || {
                        settings.lock().expect("settings manager poisoned").reload_sync();
                    }).await.map_err(|error| error.to_string())
                })
            })
        },
        get_session_file: {
            let manager = Arc::clone(&options.session_manager);
            let cached = Arc::clone(&cached_session_file);
            Arc::new(move || {
                if let Ok(manager) = manager.try_lock() {
                    let session_file = manager.get_session_file();
                    *cached
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = session_file.clone();
                    return session_file;
                }
                cached
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
            })
        },
    };
    // The lock guard must be released here: `create_agent_session` below locks the
    // same manager, so the guard cannot be held across the call.
    let _upload = {
        let manager = session_manager.lock().expect("session manager poisoned");
        install_agent_trace_upload_for_session_manager(&manager, trace_upload_options)
    };
    let telemetry_disabled = options.creation.telemetry_disabled;
    let creation = options.creation;
    let result = crate::core::sdk::create_agent_session(crate::core::sdk::CreateAgentSessionOptions {
        cwd: Some(options.services.cwd.clone()),
        agent_dir: Some(options.services.agent_dir.clone()),
        auth_storage: Some(Arc::clone(&options.services.auth_storage)),
        model_registry: Some(Arc::clone(&options.services.model_registry)),
        model: creation.model.clone(),
        thinking_level: creation.thinking_level.clone(),
        service_tier: creation.service_tier.clone(),
        scoped_models: creation.scoped_models.clone(),
        no_tools: creation.no_tools.clone(),
        tools: creation.tools.clone(),
        custom_tools: creation.custom_tools.clone(),
        resource_loader: Some(
            Arc::clone(&options.services.resource_loader) as Arc<dyn ResourceLoader>
        ),
        mcp_manager: Some(Arc::clone(&options.services.mcp_manager)),
        session_manager: Some(options.session_manager),
        settings_manager: Some(Arc::clone(&options.services.settings_manager)),
        session_start_event: options.session_start_event,
        autonomous: creation.autonomous.clone(),
        creation: AgentSessionCreationOptions {
            telemetry_disabled,
            ..creation
        },
    })
    .await?;
    // `if (result.session.rlmDepth === 0 && !options.telemetryDisabled)`
    // `installAgentTelemetry(result.session, ...)` lives in `core/telemetry.ts`,
    // which has not landed (recorded in `blocked_on`).
    if result.session.rlm_depth() == 0 && telemetry_disabled != Some(true) {
        let _ = telemetry_disabled;
    }
    Ok(result)
}

/// `join(agentDir, name)`.
fn join_path(base: &str, name: &str) -> String {
    std::path::Path::new(base)
        .join(name)
        .to_string_lossy()
        .to_string()
}

/// Keeps the `BoxFuture`/`ServiceTier` imports used by the creation surface.
#[allow(dead_code)]
fn surface_markers(_future: BoxFuture<()>, _tier: &ServiceTier) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::types::{ExtensionRuntime, ExtensionRuntimeState};

    fn runtime() -> ExtensionRuntime {
        ExtensionRuntime::new(ExtensionRuntimeState::default())
    }

    /// A1 repro (`--print --model X --api-key K`).
    ///
    /// `main.ts:882` writes the CLI override to the auth storage that
    /// `ModelRegistry.create(authStorage, ...)` received
    /// (`agent-session-services.ts:150-152`, `model-registry.ts:525`), so request
    /// auth (`getApiKeyAndHeaders`) resolves it. This port's `ModelRegistry` owns a
    /// separate `AuthStorage` by value, so the override must also be written through
    /// `ModelRegistry::set_runtime_api_key`; the registry is deliberately left to
    /// `create_agent_session_services` here to reproduce the two-instance setup.
    #[tokio::test]
    async fn runtime_api_key_reaches_model_registry_request_auth() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();
        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            indexmap::IndexMap::new(),
            Some(crate::core::auth_storage::AuthStorageOptions {
                prime_cli_config_path: None,
                use_prime_cli_config: false,
            }),
        )));
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(cwd.clone()),
            auth_storage: Some(Arc::clone(&auth_storage)),
            settings_manager: None,
            model_registry: None,
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                ..DefaultResourceLoaderOptions::new(&cwd, &cwd)
            }),
        })
        .await
        .expect("services");
        let provider = "anthropic";
        let model = services
            .model_registry
            .lock()
            .expect("model registry poisoned")
            .get_all()
            .into_iter()
            .find(|model| model.provider == provider)
            .expect("anthropic model in the built-in catalog");

        // The two writes main_entry.rs performs for one `--api-key` flag.
        services
            .auth_storage
            .lock()
            .await
            .set_runtime_api_key(provider, "cli-key");
        services
            .model_registry
            .lock()
            .expect("model registry poisoned")
            .set_runtime_api_key(provider, "cli-key");

        assert!(services
            .model_registry
            .lock()
            .expect("model registry poisoned")
            .has_configured_auth(&model));
        let auth = services
            .model_registry
            .lock()
            .expect("model registry poisoned")
            .get_api_key_and_headers(&model)
            .await;
        assert!(auth.ok);
        assert_eq!(auth.api_key.as_deref(), Some("cli-key"));
    }

    /// `createAgentSessionFromServices` must call
    /// `installAgentTraceUpload(options.sessionManager, {...})`
    /// (agent-session-services.ts:234), which registers
    /// `sessionManager.onPersist(controller.schedule)` (agent-traces.ts:1169). The
    /// audit found this call site used the options-only install, which registers no
    /// listener, so a persisted session never scheduled an upload.
    ///
    /// `schedule()` is observable without network access: it marks upload intent in
    /// the outbox (`markAgentTraceOutboxPendingSync`, agent-traces.ts:1092). The
    /// install runs before `createAgentSession` (agent-session-services.ts:234 vs
    /// :242), so the assertion holds even if the later creation steps fail.
    #[tokio::test]
    async fn a_persist_after_create_agent_session_from_services_schedules_a_trace_upload() {
        use pi_ai::providers::faux::{register_faux_provider, RegisterFauxProviderOptions};
        use std::path::Path;

        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();
        let session_dir = dir.path().join("sessions");
        std::fs::create_dir_all(&session_dir).unwrap();

        let mut manager =
            SessionManager::create(&cwd, Some(&session_dir.to_string_lossy())).unwrap();
        let session_file = manager.new_session(None).unwrap().expect("session file");
        let session_manager = Arc::new(Mutex::new(manager));

        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some(format!("session-services-{}", uuid::Uuid::new_v4())),
            tokens_per_second: Some(0.0),
            ..Default::default()
        }));
        let model = provider.get_model();
        let mut registry_auth = AuthStorage::in_memory(Default::default(), None);
        registry_auth.set_runtime_api_key(&model.provider, "synthetic-faux-key");
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "agentTraces": {"enabled": true},
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));
        assert!(
            settings.lock().expect("settings manager poisoned").get_agent_traces_enabled(),
            "the schedule() path only records intent while sharing is enabled"
        );

        // `installAgentTraceUpload` runs the one-shot startup catch-up
        // (`catchUpTriggered`, agent-traces.ts:1154-1160) in whichever process
        // installs first. Consume that one-shot here with tracing disabled against a
        // throwaway manager, so the sweep cannot read, upload or re-cursor any entry
        // in the real outbox directory; `agent_traces_enabled: false` makes
        // `catchUpAgentTraceUploads` return before it touches a file
        // (agent-traces.rs:1116-1118).
        {
            let sink = SessionManager::in_memory(Some(&cwd), None).unwrap();
            let _sink_controller = install_agent_trace_upload_for_session_manager(
                &sink,
                AgentTraceUploadInstallOptions {
                    auth_storage: Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
                        Default::default(),
                        None,
                    ))),
                    base_url: None,
                    config_path: None,
                    fetch_fn: None,
                    request_timeout_ms: None,
                    semantic_edges_ledger_path: None,
                    agent_traces_enabled: Arc::new(|| false),
                    reload_settings: Arc::new(|| Box::pin(async { Ok(()) })),
                    get_session_file: Arc::new(|| None),
                },
            );
        }

        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(cwd.clone()),
            auth_storage: Some(Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
                Default::default(),
                None,
            )))),
            settings_manager: Some(Arc::clone(&settings)),
            model_registry: Some(Arc::new(Mutex::new(ModelRegistry::in_memory(registry_auth)))),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                ..DefaultResourceLoaderOptions::new(&cwd, &cwd)
            }),
        })
        .await
        .expect("services");

        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::clone(&session_manager),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                no_tools: Some("all".to_string()),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                ..Default::default()
            },
        })
        .await;

        // Follow the same configured profile as agent-traces.ts; the test runner
        // deliberately hides the real home directory behind an isolated profile.
        let outbox_dir = Path::new(&get_agent_dir()).join("agent-traces-outbox");
        let ledger_path = crate::core::semantic_edges::semantic_edge_ledger_path(
            None,
            session_manager
                .lock()
                .expect("session manager poisoned")
                .get_session_artifact_dir()
                .as_deref(),
        );
        let entries_for = |key: &str| -> Vec<std::path::PathBuf> {
            let Ok(read_dir) = std::fs::read_dir(&outbox_dir) else {
                return Vec::new();
            };
            read_dir
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    std::fs::read_to_string(path)
                        .ok()
                        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
                        .and_then(|value| {
                            value.get("sessionFile").and_then(Value::as_str).map(str::to_string)
                        })
                        .as_deref()
                        == Some(key)
                })
                .collect()
        };
        for entry in entries_for(&session_file) {
            let _ = std::fs::remove_file(entry);
        }
        assert!(
            entries_for(&session_file).is_empty(),
            "no upload intent before the first persist"
        );

        // A real transcript persist: `flushNow()` -> `_rewriteFile()` ->
        // `_notifyPersistListeners()` (session-manager.ts:2003-2008, 1912).
        let _ = session_manager.lock().expect("session manager poisoned").flush_now();
        assert!(
            Path::new(&session_file).exists(),
            "the persist must write the session file"
        );

        let recorded = entries_for(&session_file);
        assert!(
            !recorded.is_empty(),
            "a persist after createAgentSessionFromServices must schedule an upload: \
             sessionManager.onPersist(controller.schedule) is registered by \
             installAgentTraceUpload(options.sessionManager, ...) \
             (agent-session-services.ts:234) and the listener marks upload intent for \
             {session_file} (agent-traces.ts:1169)"
        );

        // Leave no outbox entries behind for this temp session.
        for entry in recorded {
            let _ = std::fs::remove_file(entry);
        }
        if let Some(ledger_path) = ledger_path.as_deref() {
            for entry in entries_for(ledger_path) {
                let _ = std::fs::remove_file(entry);
            }
        }
        if let Ok(created) = created {
            created.session.dispose_async(Some(false)).await;
        }
        provider.unregister();
    }

    #[test]
    fn no_flag_values_produces_no_diagnostics() {
        let flags = indexmap::IndexMap::new();
        assert!(apply_extension_flag_values_from_flags(&flags, &runtime(), None).is_empty());
    }

    #[test]
    fn an_unknown_flag_is_reported_with_the_singular_label() {
        let flags = indexmap::IndexMap::new();
        let mut values = indexmap::IndexMap::new();
        values.insert("missing".to_string(), Value::Bool(true));
        let diagnostics = apply_extension_flag_values_from_flags(&flags, &runtime(), Some(&values));
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].type_, DIAGNOSTIC_ERROR);
        assert_eq!(diagnostics[0].message, "Unknown option: --missing");
    }

    #[test]
    fn several_unknown_flags_use_the_plural_label_and_declaration_order() {
        let flags = indexmap::IndexMap::new();
        let mut values = indexmap::IndexMap::new();
        values.insert("one".to_string(), Value::Bool(true));
        values.insert("two".to_string(), Value::Bool(true));
        let diagnostics = apply_extension_flag_values_from_flags(&flags, &runtime(), Some(&values));
        assert_eq!(diagnostics[0].message, "Unknown options: --one, --two");
    }

    #[test]
    fn a_boolean_flag_is_set_to_true_in_the_runtime() {
        let mut flags = indexmap::IndexMap::new();
        flags.insert("verbose".to_string(), "boolean".to_string());
        let shared = runtime();
        let mut values = indexmap::IndexMap::new();
        values.insert("verbose".to_string(), Value::Bool(false));
        assert!(apply_extension_flag_values_from_flags(&flags, &shared, Some(&values)).is_empty());
        assert_eq!(shared.flag_values_get("verbose"), Some(Value::Bool(true)));
    }

    #[test]
    fn a_string_flag_keeps_its_supplied_value() {
        let mut flags = indexmap::IndexMap::new();
        flags.insert("mode".to_string(), "string".to_string());
        let shared = runtime();
        let mut values = indexmap::IndexMap::new();
        values.insert("mode".to_string(), Value::String("fast".to_string()));
        assert!(apply_extension_flag_values_from_flags(&flags, &shared, Some(&values)).is_empty());
        assert_eq!(
            shared.flag_values_get("mode"),
            Some(Value::String("fast".to_string()))
        );
    }

    #[test]
    fn a_valueless_string_flag_reports_the_requires_a_value_error() {
        let mut flags = indexmap::IndexMap::new();
        flags.insert("mode".to_string(), "string".to_string());
        let shared = runtime();
        let mut values = indexmap::IndexMap::new();
        values.insert("mode".to_string(), Value::Bool(true));
        let diagnostics = apply_extension_flag_values_from_flags(&flags, &shared, Some(&values));
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].message, "Extension flag \"--mode\" requires a value");
        assert_eq!(shared.flag_values_get("mode"), None);
    }
}
