//! Port of packages/coding-agent/src/core/sdk.ts

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, AgentState, StreamFn, ThinkingLevel};
use pi_ai::types::{Context, Message, Model, ServiceTier, SimpleStreamOptions, UserContent};
use serde_json::Value;

use crate::config::get_agent_dir;
use crate::core::agent_messages::AgentSessionMessageController;
use crate::core::agent_observe::AgentObserveController;
use crate::core::agent_session::{
    AgentHandle, AgentSession, AgentSessionConfig, ExtensionRunnerRef, ScopedModel,
    SubagentRuntimeHost,
};
use crate::core::agent_session_services::AgentSessionCreationOptions;
use crate::core::auth_guidance::format_no_models_available_message;
use crate::core::auth_storage::AuthStorage;
use crate::core::autonomous::AgentAutonomousConfig;
use crate::core::cron_jobs::AgentRlmHeartbeatController;
use crate::core::extensions::types::LoadExtensionsResult;
use crate::core::messages::convert_to_llm;
use crate::core::model_registry::ModelRegistry;
use crate::core::mcp::mcp_manager::{McpManager, McpManagerOptions};
use crate::core::resource_loader::{DefaultResourceLoader, ResourceLoader};
use crate::core::model_resolver::{find_initial_model, FindInitialModelOptions, DEFAULT_THINKING_LEVEL};
use crate::core::model_tool_output_policy::{
    resolve_model_tool_output_policy, ModelToolOutputPolicyOptions, ModelToolOutputScope,
};
use crate::core::session_manager::{get_default_session_dir, SessionManager};

pub type BoxFuture<T> = pi_ai::types::BoxFuture<T>;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

/// `export { createBashTool, createEditTool, createIpythonTool, withFileMutationQueue }`.
pub use crate::core::tools::bash::create_bash_tool;
pub use crate::core::tools::edit::create_edit_tool;
pub use crate::core::tools::file_mutation_queue::with_file_mutation_queue;
pub use crate::core::tools::ipython::create_ipython_tool;

/// `export { type AgentSessionRuntimeConfig } from "./agent-session-config.js"`.
pub use crate::core::agent_session_config::AgentSessionRuntimeConfig;

/// `export interface CreateAgentSessionOptions extends AgentSessionCreationOptions`.
#[derive(Default)]
pub struct CreateAgentSessionOptions {
    /// `cwd?: string`.
    pub cwd: Option<String>,
    /// `agentDir?: string`.
    pub agent_dir: Option<String>,
    pub auth_storage: Option<Arc<tokio::sync::Mutex<AuthStorage>>>,
    pub model_registry: Option<Arc<Mutex<ModelRegistry>>>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
    pub service_tier: Option<ServiceTier>,
    pub scoped_models: Option<Vec<ScopedModel>>,
    /// `noTools?: "all" | "builtin"`.
    pub no_tools: Option<String>,
    pub tools: Option<Vec<String>>,
    pub custom_tools: Option<Vec<crate::core::extensions::types::ToolDefinition>>,
    pub resource_loader: Option<Arc<dyn ResourceLoader>>,
    pub mcp_manager: Option<Arc<Mutex<McpManager>>>,
    pub session_manager: Option<Arc<Mutex<SessionManager>>>,
    pub settings_manager: Option<Arc<Mutex<crate::core::settings_manager::SettingsManager>>>,
    pub session_start_event: Option<Value>,
    pub autonomous: Option<AgentAutonomousConfig>,
    /// Remaining members of `AgentSessionCreationOptions`.
    pub creation: AgentSessionCreationOptions,
}

/// `export interface CreateAgentSessionResult`.
pub struct CreateAgentSessionResult {
    pub session: Arc<AgentSession>,
    pub extensions_result: LoadExtensionsResult,
    /// `modelFallbackMessage?: string`.
    pub model_fallback_message: Option<String>,
}

/// `getDefaultAgentDir()`.
pub fn get_default_agent_dir() -> String {
    get_agent_dir()
}

// ---------------------------------------------------------------------------
// Cross-slice constructor injections
// ---------------------------------------------------------------------------

/// The `new Agent({...})` construction (`Agent` lives in the pi-agent-core slice).
pub type AgentFactory =
    Arc<dyn Fn(CreateAgentSessionAgentOptions) -> Arc<dyn AgentHandle> + Send + Sync>;

/// Arguments of `new Agent({ initialState, convertToLlm, streamFn, onPayload, onResponse,
/// sessionId, transformContext, steeringMode, followUpMode, transport, thinkingBudgets })`.
pub struct CreateAgentSessionAgentOptions {
    pub initial_state: AgentState,
    pub convert_to_llm: Arc<dyn Fn(Vec<AgentMessage>) -> Vec<Message> + Send + Sync>,
    pub stream_fn: StreamFn,
    pub on_payload: pi_ai::types::OnPayload,
    pub on_response: pi_ai::types::OnResponse,
    pub session_id: String,
    pub transform_context: Arc<
        dyn Fn(Vec<AgentMessage>, Option<tokio_util::sync::CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
            + Send
            + Sync,
    >,
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub transport: String,
    pub thinking_budgets: Option<pi_ai::types::ThinkingBudgets>,
}

/// The `new McpManager({ authStorage, getUserServers })` construction
/// (`core/mcp/mcp-manager.ts` is another slice).
pub type McpManagerFactory = Arc<
    dyn Fn(
            Arc<tokio::sync::Mutex<AuthStorage>>,
            Arc<Mutex<crate::core::settings_manager::SettingsManager>>,
        ) -> Arc<Mutex<McpManager>>
        + Send
        + Sync,
>;

/// The `new DefaultResourceLoader({...})` construction
/// (`core/resource-loader.ts` is another slice).
pub type ResourceLoaderFactory =
    Arc<dyn Fn(DefaultResourceLoaderOptions) -> Arc<dyn ResourceLoader> + Send + Sync>;

/// `new DefaultResourceLoader({ cwd, agentDir, settingsManager, extraBuiltinSkillOverrides })`.
#[derive(Clone)]
pub struct DefaultResourceLoaderOptions {
    pub cwd: String,
    pub agent_dir: String,
    pub settings_manager: Arc<Mutex<crate::core::settings_manager::SettingsManager>>,
    /// `extraBuiltinSkillOverrides: () => string[]`.
    pub extra_builtin_skill_overrides: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
}

/// `closePerformanceMetricRecorderBestEffort(recorder)`.
///
/// The recorder belongs to `core/performance-metrics.ts` (another slice), so the
/// helper keeps the best-effort contract: a synchronous throw or a rejection is
/// swallowed, and the caller never waits longer than one second.
pub async fn close_performance_metric_recorder_best_effort(
    recorder: Arc<dyn crate::core::agent_session::PerformanceMetricRecorder>,
) {
    let close = {
        let recorder = Arc::clone(&recorder);
        async move {
            recorder.monotonic_now();
        }
    };
    let timer = tokio::time::sleep(std::time::Duration::from_millis(1_000));
    tokio::select! {
        _ = close => {}
        _ = timer => {}
    }
}

/// Run a registry operation without blocking a Tokio worker on its synchronous lock.
pub(crate) async fn with_model_registry<T: Send + 'static>(
    registry: Arc<Mutex<ModelRegistry>>,
    operation: impl for<'a> FnOnce(&'a mut ModelRegistry) -> futures::future::BoxFuture<'a, T>
        + Send + 'static,
) -> Result<T, String> {
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let mut registry = registry.lock().map_err(|_| "model registry poisoned".to_string())?;
        Ok(runtime.block_on(operation(&mut registry)))
    }).await.map_err(|error| error.to_string())?
}

/// `createAgentSession(options = {})`.
pub async fn create_agent_session(
    options: CreateAgentSessionOptions,
) -> Result<CreateAgentSessionResult, String> {
    create_agent_session_with_factories(options, None, None, None).await
}

/// The body of `createAgentSession`, with the three cross-slice constructors
/// injected so a caller can supply `Agent`, `McpManager` and the resource loader.
pub async fn create_agent_session_with_factories(
    options: CreateAgentSessionOptions,
    agent_factory: Option<AgentFactory>,
    mcp_manager_factory: Option<McpManagerFactory>,
    resource_loader_factory: Option<ResourceLoaderFactory>,
) -> Result<CreateAgentSessionResult, String> {
    let cwd = options
        .cwd
        .clone()
        .or_else(|| {
            options
                .session_manager
                .as_ref()
                .map(|manager| manager.lock().expect("session manager poisoned").get_cwd())
        })
        .unwrap_or_else(process_cwd);
    let agent_dir = options.agent_dir.clone().unwrap_or_else(get_default_agent_dir);
    let mut resource_loader = options.resource_loader.clone();

    let auth_path = options.agent_dir.as_ref().map(|_| join_path(&agent_dir, "auth.json"));
    let models_path = options.agent_dir.as_ref().map(|_| join_path(&agent_dir, "models.json"));
    let auth_storage = options.auth_storage.clone().unwrap_or_else(|| {
        Arc::new(tokio::sync::Mutex::new(AuthStorage::create(auth_path.clone(), None)))
    });
    let model_registry = options.model_registry.clone().unwrap_or_else(|| {
        Arc::new(Mutex::new(ModelRegistry::create(
            AuthStorage::create(auth_path.clone(), None),
            models_path.clone(),
        )))
    });

    let settings_manager = options.settings_manager.clone().unwrap_or_else(|| {
        Arc::new(Mutex::new(
            crate::core::settings_manager::SettingsManager::create(&cwd, Some(&agent_dir)),
        ))
    });
    let session_manager = match options.session_manager.clone() {
        Some(manager) => manager,
        None => Arc::new(Mutex::new(SessionManager::create(
            &cwd,
            Some(&get_default_session_dir(&cwd, Some(&agent_dir))),
        )?)),
    };

    // Ensure MCP providers are registered and built-in MCP skills are gated by
    // auth even on the bare SDK path (not just the CLI's createAgentSessionServices).
    let mcp_manager = match options.mcp_manager.clone() {
        Some(manager) => manager,
        None => match &mcp_manager_factory {
            Some(factory) => factory(Arc::clone(&auth_storage), Arc::clone(&settings_manager)),
            None => Arc::new(Mutex::new(McpManager::new(McpManagerOptions {
                auth_storage: Arc::clone(&auth_storage),
                get_user_servers: Some({
                    let settings_manager = Arc::clone(&settings_manager);
                    Arc::new(move || {
                        settings_manager.lock().expect("settings manager poisoned")
                            .get_global_mcp_servers().map(|servers| {
                                servers.into_iter().filter_map(|(name, config)| {
                                    serde_json::from_value(config).ok().map(|config| (name, config))
                                }).collect()
                            })
                    })
                }),
                begin_login: None,
            }))),
        },
    };
    {
        let mut registry = model_registry.lock().expect("model registry poisoned");
        let mcp_manager = Arc::clone(&mcp_manager);
        registry.set_on_oauth_providers_reset(Arc::new(move || {
            mcp_manager.lock().expect("mcp manager poisoned").register_user_providers();
        }));
    }

    if resource_loader.is_none() {
        let loader_options = DefaultResourceLoaderOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            settings_manager: Arc::clone(&settings_manager),
            extra_builtin_skill_overrides: {
                let mcp_manager = Arc::clone(&mcp_manager);
                Arc::new(move || mcp_manager.lock().expect("mcp manager poisoned")
                    .get_disabled_builtin_skill_overrides())
            },
        };
        let loader: Arc<dyn ResourceLoader> = match resource_loader_factory {
            Some(factory) => factory(loader_options),
            None => Arc::new(DefaultResourceLoader::new(
                crate::core::resource_loader::DefaultResourceLoaderOptions {
                    cwd: loader_options.cwd,
                    agent_dir: loader_options.agent_dir,
                    settings_manager: Some(loader_options.settings_manager),
                    extra_builtin_skill_overrides: Some(loader_options.extra_builtin_skill_overrides),
                    ..Default::default()
                },
            )),
        };
        loader.reload().await;
        crate::core::timings::time("resourceLoader.reload");
        resource_loader = Some(loader);
    }
    let resource_loader = resource_loader.expect("resource loader initialized");

    let existing_session = session_manager
        .lock()
        .expect("session manager poisoned")
        .build_session_context(None);
    let (has_thinking_entry, has_service_tier_entry) = {
        let manager = session_manager.lock().expect("session manager poisoned");
        let branch = manager.get_branch(None);
        (
            branch
                .iter()
                .any(|entry| entry.get("type").and_then(Value::as_str) == Some("thinking_level_change")),
            branch
                .iter()
                .any(|entry| entry.get("type").and_then(Value::as_str) == Some("service_tier_change")),
        )
    };
    let has_existing_session = !existing_session.messages.is_empty();

    let mut model = options.model.clone();
    let mut model_fallback_message: Option<String> = None;

    if model.is_none() && has_existing_session {
        if let Some(existing_model) = existing_session.model.clone() {
            let restored_model = model_registry
                .lock()
                .expect("model registry poisoned")
                .find(&existing_model.provider, &existing_model.model_id);
            if let Some(restored_model) = restored_model {
                if model_registry
                    .lock()
                    .expect("model registry poisoned")
                    .has_configured_auth(&restored_model)
                {
                    model = Some(restored_model);
                }
            }
            if model.is_none() {
                model_fallback_message = Some(format!(
                    "Could not restore model {}/{}",
                    existing_model.provider, existing_model.model_id
                ));
            }
        }
    }

    if model.is_none() {
        let (default_provider, default_model_id, default_thinking_level) = {
            let settings = settings_manager.lock().expect("settings manager poisoned");
            (
                settings.get_default_provider(),
                settings.get_default_model(),
                settings.get_default_thinking_level(),
            )
        };
        let resolution_options = FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: Vec::new(),
            is_continuing: has_existing_session,
            default_provider,
            default_model_id,
            default_thinking_level,
        };
        let result = with_model_registry(Arc::clone(&model_registry), move |registry| {
            Box::pin(async move { find_initial_model(&resolution_options, registry).await })
        }).await??;
        model = result.model;
        if model.is_none() {
            model_fallback_message = Some(format_no_models_available_message());
        } else if let Some(message) = model_fallback_message.as_mut() {
            let model_ref = model.as_ref().expect("model is set");
            message.push_str(&format!(". Using {}/{}", model_ref.provider, model_ref.id));
        }
    }

    let mut thinking_level = options.thinking_level.as_ref().map(|level| level.as_str().to_string());

    if thinking_level.is_none() && has_existing_session {
        thinking_level = Some(if has_thinking_entry {
            existing_session.thinking_level.clone()
        } else {
            settings_manager
                .lock()
                .expect("settings manager poisoned")
                .get_default_thinking_level()
                .unwrap_or_else(|| DEFAULT_THINKING_LEVEL.to_string())
        });
    }

    if thinking_level.is_none() {
        thinking_level = Some(
            settings_manager
                .lock()
                .expect("settings manager poisoned")
                .get_default_thinking_level()
                .unwrap_or_else(|| DEFAULT_THINKING_LEVEL.to_string()),
        );
    }

    let thinking_level = match &model {
        None => ThinkingLevel::Off,
        Some(model) => parse_thinking_level(&pi_ai::models::clamp_thinking_level(
            model,
            thinking_level.as_deref().unwrap_or(DEFAULT_THINKING_LEVEL),
        )),
    };

    let service_tier_preference = options.service_tier.clone().unwrap_or_else(|| {
        if has_service_tier_entry {
            existing_session.service_tier.clone()
        } else {
            Some(Some(
                settings_manager
                    .lock()
                    .expect("settings manager poisoned")
                    .get_default_service_tier(),
            ))
        }
    });
    let service_tier = if service_tier_preference.as_ref().and_then(|tier| tier.as_deref())
        == Some("priority")
        && model
            .as_ref()
            .map(|model| !pi_ai::models::supports_fast_mode(model))
            .unwrap_or(true)
    {
        Some(Some("default".to_string()))
    } else {
        service_tier_preference.clone()
    };

    let allowed_tool_names = options
        .creation
        .allowed_tool_names
        .clone()
        .or_else(|| options.tools.clone())
        .or_else(|| {
            if options.no_tools.as_deref() == Some("all") {
                Some(Vec::new())
            } else {
                None
            }
        });
    let include_goals = options
        .creation
        .include_goals
        .unwrap_or(options.tools.is_some() || options.no_tools.as_deref() != Some("all"));
    let initial_active_tool_names: Vec<String> = options
        .creation
        .initial_active_tool_names
        .clone()
        .unwrap_or_else(|| {
            if let Some(tools) = &options.tools {
                tools.clone()
            } else if options.no_tools.is_some() {
                Vec::new()
            } else {
                vec!["ipython".to_string()]
            }
        });

    // `let getActiveSessionManager = () => sessionManager` - reassigned to the
    // live session's manager once the session is constructed.
    let active_session_manager: Arc<Mutex<Arc<Mutex<SessionManager>>>> =
        Arc::new(Mutex::new(Arc::clone(&session_manager)));

    let convert_to_llm_with_block_images: Arc<dyn Fn(Vec<AgentMessage>) -> Vec<Message> + Send + Sync> = {
        let settings_manager = Arc::clone(&settings_manager);
        let active_session_manager = Arc::clone(&active_session_manager);
        Arc::new(move |messages: Vec<AgentMessage>| -> Vec<Message> {
            let session_manager = active_session_manager
                .lock()
                .expect("active session manager poisoned")
                .clone();
            let (model_tool_output_artifact_dir, session_id, policy, block_images) = {
                let manager = session_manager.lock().expect("session manager poisoned");
                let settings = settings_manager.lock().expect("settings manager poisoned");
                (
                    manager.get_session_artifact_dir(),
                    manager.get_session_id(),
                    resolve_model_tool_output_policy(Some(&settings.get_model_tool_output_policy())),
                    settings.get_block_images(),
                )
            };
            let converted = convert_to_llm(
                &messages,
                &ModelToolOutputPolicyOptions {
                    policy: Some(policy),
                    scope: model_tool_output_artifact_dir.map(|dir| ModelToolOutputScope {
                        session_id,
                        session_artifact_dir: dir,
                    }),
                },
            );
            if !block_images {
                return converted;
            }
            converted.into_iter().map(block_images_in_message).collect()
        })
    };

    let extension_runner_ref = Arc::new(ExtensionRunnerRef::new(None));

    let (steering_mode, follow_up_mode, transport, thinking_budgets) = {
        let settings = settings_manager.lock().expect("settings manager poisoned");
        (
            settings.get_steering_mode(),
            settings.get_follow_up_mode(),
            settings.get_transport(),
            settings.get_thinking_budgets().and_then(|value| serde_json::from_value(value).ok()),
        )
    };

    // `streamFn: async (model, context, options) => { const auth = await modelRegistry.getApiKeyAndHeaders(model); ... }`
    let stream_fn: StreamFn = {
        let model_registry = Arc::clone(&model_registry);
        let settings_manager = Arc::clone(&settings_manager);
        Arc::new(
            move |model: Model, context: Context, mut options: SimpleStreamOptions| {
                let model_registry = Arc::clone(&model_registry);
                let settings_manager = Arc::clone(&settings_manager);
                Box::pin(async move {
                    let request_model = model.clone();
                    let auth = with_model_registry(model_registry, move |registry| {
                        Box::pin(async move { registry.get_api_key_and_headers(&request_model).await })
                    }).await.unwrap_or_else(|error| crate::core::model_registry::ResolvedRequestAuth {
                        ok: false, api_key: None, headers: None, error: Some(error),
                    });
                    if !auth.ok {
                        // `throw new Error(auth.error)`; the stream contract reports request
                        // failures through a terminal assistant message instead.
                        let mut message = pi_ai::types::AssistantMessage::default();
                        message.api = model.api.clone();
                        message.provider = model.provider.clone();
                        message.model = model.id.clone();
                        message.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
                        message.error_message = auth.error.clone();
                        let stream = pi_ai::utils::event_stream::AssistantMessageEventStream::new();
                        stream.push(pi_ai::types::AssistantMessageEvent::Start {
                            partial: message.clone(),
                        });
                        stream.end(Some(message));
                        return stream;
                    }
                    if auth.api_key.is_some() {
                        options.stream.api_key = auth.api_key.clone();
                    }
                    if auth.headers.is_some() {
                        options.stream.headers = auth.headers.clone();
                    }
                    if options.stream.timeout_ms.is_none() {
                        let timeout_ms = settings_manager
                            .lock()
                            .expect("settings manager poisoned")
                            .get_provider_retry_settings()
                            .timeout_ms;
                        if let Some(timeout_ms) = timeout_ms {
                            options.stream.timeout_ms = Some(timeout_ms);
                        }
                    }
                    pi_ai::stream::stream_simple(&model, &context, Some(&options))
                })
            },
        )
    };

    // `onPayload: async (payload, _model) => { const runner = extensionRunnerRef.current; if (!runner?.hasHandlers("before_provider_request")) return payload; return runner.emitBeforeProviderRequest(payload); }`
    let on_payload: pi_ai::types::OnPayload = {
        let extension_runner_ref = Arc::clone(&extension_runner_ref);
        Arc::new(move |payload: Value, _model: &Model| {
            let runner = extension_runner_ref.current();
            let Some(runner) = runner else {
                return Box::pin(async move { Some(payload) }) as BoxFuture<Option<Value>>;
            };
            if !runner.has_handlers("before_provider_request") {
                return Box::pin(async move { Some(payload) }) as BoxFuture<Option<Value>>;
            }
            Box::pin(async move { Some(runner.emit_before_provider_request(payload).await) }) as BoxFuture<Option<Value>>
        })
    };

    let on_response: pi_ai::types::OnResponse = {
        let extension_runner_ref = Arc::clone(&extension_runner_ref);
        Arc::new(move |response: pi_ai::types::ProviderResponse, _model: &Model| {
            let runner = extension_runner_ref.current();
            let Some(runner) = runner else {
                return Box::pin(async {}) as BoxFuture<()>;
            };
            if !runner.has_handlers("after_provider_response") {
                return Box::pin(async {}) as BoxFuture<()>;
            }
            Box::pin(async move {
                runner.emit(crate::core::extensions::types::ExtensionEvent::AfterProviderResponse(
                    crate::core::extensions::types::AfterProviderResponsePayload {
                        status: response.status as f64,
                        headers: response.headers.into_iter()
                            .map(|(key, value)| (key, Value::String(value))).collect(),
                    },
                )).await;
            }) as BoxFuture<()>
        })
    };

    let session_id = session_manager
        .lock()
        .expect("session manager poisoned")
        .get_session_id();

    // `transformContext: async (messages) => { const runner = extensionRunnerRef.current; if (!runner) return messages; return runner.emitContext(messages); }`
    let transform_context = {
        let extension_runner_ref = Arc::clone(&extension_runner_ref);
        Arc::new(
            move |messages: Vec<AgentMessage>, signal: Option<tokio_util::sync::CancellationToken>| {
                let runner = extension_runner_ref.current();
                Box::pin(async move {
                    let Some(runner) = runner else { return messages; };
                    let Ok(values) = messages.iter().map(serde_json::to_value).collect::<Result<Vec<_>, _>>() else {
                        return messages;
                    };
                    let values = runner.emit_context(values).await;
                    let ctx = runner.create_context();
                    // ROOT CONTRACT v1 (Search): ONE shared provider-context
                    // deadline across filtering, reranking and line matching;
                    // every stage spends from this single token.
                    let search_budget = pi_jev::search::SearchBudget::from_now(
                        std::time::Duration::from_millis(pi_jev::search::SHARED_SEARCH_BUDGET_MS),
                    );
                    let values = crate::core::jev_bridge::filter_context_candidates(ctx.clone(), values, &search_budget).await;
                    let values = crate::core::jev_compaction::compact_context(ctx, values, signal).await;
                    values.into_iter().map(serde_json::from_value).collect::<Result<Vec<_>, _>>().unwrap_or(messages)
                }) as BoxFuture<Vec<AgentMessage>>
            },
        ) as Arc<
            dyn Fn(Vec<AgentMessage>, Option<tokio_util::sync::CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
                + Send
                + Sync,
        >
    };

    let agent_factory = agent_factory.unwrap_or_else(|| Arc::new(|options| {
        // `AgentOptions` allows `T[] | Promise<T[]>`; the port keeps the creation
        // options synchronous and defers the call into the returned future.
        let convert_to_llm: Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<Message>> + Send + Sync> = {
            let convert = Arc::clone(&options.convert_to_llm);
            Arc::new(move |messages: Vec<AgentMessage>| -> BoxFuture<Vec<Message>> {
                let convert = Arc::clone(&convert);
                Box::pin(async move { (convert)(messages) })
            })
        };
        let transform_context: Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> BoxFuture<Vec<AgentMessage>>
                + Send
                + Sync,
        > = {
            let transform = Arc::clone(&options.transform_context);
            Arc::new(
                move |messages: Vec<AgentMessage>,
                      signal: Option<tokio_util::sync::CancellationToken>|
                      -> BoxFuture<Vec<AgentMessage>> {
                    let transform = Arc::clone(&transform);
                    Box::pin(async move { (transform)(messages, signal).await })
                },
            )
        };
        Arc::new(pi_agent_core::agent::Agent::new(pi_agent_core::agent::AgentOptions {
            initial_state: Some(options.initial_state),
            convert_to_llm: Some(convert_to_llm),
            stream_fn: Some(options.stream_fn),
            on_payload: Some(options.on_payload),
            on_response: Some(options.on_response),
            session_id: Some(options.session_id),
            transform_context: Some(transform_context),
            steering_mode: Some(if options.steering_mode == "all" {
                pi_agent_core::agent::QueueMode::All
            } else { pi_agent_core::agent::QueueMode::OneAtATime }),
            follow_up_mode: Some(if options.follow_up_mode == "all" {
                pi_agent_core::agent::QueueMode::All
            } else { pi_agent_core::agent::QueueMode::OneAtATime }),
            transport: Some(options.transport),
            thinking_budgets: options.thinking_budgets,
            ..Default::default()
        }))
    }));
    let agent = agent_factory(CreateAgentSessionAgentOptions {
        initial_state: AgentState {
            system_prompt: String::new(),
            model: model.clone().unwrap_or_default(),
            thinking_level: thinking_level.clone(),
            service_tier: service_tier.clone(),
            tools: Some(Vec::new()),
            messages: Vec::new(),
            ..Default::default()
        },
        convert_to_llm: Arc::clone(&convert_to_llm_with_block_images),
        stream_fn,
        on_payload,
        on_response,
        session_id: session_id.clone(),
        transform_context,
        steering_mode,
        follow_up_mode,
        transport,
        thinking_budgets,
    });
    crate::core::performance_monitor::attach_performance_monitor(
        agent.as_ref(), std::path::Path::new(&agent_dir), session_id,
    );

    {
        let mut manager = session_manager.lock().expect("session manager poisoned");
        if has_existing_session {
            let mut state = agent.state();
            state.messages = existing_session.messages.clone();
            agent.set_state(state);
            if !has_thinking_entry {
                let _ = manager.append_thinking_level_change(&thinking_level_name(&thinking_level));
            }
        } else {
            if let Some(model_ref) = &model {
                let _ = manager.append_model_change(&model_ref.provider, &model_ref.id);
            }
            let _ = manager.append_thinking_level_change(&thinking_level_name(&thinking_level));
        }
        if !has_service_tier_entry {
            let _ = manager.append_service_tier_change(&service_tier_preference);
        }
    }

    let session = AgentSession::new(AgentSessionConfig {
        agent: Arc::clone(&agent),
        session_manager: Arc::clone(&session_manager),
        settings_manager: Arc::clone(&settings_manager),
        service_tier_preference: Some(service_tier_preference.clone()),
        cwd: cwd.clone(),
        // Only the explicit dir - the default may not match injected custom storage.
        agent_dir: options.agent_dir.clone(),
        scoped_models: options.scoped_models.clone(),
        resource_loader: Arc::clone(&resource_loader),
        custom_tools: options.custom_tools.clone(),
        model_registry: Arc::clone(&model_registry),
        mcp_manager: Some(mcp_manager),
        initial_active_tool_names: Some(initial_active_tool_names),
        allowed_tool_names,
        include_goals: Some(include_goals),
        include_compact_skill: options.creation.include_compact_skill,
        // REPAIR CURSOR: blocked on duplicate controller/autonomous types in core/agent_session.rs
        // (owned by another worker): its local `AgentRlmHeartbeatController` /
        // `AgentSessionMessageController` / `AgentAutonomousConfig` duplicate the canonical
        // core/cron_jobs.rs, core/agent_messages.rs and core/autonomous.rs ones the creation
        // options use. agent_session.rs must re-export those instead of re-declaring them.
        rlm_heartbeat_controller: options.creation.rlm_heartbeat_controller.clone(),
        agent_message_controller: options.creation.agent_message_controller.clone(),
        agent_observe_controller: options.creation.agent_observe_controller.clone(),
        extension_runner_ref: Some(Arc::clone(&extension_runner_ref)),
        session_start_event: options.session_start_event.clone(),
        rlm_depth: options.creation.rlm_depth,
        rlm_max_depth: options.creation.rlm_max_depth,
        rlm_session_dir: options.creation.rlm_session_dir.clone(),
        rlm_parent_node_id: options.creation.rlm_parent_node_id.clone(),
        rlm_parent_agent: options.creation.rlm_parent_agent.clone(),
        semantic_parent_session_id: options.creation.semantic_parent_session_id.clone(),
        semantic_spawned_by_request_id: options.creation.semantic_spawned_by_request_id.clone(),
        subagent_runtime_host: options.creation.subagent_runtime_host.clone(),
        prewarm_ipython_kernel: options.creation.prewarm_ipython_kernel,
        // REPAIR CURSOR: blocked on duplicate controller/autonomous types in core/agent_session.rs
        // (owned by another worker) - see the note above; `agent_session::AgentAutonomousConfig`
        // keeps only enabled/cwd, so converting here would silently drop gates and limits.
        autonomous: options
            .autonomous
            .clone()
            .or_else(|| options.creation.autonomous.clone()),
        serialized_refine: options.creation.serialized_refine,
        initial_goal: options.creation.initial_goal.clone(),
        base_tools_override: None,
        auto_refine_reviewer: None,
    })?;

    // `getActiveSessionManager = () => session.sessionManager`.
    *active_session_manager
        .lock()
        .expect("active session manager poisoned") = Arc::clone(&session.session_manager);

    let extensions_result = resource_loader.get_extensions();

    Ok(CreateAgentSessionResult {
        session,
        extensions_result,
        model_fallback_message,
    })
}

/// `convertToLlmWithBlockImages`: replaces image blocks with the disabled-image
/// text and merges adjacent copies, exactly like the TypeScript `map`/`filter`.
fn block_images_in_message(message: Message) -> Message {
    let replace = |content: &mut Vec<pi_ai::types::ImageOrTextContent>| {
        if !content
            .iter()
            .any(|block| matches!(block, pi_ai::types::ImageOrTextContent::Image(_)))
        {
            return;
        }
        let mapped: Vec<pi_ai::types::ImageOrTextContent> = content
            .iter()
            .map(|block| match block {
                pi_ai::types::ImageOrTextContent::Image(_) => {
                    pi_ai::types::ImageOrTextContent::Text(pi_ai::types::TextContent::new(
                        IMAGE_READING_DISABLED,
                    ))
                }
                other => other.clone(),
            })
            .collect();
        let mut filtered: Vec<pi_ai::types::ImageOrTextContent> = Vec::new();
        for block in &mapped {
            let is_duplicate_disabled_text = match block {
                pi_ai::types::ImageOrTextContent::Text(text) => {
                    text.text == IMAGE_READING_DISABLED
                        && filtered.last().is_some_and(|previous| match previous {
                            pi_ai::types::ImageOrTextContent::Text(previous) => {
                                previous.text == IMAGE_READING_DISABLED
                            }
                            pi_ai::types::ImageOrTextContent::Image(_) => false,
                        })
                }
                pi_ai::types::ImageOrTextContent::Image(_) => false,
            };
            if is_duplicate_disabled_text && !filtered.is_empty() {
                continue;
            }
            filtered.push(block.clone());
        }
        *content = filtered;
    };

    match message {
        Message::User(mut user) => {
            if let UserContent::Blocks(blocks) = &mut user.content {
                replace(blocks);
            }
            Message::User(user)
        }
        Message::ToolResult(mut result) => {
            replace(&mut result.content);
            Message::ToolResult(result)
        }
        other => other,
    }
}

/// `text: "Image reading is disabled."`.
const IMAGE_READING_DISABLED: &str = "Image reading is disabled.";

/// `clampThinkingLevel(model, thinkingLevel) as ThinkingLevel`.
pub fn parse_thinking_level(value: &str) -> ThinkingLevel {
    match value {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => ThinkingLevel::Off,
    }
}

/// `thinkingLevel` serialized back to its wire name.
pub fn thinking_level_name(level: &ThinkingLevel) -> String {
    level.as_str().to_string()
}

/// `join(agentDir, name)`.
fn join_path(base: &str, name: &str) -> String {
    std::path::Path::new(base)
        .join(name)
        .to_string_lossy()
        .to_string()
}

/// `process.cwd()`.
fn process_cwd() -> String {
    std::env::current_dir()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Keeps the imports that only appear in the TypeScript type surface.
#[allow(dead_code)]
fn type_surface_markers(
    _host: Arc<dyn SubagentRuntimeHost>,
    _controller: Arc<dyn AgentSessionMessageController>,
    _observe: Arc<dyn AgentObserveController>,
    _heartbeat: Arc<dyn AgentRlmHeartbeatController>,
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{ImageContent, TextContent, ToolResultMessage};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_session_prompts_the_faux_provider_and_delivers_events() {
        use pi_ai::providers::faux::{register_faux_provider, RegisterFauxProviderOptions, FauxResponseStep, faux_assistant_message, FauxAssistantContent};
        let root = tempfile::Builder::new().prefix("sdk-faux-").tempdir_in(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.port-env/tmp"),
        ).unwrap();
        let cwd = root.path().to_string_lossy().to_string();
        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some(format!("session-test-{}", uuid::Uuid::new_v4())),
            tokens_per_second: Some(0.0), ..Default::default()
        }));
        let model = provider.get_model();
        provider.set_responses(vec![FauxResponseStep::Factory(Arc::new(|context, _, _, model| {
            Box::pin(async move {
                assert!(context.messages.iter().any(|message| match message {
                    Message::User(user) => match &user.content {
                        UserContent::Text(text) => text == "Say hello",
                        UserContent::Blocks(blocks) => blocks.iter().any(|block| {
                            matches!(block, pi_ai::types::ImageOrTextContent::Text(text) if text.text == "Say hello")
                        }),
                    },
                    _ => false,
                }));
                let mut response = faux_assistant_message(FauxAssistantContent::Text("Hello from faux session.".to_string()), None);
                response.api = model.api.clone(); response.provider = model.provider.clone(); response.model = model.id.clone();
                response
            })
        }))]);
        let mut registry_auth = AuthStorage::in_memory(Default::default(), None);
        registry_auth.set_runtime_api_key(&model.provider, "synthetic-faux-key");
        let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(registry_auth)));
        let settings = Arc::new(Mutex::new(crate::core::settings_manager::SettingsManager::in_memory(
            serde_json::json!({"autoRefine":{"enabled":false},"retry":{"enabled":false},"compaction":{"enabled":false},"telemetryEnabled":false,"agentTracesEnabled":false}).as_object().unwrap().clone(),
        )));
        let loader = Arc::new(DefaultResourceLoader::new(crate::core::resource_loader::DefaultResourceLoaderOptions {
            cwd: cwd.clone(), agent_dir: cwd.clone(), settings_manager: Some(settings.clone()),
            no_extensions: true, no_skills: true, no_prompt_templates: true, no_themes: true,
            no_context_files: true, bundled_skills_dir: Some(None), ..Default::default()
        }));
        loader.reload().await;
        let result = create_agent_session(CreateAgentSessionOptions {
            cwd: Some(cwd.clone()), agent_dir: Some(cwd.clone()), model: Some(model),
            auth_storage: Some(Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(Default::default(), None)))),
            model_registry: Some(registry), settings_manager: Some(settings), resource_loader: Some(loader),
            session_manager: Some(Arc::new(Mutex::new(SessionManager::in_memory(Some(&cwd), Some(&cwd)).unwrap()))),
            no_tools: Some("all".to_string()),
            creation: AgentSessionCreationOptions { prewarm_ipython_kernel: Some(false), telemetry_disabled: Some(true), ..Default::default() },
            ..Default::default()
        }).await.unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let unsubscribe = result.session.subscribe(Arc::new(move |event| captured.lock().unwrap().push(event.type_name().to_string())));
        tokio::time::timeout(std::time::Duration::from_secs(20), result.session.prompt_and_wait("Say hello", None)).await.unwrap().unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(20), result.session.wait_for_headless_idle()).await.unwrap().unwrap();
        assert_eq!(provider.call_count(), 1);
        assert_eq!(provider.get_pending_response_count(), 0);
        assert!(result.session.messages().iter().any(|message| crate::core::side_question::read_assistant_text(message) == "Hello from faux session."));
        let captured = events.lock().unwrap().clone();
        assert!(captured.iter().any(|event| event == "message_end"));
        assert!(captured.iter().any(|event| event == "agent_end"));
        unsubscribe();
        result.session.dispose_async(Some(false)).await;
        provider.unregister();
    }

    #[test]
    fn thinking_level_name_round_trips_every_level() {
        for name in ["off", "minimal", "low", "medium", "high", "xhigh", "max"] {
            assert_eq!(thinking_level_name(&parse_thinking_level(name)), name);
        }
    }

    #[test]
    fn unknown_thinking_levels_fall_back_to_off() {
        assert_eq!(parse_thinking_level("bogus"), ThinkingLevel::Off);
    }

    #[test]
    fn image_blocks_become_disabled_text_and_adjacent_copies_merge() {
        let message = Message::User(pi_ai::types::UserMessage::new(
            UserContent::Blocks(vec![
                pi_ai::types::ImageOrTextContent::Image(ImageContent::new("a", "image/png")),
                pi_ai::types::ImageOrTextContent::Image(ImageContent::new("b", "image/png")),
                pi_ai::types::ImageOrTextContent::Text(TextContent::new("after")),
            ]),
            0,
        ));
        let converted = block_images_in_message(message);
        let Message::User(user) = converted else {
            panic!("expected a user message");
        };
        let UserContent::Blocks(blocks) = user.content else {
            panic!("expected blocks");
        };
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0],
            pi_ai::types::ImageOrTextContent::Text(TextContent::new(IMAGE_READING_DISABLED))
        );
        assert_eq!(blocks[1], pi_ai::types::ImageOrTextContent::Text(TextContent::new("after")));
    }

    #[test]
    fn tool_results_without_images_are_untouched() {
        let message = Message::ToolResult(ToolResultMessage {
            content: vec![pi_ai::types::ImageOrTextContent::Text(TextContent::new("ok"))],
            ..ToolResultMessage::new("id", "name", Vec::new(), false, 0)
        });
        let converted = block_images_in_message(message.clone());
        assert_eq!(converted, message);
    }

    #[test]
    fn join_path_uses_the_platform_separator() {
        assert_eq!(join_path("dir", "auth.json"), std::path::Path::new("dir").join("auth.json").to_string_lossy());
    }
}
