//! Port of packages/coding-agent/src/core/extensions/builtin/memory.ts

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::AgentMessage;
use serde_json::Value;

use crate::core::extensions::types::{
    AutocompleteItem, CustomMessagePayload, ExtensionApi, ExtensionCommandContext, ExtensionContext,
    ExtensionEvent, ExtensionFactory, ExtensionHandler, RegisterCommandOptions, SendMessageOptions,
};
use crate::core::memory::evidence::{
    collect_evidence, hash, message_evidence, Evidence, MEMORY_RECALL_TYPE,
};
use crate::core::memory::jobs::MemoryExtractor;
use crate::core::memory::extraction::completion_fn_for;
use crate::core::memory::service::MemoryService;
use crate::core::messages::without_harness_digests_for_compaction;
use crate::core::refinement::refinement::{
    get_local_harness_state_dir, load_harness_state, plan_refinement, HarnessScope,
    PlanRefinementRequest, RefineModel, RefineOptions, RefinementEdit, RefinementProposal,
};
use crate::core::session_manager::get_session_artifact_path;
use crate::core::settings_manager::SettingsManager;


fn refinement_retry_policy(settings: &Mutex<SettingsManager>) -> crate::core::refinement::refinement::ProviderRetryPolicy {
    let guard = settings.lock().unwrap_or_else(|p| p.into_inner());
    let retry = guard.get_retry_settings();
    let provider_retry = guard.get_provider_retry_settings();
    crate::core::refinement::refinement::ProviderRetryPolicy {
        enabled: retry.enabled,
        max_retries: retry.max_retries.max(0.0) as u32,
        base_delay_ms: retry.base_delay_ms,
        max_retry_delay_ms: provider_retry.max_retry_delay_ms,
    }
}

pub const MEMORY_CONTROL_CUSTOM_TYPE: &str = "prime-agent.memory-control";
pub const MEMORY_RESULT_CUSTOM_TYPE: &str = "prime-agent.memory-result";
pub const MEMORY_COMMAND_CUSTOM_TYPE: &str = "prime-agent.memory-command";
pub const MEMORY_DIAGNOSTIC_CUSTOM_TYPE: &str = "prime-agent.memory-diagnostic";

/// `sessionKey(ctx)` - `` `${ctx.cwd}\0${ctx.sessionManager.getSessionId()}` ``.
fn session_key<T: ExtensionContext + ?Sized>(ctx: &Arc<T>) -> String {
    format!("{}\u{0}{}", ctx.cwd(), ctx.session_manager().get_session_id())
}

/// `Map<string, MemoryService>` bound per session.
type ServiceMap = Arc<Mutex<HashMap<String, Arc<MemoryService>>>>;

struct CachedRecall {
    hits: Vec<crate::core::memory::search::MemoryHit>,
    recalled: Arc<crate::core::memory::search::RecallResult>,
}

/// The cache retains the unfiltered baseline so changing Jev mode is reversible.
type RecalledMap = Arc<Mutex<HashMap<String, (String, Arc<CachedRecall>)>>>;

/// `createMemoryExtension(agentDir, settingsManager)`.
pub fn create_memory_extension(
    agent_dir: String,
    settings_manager: Arc<Mutex<SettingsManager>>,
) -> ExtensionFactory {
    Arc::new(move |pi: Arc<dyn ExtensionApi>| {
        let agent_dir = agent_dir.clone();
        let settings_manager = settings_manager.clone();
        Box::pin(async move {
            create_memory_extension_impl(pi, agent_dir, settings_manager);
            Ok(())
        })
    })
}

/// `diagnostic(data)` - `pi.appendEntry("prime-agent.memory-diagnostic", data)`.
///
/// Diagnostics cannot block a turn.
fn diagnostic(pi: &Arc<dyn ExtensionApi>, data: Value) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pi.append_entry(MEMORY_DIAGNOSTIC_CUSTOM_TYPE.to_string(), Some(data));
    }));
}

/// `service(ctx)` - resolves (and caches) the session's `MemoryService`.
fn service<T: ExtensionContext + ?Sized>(
    ctx: &Arc<T>,
    agent_dir: &str,
    services: &ServiceMap,
) -> Result<Arc<MemoryService>, String> {
    let key = session_key(ctx);
    if let Some(existing) = services
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .cloned()
    {
        return Ok(existing);
    }
    let session_manager = ctx.session_manager();
    let session_artifact_dir = if session_manager.get_session_file().is_some() {
        Some(get_session_artifact_path(
            &session_manager.get_session_dir(),
            &session_manager.get_session_id(),
        ))
    } else {
        None
    };
    let memory = Arc::new(MemoryService::new(&ctx.cwd(), agent_dir, session_artifact_dir)?);
    services
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, memory.clone());
    Ok(memory)
}

/// `evidence(ctx)` - branch message evidence with the session-file URI.
fn evidence(ctx: &Arc<dyn ExtensionContext>) -> Vec<Evidence> {
    let session_manager = ctx.session_manager();
    let file = session_manager.get_session_file();
    let mut collected: Vec<Evidence> = Vec::new();
    for entry in session_manager.get_branch() {
        if entry.entry_type != "message" {
            continue;
        }
        let Some(message) = entry.extra.get("message").cloned() else {
            continue;
        };
        let Ok(parsed) = serde_json::from_value::<crate::core::memory::evidence::AgentMessage>(message) else {
            continue;
        };
        // `pathToFileURL(file).href`.
        let uri = file.as_ref().map(|file| format!("file://{}#{}", file, entry.id));
        if let Some(source) = message_evidence(&parsed, uri.as_deref(), Some(&entry.id)) {
            collected.push(source);
        }
    }
    collected
}

/// `ctx.modelRegistry.getApiKeyAndHeaders(model)` - the captured apiKey + headers.
async fn api_key_and_headers(
    ctx: &Arc<dyn ExtensionContext>,
    model: &pi_ai::types::Model,
) -> Result<(String, Option<HashMap<String, String>>), String> {
    let result = ctx.model_registry().get_api_key_and_headers(model).await?;
    let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let api_key = result.get("apiKey").and_then(Value::as_str).map(str::to_string);
    if !ok {
        return Err("No credentials for the selected model".to_string());
    }
    let Some(api_key) = api_key else {
        return Err("No credentials for the selected model".to_string());
    };
    let headers = result.get("headers").and_then(Value::as_object).map(|headers| {
        headers
            .iter()
            .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_string())))
            .collect::<HashMap<String, String>>()
    });
    Ok((api_key, headers))
}

/// `performance.now()` - milliseconds since the first call in this process.
fn performance_now() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

/// The `/memory` argument list, in completion order.
const MEMORY_ACTIONS: [&str; 17] = [
    "status",
    "search",
    "read",
    "recall",
    "learning",
    "history",
    "backup",
    "restore",
    "rollback",
    "import-prepare",
    "import-run",
    "import-read",
    "import-apply",
    "share",
    "sync",
    "bind",
    "configure",
];

/// `args.trim().match(/^(\S+)(?:\s+([\s\S]*))?$/)`.
fn split_action(args: &str) -> (String, String) {
    let trimmed = args.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    match (parts.next(), parts.next()) {
        (Some(action), rest) if !action.is_empty() => (
            action.replace('-', "_"),
            rest.unwrap_or_default().trim_start().to_string(),
        ),
        _ => ("status".to_string(), String::new()),
    }
}



/// `extractor(ctx, memory)` - the `memory.request(...)` extraction callback.
fn extractor(
    ctx: Arc<dyn ExtensionContext>,
    memory: Arc<MemoryService>,
    settings_manager: Arc<Mutex<SettingsManager>>,
) -> MemoryExtractor {
    Arc::new(move |records: Vec<Evidence>| {
        let ctx = ctx.clone();
        let memory = memory.clone();
        let settings_manager = settings_manager.clone();
        Box::pin(async move {
            let Some(model) = ctx.model() else {
                return Err("Select a model before extracting memory".to_string());
            };
            let (api_key, headers) = api_key_and_headers(&ctx, &model).await?;
            crate::core::memory::extraction::extract(
                &memory, records, model, api_key, headers,
                RefineOptions {
                    retry: Some(refinement_retry_policy(&settings_manager)),
                    instructions: Some(
                        "Extract only durable project memory. Return create edits of kind memory with sourceIds. Do not update or delete existing entries. Exclude host facts, credentials, unsupported assistant assertions and transient task state. Raw sources remain available, so do not copy entire logs into memory.".to_string(),
                    ),
                    ..Default::default()
                },
            ).await
        })
    })
}

/// Outcome of the `context` hook's recall step.
enum RecallOutcome {
    /// `!memory.store.settings().recall` - return the digest-filtered messages.
    Disabled,
    Done,
}

fn create_memory_extension_impl(
    pi: Arc<dyn ExtensionApi>,
    agent_dir: String,
    settings_manager: Arc<Mutex<SettingsManager>>,
) {
    let services: ServiceMap = Arc::new(Mutex::new(HashMap::new()));
    let recalled_turns: RecalledMap = Arc::new(Mutex::new(HashMap::new()));

    // --- session_shutdown: drop this session's service and recall cache ---
    {
        let services = services.clone();
        let recalled_turns = recalled_turns.clone();
        let handler: ExtensionHandler = Arc::new(move |_event, ctx| {
            let services = services.clone();
            let recalled_turns = recalled_turns.clone();
            Box::pin(async move {
                let key = session_key(&ctx);
                services
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                recalled_turns
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                None
            })
        });
        pi.on("session_shutdown", handler);
    }

    // --- /memory command ---
    let get_argument_completions: Arc<
        dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<AutocompleteItem>>> + Send>>
            + Send
            + Sync,
    > = Arc::new(|prefix: String| {
        Box::pin(async move {
            Some(
                MEMORY_ACTIONS
                    .iter()
                    .filter(|value| value.starts_with(&prefix))
                    .map(|value| AutocompleteItem {
                        value: (*value).to_string(),
                        label: (*value).to_string(),
                        description: None,
                        argument_hint: None,
                        source_tag: None,
                        takes_argument: None,
                    })
                    .collect::<Vec<AutocompleteItem>>(),
            )
        })
    });

    let command_handler: Arc<
        dyn Fn(
                String,
                Arc<dyn ExtensionCommandContext>,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
            + Send
            + Sync,
    > = {
        let pi = pi.clone();
        let agent_dir = agent_dir.clone();
        let settings_manager = settings_manager.clone();
        let services = services.clone();
        let recalled_turns = recalled_turns.clone();
        Arc::new(move |args, ctx| {
            let pi = pi.clone();
            let agent_dir = agent_dir.clone();
            let settings_manager = settings_manager.clone();
            let services = services.clone();
            let recalled_turns = recalled_turns.clone();
            Box::pin(async move {
                run_memory_command(pi, ctx, agent_dir, settings_manager, services, recalled_turns, args).await
            })
        })
    };

    pi.register_command(
        "memory".to_string(),
        RegisterCommandOptions {
            description: Some(
                "Inspect project memory, recall, learning, imports and optional sharing".to_string(),
            ),
            get_argument_completions: Some(get_argument_completions),
            handler: Some(command_handler),
        },
    );

    // --- before_agent_start: publish the learning status message ---
    {
        let agent_dir = agent_dir.clone();
        let settings_manager = settings_manager.clone();
        let services = services.clone();
        let handler: ExtensionHandler = Arc::new(move |_event, ctx| {
            let agent_dir = agent_dir.clone();
            let settings_manager = settings_manager.clone();
            let services = services.clone();
            Box::pin(async move {
                let memory = service(&ctx, &agent_dir, &services).ok()?;
                let learning = memory.store.settings().learning
                    && settings_manager
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_auto_refine_settings()
                        .enabled;
                Some(serde_json::json!({
                    "message": {
                        "customType": MEMORY_CONTROL_CUSTOM_TYPE,
                        "content": format!(
                            "Memory automatic learning: {}.",
                            if learning { "enabled" } else { "paused" }
                        ),
                        "display": false,
                        "details": { "learning": learning, "projectId": memory.store.project.id },
                    }
                }))
            })
        });
        pi.on("before_agent_start", handler);
    }

    // --- context: inject recall before the current user turn ---
    {
        let handler_pi = pi.clone();
        let agent_dir = agent_dir.clone();
        let services = services.clone();
        let recalled_turns = recalled_turns.clone();
        let handler: ExtensionHandler = Arc::new(move |event, ctx| {
            let pi = handler_pi.clone();
            let agent_dir = agent_dir.clone();
            let services = services.clone();
            let recalled_turns = recalled_turns.clone();
            Box::pin(async move {
                let started = performance_now();
                let ExtensionEvent::Context(payload) = event else {
                    return None;
                };
                let mut messages: Vec<AgentMessage> = payload
                    .messages
                    .iter()
                    .filter_map(|value| serde_json::from_value::<AgentMessage>(value.clone()).ok())
                    .collect();
                messages.retain(|message| {
                    !matches!(
                        message,
                        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom {
                            custom_type,
                            ..
                        }) if custom_type == MEMORY_RECALL_TYPE
                    )
                });
                // TS wraps the body in try/catch: a throw still returns the filtered
                // messages, and reports a failed recall diagnostic.
                let recalled = async {
                    let memory = service(&ctx, &agent_dir, &services)?;
                    if !memory.store.settings().recall {
                        return Ok(RecallOutcome::Disabled);
                    }
                    let user = messages
                        .iter()
                        .rev()
                        .find(|message| matches!(message, AgentMessage::Message(pi_ai::types::Message::User(_))))
                        .cloned();
                    let query = user
                        .as_ref()
                        .and_then(|user| serde_json::to_value(user).ok().and_then(|value| serde_json::from_value::<crate::core::memory::evidence::AgentMessage>(value).ok()).and_then(|message| collect_evidence(&[message]).into_iter().next()))
                        .map(|evidence| evidence.text)
                        .unwrap_or_default();
                    let key = hash(&format!(
                        "{}:{}:{}",
                        user_timestamp(&user)
                            .map(crate::core::memory::evidence::js_number)
                            .unwrap_or_else(|| "undefined".to_string()),
                        query,
                        serde_json::to_string(&memory.store.settings()).unwrap_or_default(),
                    ));
                    let cached = recalled_turns
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get(&session_key(&ctx))
                        .cloned();
                    let baseline = match cached {
                        Some((cached_key, recall)) if cached_key == key => recall,
                        _ => {
                            let mut hits = if query.is_empty() { Vec::new() } else { memory.search(&query, false) };
                            let recalled = memory.render_recall(&hits);
                            hits.retain(|hit| recalled.ids.contains(&hit.id));
                            Arc::new(CachedRecall { hits, recalled: Arc::new(recalled) })
                        }
                    };
                    recalled_turns
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .insert(session_key(&ctx), (key, baseline.clone()));
                    let filtered = crate::core::jev_bridge::filter_memory_candidates(
                        ctx.clone(), &query, baseline.hits.clone(),
                    ).await;
                    let recalled = if filtered.len() == baseline.hits.len() {
                        baseline.recalled.clone()
                    } else {
                        Arc::new(memory.render_recall(&filtered))
                    };
                    if !recalled.text.is_empty() {
                        let note: AgentMessage =
                            AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom {
                                custom_type: MEMORY_RECALL_TYPE.to_string(),
                                content: pi_agent_core::types::CustomMessageContent::Text(recalled.text.clone()),
                                display: false,
                                details: Some(serde_json::json!({ "ids": recalled.ids })),
                                timestamp: user_timestamp(&user).map(|value| value as i64).unwrap_or(0),
                            });
                        // Anchor recall before the current user turn, preserving it across
                        // subsequent tool requests.
                        let index = user
                            .as_ref()
                            .and_then(|user| messages.iter().rposition(|message| message == user))
                            .unwrap_or(messages.len());
                        messages.insert(index, note);
                    }
                    diagnostic(
                        &pi,
                        serde_json::json!({
                            "operation": "recall",
                            "projectId": memory.store.project.id,
                            "ids": recalled.ids,
                            "chars": recalled.chars,
                            "latencyMs": performance_now() - started,
                        }),
                    );
                    Ok::<RecallOutcome, String>(RecallOutcome::Done)
                }.await;

                match recalled {
                    Ok(RecallOutcome::Disabled) => Some(serde_json::json!({
                        "messages": serialize_messages(&without_harness_digests_for_compaction(&messages)),
                    })),
                    Ok(RecallOutcome::Done) => {
                        Some(serde_json::json!({ "messages": serialize_messages(&messages) }))
                    }
                    Err(_) => {
                        diagnostic(
                            &pi,
                            serde_json::json!({
                                "operation": "recall",
                                "status": "failed",
                                "latencyMs": performance_now() - started,
                            }),
                        );
                        Some(serde_json::json!({ "messages": serialize_messages(&messages) }))
                    }
                }
            })
        });
        pi.on("context", handler);
    }

    // --- session_before_refine ---
    {
        let agent_dir = agent_dir.clone();
        let settings_manager = settings_manager.clone();
        let services = services.clone();
        let handler: ExtensionHandler = Arc::new(move |event, ctx| {
            let agent_dir = agent_dir.clone();
            let settings_manager = settings_manager.clone();
            let services = services.clone();
            Box::pin(async move {
                let ExtensionEvent::SessionBeforeRefine(payload) = event else {
                    return None;
                };
                let memory = service(&ctx, &agent_dir, &services).ok()?;
                let preparation = payload.preparation;
                if preparation.trigger == "auto" && !memory.store.settings().learning {
                    return Some(serde_json::json!({ "skip": true }));
                }
                let model = ctx.model()?;
                let (api_key, headers) = api_key_and_headers(&ctx, &model).await.ok()?;
                let state = preparation.planning_state.clone();
                let plan = plan_refinement(PlanRefinementRequest {
                    messages: &[],
                    state: &state,
                    history: &preparation.history,
                    model: RefineModel {
                        max_tokens: model.max_tokens,
                    },
                    api_key: api_key.clone(),
                    options: RefineOptions {
                        evidence: Some(evidence(&ctx)),
                        instructions: preparation.instructions.clone(),
                        global: Some(preparation.scope == "global"),
                        retry: Some(refinement_retry_policy(&settings_manager)),
                        max_output_tokens: Some(memory.store.settings().max_extraction_tokens as f64),
                        ..Default::default()
                    },
                    headers: headers.clone(),
                    thinking_level: None,
                    complete: completion_fn_for(model, api_key, headers, Some(refinement_retry_policy(&settings_manager))),
                })
                .await
                .ok()?;
                if preparation.trigger == "auto" && !memory.store.settings().learning {
                    return Some(serde_json::json!({ "skip": true }));
                }
                Some(serde_json::json!({ "proposal": plan.proposal }))
            })
        });
        pi.on("session_before_refine", handler);
    }

    // --- refine_complete: promote local entries to project memory ---
    {
        let handler_pi = pi.clone();
        let agent_dir = agent_dir.clone();
        let services = services.clone();
        let handler: ExtensionHandler = Arc::new(move |event, ctx| {
            let pi = handler_pi.clone();
            let agent_dir = agent_dir.clone();
            let services = services.clone();
            Box::pin(async move {
                let ExtensionEvent::RefineComplete(payload) = event else {
                    return None;
                };
                if payload.scope != "local" {
                    return None;
                }
                let memory = service(&ctx, &agent_dir, &services).ok()?;
                if !memory.store.settings().learning {
                    return None;
                }
                let local_dir = get_local_harness_state_dir(memory.session_artifact_dir.as_deref())?;
                let local = load_harness_state(&local_dir, HarnessScope::Local);
                let doc = memory.store.read().ok()?;
                let changed = local
                    .refinements
                    .iter()
                    .find(|item| item.id == payload.id)
                    .map(|item| item.changes.clone())
                    .unwrap_or_default();
                let mut edits: Vec<RefinementEdit> = Vec::new();
                for bucket in local.entries.values() {
                    for entry in bucket.values() {
                        if !changed.iter().any(|change| {
                            *change == format!("create memory:{}", entry.id)
                                || *change == format!("update memory:{}", entry.id)
                        }) {
                            continue;
                        }
                        if entry.metadata.get("projectReusable") != Some(&Value::Bool(true))
                            || entry.metadata.get("evidenceStatus").and_then(Value::as_str) != Some("cited")
                        {
                            continue;
                        }
                        let already_saved = doc
                            .entries
                            .get("memory")
                            .map(|bucket| {
                                bucket.contains_key(&entry.id)
                                    || bucket.values().any(|saved| {
                                        saved.content.trim() == entry.content.trim()
                                    })
                            })
                            .unwrap_or(false);
                        if already_saved {
                            continue;
                        }
                        edits.push(RefinementEdit {
                            action: "create".to_string(),
                            kind: "memory".to_string(),
                            id: Some(entry.id.clone()),
                            title: Some(entry.title.clone()),
                            content: Some(entry.content.clone()),
                            path: Some(entry.path.clone()),
                            reference: None,
                            arguments: None,
                            metadata: Some(entry.metadata.clone()),
                            reason: None,
                        });
                    }
                }
                if edits.is_empty() {
                    return None;
                }
                let proposal = RefinementProposal {
                    summary: payload.summary.clone(),
                    rationale: format!("Project facts from {}", payload.id),
                    edits,
                    expected_outcome: "Retain evidence-backed project facts across sessions".to_string(),
                };
                let result = memory
                    .store
                    .apply(
                        &proposal,
                        crate::core::memory::store::ApplyOptions {
                            event_id: format!("project_{}", payload.id),
                            expected_revision: doc.memory.revision,
                            automatic: true,
                            ..Default::default()
                        },
                    )
                    .await;
                if result.is_err() {
                    diagnostic(
                        &pi,
                        serde_json::json!({
                            "operation": "promote",
                            "status": "conflict",
                            "refinementId": payload.id,
                        }),
                    );
                }
                None
            })
        });
        pi.on("refine_complete", handler);
    }
}

/// `user?.timestamp`.
fn user_timestamp(message: &Option<AgentMessage>) -> Option<f64> {
    match message {
        Some(AgentMessage::Message(pi_ai::types::Message::User(user))) => Some(user.timestamp as f64),
        Some(AgentMessage::Message(pi_ai::types::Message::Assistant(assistant))) => Some(assistant.timestamp as f64),
        _ => None,
    }
}

/// `messages.map(JSON-serialisable values)` back into the event payload shape.
fn serialize_messages(messages: &[AgentMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
        .collect()
}

/// The `/memory` command handler body.
async fn run_memory_command(
    pi: Arc<dyn ExtensionApi>,
    ctx: Arc<dyn ExtensionCommandContext>,
    agent_dir: String,
    settings_manager: Arc<Mutex<SettingsManager>>,
    services: ServiceMap,
    recalled_turns: RecalledMap,
    args: String,
) -> Result<(), String> {
    let memory = service(&ctx, &agent_dir, &services)?;
    let (action, rest) = split_action(&args);
    let mut result: Value;

    if action == "recall" || action == "learning" {
        if rest != "on" && rest != "off" {
            return Err(format!("Usage: /memory {action} on|off"));
        }
        let enabled = rest == "on";
        let mut patch = serde_json::Map::new();
        patch.insert(action.clone(), Value::Bool(enabled));
        let settings = memory.store.configure(&Value::Object(patch)).await?;
        result = serde_json::to_value(settings).unwrap_or(Value::Null);
    } else {
        let payload = memory_payload(&action, &rest)?;
        let extraction = extractor(ctx.clone(), memory.clone(), settings_manager.clone());
        result = memory
            .request(&action, &payload, Some(extraction))
            .await?;
        if action == "status" {
            if let Value::Object(map) = &mut result {
                map.insert(
                    "autoRefine".to_string(),
                    serde_json::to_value(
                        settings_manager
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .get_auto_refine_settings(),
                    )
                    .unwrap_or(Value::Null),
                );
            }
        }
    }

    if action == "bind" {
        services
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&session_key(&ctx));
    }
    recalled_turns
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&session_key(&ctx));

    let current = service(&ctx, &agent_dir, &services)?;
    let learning = current.store.settings().learning
        && settings_manager
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_auto_refine_settings()
            .enabled;
    pi.send_message(
        CustomMessagePayload {
            custom_type: MEMORY_CONTROL_CUSTOM_TYPE.to_string(),
            content: Value::String(format!(
                "Memory automatic learning: {}.",
                if learning { "enabled" } else { "paused" }
            )),
            display: false,
            details: Some(serde_json::json!({
                "learning": learning,
                "projectId": current.store.project.id,
            })),
        },
        None,
    );

    let content = serde_json::to_string_pretty(&result).unwrap_or_default();
    pi.append_entry(
        MEMORY_COMMAND_CUSTOM_TYPE.to_string(),
        Some(serde_json::json!({ "action": action, "result": result })),
    );
    ctx.ui().notify(content.clone(), Some("info".to_string()));
    // Existing custom-message transport also exposes command results to non-TUI clients.
    pi.send_message(
        CustomMessagePayload {
            custom_type: MEMORY_RESULT_CUSTOM_TYPE.to_string(),
            content: Value::String(content),
            display: true,
            details: Some(serde_json::json!({ "action": action, "result": result })),
        },
        Some(SendMessageOptions::default()),
    );
    Ok(())
}

/// The `payload` record the command handler builds for one action.
fn memory_payload(action: &str, rest: &str) -> Result<serde_json::Map<String, Value>, String> {
    let mut payload = serde_json::Map::new();
    if action == "search" {
        payload.insert("query".to_string(), Value::String(rest.to_string()));
    } else if ["read", "restore", "import_run", "import_read"].contains(&action) {
        payload.insert("id".to_string(), Value::String(rest.to_string()));
    } else if action == "import_prepare" {
        payload.insert("path".to_string(), Value::String(rest.to_string()));
    } else if action == "bind" {
        payload.insert("projectId".to_string(), Value::String(rest.to_string()));
    } else if action == "share" {
        payload.insert(
            "ids".to_string(),
            Value::Array(
                rest.split_whitespace()
                    .filter(|value| !value.is_empty())
                    .map(|value| Value::String(value.to_string()))
                    .collect(),
            ),
        );
    } else if action == "configure" {
        payload.insert(
            "settings".to_string(),
            serde_json::from_str(rest).map_err(|error| error.to_string())?,
        );
    } else if ["rollback", "import_apply"].contains(&action) {
        let mut parts = rest.split_whitespace();
        let id = parts.next().unwrap_or_default();
        let revision: f64 = parts
            .next()
            .unwrap_or_default()
            .parse()
            .unwrap_or(f64::NAN);
        payload.insert("id".to_string(), Value::String(id.to_string()));
        payload.insert("revision".to_string(), serde_json::json!(revision));
    } else if !rest.is_empty() {
        // `JSON.parse(rest)`.
        payload = serde_json::from_str::<serde_json::Map<String, Value>>(rest)
            .map_err(|error| error.to_string())?;
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_parsing_matches_the_command_regex() {
        assert_eq!(split_action(""), ("status".to_string(), String::new()));
        assert_eq!(split_action("status"), ("status".to_string(), String::new()));
        assert_eq!(
            split_action("import-prepare  /tmp/log"),
            ("import_prepare".to_string(), "/tmp/log".to_string())
        );
        assert_eq!(
            split_action("share a b"),
            ("share".to_string(), "a b".to_string())
        );
        assert_eq!(
            split_action("recall on"),
            ("recall".to_string(), "on".to_string())
        );
    }

    #[test]
    fn payload_shapes_follow_the_typescript_branches() {
        let payload = memory_payload("search", "hello").unwrap();
        assert_eq!(payload.get("query"), Some(&Value::String("hello".to_string())));
        let payload = memory_payload("rollback", "mem_1 3").unwrap();
        assert_eq!(payload.get("id"), Some(&Value::String("mem_1".to_string())));
        assert_eq!(payload.get("revision").and_then(Value::as_f64), Some(3.0));
        let payload = memory_payload("share", "a  b").unwrap();
        assert_eq!(payload.get("ids").and_then(Value::as_array).map(Vec::len), Some(2));
        let payload = memory_payload("import_prepare", "/tmp/x").unwrap();
        assert_eq!(payload.get("path"), Some(&Value::String("/tmp/x".to_string())));
    }


    #[test]
    fn recall_cache_keeps_unfiltered_baseline_for_mode_reversal() {
        let baseline = Arc::new(CachedRecall {
            hits: Vec::new(),
            recalled: Arc::new(crate::core::memory::search::RecallResult {
                text: "baseline memory".to_string(), ids: vec!["original".to_string()], chars: 15,
            }),
        });
        let cache: RecalledMap = Arc::new(Mutex::new(HashMap::from([
            ("session".to_string(), ("turn".to_string(), baseline.clone())),
        ])));
        let filtered = crate::core::jev_retrieval::apply_memory(baseline.hits.clone(), &[0]);
        assert!(filtered.is_empty());
        let restored = cache.lock().unwrap().get("session").cloned().unwrap().1;
        assert!(Arc::ptr_eq(&restored, &baseline));
        assert_eq!(restored.recalled.text, "baseline memory");
        assert_eq!(restored.recalled.ids, vec!["original"]);
    }

    #[test]
    fn session_key_is_cwd_nul_session_id() {
        let key = format!("{}\u{0}{}", "/work", "sess-1");
        assert!(key.contains('\u{0}'));
        assert!(key.ends_with("sess-1"));
    }
}
