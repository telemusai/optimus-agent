//! OCC3.1 native execution through a registered offline, non-latent custom provider.
use pi_ai::providers::faux::*;
use pi_ai::types::{AssistantMessage, ContentBlock, Context, Message, Model, ToolCall};
use pi_coding_agent::core::occ::{textmas::{MAX_CONTEXT_BYTES, TextMasCallStatus, TextMasMode, TextMasRole, TextMasUsageLedger, TextMasUsageRecord, usage_record}, textmas_settings};
use pi_coding_agent::core::occ::textmas_activity::{TextMasActivitySnapshot, TextMasActivityStatus};
use pi_coding_agent::core::extensions::types::{ExtensionEvent, ExtensionFactory, InputSource};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::*;
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

static ENV: Mutex<()> = Mutex::new(());

const HIERARCHICAL_CONCURRENCY_ENV: &str = "PRIME_AGENT_OCC3_TEXTMAS_HIERARCHICAL_CONCURRENCY";

// All suites hold ENV while using Fixture because profiles are process-global.
// Scope this second process-global setting too, and restore it on every exit path.
struct ScopedHierarchicalConcurrencyEnv(Option<std::ffi::OsString>);
impl ScopedHierarchicalConcurrencyEnv {
    fn set(value: Option<&str>) -> Self {
        let previous = std::env::var_os(HIERARCHICAL_CONCURRENCY_ENV);
        match value {
            Some(value) => std::env::set_var(HIERARCHICAL_CONCURRENCY_ENV, value),
            None => std::env::remove_var(HIERARCHICAL_CONCURRENCY_ENV),
        }
        Self(previous)
    }
}
impl Drop for ScopedHierarchicalConcurrencyEnv {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => std::env::set_var(HIERARCHICAL_CONCURRENCY_ENV, value),
            None => std::env::remove_var(HIERARCHICAL_CONCURRENCY_ENV),
        }
    }
}

// A bounded fixture assertion must not detach its still-waiting prompt task.
struct AbortOnDrop(tokio::task::AbortHandle);
impl Drop for AbortOnDrop {
    fn drop(&mut self) { self.0.abort(); }
}

struct Fixture {
    root: tempfile::TempDir,
    old_profile: Option<std::ffi::OsString>,
    provider: FauxProviderRegistration,
    captured: Arc<Mutex<Vec<Context>>>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("workspace")).unwrap();
        std::fs::create_dir_all(root.path().join("profile")).unwrap();
        std::fs::write(
            root.path().join("workspace/AGENTS.md"),
            "Preserve TEXTMAS_PROJECT_INSTRUCTIONS.",
        )
        .unwrap();
        let old_profile = std::env::var_os("PRIME_AGENT_CODING_AGENT_DIR");
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", root.path().join("profile"));
        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some("textmas-custom-nonlatent".into()),
            api: Some("offline-textmas-api".into()),
            models: Some(vec![FauxModelDefinition { id: "ordinary-model".into(), reasoning: Some(false), max_tokens: Some(2048.0), context_window: Some(128000.0), ..Default::default() }]),
            tokens_per_second: Some(0.0),
            ..Default::default()
        }));
        Self {
            root,
            old_profile,
            provider,
            captured: Default::default(),
        }
    }

    async fn session(&self, depth: i64, retry: bool) -> Arc<AgentSession> {
        self.session_with_extensions(depth, retry, Vec::new()).await
    }

    async fn session_with_extensions(
        &self,
        depth: i64,
        retry: bool,
        extension_factories: Vec<ExtensionFactory>,
    ) -> Arc<AgentSession> {
        let cwd = self
            .root
            .path()
            .join("workspace")
            .to_string_lossy()
            .into_owned();
        let profile = self
            .root
            .path()
            .join("profile")
            .to_string_lossy()
            .into_owned();
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(json!({
            "autoRefine":{"enabled":false}, "retry":{"enabled":retry,"maxRetries":2,"baseDelayMs":1}, "compaction":{"enabled":false},
            "telemetryEnabled":false, "agentTracesEnabled":false
        }).as_object().unwrap().clone())));
        let mut model: Model = self.provider.get_model();
        model.input.push(pi_ai::types::InputModality::Image);
        let auth = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            Default::default(),
            None,
        )));
        auth.lock()
            .await
            .set_runtime_api_key(&model.provider, "synthetic-textmas-key");
        let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(Default::default(), None),
        )));
        registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, "synthetic-textmas-key");
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(profile.clone()),
            auth_storage: Some(auth),
            settings_manager: Some(settings.clone()),
            model_registry: Some(registry),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir: profile.clone(),
                settings_manager: Some(settings),
                extension_factories,
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            }),
        })
        .await
        .unwrap();
        let manager = SessionManager::create(&cwd, Some(&profile)).unwrap();
        create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::new(Mutex::new(manager)),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                rlm_depth: Some(depth),
                ..Default::default()
            },
        })
        .await
        .unwrap()
        .session
    }

    fn reply(&self, message: AssistantMessage, file: Option<String>, gate: Option<Arc<tokio::sync::Semaphore>>) {
        let captured = self.captured.clone();
        self.provider.append_responses(vec![FauxResponseStep::Factory(Arc::new(move |context, options, _, model| {
            assert_eq!(model.provider, "textmas-custom-nonlatent");
            assert_eq!(model.api, "offline-textmas-api");
            assert!(!model.reasoning);
            let options = options.unwrap();
            assert_eq!(options.api_key.as_deref(), Some("synthetic-textmas-key"));
            if context.tools.is_none() {
                assert!(options.inference_state.is_none());
                assert!(options.source_capture.is_none());
                assert!(options.max_tokens.unwrap() <= 2048.0);
                assert_eq!(options.timeout_ms, Some(120000.0));
            }
            if let Some(file) = &file {
                let saved = std::fs::read_to_string(file).expect("input is durable before advisory spend");
                let entries: Vec<serde_json::Value> = saved.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
                assert!(entries.iter().any(|entry| entry["message"]["role"] == "user" && entry["message"]["content"].to_string().contains("CURRENT_REQUEST")));
                assert!(entries.iter().any(|entry| entry["customType"] == "occ31_textmas_usage" && entry["data"]["status"] == "started"));
            }
            captured.lock().unwrap().push(context.clone());
            let gate = gate.clone();
            let signal = options.signal.clone();
            let message = message.clone();
            Box::pin(async move {
                if let Some(gate) = gate {
                    tokio::select! {
                        permit = gate.acquire() => { permit.unwrap().forget(); },
                        _ = async { if let Some(signal) = signal { signal.cancelled().await } else { std::future::pending::<()>().await } } => {},
                    }
                }
                message
            })
        }))]);
    }

    fn drafts(&self, file: &str) {
        for role in ["PLANNER_DRAFT", "CRITIC_DRAFT", "REFINER_DRAFT"] {
            self.reply(faux_assistant_message(FauxAssistantContent::Blocks(vec![ContentBlock::Thinking(faux_thinking("PRIVATE_ROLE_THOUGHT")), ContentBlock::Text(faux_text(role))]), None), Some(file.into()), None);
        }
    }

    fn dynamic_preparation_replies(&self, count: usize) {
        self.dynamic_preparation_replies_with_parent_failures(count, 0);
    }

    fn dynamic_preparation_replies_with_parent_failures(&self, count: usize, parent_failures: usize) {
        self.dynamic_preparation_replies_impl(count, parent_failures, false);
    }

    fn dynamic_dense_preparation_replies(&self, count: usize) {
        self.dynamic_preparation_replies_impl(count, 0, true);
    }

    fn dynamic_preparation_replies_impl(&self, count: usize, parent_failures: usize, dense: bool) {
        let captured = self.captured.clone();
        let parent_attempt = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let steps = (0..count).map(|_| {
            let captured = captured.clone();
            let parent_attempt = parent_attempt.clone();
            FauxResponseStep::Factory(Arc::new(move |context, options, _, model| {
                assert_eq!(model.provider, "textmas-custom-nonlatent");
                assert_eq!(model.api, "offline-textmas-api");
                let options = options.unwrap();
                assert_eq!(options.api_key.as_deref(), Some("synthetic-textmas-key"));
                captured.lock().unwrap().push(context.clone());
                let prompt = context.system_prompt.as_deref().unwrap_or_default();
                if context.tools.is_none() {
                    assert!(serde_json::to_vec(&context).unwrap().len() <= MAX_CONTEXT_BYTES);
                }
                let message = if context.tools.is_some() {
                    let attempt = parent_attempt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if attempt < parent_failures {
                        faux_assistant_message("".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("error".into()), error_message: Some("synthetic parent retry".into()), ..Default::default() }))
                    } else {
                        text_reply("PRACTICAL_FINAL_ANSWER")
                    }
                } else {
                    assert!(options.inference_state.is_none());
                    assert!(options.source_capture.is_none());
                    assert!(options.max_tokens.unwrap() <= 2048.0);
                    assert_eq!(options.timeout_ms, Some(120000.0));
                    let text = if prompt.contains("advisory role: mapper") {
                        if dense { "\\".repeat(pi_coding_agent::core::occ::textmas::MAX_DRAFT_BYTES) } else { "MAPPER_BRIEF".to_string() }
                    } else if prompt.contains("advisory role: reducer") {
                        if dense { "\\".repeat(pi_coding_agent::core::occ::textmas::MAX_DRAFT_BYTES) } else { "REDUCED_BRIEF".to_string() }
                    } else if prompt.contains("advisory role: planner") { "PLANNER_DRAFT".to_string() }
                        else if prompt.contains("advisory role: critic") { "CRITIC_DRAFT".to_string() }
                        else if prompt.contains("advisory role: refiner") { "REFINER_DRAFT".to_string() }
                        else { panic!("unexpected tool-free OCC3.1 role") };
                    text_reply(&text)
                };
                Box::pin(async move { message })
            }))
        }).collect();
        self.provider.append_responses(steps);
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.provider.unregister();
        match self.old_profile.take() {
            Some(value) => std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", value),
            None => std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR"),
        }
    }
}

#[derive(Clone, Default)]
struct RoleProbe {
    active: Arc<std::sync::atomic::AtomicUsize>,
    peak: Arc<std::sync::atomic::AtomicUsize>,
    entered: Arc<Mutex<Vec<String>>>,
    exited: Arc<Mutex<Vec<String>>>,
    parent_calls: Arc<std::sync::atomic::AtomicUsize>,
    capacity_refusals: Arc<std::sync::atomic::AtomicUsize>,
}

impl RoleProbe {
    async fn wait_for_roles(&self, count: usize) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while self.entered.lock().unwrap().len() < count { tokio::task::yield_now().await; }
        }).await.expect("all independent roles entered");
    }
    async fn wait_for_exit(&self, role: &str) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !self.exited.lock().unwrap().iter().any(|entry| entry == role) { tokio::task::yield_now().await; }
        }).await.expect("released role did not finish");
    }
}

struct ActiveRole(RoleProbe, &'static str);
impl Drop for ActiveRole {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        self.0.exited.lock().unwrap().push(self.1.into());
    }
}

#[derive(Clone)]
struct ControlledRole {
    message: AssistantMessage,
    gate: Option<Arc<tokio::sync::Semaphore>>,
    admission_gate: Option<Arc<tokio::sync::Semaphore>>,
    reject_if_busy: bool,
}
impl Default for ControlledRole {
    fn default() -> Self {
        Self { message: text_reply(""), gate: None, admission_gate: None, reject_if_busy: false }
    }
}

fn provider_error(text: &str) -> AssistantMessage {
    faux_assistant_message("".into(), Some(FauxAssistantMessageOptions {
        stop_reason: Some("error".into()), error_message: Some(text.into()), ..Default::default()
    }))
}

async fn controlled_wait(gate: &Arc<tokio::sync::Semaphore>, signal: Option<&tokio_util::sync::CancellationToken>) -> bool {
    tokio::select! {
        permit = gate.acquire() => { permit.unwrap().forget(); true },
        _ = async { if let Some(signal) = signal { signal.cancelled().await } else { std::future::pending::<()>().await } } => false,
    }
}

fn controlled_roles(f: &Fixture, roles: [ControlledRole; 3], probe: RoleProbe) {
    let roles = Arc::new(roles);
    let steps = (0..4).map(|_| {
        let roles = roles.clone();
        let probe = probe.clone();
        let captured = f.captured.clone();
        FauxResponseStep::Factory(Arc::new(move |context, options, _, model| {
            assert_eq!(model.provider, "textmas-custom-nonlatent");
            assert_eq!(model.api, "offline-textmas-api");
            let options = options.unwrap();
            assert_eq!(options.api_key.as_deref(), Some("synthetic-textmas-key"));
            captured.lock().unwrap().push(context.clone());
            if context.tools.is_some() {
                probe.parent_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let message = text_reply("PRACTICAL_FINAL_ANSWER");
                return Box::pin(async move { message });
            }
            assert!(options.inference_state.is_none());
            assert!(options.source_capture.is_none());
            assert!(options.max_tokens.unwrap() <= 2048.0);
            assert_eq!(options.timeout_ms, Some(120000.0));
            assert!(serde_json::to_vec(&context).unwrap().len() <= MAX_CONTEXT_BYTES);
            let prompt = context.system_prompt.as_deref().unwrap_or_default();
            let (index, role): (usize, &'static str) = if prompt.contains("advisory role: planner") {
                (0, "planner")
            } else if prompt.contains("advisory role: critic") {
                (1, "critic")
            } else if prompt.contains("advisory role: refiner") {
                (2, "refiner")
            } else { panic!("expected one independent reasoning role") };
            let behavior = roles[index].clone();
            let signal = options.signal.clone();
            // Keep the response factory reusable: only this per-call clone enters the future.
            let call_probe = probe.clone();
            Box::pin(async move {
                if let Some(gate) = behavior.admission_gate.as_ref() {
                    if !controlled_wait(gate, signal.as_ref()).await {
                        return provider_error("synthetic admission cancelled; no active work was published");
                    }
                }
                if behavior.reject_if_busy && call_probe.active.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                    call_probe.entered.lock().unwrap().push(role.into());
                    call_probe.capacity_refusals.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return provider_error("synthetic provider capacity limit");
                }
                let active = call_probe.active.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                call_probe.peak.fetch_max(active, std::sync::atomic::Ordering::SeqCst);
                // Waiters know this active call is visible before they release a gated peer.
                call_probe.entered.lock().unwrap().push(role.into());
                let _active = ActiveRole(call_probe, role);
                if let Some(gate) = behavior.gate.as_ref() {
                    let _ = controlled_wait(gate, signal.as_ref()).await;
                }
                behavior.message
            })
        }))
    }).collect();
    f.provider.append_responses(steps);
}

fn role_usage(session: &AgentSession) -> Vec<TextMasUsageRecord> {
    session.session_manager.lock().unwrap().get_entries().iter().filter_map(usage_record).collect()
}
fn terminal_status(session: &AgentSession, role: TextMasRole) -> Vec<TextMasCallStatus> {
    role_usage(session).into_iter().filter(|record| record.role == role && record.status != TextMasCallStatus::Started)
        .map(|record| record.status).collect()
}

fn settled_activity(session: &AgentSession) -> TextMasActivitySnapshot {
    let activity = session.textmas_activity_snapshot().expect("settled run has an activity projection");
    assert!(activity.is_valid() && !activity.active);
    activity
}

fn activity_role_status(activity: &TextMasActivitySnapshot, role: TextMasRole) -> TextMasActivityStatus {
    activity.calls.iter().find(|call| call.role == role).expect("one real role call").status
}

async fn idle(session: &Arc<AgentSession>) {

    let _ = tokio::time::timeout(Duration::from_secs(20), session.wait_for_headless_idle()).await.expect("session settled");
}
async fn turn(session: &Arc<AgentSession>, text: &str, options: Option<PromptOptions>) -> Result<(), String> {
    let result = tokio::time::timeout(Duration::from_secs(20), session.prompt(text, options)).await.expect("prompt settled");
    idle(session).await;
    result
}
fn text_reply(text: &str) -> AssistantMessage { faux_assistant_message(text.into(), None) }

const DENSE_IMAGE_BYTES: usize = 150_000;
const DENSE_IMAGE_MIME: &str = "application/x-occ31-image-density";

fn dense_image_block(block: &serde_json::Value) -> bool {
    block.get("type").and_then(serde_json::Value::as_str) == Some("image")
        && block.get("mimeType").and_then(serde_json::Value::as_str) == Some(DENSE_IMAGE_MIME)
        && block
            .get("data")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|data| data.len() == DENSE_IMAGE_BYTES)
}

fn image_density_extension(audit: Arc<Mutex<Vec<usize>>>) -> ExtensionFactory {
    Arc::new(move |api| {
        let audit = audit.clone();
        Box::pin(async move {
            api.on(
                "context",
                Arc::new(move |event, _| {
                    let audit = audit.clone();
                    Box::pin(async move {
                        let ExtensionEvent::Context(payload) = event else {
                            return None;
                        };
                        let mut messages = payload.messages;
                        let dense_blocks = messages
                            .iter()
                            .filter_map(|message| {
                                message.get("content").and_then(serde_json::Value::as_array)
                            })
                            .map(|blocks| {
                                blocks
                                    .iter()
                                    .filter(|block| dense_image_block(block))
                                    .count()
                            })
                            .sum::<usize>();
                        audit.lock().unwrap().push(dense_blocks);
                        let mut retained = false;
                        for message in messages.iter_mut().rev() {
                            let Some(blocks) = message
                                .get_mut("content")
                                .and_then(serde_json::Value::as_array_mut)
                            else {
                                continue;
                            };
                            for block in blocks {
                                if dense_image_block(block) {
                                    if retained {
                                        block["data"] = json!("OCC31_DUPLICATE_IMAGE_DEDUPED");
                                    } else {
                                        retained = true;
                                    }
                                }
                            }
                        }
                        Some(json!({"messages": messages}))
                    })
                }),
            );
            Ok(())
        })
    })
}

fn dense_image_count(context: &Context) -> usize {
    context.messages.iter().map(|message| match message {
        Message::User(user) => match &user.content {
            pi_ai::types::UserContent::Blocks(blocks) => blocks.iter().filter(|block| matches!(
                block,
                pi_ai::types::ImageOrTextContent::Image(image) if image.mime_type == DENSE_IMAGE_MIME && image.data.len() == DENSE_IMAGE_BYTES
            )).count(),
            _ => 0,
        },
        _ => 0,
    }).sum()
}

fn ledger(session: &AgentSession) -> TextMasUsageLedger {
    let mut ledger = TextMasUsageLedger::default();
    for entry in session.session_manager.lock().unwrap().get_entries() {
        if let Some(record) = usage_record(&entry) { ledger.observe(record); }
    }
    ledger
}

#[tokio::test]
async fn sequential_and_hierarchical_are_real_distinct_graphs_and_keep_final_tools_history_images() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(None);
    for mode in [TextMasMode::Sequential, TextMasMode::Hierarchical] {
        let f = Fixture::new();
        let session = f.session(0, false).await;
        f.reply(text_reply("HISTORY_ANSWER"), None, None);
        turn(&session, "HISTORY_REQUEST", None).await.unwrap();
        let before = session.messages();
        turn(&session, "/mode direct", None).await.unwrap();
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, mode).unwrap();
        let side_effect = f.root.path().join("final-tool.txt");
        f.drafts(&file);
        f.reply(faux_assistant_message(ContentBlock::ToolCall(ToolCall::new("final-tool", "bash", json!({"command":format!("printf FINAL_EXECUTED > '{}'",side_effect.display())}).as_object().unwrap().clone())).into(), Some(FauxAssistantMessageOptions { stop_reason: Some("toolUse".into()), ..Default::default() })), None, None);
        f.reply(text_reply("FINAL_ANSWER"), None, None);
        let image: pi_ai::types::ImageContent = serde_json::from_value(json!({"type":"image","data":"AA==","mimeType":"image/png"})).unwrap();
        turn(&session, "CURRENT_REQUEST", Some(PromptOptions { images: Some(vec![image]), source: Some(InputSource::Rpc), ..Default::default() })).await.unwrap();
        let captured = f.captured.lock().unwrap().clone();
        assert_eq!(captured.len(), 6, "one old turn, three roles, normal judger tool turn, normal continuation");
        for (offset, role) in ["planner", "critic", "refiner"].iter().enumerate() {
            let context = &captured[offset + 1];
            let text = serde_json::to_string(context).unwrap();
            assert!(context.tools.is_none());
            assert!(context.system_prompt.as_ref().unwrap().contains(&format!("advisory role: {role}")));
            assert!(text.contains("TEXTMAS_PROJECT_INSTRUCTIONS"));
            assert!(text.contains("HISTORY_REQUEST") && text.contains("HISTORY_ANSWER") && text.contains("CURRENT_REQUEST"));
            assert!(text.contains("AA==") && text.contains("image/png"));
            assert_eq!(text.contains("PLANNER_DRAFT"), mode == TextMasMode::Sequential && offset > 0);
            assert_eq!(text.contains("CRITIC_DRAFT"), mode == TextMasMode::Sequential && offset > 1);
            assert!(!text.contains("PRIVATE_ROLE_THOUGHT"));
        }
        for context in &captured[4..] {
            let text = serde_json::to_string(&context.messages).unwrap();
            assert!(text.contains("PLANNER_DRAFT") && text.contains("CRITIC_DRAFT") && text.contains("REFINER_DRAFT"));
            assert!(!text.contains("PRIVATE_ROLE_THOUGHT"));
            assert!(!context.system_prompt.as_ref().unwrap().contains("PLANNER_DRAFT"));
            assert!(context.tools.as_ref().unwrap().iter().any(|tool| tool.name == "bash"));
        }
        assert_eq!(std::fs::read_to_string(&side_effect).unwrap(), "FINAL_EXECUTED");
        assert!(session.messages().starts_with(&before));
        assert!(!serde_json::to_string(&session.messages()).unwrap().contains("PLANNER_DRAFT"), "advice is request-owned, not transcript history");
        let usage = ledger(&session);
        let summary = usage.summary().unwrap();
        assert_eq!(summary.calls, 3);
        assert_eq!(summary.unknown_cost_calls, 3, "normalized zero price does not prove free use");
        assert!(summary.known_subtotal.input > 0.0);
        assert_eq!(session.get_session_stats().textmas_usage.as_ref(), Some(&summary));
        let stored = std::fs::read_to_string(&file).unwrap();
        assert!(!stored.contains("PRIVATE_ROLE_THOUGHT") && !stored.contains("PLANNER_DRAFT"));
        // Configuration off removes run-owned context and does not charge another advisory call.
        textmas_settings::set_mode(&file, TextMasMode::Off).unwrap();
        f.reply(text_reply("NORMAL_AFTER"), None, None);
        turn(&session, "next ordinary turn", None).await.unwrap();
        assert!(!serde_json::to_string(f.captured.lock().unwrap().last().unwrap()).unwrap().contains("PLANNER_DRAFT"));
        assert_eq!(ledger(&session).summary().unwrap().calls, 3);
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn hidden_thinking_over_cap_does_not_block_roles_or_the_normal_parent() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let transform_audit = Arc::new(Mutex::new(Vec::new()));
    let session = f.session_with_extensions(0, false, vec![image_density_extension(transform_audit.clone())]).await;
    let dense_image = pi_ai::types::ImageContent::new("A".repeat(DENSE_IMAGE_BYTES), DENSE_IMAGE_MIME);
    let private_history = format!("LARGE_PRIVATE_HISTORY{}", "x".repeat(MAX_CONTEXT_BYTES));
    let private_thinking = pi_ai::types::ThinkingContent {
        thinking_signature: Some("PRIVATE_THINKING_SIGNATURE".into()),
        ..faux_thinking(&private_history)
    };
    f.reply(
        faux_assistant_message(
            FauxAssistantContent::Blocks(vec![
                ContentBlock::Thinking(private_thinking.clone()),
                ContentBlock::Text(faux_text("HISTORY_ANSWER")),
            ]),
            None,
        ),
        None,
        None,
    );
    turn(&session, "HISTORY_REQUEST", Some(PromptOptions {
        images: Some(vec![dense_image.clone()]), source: Some(InputSource::Rpc), ..Default::default()
    })).await.unwrap();
    transform_audit.lock().unwrap().clear();
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.drafts(&file);
    f.reply(text_reply("FINAL_ANSWER"), None, None);
    turn(&session, "CURRENT_REQUEST", Some(PromptOptions {
        images: Some(vec![dense_image]), source: Some(InputSource::Rpc), ..Default::default()
    })).await.unwrap();
    assert_eq!(transform_audit.lock().unwrap().as_slice(), &[2], "one extension transform sees both dense images; role and parent reuse it");

    let contexts = f.captured.lock().unwrap().clone();
    assert_eq!(
        contexts.len(),
        5,
        "ordinary history, three roles, normal parent"
    );
    for (offset, role) in ["planner", "critic", "refiner"].iter().enumerate() {
        let context = &contexts[offset + 1];
        assert!(context.tools.is_none());
        assert!(context
            .system_prompt
            .as_ref()
            .unwrap()
            .contains(&format!("advisory role: {role}")));
        assert!(serde_json::to_vec(context).unwrap().len() <= MAX_CONTEXT_BYTES);
        assert_eq!(dense_image_count(context), 1, "extension-deduplicated image density is bounded for {role}");
        let text = serde_json::to_string(context).unwrap();
        assert!(text.contains("HISTORY_ANSWER") && text.contains("CURRENT_REQUEST"));
        assert!(!text.contains("LARGE_PRIVATE_HISTORY"));
    }
    let parent = &contexts[4];
    assert!(parent.tools.is_some());
    assert!(serde_json::to_vec(parent).unwrap().len() > MAX_CONTEXT_BYTES);
    assert_eq!(dense_image_count(parent), 1);
    let parent_text = serde_json::to_string(parent).unwrap();
    assert!(parent_text.contains("OCC31_DUPLICATE_IMAGE_DEDUPED"));
    let parent_thinking = parent
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant),
            _ => None,
        })
        .flat_map(|assistant| &assistant.content)
        .find_map(|block| match block {
            ContentBlock::Thinking(thinking) if thinking.thinking == private_history => Some(thinking),
            _ => None,
        })
        .expect("normal parent retains the exact hidden Thinking block");
    assert_eq!(parent_thinking, &private_thinking);
    assert!(
        parent_text.contains("LARGE_PRIVATE_HISTORY"),
        "normal parent history is not truncated"
    );
    assert!(
        parent_text.contains("PLANNER_DRAFT")
            && parent_text.contains("CRITIC_DRAFT")
            && parent_text.contains("REFINER_DRAFT")
    );
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn oversized_single_message_uses_bounded_chunked_preparation_in_both_modes() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(None);
    for mode in [TextMasMode::Sequential, TextMasMode::Hierarchical] {
        let f = Fixture::new();
        let session = f.session(0, false).await;
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, mode).unwrap();
        f.dynamic_preparation_replies(64);
        let oversized_request = format!(
            "BEGIN_SENTINEL:{}:MIDDLE_SENTINEL:{}:END_SENTINEL",
            "é雪\"\n≤".repeat(40_000),
            "z".repeat(MAX_CONTEXT_BYTES * 2),
        );
        turn(&session, &oversized_request, None).await.unwrap();
        let contexts = f.captured.lock().unwrap().clone();
        let role_name = |context: &Context| {
            let prompt = context.system_prompt.as_deref().unwrap_or_default();
            ["mapper", "reducer", "planner", "critic", "refiner"].into_iter()
                .find(|role| prompt.contains(&format!("advisory role: {role}")))
        };
        let mapper_count = contexts.iter().filter(|context| role_name(context) == Some("mapper")).count();
        let reducer_count = contexts.iter().filter(|context| role_name(context) == Some("reducer")).count();
        assert!(mapper_count > 1, "large source must use multiple mapper calls ({mode:?})");
        assert!(reducer_count >= 1, "multiple briefs must be reduced ({mode:?})");
        for context in contexts.iter().filter(|context| context.tools.is_none()) {
            assert!(serde_json::to_vec(context).unwrap().len() <= MAX_CONTEXT_BYTES);
            assert!(role_name(context).is_some());
        }
        for role in ["planner", "critic", "refiner"] {
            assert_eq!(contexts.iter().filter(|context| role_name(context) == Some(role)).count(), 1);
        }
        let sequential = contexts.iter().find(|context| role_name(context) == Some("critic")).unwrap();
        let refiner = contexts.iter().find(|context| role_name(context) == Some("refiner")).unwrap();
        assert_eq!(serde_json::to_string(sequential).unwrap().contains("PLANNER_DRAFT"), mode == TextMasMode::Sequential);
        assert_eq!(serde_json::to_string(refiner).unwrap().contains("CRITIC_DRAFT"), mode == TextMasMode::Sequential);
        let parents = contexts.iter().filter(|context| context.tools.is_some()).collect::<Vec<_>>();
        assert_eq!(parents.len(), 1);
        let parent = serde_json::to_string(parents[0]).unwrap();
        assert!(parent.contains("BEGIN_SENTINEL") && parent.contains("MIDDLE_SENTINEL") && parent.contains("END_SENTINEL"));
        assert!(parent.contains("PLANNER_DRAFT") && parent.contains("CRITIC_DRAFT") && parent.contains("REFINER_DRAFT"));
        assert!(parent.contains("shared context-map") && parent.contains("lossy advisory"));
        assert_eq!(ledger(&session).summary().unwrap().calls, mapper_count + reducer_count + 3);
        let activity = session.textmas_activity_snapshot().unwrap();
        assert!(activity.is_valid() && activity.mode == Some(mode));
        assert_eq!(activity.calls.iter().filter(|call| call.role == pi_coding_agent::core::occ::textmas::TextMasRole::Mapper).count(), mapper_count);
        let reducers = activity.calls.iter().filter(|call| call.role == pi_coding_agent::core::occ::textmas::TextMasRole::Reducer).collect::<Vec<_>>();
        assert_eq!(reducers.len(), reducer_count);
        for reducer in &reducers {
            assert!(reducer.level.is_some());
            assert!(!reducer.depends_on.is_empty());
            for dependency in &reducer.depends_on {
                assert!(activity.calls.iter().any(|call| &call.call_id == dependency));
            }
        }
        assert!(reducers.iter().any(|reducer| pi_coding_agent::core::occ::textmas::TextMasRole::ALL.iter().all(|role| {
            activity.calls.iter().find(|call| call.role == *role).unwrap().depends_on.contains(&reducer.call_id)
        })), "all real roles consume the actual final reduction brief");
        assert!(!serde_json::to_string(&session.messages()).unwrap().contains("MAPPER_BRIEF"));
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn large_history_reduction_and_parent_retry_keep_exact_latest_and_do_not_respend() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let transform_audit = Arc::new(Mutex::new(Vec::new()));
    let session = f
        .session_with_extensions(0, true, vec![image_density_extension(transform_audit.clone())])
        .await;
    f.reply(text_reply("HISTORY_ANSWER"), None, None);
    let large_history = format!("HISTORY_BEGIN{}HISTORY_END", "h".repeat(MAX_CONTEXT_BYTES * 2));
    turn(&session, &large_history, None).await.unwrap();
    transform_audit.lock().unwrap().clear();
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.dynamic_preparation_replies_with_parent_failures(64, 1);
    turn(&session, "CURRENT_REQUEST", None).await.unwrap();
    let contexts = f.captured.lock().unwrap().clone();
    let prepared = &contexts[1..];
    let role_name = |context: &Context| {
        let prompt = context.system_prompt.as_deref().unwrap_or_default();
        ["mapper", "reducer", "planner", "critic", "refiner"].into_iter()
            .find(|role| prompt.contains(&format!("advisory role: {role}")))
    };
    let mapper_count = prepared.iter().filter(|context| role_name(context) == Some("mapper")).count();
    let reducer_count = prepared.iter().filter(|context| role_name(context) == Some("reducer")).count();
    assert!(mapper_count > 1 && reducer_count >= 1);
    for role in ["planner", "critic", "refiner"] {
        let context = prepared.iter().find(|context| role_name(context) == Some(role)).unwrap();
        let encoded = serde_json::to_string(context).unwrap();
        assert!(encoded.contains("CURRENT_REQUEST"), "small latest request stays exact in the reduced role");
        assert!(!encoded.contains("HISTORY_BEGIN"), "roles consume the brief, not the oversized raw history");
    }
    let parents = prepared.iter().filter(|context| context.tools.is_some()).collect::<Vec<_>>();
    assert_eq!(parents.len(), 2, "one failed normal parent attempt and its ordinary retry");
    for parent in parents {
        let encoded = serde_json::to_string(parent).unwrap();
        assert!(encoded.contains("HISTORY_BEGIN") && encoded.contains("HISTORY_END"));
        assert!(encoded.contains("CURRENT_REQUEST") && encoded.contains("REFINER_DRAFT"));
        assert!(serde_json::to_vec(parent).unwrap().len() > MAX_CONTEXT_BYTES);
    }
    assert_eq!(ledger(&session).summary().unwrap().calls, mapper_count + reducer_count + 3);
    assert_eq!(*transform_audit.lock().unwrap(), vec![0], "one stateful transform is reused by both parent attempts");
    session.dispose_async(Some(false)).await;
}


#[tokio::test]
async fn large_project_instructions_trigger_bounded_source_preparation() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let instructions = format!(
        "PROJECT_INSTRUCTIONS_BEGIN\n{}\nPROJECT_INSTRUCTIONS_END",
        "Preserve exact UTF-8 é雪 and JSON {\"key\": true} in this instruction.\n"
            .repeat(4_500)
    );
    assert!(instructions.len() > MAX_CONTEXT_BYTES);
    std::fs::write(
        f.root.path().join("workspace/AGENTS.md"),
        &instructions,
    )
    .unwrap();
    let session = f.session(0, false).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.dynamic_preparation_replies(64);
    turn(&session, "CURRENT_REQUEST", None).await.unwrap();

    let contexts = f.captured.lock().unwrap().clone();
    let role_name = |context: &Context| {
        let prompt = context.system_prompt.as_deref().unwrap_or_default();
        ["mapper", "reducer", "planner", "critic", "refiner"]
            .into_iter()
            .find(|role| prompt.contains(&format!("advisory role: {role}")))
    };
    let parent = contexts
        .iter()
        .find(|context| context.tools.is_some())
        .expect("the ordinary parent still runs after preparation");
    let original_system = parent.system_prompt.as_deref().unwrap();
    assert!(original_system.contains(&instructions));
    assert!(original_system.len() > MAX_CONTEXT_BYTES);

    // All original system-instruction bytes are losslessly available to the
    // source mapper. They are neither truncated nor copied back into an
    // oversized auxiliary system prompt.
    let mut recovered_system = String::new();
    for context in contexts
        .iter()
        .filter(|context| role_name(context) == Some("mapper"))
    {
        for message in &context.messages {
            let Message::User(user) = message else {
                continue;
            };
            let pi_ai::types::UserContent::Text(text) = &user.content else {
                continue;
            };
            if text.starts_with("[OCC3.1 source system instructions") {
                let (_, payload) = text.split_once("]\n").unwrap();
                recovered_system.push_str(payload);
            }
        }
    }
    assert_eq!(recovered_system, original_system);
    for context in contexts.iter().filter(|context| context.tools.is_none()) {
        assert!(serde_json::to_vec(context).unwrap().len() <= MAX_CONTEXT_BYTES);
    }
    for role in ["planner", "critic", "refiner"] {
        let context = contexts
            .iter()
            .find(|context| role_name(context) == Some(role))
            .unwrap();
        let encoded = serde_json::to_string(context).unwrap();
        assert!(encoded.contains("CURRENT_REQUEST"));
        assert!(!encoded.contains("PROJECT_INSTRUCTIONS_BEGIN"));
    }
    let mapper_count = contexts
        .iter()
        .filter(|context| role_name(context) == Some("mapper"))
        .count();
    let reducer_count = contexts
        .iter()
        .filter(|context| role_name(context) == Some("reducer"))
        .count();
    assert!(mapper_count > 1 && reducer_count >= 1);
    let parent_text = serde_json::to_string(parent).unwrap();
    assert!(parent_text.contains("PROJECT_INSTRUCTIONS_BEGIN"));
    assert!(parent_text.contains("PROJECT_INSTRUCTIONS_END"));
    assert!(parent_text.contains("complete system instructions were mapped"));
    assert!(parent_text.contains("REFINER_DRAFT"));
    assert!(serde_json::to_vec(parent).unwrap().len() > MAX_CONTEXT_BYTES);
    assert_eq!(
        ledger(&session).summary().unwrap().calls,
        mapper_count + reducer_count + 3
    );
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn dense_auxiliary_outputs_force_real_recursive_reduction_and_unique_ordinals() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, true).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Sequential).unwrap();
    f.dynamic_dense_preparation_replies(96);
    let source = format!("DENSE_SOURCE_BEGIN{}DENSE_SOURCE_END", "s".repeat(MAX_CONTEXT_BYTES * 20));
    tokio::time::timeout(Duration::from_secs(120), session.prompt(&source, None))
        .await
        .expect("dense stress prompt settled within its explicit bound")
        .unwrap();
    idle(&session).await;
    let contexts = f.captured.lock().unwrap().clone();
    let role_name = |context: &Context| {
        let prompt = context.system_prompt.as_deref().unwrap_or_default();
        ["mapper", "reducer", "planner", "critic", "refiner"].into_iter()
            .find(|role| prompt.contains(&format!("advisory role: {role}")))
    };
    let mapper_count = contexts.iter().filter(|context| role_name(context) == Some("mapper")).count();
    let reducer_count = contexts.iter().filter(|context| role_name(context) == Some("reducer")).count();
    assert!(mapper_count > 16 && reducer_count > 3, "dense escaped outputs require multiple actual reduction rounds: {mapper_count} maps, {reducer_count} reductions");
    let usage = ledger(&session);
    let ids = usage.records().map(|record| record.call_id()).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), mapper_count + reducer_count + 3);
    let mut mapper_ordinals = usage.records().filter(|record| record.role == pi_coding_agent::core::occ::textmas::TextMasRole::Mapper)
        .map(|record| record.call_index.expect("mapper ordinal")).collect::<Vec<_>>();
    mapper_ordinals.sort_unstable();
    assert_eq!(mapper_ordinals, (0..u32::try_from(mapper_count).unwrap()).collect::<Vec<_>>());
    let mut reducer_ordinals = usage.records().filter(|record| record.role == pi_coding_agent::core::occ::textmas::TextMasRole::Reducer)
        .map(|record| record.call_index.expect("reducer ordinal")).collect::<Vec<_>>();
    reducer_ordinals.sort_unstable();
    assert_eq!(reducer_ordinals, (0..u32::try_from(reducer_count).unwrap()).collect::<Vec<_>>());
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn valid_oversized_png_is_resized_only_for_advisory_preparation() {
    use base64::Engine as _;
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, true).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Sequential).unwrap();
    let pixels = image::RgbaImage::from_fn(512, 512, |x, y| {
        let value = x.wrapping_mul(1_103_515_245).wrapping_add(y.wrapping_mul(12_345));
        image::Rgba([value as u8, (value >> 8) as u8, (value >> 16) as u8, 255])
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(pixels).write_to(&mut encoded, image::ImageFormat::Png).unwrap();
    let original_data = base64::engine::general_purpose::STANDARD.encode(encoded.into_inner());
    assert!(original_data.len() > MAX_CONTEXT_BYTES, "fixture must actually exceed the role bound");
    let original = pi_ai::types::ImageContent::new(original_data.clone(), "image/png");
    f.dynamic_preparation_replies_with_parent_failures(64, 0);
    turn(&session, "VALID_IMAGE_RESIZE", Some(PromptOptions {
        images: Some(vec![original]), ..Default::default()
    })).await.unwrap();
    let contexts = f.captured.lock().unwrap().clone();
    let mapper = contexts.iter().find(|context| {
        context.system_prompt.as_deref().unwrap_or_default().contains("advisory role: mapper")
    }).expect("valid image reaches mapper preparation");
    let mapper_json = serde_json::to_string(mapper).unwrap();
    assert!(!mapper_json.contains(&original_data), "mapper must not receive the oversized original image");
    let parent = contexts.iter().find(|context| context.tools.is_some()).expect("ordinary parent");
    let parent_json = serde_json::to_string(parent).unwrap();
    assert!(parent_json.contains(&original_data), "ordinary parent image bytes are not resized or replaced");
    assert!(parent_json.contains("resized 1 preparation-only image(s)"), "parent advice records advisory-only resizing");
    session.dispose_async(Some(false)).await;
}


#[tokio::test]
async fn activity_events_and_snapshot_track_running_then_authoritative_completion() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    use pi_coding_agent::core::agent_session::AgentSessionEvent;
    use pi_coding_agent::core::occ::textmas_activity::{TextMasActivitySnapshot, TextMasActivityStatus};
    let f = Fixture::new();
    let session = f.session(0, false).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    let events = Arc::new(Mutex::new(Vec::<TextMasActivitySnapshot>::new()));
    let unsubscribe = session.subscribe(Arc::new({ let events = events.clone(); move |event| {
        if let AgentSessionEvent::TextMasActivity { activity } = event { events.lock().unwrap().push(activity); }
    }}));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    f.reply(text_reply("PLANNER_DRAFT"), Some(file), Some(release.clone()));
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _running_cleanup = AbortOnDrop(running.abort_handle());
    tokio::time::timeout(Duration::from_secs(10), async {
        while f.captured.lock().unwrap().is_empty() { tokio::task::yield_now().await; }
    }).await.unwrap();
    let running_activity = session.textmas_activity_snapshot().unwrap();
    assert_eq!(running_activity.session_id, session.session_id());
    assert!(running_activity.active && !running_activity.stale);
    assert_eq!(running_activity.calls[0].status, TextMasActivityStatus::Running);
    assert!(running_activity.calls[1..].iter().all(|call| call.status == TextMasActivityStatus::Queued));
    assert_eq!(f.captured.lock().unwrap().len(), 1, "planning and snapshots do not spend tokens");
    release.add_permits(1);
    f.reply(text_reply("CRITIC_DRAFT"), None, None);
    f.reply(text_reply("REFINER_DRAFT"), None, None);
    f.reply(text_reply("PARENT"), None, None);
    tokio::time::timeout(Duration::from_secs(20), running).await.unwrap().unwrap().unwrap();
    idle(&session).await;
    let final_activity = session.textmas_activity_snapshot().unwrap();
    assert!(!final_activity.active);
    assert!(final_activity.calls.iter().all(|call| call.status == TextMasActivityStatus::Completed));
    assert_eq!(f.captured.lock().unwrap().len(), 4, "three role calls plus exactly one parent");
    let events = events.lock().unwrap();
    assert!(events.iter().any(|activity| activity.active && activity.calls.iter().any(|call| call.status == TextMasActivityStatus::Running)));
    assert!(events.iter().all(|activity| activity.session_id == session.session_id() && activity.is_valid()));
    assert_eq!(events.last().unwrap().calls, final_activity.calls);
    drop(events);
    unsubscribe();
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn off_internal_extension_and_child_turns_never_fan_out() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(None);
    let f = Fixture::new();
    let session = f.session(0, false).await;
    f.reply(text_reply("ORDINARY"), None, None);
    turn(&session, "off by default", None).await.unwrap();
    assert_eq!(f.captured.lock().unwrap().len(), 1);
    assert!(session.get_session_stats().textmas_usage.is_none());
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Sequential).unwrap();
    for options in [PromptOptions { internal_prompt: Some(true), ..Default::default() }, PromptOptions { source: Some(InputSource::Extension), ..Default::default() }] {
        f.reply(text_reply("INTERNAL"), None, None);
        turn(&session, "not a user turn", Some(options)).await.unwrap();
    }
    assert_eq!(f.captured.lock().unwrap().len(), 3);
    assert!(session.get_session_stats().textmas_usage.is_none());
    session.dispose_async(Some(false)).await;
    let child = f.session(1, false).await;
    textmas_settings::set_mode(&child.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    f.reply(text_reply("CHILD"), None, None);
    turn(&child, "child request", None).await.unwrap();
    assert_eq!(f.captured.lock().unwrap().len(), 4);
    assert!(child.get_session_stats().textmas_usage.is_none());
    child.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn provider_error_refusal_empty_and_tool_attempt_fail_without_retry_or_execution() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    for case in ["error", "length", "refusal", "empty", "tool", "oversized"] {
        let f = Fixture::new();
        let session = f.session(0, true).await;
        turn(&session, "/mode direct", None).await.unwrap();
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
        let side_effect = f.root.path().join("must-not-exist");
        let message = match case {
            "error" => faux_assistant_message("".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("error".into()), error_message: Some("temporary provider failure".into()), ..Default::default() })),
            "length" => faux_assistant_message("partial unpublished draft".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("length".into()), ..Default::default() })),
            "refusal" => { let mut m = text_reply("declined"); m.stop_reason_raw = Some("refusal".into()); m },
            "empty" => faux_assistant_message(FauxAssistantContent::Blocks(vec![ContentBlock::Thinking(faux_thinking("PRIVATE_ONLY"))]), None),
            "tool" => faux_assistant_message(ContentBlock::ToolCall(ToolCall::new("unsafe", "bash", json!({"command":format!("touch '{}'",side_effect.display())}).as_object().unwrap().clone())).into(), Some(FauxAssistantMessageOptions { stop_reason: Some("toolUse".into()), ..Default::default() })),
            _ => text_reply(&"x".repeat(16_385)),
        };
        f.reply(message, Some(file), None);
        let error = turn(&session, "CURRENT_REQUEST", None).await.unwrap_err();
        if case == "length" {
            assert!(error.contains("stop: length"), "{error}");
            assert!(!error.contains("partial unpublished draft"));
        }
        assert_eq!(f.captured.lock().unwrap().len(), 1, "no retry/fallback after {case}");
        let activity = session.textmas_activity_snapshot().unwrap();
        assert!(activity.is_valid() && !activity.active);
        assert_eq!(activity.calls[0].status, pi_coding_agent::core::occ::textmas_activity::TextMasActivityStatus::Failed);
        assert!(activity.calls[1..].iter().all(|call| call.status == pi_coding_agent::core::occ::textmas_activity::TextMasActivityStatus::Skipped));
        assert!(!side_effect.exists());
        assert_eq!(ledger(&session).summary().unwrap().calls, 1);
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn hierarchical_overlap_preserves_order_usage_and_sequential_dependencies() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("3"));
    for mode in [TextMasMode::Hierarchical, TextMasMode::Sequential] {
        let f = Fixture::new();
        let session = f.session(0, false).await;
        textmas_settings::set_mode(&session.session_file().unwrap(), mode).unwrap();
        let gates = [
            Arc::new(tokio::sync::Semaphore::new(0)),
            Arc::new(tokio::sync::Semaphore::new(0)),
            Arc::new(tokio::sync::Semaphore::new(0)),
        ];
        let texts = ["PLANNER_DRAFT", "CRITIC_DRAFT", "REFINER_DRAFT"];
        let roles = gates.iter().zip(texts).map(|(gate, text)| ControlledRole {
            message: text_reply(text),
            gate: if mode == TextMasMode::Hierarchical { Some(gate.clone()) } else { None },
            ..Default::default()
        }).collect::<Vec<_>>();
        let probe = RoleProbe::default();
        controlled_roles(&f, roles.try_into().ok().unwrap(), probe.clone());
        if mode == TextMasMode::Hierarchical {
            let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
            let _cleanup = AbortOnDrop(running.abort_handle());
            probe.wait_for_roles(3).await;
            // Synthetic inverse completion: no ordering result depends on timing granularity.
            gates[1].add_permits(1); probe.wait_for_exit("critic").await;
            gates[2].add_permits(1); probe.wait_for_exit("refiner").await;
            gates[0].add_permits(1);
            tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().unwrap();
        } else {
            turn(&session, "CURRENT_REQUEST", None).await.unwrap();
        }
        idle(&session).await;
        assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), if mode == TextMasMode::Hierarchical { 3 } else { 1 });
        assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let contexts = f.captured.lock().unwrap().clone();
        assert_eq!(contexts.len(), 4);
        for name in ["planner", "critic", "refiner"] {
            let label = format!("advisory role: {name}");
            let matched = contexts.iter().filter(|context| context.system_prompt.as_deref().is_some_and(|prompt| prompt.contains(&label))).collect::<Vec<_>>();
            assert_eq!(matched.len(), 1, "role {name} must have exactly one correlated context");
            let text = serde_json::to_string(&matched[0]).unwrap();
            assert!(text.contains("CURRENT_REQUEST") && text.contains("TEXTMAS_PROJECT_INSTRUCTIONS"));
            if name == "critic" {
                assert_eq!(text.contains("PLANNER_DRAFT"), mode == TextMasMode::Sequential);
            } else if name == "refiner" {
                assert_eq!(text.contains("PLANNER_DRAFT"), mode == TextMasMode::Sequential);
                assert_eq!(text.contains("CRITIC_DRAFT"), mode == TextMasMode::Sequential);
            }
        }
        let parent = serde_json::to_string(&contexts[3].messages).unwrap();
        let planner_pos = parent.find("PLANNER_DRAFT").unwrap();
        let critic_pos = parent.find("CRITIC_DRAFT").unwrap();
        let refiner_pos = parent.find("REFINER_DRAFT").unwrap();
        assert!(planner_pos < critic_pos && critic_pos < refiner_pos);
        for role in TextMasRole::ALL {
            assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Completed]);
        }
        if mode == TextMasMode::Hierarchical {
            assert_eq!(probe.exited.lock().unwrap().as_slice(), ["critic", "refiner", "planner"]);
            let records = role_usage(&session);
            let terminal = records.iter().filter(|record| record.status != TextMasCallStatus::Started).collect::<Vec<_>>();
            assert_eq!(terminal.len(), 3);
            assert!(terminal.iter().all(|record| record.usage_known));
            // Ordinary faux responses estimate token counters; a zero price is still unknown cost.
            assert!(terminal.iter().all(|record| !record.cost_known));
            let expected_tokens = terminal.iter().map(|record| {
                let usage = record.usage.as_ref().unwrap();
                if usage.total_tokens > 0.0 { usage.total_tokens }
                else { usage.input + usage.output + usage.cache_read + usage.cache_write }
            }).sum::<f64>();
            let expected_input = terminal.iter().map(|record| record.usage.as_ref().unwrap().input).sum::<f64>();
            let expected_output = terminal.iter().map(|record| record.usage.as_ref().unwrap().output).sum::<f64>();
            let summary = ledger(&session).summary().unwrap();
            assert_eq!(summary.calls, 3);
            assert_eq!(summary.unknown_usage_calls, 0);
            assert_eq!(summary.unknown_cost_calls, 3);
            assert_eq!(summary.known_tokens, expected_tokens);
            assert_eq!(summary.known_subtotal.input, expected_input);
            assert_eq!(summary.known_subtotal.output, expected_output);
        }
        session.dispose_async(Some(false)).await;
    }

    // Controlled synthetic simultaneous completion: release all three in-flight roles together.
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let role = |text| ControlledRole { message: text_reply(text), gate: Some(gate.clone()), ..Default::default() };
    let probe = RoleProbe::default();
    controlled_roles(&f, [role("PLANNER_DRAFT"), role("CRITIC_DRAFT"), role("REFINER_DRAFT")], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(3).await;
    gate.add_permits(3);
    tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().unwrap();
    idle(&session).await;
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    for role in TextMasRole::ALL {
        assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Completed]);
    }
    let parent = serde_json::to_string(&f.captured.lock().unwrap().last().unwrap().messages).unwrap();
    let planner_pos = parent.find("PLANNER_DRAFT").unwrap();
    let critic_pos = parent.find("CRITIC_DRAFT").unwrap();
    let refiner_pos = parent.find("REFINER_DRAFT").unwrap();
    assert!(planner_pos < critic_pos && critic_pos < refiner_pos);
    session.dispose_async(Some(false)).await;
}


#[tokio::test]
async fn hierarchical_each_role_failure_promptly_cancels_inflight_siblings() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("3"));
    for (failed_index, failed_role) in TextMasRole::ALL.into_iter().enumerate() {
        let f = Fixture::new();
        let session = f.session(0, false).await;
        textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
        let slow = Arc::new(tokio::sync::Semaphore::new(0));
        let fail = Arc::new(tokio::sync::Semaphore::new(0));
        let roles = TextMasRole::ALL.into_iter().enumerate().map(|(index, role)| ControlledRole {
            message: if index == failed_index { provider_error(&format!("selected {} failure", role.as_str())) } else { text_reply(role.as_str()) },
                        gate: Some(if index == failed_index { fail.clone() } else { slow.clone() }),
            reject_if_busy: false, ..Default::default()
        }).collect::<Vec<_>>();
        let probe = RoleProbe::default();
        controlled_roles(&f, roles.try_into().ok().unwrap(), probe.clone());
        let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
        let _cleanup = AbortOnDrop(running.abort_handle());
        probe.wait_for_roles(3).await;
        fail.add_permits(1);
        let result = tokio::time::timeout(Duration::from_secs(5), running).await.unwrap().unwrap();
        assert!(result.unwrap_err().contains(&format!("OCC3.1 {}", failed_role.as_str())));
        assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(f.captured.lock().unwrap().len(), 3);
        for role in TextMasRole::ALL {
            let expected = if role == failed_role { TextMasCallStatus::Failed } else { TextMasCallStatus::Cancelled };
            assert_eq!(terminal_status(&session, role), vec![expected], "failed role {failed_role:?}; sibling {role:?}");
        }
        let activity = settled_activity(&session);
        assert_eq!(activity.calls.len(), 3, "no synthetic roles come from sibling cancellation");
        for role in TextMasRole::ALL {
            let expected = if role == failed_role { TextMasActivityStatus::Failed } else { TextMasActivityStatus::Cancelled };
            assert_eq!(activity_role_status(&activity, role), expected);
        }
        let records = role_usage(&session);
        assert_eq!(records.iter().filter(|record| record.status == TextMasCallStatus::Started).count(), 3);
        assert_eq!(records.iter().filter(|record| record.call_index.is_some()).count(), 0);
        assert_eq!(records.iter().map(|record| record.action_id.clone()).collect::<std::collections::HashSet<_>>().len(), 1);
        let terminal = records.iter().filter(|record| record.status != TextMasCallStatus::Started).collect::<Vec<_>>();
        let cancelled = terminal.iter().filter(|record| record.status == TextMasCallStatus::Cancelled).collect::<Vec<_>>();
        assert_eq!(cancelled.len(), 2);
        assert!(cancelled.iter().all(|record| !record.usage_known && !record.cost_known));
        let failed = terminal.iter().find(|record| record.status == TextMasCallStatus::Failed).unwrap();
        assert!(failed.usage_known, "the deterministic provider error retains estimated tokens");
        let summary = ledger(&session).summary().unwrap();
        assert_eq!(summary.calls, 3);
        assert_eq!(summary.unknown_usage_calls, 2, "failed role has usage; both cancelled siblings remain unknown");
        assert_eq!(summary.unknown_cost_calls, 3);
        slow.add_permits(2);
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn hierarchical_parent_abort_cancels_every_role_without_parent_execution() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("3"));
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let role = |text| ControlledRole { message: text_reply(text), gate: Some(gate.clone()), reject_if_busy: false, ..Default::default() };
    let probe = RoleProbe::default();
    controlled_roles(&f, [role("STALE_PLANNER"), role("STALE_CRITIC"), role("STALE_REFINER")], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(3).await;
    session.abort().await.unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().is_err());
    idle(&session).await;
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(f.captured.lock().unwrap().len(), 3);
    for role in TextMasRole::ALL {
        assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Cancelled]);
    }
    assert!(!serde_json::to_string(&session.messages()).unwrap().contains("STALE_"));
    gate.add_permits(3);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn hierarchical_default_preserves_provider_capacity_one() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(None);
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let planner_gate = Arc::new(tokio::sync::Semaphore::new(0));
    let probe = RoleProbe::default();
    // No provider capability is advertised. The synthetic provider refuses the
    // critic if the previous role is still active. Default hierarchical execution
    // must finish all three ordered roles without admitting that busy request.
    controlled_roles(&f, [
        ControlledRole { message: text_reply("PLANNER_DRAFT"), gate: Some(planner_gate.clone()), ..Default::default() },
        ControlledRole { message: text_reply("CRITIC_DRAFT"), reject_if_busy: true, ..Default::default() },
        ControlledRole { message: text_reply("REFINER_DRAFT"), ..Default::default() },
    ], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(1).await;
    planner_gate.add_permits(1);
    probe.wait_for_exit("planner").await;
    let result = tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap();
    assert!(result.is_ok(), "capacity-one hierarchical preparation must complete, not fail: {result:?}");
    idle(&session).await;
    assert_eq!(probe.capacity_refusals.load(std::sync::atomic::Ordering::SeqCst), 0, "no provider-aware capability was advertised, so serial scheduling must avoid a busy refusal");
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(probe.entered.lock().unwrap().as_slice(), ["planner", "critic", "refiner"]);
    for role in TextMasRole::ALL {
        assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Completed]);
    }
    let parent = serde_json::to_string(&f.captured.lock().unwrap().last().unwrap().messages).unwrap();
    assert!(parent.find("PLANNER_DRAFT").unwrap() < parent.find("CRITIC_DRAFT").unwrap());
    assert!(parent.find("CRITIC_DRAFT").unwrap() < parent.find("REFINER_DRAFT").unwrap());
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn hierarchical_configured_two_is_bounded_ordered_and_work_conserving() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("2"));
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let gates = [Arc::new(tokio::sync::Semaphore::new(0)), Arc::new(tokio::sync::Semaphore::new(0)), Arc::new(tokio::sync::Semaphore::new(0))];
    let probe = RoleProbe::default();
    controlled_roles(&f, [
        ControlledRole { message: text_reply("PLANNER_DRAFT"), gate: Some(gates[0].clone()), ..Default::default() },
        ControlledRole { message: text_reply("CRITIC_DRAFT"), gate: Some(gates[1].clone()), ..Default::default() },
        ControlledRole { message: text_reply("REFINER_DRAFT"), gate: Some(gates[2].clone()), ..Default::default() },
    ], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(2).await;
    assert_eq!(probe.entered.lock().unwrap().as_slice(), ["planner", "critic"]);
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    // Complete the critic while the planner is blocked. The third role must take
    // the freed capacity-two slot without waiting for a two-item chunk to finish.
    gates[1].add_permits(1);
    probe.wait_for_exit("critic").await;
    probe.wait_for_roles(3).await;
    assert_eq!(probe.entered.lock().unwrap().as_slice(), ["planner", "critic", "refiner"]);
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    gates[0].add_permits(1);
    gates[2].add_permits(1);
    tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().unwrap();
    idle(&session).await;
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    let parent = serde_json::to_string(&f.captured.lock().unwrap().last().unwrap().messages).unwrap();
    assert!(parent.find("PLANNER_DRAFT").unwrap() < parent.find("CRITIC_DRAFT").unwrap());
    assert!(parent.find("CRITIC_DRAFT").unwrap() < parent.find("REFINER_DRAFT").unwrap());
    for role in TextMasRole::ALL {
        assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Completed]);
    }
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn hierarchical_explicit_one_and_invalid_settings_stay_serial() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    for value in [Some("1"), Some("0"), Some("4"), Some("not-a-number")] {
        let _concurrency = ScopedHierarchicalConcurrencyEnv::set(value);
        let expected_label = format!("configuration {value:?}");
        let f = Fixture::new();
        let session = f.session(0, false).await;
        textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
        let planner_gate = Arc::new(tokio::sync::Semaphore::new(0));
        let probe = RoleProbe::default();
        controlled_roles(&f, [
            ControlledRole { message: text_reply("PLANNER_DRAFT"), gate: Some(planner_gate.clone()), ..Default::default() },
            ControlledRole { message: text_reply("CRITIC_DRAFT"), reject_if_busy: true, ..Default::default() },
            ControlledRole { message: text_reply("REFINER_DRAFT"), ..Default::default() },
        ], probe.clone());
        let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
        let _cleanup = AbortOnDrop(running.abort_handle());
        probe.wait_for_roles(1).await;
        assert_eq!(probe.entered.lock().unwrap().len(), 1, "{expected_label} must not admit a sibling before the planner completes");
        planner_gate.add_permits(1);
        probe.wait_for_exit("planner").await;
        tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().unwrap();
        idle(&session).await;
        assert_eq!(probe.capacity_refusals.load(std::sync::atomic::Ordering::SeqCst), 0, "{expected_label}");
        assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 1, "{expected_label}");
        assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 1, "{expected_label}");
        for role in TextMasRole::ALL {
            assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Completed], "{expected_label}");
        }
        assert_eq!(ledger(&session).summary().unwrap().calls, 3, "{expected_label}");
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn hierarchical_provider_capacity_refusal_fails_without_serial_fallback() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("3"));
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let planner_gate = Arc::new(tokio::sync::Semaphore::new(0));
    let critic_admission = Arc::new(tokio::sync::Semaphore::new(0));
    let refiner_admission = Arc::new(tokio::sync::Semaphore::new(0));
    let probe = RoleProbe::default();
    controlled_roles(&f, [
        ControlledRole { message: text_reply("PLANNER_DRAFT"), gate: Some(planner_gate.clone()), ..Default::default() },
        ControlledRole { message: provider_error("unused rejection message"), admission_gate: Some(critic_admission.clone()), reject_if_busy: true, ..Default::default() },
        ControlledRole { message: text_reply("REFINER_DRAFT"), admission_gate: Some(refiner_admission.clone()), ..Default::default() },
    ], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(1).await;
    assert_eq!(probe.entered.lock().unwrap().as_slice(), ["planner"]);
    critic_admission.add_permits(1);
    let error = tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().unwrap_err();
    assert!(error.contains("OCC3.1 critic"), "{error}");
    assert_eq!(probe.capacity_refusals.load(std::sync::atomic::Ordering::SeqCst), 1, "the gated busy-refusal path must execute");
    assert_eq!(probe.peak.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(probe.entered.lock().unwrap().as_slice(), ["planner", "critic"]);
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(terminal_status(&session, TextMasRole::Planner), vec![TextMasCallStatus::Cancelled]);
    assert_eq!(terminal_status(&session, TextMasRole::Critic), vec![TextMasCallStatus::Failed]);
    let refiner = terminal_status(&session, TextMasRole::Refiner);
    assert!(refiner.is_empty() || refiner == [TextMasCallStatus::Cancelled]);
    let activity = settled_activity(&session);
    assert_eq!(activity.calls.len(), 3, "only the three real planned roles appear");
    assert_eq!(activity_role_status(&activity, TextMasRole::Planner), TextMasActivityStatus::Cancelled);
    assert_eq!(activity_role_status(&activity, TextMasRole::Critic), TextMasActivityStatus::Failed);
    let expected_refiner = if refiner.is_empty() { TextMasActivityStatus::Skipped } else { TextMasActivityStatus::Cancelled };
    assert_eq!(activity_role_status(&activity, TextMasRole::Refiner), expected_refiner);
    let records = role_usage(&session);
    let started = records.iter().filter(|record| record.status == TextMasCallStatus::Started).count();
    let terminal = records.iter().filter(|record| record.status != TextMasCallStatus::Started).count();
    assert!((2..=3).contains(&started) && started == terminal);
    assert_eq!(ledger(&session).summary().unwrap().calls, started);
    planner_gate.add_permits(1);
    refiner_admission.add_permits(1);
    session.dispose_async(Some(false)).await;
}


#[tokio::test(start_paused = true)]
async fn hierarchical_timeout_cancels_siblings_without_resetting_role_deadlines() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(Some("3"));
    let f = Fixture::new();
    let session = f.session(0, false).await;
    textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let role = |text| ControlledRole { message: text_reply(text), gate: Some(gate.clone()), reject_if_busy: false, ..Default::default() };
    let probe = RoleProbe::default();
    controlled_roles(&f, [role("LATE_PLANNER"), role("LATE_CRITIC"), role("LATE_REFINER")], probe.clone());
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _cleanup = AbortOnDrop(running.abort_handle());
    probe.wait_for_roles(3).await;
    tokio::time::advance(Duration::from_secs(121)).await;
    let result = tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap();
    assert!(result.unwrap_err().contains("timed out"));
    assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    let statuses = TextMasRole::ALL.into_iter().flat_map(|role| terminal_status(&session, role)).collect::<Vec<_>>();
    assert_eq!(statuses.iter().filter(|status| **status == TextMasCallStatus::TimedOut).count(), 1);
    assert_eq!(statuses.iter().filter(|status| **status == TextMasCallStatus::Cancelled).count(), 2);
    let activity = settled_activity(&session);
    assert_eq!(activity.calls.len(), 3);
    let activity_statuses = activity.calls.iter().map(|call| call.status).collect::<Vec<_>>();
    assert_eq!(activity_statuses.iter().filter(|status| **status == TextMasActivityStatus::TimedOut).count(), 1);
    assert_eq!(activity_statuses.iter().filter(|status| **status == TextMasActivityStatus::Cancelled).count(), 2);
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    gate.add_permits(3);
    session.dispose_async(Some(false)).await;
}
#[tokio::test]
async fn cancellation_settles_without_final_execution_and_next_turn_has_fresh_roles() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, true).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    f.reply(text_reply("CANCELLED_DRAFT"), Some(file.clone()), Some(release));
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    let _running_cleanup = AbortOnDrop(running.abort_handle());
    tokio::time::timeout(Duration::from_secs(10), async {
        while f.captured.lock().unwrap().is_empty() { tokio::task::yield_now().await; }
    }).await.unwrap();
    let running_id = session.get_session_action_recovery_snapshot().actions[0].id.clone();
    tokio::time::timeout(Duration::from_secs(10), session.prompt("UNRELATED_QUEUED_REQUEST", Some(PromptOptions {
        source: Some(InputSource::Rpc), streaming_behavior: Some("followUp".into()),
        queue_if_busy: Some(true), follow_up_queue_key: Some("unrelated-fixture".into()), ..Default::default()
    }))).await.unwrap().unwrap();
    session.abort().await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap();
    assert!(result.is_err(), "the aborted running input must settle with failure");
    let remaining = session.get_session_action_recovery_snapshot().actions;
    assert!(!remaining.iter().any(|action| action.id == running_id), "aborted ticket is terminal");
    assert_eq!(remaining.len(), 1, "only the unrelated queued input remains");
    assert_eq!(remaining[0].queue_key.as_deref(), Some("unrelated-fixture"));
    // Only this explicit fixture cleanup, not failed TextMAS preparation, clears it.
    session.clear_queue();
    idle(&session).await;
    assert_eq!(f.captured.lock().unwrap().len(), 1);
    let summary = ledger(&session).summary().unwrap();
    assert_eq!(summary.calls, 1);
    assert_eq!(summary.unknown_usage_calls, 1);
    assert_eq!(summary.unknown_cost_calls, 1);
    f.drafts(&file);
    f.reply(text_reply("AFTER_CANCEL"), None, None);
    turn(&session, "CURRENT_REQUEST fresh", None).await.unwrap();
    assert_eq!(f.captured.lock().unwrap().len(), 5);
    assert!(!serde_json::to_string(f.captured.lock().unwrap().last().unwrap()).unwrap().contains("CANCELLED_DRAFT"));
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn final_provider_retry_reuses_three_drafts_without_another_stage_call() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    for (mode, configured_concurrency) in [
        (TextMasMode::Sequential, None),
        (TextMasMode::Hierarchical, None),
        (TextMasMode::Hierarchical, Some("3")),
    ] {
        let _concurrency = ScopedHierarchicalConcurrencyEnv::set(configured_concurrency);
        let f = Fixture::new();
        let session = f.session(0, true).await;
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, mode).unwrap();
        f.drafts(&file);
        f.reply(faux_assistant_message("".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("error".into()), error_message: Some("temporary provider failure".into()), ..Default::default() })), None, None);
        f.reply(text_reply("RETRIED_JUDGER"), None, None);
        turn(&session, "CURRENT_REQUEST", None).await.unwrap();
        let contexts = f.captured.lock().unwrap();
        assert_eq!(contexts.len(), 5);
        assert!(contexts[3].tools.is_some() && contexts[4].tools.is_some());
        assert!(serde_json::to_string(&contexts[4].messages).unwrap().contains("REFINER_DRAFT"));
        drop(contexts);
        let activity = session.textmas_activity_snapshot().unwrap();
        assert!(activity.is_valid() && !activity.active);
        assert_eq!(activity.mode, Some(mode));
        assert_eq!(activity.calls.len(), 3, "a parent retry cannot create fresh role identities");
        if mode == TextMasMode::Sequential {
            assert_eq!(activity.calls[0].depends_on.len(), 0);
            assert_eq!(activity.calls[1].depends_on, vec![activity.calls[0].call_id.clone()]);
            assert_eq!(activity.calls[2].depends_on, vec![activity.calls[1].call_id.clone()]);
        } else {
            assert!(activity.calls.iter().all(|call| call.depends_on.is_empty()));
        }
        assert!(activity.calls.iter().all(|call| call.status == pi_coding_agent::core::occ::textmas_activity::TextMasActivityStatus::Completed));
        assert_eq!(ledger(&session).summary().unwrap().calls, 3);
        session.dispose_async(Some(false)).await;
    }
}
#[tokio::test]
async fn usage_records_are_deduplicated_in_disk_inventory_and_context_tree() {
    use pi_coding_agent::core::occ::textmas::{TextMasCallStatus, TextMasRole, TextMasUsageRecord, USAGE_CUSTOM_TYPE};
    use pi_coding_agent::core::context_tree::{context_tree_entry_from_json, compute_own_and_total_usage};
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("usage.jsonl");
    let mut usage = pi_ai::types::Usage::default();
    usage.input = 17.0; usage.output = 4.0; usage.total_tokens = 21.0; usage.cost.total = 0.25;
    let start = TextMasUsageRecord::new("a", TextMasRole::Planner, TextMasCallStatus::Started, None);
    let done = TextMasUsageRecord::new("a", TextMasRole::Planner, TextMasCallStatus::Failed, Some(usage.clone()));
    let unknown = TextMasUsageRecord::new("a", TextMasRole::Critic, TextMasCallStatus::Cancelled, None);
    let values = vec![
        json!({"type":"session","id":"usage","version":3,"timestamp":"2026-01-01T00:00:00Z","cwd":"/workspace"}),
        json!({"type":"custom","id":"one","customType":USAGE_CUSTOM_TYPE,"data":start}),
        json!({"type":"custom","id":"two","parentId":"one","customType":USAGE_CUSTOM_TYPE,"data":done}),
        json!({"type":"custom","id":"three","parentId":"two","customType":USAGE_CUSTOM_TYPE,"data":done}),
        json!({"type":"custom","id":"four","parentId":"three","customType":USAGE_CUSTOM_TYPE,"data":unknown}),
    ];
    std::fs::write(&file, values.iter().map(|value| format!("{value}\n")).collect::<String>()).unwrap();
    let entries: Vec<_> = values.iter().filter_map(context_tree_entry_from_json).collect();
    let (own, total) = compute_own_and_total_usage(&entries, &entries);
    assert_eq!(own, usage); assert_eq!(total, usage);
    let info = pi_coding_agent::core::session_manager::read_session_info(file.to_str().unwrap()).await.unwrap();
    let summary = info.usage.unwrap();
    assert_eq!(summary.input_tokens, 17.0);
    assert_eq!(summary.output_tokens, 4.0);
    assert_eq!(summary.cost, 0.25);
}

#[tokio::test]
async fn invalid_preferences_and_transcript_persistence_fail_before_any_advisory_spend() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    for preference_error in [true, false] {
        let f = Fixture::new();
        let session = f.session(0, true).await;
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
        if preference_error {
            std::fs::write(textmas_settings::preference_path(&file), "invalid settings").unwrap();
        } else {
            if std::path::Path::new(&file).exists() { std::fs::remove_file(&file).unwrap(); }
            std::fs::create_dir(&file).unwrap();
        }
        assert!(turn(&session, "CURRENT_REQUEST", None).await.is_err());
        assert!(f.captured.lock().unwrap().is_empty());
        assert_eq!(ledger(&session).len(), 0);
        if !preference_error { std::fs::remove_dir(&file).unwrap(); }
        session.dispose_async(Some(false)).await;
    }
}



#[tokio::test]
async fn model_change_cancels_every_admitted_mapper_only_once() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, true).await;
    f.reply(text_reply("HISTORY_ANSWER"), None, None);
    let history = "h".repeat(MAX_CONTEXT_BYTES * 12);
    turn(&session, &history, None).await.unwrap();
    let baseline_calls = f.captured.lock().unwrap().len();
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    let mapper_gate = Arc::new(tokio::sync::Semaphore::new(0));
    // Faux responses are consumed once per provider call. Arm every admitted
    // mapper with the same gate so model invalidation, rather than an empty
    // scripted-response queue, determines the terminal behavior.
    for _ in 0..3 {
        f.reply(
            text_reply("STALE_MODEL_DRAFT"),
            Some(file.clone()),
            Some(mapper_gate.clone()),
        );
    }
    let running = tokio::spawn({
        let session = session.clone();
        async move { session.prompt("CURRENT_REQUEST", None).await }
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while f.captured.lock().unwrap().len() < baseline_calls + 3 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("bounded mapper admission follows the oversized history");
    let mut other = session.model().unwrap();
    other.id = "another-ordinary-model".into();
    session.set_model(other, Default::default()).await.unwrap();
    let _ = tokio::time::timeout(Duration::from_secs(10), running)
        .await
        .expect("model invalidation settles the mapper fan-out");
    idle(&session).await;
    {
        let captured = f.captured.lock().unwrap();
        assert_eq!(
            captured.len(),
            baseline_calls + 3,
            "no later level, role, parent, or unadmitted mapper"
        );
        assert!(captured[baseline_calls..].iter().all(|context| {
            context
                .system_prompt
                .as_deref()
                .unwrap_or_default()
                .contains("advisory role: mapper")
        }));
    }
    let mut identities = std::collections::BTreeMap::<u64, Vec<String>>::new();
    for line in std::fs::read_to_string(session.session_file().unwrap())
        .unwrap()
        .lines()
    {
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        if entry["customType"] == "occ31_textmas_usage" && entry["data"]["role"] == "mapper" {
            identities
                .entry(entry["data"]["callIndex"].as_u64().unwrap())
                .or_default()
                .push(entry["data"]["status"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(identities.len(), 3);
    for (index, statuses) in identities {
        assert_eq!(
            statuses.iter().map(String::as_str).collect::<Vec<_>>(),
            ["started", "cancelled"],
            "mapper {index}: one Started and exactly one Cancelled"
        );
    }
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    assert!(!serde_json::to_string(&session.messages())
        .unwrap()
        .contains("STALE_MODEL_DRAFT"));
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn model_change_cancels_the_advisory_call_instead_of_reinterpreting_its_draft() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    for configured_concurrency in [None, Some("3")] {
        let _concurrency = ScopedHierarchicalConcurrencyEnv::set(configured_concurrency);
        let expected = if configured_concurrency.is_some() { 3 } else { 1 };
        let f = Fixture::new();
        let session = f.session(0, true).await;
        textmas_settings::set_mode(&session.session_file().unwrap(), TextMasMode::Hierarchical).unwrap();
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let role = |text| ControlledRole { message: text_reply(text), gate: Some(gate.clone()), reject_if_busy: false, ..Default::default() };
        let probe = RoleProbe::default();
        controlled_roles(&f, [role("STALE_PLANNER"), role("STALE_CRITIC"), role("STALE_REFINER")], probe.clone());
        let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
        let _cleanup = AbortOnDrop(running.abort_handle());
        probe.wait_for_roles(expected).await;
        let mut other = session.model().unwrap();
        other.id = "another-ordinary-model".into();
        session.set_model(other, Default::default()).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(10), running).await.unwrap().unwrap().is_err());
        idle(&session).await;
        assert_eq!(f.captured.lock().unwrap().len(), expected);
        assert_eq!(probe.parent_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        for (index, role) in TextMasRole::ALL.into_iter().enumerate() {
            if index < expected {
                assert_eq!(terminal_status(&session, role), vec![TextMasCallStatus::Cancelled]);
            } else {
                assert!(terminal_status(&session, role).is_empty(), "an unadmitted role has no fabricated usage record");
            }
        }
        assert_eq!(ledger(&session).summary().unwrap().calls, expected);
        assert!(!serde_json::to_string(&session.messages()).unwrap().contains("STALE_"));
        gate.add_permits(expected);
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test]
async fn queued_all_batch_keeps_both_user_inputs_and_delivers_all_three_drafts() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let _concurrency = ScopedHierarchicalConcurrencyEnv::set(None);
    let f = Fixture::new();
    let session = f.session(0, false).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Hierarchical).unwrap();
    session.set_follow_up_mode("all");
    let pause = session.acquire_queued_work_pause();
    // The first prompt waits for completion even under a queue pause. Admit it in
    // an owned task, then admit the second while busy; release before awaiting both.
    let first = tokio::spawn({ let session = session.clone(); async move {
        session.prompt("CURRENT_REQUEST first batched input", Some(PromptOptions {
            source: Some(InputSource::Rpc), streaming_behavior: Some("followUp".into()),
            queue_if_busy: Some(true), ..Default::default()
        })).await
    }});
    let _first_cleanup = AbortOnDrop(first.abort_handle());
    tokio::time::timeout(Duration::from_secs(10), async {
        while session.unfinished_action_count() != 1 { tokio::task::yield_now().await; }
    }).await.unwrap();
    let second = tokio::spawn({ let session = session.clone(); async move {
        session.prompt("CURRENT_REQUEST second batched input", Some(PromptOptions {
            source: Some(InputSource::Rpc), streaming_behavior: Some("followUp".into()),
            queue_if_busy: Some(true), ..Default::default()
        })).await
    }});
    let _second_cleanup = AbortOnDrop(second.abort_handle());
    tokio::time::timeout(Duration::from_secs(10), async {
        while session.unfinished_action_count() != 2 { tokio::task::yield_now().await; }
    }).await.unwrap();
    let admitted = session.get_session_action_recovery_snapshot().actions;
    assert_eq!(admitted.len(), 2);
    assert!(admitted.iter().all(|action| action.source.as_str() == "rpc"));
    f.drafts(&file);
    f.reply(text_reply("BATCH_JUDGER"), None, None);
    pause.release();
    tokio::time::timeout(Duration::from_secs(20), first).await.unwrap().unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(20), second).await.unwrap().unwrap().unwrap();
    idle(&session).await;
    let contexts = f.captured.lock().unwrap().clone();
    assert_eq!(contexts.len(), 4);
    for context in &contexts {
        let text = serde_json::to_string(&context.messages).unwrap();
        assert!(text.contains("first batched input") && text.contains("second batched input"));
    }
    let final_context = serde_json::to_string(&contexts[3].messages).unwrap();
    assert!(final_context.contains("PLANNER_DRAFT") && final_context.contains("CRITIC_DRAFT") && final_context.contains("REFINER_DRAFT"));
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    session.dispose_async(Some(false)).await;
}


#[tokio::test]
async fn successful_turn_clears_activity_and_admits_a_subsequent_turn() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, false).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.drafts(&file);
    f.reply(text_reply("FIRST_PARENT"), None, None);
    turn(&session, "CURRENT_REQUEST first successful turn", None).await.unwrap();
    assert!(!session.is_textmas_active(), "the settled parent must clear TextMAS activity");
    f.drafts(&file);
    f.reply(text_reply("SECOND_PARENT"), None, None);
    turn(&session, "CURRENT_REQUEST subsequent user turn", None).await.unwrap();
    assert!(!session.is_textmas_active(), "the second settled parent must also clear TextMAS activity");
    let contexts = f.captured.lock().unwrap();
    assert_eq!(contexts.len(), 8, "both turns must have fresh roles and a parent");
    assert_eq!(contexts.iter().filter(|context| context.tools.is_some()).count(), 2);
    drop(contexts);
    assert_eq!(ledger(&session).summary().unwrap().calls, 6);
    session.dispose_async(Some(false)).await;
}
