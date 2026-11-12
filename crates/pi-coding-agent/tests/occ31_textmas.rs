//! OCC3.1 native execution through a registered offline, non-latent custom provider.
use pi_ai::providers::faux::*;
use pi_ai::types::{AssistantMessage, ContentBlock, Context, Message, Model, ToolCall};
use pi_coding_agent::core::occ::{textmas::{MAX_CONTEXT_BYTES, TextMasMode, TextMasUsageLedger, usage_record}, textmas_settings};
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
async fn oversized_transformed_planner_context_fails_before_any_stage_spending() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, false).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.reply(text_reply("MUST_NOT_CALL"), None, None);
    let oversized_request = "x".repeat(MAX_CONTEXT_BYTES);
    let error = turn(&session, &oversized_request, None).await.unwrap_err();
    assert!(error.contains("256 KiB preparation limit"), "{error}");
    assert!(
        f.captured.lock().unwrap().is_empty(),
        "Planner must fail before its provider call"
    );
    assert_eq!(
        ledger(&session).len(),
        0,
        "Planner must fail before its Started reservation"
    );
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn off_internal_extension_and_child_turns_never_fan_out() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
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
    for case in ["error", "refusal", "empty", "tool", "oversized"] {
        let f = Fixture::new();
        let session = f.session(0, true).await;
        turn(&session, "/mode direct", None).await.unwrap();
        let file = session.session_file().unwrap();
        textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
        let side_effect = f.root.path().join("must-not-exist");
        let message = match case {
            "error" => faux_assistant_message("".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("error".into()), error_message: Some("temporary provider failure".into()), ..Default::default() })),
            "refusal" => { let mut m = text_reply("declined"); m.stop_reason_raw = Some("refusal".into()); m },
            "empty" => faux_assistant_message(FauxAssistantContent::Blocks(vec![ContentBlock::Thinking(faux_thinking("PRIVATE_ONLY"))]), None),
            "tool" => faux_assistant_message(ContentBlock::ToolCall(ToolCall::new("unsafe", "bash", json!({"command":format!("touch '{}'",side_effect.display())}).as_object().unwrap().clone())).into(), Some(FauxAssistantMessageOptions { stop_reason: Some("toolUse".into()), ..Default::default() })),
            _ => text_reply(&"x".repeat(16_385)),
        };
        f.reply(message, Some(file), None);
        assert!(turn(&session, "CURRENT_REQUEST", None).await.is_err(), "{case}");
        assert_eq!(f.captured.lock().unwrap().len(), 1, "no retry/fallback after {case}");
        assert!(!side_effect.exists());
        assert_eq!(ledger(&session).summary().unwrap().calls, 1);
        session.dispose_async(Some(false)).await;
    }
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
    let f = Fixture::new();
    let session = f.session(0, true).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Sequential).unwrap();
    f.drafts(&file);
    f.reply(faux_assistant_message("".into(), Some(FauxAssistantMessageOptions { stop_reason: Some("error".into()), error_message: Some("temporary provider failure".into()), ..Default::default() })), None, None);
    f.reply(text_reply("RETRIED_JUDGER"), None, None);
    turn(&session, "CURRENT_REQUEST", None).await.unwrap();
    let contexts = f.captured.lock().unwrap();
    assert_eq!(contexts.len(), 5);
    assert!(contexts[3].tools.is_some() && contexts[4].tools.is_some());
    assert!(serde_json::to_string(&contexts[4].messages).unwrap().contains("REFINER_DRAFT"));
    drop(contexts);
    assert_eq!(ledger(&session).summary().unwrap().calls, 3);
    session.dispose_async(Some(false)).await;
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
async fn model_change_cancels_the_advisory_call_instead_of_reinterpreting_its_draft() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
    let f = Fixture::new();
    let session = f.session(0, true).await;
    let file = session.session_file().unwrap();
    textmas_settings::set_mode(&file, TextMasMode::Hierarchical).unwrap();
    f.reply(text_reply("STALE_MODEL_DRAFT"), Some(file), Some(Arc::new(tokio::sync::Semaphore::new(0))));
    let running = tokio::spawn({ let session = session.clone(); async move { session.prompt("CURRENT_REQUEST", None).await } });
    tokio::time::timeout(Duration::from_secs(10), async {
        while f.captured.lock().unwrap().is_empty() { tokio::task::yield_now().await; }
    }).await.unwrap();
    let mut other = session.model().unwrap(); other.id = "another-ordinary-model".into();
    session.set_model(other, Default::default()).await.unwrap();
    let _ = tokio::time::timeout(Duration::from_secs(10), running).await.unwrap();
    idle(&session).await;
    assert_eq!(f.captured.lock().unwrap().len(), 1);
    assert_eq!(ledger(&session).summary().unwrap().calls, 1);
    assert!(!serde_json::to_string(&session.messages()).unwrap().contains("STALE_MODEL_DRAFT"));
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn queued_all_batch_keeps_both_user_inputs_and_delivers_all_three_drafts() {
    let _env = ENV.lock().unwrap_or_else(|p| p.into_inner());
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
