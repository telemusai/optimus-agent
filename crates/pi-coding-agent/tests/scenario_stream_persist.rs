
//! Campaign section 35 end-to-end scenarios owned by w6-providers.
//!
//! Scenario K - provider interruption: the stream fails after a partial
//! tool-call event. The engine must not execute the incomplete tool call and
//! must not record a fabricated success in the transcript. The unexecuted call
//! is still answered exactly once, with an error result, so the persisted
//! history stays protocol-valid without ever claiming the tool ran.
//!
//! Scenario L - session write failure: a failed session write must keep the
//! previous authoritative state intact and report the failure to the caller
//! instead of swallowing it.
//!
//! Both scenarios drive production code paths only (the real agent loop, the
//! real `AgentSession`, the real `SessionManager` JSONL store). No provider
//! network calls: the interrupted stream is a local fixture and every writable
//! path lives under a private temp root.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pi_agent_core::agent_loop::agent_loop;
use pi_agent_core::types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool, AgentToolResult,
};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, ImageOrTextContent, Model,
    SimpleStreamOptions, ToolCall, ToolResultMessage, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::core::session_manager::SessionManager;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// V00 isolation harness
// ---------------------------------------------------------------------------

fn state_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let root = std::env::var_os("OPTIMUS_PARITY_STATE_ROOT")
        .map(|raw| {
            assert!(
                !raw.is_empty(),
                "OPTIMUS_PARITY_STATE_ROOT must not be empty"
            );
            std::path::absolute(raw).expect("absolute state root")
        })
        .unwrap_or_else(|| {
            ROOT.get_or_init(|| {
                tempfile::Builder::new()
                    .prefix("optimus-scenario-stream-persist-")
                    .tempdir()
                    .expect("private state root")
            })
            .path()
            .to_path_buf()
        });
    let lowered = root.to_string_lossy().to_lowercase();
    assert!(!lowered.contains(".prime"), "state root must not touch .prime");
    std::fs::create_dir_all(&root).expect("create state root");
    root
}

fn case_dir(tag: &str) -> PathBuf {
    let dir = state_root()
        .join(tag)
        .join(format!("{}-{}", std::process::id(), unique_tag()));
    std::fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn unique_tag() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    static TAG_SEQ: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
        .unwrap_or(0);
    format!("{}-{}", nanos, TAG_SEQ.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Scenario K fixtures: a provider that dies mid tool call
// ---------------------------------------------------------------------------

const INTERRUPTED_TOOL_ID: &str = "call-interrupted-1";
const INTERRUPTED_TOOL_NAME: &str = "probe_write";
const INTERRUPTED_ARGS_PREFIX: &str = "{\"path\":\"marker.txt\"";

fn fixture_model() -> Model {
    let mut model = Model::new(
        "scenario-k-interrupted",
        "Scenario K interrupted model",
        "scenario-k",
        "scenario-k",
        "https://scenario.invalid",
    );
    model.context_window = 32_000.0;
    model.max_tokens = 4_096.0;
    model
}

fn empty_partial(model: &Model) -> AssistantMessage {
    let mut partial = AssistantMessage::new(
        model.api.clone(),
        model.provider.clone(),
        model.id.clone(),
        1,
    );
    partial.stop_reason = "toolUse".to_string();
    partial
}

/// The partial assistant message a real provider holds after a tool-call
/// delta: the call id and name are known, the JSON arguments are half written.
fn partial_tool_call_message(model: &Model) -> AssistantMessage {
    let mut partial = empty_partial(model);
    partial.content = vec![ContentBlock::ToolCall(ToolCall::new(
        INTERRUPTED_TOOL_ID.to_string(),
        INTERRUPTED_TOOL_NAME.to_string(),
        json!({"path": "marker.txt", "content": "must not be written"})
            .as_object()
            .cloned()
            .unwrap_or_default(),
    ))];
    partial
}

/// The terminal message a failed transport reports. A real provider keeps the
/// content it had already streamed, so the partial tool call is still present
/// in the error message: the engine, not the provider, owns the guarantee
/// that an incomplete call never runs.
fn interrupted_error_message(model: &Model) -> AssistantMessage {
    let mut error = partial_tool_call_message(model);
    error.stop_reason = "error".to_string();
    error.error_message = Some("connection reset by peer".to_string());
    error.usage = pi_ai::types::Usage::zero();
    error
}

/// Push one interrupted provider response: start, tool-call start, one
/// fragmented arguments delta, then a transport error.
fn interrupted_stream(model: &Model) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let partial = partial_tool_call_message(model);
    stream.push(AssistantMessageEvent::Start {
        partial: empty_partial(model),
    });
    stream.push(AssistantMessageEvent::ToolCallStart {
        content_index: 0,
        partial: partial.clone(),
    });
    stream.push(AssistantMessageEvent::ToolCallDelta {
        content_index: 0,
        delta: format!("{INTERRUPTED_ARGS_PREFIX}, \"content\": \"mu"),
        partial: partial.clone(),
    });
    let error = interrupted_error_message(model);
    stream.push(AssistantMessageEvent::Error {
        reason: "transport".to_string(),
        error: error.clone(),
    });
    stream.end(Some(error));
    stream
}

/// A tool whose only job is to prove it never ran: a counter and a file.
fn probe_tool(dir: &Path, calls: &Arc<AtomicUsize>) -> AgentTool {
    let marker = dir.join("probe-marker.txt");
    let calls = Arc::clone(calls);
    AgentTool {
        name: INTERRUPTED_TOOL_NAME.to_string(),
        description: "writes a marker file".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }),
        label: "probe".to_string(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, args, _signal, _on_update| {
            let calls = Arc::clone(&calls);
            let marker = marker.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let content = args
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                std::fs::write(&marker, content).expect("probe write");
                Ok(AgentToolResult::new(
                    vec![pi_agent_core::types::ContentBlock::text("written")],
                    json!({"ok": true}),
                ))
            })
        }),
        execution_mode: None,
    }
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Message(pi_ai::types::Message::User(UserMessage::new(
        UserContent::Text(text.to_string()),
        1,
    )))
}

fn assistant_of(message: &AgentMessage) -> Option<&AssistantMessage> {
    match message {
        AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => Some(assistant),
        _ => None,
    }
}

fn tool_result_of(message: &AgentMessage) -> Option<&ToolResultMessage> {
    match message {
        AgentMessage::Message(pi_ai::types::Message::ToolResult(result)) => Some(result),
        _ => None,
    }
}

fn tool_results_of(messages: &[AgentMessage]) -> Vec<&ToolResultMessage> {
    messages.iter().filter_map(tool_result_of).collect()
}

fn tool_result_text(result: &ToolResultMessage) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            ImageOrTextContent::Text(text) => Some(text.text.clone()),
            ImageOrTextContent::Image(_) => None,
        })
        .collect::<Vec<String>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Scenario K (engine level)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scenario_k_partial_tool_call_then_stream_error_never_executes_the_tool() {
    let dir = case_dir("scenario-k");
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fixture_model();

    let mut config = AgentLoopConfig::new(model.clone());
    config.convert_to_llm = Some(Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|message| message.as_message().cloned())
                .collect::<Vec<_>>()
        })
    }));

    // The prompt is added to the context by the loop itself.
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![probe_tool(&dir, &calls)]),
    };

    let stream_model = model.clone();
    let stream_fn: pi_agent_core::types::StreamFn =
        Arc::new(move |_model, _context, _options: SimpleStreamOptions| {
            let stream_model = stream_model.clone();
            Box::pin(async move { interrupted_stream(&stream_model) })
        });

    let stream = agent_loop(
        vec![user_message("write the marker file")],
        context,
        config,
        None,
        Some(stream_fn),
    );

    let mut events: Vec<AgentEvent> = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(event) = stream.next().await {
            events.push(event);
        }
    })
    .await
    .expect("the interrupted run must terminate");
    let produced = tokio::time::timeout(Duration::from_secs(30), stream.result())
        .await
        .expect("the interrupted run must settle its transcript result")
        .expect("the interrupted run must publish its transcript");

    // The tool never ran, and no side effect reached the filesystem.
    assert_eq!(calls.load(Ordering::SeqCst), 0, "the incomplete tool call must not execute");
    assert!(
        !dir.join("probe-marker.txt").exists(),
        "an interrupted tool call must leave no side effect"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolExecutionStart { .. })),
        "no tool execution may start for an interrupted stream"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolExecutionEnd { .. })),
        "no tool execution may end for an interrupted stream"
    );

    // No fabricated success reached the produced transcript. The unexecuted
    // call is still answered exactly once, with the dedicated error result
    // (`answer_unexecuted_tool_calls`, agent_loop.rs), so every assistant tool
    // call stays paired with a tool result - but the answer never claims the
    // tool ran and never carries the probe's success details.
    let produced_results = tool_results_of(&produced);
    assert_eq!(
        produced_results.len(),
        1,
        "the unexecuted call is answered exactly once"
    );
    assert_eq!(produced_results[0].tool_call_id, INTERRUPTED_TOOL_ID);
    assert_eq!(produced_results[0].tool_name, INTERRUPTED_TOOL_NAME);
    assert!(
        produced_results[0].is_error,
        "the answer for an interrupted call must be an error"
    );
    assert_eq!(
        tool_result_text(produced_results[0]),
        pi_agent_core::agent_loop::TOOL_EXECUTION_ABORTED_MESSAGE,
        "the answer must state that the call never ran"
    );
    assert_ne!(
        produced_results[0].details,
        Some(json!({"ok": true})),
        "the probe's success details must never be fabricated"
    );
    assert_eq!(
        produced_results.iter().filter(|result| !result.is_error).count(),
        0,
        "no tool result may report success"
    );
    for event in &events {
        if let AgentEvent::TurnEnd { tool_results, .. } = event {
            assert_eq!(
                tool_results.len(),
                1,
                "an interrupted turn ends with the one error answer"
            );
            assert!(
                tool_results.iter().all(|result| result.is_error),
                "an interrupted turn must never end with a successful tool result"
            );
        }
        if let AgentEvent::MessageEnd { message } = event {
            if message.role() == "toolResult" {
                let result = tool_result_of(message).expect("a tool-result message");
                assert!(
                    result.is_error,
                    "no success tool-result message may be emitted"
                );
            }
        }
    }

    // The turn is an explicit provider failure, not a success.
    let assistant_messages: Vec<&AssistantMessage> =
        produced.iter().filter_map(assistant_of).collect();
    assert_eq!(assistant_messages.len(), 1, "exactly one assistant response");
    assert_eq!(
        assistant_messages[0].stop_reason, "error",
        "the interrupted response must be a provider error"
    );
    assert_eq!(
        assistant_messages[0].error_message.as_deref(),
        Some("connection reset by peer")
    );
}

// ---------------------------------------------------------------------------
// Scenario L: a failed session write must not destroy or hide state
// ---------------------------------------------------------------------------

fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read session file")
}

#[allow(clippy::permissions_set_readonly_false)]
fn set_read_only(path: &Path, read_only: bool) {
    let mut permissions = std::fs::metadata(path).expect("metadata").permissions();
    permissions.set_readonly(read_only);
    std::fs::set_permissions(path, permissions).expect("set permissions");
}

/// Atomic-write temp files left in `dir`. `write_file_atomic_sync` writes
/// `<target>.<pid>.<uuid>.tmp` beside the destination and removes it on every
/// path, success or failure, so a leftover name means a torn write escaped.
fn temp_files_in(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect("read the session directory")
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".tmp"))
        .collect()
}

fn parse_lines(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("valid session line"))
        .collect()
}

#[test]
fn scenario_l_failed_session_write_keeps_the_authoritative_file_and_reports_the_error() {
    let dir = case_dir("scenario-l");
    let session_dir = dir.join("sessions");
    let mut manager = SessionManager::create(dir.to_string_lossy().as_ref(), Some(
        session_dir.to_string_lossy().as_ref(),
    ))
    .expect("create session");

    manager
        .append_message(user_message("keep this turn"))
        .expect("append user turn");
    manager.flush_now().expect("flush the authoritative state");

    let session_file = manager
        .get_session_file()
        .map(PathBuf::from)
        .expect("a persisted session has a file");
    let before = read_bytes(&session_file);
    let leaf_before = manager.get_leaf_id();
    let count_before = manager.get_entry_count();
    assert!(!before.is_empty(), "the authoritative state must be on disk");

    // A write failure that a real disk throws: the session file is read-only,
    // so the append cannot open it for writing.
    set_read_only(&session_file, true);
    let epoch_before = manager.persist_write_epoch();
    let failed = manager.append_custom_entry_with_rollback("scenario_l_probe", Some(json!({"n": 1})));
    // Restore writability before any assertion so a failure cannot leak the
    // read-only attribute into the next test, and so the recovery append below
    // runs against a writable file.
    set_read_only(&session_file, false);

    let error = failed.expect_err("a failed session write must be reported, not swallowed");
    assert!(
        !error.trim().is_empty(),
        "the reported failure must carry a reason, got {error:?}"
    );

    // The previous authoritative state is intact, byte for byte.
    let after_bytes = read_bytes(&session_file);
    assert_eq!(
        after_bytes, before,
        "a failed append must not change the committed session file"
    );
    // The write path is append-only plus an atomic rename, so a failure leaves
    // neither a torn line tail nor a temp file beside the session.
    assert!(
        after_bytes.ends_with(b"\n"),
        "a failed append must not leave a torn line tail"
    );
    let leftovers = temp_files_in(&session_dir);
    assert!(
        leftovers.is_empty(),
        "an atomic write must not leave temp files behind, got {leftovers:?}"
    );
    let after_lines = parse_lines(&after_bytes);
    assert_eq!(
        after_lines.len(),
        count_before + 1,
        "the header plus the committed turn must still be the whole file"
    );

    // The in-memory transcript rolled back with the failure: no phantom entry.
    assert_eq!(manager.get_leaf_id(), leaf_before, "the leaf must roll back");
    assert_eq!(manager.get_entry_count(), count_before, "no entry may survive a failed write");
    assert!(
        !manager.get_entries().iter().any(|entry| {
            entry.get("customType").and_then(Value::as_str) == Some("scenario_l_probe")
        }),
        "the rejected entry must not stay in memory"
    );
    // `persist_write_epoch` is an accept-time dirty marker, not a commit
    // counter: the rejected entry dirties the session even though nothing
    // reached the file, which is what forces the next durability boundary to
    // drain and fsync. The file bytes above remain the authoritative check.
    assert!(
        manager.persist_write_epoch() > epoch_before,
        "a rejected append must still mark the session dirty for the next durability boundary"
    );

    // Recovery still works: the next append rewrites the transcript and lands
    // exactly one copy of the new entry.
    manager
        .append_custom_entry_with_rollback("scenario_l_probe", Some(json!({"n": 2})))
        .expect("the session recovers after the failed write");
    manager.flush_now().expect("flush the recovered state");

    let recovered_bytes = read_bytes(&session_file);
    assert!(
        recovered_bytes.ends_with(b"\n"),
        "the recovered transcript must end on a whole line"
    );
    let recovered_leftovers = temp_files_in(&session_dir);
    assert!(
        recovered_leftovers.is_empty(),
        "the recovered transcript must leave no temp files, got {recovered_leftovers:?}"
    );
    let recovered = parse_lines(&recovered_bytes);
    let probe_entries: Vec<&Value> = recovered
        .iter()
        .filter(|entry| {
            entry.get("customType").and_then(Value::as_str) == Some("scenario_l_probe")
        })
        .collect();
    assert_eq!(probe_entries.len(), 1, "exactly one copy of the recovered entry");
    assert_eq!(probe_entries[0]["data"]["n"], json!(2));

    drop(manager);
    let reopened = SessionManager::open(session_file.to_string_lossy().as_ref(), None, None)
        .expect("the recovered session reopens");
    assert_eq!(
        reopened.get_entry_count(),
        count_before + 1,
        "the reopened transcript holds the committed entries only"
    );
}

// ---------------------------------------------------------------------------
// Scenario K (real AgentSession): the transcript keeps no fabricated success
// ---------------------------------------------------------------------------

mod session_level {
    use super::*;

    use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
    use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
    use pi_coding_agent::core::agent_session_services::{
        create_agent_session_from_services, create_agent_session_services,
        AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
        CreateAgentSessionServicesOptions,
    };
    use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
    use pi_coding_agent::core::model_registry::ModelRegistry;
    use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
    use pi_coding_agent::core::settings_manager::SettingsManager;
    use std::sync::Mutex;

    const API_KEY: &str = "scenario-k-fixture-key";

    /// Registers the interrupted provider under a private api id so this test
    /// cannot collide with any other fixture in the binary.
    fn register_interrupted_api() {
        register_api_provider_simple(
            ApiProviderSimple {
                api: "scenario-k".to_string(),
                stream: Arc::new(|model: &Model, _context: &pi_ai::types::Context, _options| {
                    interrupted_stream(model)
                }),
                stream_simple: Arc::new(
                    |model: &Model, _context: &pi_ai::types::Context, _options| {
                        interrupted_stream(model)
                    },
                ),
                compact: None,
                supports_compaction: None,
            },
            None,
        );
    }

    async fn fixture() -> (Arc<AgentSession>, tempfile::TempDir) {
        register_interrupted_api();
        let root = tempfile::Builder::new()
            .prefix("scenario-k-session-")
            .tempdir()
            .expect("temp root");
        let agent_dir = root.path().join("agent");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let cwd_string = root.path().to_string_lossy().to_string();
        let agent_dir_string = agent_dir.to_string_lossy().to_string();
        let model = fixture_model();

        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));

        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            AuthStorageData::new(),
            None,
        )));
        let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(AuthStorage::in_memory(
            AuthStorageData::new(),
            None,
        ))));
        auth_storage
            .lock()
            .await
            .set_runtime_api_key(&model.provider, API_KEY);
        model_registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, API_KEY);

        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd_string.clone(),
            agent_dir: Some(agent_dir_string.clone()),
            auth_storage: Some(Arc::clone(&auth_storage)),
            settings_manager: Some(Arc::clone(&settings)),
            model_registry: Some(Arc::clone(&model_registry)),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd_string.clone(),
                agent_dir: agent_dir_string.clone(),
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            }),
        })
        .await
        .expect("services");

        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::new(Mutex::new(
                SessionManager::create(&cwd_string, Some(&agent_dir_string)).expect("session"),
            )),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                rlm_depth: Some(0),
                no_tools: Some("all".to_string()),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                ..Default::default()
            },
        })
        .await
        .expect("agent session");
        (created.session, root)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scenario_k_session_answers_the_interrupted_call_with_an_error_only() {
        let (session, _root) = fixture().await;

        // The provider dies mid tool call. A terminal provider failure settles
        // the prompt normally (see `transport_retry_tests`: "prompt settled")
        // and is reported through the transcript instead of rejecting the
        // caller. Bound the wait: a hung prompt must fail the test, not the
        // suite.
        let outcome = tokio::time::timeout(
            Duration::from_secs(30),
            session.prompt_and_wait("write the marker file", Some(PromptOptions::default())),
        )
        .await
        .expect("the interrupted prompt must not hang");
        assert!(
            outcome.is_ok(),
            "a terminal provider error settles the prompt and is reported through the transcript, got {outcome:?}"
        );

        let messages = session.agent.messages();
        let tool_results = tool_results_of(&messages);
        assert_eq!(
            tool_results.len(),
            1,
            "the unexecuted call is answered exactly once, got {tool_results:?}"
        );
        assert_eq!(tool_results[0].tool_call_id, INTERRUPTED_TOOL_ID);
        assert_eq!(tool_results[0].tool_name, INTERRUPTED_TOOL_NAME);
        assert!(
            tool_results[0].is_error,
            "an interrupted tool call must never produce a successful tool result"
        );
        assert_eq!(
            tool_result_text(tool_results[0]),
            pi_agent_core::agent_loop::TOOL_EXECUTION_ABORTED_MESSAGE,
            "the recorded answer must state that the call never ran"
        );
        assert_ne!(
            tool_results[0].details,
            Some(json!({"ok": true})),
            "the probe's success details must never be fabricated"
        );

        let assistants: Vec<_> = messages.iter().filter_map(assistant_of).collect();
        assert_eq!(assistants.len(), 1, "exactly one assistant response is recorded");
        assert_eq!(
            assistants[0].stop_reason, "error",
            "the recorded response is a provider error, not a success"
        );
        assert_eq!(
            assistants[0].error_message.as_deref(),
            Some("connection reset by peer"),
            "the provider failure must be surfaced in the transcript, not swallowed"
        );

        // The durable transcript agrees with the live one.
        session
            .session_manager
            .lock()
            .unwrap()
            .flush_now()
            .expect("flush transcript");
        let session_file = session
            .session_manager
            .lock()
            .unwrap()
            .get_session_file()
            .expect("persisted session");
        let entries = parse_lines(&read_bytes(Path::new(&session_file)));
        let persisted_tool_results: Vec<&Value> = entries
            .iter()
            .filter(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("message")
                    && entry
                        .get("message")
                        .and_then(|message| message.get("role"))
                        .and_then(Value::as_str)
                        == Some("toolResult")
            })
            .collect();
        assert_eq!(
            persisted_tool_results.len(),
            1,
            "the persisted transcript answers the interrupted call exactly once"
        );
        assert_eq!(
            persisted_tool_results[0]["message"]["isError"],
            json!(true),
            "the persisted answer is an error, never a fabricated success"
        );
        assert_eq!(
            persisted_tool_results[0]["message"]["toolCallId"],
            json!(INTERRUPTED_TOOL_ID)
        );

        // Reopening must not resurrect the call as a success either.
        let reopened =
            SessionManager::open(&session_file, None, None).expect("reopen the interrupted session");
        let context = reopened.build_session_context(None);
        let replayed_results = tool_results_of(&context.messages);
        assert_eq!(
            replayed_results.len(),
            1,
            "a replayed context answers the interrupted call exactly once"
        );
        assert!(
            replayed_results[0].is_error,
            "a replayed answer must stay an error"
        );
    }
}
