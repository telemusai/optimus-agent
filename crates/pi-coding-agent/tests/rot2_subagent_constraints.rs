//! ROT2 W8: compaction -> subagent constraint propagation tests (campaign 15).
//!
//! What these tests pin: a parent session that has crossed a compaction
//! boundary must still be able to hand a constraint-carrying task to an RLM
//! child, a child that compacts mid-task must keep the parent-imposed
//! constraint in its continuation context, and a child that finishes after
//! the parent compacted must deliver a result the parent can still resolve
//! against its compacted context.
//!
//! All scenarios run offline on the faux provider (agent turns and summary
//! wire calls alike), over the REAL `AgentSession` auto-compaction trigger,
//! the REAL session-action/continuation machinery and the REAL
//! `SessionManager` JSONL persistence. Agent-message routing between the
//! two sessions goes through an in-process `AgentSessionMessageController`
//! (`FamilyBus`) that routes exactly one parent and one child, mirroring the
//! daemon delivery surface (`roster` + `send_agent_message` ->
//! `accept_agent_message_prompt` on the target session).
//!
//! Coverage map (brief scenarios 1-3):
//! - (1) `rot2_w8_parent_compaction_then_child_task_prompt_carries_constraints`
//! - (2) `rot2_w8_agent_message_task_crosses_compaction_mid_child`
//! - (3) `rot2_w8_child_result_after_parent_compaction_resolves_on_compacted_context`
//!
//! Findings recorded against the brief (public-API gaps, not forced):
//! - F1 (scenario 1, spawn flavor): the production spawn path (the `rlm`
//!   tool -> `SubagentRuntimeHost::create_rlm_subagent_runtime` ->
//!   `RlmSubagentRuntime`) has no public in-process constructor:
//!   `RlmSubagentRuntime::new` does not exist and
//!   `CreateRlmSubagentRuntimeOptions` is a 20-field host DTO. An offline
//!   test cannot drive a real spawn, so scenario 1 drives the parent's
//!   public `send_agent_message` and the child's public
//!   `accept_agent_message_prompt` with the SPAWN durable shape (an
//!   `agent_message` custom entry whose details id starts with `spawn:`,
//!   the shape `rlm_child_usage_origin` classifies as a spawned task).
//!   Whether the runtime injects parent context into a spawned child's
//!   initial prompt is therefore unmeasured here.
//! - F2 (scenario 1, propagation semantics): with every public API
//!   reachable in-process, NOTHING injects the parent's compaction summary
//!   (or any constraint ledger) into a delivered child task prompt. The
//!   child receives exactly the text the parent sends. Constraint
//!   propagation to children is therefore carried by the parent's own
//!   compacted context plus the parent restating the constraint in the task
//!   text; the tests pin both halves of that chain. A dedicated constraint
//!   ledger does not exist in the audited surface.
//! - F3 (scenario 3, acknowledgement): the child's settlement only marks a
//!   parent task `replied` when `rlm_result_receipt_is_valid` (private, in
//!   core/rlm_continuation) accepts the controller's receipt. That contract
//!   could not be satisfied from a test-side controller across 32 receipt
//!   shapes, so the offline bus delivers successfully while the child keeps
//!   its durable delivery intent instead of the replied flag. The
//!   substantive scenario assertion (the receipt's entry id resolving in
//!   the parent's compacted context) is unaffected.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pi_agent_core::types::{AgentMessage, CustomMessageContent};
use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent,
    FauxAssistantMessageOptions, FauxModelDefinition, FauxResponseFactory, FauxResponseStep,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{ContentBlock, Context, Message, Model};
use pi_coding_agent::core::agent_messages::{
    AgentFamilyRosterCurrent, AgentFamilyRosterEntry, AgentFamilyRosterResult,
    AgentSessionMessageController, AgentSessionMessageEndpoint, AgentSessionMessageReceipt,
    AgentSessionMessageSendInput, AgentSessionMessageSender, FAMILY_RELATIONSHIP_CHILD,
    FAMILY_RELATIONSHIP_PARENT,
};
use pi_coding_agent::core::agent_session::{AgentSession, PromptOptions};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::messages::CustomMessage;
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::rlm_continuation::{
    parse_rlm_continuation_state, RlmContinuationState, RLM_CONTINUATION_STATE_CUSTOM_TYPE,
};
use pi_coding_agent::core::session_manager::{NewSessionOptions, SessionManager};
use pi_coding_agent::core::settings_manager::SettingsManager;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// V00 isolation harness (same contract as compaction_recovery.rs)
// ---------------------------------------------------------------------------

fn unique_tag() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static TAG_SEQ: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{}-{}", nanos, TAG_SEQ.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Shared constants and fixtures
// ---------------------------------------------------------------------------

/// The constraint every scenario plants. Unique enough that a hit means the
/// exact line survived, not a coincidence.
const W8_CONSTRAINT: &str = "CONSTRAINT: W8-CONSTRAINT never drop the payments ledger.";
const W8_GOAL: &str = "TASK GOAL: ship the migration report before Friday.";
const W8_EVIDENCE: &str = "Evidence reference: reports/w8-audit.md#findings.";
/// Marker planted ONLY in the parent's post-compaction turn; the child task
/// prompt never contains it. Its absence from the child's request proves no
/// parent context is injected into the delivered prompt (finding F2).
const W8_PARENT_ONLY_MARKER: &str = "PARENT-ONLY-MARKER-W8-7f3a";
/// A section header the fixture summarizer emits; the child task prompt
/// never contains it.
const W8_SUMMARY_SECTION: &str = "## Critical Context";
const W8_FILLER: &str = "planning notes: the ledger export needs a dry run first. ";
const W8_CHILD_FILLER: &str = "child work notes: section 3 is in progress. ";

const API_KEY: &str = "rot2-w8-constraints-fixture-key";

/// One captured provider request: the joined visible text plus whether it
/// was a summary wire call (the compaction module wraps summarizer input in
/// a `<conversation>` block).
#[derive(Debug, Clone)]
struct CapturedRequest {
    text: String,
    is_summary: bool,
}

/// How the fixture model answers agent turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerPolicy {
    /// Plain non-terminal text: the RLM child machinery queues a
    /// continuation after it.
    Ack,
    /// Terminal child answer: the visible work, then the terminal line the
    /// continuation classifier requires on its own last line.
    TerminalComplete,
}

fn fixture_answer(policy: AnswerPolicy) -> String {
    match policy {
        AnswerPolicy::Ack => "ack: work started.".to_string(),
        AnswerPolicy::TerminalComplete => {
            "Section 3 of the migration is fixed; the payments ledger stayed intact.\nRLM_CHILD_STATUS: complete".to_string()
        }
    }
}

/// The summary body each summary wire call returns. Anchors are echoed from
/// the request text (which includes the `<previous-summary>` of earlier
/// chunks, so chained chunks re-anchor) so the compacted context demonstrably
/// retains the goal, the constraint and the evidence reference.
fn summary_text(request_text: &str) -> String {
    if request_text.contains("This is the PREFIX of a turn") {
        return format!(
            "## Original Request\n{}\n## Early Progress\n- Section 1 audited.\n## Context for Suffix\n- {}",
            anchor_lines(request_text).join("\n"),
            "reports/w8-audit.md stays available."
        );
    }
    let anchors = anchor_lines(request_text);
    format!(
        "## Goal\n{}\n## Constraints & Preferences\n{}\n## Progress\n### Done\n- [x] Section 1 audited.\n\n### In Progress\n- [ ] Section 3.\n\n### Blocked\n- None.\n\n## Key Decisions\n- Incremental rollout.\n## Next Steps\n1. Publish the report.\n## Critical Context\n{} reports/w8-audit.md stays available.",
        anchors[0], anchors[1], anchors[2]
    )
}

/// The anchor lines each summary re-anchors on: the goal, the constraint
/// and the evidence reference, found verbatim in the request text.
fn anchor_lines(request_text: &str) -> Vec<String> {
    ["TASK GOAL:", "CONSTRAINT:", "Evidence reference:"]
        .iter()
        .map(|prefix| {
            request_text
                .lines()
                .find(|line| line.trim_start().starts_with(prefix))
                .map(|line| line.trim().to_string())
                .unwrap_or_else(|| "None.".to_string())
        })
        .collect()
}

struct E2eFixture {
    session: Arc<AgentSession>,
    _dir: tempfile::TempDir,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
}

/// One real `AgentSession` on the faux provider with a small context window,
/// compaction enabled, and a factory that answers agent turns (per policy)
/// and summary wire calls (echoing the anchors). Every provider request is
/// captured.
async fn fixture(
    label: &str,
    rlm_depth: i64,
    message_controller: Option<Arc<dyn AgentSessionMessageController>>,
    answer: AnswerPolicy,
) -> E2eFixture {
    let root = tempfile::Builder::new()
        .prefix(&format!("rot2-w8-{label}-{}", unique_tag()))
        .tempdir()
        .unwrap();
    let cwd = root.path().join("workspace");
    let agent_dir = root.path().join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let cwd_string = cwd.to_string_lossy().to_string();
    let agent_dir_string = agent_dir.to_string_lossy().to_string();

    let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
        provider: Some(format!("rot2-w8-{label}-{}", unique_tag())),
        models: Some(vec![FauxModelDefinition {
            id: "rot2-w8-model".into(),
            name: Some("ROT2 W8 fixture model".into()),
            reasoning: Some(false),
            input: Some(vec!["text".into()]),
            cost: None,
            context_window: Some(8_192.0),
            max_tokens: Some(8_192.0),
        }]),
        tokens_per_second: Some(0.0),
        ..Default::default()
    }));
    let model = provider.get_model();

    let requests: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let factory: FauxResponseFactory =
        Arc::new(move |context: &Context, _options, _state, _model: &Model| {
            let request_text = context
                .messages
                .iter()
                .map(|message| match message {
                    Message::User(user) => user.content.text(),
                    Message::Assistant(assistant) => assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            let is_summary = request_text.contains("<conversation>");
            let text = if is_summary {
                summary_text(&request_text)
            } else {
                fixture_answer(answer)
            };
            seen.lock().unwrap().push(CapturedRequest {
                text: request_text,
                is_summary,
            });
            Box::pin(async move {
                faux_assistant_message(
                    FauxAssistantContent::Text(text),
                    Some(FauxAssistantMessageOptions {
                        stop_reason: Some("stop".into()),
                        ..Default::default()
                    }),
                )
            })
        });
    // Every stream call pops one queued step; continuation chains make
    // several calls per session, so queue generously.
    let steps: Vec<FauxResponseStep> = (0..96)
        .map(|_| FauxResponseStep::Factory(factory.clone()))
        .collect();
    provider.set_responses(steps);

    let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
        serde_json::json!({
            "autoRefine": {"enabled": false},
            "retry": {"enabled": false},
            "compaction": {
                "enabled": true,
                "reserveTokens": 512.0,
                "keepRecentTokens": 5.0
            },
            "telemetryEnabled": false,
            "agentTracesEnabled": false,
        })
        .as_object()
        .unwrap()
        .clone(),
    )));

    let mut session_manager =
        SessionManager::create(&cwd_string, Some(&agent_dir_string)).unwrap();
    if rlm_depth > 0 {
        session_manager
            .new_session(Some(&NewSessionOptions {
                rlm_depth: Some(rlm_depth),
                ..Default::default()
            }))
            .unwrap();
    }
    let session_manager = Arc::new(Mutex::new(session_manager));

    let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
        AuthStorageData::new(),
        None,
    )));
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
        AuthStorage::in_memory(AuthStorageData::new(), None),
    )));
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
        session_manager: Arc::clone(&session_manager),
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model),
            rlm_depth: Some(rlm_depth),
            agent_message_controller: message_controller,
            no_tools: Some("all".to_string()),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .expect("agent session");
    E2eFixture {
        session: created.session,
        _dir: root,
        requests,
    }
}

async fn wait_until(
    description: &str,
    timeout: Duration,
    mut predicate: impl FnMut() -> bool,
) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if predicate() {
            return true;
        }
        if Instant::now() >= deadline {
            eprintln!("timed out waiting for {description}");
            return false;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The agent-turn (non-summary) requests captured so far.
fn agent_turn_requests(fixture: &E2eFixture) -> Vec<String> {
    fixture
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|request| !request.is_summary)
        .map(|request| request.text.clone())
        .collect()
}

fn summary_requests(fixture: &E2eFixture) -> Vec<String> {
    fixture
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|request| request.is_summary)
        .map(|request| request.text.clone())
        .collect()
}

fn has_compaction_entry(session: &Arc<AgentSession>) -> bool {
    session
        .session_manager
        .lock()
        .unwrap()
        .get_branch(None)
        .iter()
        .any(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
}

/// The last continuation-state ledger entry on the branch, parsed.
fn ledger_state(session: &Arc<AgentSession>) -> Option<RlmContinuationState> {
    let manager = session.session_manager.lock().unwrap();
    let branch = manager.get_branch(None);
    let ledger = branch.iter().rev().find(|entry| {
        entry.get("type").and_then(Value::as_str) == Some("custom")
            && entry.get("customType").and_then(Value::as_str)
                == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
    })?;
    parse_rlm_continuation_state(&ledger["data"])
}

fn live_context_is_summary_first(session: &Arc<AgentSession>) -> bool {
    matches!(
        session.agent.messages().first(),
        Some(AgentMessage::Custom(
            pi_agent_core::types::CustomAgentMessage::CompactionSummary { .. }
        ))
    )
}

/// Find the delivered `agent_message` custom entry in the session's built
/// context by its details id. Returns (entry id, from relationship, content
/// text). Entry ids and messages are parallel in the built context, so the
/// returned id is the durable branch id the context resolves.
fn delivered_agent_message(
    session: &Arc<AgentSession>,
    message_id: &str,
) -> Option<(String, String, String)> {
    let manager = session.session_manager.lock().unwrap();
    let context = manager.build_session_context_with_entry_ids(None);
    for (index, message) in context.messages.iter().enumerate() {
        if let AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom {
            custom_type,
            details,
            content,
            ..
        }) = message
        {
            if custom_type != "agent_message" {
                continue;
            }
            let details = details.as_ref()?;
            if details.get("id").and_then(Value::as_str) != Some(message_id) {
                continue;
            }
            let entry_id = context.entry_ids.get(index)?.clone();
            let relationship = details
                .get("fromRelationship")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let text = match content {
                CustomMessageContent::Text(text) => text.clone(),
                _ => String::new(),
            };
            return Some((entry_id, relationship, text));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// In-process agent-message routing (the daemon delivery surface, offline)
// ---------------------------------------------------------------------------

struct FamilyBusInner {
    parent: std::sync::OnceLock<Arc<AgentSession>>,
    child: std::sync::OnceLock<Arc<AgentSession>>,
    /// The spawn counter for parent -> child task ids.
    next_spawn_id: std::sync::atomic::AtomicUsize,
    /// The message id of the next child -> parent report.
    next_report_id: std::sync::atomic::AtomicUsize,
    /// Successfully delivered sends.
    receipts: Mutex<Vec<DeliveredRecord>>,
}

/// One successful bus delivery, as the test needs it: the receiver, the
/// `agent_message` details id, and the receipt the sender got back.
#[derive(Clone)]
struct DeliveredRecord {
    receiver_id: String,
    message_id: String,
    receipt: AgentSessionMessageReceipt,
}

/// Routes exactly one parent and one child. `send_agent_message` delivers
/// into the target session through the same public entry point the daemon
/// uses (`accept_agent_message_prompt` with an `agent_message` custom
/// envelope), then returns a receipt whose id is the durable entry id of
/// the delivered message in the target's built context.
struct FamilyBus {
    inner: Arc<FamilyBusInner>,
}

impl FamilyBus {
    fn new() -> Self {
        Self {
            inner: Arc::new(FamilyBusInner {
                parent: std::sync::OnceLock::new(),
                child: std::sync::OnceLock::new(),
                next_spawn_id: std::sync::atomic::AtomicUsize::new(0),
                next_report_id: std::sync::atomic::AtomicUsize::new(0),
                receipts: Mutex::new(Vec::new()),
            }),
        }
    }

    fn bind(&self, parent: Arc<AgentSession>, child: Arc<AgentSession>) {
        assert!(self.inner.parent.set(parent).is_ok(), "parent bound once");
        assert!(self.inner.child.set(child).is_ok(), "child bound once");
    }

    fn receipts_for(&self, target: &str) -> Vec<DeliveredRecord> {
        self.inner
            .receipts
            .lock()
            .unwrap()
            .iter()
            .filter(|record| record.receiver_id == target)
            .cloned()
            .collect()
    }

    fn endpoint(session: &Arc<AgentSession>) -> AgentSessionMessageEndpoint {
        AgentSessionMessageEndpoint {
            session_id: session.session_id(),
            session_name: session.session_name(),
            active_session_id: session.session_id(),
            runtime_kind: Some("fixture".into()),
        }
    }

    fn sender(session: &Arc<AgentSession>) -> AgentSessionMessageSender {
        AgentSessionMessageSender {
            session_id: Some(session.session_id()),
            session_name: session.session_name(),
            active_session_id: Some(session.session_id()),
            client_id: None,
            runtime_kind: Some("fixture".into()),
        }
    }
}

impl FamilyBusInner {
    /// Deliver `message` into `target` as an `agent_message` custom envelope
    /// from `from`. Returns the durable entry id of the delivered message.
    async fn deliver(
        &self,
        target: &Arc<AgentSession>,
        from: &Arc<AgentSession>,
        message_id: &str,
        relationship: &str,
        message: &str,
    ) -> Result<String, String> {
        let custom = CustomMessage {
            role: "custom".into(),
            custom_type: "agent_message".into(),
            content: CustomMessageContent::Text(message.to_string()),
            display: true,
            details: Some(json!({
                "id": message_id,
                "message": message,
                "fromRelationship": relationship,
                "fromName": from.session_name().unwrap_or_else(|| from.session_id()),
            })),
            timestamp: 1_000,
        };
        target
            .accept_agent_message_prompt(
                message,
                Some(PromptOptions {
                    custom_message: Some(custom),
                    agent_message_id: Some(message_id.to_string()),
                    ..Default::default()
                }),
            )
            .await?;
        let (entry_id, delivered_relationship, delivered_text) =
            delivered_agent_message(target, message_id)
                .ok_or_else(|| "delivered entry not found in the target context".to_string())?;
        if delivered_relationship != relationship {
            return Err(format!(
                "delivered relationship {delivered_relationship} != {relationship}"
            ));
        }
        if delivered_text != message {
            return Err("delivered text differs from the sent message".to_string());
        }
        Ok(entry_id)
    }
}

impl AgentSessionMessageController for FamilyBus {
    fn roster(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentFamilyRosterResult, String>> + Send>,
    > {
        let parent = self
            .inner
            .parent
            .get()
            .expect("parent bound")
            .clone();
        let child = self.inner.child.get().expect("child bound").clone();
        Box::pin(async move {
            Ok(AgentFamilyRosterResult {
                current: AgentFamilyRosterCurrent {
                    id: child.session_id(),
                    name: child.session_name().unwrap_or_else(|| child.session_id()),
                    depth: 1.0,
                },
                entries: vec![
                    AgentFamilyRosterEntry {
                        id: parent.session_id(),
                        name: parent.session_name().unwrap_or_else(|| parent.session_id()),
                        relationship: FAMILY_RELATIONSHIP_PARENT.to_string(),
                        depth: 0.0,
                        status: "running".to_string(),
                        replied_since_task: None,
                    },
                    AgentFamilyRosterEntry {
                        id: child.session_id(),
                        name: child.session_name().unwrap_or_else(|| child.session_id()),
                        relationship: FAMILY_RELATIONSHIP_CHILD.to_string(),
                        depth: 1.0,
                        status: "running".to_string(),
                        replied_since_task: None,
                    },
                ],
            })
        })
    }

    fn await_pending_child_publication(
        &self,
        _child_id: String,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send>,
    > {
        // No daemon publication backlog in-process.
        Box::pin(async move { Ok(None) })
    }

    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentSessionMessageReceipt, String>> + Send>,
    > {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let parent = inner.parent.get().expect("parent bound").clone();
            let child = inner.child.get().expect("child bound").clone();
            let target = input.target;
            let message = input.message;
            let (receiver, sender, relationship, message_id) = if target == parent.session_id() {
                let report_id = format!(
                    "agentmsg-child-{}",
                    inner
                        .next_report_id
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                );
                (parent, child, FAMILY_RELATIONSHIP_CHILD, report_id)
            } else if target == child.session_id() {
                // The spawned-child durable shape (finding F1): the details
                // id carries the `spawn:` prefix.
                let spawn_id = format!(
                    "spawn:w8-child-{}",
                    inner
                        .next_spawn_id
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                );
                (child, parent, FAMILY_RELATIONSHIP_PARENT, spawn_id)
            } else {
                return Err(format!("unknown agent-message target {target}"));
            };
            let entry_id = inner
                .deliver(&receiver, &sender, &message_id, relationship, &message)
                .await?;
            // A direct-delivery receipt: the id is the durable entry id of
            // the delivered message on the receiver's branch. Finding F3:
            // the child's settlement does not acknowledge this receipt (the
            // private `rlm_result_receipt_is_valid` contract cannot be
            // satisfied from a test-side controller), so the child's task
            // keeps its durable delivery intent instead of flipping to
            // replied; see the scenario 3 test.
            let receipt = AgentSessionMessageReceipt {
                id: entry_id,
                source: sender.session_id(),
                target: FamilyBus::endpoint(&receiver),
                from: Some(FamilyBus::sender(&sender)),
                message: message.clone(),
                queued_at: None,
                delivered_at: Some("fixture-delivered".to_string()),
                delivery_mode: Some("direct".to_string()),
                delivery_status: "delivered".to_string(),
                delivered_to_hydrated: None,
            };
            inner
                .receipts
                .lock()
                .unwrap()
                .push(DeliveredRecord {
                    receiver_id: receiver.session_id(),
                    message_id,
                    receipt: receipt.clone(),
                });
            Ok(receipt)
        })
    }
}

// ---------------------------------------------------------------------------
// Scenario 1: parent with active constraints compacts, then hands a
// constraint-carrying task prompt to a child
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rot2_w8_parent_compaction_then_child_task_prompt_carries_constraints() {
    let bus = Arc::new(FamilyBus::new());
    let child = fixture("s1-child", 1, Some(bus.clone()), AnswerPolicy::Ack).await;
    let parent = fixture("s1-parent", 0, Some(bus.clone()), AnswerPolicy::Ack).await;
    bus.bind(parent.session.clone(), child.session.clone());
    assert_eq!(parent.session.rlm_depth(), 0);
    assert_eq!(child.session.rlm_depth(), 1);

    // Turn 1: the constraint enters the parent's context with enough bulk to
    // cross the compaction boundary.
    let prompt = format!(
        "{W8_GOAL}\n{W8_CONSTRAINT}\n{W8_EVIDENCE}\n{}",
        W8_FILLER.repeat(700)
    );
    parent
        .session
        .prompt_and_wait(&prompt, None)
        .await
        .expect("parent turn 1 accepted");

    let compacted = wait_until(
        "parent compaction entry",
        Duration::from_secs(60),
        || has_compaction_entry(&parent.session),
    )
    .await;
    assert!(compacted, "the parent never auto-compacted");

    // The compaction summary is the only carrier of the constraint now: the
    // summarizer saw it, and the live context is summary-first.
    let summaries = summary_requests(&parent);
    assert!(
        summaries.iter().any(|summary| summary.contains(W8_CONSTRAINT)),
        "the constraint never reached the summarizer input"
    );
    assert!(live_context_is_summary_first(&parent.session));

    // Turn 2 (post-compaction): the marker rides only in this prompt. The
    // captured request must carry the constraint through the summary while
    // the summarized filler is gone.
    parent
        .session
        .prompt_and_wait(
            &format!("Delegate section 3 of the migration to a child agent. {W8_PARENT_ONLY_MARKER}"),
            None,
        )
        .await
        .expect("parent turn 2 accepted");
    let delegated = wait_until(
        "post-compaction parent request",
        Duration::from_secs(30),
        || {
            agent_turn_requests(&parent)
                .iter()
                .any(|text| text.contains(W8_PARENT_ONLY_MARKER))
        },
    )
    .await;
    assert!(delegated, "the parent's post-compaction turn never ran");
    let post_compaction_request = agent_turn_requests(&parent)
        .into_iter()
        .rev()
        .find(|text| text.contains(W8_PARENT_ONLY_MARKER))
        .expect("captured post-compaction request");
    assert!(
        post_compaction_request.contains(W8_CONSTRAINT),
        "the parent's post-compaction context lost the constraint: the model has nothing to propagate"
    );
    assert!(
        !post_compaction_request.contains(&W8_FILLER.repeat(2)),
        "the summarized filler must be gone from the compacted context"
    );

    // The parent hands the task to the child through the public send path.
    // The task text restates the constraint, exactly as a parent model that
    // sees it in its compacted context would.
    let task_text = format!(
        "TASK GOAL: finish section 3 of the migration and report back.\n{W8_CONSTRAINT}\n{W8_EVIDENCE}"
    );
    let receipt = parent
        .session
        .send_agent_message(
            child.session.session_id(),
            task_text.clone(),
            Some("child".into()),
        )
        .await
        .expect("the parent's child-task send must succeed");
    assert_eq!(receipt.message, task_text);

    // The child's first model request carries the task prompt with the
    // constraint, delivered verbatim through the real delivery pipeline.
    let child_prompted = wait_until(
        "child first request",
        Duration::from_secs(60),
        || {
            agent_turn_requests(&child)
                .iter()
                .any(|text| text.contains("finish section 3 of the migration"))
        },
    )
    .await;
    assert!(child_prompted, "the child never received the task prompt");
    let child_request = agent_turn_requests(&child)
        .into_iter()
        .find(|text| text.contains("finish section 3 of the migration"))
        .expect("captured child request");
    assert!(
        child_request.contains(W8_CONSTRAINT),
        "the delivered child task prompt lost the active constraint"
    );
    assert!(
        child_request.contains(W8_EVIDENCE),
        "the delivered child task prompt lost the evidence reference"
    );
    // Finding F2: nothing injects the parent's compacted context into the
    // child. The parent-only marker and the parent's summary body are both
    // absent from the child's request.
    assert!(
        !child_request.contains(W8_PARENT_ONLY_MARKER),
        "parent post-compaction context leaked into the child prompt"
    );
    assert!(
        !child_request.contains(W8_SUMMARY_SECTION),
        "the parent's compaction summary body leaked into the child prompt"
    );

    // The delivered task is registered in the child's continuation ledger
    // under its spawn-flavored id.
    let ledger_settled = wait_until(
        "child ledger with the spawn task",
        Duration::from_secs(60),
        || {
            ledger_state(&child.session)
                .map(|state| {
                    state.tasks.len() == 1 && state.tasks[0].id == "spawn:w8-child-0"
                })
                .unwrap_or(false)
        },
    )
    .await;
    assert!(
        ledger_settled,
        "the spawned task never registered in the child's continuation ledger"
    );

    // The receipt resolves on the child's branch: its id is the durable
    // entry id of the delivered task prompt.
    let (entry_id, relationship, text) = delivered_agent_message(&child.session, "spawn:w8-child-0")
        .expect("durable task entry");
    assert_eq!(entry_id, receipt.id);
    assert_eq!(relationship, FAMILY_RELATIONSHIP_PARENT);
    assert_eq!(text, task_text);
}

// ---------------------------------------------------------------------------
// Scenario 2: an agent_message-delivered parent task crosses the compaction
// boundary mid-child; the child's continuation keeps the constraint
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rot2_w8_agent_message_task_crosses_compaction_mid_child() {
    let child = fixture("s2-child", 1, None, AnswerPolicy::Ack).await;
    let session = &child.session;
    assert_eq!(session.rlm_depth(), 1);

    // The delivered message IS the custom agent-session message: the whole
    // oversized body goes into its content so the model context crosses the
    // compaction boundary.
    let task_text = format!(
        "{W8_GOAL}\n{W8_CONSTRAINT}\n{W8_EVIDENCE}\n{}",
        W8_CHILD_FILLER.repeat(700)
    );
    let custom = CustomMessage {
        role: "custom".into(),
        custom_type: "agent_message".into(),
        content: CustomMessageContent::Text(task_text.clone()),
        display: true,
        details: Some(json!({
            "id": "agentmsg-parent-1",
            "message": task_text,
            "fromRelationship": "parent",
            "fromName": "parent-session"
        })),
        timestamp: 1_000,
    };
    session
        .accept_agent_message_prompt(
            &task_text,
            Some(PromptOptions {
                custom_message: Some(custom),
                agent_message_id: Some("agentmsg-parent-1".into()),
                ..Default::default()
            }),
        )
        .await
        .expect("parent task delivered");

    // The child turn crosses the boundary, the threshold compaction runs,
    // and the rebuilt model context settles into its summary-first shape.
    // (The durable compaction entry lands at turn end; the context rebuild
    // follows it, so both conditions are polled together.)
    let compacted = wait_until(
        "child compaction entry and summary-first context",
        Duration::from_secs(90),
        || has_compaction_entry(session) && live_context_is_summary_first(session),
    )
    .await;
    assert!(compacted, "the child never auto-compacted mid-task");

    // The compaction summary retains the constraint.
    let summaries = summary_requests(&child);
    assert!(
        summaries.iter().any(|summary| summary.contains(W8_CONSTRAINT)),
        "the constraint never reached the summarizer input"
    );
    let live_summary = match session.agent.messages().first() {
        Some(AgentMessage::Custom(
            pi_agent_core::types::CustomAgentMessage::CompactionSummary { summary, .. },
        )) => summary.clone(),
        other => panic!(
            "child context must be summary-first, got {:?}",
            other.map(|message| message.role())
        ),
    };
    assert!(
        live_summary.contains(W8_CONSTRAINT),
        "the compacted context lost the parent-imposed constraint: {live_summary}"
    );

    // The child's continuation (the automatic recovery turn the RLM
    // machinery queues after a non-terminal answer) must still see the
    // constraint through the compaction summary.
    let continued = wait_until(
        "post-compaction continuation request",
        Duration::from_secs(90),
        || agent_turn_requests(&child).len() >= 2,
    )
    .await;
    assert!(
        continued,
        "the child's continuation never ran after compaction"
    );
    let continuation_request = agent_turn_requests(&child)
        .into_iter()
        .nth(1)
        .expect("the continuation request");
    assert!(
        continuation_request.contains(W8_CONSTRAINT),
        "the child's continuation lost the constraint after compaction"
    );
    assert!(
        !continuation_request.contains(&W8_CHILD_FILLER.repeat(2)),
        "the summarized task body must be gone; only the summary may carry the constraint"
    );

    // The continuation ledger survives the compaction with the parent task
    // still tracked (open, or replied with a recorded result).
    let ledger = wait_until(
        "child ledger after compaction",
        Duration::from_secs(60),
        || {
            ledger_state(session)
                .map(|state| {
                    state.tasks.len() == 1
                        && state.tasks[0].id == "agentmsg-parent-1"
                        && (state.tasks[0].replied == state.tasks[0].result.is_some()
                            || !state.tasks[0].replied)
                })
                .unwrap_or(false)
        },
    )
    .await;
    assert!(
        ledger,
        "the continuation ledger lost the parent task across the compaction"
    );
    let state = ledger_state(session).expect("ledger");
    assert_eq!(state.tasks[0].id, "agentmsg-parent-1");
    if state.tasks[0].replied {
        let result = state.tasks[0].result.as_ref().expect("recorded result");
        assert!(!result.text.trim().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Scenario 3: a child completes AFTER the parent compacted; the delivered
// result resolves against the parent's compacted context
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rot2_w8_child_result_after_parent_compaction_resolves_on_compacted_context() {
    let bus = Arc::new(FamilyBus::new());
    // The child answers terminal-complete so its result is recorded and
    // automatically delivered to the parent.
    let child = fixture(
        "s3-child",
        1,
        Some(bus.clone()),
        AnswerPolicy::TerminalComplete,
    )
    .await;
    let parent = fixture("s3-parent", 0, Some(bus.clone()), AnswerPolicy::Ack).await;
    bus.bind(parent.session.clone(), child.session.clone());

    // Parent turn 1 crosses the compaction boundary; the constraint survives
    // in the summary.
    let prompt = format!(
        "{W8_GOAL}\n{W8_CONSTRAINT}\n{W8_EVIDENCE}\n{}",
        W8_FILLER.repeat(700)
    );
    parent
        .session
        .prompt_and_wait(&prompt, None)
        .await
        .expect("parent turn 1 accepted");
    let compacted = wait_until(
        "parent compaction entry",
        Duration::from_secs(60),
        || has_compaction_entry(&parent.session),
    )
    .await;
    assert!(compacted, "the parent never auto-compacted");
    assert!(live_context_is_summary_first(&parent.session));

    // Snapshot the compacted context before the child reports back.
    let context_before = parent
        .session
        .session_manager
        .lock()
        .unwrap()
        .build_session_context_with_entry_ids(None);
    assert!(
        context_before
            .entry_ids
            .iter()
            .all(|id| parent
                .session
                .session_manager
                .lock()
                .unwrap()
                .get_entry(id)
                .is_some()),
        "pre-report context ids must be durable"
    );

    // The parent delegates AFTER the compaction; the child completes after
    // it. The task text restates the constraint from the parent's compacted
    // context (pinned in scenario 1).
    let task_text = format!(
        "TASK GOAL: finish section 3 of the migration and report back.\n{W8_CONSTRAINT}"
    );
    parent
        .session
        .send_agent_message(
            child.session.session_id(),
            task_text.clone(),
            Some("child".into()),
        )
        .await
        .expect("the parent's child-task send must succeed");

    // The child finishes, the automatic result delivery routes through the
    // bus into the parent, and the parent runs a turn over the report.
    let report_processed = wait_until(
        "parent processes the child report",
        Duration::from_secs(120),
        || {
            agent_turn_requests(&parent)
                .iter()
                .any(|text| text.contains("RLM child automatic result"))
        },
    )
    .await;
    assert!(
        report_processed,
        "the child's result never reached the parent's model context"
    );

    // The receipt the child's settlement consumed: its id is the durable
    // entry id of the delivered report on the parent's branch, and the entry
    // resolves with the exact delivered text.
    let receipts = bus.receipts_for(&parent.session.session_id());
    assert_eq!(receipts.len(), 1, "exactly one child-to-parent delivery");
    let delivered = receipts.into_iter().next().expect("receipt");
    let receipt = delivered.receipt;
    let (report_entry_id, report_relationship, report_text) =
        delivered_agent_message(&parent.session, &delivered.message_id)
            .expect("the receipt id resolves to a durable agent_message entry");
    assert_eq!(report_entry_id, receipt.id);
    assert_eq!(report_relationship, FAMILY_RELATIONSHIP_CHILD);
    assert_eq!(report_text, receipt.message);
    assert!(
        report_text.contains("RLM child automatic result"),
        "the delivered report must be the automatic result message"
    );
    assert!(
        report_text.contains("task_id: spawn:w8-child-0"),
        "the delivered report must name the spawned task"
    );

    // The report entry id resolves against the parent's COMPACTED context:
    // entry ids survive the compaction (the pre-report ids are unchanged and
    // still lead), and the delivered entry lands after the summary.
    let context_after = parent
        .session
        .session_manager
        .lock()
        .unwrap()
        .build_session_context_with_entry_ids(None);
    assert!(
        context_after.entry_ids.contains(&receipt.id),
        "the child report entry id must resolve in the parent's compacted context"
    );
    assert!(
        context_after.entry_ids.len() >= context_before.entry_ids.len(),
        "the compacted context must keep its pre-report entries"
    );
    assert_eq!(
        &context_after.entry_ids[..context_before.entry_ids.len()],
        &context_before.entry_ids[..],
        "the pre-report (post-compaction) entry ids must be unchanged"
    );
    assert!(live_context_is_summary_first(&parent.session));
    assert!(
        parent
            .session
            .session_manager
            .lock()
            .unwrap()
            .get_entry(&receipt.id)
            .is_some(),
        "the delivered report entry must stay durable"
    );

    // The parent's post-report model context carries the child's visible
    // result text and the constraint from the parent's own summary.
    let report_request = agent_turn_requests(&parent)
        .into_iter()
        .rev()
        .find(|text| text.contains("RLM child automatic result"))
        .expect("captured report request");
    assert!(
        report_request.contains("Section 3 of the migration is fixed"),
        "the parent's model context lost the child's visible result"
    );
    assert!(
        report_request.contains(W8_CONSTRAINT),
        "the parent's compacted context lost the constraint while processing the report"
    );

    // The child's ledger: the task carries the recorded result and a
    // durable delivery intent. Finding F3: the task does NOT flip to
    // replied here. The settlement's receipt acknowledgement runs through
    // the private `rlm_result_receipt_is_valid`, whose contract is not
    // satisfiable from a test-side controller (32 receipt shapes were
    // tried: delivered/queued/acknowledged statuses, direct/steer/followUp/
    // queued/absent modes, RFC3339 and plain timestamps, session names set
    // and unset, UUID and entry ids, several runtime kinds - all rejected
    // as "invalid_receipt"). The real contract lives in the daemon's
    // controller (crate::core::agent_messages, outside this brief's file
    // map), so the acknowledged settlement cannot be driven offline. What
    // the durable state does prove: the result is recorded exactly once and
    // the delivery intent is persisted, which is the no-replay evidence the
    // interrupted-delivery paths rely on.
    let state = wait_until(
        "child ledger with a recorded result and delivery intent",
        Duration::from_secs(60),
        || {
            ledger_state(&child.session)
                .map(|state| {
                    state.tasks.len() == 1
                        && state.tasks[0].id == "spawn:w8-child-0"
                        && state.tasks[0].result.is_some()
                        && state.tasks[0].result_delivery.is_some()
                })
                .unwrap_or(false)
        },
    )
    .await;
    assert!(
        state,
        "the child's task never recorded its result and delivery intent"
    );
    let state = ledger_state(&child.session).expect("ledger");
    let result = state.tasks[0].result.as_ref().expect("recorded result");
    assert_eq!(result.status, "complete");
    assert!(
        result.text.contains("Section 3 of the migration is fixed"),
        "the recorded result lost the visible text: {}",
        result.text
    );
    let delivery = state.tasks[0].result_delivery.as_ref().expect("delivery intent");
    assert!(
        !delivery.attempt_id.is_empty(),
        "the durable delivery intent must carry its attempt id"
    );
}
