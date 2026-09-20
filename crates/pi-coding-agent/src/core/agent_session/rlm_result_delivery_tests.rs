//! B2 regressions: no network, daemon, provider, kernel, or live child messages.
use super::*;
use crate::core::agent_messages::{
    AgentFamilyRosterEntry, AgentFamilyRosterResult, AgentSessionMessageController,
    AgentSessionMessageEndpoint, AgentSessionMessageSendInput,
};
use crate::core::session_manager::SessionManager;
use pi_agent_core::types::{ShouldStopAfterTurnContext, StreamFn};
use pi_ai::types::{OnPayload, OnResponse};
use tokio::sync::oneshot;

type Listener = Arc<dyn Fn(AgentEvent, Option<CancellationToken>) -> BoxFuture<()> + Send + Sync>;

/// Minimal scripted agent: idle, no queue, `continue()` counted.
///
/// A plain turn is never dispatched by this suite. `prompt` rejects so that an
/// accidental dispatch is loud instead of silently completing.
struct ScriptedAgent {
    state: Mutex<AgentState>,
    continue_calls: std::sync::atomic::AtomicUsize,
    stream_fn: Mutex<StreamFn>,
}

impl ScriptedAgent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(AgentState::default()),
            continue_calls: std::sync::atomic::AtomicUsize::new(0),
            stream_fn: Mutex::new(pi_agent_core::agent::default_stream_fn()),
        })
    }
}

impl AgentHandle for ScriptedAgent {
    fn state(&self) -> AgentState {
        self.state.lock().unwrap().clone()
    }

    fn set_state(&self, state: AgentState) {
        *self.state.lock().unwrap() = state;
    }

    fn subscribe(&self, _listener: Listener) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn set_before_tool_call(&self, _hook: BeforeToolCallHook) {}
    fn set_after_tool_call(&self, _hook: AfterToolCallHook) {}
    fn set_get_continuation_messages(&self, _hook: GetContinuationMessagesHook) {}
    fn set_should_stop_before_turn(&self, _hook: Arc<dyn Fn() -> bool + Send + Sync>) {}
    fn set_should_stop_after_turn(
        &self,
        _hook: Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxFuture<bool> + Send + Sync>,
    ) {
    }

    fn set_stream_fn(&self, stream_fn: StreamFn) {
        *self.stream_fn.lock().unwrap() = stream_fn;
    }

    fn stream_fn(&self) -> StreamFn {
        self.stream_fn.lock().unwrap().clone()
    }

    fn abort(&self) {}

    fn wait_for_idle(&self) -> BoxFuture<()> {
        Box::pin(async {})
    }

    fn prompt(&self, _messages: Vec<AgentMessage>) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Err("scripted agent does not run turns".to_string()) })
    }

    fn continue_(&self) -> BoxFuture<Result<(), AgentContinueError>> {
        self.continue_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn is_streaming(&self) -> bool {
        self.state.lock().unwrap().is_streaming
    }

    fn has_queued_messages(&self) -> bool {
        false
    }

    fn clear_all_queues(&self) {}

    fn remove_queued_messages(
        &self,
        _predicate: Arc<dyn Fn(&AgentMessage) -> bool + Send + Sync>,
    ) -> Vec<AgentMessage> {
        Vec::new()
    }

    fn follow_up(&self, _message: AgentMessage) {}
    fn set_follow_up_mode(&self, _mode: String) {}
    fn set_steering_mode(&self, _mode: String) {}

    fn set_convert_to_llm(
        &self,
        _convert: Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<Message>> + Send + Sync>,
    ) {
    }

    fn set_transform_context(
        &self,
        _transform: Arc<
            dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    ) {
    }

    fn set_get_api_key(
        &self,
        _get_api_key: Arc<dyn Fn(String) -> BoxFuture<Option<String>> + Send + Sync>,
    ) {
    }

    fn set_on_payload(&self, _hook: OnPayload) {}
    fn set_on_response(&self, _hook: OnResponse) {}
    fn set_tool_execution(&self, _mode: String) {}
    fn performance_metrics(&self) -> Option<AgentLoopPerformanceMetrics> {
        None
    }
    fn set_performance_metrics(&self, _metrics: Option<AgentLoopPerformanceMetrics>) {}
    fn signal(&self) -> Option<CancellationToken> {
        None
    }
}

fn session_with_manager(
    root: &Path,
    manager: SessionManager,
    controller: Arc<Controller>,
) -> Arc<AgentSession> {
    let cwd = root.to_string_lossy().to_string();
    let settings = Arc::new(Mutex::new(
        crate::core::settings_manager::SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    ));
    let loader = Arc::new(crate::core::resource_loader::DefaultResourceLoader::new(
        crate::core::resource_loader::DefaultResourceLoaderOptions {
            cwd: cwd.clone(),
            agent_dir: cwd.clone(),
            settings_manager: Some(settings.clone()),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        },
    ));
    AgentSession::new(AgentSessionConfig {
        agent: ScriptedAgent::new() as Arc<dyn AgentHandle>,
        session_manager: Arc::new(Mutex::new(manager)),
        settings_manager: settings,
        service_tier_preference: None,
        cwd: cwd.clone(),
        agent_dir: Some(cwd.clone()),
        scoped_models: None,
        resource_loader: loader,
        custom_tools: None,
        model_registry: Arc::new(Mutex::new(
            crate::core::model_registry::ModelRegistry::in_memory(
                crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
            ),
        )),
        initial_active_tool_names: None,
        allowed_tool_names: None,
        include_goals: Some(false),
        agent_message_controller: Some(controller),
        agent_observe_controller: None,
        include_compact_skill: Some(false),
        rlm_heartbeat_controller: None,
        mcp_manager: None,
        base_tools_override: Some(Vec::new()),
        extension_runner_ref: None,
        session_start_event: None,
        rlm_depth: Some(1),
        rlm_max_depth: Some(2),
        rlm_session_dir: None,
        rlm_parent_node_id: None,
        rlm_parent_agent: None,
        semantic_parent_session_id: None,
        semantic_spawned_by_request_id: None,
        subagent_runtime_host: None,
        autonomous: None,
        prewarm_ipython_kernel: Some(false),
        auto_refine_reviewer: None,
        serialized_refine: None,
        initial_goal: None,
    })
    .unwrap()
}

#[derive(Clone, Copy)]
enum Reply {
    Status(&'static str),
    WrongTarget,
    MissingTimestamp,
    Error(&'static str),
}

struct Plan {
    reply: Reply,
    started: Option<oneshot::Sender<()>>,
    release: Option<oneshot::Receiver<()>>,
}

#[derive(Default)]
struct Controller {
    roster_errors: Mutex<VecDeque<String>>,
    plans: Mutex<VecDeque<Plan>>,
    calls: Arc<Mutex<Vec<AgentSessionMessageSendInput>>>,
    journal: Mutex<Option<PathBuf>>,
}

impl Controller {
    fn push(&self, reply: Reply) {
        self.plans.lock().unwrap().push_back(Plan {
            reply,
            started: None,
            release: None,
        });
    }

    fn pause(&self, reply: Reply) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        self.plans.lock().unwrap().push_back(Plan {
            reply,
            started: Some(started_tx),
            release: Some(release_rx),
        });
        (started_rx, release_tx)
    }

    fn len(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

fn saved_ledger(path: &Path) -> RlmContinuationState {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .rev()
        .find_map(|line| {
            let entry: Value = serde_json::from_str(line).unwrap();
            (entry["customType"] == RLM_CONTINUATION_STATE_CUSTOM_TYPE)
                .then(|| parse_rlm_continuation_state(&entry["data"]).unwrap())
        })
        .expect("saved continuation ledger")
}

impl AgentSessionMessageController for Controller {
    fn roster(&self) -> BoxFuture<Result<AgentFamilyRosterResult, String>> {
        let error = self.roster_errors.lock().unwrap().pop_front();
        Box::pin(async move {
            if let Some(error) = error {
                return Err(error);
            }
            Ok(AgentFamilyRosterResult {
                entries: vec![
                    AgentFamilyRosterEntry {
                        id: "parent".into(),
                        name: "parent-name".into(),
                        relationship: "parent".into(),
                        ..Default::default()
                    },
                    AgentFamilyRosterEntry {
                        id: "sibling".into(),
                        name: "sibling-name".into(),
                        relationship: "sibling".into(),
                        ..Default::default()
                    },
                    AgentFamilyRosterEntry {
                        id: "child".into(),
                        name: "child-name".into(),
                        relationship: "child".into(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            })
        })
    }

    fn await_pending_child_publication(
        &self,
        _: String,
    ) -> BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let plan = self
            .plans
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected duplicate send");
        let calls = self.calls.clone();
        let journal = self.journal.lock().unwrap().clone();
        Box::pin(async move {
            if let Some(path) = journal {
                let saved = saved_ledger(&path);
                assert!(
                    saved
                        .tasks
                        .iter()
                        .any(|task| task.result_delivery.is_some() || task.replied),
                    "the intent must reach the actual journal before the external call"
                );
            }
            calls.lock().unwrap().push(input.clone());
            if let Some(started) = plan.started {
                started.send(()).unwrap();
            }
            if let Some(release) = plan.release {
                release.await.unwrap();
            }
            let status = match plan.reply {
                Reply::Error(error) => return Err(error.to_string()),
                Reply::Status(status) => status,
                _ => "queued",
            };
            let at = "2026-09-20T00:00:00Z".to_string();
            Ok(AgentSessionMessageReceipt {
                id: format!("agentmsg_{}", uuid::Uuid::new_v4()),
                source: "agent_message".into(),
                target: AgentSessionMessageEndpoint {
                    session_id: if matches!(plan.reply, Reply::WrongTarget) {
                        "other".into()
                    } else {
                        "parent".into()
                    },
                    active_session_id: "parent-active".into(),
                    session_name: Some("parent-name".into()),
                    ..Default::default()
                },
                message: input.message.trim().to_string(),
                delivery_status: status.into(),
                queued_at: (status == "queued" && !matches!(plan.reply, Reply::MissingTimestamp))
                    .then_some(at.clone()),
                delivered_at: (status == "delivered").then_some(at),
                ..Default::default()
            })
        })
    }
}

struct Fixture {
    root: tempfile::TempDir,
    session: Arc<AgentSession>,
    controller: Arc<Controller>,
    journal: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let cwd = root.path().to_string_lossy();
        let manager = SessionManager::create(&cwd, Some(&cwd)).unwrap();
        let journal = PathBuf::from(manager.get_session_file().unwrap());
        let controller = Arc::new(Controller::default());
        *controller.journal.lock().unwrap() = Some(journal.clone());
        let session = session_with_manager(root.path(), manager, controller.clone());
        seed(&session, "task-1");
        Self {
            root,
            session,
            controller,
            journal,
        }
    }

    fn reload(&self) -> Arc<AgentSession> {
        let manager = SessionManager::open(self.journal.to_str().unwrap(), None, None).unwrap();
        session_with_manager(self.root.path(), manager, self.controller.clone())
    }
}

fn seed(session: &Arc<AgentSession>, id: &str) {
    let mut state = session.rlm_continuation.lock().unwrap();
    state.tasks.push(RlmParentTask {
        id: id.into(),
        received_at: 1.0,
        replied: false,
        result: Some(RlmPendingResult {
            status: "complete".into(),
            text: "visible result only".into(),
            ..Default::default()
        }),
        result_delivery: None,
    });
}

fn task(session: &Arc<AgentSession>) -> RlmParentTask {
    session.rlm_continuation.lock().unwrap().tasks[0].clone()
}

fn count(session: &Arc<AgentSession>) -> u64 {
    session.parent_reply_count.load(Ordering::SeqCst)
}

fn explicit(
    session: &Arc<AgentSession>,
    text: &str,
) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
    session.send_agent_message("parent".into(), text.into(), Some("parent".into()))
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(std::time::Duration::from_secs(3), future)
        .await
        .expect("no gate deadlock")
}

#[tokio::test]
async fn rejected_automatic_result_retries_only_after_known_pre_admission_failure() {
    for error in [
        "Agent messaging is paused",
        "Agent messaging rate limit exceeded; retry after 7ms",
    ] {
        let f = Fixture::new();
        f.controller.push(Reply::Error(error));
        f.session.deliver_pending_rlm_results_once().await;
        assert!(!task(&f.session).replied);
        assert!(task(&f.session).result_delivery.is_none());
        assert!(task(&f.session).result.is_some());
        assert_eq!(count(&f.session), 0);
        f.controller.push(Reply::Status("queued"));
        f.session.deliver_pending_rlm_results_once().await;
        f.session.deliver_pending_rlm_results_once().await;
        assert!(task(&f.session).replied);
        assert_eq!(f.controller.len(), 2);
        assert_eq!(count(&f.session), 1);
    }
}

#[tokio::test]
async fn acknowledged_automatic_result_is_not_repeated_even_after_reload() {
    for status in ["queued", "delivered"] {
        let f = Fixture::new();
        f.controller.push(Reply::Status(status));
        f.session.deliver_pending_rlm_results_once().await;
        f.session.deliver_pending_rlm_results_once().await;
        assert_eq!(count(&f.session), 1);
        assert!(task(&f.session).replied);
        let reloaded = f.reload();
        reloaded.deliver_pending_rlm_results_once().await;
        assert!(task(&reloaded).replied);
        assert_eq!(f.controller.len(), 1);
    }
}

#[tokio::test]
async fn uncertain_and_invalid_automatic_receipts_retain_evidence_without_reload_replay() {
    for reply in [
        Reply::Error("transport disconnected"),
        Reply::Status("unknown"),
        Reply::WrongTarget,
        Reply::MissingTimestamp,
    ] {
        let f = Fixture::new();
        f.controller.push(reply);
        f.session.deliver_pending_rlm_results_once().await;
        assert!(!task(&f.session).replied);
        assert_eq!(count(&f.session), 0);
        assert!(task(&f.session).result_delivery.is_some());
        assert_eq!(
            f.session.rlm_diagnostics().unwrap()["diagnosticState"],
            "rlm_result_acknowledgement_uncertain"
        );
        let reloaded = f.reload();
        assert_eq!(task(&f.session), task(&reloaded));
        reloaded.deliver_pending_rlm_results_once().await;
        assert_eq!(f.controller.len(), 1);
        assert_eq!(task(&reloaded).result.unwrap().text, "visible result only");
        assert!(!std::fs::read_to_string(&f.journal)
            .unwrap()
            .contains("private reasoning"));
    }
}

#[tokio::test]
async fn explicit_reply_wins_while_automatic_waits_without_gate_deadlock() {
    let f = Fixture::new();
    let (started, release) = f.controller.pause(Reply::Status("delivered"));
    let reply = explicit(&f.session, "deliberate answer");
    let automatic = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    bounded(started).await.unwrap();
    release.send(()).unwrap();
    bounded(automatic).await.unwrap();
    bounded(reply).await.unwrap();
    assert_eq!(f.controller.len(), 1);
    assert_eq!(count(&f.session), 1);
    assert!(task(&f.session).replied);
    assert!(!f.controller.calls.lock().unwrap()[0]
        .message
        .contains("automatic result"));
}

#[tokio::test]
async fn automatic_claim_wins_but_later_deliberate_explicit_messages_are_not_suppressed() {
    let f = Fixture::new();
    let (started, release) = f.controller.pause(Reply::Status("queued"));
    let automatic = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    bounded(started).await.unwrap();
    f.controller.push(Reply::Status("delivered"));
    let reply = explicit(&f.session, "additional deliberate answer");
    release.send(()).unwrap();
    let (automatic, reply) = bounded(async { tokio::join!(automatic, reply) }).await;
    automatic.unwrap();
    reply.unwrap();
    f.controller.push(Reply::Status("queued"));
    explicit(&f.session, "a second distinct explicit message")
        .await
        .unwrap();
    f.session.deliver_pending_rlm_results_once().await;
    assert_eq!(f.controller.len(), 3);
    assert_eq!(count(&f.session), 3);
    assert_eq!(
        f.controller
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.message.contains("automatic result"))
            .count(),
        1
    );
}

#[tokio::test]
async fn concurrent_automatic_passes_claim_once() {
    let f = Fixture::new();
    let (started, release) = f.controller.pause(Reply::Status("queued"));
    let first = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    bounded(started).await.unwrap();
    let second = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    release.send(()).unwrap();
    let (first, second) = bounded(async { tokio::join!(first, second) }).await;
    first.unwrap();
    second.unwrap();
    assert_eq!(count(&f.session), 1);
    assert_eq!(f.controller.len(), 1);
}

fn obstruct_journal(path: &Path) {
    if path.exists() {
        std::fs::remove_file(path).unwrap();
    }
    std::fs::create_dir(path).unwrap();
}

#[tokio::test]
async fn intent_persistence_failure_prevents_automatic_and_correlated_explicit_send() {
    for use_explicit in [false, true] {
        let f = Fixture::new();
        obstruct_journal(&f.journal);
        if use_explicit {
            assert!(explicit(&f.session, "answer").await.is_err());
        } else {
            f.session.deliver_pending_rlm_results_once().await;
        }
        assert_eq!(f.controller.len(), 0);
        assert_eq!(count(&f.session), 0);
        assert!(!task(&f.session).replied);
        assert!(task(&f.session).result.is_some());
    }
}

#[tokio::test]
async fn nonpersistent_correlated_task_fails_closed_but_uncorrelated_explicit_still_sends() {
    let root = tempfile::tempdir().unwrap();
    let manager = SessionManager::in_memory(Some(root.path().to_str().unwrap()), None).unwrap();
    let controller = Arc::new(Controller::default());
    let session = session_with_manager(root.path(), manager, controller.clone());
    controller.push(Reply::Status("queued"));
    explicit(&session, "uncorrelated message").await.unwrap();
    assert_eq!(count(&session), 1);
    seed(&session, "task-1");
    let error = explicit(&session, "correlated answer").await.unwrap_err();
    assert!(error.contains("persistent session journal"), "{error}");
    session.deliver_pending_rlm_results_once().await;
    assert_eq!(controller.len(), 1);
    assert_eq!(count(&session), 1);
}

#[tokio::test]
async fn lazy_persistent_journal_is_materialized_before_send() {
    let f = Fixture::new();
    if f.journal.exists() {
        std::fs::remove_file(&f.journal).unwrap();
    }
    f.controller.push(Reply::Status("queued"));
    f.session.deliver_pending_rlm_results_once().await;
    assert!(saved_ledger(&f.journal).tasks[0].replied);
    assert_eq!(f.controller.len(), 1);
}

#[tokio::test]
async fn interrupted_submission_keeps_durable_intent_without_reload_replay() {
    let f = Fixture::new();
    let (started, _release) = f.controller.pause(Reply::Status("queued"));
    let automatic = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    bounded(started).await.unwrap();
    automatic.abort();
    assert!(automatic.await.unwrap_err().is_cancelled());
    assert_eq!(count(&f.session), 0);
    let reloaded = f.reload();
    reloaded.deliver_pending_rlm_results_once().await;
    assert!(!task(&reloaded).replied);
    assert!(task(&reloaded).result_delivery.is_some());
    assert_eq!(f.controller.len(), 1);
}

#[tokio::test]
async fn disposal_after_durable_claim_does_not_send_or_acknowledge() {
    let f = Fixture::new();
    let weak = Arc::downgrade(&f.session);
    let unsubscribe = f
        .session
        .session_manager
        .lock()
        .unwrap()
        .on_persist(Box::new(move |_| {
            if let Some(session) = weak.upgrade() {
                session.disposing.store(true, Ordering::SeqCst);
            }
        }));
    f.session.deliver_pending_rlm_results_once().await;
    unsubscribe();
    assert_eq!(f.controller.len(), 0);
    assert_eq!(count(&f.session), 0);
    assert!(!saved_ledger(&f.journal).tasks[0].replied);
    let reloaded = f.reload();
    reloaded.deliver_pending_rlm_results_once().await;
    assert_eq!(f.controller.len(), 0);
}

#[tokio::test]
async fn acknowledgement_settlement_failure_retains_durable_no_replay_evidence() {
    let f = Fixture::new();
    let (started, release) = f.controller.pause(Reply::Status("queued"));
    let automatic = tokio::spawn({
        let session = f.session.clone();
        async move { session.deliver_pending_rlm_results_once().await }
    });
    bounded(started).await.unwrap();
    let durable_intent = std::fs::read(&f.journal).unwrap();
    obstruct_journal(&f.journal);
    release.send(()).unwrap();
    bounded(automatic).await.unwrap();
    assert!(task(&f.session).replied);
    assert_eq!(count(&f.session), 1);
    std::fs::remove_dir(&f.journal).unwrap();
    std::fs::write(&f.journal, durable_intent).unwrap();
    let reloaded = f.reload();
    assert!(!task(&reloaded).replied);
    reloaded.deliver_pending_rlm_results_once().await;
    assert_eq!(f.controller.len(), 1);
}

#[tokio::test]
async fn ambiguous_explicit_reply_blocks_fallback_and_later_success_reconciles() {
    for reply in [Reply::Error("worker reply lost"), Reply::Status("invalid")] {
        let f = Fixture::new();
        f.controller.push(reply);
        let _ = explicit(&f.session, "deliberate answer").await;
        assert!(!task(&f.session).replied);
        assert_eq!(count(&f.session), 0);
        f.session.deliver_pending_rlm_results_once().await;
        let reloaded = f.reload();
        reloaded.deliver_pending_rlm_results_once().await;
        assert_eq!(f.controller.len(), 1);
        // A known rejection must not erase an older unknown acknowledgement.
        f.controller.push(Reply::Error("Agent messaging is paused"));
        assert!(explicit(&reloaded, "deliberate reconciliation")
            .await
            .is_err());
        reloaded.deliver_pending_rlm_results_once().await;
        assert!(task(&reloaded).result_delivery.is_some());
        f.controller.push(Reply::Status("delivered"));
        explicit(&reloaded, "deliberate reconciliation")
            .await
            .unwrap();
        assert!(task(&reloaded).replied);
        assert!(task(&reloaded).result_delivery.is_none());
        assert_eq!(count(&reloaded), 1);
        reloaded.deliver_pending_rlm_results_once().await;
        assert_eq!(f.controller.len(), 3);
    }
}

#[tokio::test]
async fn preadmission_explicit_rejection_allows_one_automatic_fallback() {
    let f = Fixture::new();
    f.controller.push(Reply::Error("Agent messaging is paused"));
    assert!(explicit(&f.session, "answer").await.is_err());
    assert!(task(&f.session).result_delivery.is_none());
    f.controller.push(Reply::Status("queued"));
    f.session.deliver_pending_rlm_results_once().await;
    f.session.deliver_pending_rlm_results_once().await;
    assert_eq!(count(&f.session), 1);
    assert_eq!(f.controller.len(), 2);
}

#[tokio::test]
async fn follow_up_has_distinct_identity_and_preserves_old_uncertain_result() {
    let f = Fixture::new();
    f.controller.push(Reply::Error("unknown acknowledgement"));
    f.session.deliver_pending_rlm_results_once().await;
    let old = task(&f.session);
    let message = crate::core::agent_messages::create_agent_session_message(
        &crate::core::agent_messages::AgentSessionMessagePayload {
            id: "task-2".into(),
            source: "agent_message".into(),
            message: "follow up".into(),
            from_relationship: Some("parent".into()),
            ..Default::default()
        },
        2,
    );
    f.session
        .begin_rlm_parent_task(&AgentMessage::Custom(message))
        .unwrap();
    f.session.record_rlm_terminal_result(RlmPendingResult {
        status: "complete".into(),
        text: "new visible result".into(),
        ..Default::default()
    });
    f.controller.push(Reply::Status("queued"));
    f.session.deliver_pending_rlm_results_once().await;
    assert_eq!(task(&f.session), old);
    let state = f.session.rlm_continuation.lock().unwrap();
    assert_eq!(state.tasks[1].id, "task-2");
    assert!(state.tasks[1].replied);
    assert_eq!(
        state.tasks[1].result.as_ref().unwrap().text,
        "new visible result"
    );
    assert_eq!(f.controller.len(), 2);
}

#[tokio::test]
async fn later_explicit_success_reconciles_automatic_uncertainty_without_reasoning_disclosure() {
    let f = Fixture::new();
    f.session.rlm_continuation.lock().unwrap().tasks[0].result = None;
    let assistant = AssistantMessage {
        content: vec![
            pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::new(
                "private reasoning sentinel",
            )),
            pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(
                "visible answer\nRLM_CHILD_STATUS: complete",
            )),
        ],
        stop_reason: "stop".into(),
        timestamp: 3,
        ..Default::default()
    };
    f.session
        .handle_rlm_child_turn_outcome(&assistant, false, None)
        .unwrap();
    f.controller.push(Reply::Error("response lost"));
    f.session.deliver_pending_rlm_results_once().await;
    let reloaded = f.reload();
    assert!(task(&reloaded).result_delivery.is_some());
    f.controller.push(Reply::Status("delivered"));
    explicit(&reloaded, "visible answer").await.unwrap();
    assert!(task(&reloaded).replied);
    assert!(task(&reloaded).result_delivery.is_none());
    assert_eq!(count(&reloaded), 1);
    reloaded.deliver_pending_rlm_results_once().await;
    assert_eq!(f.controller.len(), 2);
    assert!(f
        .controller
        .calls
        .lock()
        .unwrap()
        .iter()
        .all(|call| !call.message.contains("private reasoning sentinel")));
    assert!(!std::fs::read_to_string(&f.journal)
        .unwrap()
        .contains("private reasoning sentinel"));
}

#[tokio::test]
async fn failed_parent_resolution_cannot_bypass_intent_for_a_correlated_explicit_reply() {
    let f = Fixture::new();
    f.controller
        .roster_errors
        .lock()
        .unwrap()
        .push_back("transient roster failure".into());
    f.controller.push(Reply::Status("delivered"));
    let error = f
        .session
        .send_agent_message("parent".into(), "answer".into(), None)
        .await
        .unwrap_err();
    assert!(error.contains("not submitted"), "{error}");
    assert_eq!(f.controller.len(), 0);
    assert_eq!(count(&f.session), 0);
    assert!(!task(&f.session).replied);
    assert!(task(&f.session).result_delivery.is_none());
    // The unconsumed valid receipt belongs to the only actual submission.
    f.session.deliver_pending_rlm_results_once().await;
    f.session.deliver_pending_rlm_results_once().await;
    assert_eq!(f.controller.len(), 1);
    assert_eq!(count(&f.session), 1);
}

#[tokio::test]
async fn padded_parent_id_and_name_are_normalized_before_relation_and_receipt_validation() {
    for target in [" parent ", " parent-name "] {
        let f = Fixture::new();
        f.controller.push(Reply::Status("queued"));
        f.session
            .send_agent_message(target.into(), "answer".into(), None)
            .await
            .unwrap();
        assert!(task(&f.session).replied);
        assert!(task(&f.session).result_delivery.is_none());
        assert_eq!(count(&f.session), 1);
        f.session.deliver_pending_rlm_results_once().await;
        assert_eq!(f.controller.len(), 1);
        assert_eq!(f.controller.calls.lock().unwrap()[0].target, target.trim());
    }
}

#[tokio::test]
async fn unresolved_direct_alias_cannot_bypass_intent_for_a_correlated_explicit_reply() {
    for target in ["parent-active", "rent"] {
        let f = Fixture::new();
        f.controller.push(Reply::Status("delivered"));
        let error = f
            .session
            .send_agent_message(target.into(), "answer".into(), None)
            .await
            .unwrap_err();
        assert!(error.contains("not submitted"), "{error}");
        assert!(error.contains("receiver_role=parent"), "{error}");
        assert!(error.contains("canonical roster ID/name"), "{error}");
        assert_eq!(f.controller.len(), 0);
        assert_eq!(count(&f.session), 0);
        assert!(!task(&f.session).replied);
        assert!(task(&f.session).result_delivery.is_none());
        f.session.deliver_pending_rlm_results_once().await;
        assert_eq!(f.controller.len(), 1);
        assert_eq!(count(&f.session), 1);
    }
}

#[tokio::test]
async fn nonparent_roles_known_roster_routes_and_uncorrelated_aliases_still_send() {
    for (target, role) in [
        ("sibling", None),
        ("sibling-name", None),
        ("child", None),
        ("child-name", None),
        ("unresolved-sibling-alias", Some("sibling")),
        ("unresolved-child-alias", Some("child")),
    ] {
        let f = Fixture::new();
        *f.controller.journal.lock().unwrap() = None;
        f.controller.push(Reply::Status("queued"));
        f.session
            .send_agent_message(
                target.into(),
                "not a parent answer".into(),
                role.map(str::to_string),
            )
            .await
            .unwrap();
        assert_eq!(f.controller.len(), 1);
        assert_eq!(f.controller.calls.lock().unwrap()[0].target, target);
        assert_eq!(count(&f.session), 0);
        assert!(!task(&f.session).replied);
        assert!(task(&f.session).result_delivery.is_none());
    }
    let f = Fixture::new();
    f.session.rlm_continuation.lock().unwrap().tasks.clear();
    *f.controller.journal.lock().unwrap() = None;
    f.controller.push(Reply::Status("queued"));
    f.session
        .send_agent_message("uncorrelated-alias".into(), "message".into(), None)
        .await
        .unwrap();
    assert_eq!(f.controller.len(), 1);
}
