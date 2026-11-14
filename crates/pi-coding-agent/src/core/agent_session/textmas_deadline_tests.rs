use super::*;
use futures::StreamExt;
use pi_ai::types::{AssistantMessage, ContentBlock, TextContent, Usage, UsageCost};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const FLOOD_EVENTS: usize = 512;
const CANCEL_AFTER: usize = 8;

fn partial_message() -> AssistantMessage {
    AssistantMessage::default()
}

fn draft_message(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new(text))],
        usage: Usage {
            input: 11.0,
            output: 7.0,
            total_tokens: 18.0,
            cost: UsageCost { total: 0.125, ..Default::default() },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn flood_events(ready_events: usize) -> Vec<AssistantMessageEvent> {
    let partial = partial_message();
    let final_message = draft_message("ROLE_DRAFT");
    let mut events = (0..ready_events)
        .map(|index| AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: format!("ready-{index}"),
            partial: partial.clone(),
        })
        .collect::<Vec<_>>();
    events.push(AssistantMessageEvent::Done {
        reason: final_message.stop_reason.clone(),
        message: final_message,
    });
    events
}

async fn run_bounded_stream<S>(
    stream: S,
    stage_timeout: Duration,
    run_signal: &CancellationToken,
    parent_signal: Option<&CancellationToken>,
    terminal_result: Option<AssistantMessage>,
) -> (TextMasProviderResult, Option<Usage>)
where
    S: futures::Stream<Item = AssistantMessageEvent> + Unpin,
{
    let mut usage = None;
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        run_textmas_provider_request(
            run_signal,
            parent_signal,
            stage_timeout,
            &mut usage,
            async move { (stream, move || terminal_result.clone()) },
        ),
    )
    .await
    .expect("ready-event regression exceeded its two-second watchdog");
    (result, usage)
}

#[tokio::test]
async fn normal_ready_event_then_terminal_preserves_draft_and_usage() {
    let run_signal = CancellationToken::new();
    let (result, usage) = run_bounded_stream(
        futures::stream::iter(flood_events(1)),
        Duration::from_secs(60),
        &run_signal,
        None,
        None,
    )
    .await;
    assert_eq!(result.expect("normal stream must complete"), "ROLE_DRAFT");
    assert_eq!(usage, Some(draft_message("ROLE_DRAFT").usage));
}

#[tokio::test]
async fn zero_total_deadline_wins_ready_event_flood_without_consuming_terminal() {
    let run_signal = CancellationToken::new();
    let observed = Arc::new(AtomicUsize::new(0));
    let observed_by_stream = observed.clone();
    let stream = futures::stream::iter(flood_events(FLOOD_EVENTS)).inspect(move |_| {
        observed_by_stream.fetch_add(1, Ordering::SeqCst);
    });
    let (result, _) = run_bounded_stream(stream, Duration::ZERO, &run_signal, None, None).await;
    let consumed = observed.load(Ordering::SeqCst);
    assert!(
        matches!(result, Err((TextMasCallStatus::TimedOut, _))),
        "zero total deadline must be observed before the ready terminal; consumed {consumed} of {} events; result: {result:?}",
        FLOOD_EVENTS + 1,
    );
    assert_eq!(consumed, 0, "an already expired shared deadline must stop before stream consumption");
}

#[tokio::test]
async fn run_cancellation_during_ready_flood_stops_before_terminal() {
    let run_signal = CancellationToken::new();
    let cancel_at_observation = run_signal.clone();
    let observed = Arc::new(AtomicUsize::new(0));
    let observed_by_stream = observed.clone();
    let stream = futures::stream::iter(flood_events(FLOOD_EVENTS)).inspect(move |_| {
        let current = observed_by_stream.fetch_add(1, Ordering::SeqCst) + 1;
        if current == CANCEL_AFTER { cancel_at_observation.cancel(); }
    });
    let (result, _) = run_bounded_stream(stream, Duration::from_secs(60), &run_signal, None, None).await;
    let consumed = observed.load(Ordering::SeqCst);
    assert!(
        matches!(result, Err((TextMasCallStatus::Cancelled, _))),
        "run/model-change cancellation must be observed during the flood; consumed {consumed} events; result: {result:?}",
    );
    assert_eq!(consumed, CANCEL_AFTER);
}

#[tokio::test]
async fn parent_cancellation_during_ready_flood_stops_before_terminal() {
    let run_signal = CancellationToken::new();
    let parent_signal = CancellationToken::new();
    let cancel_at_observation = parent_signal.clone();
    let observed = Arc::new(AtomicUsize::new(0));
    let observed_by_stream = observed.clone();
    let stream = futures::stream::iter(flood_events(FLOOD_EVENTS)).inspect(move |_| {
        let current = observed_by_stream.fetch_add(1, Ordering::SeqCst) + 1;
        if current == CANCEL_AFTER { cancel_at_observation.cancel(); }
    });
    let (result, _) = run_bounded_stream(
        stream,
        Duration::from_secs(60),
        &run_signal,
        Some(&parent_signal),
        None,
    )
    .await;
    let consumed = observed.load(Ordering::SeqCst);
    assert!(
        matches!(result, Err((TextMasCallStatus::Cancelled, _))),
        "parent cancellation must be observed during the flood; consumed {consumed} events; result: {result:?}",
    );
    assert_eq!(consumed, CANCEL_AFTER);
}

#[tokio::test]
async fn terminal_error_remains_failed_during_ready_stream() {
    let run_signal = CancellationToken::new();
    let mut error = draft_message("");
    error.stop_reason = "error".into();
    error.error_message = Some("synthetic provider failure".into());
    let mut events = flood_events(1);
    events.pop();
    events.push(AssistantMessageEvent::Error { reason: "error".into(), error });
    let (result, usage) = run_bounded_stream(
        futures::stream::iter(events),
        Duration::from_secs(60),
        &run_signal,
        None,
        None,
    )
    .await;
    assert!(matches!(result, Err((TextMasCallStatus::Failed, _))), "{result:?}");
    assert_eq!(usage.expect("terminal error usage must be retained").output, 7.0);
}

#[tokio::test]
async fn late_terminal_result_still_requires_a_complete_public_draft() {
    let run_signal = CancellationToken::new();
    let (result, usage) = run_bounded_stream(
        futures::stream::empty(),
        Duration::from_secs(60),
        &run_signal,
        None,
        Some(draft_message("LATE_DRAFT")),
    )
    .await;
    assert_eq!(result.expect("late terminal result must remain supported"), "LATE_DRAFT");
    assert_eq!(usage, Some(draft_message("LATE_DRAFT").usage));
}


struct RoleFixture {
    root: tempfile::TempDir,
}

impl RoleFixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("workspace")).unwrap();
        std::fs::create_dir_all(root.path().join("profile")).unwrap();
        Self { root }
    }

    async fn session(&self) -> Arc<AgentSession> {
        use crate::core::agent_session_services::{
            create_agent_session_from_services, create_agent_session_services,
            AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
            CreateAgentSessionServicesOptions,
        };
        use crate::core::auth_storage::AuthStorage;
        use crate::core::model_registry::ModelRegistry;
        use crate::core::resource_loader::DefaultResourceLoaderOptions;
        use crate::core::session_manager::SessionManager;
        use crate::core::settings_manager::SettingsManager;
        use std::sync::Mutex;

        let cwd = self.root.path().join("workspace").to_string_lossy().into_owned();
        let profile = self.root.path().join("profile").to_string_lossy().into_owned();
        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false
            })
            .as_object()
            .unwrap()
            .clone(),
        )));
        let mut model = Model::new(
            "deadline-model",
            "Deadline test model",
            "offline-deadline-api",
            "offline-deadline-provider",
            "http://localhost:0",
        );
        model.context_window = 128_000.0;
        model.max_tokens = 2_048.0;
        let registry = Arc::new(Mutex::new(ModelRegistry::in_memory(AuthStorage::in_memory(
            Default::default(),
            None,
        ))));
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(profile.clone()),
            auth_storage: Some(Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
                Default::default(),
                None,
            )))),
            settings_manager: Some(settings.clone()),
            model_registry: Some(registry),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir: profile,
                settings_manager: Some(settings),
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
        let manager = SessionManager::create(&cwd, None).unwrap();
        create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::new(services),
            session_manager: Arc::new(Mutex::new(manager)),
            session_start_event: None,
            creation: AgentSessionCreationOptions {
                model: Some(model),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                rlm_depth: Some(0),
                ..Default::default()
            },
        })
        .await
        .unwrap()
        .session
    }
}

fn install_prefilled_role_stream(session: &AgentSession, events: Vec<AssistantMessageEvent>) {
    let stream_fn: pi_agent_core::types::StreamFn = Arc::new(move |_, context, _| {
        assert!(context.tools.is_none(), "auxiliary roles must not receive tools");
        let stream = pi_ai::utils::event_stream::AssistantMessageEventStream::new();
        let terminal = events.iter().rev().find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
            _ => None,
        });
        for event in events.clone() { stream.push(event); }
        stream.end(terminal);
        Box::pin(async move { stream })
    });
    session.agent.set_stream_fn(stream_fn);
}

fn role_run(session: &AgentSession, action_id: &str) -> (TextMasRun, Context) {
    let user = UserMessage::new(UserContent::Text("CURRENT_REQUEST".into()), 1);
    let context = Context::new(
        Some("ROLE_SYSTEM".into()),
        vec![Message::User(user.clone())],
        None,
    );
    let run = TextMasRun {
        action_id: action_id.into(),
        input_ids: Vec::new(),
        last_user: AgentMessage::Message(Message::User(user)),
        mode: TextMasMode::Sequential,
        model: session.agent.model(),
        signal: CancellationToken::new(),
        phase: Phase::Armed,
        activity: Default::default(),
    };
    (run, context)
}

fn durable_usage(session: &AgentSession) -> Vec<TextMasUsageRecord> {
    session
        .session_manager
        .lock()
        .unwrap()
        .get_entries()
        .iter()
        .filter_map(contract::usage_record)
        .collect()
}

#[tokio::test]
async fn timeout_during_ready_flood_persists_started_and_timed_out_once() {
    let fixture = RoleFixture::new();
    let session = fixture.session().await;
    install_prefilled_role_stream(&session, flood_events(FLOOD_EVENTS));
    let (run, context) = role_run(&session, "deadline-timeout");
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        session.run_textmas_role_with_timeout(
            &run,
            TextMasRole::Planner,
            None,
            context,
            None,
            Duration::ZERO,
        ),
    )
    .await
    .expect("role-path ready flood exceeded its watchdog");
    assert!(result.is_err(), "expired role must not return a draft: {result:?}");
    let records = durable_usage(&session);
    assert_eq!(records.len(), 2, "one Started and one terminal record required: {records:?}");
    assert_eq!(records[0].status, TextMasCallStatus::Started);
    assert_eq!(records[1].status, TextMasCallStatus::TimedOut);
    assert_eq!(records[0].call_id(), records[1].call_id());
    let mut ledger = contract::TextMasUsageLedger::default();
    for record in records { ledger.observe(record); }
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger.records().next().unwrap().status, TextMasCallStatus::TimedOut);
}

#[tokio::test]
async fn normal_role_terminal_persists_started_and_completed_once_with_usage() {
    let fixture = RoleFixture::new();
    let session = fixture.session().await;
    install_prefilled_role_stream(&session, flood_events(1));
    let (run, context) = role_run(&session, "deadline-completed");
    let result = session
        .run_textmas_role_with_timeout(
            &run,
            TextMasRole::Planner,
            None,
            context,
            None,
            Duration::from_secs(60),
        )
        .await;
    assert_eq!(result.expect("normal role call must complete"), "ROLE_DRAFT");
    let records = durable_usage(&session);
    assert_eq!(records.len(), 2, "one Started and one terminal record required: {records:?}");
    assert_eq!(records[0].status, TextMasCallStatus::Started);
    assert_eq!(records[1].status, TextMasCallStatus::Completed);
    assert_eq!(records[0].call_id(), records[1].call_id());
    assert_eq!(records[1].usage.as_ref().unwrap().total_tokens, 18.0);
    assert_eq!(records[1].usage.as_ref().unwrap().cost.total, 0.125);
    let mut ledger = contract::TextMasUsageLedger::default();
    for record in records { ledger.observe(record); }
    let summary = ledger.summary().unwrap();
    assert_eq!(summary.calls, 1);
    assert_eq!(summary.known_tokens, 18.0);
    assert_eq!(summary.known_subtotal.cost.total, 0.125);
}
