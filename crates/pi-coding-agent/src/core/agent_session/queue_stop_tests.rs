use super::*;
use pi_ai::types::{AssistantMessageEvent, ContentBlock, ToolCall};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use std::sync::atomic::AtomicUsize;

async fn fixture() -> Arc<AgentSession> {
    post_compaction_continuation_tests::test_session_with_credentials().await
}

fn report(id: &str) -> CustomMessage {
    CustomMessage {
        role: "custom".into(),
        custom_type: "agent_message".into(),
        content: CustomMessageContent::Text(format!("Old report {id}: do not lose this evidence")),
        details: Some(
            serde_json::json!({"id": id, "message": "old child report", "fromRelationship": "child", "fromName": "child-fixture"}),
        ),
        display: true,
        timestamp: 1000,
    }
}

fn install_tool(
    session: &Arc<AgentSession>,
    entered: CancellationToken,
    release: CancellationToken,
    calls: Arc<AtomicUsize>,
) {
    let tool = AgentTool {
        name: "boundary_fixture".into(),
        description: "isolated deterministic tool".into(),
        label: "fixture".into(),
        parameters: serde_json::json!({"type":"object","properties":{}}),
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(move |_, _, signal, _| {
            let (entered, release, calls) = (entered.clone(), release.clone(), calls.clone());
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                entered.cancel();
                if let Some(signal) = signal {
                    tokio::select! { _ = release.cancelled() => {}, _ = signal.cancelled() => {} }
                } else {
                    release.cancelled().await;
                }
                Ok(pi_agent_core::types::AgentToolResult {
                    content: vec![pi_agent_core::types::ContentBlock::text("settled")],
                    details: Value::Null,
                    is_error: None,
                    terminate: None,
                })
            })
        }),
    };
    session
        .tool_registry
        .lock()
        .unwrap()
        .insert(tool.name.clone(), tool);
    session.set_active_tools_by_name(&["boundary_fixture".into()]);
}

fn install_provider(
    session: &Arc<AgentSession>,
    contexts: Arc<Mutex<Vec<pi_ai::types::Context>>>,
    tool_turns: usize,
) {
    // A real child requires its terminal protocol marker; bare prose correctly
    // triggers recovery continuations and is not a completed-child fixture.
    let completed_text = if session.rlm_depth > 0 {
        "done\nRLM_CHILD_STATUS: complete"
    } else {
        "done"
    };
    session
        .agent
        .set_stream_fn(Arc::new(move |model, context, _options| {
            let index = {
                let mut seen = contexts.lock().unwrap();
                seen.push(context);
                seen.len()
            };
            Box::pin(async move {
                let stream = AssistantMessageEventStream::new();
                let message = AssistantMessage {
                    model: model.id,
                    provider: model.provider,
                    api: model.api,
                    content: if index <= tool_turns {
                        vec![ContentBlock::ToolCall(ToolCall::new(
                            format!("call-{index}"),
                            "boundary_fixture",
                            Default::default(),
                        ))]
                    } else {
                        vec![ContentBlock::Text(TextContent::new(completed_text))]
                    },
                    stop_reason: if index <= tool_turns {
                        "toolUse"
                    } else {
                        "stop"
                    }
                    .into(),
                    timestamp: now_ms() as i64,
                    ..Default::default()
                };
                stream.push(AssistantMessageEvent::Start {
                    partial: message.clone(),
                });
                stream.push(AssistantMessageEvent::Done {
                    reason: message.stop_reason.clone(),
                    message,
                });
                stream
            })
        }));
}

fn has_text(context: &pi_ai::types::Context, text: &str) -> bool {
    serde_json::to_string(&context.messages)
        .unwrap()
        .contains(text)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_session_steer_reaches_next_tool_boundary_in_both_modes() {
    for mode in ["one-at-a-time", "all"] {
        let session = fixture().await;
        session.set_steering_mode(mode);
        let entered = CancellationToken::new();
        let release = CancellationToken::new();
        let tool_calls = Arc::new(AtomicUsize::new(0));
        install_tool(
            &session,
            entered.clone(),
            release.clone(),
            tool_calls.clone(),
        );
        let contexts = Arc::new(Mutex::new(Vec::new()));
        install_provider(&session, contexts.clone(), 3);
        let owner = session.clone();
        let initial = tokio::spawn(async move { owner.prompt("initial work", None).await });
        tokio::time::timeout(std::time::Duration::from_secs(5), entered.cancelled())
            .await
            .unwrap();
        for text in ["human first", "human second"] {
            session
                .prompt_until_accepted(
                    text,
                    Some(PromptOptions {
                        streaming_behavior: Some("steer".into()),
                        queue_if_busy: Some(true),
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            session.get_session_action_snapshot().steering,
            ["human first", "human second"]
        );
        assert!(session.steering_stop_pending());
        assert_eq!(
            tool_calls.load(Ordering::SeqCst),
            1,
            "a steer must not abort or duplicate a running tool"
        );
        release.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(8), initial)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(8), session.wait_for_idle())
            .await
            .unwrap()
            .unwrap();
        let seen = contexts.lock().unwrap();
        assert!(
            has_text(&seen[1], "human first"),
            "{mode}: first steer must reach call 2, not call 5"
        );
        let first = seen
            .iter()
            .position(|c| has_text(c, "human first"))
            .unwrap();
        let second = seen
            .iter()
            .position(|c| has_text(c, "human second"))
            .unwrap();
        assert!(first <= second, "human ordering");
        assert_eq!(
            tool_calls.load(Ordering::SeqCst),
            3,
            "each tool executes once"
        );
        drop(seen);
        session.dispose_async(Some(false)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_session_follow_up_does_not_interrupt_tool_continuation() {
    let session = fixture().await;
    let entered = CancellationToken::new();
    let release = CancellationToken::new();
    install_tool(
        &session,
        entered.clone(),
        release.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 2);
    let owner = session.clone();
    let initial = tokio::spawn(async move { owner.prompt("initial work", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.cancelled())
        .await
        .unwrap();
    session
        .prompt_until_accepted(
            "only after run",
            Some(PromptOptions {
                streaming_behavior: Some("followUp".into()),
                queue_if_busy: Some(true),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert!(!session.steering_stop_pending());
    release.cancel();
    tokio::time::timeout(std::time::Duration::from_secs(8), initial)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(8), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    let seen = contexts.lock().unwrap();
    assert!(!has_text(&seen[1], "only after run"));
    assert!(!has_text(&seen[2], "only after run"));
    assert!(has_text(&seen[3], "only after run"));
    drop(seen);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_stop_preserves_reports_without_model_wakeup_or_replay() {
    let session = fixture().await;
    let journal_root = tempfile::tempdir().unwrap();
    *session.session_manager.lock().unwrap() =
        SessionManager::create(&session.cwd, Some(&journal_root.path().to_string_lossy())).unwrap();
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 0);
    let pause = session.acquire_queued_work_pause();
    assert!(session
        .queue_agent_message_prompt("old report", "steer", Some(report("before-stop")))
        .await
        .unwrap());
    session.request_abort();
    let delivery = session.wait_for_agent_message_prompt_delivery("after-stop");
    for _ in 0..2 {
        session
            .accept_agent_message_prompt(
                "late report",
                Some(PromptOptions {
                    custom_message: Some(report("after-stop")),
                    agent_message_id: Some("after-stop".into()),
                    streaming_behavior: Some("steer".into()),
                    queue_if_busy: Some(true),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), delivery.wait())
            .await
            .unwrap()
            .is_err(),
        "a saved report must not falsely acknowledge delivery to the model"
    );
    pause.release();
    session.resume_queued_work(); // automatic checkpoint resumption is not consent
    assert!(session.explicitly_stopped());
    assert!(session.session_input_pump_suspended.load(Ordering::SeqCst));
    assert!(contexts.lock().unwrap().is_empty());
    let entries = session.session_manager.lock().unwrap().get_branch(None);
    let reports: Vec<_> = entries
        .iter()
        .filter(|e| e.get("customType").and_then(Value::as_str) == Some(DEFERRED_REPORT_ENTRY))
        .collect();
    assert_eq!(reports.len(), 2, "deduplicated durable first delivery");
    assert!(serde_json::to_string(&reports)
        .unwrap()
        .contains("do not lose this evidence"));
    assert!(reports
        .iter()
        .all(|e| e["data"]["message"]["timestamp"] == 1000));
    // Rehydration derives its latch from the durable journal, not process state.
    let journal_path = session
        .session_manager
        .lock()
        .unwrap()
        .get_session_file()
        .unwrap();
    let reopened = SessionManager::open(&journal_path, None, None).unwrap();
    *session.session_manager.lock().unwrap() = reopened;
    *session.explicit_stop.lock().unwrap() = ExplicitStopState::default();
    session.restore_explicit_stop();
    assert!(session.explicitly_stopped());
    // Distinguish reports authored during the pause from those preceding stop.
    // The short waits establish timestamp ordering, not task-cleanup settlement.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let mut during_pause = report("authored-during-pause");
    during_pause.timestamp = now_ms_i64();
    assert!((during_pause.timestamp as f64) > session.explicit_stop.lock().unwrap().stopped_at);
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    session.prompt("new task only", None).await.unwrap();
    assert!(!session.explicitly_stopped());
    let seen = contexts.lock().unwrap();
    assert_eq!(
        seen.len(),
        1,
        "resume cannot create one model call per old report"
    );
    assert!(has_text(&seen[0], "2 child reports were saved"));
    assert!(
        !has_text(&seen[0], "do not lose this evidence"),
        "old instructions cannot be replayed"
    );
    drop(seen);
    // Restore again after resume: the extended cutoff must survive a reboot too.
    *session.session_manager.lock().unwrap() =
        SessionManager::open(&journal_path, None, None).unwrap();
    *session.explicit_stop.lock().unwrap() = ExplicitStopState::default();
    session.restore_explicit_stop();
    assert!(!session.explicitly_stopped());
    session
        .accept_agent_message_prompt(
            "pause-period report delivered late",
            Some(PromptOptions {
                custom_message: Some(during_pause),
                agent_message_id: Some("authored-during-pause".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(
        contexts.lock().unwrap().len(),
        1,
        "pause-period backlog cannot wake the resumed task"
    );
    // Resuming a new task does not authorize replay of transport-buffered reports
    // authored before the stop, even if they arrive after that explicit resume.
    session
        .accept_agent_message_prompt(
            "late old backlog",
            Some(PromptOptions {
                custom_message: Some(report("after-resume")),
                agent_message_id: Some("after-resume".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(contexts.lock().unwrap().len(), 1);
    assert_eq!(
        session
            .session_manager
            .lock()
            .unwrap()
            .get_branch(None)
            .iter()
            .filter(|e| e.get("customType").and_then(Value::as_str) == Some(DEFERRED_REPORT_ENTRY))
            .count(),
        4
    );
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_stop_cancels_active_tool_without_cleanup_model_calls() {
    let session = fixture().await;
    let entered = CancellationToken::new();
    let release = CancellationToken::new();
    install_tool(
        &session,
        entered.clone(),
        release.clone(),
        Arc::new(AtomicUsize::new(0)),
    );
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 5);
    let owner = session.clone();
    let initial = tokio::spawn(async move { owner.prompt("work", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.cancelled())
        .await
        .unwrap();
    *session.pending_requested_compaction.lock().unwrap() =
        Some(PendingRequestedCompaction::default());
    tokio::time::timeout(std::time::Duration::from_secs(3), session.abort())
        .await
        .unwrap()
        .unwrap();
    assert!(session.explicitly_stopped());
    assert!(
        !session.is_streaming(),
        "a drained event callback must not restore stale busy state"
    );
    assert!(session
        .pending_requested_compaction
        .lock()
        .unwrap()
        .is_none());
    assert_eq!(contexts.lock().unwrap().len(), 1);
    release.cancel();
    tokio::time::timeout(std::time::Duration::from_secs(3), initial)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stopped_session_skips_late_retry_and_compaction_requests() {
    let session = fixture().await;
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 0);
    session.request_abort();
    session
        .settings_manager
        .lock()
        .unwrap()
        .set_retry_enabled(true);
    let error = AssistantMessage {
        stop_reason: "error".into(),
        error_message: Some("server error".into()),
        ..Default::default()
    };
    assert!(!session.handle_retryable_error(&error).await);
    assert!(
        !session
            .run_auto_compaction(COMPACTION_REASON_REQUESTED, false)
            .await
    );
    assert_eq!(
        session
            .check_compaction_overflow_with(&error, &session.compaction_settings(), true)
            .await,
        Some(false)
    );
    assert!(!session.is_retrying());
    assert!(contexts.lock().unwrap().is_empty());
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fresh_parent_command_resumes_stopped_child_through_real_admission() {
    let session =
        post_compaction_continuation_tests::test_session_with_credentials_at_depth(1).await;
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 0);
    session.request_abort();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let mut command = report("new-parent-task");
    command.timestamp = now_ms_i64();
    command.content = CustomMessageContent::Text("FRESH_PARENT_TASK".into());
    command.details = Some(serde_json::json!({
        "id": "new-parent-task", "message": "FRESH_PARENT_TASK", "fromRelationship": "parent", "fromName": "parent-fixture"
    }));
    let delivery = session.wait_for_agent_message_prompt_delivery("new-parent-task");
    session
        .accept_agent_message_prompt(
            "FRESH_PARENT_TASK",
            Some(PromptOptions {
                custom_message: Some(command),
                agent_message_id: Some("new-parent-task".into()),
                streaming_behavior: Some("steer".into()),
                queue_if_busy: Some(true),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), delivery.wait())
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    assert!(!session.explicitly_stopped());
    let seen = contexts.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(has_text(&seen[0], "FRESH_PARENT_TASK"));
    drop(seen);
    assert!(!session
        .explicit_stop
        .lock()
        .unwrap()
        .report_ids
        .contains("new-parent-task"));
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn message_patch_preserves_concurrent_appends_and_live_busy_state() {
    let session = fixture().await;
    let original = AgentMessage::Message(Message::Assistant(AssistantMessage {
        timestamp: 1,
        content: vec![ContentBlock::Text(TextContent::new("original"))],
        ..Default::default()
    }));
    let replacement = AgentMessage::Message(Message::Assistant(AssistantMessage {
        timestamp: 1,
        content: vec![ContentBlock::Text(TextContent::new("replacement"))],
        ..Default::default()
    }));
    let seed = original.clone();
    session.agent.update_state(Box::new(move |state| {
        state.messages.push(seed);
    }));
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let writer = session.agent.clone();
    let writer_barrier = barrier.clone();
    let append = std::thread::spawn(move || {
        writer_barrier.wait();
        for index in 0..500 {
            writer.update_state(Box::new(move |state| {
                state.is_streaming = true;
                state
                    .messages
                    .push(AgentMessage::Message(Message::Assistant(
                        AssistantMessage {
                            timestamp: index + 100,
                            content: vec![ContentBlock::Text(TextContent::new("new message"))],
                            ..Default::default()
                        },
                    )));
            }));
        }
        writer.update_state(Box::new(|state| {
            state.is_streaming = false;
        }));
    });
    let patcher = session.clone();
    let patch = std::thread::spawn(move || {
        barrier.wait();
        for _ in 0..500 {
            patcher.replace_message_in_place(&original, replacement.clone());
            patcher.replace_message_in_place(&replacement, original.clone());
        }
    });
    append.join().unwrap();
    patch.join().unwrap();
    let live = session.agent.state();
    assert_eq!(
        live.messages.len(),
        501,
        "a callback must not replace the complete history snapshot"
    );
    assert!(
        !live.is_streaming,
        "a callback cannot resurrect a completed run"
    );
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn steering_predicate_checks_selected_preparing_and_not_running_or_followups() {
    let session = fixture().await;
    let pause = session.acquire_queued_work_pause();
    let mut action = session.create_prepared_turn_action(
        "steer",
        "queued",
        None,
        Some(PreparedTurnActionOptions {
            queue_visible: Some(true),
            ..Default::default()
        }),
    );
    session
        .action_store
        .lock()
        .unwrap()
        .enqueue(action.clone())
        .unwrap();
    assert!(session.steering_stop_pending());
    action = session
        .action_store
        .lock()
        .unwrap()
        .select_first()
        .unwrap()
        .unwrap();
    assert!(session.steering_stop_pending());
    transition_session_action(
        &mut action,
        ActionLifecycle::Preparing { preparation: None },
        &TransitionOptions::default(),
    )
    .unwrap();
    session
        .action_store
        .lock()
        .unwrap()
        .update_action(&action)
        .unwrap();
    assert!(session.steering_stop_pending());
    transition_session_action(
        &mut action,
        ActionLifecycle::Committing,
        &TransitionOptions::default(),
    )
    .unwrap();
    transition_session_action(
        &mut action,
        ActionLifecycle::Running {
            execution: crate::core::session_action_store::ActionExecution::AgentTurn,
        },
        &TransitionOptions::default(),
    )
    .unwrap();
    session
        .action_store
        .lock()
        .unwrap()
        .update_action(&action)
        .unwrap();
    assert!(
        !session.steering_stop_pending(),
        "the already-running steer must not interrupt itself"
    );
    session.clear_queue();
    assert!(!session.steering_stop_pending());
    let follow = session.create_prepared_turn_action("followUp", "later", None, None);
    session
        .action_store
        .lock()
        .unwrap()
        .enqueue(follow)
        .unwrap();
    session.action_store.lock().unwrap().select_first().unwrap();
    assert!(!session.steering_stop_pending());
    session.request_abort();
    pause.release();
    session.dispose_async(Some(false)).await;
}

#[derive(Default)]
struct QueueRecorder(Mutex<Vec<pi_agent_core::performance_metrics::PerformanceMetricEvent>>);

#[tokio::test]
async fn failed_response_removal_allows_login_annotation_but_preserves_new_or_changed_messages() {
    let session = fixture().await;
    let failed = AssistantMessage {
        timestamp: 42,
        stop_reason: "error".into(),
        error_message: Some("401 Unauthorized".into()),
        content: vec![ContentBlock::Text(TextContent::new("original content"))],
        ..Default::default()
    };
    let mut annotated = failed.clone();
    annotated.error_message = Some(add_login_guidance_to_auth_error("401 Unauthorized"));
    assert_ne!(
        annotated.error_message, failed.error_message,
        "fixture must exercise an actual annotation"
    );
    for (candidate, removed) in [
        (failed.clone(), true),
        (annotated.clone(), true),
        (
            AssistantMessage {
                timestamp: 43,
                ..annotated.clone()
            },
            false,
        ),
        (
            AssistantMessage {
                content: vec![ContentBlock::Text(TextContent::new("different content"))],
                ..annotated.clone()
            },
            false,
        ),
        (
            AssistantMessage {
                error_message: Some("a different failure".into()),
                ..annotated
            },
            false,
        ),
    ] {
        let expected = AgentMessage::Message(Message::Assistant(candidate.clone()));
        session.agent.update_state(Box::new(move |state| {
            state.is_streaming = true;
            state.messages = vec![AgentMessage::Message(Message::Assistant(candidate))];
        }));
        session.remove_failed_assistant_from_state(&failed);
        let live = session.agent.state();
        assert!(
            live.is_streaming,
            "message removal cannot overwrite live run flags"
        );
        if removed {
            assert!(live.messages.is_empty());
        } else {
            assert_eq!(live.messages, vec![expected]);
        }
    }
    session.agent.update_state(Box::new(|state| {
        state.is_streaming = false;
    }));
    session.dispose_async(Some(false)).await;
}

impl PerformanceMetricRecorder for QueueRecorder {
    fn session_id(&self) -> &str {
        "queue-timing-session"
    }
    fn monotonic_now(&self) -> f64 {
        now_ms()
    }
    fn next_id(&self, _: pi_agent_core::performance_metrics::PerformanceMetricIdScope) -> String {
        uuid::Uuid::new_v4().to_string()
    }
    fn record(&self, event: pi_agent_core::performance_metrics::PerformanceMetricEvent) {
        self.0.lock().unwrap().push(event);
    }
    fn flush(&self) {}
    fn close(&self) {}
}

fn terminal_notice(id: &str) -> CustomMessage {
    CustomMessage {
        custom_type: RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE.into(),
        content: CustomMessageContent::Text(format!("Child {id} was cancelled")),
        ..report(id)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_notice_admission_cannot_race_next_turn_drain() {
    let session = fixture().await;
    let contexts = Arc::new(Mutex::new(Vec::new()));
    install_provider(&session, contexts.clone(), 0);
    let drained = Arc::new(Mutex::new(Vec::new()));
    let observed = drained.clone();
    let owner = Arc::downgrade(&session);
    let unsubscribe = session.subscribe(Arc::new(move |event| {
        if matches!(event, AgentSessionEvent::SessionActionUpdate { .. }) {
            // Admission publishes before returning. Reproduce a queued turn
            // taking its next-turn context during that window.
            observed
                .lock()
                .unwrap()
                .extend(owner.upgrade().unwrap().take_pending_next_turn_messages());
        }
    }));
    session
        .pending_next_turn_messages
        .lock()
        .unwrap()
        .push(report("context"));
    session
        .defer_rlm_terminal_notice(terminal_notice("cancelled-child"))
        .await
        .unwrap();
    unsubscribe();
    let drained = drained.lock().unwrap().clone();
    assert_eq!(
        drained.len(),
        1,
        "the notice must have only one delivery owner"
    );
    assert_eq!(drained[0].custom_type, "agent_message");
    tokio::time::timeout(std::time::Duration::from_secs(5), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    session
        .queue_jev_control_feedback(CustomMessage {
            custom_type: "jevControl".into(),
            content: CustomMessageContent::Text("CONTINUE_AFTER_CANCEL".into()),
            ..report("continuation")
        })
        .await;
    tokio::time::timeout(std::time::Duration::from_secs(5), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        session.prompt("USER_AFTER_CANCEL", None),
    )
    .await
    .unwrap()
    .unwrap();
    let seen = contexts.lock().unwrap();
    assert!(seen
        .iter()
        .any(|context| has_text(context, "CONTINUE_AFTER_CANCEL")));
    assert!(has_text(seen.last().unwrap(), "USER_AFTER_CANCEL"));
    let messages = session.messages();
    assert_eq!(messages.iter().filter(|message| {
        matches!(message, AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, .. }) if custom_type == RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE)
    }).count(), 1);
    drop(seen);
    assert_eq!(session.queued_action_count(), 0);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn terminal_notice_failed_admission_preserves_pending_context() {
    let session = fixture().await;
    install_provider(&session, Arc::new(Mutex::new(Vec::new())), 0);
    let first = terminal_notice("first");
    let context = report("context");
    let second = terminal_notice("second");
    *session.pending_next_turn_messages.lock().unwrap() =
        vec![first, context.clone(), second.clone()];
    let owner = Arc::downgrade(&session);
    let unsubscribe = session.subscribe(Arc::new(move |event| {
        if matches!(event, AgentSessionEvent::SessionActionUpdate { .. }) {
            owner
                .upgrade()
                .unwrap()
                .session_input_admission_pauses
                .lock()
                .unwrap()
                .insert("notice-admission-test".into());
        }
    }));
    session.flush_deferred_rlm_terminal_notices();
    unsubscribe();
    assert_eq!(session.get_session_action_snapshot().follow_ups.len(), 1);
    assert_eq!(
        *session.pending_next_turn_messages.lock().unwrap(),
        vec![context, second]
    );
    assert_eq!(
        session
            .durable_rlm_terminal_notice_action_ids
            .lock()
            .unwrap()
            .len(),
        1
    );
    session
        .session_input_admission_pauses
        .lock()
        .unwrap()
        .clear();
    session.flush_deferred_rlm_terminal_notices();
    tokio::time::timeout(std::time::Duration::from_secs(5), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    let messages = session.messages();
    for id in ["first", "second"] {
        assert_eq!(messages.iter().filter(|message| {
            matches!(message, AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, content, .. })
                if custom_type == RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE
                    && *content == CustomMessageContent::Text(format!("Child {id} was cancelled")))
        }).count(), 1);
    }
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_metrics_record_actual_delivery_once_without_prompt_content() {
    use pi_agent_core::performance_metrics::{
        PerformanceMetricMeasurement, PerformanceMetricOperation, PerformanceMetricOutcome,
    };
    let session = fixture().await;
    let recorder = Arc::new(QueueRecorder::default());
    session
        .agent
        .set_performance_metrics(Some(AgentLoopPerformanceMetrics {
            recorder: recorder.clone(),
            logical_request_id: None,
            logical_request_started_at: None,
            provider_attempt_number: None,
            host_owns_logical_request_terminal: false,
            logical_request_settlement: None,
        }));
    install_provider(&session, Arc::new(Mutex::new(Vec::new())), 0);
    let pause = session.acquire_queued_work_pause();
    let action = session.create_prepared_turn_action(
        "steer",
        "PRIVATE-PROMPT-MUST-NOT-BE-LOGGED",
        None,
        Some(PreparedTurnActionOptions {
            queue_visible: Some(true),
            ..Default::default()
        }),
    );
    let id = action.id.clone();
    session.admit_session_input(action, false).unwrap();
    assert!(!recorder
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.operation == PerformanceMetricOperation::SessionInput));
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    pause.release();
    tokio::time::timeout(std::time::Duration::from_secs(5), session.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    let events = recorder.0.lock().unwrap();
    let delivered: Vec<_> = events
        .iter()
        .filter(|e| e.operation == PerformanceMetricOperation::SessionInput)
        .collect();
    assert_eq!(delivered.len(), 1);
    assert_eq!(
        delivered[0]
            .correlation
            .as_ref()
            .unwrap()
            .action_id
            .as_deref(),
        Some(id.as_str())
    );
    assert_eq!(
        delivered[0].outcome,
        Some(PerformanceMetricOutcome::Success)
    );
    assert!(
        delivered[0].measurements.as_ref().unwrap()[&PerformanceMetricMeasurement::QueueMs]
            .unwrap()
            >= 20.0
    );
    assert!(!serde_json::to_string(&delivered)
        .unwrap()
        .contains("PRIVATE-PROMPT"));
    drop(events);
    session.dispose_async(Some(false)).await;
}
