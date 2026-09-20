use super::*;
use crate::core::session_action_store::ActionTicketController;
use pi_ai::types::{AssistantMessageEvent, ContentBlock};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use std::sync::atomic::AtomicUsize;

#[derive(Default)]
struct DispatchRecorder(Mutex<Vec<pi_agent_core::performance_metrics::PerformanceMetricEvent>>);

impl PerformanceMetricRecorder for DispatchRecorder {
    fn session_id(&self) -> &str { "dispatch-fixture" }
    fn monotonic_now(&self) -> f64 { now_ms() }
    fn next_id(&self, _: pi_agent_core::performance_metrics::PerformanceMetricIdScope) -> String { uuid::Uuid::new_v4().to_string() }
    fn record(&self, event: pi_agent_core::performance_metrics::PerformanceMetricEvent) { self.0.lock().unwrap().push(event); }
    fn flush(&self) {}
    fn close(&self) {}
}

async fn fixture() -> (Arc<AgentSession>, tempfile::TempDir) {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    let root = tempfile::tempdir().unwrap();
    *session.session_manager.lock().unwrap() = SessionManager::create(
        &session.cwd, Some(&root.path().to_string_lossy()),
    ).unwrap();
    (session, root)
}

fn child_report(id: &str) -> CustomMessage {
    CustomMessage {
        role: "custom".into(), custom_type: "agent_message".into(),
        content: CustomMessageContent::Text(format!("child evidence {id}")),
        details: Some(serde_json::json!({"id":id,"message":format!("child evidence {id}"),"fromRelationship":"child","fromName":"fixture"})),
        display: true, timestamp: 1000,
    }
}

fn family_message(id: &str, relationship: &str) -> CustomMessage {
    let mut message = child_report(id);
    message.content = CustomMessageContent::Text(format!("{relationship} instruction {id}"));
    message.details.as_mut().unwrap()["fromRelationship"] = relationship.into();
    message.details.as_mut().unwrap()["message"] = format!("{relationship} instruction {id}").into();
    message
}

fn enqueue(session: &Arc<AgentSession>, text: &str, custom: Option<CustomMessage>) -> (QueuedSessionAction, Arc<ActionTicketController>) {
    let agent_message_id = custom.as_ref().and_then(|m| m.details.as_ref())
        .and_then(|d| d.get("id")).and_then(Value::as_str).map(str::to_string);
    let action = session.create_prepared_turn_action("steer", text, None, Some(PreparedTurnActionOptions {
        source: Some(if custom.is_some() { "internal" } else { "interactive" }.into()),
        accepted_agent_message: Some(custom.is_some()),
        custom_message: custom, agent_message_id, queue_visible: Some(true), ..Default::default()
    }));
    let (_, ticket, _) = session.admit_session_input_with_options(action.clone(), false, false, false, false).unwrap();
    (action, ticket.unwrap())
}

fn response(model: Model) -> AssistantMessageEventStream {
    response_text(model, "done")
}

fn response_text(model: Model, text: &str) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let mut message = AssistantMessage::new(model.api, model.provider, model.id, now_ms_i64());
    message.content = vec![ContentBlock::Text(TextContent::new(text))];
    stream.push(AssistantMessageEvent::Start { partial: message.clone() });
    stream.push(AssistantMessageEvent::Done { reason: "stop".into(), message });
    stream.end(None);
    stream
}

async fn pump(session: &Arc<AgentSession>) {
    tokio::time::timeout(std::time::Duration::from_secs(10), session.pump_session_inputs(
        session.session_input_pump_epoch.load(Ordering::SeqCst),
    )).await.expect("bounded dispatch");
}

fn count_input(entries: &[crate::core::session_manager::SessionEntry], text: &str) -> usize {
    entries.iter().filter(|entry| {
        matches!(entry.get("type").and_then(Value::as_str), Some("message" | "custom_message"))
            && serde_json::to_string(entry).unwrap().contains(text)
    }).count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn durable_receipts_survive_compaction_context_replacement_and_reopen() {
    let (session, _root) = fixture().await;
    session.set_steering_mode("all");
    let (human, human_ticket) = enqueue(&session, "human durable fixture", None);
    let (child, child_ticket) = enqueue(&session, "child durable fixture", Some(child_report("durable-child")));
    let originals = vec![human.clone(), child.clone()];
    let weak = Arc::downgrade(&session);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        let session = weak.upgrade().unwrap();
        let originals = originals.clone();
        Box::pin(async move {
            session.await_agent_event_queue().await;
            assert!(session.settled_turn_delivery_error(&originals).is_none());
            let last_id = session.session_manager.lock().unwrap().get_leaf_id().unwrap();
            session.session_manager.lock().unwrap().append_compaction(
                "earlier work retained in summary", &last_id, 9999.0, None, None, None, None, None,
            ).unwrap();
            let context = session.build_session_context();
            session.agent.update_state(Box::new(move |state| state.messages = context.messages));
            let live = session.messages();
            assert!(originals.iter().all(|action| {
                !live.contains(&agent_message_from_delivery(&primary_delivery_record(action).unwrap().message))
            }), "human was compacted and custom timestamp was normalized");
            assert!(session.settled_turn_delivery_error(&originals).is_none());
            response(model)
        })
    }));
    pump(&session).await;
    human_ticket.ticket.completed.clone().await.unwrap();
    child_ticket.ticket.completed.clone().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1, "no replay to repair a stale snapshot");
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    assert_eq!(count_input(&reopened.get_entries(), "human durable fixture"), 1);
    assert_eq!(count_input(&reopened.get_entries(), "child evidence durable-child"), 1);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivery_is_durable_before_model_execution_even_when_event_processing_is_delayed() {
    let (session, _root) = fixture().await;
    let release = CancellationToken::new();
    let held = release.clone();
    session.push_agent_event_task(async move { held.cancelled().await; });
    // Select directly so the pump's initial event barrier does not mask the
    // immediate write/receipt while extension event handling remains delayed.
    let (action, ticket) = enqueue(&session, "delayed durable event", None);
    let mut selected = session.action_store.lock().unwrap().select_first().unwrap().unwrap();
    transition_session_action(&mut selected, ActionLifecycle::Preparing { preparation: None }, &TransitionOptions::default()).unwrap();
    session.action_store.lock().unwrap().update_action(&selected).unwrap();
    let message = agent_message_from_delivery(&primary_delivery_record(&action).unwrap().message);
    transition_session_action(&mut selected, ActionLifecycle::Committing, &TransitionOptions::default()).unwrap();
    session.action_store.lock().unwrap().update_action(&selected).unwrap();
    session.agent.update_state(Box::new({ let message = message.clone(); move |state| state.messages.push(message) }));
    session.handle_agent_event(AgentEvent::MessageStart { message: message.clone() });
    session.handle_agent_event(AgentEvent::MessageEnd { message });
    ticket.ticket.delivered.clone().await.unwrap();
    assert!(session.settled_turn_delivery_error(&[action.clone()]).is_none());
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    assert_eq!(count_input(&reopened.get_entries(), "delayed durable event"), 1);
    release.cancel();
    session.await_agent_event_queue().await;
    assert!(session.settled_turn_delivery_error(&[action]).is_none());
    assert_eq!(count_input(&session.session_manager.lock().unwrap().get_entries(), "delayed durable event"), 1);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_transcript_write_rejects_delivery_and_retains_input_for_flush_recovery() {
    let (session, _root) = fixture().await;
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    // Block the exact new fixture transcript path with a directory. No live files.
    std::fs::create_dir(&path).unwrap();
    let (action, ticket) = enqueue(&session, "recover-unsaved-input", None);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { response(model) })
    }));
    pump(&session).await;
    let failure = ticket.ticket.delivered.clone().await.unwrap_err();
    assert!(failure.contains("transcript persistence failed"), "{failure}");
    assert!(failure.contains(&action.id));
    assert!(ticket.ticket.completed.clone().await.is_err());
    assert_eq!(count_input(&session.session_manager.lock().unwrap().get_entries(), "recover-unsaved-input"), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "failed initial persistence must prevent provider/tool work");
    std::fs::remove_dir(&path).unwrap();
    session.session_manager.lock().unwrap().flush_now().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    assert_eq!(count_input(&reopened.get_entries(), "recover-unsaved-input"), 1);
    assert!(!session.has_failed_dispatch_persistence(), "settled failure must not suppress a later independent turn");
    session.prompt("new task after persistence repair", None).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1, "a new explicit task runs normally without replaying the failed action");
    // Repeat after a successful flush: append failure must invalidate the old
    // flushed marker, or recovery would silently omit this retained new input.
    let saved = std::path::Path::new(&path).with_extension("saved-fixture");
    std::fs::rename(&path, &saved).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(session.prompt("unsaved-after-flushed", None).await.unwrap_err().contains("persistence failed"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    std::fs::remove_dir(&path).unwrap();
    std::fs::rename(&saved, &path).unwrap();
    session.session_manager.lock().unwrap().flush_now().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    assert_eq!(count_input(&reopened.get_entries(), "unsaved-after-flushed"), 1);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_backlog_batches_once_while_human_order_and_one_at_a_time_are_preserved() {
    let (session, _root) = fixture().await;
    session.set_steering_mode("one-at-a-time");
    let recorder = Arc::new(DispatchRecorder::default());
    session.agent.set_performance_metrics(Some(AgentLoopPerformanceMetrics::new(recorder.clone())));
    let mut tickets = Vec::new();
    for id in ["one", "two", "three"] {
        tickets.push(enqueue(&session, "child report", Some(child_report(id))).1);
    }
    tickets.push(enqueue(&session, "human-first", None).1);
    tickets.push(enqueue(&session, "human-second", None).1);
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let seen = contexts.clone();
    session.agent.set_stream_fn(Arc::new(move |model, context, _| {
        seen.lock().unwrap().push(serde_json::to_string(&context.messages).unwrap());
        Box::pin(async move { response(model) })
    }));
    pump(&session).await;
    for ticket in tickets { ticket.ticket.completed.clone().await.unwrap(); }
    let seen = contexts.lock().unwrap();
    assert_eq!(seen.len(), 3, "two ordered human turns plus one complete child batch");
    assert!(seen[0].contains("human-first"));
    assert!(!seen[0].contains("human-second"));
    assert!(!seen[0].contains("child evidence"));
    assert!(seen[1].contains("human-second"));
    for id in ["one", "two", "three"] { assert!(seen[2].contains(&format!("child evidence {id}"))); }
    drop(seen);
    let entries = session.session_manager.lock().unwrap().get_entries();
    for id in ["one", "two", "three"] { assert_eq!(count_input(&entries, &format!("child evidence {id}")), 1); }
    let queue_metrics: Vec<_> = recorder.0.lock().unwrap().iter().filter(|event| {
        event.operation == pi_agent_core::performance_metrics::PerformanceMetricOperation::SessionInput
    }).cloned().collect();
    let classifications: Vec<_> = queue_metrics.iter().map(|event| event.measurements.as_ref().unwrap()[
        &pi_agent_core::performance_metrics::PerformanceMetricMeasurement::InputAgentMessage
    ].unwrap()).collect();
    assert_eq!(classifications, [0.0, 0.0, 1.0, 1.0, 1.0]);
    let serialized = serde_json::to_string(&queue_metrics).unwrap();
    assert!(!serialized.contains("human-first") && !serialized.contains("child evidence"));
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn family_backlog_batches_without_combining_messages_or_overtaking_human_input() {
    let (session, _root) = fixture().await;
    session.set_steering_mode("one-at-a-time");
    let messages = [
        ("parent-start", "parent"), ("sibling-evidence", "sibling"),
        ("child-result", "child"), ("parent-correction", "parent"),
    ];
    let mut tickets = Vec::new();
    for (id, relationship) in messages {
        tickets.push(enqueue(&session, id, Some(family_message(id, relationship))).1);
    }
    tickets.push(enqueue(&session, "human-priority-one", None).1);
    tickets.push(enqueue(&session, "human-priority-two", None).1);
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let seen = contexts.clone();
    session.agent.set_stream_fn(Arc::new(move |model, context, _| {
        seen.lock().unwrap().push(serde_json::to_string(&context.messages).unwrap());
        Box::pin(async move { response(model) })
    }));
    pump(&session).await;
    for ticket in tickets {
        ticket.ticket.delivered.clone().await.unwrap();
        ticket.ticket.completed.clone().await.unwrap();
    }
    let seen = contexts.lock().unwrap();
    assert_eq!(seen.len(), 3, "two human turns then one family batch, not four extra model cycles");
    assert!(seen[0].contains("human-priority-one"));
    assert!(!seen[0].contains("human-priority-two"));
    assert!(!seen[1].contains("parent-start"));
    let positions: Vec<_> = messages.iter().map(|(id, _)| seen[2].find(id).unwrap()).collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]), "corrections retain admission order");
    drop(seen);
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    for (id, _) in messages { assert_eq!(count_input(&reopened.get_entries(), id), 1); }
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batched_parent_instructions_each_create_a_durable_continuation_task() {
    let session = post_compaction_continuation_tests::test_session_with_credentials_at_depth(1).await;
    let root = tempfile::tempdir().unwrap();
    *session.session_manager.lock().unwrap() = SessionManager::create(
        &session.cwd, Some(&root.path().to_string_lossy()),
    ).unwrap();
    session.set_steering_mode("one-at-a-time");
    let mut tickets = Vec::new();
    for id in ["parent-original", "parent-latest-correction"] {
        tickets.push(enqueue(&session, id, Some(family_message(id, "parent"))).1);
    }
    // Block deferred event handling only after the pump's initial event barrier.
    // Input persistence and parent-task registration must not need this queue.
    let release = CancellationToken::new();
    let _release_on_drop = release.clone().drop_guard();
    let queued = Arc::new(AtomicBool::new(false));
    let observed_queue = queued.clone();
    let release_events = release.clone();
    let weak_events = Arc::downgrade(&session);
    let unsubscribe = session.agent.subscribe(Arc::new(move |event, _| {
        if matches!(event, AgentEvent::AgentStart) {
            let session = weak_events.upgrade().unwrap();
            let held = release_events.clone();
            session.push_agent_event_task(async move { held.cancelled().await; });
            observed_queue.store(true, Ordering::SeqCst);
        }
        Box::pin(async {})
    }));
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let weak = Arc::downgrade(&session);
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        let session = weak.upgrade().unwrap();
        let release = release.clone();
        let queued = queued.clone();
        Box::pin(async move {
            let ledger = session.rlm_continuation.lock().unwrap().clone();
            let ids: Vec<_> = ledger.tasks.iter().map(|task| task.id.clone()).collect();
            let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
            let reopened = SessionManager::open(&path, None, None).unwrap();
            let entries = reopened.get_entries();
            release.cancel();
            assert!(queued.load(Ordering::SeqCst), "the deferred event queue was pinned before input delivery");
            assert_eq!(ids, ["parent-original", "parent-latest-correction"]);
            assert!(ledger.tasks.iter().all(|task| !task.replied));
            let saved = entries.iter().rev().find(|entry| {
                entry.get("customType").and_then(Value::as_str) == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
            }).expect("parent-task ledger saved before model work");
            let restored = parse_rlm_continuation_state(saved.get("data").unwrap()).unwrap();
            assert_eq!(restored.tasks, ledger.tasks);
            // Settle one synthetic acknowledged reply through the production
            // claim/receipt path. This batching fixture has no parent transport.
            let previous = {
                let mut ledger = session.rlm_continuation.lock().unwrap();
                session.claim_rlm_parent_delivery(&mut ledger, &ids).unwrap()
            };
            assert_eq!(previous.len(), 2);
            let payload = crate::core::agent_messages::AgentSessionMessagePayload {
                id: "agentmsg_batched-parent-fixture".into(),
                source: "agent_message".into(),
                message: "batched parent task result".into(),
                target: crate::core::agent_messages::AgentSessionMessageEndpoint {
                    session_id: "fixture-parent".into(),
                    active_session_id: "fixture-parent-active".into(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let receipt = crate::core::agent_messages::create_agent_session_message_receipt(
                &payload, &"delivered".to_string(), "2026-09-20T00:00:00Z",
            );
            session.settle_rlm_parent_delivery(
                &previous, &Ok(receipt), &payload.target.session_id, &payload.message, true,
            );
            response_text(model, "done\nRLM_CHILD_STATUS: complete")
        })
    }));
    pump(&session).await;
    unsubscribe();
    for ticket in tickets { ticket.ticket.completed.clone().await.unwrap(); }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(session.rlm_continuation.lock().unwrap().tasks.iter().all(|task| task.replied));
    assert_eq!(session.parent_reply_count.load(Ordering::SeqCst), 1);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn failed_parent_task_ledger_write_does_not_publish_or_duplicate_task_state() {
    let session = post_compaction_continuation_tests::test_session_with_credentials_at_depth(1).await;
    let root = tempfile::tempdir().unwrap();
    *session.session_manager.lock().unwrap() = SessionManager::create(
        &session.cwd, Some(&root.path().to_string_lossy()),
    ).unwrap();
    let action = session.create_prepared_turn_action("steer", "parent-ledger", None,
        Some(PreparedTurnActionOptions {
            custom_message: Some(family_message("ledger-failure", "parent")), ..Default::default()
        }));
    let message = agent_message_from_delivery(&primary_delivery_record(&action).unwrap().message);
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let before = session.rlm_continuation.lock().unwrap().clone();
    let replied_before = *session.replied_to_parent_since_task.lock().unwrap();
    // Fail the checked ledger flush even when there is no assistant yet.
    std::fs::create_dir(&path).unwrap();
    let error = session.begin_rlm_parent_task(&message).unwrap_err();
    assert!(error.contains("Parent-task ledger persistence failed"), "{error}");
    assert_eq!(session.rlm_continuation.lock().unwrap().tasks, before.tasks);
    assert_eq!(*session.replied_to_parent_since_task.lock().unwrap(), replied_before);
    assert!(!session.session_manager.lock().unwrap().get_entries().iter().any(|entry| {
        entry.get("customType").and_then(Value::as_str) == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
    }), "failed ledger append is rolled back, not advertised as durable");
    std::fs::remove_dir(&path).unwrap();
    session.begin_rlm_parent_task(&message).unwrap();
    session.begin_rlm_parent_task(&message).unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    let entries = reopened.get_entries();
    let ledgers: Vec<_> = entries.iter().filter(|entry| {
        entry.get("customType").and_then(Value::as_str) == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
    }).collect();
    assert_eq!(ledgers.len(), 1, "later event handling cannot register the same task twice");
    let restored = parse_rlm_continuation_state(ledgers[0].get("data").unwrap()).unwrap();
    assert_eq!(restored.tasks.len(), 1);
    assert_eq!(restored.tasks[0].id, "ledger-failure");
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn older_continuation_snapshot_cannot_append_after_a_new_parent_task() {
    let session = post_compaction_continuation_tests::test_session_with_credentials_at_depth(1).await;
    let root = tempfile::tempdir().unwrap();
    *session.session_manager.lock().unwrap() = SessionManager::create(
        &session.cwd, Some(&root.path().to_string_lossy()),
    ).unwrap();
    let parent_message = |id: &str| {
        let action = session.create_prepared_turn_action("steer", id, None,
            Some(PreparedTurnActionOptions {
                custom_message: Some(family_message(id, "parent")), ..Default::default()
            }));
        agent_message_from_delivery(&primary_delivery_record(&action).unwrap().message)
    };
    session.begin_rlm_parent_task(&parent_message("older-task")).unwrap();
    let newer = parent_message("newer-task");
    // Pin the append destination so the older writer cannot finish. Its ledger
    // lock must stay held, preventing a newer snapshot from overtaking it.
    let manager = session.session_manager.lock().unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let older_session = session.clone();
    let old_writer = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        older_session.persist_rlm_continuation_state();
    });
    started_rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut held_through_append = false;
    while std::time::Instant::now() < deadline {
        if session.rlm_continuation.try_lock().is_err() {
            // Distinguish a transient snapshot lock from ownership while the
            // writer is blocked on the deliberately held destination lock.
            std::thread::sleep(std::time::Duration::from_millis(10));
            if session.rlm_continuation.try_lock().is_err() {
                held_through_append = true;
                break;
            }
        }
        std::thread::yield_now();
    }
    let newer_session = session.clone();
    let new_writer = std::thread::spawn(move || newer_session.begin_rlm_parent_task(&newer));
    drop(manager);
    old_writer.join().unwrap();
    new_writer.join().unwrap().unwrap();
    assert!(held_through_append, "older snapshot released its ledger lock before the append and can overwrite newer state");
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    let entries = reopened.get_entries();
    let saved = entries.iter().rev().find(|entry| {
        entry.get("customType").and_then(Value::as_str) == Some(RLM_CONTINUATION_STATE_CUSTOM_TYPE)
    }).unwrap();
    let restored = parse_rlm_continuation_state(saved.get("data").unwrap()).unwrap();
    let ids: Vec<_> = restored.tasks.iter().map(|task| task.id.as_str()).collect();
    assert_eq!(ids, ["older-task", "newer-task"]);
    assert_eq!(restored.tasks, session.rlm_continuation.lock().unwrap().tasks);
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn automatic_family_batch_requires_matching_policies_and_known_agent_messages() {
    let (session, _root) = fixture().await;
    let create = |id: &str, relationship: &str| session.create_prepared_turn_action(
        "steer", id, None, Some(PreparedTurnActionOptions {
            custom_message: Some(family_message(id, relationship)), ..Default::default()
        }),
    );
    let first = create("first", "parent");
    for relationship in ["parent", "sibling", "child"] {
        assert!(compatible_family_message_actions(&first, &create("next", relationship)));
    }
    let mut follow_first = first.clone();
    let mut follow_next = create("follow-next", "sibling");
    follow_first.delivery = DeliveryPolicy::WhenRunIdle;
    follow_next.delivery = DeliveryPolicy::WhenRunIdle;
    assert!(compatible_family_message_actions(&follow_first, &follow_next));
    let mut excluded = Vec::new();
    let mut next = create("lane", "sibling");
    next.delivery = DeliveryPolicy::WhenRunIdle;
    excluded.push(next);
    let mut next = create("wake", "sibling");
    next.wake = WakePolicy::ExternalResume;
    excluded.push(next);
    let mut next = create("priority", "sibling");
    next.priority = Some(crate::core::session_action_store::SessionActionPriority::Pinned);
    excluded.push(next);
    let mut next = create("suppressed", "sibling");
    next.suppress_autonomous_continuation = Some(true);
    excluded.push(next);
    let mut next = create("execution", "sibling");
    if let QueuedActionPayload::Turn(turn) = &mut next.payload {
        turn.execution_policy.run_before_agent_start = !turn.execution_policy.run_before_agent_start;
    }
    excluded.push(next);
    excluded.push(create("unknown-origin", "unknown"));
    excluded.push(session.create_prepared_turn_action("steer", "human", None, None));
    excluded.push(session.create_session_command_action(
        "/compact", SessionSlashCommand { name: "compact".into(), args: String::new(), text: "/compact".into() },
        None, "steer", None, None,
    ));
    let mut next = create("other-custom-type", "parent");
    if let QueuedActionPayload::Turn(turn) = &mut next.payload {
        if let DeliveryMessage::Custom(custom) = &mut turn.base.records[0].message {
            custom.custom_type = "rlm_continuation".into();
        }
    }
    excluded.push(next);
    for next in excluded {
        assert!(!compatible_family_message_actions(&first, &next), "must not absorb {}", action_text(&next));
    }
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn family_batch_does_not_skip_a_contiguous_queue_barrier_or_replay_a_cancelled_message() {
    let (session, _root) = fixture().await;
    session.set_steering_mode("one-at-a-time");
    let (_, first) = enqueue(&session, "first", Some(family_message("first", "parent")));
    let (_, barrier) = enqueue(&session, "unknown-barrier", Some(family_message("unknown-barrier", "unknown")));
    let (cancelled, cancelled_ticket) = enqueue(&session, "cancelled", Some(family_message("cancelled", "sibling")));
    let (_, last) = enqueue(&session, "last", Some(family_message("last", "child")));
    session.cancel_session_actions(&|action| action.id == cancelled.id, "cancel queued fixture", None);
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let seen = contexts.clone();
    session.agent.set_stream_fn(Arc::new(move |model, context, _| {
        seen.lock().unwrap().push(serde_json::to_string(&context.messages).unwrap());
        Box::pin(async move { response(model) })
    }));
    pump(&session).await;
    for ticket in [first, barrier, last] { ticket.ticket.completed.clone().await.unwrap(); }
    assert!(cancelled_ticket.ticket.delivered.clone().await.is_err());
    let seen = contexts.lock().unwrap();
    assert_eq!(seen.len(), 3, "do not scan past an incompatible queued message");
    assert!(!seen[0].contains("unknown-barrier"));
    assert!(!seen[1].contains("child instruction last"));
    assert!(seen.iter().all(|context| !context.contains("sibling instruction cancelled")));
    drop(seen);
    assert_eq!(count_input(&session.session_manager.lock().unwrap().get_entries(), "sibling instruction cancelled"), 0);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn family_batch_persistence_failure_rejects_every_receipt_before_provider_work() {
    let (session, _root) = fixture().await;
    session.set_steering_mode("one-at-a-time");
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    std::fs::create_dir(&path).unwrap();
    let mut tickets = Vec::new();
    for relationship in ["parent", "sibling", "child"] {
        tickets.push(enqueue(&session, relationship, Some(family_message(relationship, relationship))).1);
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { response(model) })
    }));
    pump(&session).await;
    for ticket in tickets {
        assert!(ticket.ticket.delivered.clone().await.unwrap_err().contains("persistence failed"));
        assert!(ticket.ticket.completed.clone().await.is_err());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(session.action_store.lock().unwrap().unfinished_actions(None).is_empty());
    std::fs::remove_dir(&path).unwrap();
    session.session_manager.lock().unwrap().flush_now().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    for relationship in ["parent", "sibling", "child"] {
        assert!(count_input(&reopened.get_entries(), &format!("{relationship} instruction {relationship}")) <= 1);
    }
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_before_delivery_never_receives_durable_ack_or_transcript_entry() {
    let (session, _root) = fixture().await;
    let (action, ticket) = enqueue(&session, "cancel-before-delivery", Some(child_report("cancelled-report")));
    let mut selected = session.action_store.lock().unwrap().select_first().unwrap().unwrap();
    transition_session_action(&mut selected, ActionLifecycle::Preparing { preparation: None }, &TransitionOptions::default()).unwrap();
    session.action_store.lock().unwrap().update_action(&selected).unwrap();
    transition_session_action(&mut selected, ActionLifecycle::Committing, &TransitionOptions::default()).unwrap();
    session.action_store.lock().unwrap().update_action(&selected).unwrap();
    session.cancel_session_actions(&|candidate| candidate.id == action.id, "cancel fixture", Some(vec![selected]));
    let message = agent_message_from_delivery(&primary_delivery_record(&action).unwrap().message);
    session.handle_agent_event(AgentEvent::MessageStart { message: message.clone() });
    session.handle_agent_event(AgentEvent::MessageEnd { message });
    session.await_agent_event_queue().await;
    assert!(ticket.ticket.delivered.clone().await.is_err());
    assert_eq!(count_input(&session.session_manager.lock().unwrap().get_entries(), "cancelled-report"), 0);
    assert!(session.settled_turn_delivery_error(&[action]).is_none());
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_retry_reuses_the_same_durable_input_without_duplicate_transcript_entries() {
    use pi_ai::utils::stream_failure::{record_stream_failure, StreamFailureError, StreamFailureInfo, ThrownStreamError};
    let (session, _root) = fixture().await;
    session.settings_manager.lock().unwrap().set_retry_enabled(true);
    let (_, ticket) = enqueue(&session, "one input across provider retry", None);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let index = counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if index > 0 { return response(model); }
            let stream = AssistantMessageEventStream::new();
            let mut message = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), now_ms_i64());
            message.stop_reason = "error".into();
            message.error_message = Some("isolated rate limit".into());
            let failure = StreamFailureError::new("isolated rate limit", StreamFailureInfo {
                kind: "rate_limit".into(), ..Default::default()
            });
            record_stream_failure(&model, &mut message, &ThrownStreamError::Failure(&failure));
            stream.push(AssistantMessageEvent::Error { reason: "error".into(), error: message });
            stream.end(None);
            stream
        })
    }));
    pump(&session).await;
    ticket.ticket.delivered.clone().await.unwrap();
    ticket.ticket.completed.clone().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    assert_eq!(count_input(&reopened.get_entries(), "one input across provider retry"), 1);
    session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn new_dispatch_cannot_overtake_delayed_previous_assistant_or_tool_persistence() {
    let (session, _root) = fixture().await;
    let release = CancellationToken::new();
    let held = release.clone();
    session.push_agent_event_task(async move { held.cancelled().await; });
    let old_assistant = AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new("older-assistant-entry"))],
        ..Default::default()
    };
    let old_tool = pi_ai::types::ToolResultMessage::new("older-call", "fixture", vec![
        pi_ai::types::ImageOrTextContent::Text(TextContent::new("older-tool-entry")),
    ], false, 1000);
    session.handle_agent_event(AgentEvent::MessageEnd { message: AgentMessage::Message(Message::Assistant(old_assistant)) });
    session.handle_agent_event(AgentEvent::MessageEnd { message: AgentMessage::Message(Message::ToolResult(old_tool)) });
    let (_, ticket) = enqueue(&session, "new-ordered-input", None);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { response(model) })
    }));
    let owner = session.clone();
    let task = tokio::spawn(async move { pump(&owner).await });
    tokio::task::yield_now().await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(count_input(&session.session_manager.lock().unwrap().get_entries(), "new-ordered-input"), 0);
    release.cancel();
    task.await.unwrap();
    ticket.ticket.completed.clone().await.unwrap();
    let path = session.session_manager.lock().unwrap().get_session_file().unwrap();
    let reopened = SessionManager::open(&path, None, None).unwrap();
    let entries: Vec<_> = reopened.get_entries().iter().map(|entry| serde_json::to_string(entry).unwrap()).collect();
    let position = |text: &str| entries.iter().position(|entry| entry.contains(text)).unwrap();
    assert!(position("older-assistant-entry") < position("older-tool-entry"));
    assert!(position("older-tool-entry") < position("new-ordered-input"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    session.dispose_async(Some(false)).await;
}
