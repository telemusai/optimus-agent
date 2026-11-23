use super::*;

#[tokio::test]
async fn viewer_delivery_detach_releases_a_stalled_listener() {
    use crate::main_entry::MainEntryDaemonTransport;
    use crate::modes::daemon::daemon_client::DaemonClient;
    let client = DaemonClient::create("unused-viewer-detach-fixture");
    let connection = Arc::new(DaemonAgentConnection::new(
        Arc::new(MainEntryDaemonTransport::new(client.clone())),
        "chat".into(),
        DaemonAgentConnectionOptions {
            defer_session_events: true,
            ..Default::default()
        },
    ));
    let unsubscribe_transport = bind(&connection).unwrap();
    let entered = Arc::new(Notify::new());
    let started = entered.clone();
    let (sender, mut released) = tokio::sync::mpsc::channel::<()>(1);
    let remove_listener = connection.subscribe(Arc::new(move |_| {
        let held = sender.clone();
        started.notify_one();
        Box::pin(async move {
            std::future::pending::<()>().await;
            drop(held);
        })
    }));
    client
        .receive_test_line(r#"{"type":"heartbeats_changed"}"#)
        .await;
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    remove_listener();
    drop(remove_listener);
    unsubscribe_transport();
    drop(unsubscribe_transport);
    assert!(tokio::time::timeout(Duration::from_secs(2), released.recv())
        .await
        .unwrap()
        .is_none());
    assert!(!*connection.disposed.lock().unwrap());
}

fn key(generation: &str, sequence: i64) -> Option<UpdateKey> {
    Some(UpdateKey::Assistant {
        session: "chat".into(),
        generation: generation.into(),
        timestamp: 100,
        sequence,
    })
}

#[test]
fn viewer_delivery_folds_only_adjacent_newer_snapshots() {
    let mut inbox = Inbox::default();
    for i in 0..10_000 {
        inbox.push(i.to_string().as_bytes(), key("a", i)).unwrap();
    }
    assert!(inbox.memory.is_empty() && inbox.segments.is_empty());
    inbox.push(b"older", key("a", 21)).unwrap();
    inbox.push(b"sequence gap", key("a", 24)).unwrap();
    inbox.push(b"new generation", key("b", 1)).unwrap();
    inbox.push(b"final message", None).unwrap();
    inbox.push(b"unsequenced", None).unwrap();
    inbox
        .push(b"heartbeat", Some(UpdateKey::Heartbeat))
        .unwrap();
    inbox.push(b"refresh", Some(UpdateKey::Heartbeat)).unwrap();
    let mut delivered = vec![];
    while let Some(bytes) = inbox.pop().unwrap() {
        delivered.push(String::from_utf8(bytes).unwrap());
    }
    assert_eq!(
        delivered,
        [
            "9999",
            "older",
            "sequence gap",
            "new generation",
            "final message",
            "unsequenced",
            "refresh"
        ]
    );
}

#[test]
fn viewer_delivery_rolls_temp_segments_and_reclaims_them_in_order() {
    let mut inbox = Inbox::default();
    let size = 256 * 1024;
    for i in 0..80 {
        inbox.push(&vec![i; size], None).unwrap();
    }
    assert!(inbox.memory_bytes <= MEMORY_BYTES);
    assert!(inbox.segments.len() > 1);
    let old_disk = inbox.disk_bytes;
    for i in 0..55 {
        assert_eq!(inbox.pop().unwrap().unwrap(), vec![i; size]);
    }
    assert!(inbox.disk_bytes < old_disk);
    for i in 80..100 {
        inbox.push(&vec![i; size], None).unwrap();
    }
    for i in 55..100 {
        assert_eq!(inbox.pop().unwrap().unwrap(), vec![i; size]);
    }
    assert!(inbox.pop().unwrap().is_none());
    assert!(inbox.segments.is_empty());
    assert_eq!(inbox.disk_bytes, 0);
}

#[test]
fn viewer_delivery_storage_limit_releases_backlog_and_records_failure() {
    let mut inbox = Inbox::default();
    inbox.spill(b"saved first").unwrap();
    inbox.disk_bytes = SPILL_BYTES - 8;
    let error = inbox.spill(b"cannot fit").unwrap_err();
    inbox.fail(error);
    assert!(inbox.failure.as_deref().unwrap().contains("128 MiB"));
    assert!(inbox.segments.is_empty());
}

fn chat_event(session: &str, sequence: i64, event: Value) -> String {
    json!({"type":"session_event","activeSessionId":session,"event":event,
        "meta":{"sequence":sequence,"cursor":{"generation":"run","sequence":sequence}}})
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn viewer_delivery_stalled_terminal_keeps_every_final_message_and_resumes() {
    exercise_stalled_viewer(1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "sustained native viewer delivery stress test"]
async fn viewer_delivery_sustained_stalls_keep_one_connection_and_exact_transcript() {
    exercise_stalled_viewer(128).await;
}

async fn exercise_stalled_viewer(rounds: i64) {
    use crate::main_entry::MainEntryDaemonTransport;
    use crate::modes::daemon::daemon_client::DaemonClient;
    use pi_ai::providers::faux::{faux_assistant_message, FauxAssistantContent};

    let client = DaemonClient::create("unused-viewer-delivery-fixture");
    let connection = Arc::new(DaemonAgentConnection::new(
        Arc::new(MainEntryDaemonTransport::new(client.clone())),
        "chat".into(),
        DaemonAgentConnectionOptions {
            defer_session_events: true,
            ..Default::default()
        },
    ));
    connection.bind_transport(&connection);
    let permit = Arc::new(tokio::sync::Semaphore::new(0));
    let gate = permit.clone();
    let (sent, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let _unsubscribe = connection.subscribe(Arc::new(move |event| {
        let first = matches!(event, AgentConnectionEvent::HeartbeatsChanged);
        sent.send(event).unwrap();
        let gate = gate.clone();
        Box::pin(async move {
            if first {
                gate.acquire().await.unwrap().forget();
            }
        })
    }));
    for round in 0..rounds {
        let base = round * 1400;
        client
            .receive_test_line(r#"{"type":"heartbeats_changed"}"#)
            .await;
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(10), observed.recv())
                .await
                .unwrap()
                .unwrap(),
            AgentConnectionEvent::HeartbeatsChanged
        ));

        let partial = faux_assistant_message(FauxAssistantContent::Text("streaming".into()), None);
        let update = json!({"type":"message_update","message":partial,
        "assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"g","partial":partial}});
        for n in 1..=1000 {
            client
                .receive_test_line(&chat_event("other", base + n, update.clone()))
                .await;
            client
                .receive_test_line(&chat_event("chat", base + n, update.clone()))
                .await;
        }
        for n in 1..=400 {
            client.receive_test_line(&chat_event("chat", base + 1000 + n, json!({
            "type":"message_end", "message":{"role":"user","content":(base+n).to_string(),"timestamp":base+n}
        }))).await;
        }
        // Global shutdown intent is observed on the reader while the UI is still paused.
        client
            .receive_test_line(r#"{"type":"daemon_closing","reason":"shutdown"}"#)
            .await;
        assert!(*connection.shutdown_restart_pending.lock().unwrap());
        assert!(observed.try_recv().is_err());

        permit.add_permits(1);
        tokio::time::timeout(Duration::from_secs(10), async {
            assert!(matches!(
                observed.recv().await.unwrap(),
                AgentConnectionEvent::SessionEvent {
                    event: AgentConnectionSessionEvent::MessageUpdate { .. }
                }
            ));
            for n in 1..=400 {
                let Some(AgentConnectionEvent::SessionEvent {
                    event: AgentConnectionSessionEvent::MessageEnd { message },
                }) = observed.recv().await
                else {
                    panic!("expected final message {n}");
                };
                assert_eq!(
                    serde_json::to_value(message).unwrap()["content"],
                    (base + n).to_string()
                );
            }
        })
        .await
        .expect("viewer must catch up");
        assert_eq!(
            *connection.last_event_sequence.lock().unwrap(),
            Some(base + 1400)
        );
        assert!(observed.try_recv().is_err());
    }
    connection
        .unsubscribe_daemon_messages
        .lock()
        .unwrap()
        .take()
        .unwrap()();
}
