//! Painted prompt receipts with held acknowledgements and authoritative events.
use super::*;
use super::tests::{RecordingConnection, SpeedRecorder};
use pi_ai::types::{UserContent, UserMessage};

fn user(text: &str) -> AgentMessage {
    UserMessage::new(UserContent::Text(text.into()), 1).into()
}

fn confirm(h: &ui_tests::FrameHarness, text: &str) {
    apply_event(&h.mode, &h.transcript, wire::AgentConnectionSessionEvent::MessageEnd { message: user(text) });
}

#[tokio::test]
async fn long_chat_paints_submitted_text_before_a_held_daemon_request() {
    let h = ui_tests::FrameHarness::new("immediate-long-chat");
    let fake = Arc::new(RecordingConnection::default());
    let connection: Arc<dyn wire::AgentConnection> = fake.clone();
    let mut history = native_history::HistoryRuntime::new(connection.clone());
    let messages = (0..10_000).map(|index| user(&format!("History {index}: {}", "context ".repeat(40)))).collect();
    apply_history_snapshot(None, messages, None, &h.transcript, &h.editor, &mut history,
        Some(native_history::ViewportFill { width: 80, rows: 24 }));
    h.paint();
    let (reply, wait) = tokio::sync::oneshot::channel();
    fake.prompt_replies.lock().unwrap().push_back(wait);
    let (send, receive) = mpsc::channel();
    let recorder = Arc::new(SpeedRecorder::default());
    let mut metrics = native_metrics::UiMetrics::with_recorder("immediate-long-chat", recorder.clone());
    let started = Instant::now();
    submit_from_editor(&connection, &send, "IMMEDIATE_NEW_PROMPT".into(), false, &mut metrics, &h.transcript);
    let frame = h.paint().join("\n");
    eprintln!("10,000-message chat submit-to-paint: {:.2}ms", started.elapsed().as_secs_f64() * 1000.0);
    assert!(fake.calls().is_empty(), "paint must precede even starting the daemon request");
    assert!(frame.contains("IMMEDIATE_NEW_PROMPT") && frame.contains("not yet confirmed"), "{frame}");
    assert!(h.ui.borrow().get_scroll_info().unwrap().following);
    assert!(h.transcript.borrow().rows.is_empty(), "receipt must not invent authoritative history");
    metrics.receipt_rendered();
    tokio::task::yield_now().await;
    assert!(receive.try_recv().is_err());
    assert!(h.paint().join("\n").contains("IMMEDIATE_NEW_PROMPT"));
    reply.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    let HostEvent::SubmissionReply(ticket, result) = receive.try_recv().unwrap() else { panic!("identified acknowledgement") };
    assert!(apply_submission_reply(&h.mode, &h.transcript, &mut metrics, &ticket, result));
    h.transcript.borrow_mut().local_sending_count = metrics.pending_count();
    let accepted = h.paint().join("\n");
    assert!(accepted.contains("IMMEDIATE_NEW_PROMPT") && accepted.contains("Accepted"), "{accepted}");
    confirm(&h, "IMMEDIATE_NEW_PROMPT");
    let confirmed = h.paint().join("\n");
    assert_eq!(confirmed.matches("IMMEDIATE_NEW_PROMPT").count(), 1, "{confirmed}");
    assert!(!confirmed.contains("Accepted") && !confirmed.contains("Sending"));
    assert!(!serde_json::to_string(&*recorder.0.lock().unwrap()).unwrap().contains("IMMEDIATE_NEW_PROMPT"));
}

#[tokio::test]
async fn identical_prompts_reconcile_once_with_messages_before_or_after_ack() {
    let h = ui_tests::FrameHarness::new("receipt-order");
    let fake = Arc::new(RecordingConnection::default());
    let connection: Arc<dyn wire::AgentConnection> = fake.clone();
    let (send, receive) = mpsc::channel();
    let mut metrics = native_metrics::UiMetrics::with_recorder("receipt-order", Arc::new(SpeedRecorder::default()));
    let mut replies = Vec::new();
    for _ in 0..2 {
        let (reply, wait) = tokio::sync::oneshot::channel();
        fake.prompt_replies.lock().unwrap().push_back(wait);
        replies.push(reply);
        submit_from_editor(&connection, &send, "SAME_PROMPT".into(), false, &mut metrics, &h.transcript);
    }
    tokio::task::yield_now().await;
    assert_eq!(h.paint().join("\n").matches("SAME_PROMPT").count(), 2);
    confirm(&h, "SAME_PROMPT");
    assert_eq!(h.paint().join("\n").matches("SAME_PROMPT").count(), 2);
    for reply in replies {
        reply.send(Ok(())).unwrap();
        tokio::task::yield_now().await;
        let HostEvent::SubmissionReply(ticket, result) = receive.try_recv().unwrap() else { panic!("reply") };
        assert!(apply_submission_reply(&h.mode, &h.transcript, &mut metrics, &ticket, result));
    }
    h.transcript.borrow_mut().local_sending_count = 0;
    confirm(&h, "SAME_PROMPT");
    let frame = h.paint().join("\n");
    assert_eq!(frame.matches("SAME_PROMPT").count(), 2, "{frame}");
    assert!(!frame.contains("Accepted") && !frame.contains("Sending"));
}

#[test]
fn queue_confirmation_and_attachment_reset_remove_only_local_previews() {
    let h = ui_tests::FrameHarness::new("receipt-queue");
    let mut metrics = native_metrics::UiMetrics::with_recorder("receipt-queue", Arc::new(SpeedRecorder::default()));
    let ticket = metrics.begin_submission();
    h.transcript.borrow_mut().local_receipts.begin(ticket.id(), "QUEUED_PROMPT", &h.mode.borrow());
    let queue = vec!["QUEUED_PROMPT".to_string()];
    h.mode.borrow_mut().patch_connection_queue(|state| state.session_actions.steering = queue.clone());
    assert!(h.transcript.borrow_mut().local_receipts.queue(queue.clone().into_iter()));
    assert!(!h.transcript.borrow_mut().local_receipts.queue(queue.into_iter()), "repeated snapshot is unchanged");
    assert!(apply_submission_reply(&h.mode, &h.transcript, &mut metrics, &ticket, Ok(())));
    let queued = h.paint().join("\n");
    assert_eq!(queued.matches("QUEUED_PROMPT").count(), 1, "{queued}");
    assert!(!queued.contains("Accepted"));
    h.mode.borrow_mut().patch_connection_queue(|state| state.session_actions.steering.clear());
    assert!(h.transcript.borrow_mut().local_receipts.queue(std::iter::empty()));
    confirm(&h, "QUEUED_PROMPT");
    let old = metrics.begin_submission();
    h.transcript.borrow_mut().local_receipts.begin(old.id(), "OLD_LOCAL_PROMPT", &h.mode.borrow());
    metrics.reset_attachment();
    h.transcript.borrow_mut().local_receipts.clear();
    assert!(!apply_submission_reply(&h.mode, &h.transcript, &mut metrics, &old, Err("old rejection".into())));
    let frame = h.paint().join("\n");
    assert!(frame.contains("QUEUED_PROMPT"));
    assert!(!frame.contains("OLD_LOCAL_PROMPT") && !frame.contains("old rejection"));
}
