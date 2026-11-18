use super::*;

/// Arm through the public tool so capture and sender-queue ownership agree.
async fn arm_queued_source_and_prompt(pf: &ProofFamily) {
    let tool = pf.source.handoff_tool(pf.family.controller("source-session").unwrap());
    let result = (tool.execute)(
        "queue-proof-arm".into(),
        json!({"receiver_role": "child", "receiver_name": "receiver"}),
        None,
        None,
    ).await.expect("initial handoff arms capture");
    assert_eq!(result.details["status"], "capture_armed");
    let (entries, _) = pf.source.sender_queue_snapshot().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].state.as_str(), "armed");
    pf.source_agent.prompt(PromptInput::Text {
        input: "Retained-source fixture prompt. Call occ3_handoff to child receiver. Arguments contain only role and name.".into(),
        images: Vec::new(),
    }).await.expect("source prompt completes");
    let (entries, _) = pf.source.sender_queue_snapshot().unwrap();
    assert_eq!(entries.len(), 1, "sender queue must own admitted delivery");
    assert_eq!(entries[0].state.as_str(), "admitted");
}

/// Consumption completes the sender queue even after an unrelated session closes.
/// This fixture does not populate the ready cache; the bug-2 cache regression is
/// `unrelated_close_preserves_shared_ready_connector`, not this provider proof.
#[tokio::test]
async fn consumed_delivery_completes_sender_queue_after_unrelated_close() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    }).await;
    let unrelated = Occ3NativeSession::enroll(
        "unrelated-session", &pf.receiver_agent.state().model, pf.connector.clone()
    ).await.expect("unrelated enrollment");
    pf.receiver.install_delivery_restamp(Arc::new(pf.receiver_agent.clone()))
        .expect("install production restamp listener");
    arm_queued_source_and_prompt(&pf).await;
    assert_eq!(pf.receiver.audit()["receiver"][0]["status"], "ADMITTED");
    unrelated.close().await.expect("close unrelated session");
    prompt_receiver_with_delivery(&pf).await;
    assert!(pf.receiver_agent.state().error_message.is_none(), "{:?}", pf.receiver_agent.state().error_message);
    assert_eq!(pf.occ3_bodies().len(), 1, "actual provider attachment request");
    assert_eq!(pf.receiver.audit()["receiver"][0]["status"], "CONSUMED");
    assert_eq!(pf.receiver.audit()["receiver"][0]["lease_state"], "Consumed");
    let receipts = pf.state.records.lock().unwrap().receiver_receipts.clone();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0]["consumed"], true);
    assert_eq!(receipts[0]["injected_state"], true);
    assert_eq!(receipts[0]["fallback_used"], false);
    assert_eq!(receipts[0]["attachment"], pf.occ3_bodies()[0]["occ3"]);
    let (entries, _) = pf.source.sender_queue_snapshot().unwrap();
    let notices = pf.source.take_pending_queue_notices();
    cleanup(&pf).await;
    assert!(entries.is_empty(), "consumption must advance the sender queue: {entries:?}");
    assert_eq!(notices.iter().filter(|notice| notice.state == "completed").count(), 1);
    assert!(!notices.iter().any(|notice| matches!(notice.state.as_str(), "uncertain" | "fallback_pending")));
}

/// Observe the queue during TurnEnd, before AgentEnd can mask a missing dispatch.
#[tokio::test]
async fn occ3_restamp_turn_end_completes_sender_with_receipt_digest() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    }).await;
    pf.receiver.install_delivery_restamp(Arc::new(pf.receiver_agent.clone()))
        .expect("install production restamp listener");
    arm_queued_source_and_prompt(&pf).await;
    let _setup = pf.source.take_pending_queue_notices();
    let observed = Arc::new(Mutex::new(None));
    let observed_callback = observed.clone();
    let source = pf.source.clone();
    // Agent awaits subscriptions in order: this runs after the real listener.
    let _subscription = pf.receiver_agent.subscribe(Arc::new(move |event, _| {
        let observed = observed_callback.clone();
        let source = source.clone();
        Box::pin(async move {
            if matches!(event, pi_agent_core::types::AgentEvent::TurnEnd { .. }) {
                *observed.lock().unwrap() = Some((
                    source.sender_queue_snapshot().unwrap().0,
                    source.take_pending_queue_notices(),
                ));
            }
        })
    }));
    prompt_receiver_with_delivery(&pf).await;
    let observed = observed.lock().unwrap().take().expect("real TurnEnd observed");
    let late_notices = pf.source.take_pending_queue_notices();
    cleanup(&pf).await;
    assert!(observed.0.is_empty(), "TurnEnd must advance sender queue: {:?}", observed.0);
    assert_eq!(observed.1.len(), 1, "one terminal sender notice");
    assert_eq!(observed.1[0].state, "completed", "receipt digest must reach sender");
    assert!(late_notices.is_empty(), "AgentEnd must not duplicate consumption");
}
