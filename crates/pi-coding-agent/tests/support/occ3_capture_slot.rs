// The capture slot belongs to the capture/publication, not the sender's lifetime.
#[tokio::test]
async fn occ3_admitted_source_allows_other_session_to_arm_while_source_is_alive() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    }).await;
    arm_and_prompt_source(&pf).await;
    assert!(pf.queued_delivery.lock().unwrap().is_some(), "source handoff admitted");
    let tool = pf.receiver.handoff_tool(pf.family.controller("receiver-session").unwrap());
    let armed = tokio::time::timeout(Duration::from_secs(2), (tool.execute)(
        "other-session-arm".into(),
        json!({"receiver_role":"parent", "receiver_name":"source"}),
        None,
        None,
    )).await;
    // Cleanup even on the red path so the private fixture does not leak.
    cleanup(&pf).await;
    let result = armed.expect("completed sender must not monopolize capture capacity")
        .expect("other session arms");
    assert_eq!(result.details["status"], "capture_armed");
}

#[tokio::test]
async fn occ3_capture_capacity_wait_cancels_without_arming_or_sending() {
    let pf = build_family(FamilyConfig::default()).await;
    pf.source.arm_source(NativeSourceBudget {
        retained_count_limit: 128,
        max_capture_bytes: 1048576,
        deadline: Instant::now() + Duration::from_secs(30),
        max_response_bytes: 65536,
    }).unwrap();
    let tool = pf.receiver.handoff_tool(pf.family.controller("receiver-session").unwrap());
    let signal = tokio_util::sync::CancellationToken::new();
    let wait = (tool.execute)(
        "cancelled-other-arm".into(),
        json!({"receiver_role":"parent", "receiver_name":"source"}),
        Some(signal.clone()),
        None,
    );
    // Poll the wait first, then cancel while the source still owns capacity.
    let (result, ()) = tokio::join!(biased; wait, async { signal.cancel(); });
    assert!(result.unwrap_err().to_string().contains("occ3_capture_capacity_cancelled"));
    assert!(pf.receiver.audit()["source"].is_null());
    let (entries, _) = pf.receiver.sender_queue_snapshot().unwrap();
    assert_eq!(entries[0].state.as_str(), "pending", "no arm transition");
    assert!(pf.chat_bodies().is_empty(), "waiting never starts inference");
    cleanup(&pf).await;
}
