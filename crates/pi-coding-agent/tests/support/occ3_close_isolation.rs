use super::*;

/// Closing a third session sharing the connector must not revoke an admitted delivery.
#[tokio::test]
async fn unrelated_close_preserves_admitted_provider_consume() {
    let pf = build_family(FamilyConfig {
        source_tool: SourceTool::QueueController,
        ..Default::default()
    }).await;
    let unrelated = Occ3NativeSession::enroll(
        "unrelated-session", &pf.receiver_agent.state().model, pf.connector.clone()
    ).await.expect("unrelated enrollment");
    arm_and_prompt_source(&pf).await;
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
    cleanup(&pf).await;
}
