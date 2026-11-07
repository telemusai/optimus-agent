//! Offline native attachment control-plane regressions. No model or GPU calls.
//! Native provider/agent fixtures separately exercise actual request dispatch.

use std::net::TcpListener;
use std::time::Duration;

use pi_ai::inference_attachment::{InferenceStateLeaseState, InferenceStateTarget};
use pi_coding_agent::core::occ::occ3::inference::{
    inference_attachment, permits_text_fallback, Occ3InferenceInbox, PrepareInferenceRequest,
    VerifiedInferenceRuntime,
};
use pi_coding_agent::core::occ::occ3::mock::MockOcc3Connector;
use pi_coding_agent::core::occ::occ3::{
    build_envelope, CaptureRequest, ConsumeRequest, Occ3Connector, Occ3DeliveryStatus,
    Occ3Envelope, Occ3Error, Occ3Method, Occ3ModelIdentity,
};
use pi_coding_agent::core::occ3_connectors::sglang::{SglangOcc3Config, SglangOcc3Connector};

fn identity() -> Occ3ModelIdentity {
    Occ3ModelIdentity {
        model_id: "fixture-model".into(),
        architecture: "fixture-arch".into(),
        checkpoint_revision: "fixture-revision".into(),
        weight_fingerprint: "fixture-weights".into(),
        tokenizer_fingerprint: "fixture-tokenizer".into(),
        chat_template_fingerprint: "fixture-template".into(),
        quantisation: "fixture-quantisation".into(),
        tensor_geometry: "fixture-geometry".into(),
        kv_layout: "fixture-layout".into(),
        latent_method: "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1".into(),
        runtime: "fixture-runtime-v1".into(),
    }
}

fn capture_request() -> CaptureRequest {
    CaptureRequest {
        message_id: "message-1".into(),
        sender_session_id: "sender".into(),
        receiver_session_id: "receiver".into(),
        method: Occ3Method::KvHandoff,
        capture_context: "SOURCE_ONLY_SECRET_FIXTURE".into(),
        ttl: Duration::from_secs(60),
    }
}

async fn fixture() -> (
    MockOcc3Connector,
    PrepareInferenceRequest,
    VerifiedInferenceRuntime,
) {
    let connector = MockOcc3Connector::new(identity(), vec![Occ3Method::KvHandoff]);
    let handle = connector.capture_latent(capture_request()).await.unwrap();
    let envelope = build_envelope(
        "message-1".into(),
        "sender".into(),
        "receiver".into(),
        identity().fingerprint(),
        connector.runtime_id(),
        "mock-occ3".into(),
        Occ3Method::KvHandoff,
        handle,
        "OCC/1\nTASK=fixture\nKIND=RESULT\nSTATUS=done".into(),
        "occ1".into(),
    )
    .unwrap();
    let runtime = VerifiedInferenceRuntime {
        model_identity: identity(),
        runtime_id: connector.runtime_id(),
        generation: 0,
        methods: vec![Occ3Method::KvHandoff],
    };
    let request = PrepareInferenceRequest {
        envelope,
        request_id: "inference-1".into(),
        receiver_session_id: "receiver".into(),
        receiver_identity: identity(),
        now_epoch: 1,
        target: InferenceStateTarget {
            provider: "fixture-provider".into(),
            model_id: "fixture-model".into(),
            base_url: "http://127.0.0.1:1/v1".into(),
        },
    };
    (connector, request, runtime)
}

fn admitted(envelope: Occ3Envelope) -> Occ3InferenceInbox {
    let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
    let id = envelope.message_id.clone();
    assert!(inbox.admit(envelope, &id, "sender", 1).unwrap());
    inbox
}


/// Method identity is the full dotted id, never just the coarse family: two
/// complete identities of one family differ, and `bind_lease` compares the
/// attachment's full method id against the verified identity's full method id
/// by exact string equality (campaign spec 14/18).
#[tokio::test]
async fn bind_lease_binds_a_full_method_id_only_against_its_own_identity() {
    let method_a = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
    let method_b = "statebridge.glm53.calibrated-lowrank-procrustes.v1";
    let mut identity_a = identity();
    identity_a.latent_method = method_a.into();
    let mut identity_b = identity();
    identity_b.latent_method = method_b.into();
    // One family either way: the family alone must never pass the bind.
    assert_eq!(
        pi_coding_agent::core::occ3_connectors::method_family(method_a),
        pi_coding_agent::core::occ3_connectors::method_family(method_b)
    );

    let connector = MockOcc3Connector::new(identity_a.clone(), vec![Occ3Method::StatebridgePrefix]);
    let handle = connector
        .capture_latent(CaptureRequest {
            message_id: "message-1".into(),
            sender_session_id: "sender".into(),
            receiver_session_id: "receiver".into(),
            method: Occ3Method::StatebridgePrefix,
            capture_context: "SOURCE_ONLY_SECRET_FIXTURE".into(),
            ttl: Duration::from_secs(60),
        })
        .await
        .unwrap();
    let envelope = build_envelope(
        "message-1".into(),
        "sender".into(),
        "receiver".into(),
        identity_a.fingerprint(),
        connector.runtime_id(),
        "mock-occ3".into(),
        Occ3Method::StatebridgePrefix,
        handle,
        "OCC/1\nTASK=fixture\nKIND=RESULT\nSTATUS=done".into(),
        "occ1".into(),
    )
    .unwrap();
    let target = InferenceStateTarget {
        provider: "fixture-provider".into(),
        model_id: "fixture-model".into(),
        base_url: "http://127.0.0.1:1/v1".into(),
    };
    let runtime = VerifiedInferenceRuntime {
        model_identity: identity_a.clone(),
        runtime_id: connector.runtime_id(),
        generation: 0,
        methods: vec![Occ3Method::StatebridgePrefix],
    };
    let request = PrepareInferenceRequest {
        envelope: envelope.clone(),
        request_id: "inference-1".into(),
        receiver_session_id: "receiver".into(),
        receiver_identity: identity_a.clone(),
        now_epoch: 1,
        target: target.clone(),
    };
    // The attachment carries the full dotted method id, not the family.
    let attachment = inference_attachment(&request, "mock-occ3", &runtime).unwrap();
    assert_eq!(attachment.method, method_a);
    let lease = pi_ai::inference_attachment::InferenceStateLease::new(target, attachment).unwrap();

    let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(inbox.admit(envelope, "message-1", "sender", 1).unwrap());
    assert!(inbox.claim(&["message-1".into()], &[], 1).unwrap().is_some());
    // The same family under a different full method id must not bind.
    assert!(matches!(
        inbox.bind_lease(&lease, &identity_b),
        Err(Occ3Error::Unauthorized)
    ));
    // The exact full method id binds.
    assert!(inbox.bind_lease(&lease, &identity_a).is_ok());
}

#[tokio::test]
async fn native_attachment_is_handle_only_and_not_detached_consumption() {
    let (connector, request, runtime) = fixture().await;
    let attachment = inference_attachment(&request, "mock-occ3", &runtime).unwrap();
    let wire = serde_json::to_string(&attachment).unwrap();
    assert!(!wire.contains("SOURCE_ONLY_SECRET_FIXTURE"));
    assert!(!wire.contains("control_summary"));
    assert_eq!(attachment.receiver_session_id, "receiver");
    let lease = connector.prepare_inference(request.clone()).await.unwrap();
    assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
    assert_eq!(connector.live_states(), 1, "preparation is not consumption");
    assert!(matches!(
        connector.prepare_inference(request).await,
        Err(Occ3Error::HandleAlreadyConsumed { .. })
    ));
}

#[tokio::test]
async fn strict_identity_model_checkpoint_tokenizer_and_unknown_rejected() {
    let (_, request, runtime) = fixture().await;
    for field in [
        "model",
        "checkpoint",
        "weights",
        "tokenizer",
        "geometry",
        "unknown",
    ] {
        let mut changed = request.clone();
        match field {
            "model" => changed.receiver_identity.model_id = "different-model".into(),
            "checkpoint" => {
                changed.receiver_identity.checkpoint_revision = "different-checkpoint".into()
            }
            "weights" => changed.receiver_identity.weight_fingerprint = "different-weights".into(),
            "tokenizer" => {
                changed.receiver_identity.tokenizer_fingerprint = "different-tokenizer".into()
            }
            "geometry" => changed.receiver_identity.tensor_geometry = "different-geometry".into(),
            _ => changed.receiver_identity.weight_fingerprint = "unknown".into(),
        }
        assert!(
            matches!(
                inference_attachment(&changed, "mock-occ3", &runtime),
                Err(Occ3Error::IncompatibleModel { .. })
            ),
            "{field}"
        );
    }
    let mut changed = request;
    changed.target.model_id = "same-display-name-other-model".into();
    assert!(inference_attachment(&changed, "mock-occ3", &runtime).is_err());
}

#[tokio::test]
async fn session_connector_runtime_generation_domain_method_expiry_are_bound() {
    let (_, request, runtime) = fixture().await;
    let mut changed = request.clone();
    changed.receiver_session_id = "unrelated".into();
    assert!(matches!(
        inference_attachment(&changed, "mock-occ3", &runtime),
        Err(Occ3Error::Unauthorized)
    ));
    assert!(matches!(
        inference_attachment(&request, "other", &runtime),
        Err(Occ3Error::UnknownConnector { .. })
    ));
    let mut current = runtime.clone();
    current.generation = 1;
    assert!(matches!(
        inference_attachment(&request, "mock-occ3", &current),
        Err(Occ3Error::RuntimeGenerationChanged { .. })
    ));
    current = runtime.clone();
    current.runtime_id = "different-access-domain".into();
    assert!(inference_attachment(&request, "mock-occ3", &current).is_err());
    current = runtime.clone();
    current.methods.clear();
    assert!(matches!(
        inference_attachment(&request, "mock-occ3", &current),
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    changed = request;
    changed.envelope.state_handle.expires_at_epoch = changed.now_epoch;
    assert!(matches!(
        inference_attachment(&changed, "mock-occ3", &runtime),
        Err(Occ3Error::HandleExpired { .. })
    ));
}

#[tokio::test]
async fn mock_rechecks_authorized_receiver_and_exact_handle_before_preparing() {
    let (connector, request, _) = fixture().await;
    let mut changed = request.clone();
    changed.receiver_session_id = "unrelated".into();
    changed.envelope.receiver_session_id = "unrelated".into();
    assert!(matches!(
        connector.prepare_inference(changed).await,
        Err(Occ3Error::Unauthorized)
    ));
    let mut changed = request.clone();
    changed.envelope.state_handle.state_size_bytes += 1;
    assert!(matches!(
        connector.prepare_inference(changed).await,
        Err(Occ3Error::Unauthorized)
    ));
    connector.restart_runtime();
    assert!(matches!(
        connector.prepare_inference(request).await,
        Err(Occ3Error::RuntimeGenerationChanged { .. })
    ));
}

#[tokio::test]
async fn authorized_admission_and_duplicate_id_are_checked_not_prompt_text() {
    let (_, request, _) = fixture().await;
    let envelope = request.envelope;
    let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(matches!(
        inbox.admit(envelope.clone(), "forged-id", "sender", 1),
        Err(Occ3Error::Unauthorized)
    ));
    assert!(matches!(
        inbox.admit(envelope.clone(), "message-1", "unrelated", 1),
        Err(Occ3Error::Unauthorized)
    ));
    assert!(inbox
        .admit(envelope.clone(), "message-1", "sender", 1)
        .unwrap());
    assert!(!inbox
        .admit(envelope.clone(), "message-1", "sender", 1)
        .unwrap());
    let mut forged = envelope;
    forged.control_summary = "changed".into();
    assert!(matches!(
        inbox.admit(forged, "message-1", "sender", 1),
        Err(Occ3Error::Unauthorized)
    ));
    assert!(inbox
        .claim(&["unrelated-message".into()], &[], 1)
        .unwrap()
        .is_none());
    assert_eq!(
        inbox.status("message-1"),
        Some(Occ3DeliveryStatus::Admitted)
    );
}

#[tokio::test]
async fn one_request_claim_blocks_automatic_text_retry_and_persistent_replay() {
    let (connector, request, _) = fixture().await;
    let mut inbox = admitted(request.envelope.clone());
    assert!(inbox.claim(&["message-1".into()], &[], 1).unwrap().is_some());
    assert!(matches!(
        inbox.claim(&["message-1".into()], &[], 1),
        Err(Occ3Error::InferenceOutcomeUncertain { .. })
    ));
    let lease = connector.prepare_inference(request).await.unwrap();
    assert!(
        inbox.record_outcome("message-1").is_err(),
        "request must be bound before dispatch"
    );
    let verified_identity = identity();
    assert_eq!(lease.attachment().method, verified_identity.latent_method);
    let mut mismatched_identity = verified_identity.clone();
    mismatched_identity.latent_method = "statebridge-prefix".into();
    assert!(inbox.bind_lease(&lease, &mismatched_identity).is_err());
    mismatched_identity = verified_identity.clone();
    mismatched_identity.checkpoint_revision = "other-checkpoint".into();
    assert!(inbox.bind_lease(&lease, &mismatched_identity).is_err());
    inbox.bind_lease(&lease, &verified_identity).unwrap();
    assert!(
        inbox.bind_lease(&lease, &verified_identity).is_err(),
        "cannot rebind the same admission"
    );
    lease.cancel();
    assert_eq!(
        inbox.record_outcome("message-1").unwrap(),
        Occ3DeliveryStatus::Failed
    );
    assert!(!permits_text_fallback(inbox.status("message-1").unwrap()));
    // A terminal failure never re-arms and never yields the envelope again
    // (no replay, no text fallback); it is journaled and skipped instead, so
    // the receiver's later inference proceeds without a re-attachment rather
    // than being bricked by the dead delivery.
    assert!(inbox.claim(&["message-1".into()], &[], 1).unwrap().is_none());
    let mut restarted = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(
        restarted.claim(&["message-1".into()], &[], 1).unwrap().is_none(),
        "no transcript hydration"
    );
}

/// Pending admissions queue in arrival order: a second sender may admit
/// while a first delivery is outstanding, each message keeps its own
/// receipt record, and `claim` consumes them one per request in context
/// (arrival) order. A terminal failure of the first never blocks the
/// second; an in-flight or unknown first delivery still refuses loudly.
#[tokio::test]
async fn pending_admissions_from_multiple_senders_queue_in_arrival_order() {
    let (connector, request, _) = fixture().await;
    let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(inbox
        .admit(request.envelope.clone(), "message-1", "sender", 1)
        .unwrap());

    // A second sender's genuinely new admission is accepted while the
    // first is still merely pending.
    let mut capture = capture_request();
    capture.message_id = "message-2".into();
    capture.sender_session_id = "sender-b".into();
    let handle = connector.capture_latent(capture).await.unwrap();
    let second = build_envelope(
        "message-2".into(),
        "sender-b".into(),
        "receiver".into(),
        identity().fingerprint(),
        connector.runtime_id(),
        "mock-occ3".into(),
        Occ3Method::KvHandoff,
        handle,
        "OCC/1\nTASK=fixture\nKIND=RESULT\nSTATUS=done".into(),
        "occ1".into(),
    )
    .unwrap();
    assert!(inbox.admit(second.clone(), "message-2", "sender-b", 1).unwrap());
    // Duplicate admission stays idempotent alongside the pending queue.
    assert!(!inbox
        .admit(second.clone(), "message-2", "sender-b", 1)
        .unwrap());

    // Arrival order is explicit in the audit projection.
    let audit = inbox.audit();
    let ids: Vec<&str> = audit
        .iter()
        .filter_map(|record| record["message_id"].as_str())
        .collect();
    assert_eq!(ids, vec!["message-1", "message-2"]);
    assert_eq!(
        audit[0]["admission_seq"].as_u64(),
        Some(1),
        "arrival sequence is monotonic and explicit"
    );

    // Reversed context order cannot reorder the queue: the earliest
    // arrival (message-1) is the only claimable record; message-2 stays
    // pending no matter where the context lists it.
    let claimed = inbox
        .claim(
            &["message-2".to_string(), "message-1".to_string()],
            &[],
            1,
        )
        .unwrap()
        .expect("earliest pending delivery is claimable");
    assert_eq!(claimed.message_id, "message-1");
    assert_eq!(inbox.status("message-2"), Some(Occ3DeliveryStatus::Admitted));

    // While the first is attached (Delivered) every further claim refuses
    // loudly receiver-globally: at-most-once, never a second attachment.
    // The refusal holds even when the active id is absent from the context
    // (an in-flight delivery dropped from context must not be shadowed).
    for ids in [
        vec!["message-1".to_string(), "message-2".to_string()],
        vec!["message-2".to_string()],
    ] {
        assert!(matches!(
            inbox.claim(&ids, &[], 1),
            Err(Occ3Error::InferenceOutcomeUncertain { .. })
        ));
    }

    // Cancel the first delivery's lease: a Prepared lease cancel is a
    // proven non-use, terminal and journaled. The pending second delivery
    // stays claimable instead of being bricked by the dead first one.
    let lease = connector.prepare_inference(request.clone()).await.unwrap();
    inbox.bind_lease(&lease, &identity()).unwrap();
    lease.cancel();
    assert_eq!(
        inbox.record_outcome("message-1").unwrap(),
        Occ3DeliveryStatus::Failed
    );
    let next = inbox
        .claim(
            &["message-1".to_string(), "message-2".to_string()],
            &[],
            1,
        )
        .unwrap()
        .expect("pending delivery after a terminal failure");
    assert_eq!(next.message_id, "message-2");
    assert_eq!(inbox.status("message-2"), Some(Occ3DeliveryStatus::Delivered));

    // The now-active second delivery blocks claims the same way, including
    // when its id is absent from the context.
    assert!(matches!(
        inbox.claim(&["message-1".to_string()], &[], 1),
        Err(Occ3Error::InferenceOutcomeUncertain { .. })
    ));

    // Resolve the second delivery terminally as well: no pending admission
    // remains, so a further claim attaches nothing and errors nothing.
    let mut second_request = request.clone();
    second_request.envelope = second.clone();
    second_request.request_id = "inference-2".into();
    let second_lease = connector
        .prepare_inference(second_request)
        .await
        .unwrap();
    inbox.bind_lease(&second_lease, &identity()).unwrap();
    second_lease.cancel();
    assert_eq!(
        inbox.record_outcome("message-2").unwrap(),
        Occ3DeliveryStatus::Failed
    );
    assert!(inbox
        .claim(
            &["message-1".to_string(), "message-2".to_string()],
            &[],
            1
        )
        .unwrap()
        .is_none());

    // No completion is reported while nothing reached Consumed, and
    // terminal-but-not-consumed deliveries never re-arm.
    assert!(inbox.poll_consumed().is_empty());
    assert!(inbox.poll_consumed().is_empty());
    assert!(inbox
        .claim(&["message-1".to_string()], &[], 1)
        .unwrap()
        .is_none());
    assert!(inbox
        .claim(&["message-2".to_string()], &[], 1)
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn expiry_after_admission_is_recorded_and_not_silent_fallback() {
    let (_, mut request, _) = fixture().await;
    request.envelope.state_handle.expires_at_epoch = 5;
    let mut inbox = admitted(request.envelope);
    assert!(matches!(
        inbox.claim(&["message-1".into()], &[], 5),
        Err(Occ3Error::HandleExpired { .. })
    ));
    assert_eq!(inbox.status("message-1"), Some(Occ3DeliveryStatus::Expired));
    // An expired admission is terminal and journaled: it never re-arms and
    // never silently becomes text, but it also no longer blocks later
    // pending deliveries from being claimed.
    assert!(inbox.claim(&["message-1".into()], &[], 5).unwrap().is_none());
    assert!(!permits_text_fallback(Occ3DeliveryStatus::Expired));
}

#[test]
fn fallback_is_only_before_admission_not_after_uncertain_delivery() {
    assert!(permits_text_fallback(Occ3DeliveryStatus::Prepared));
    assert!(permits_text_fallback(Occ3DeliveryStatus::Captured));
    for status in [
        Occ3DeliveryStatus::Admitted,
        Occ3DeliveryStatus::Delivered,
        Occ3DeliveryStatus::Consumed,
        Occ3DeliveryStatus::Acknowledged,
        Occ3DeliveryStatus::Failed,
        Occ3DeliveryStatus::Expired,
        Occ3DeliveryStatus::Uncertain,
    ] {
        assert!(!permits_text_fallback(status));
    }
}

#[tokio::test]
async fn forged_model_binding_and_embedded_source_fallback_ref_are_rejected() {
    let (_, mut request, runtime) = fixture().await;
    request.envelope.state_handle.model_fingerprint = "forged".into();
    assert!(inference_attachment(&request, "mock-occ3", &runtime).is_err());
    request.envelope.state_handle.model_fingerprint = request.envelope.model_fingerprint.clone();
    request.envelope.fallback_reference = "SOURCE_ONLY_SECRET_FIXTURE".into();
    assert!(inference_attachment(&request, "mock-occ3", &runtime).is_err());
}

#[tokio::test]
async fn sglang_fails_closed_without_legacy_send_or_detached_recv_network_call() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let connector = SglangOcc3Connector::new(SglangOcc3Config {
        endpoint: format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap(),
        api_key: Some("synthetic-fixture-key".into()),
        control_headers: None,
    })
    .unwrap();
    let (_, request, _) = fixture().await;
    let handle = request.envelope.state_handle.clone();
    assert!(matches!(
        connector.discover_capabilities().await,
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    assert!(matches!(
        connector.capture_latent(capture_request()).await,
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    assert!(matches!(
        connector.deliver_latent(&handle).await,
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    assert!(matches!(
        connector.prepare_inference(request).await,
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    assert!(matches!(
        connector
            .consume_latent(ConsumeRequest {
                handle: handle.clone(),
                receiver_session_id: "receiver".into(),
                injection_target: "next-turn".into(),
            })
            .await,
        Err(Occ3Error::NativeInferenceUnsupported { .. })
    ));
    assert!(connector.acknowledge_delivery(&handle).await.is_err());
    assert!(connector.release_handle(&handle).await.is_err());
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[tokio::test]
async fn control_header_enforcement_refuses_before_any_network_call() {
    use indexmap::IndexMap;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint: url::Url = format!("http://{}", listener.local_addr().unwrap())
        .parse()
        .unwrap();
    let mut invalid: Vec<IndexMap<String, String>> = Vec::new();
    // Forbidden transport-owned or unapproved names (case variants included).
    for name in [
        "authorization",
        "AUTHORIZATION",
        "host",
        "proxy-authorization",
        "content-type",
        "accept-encoding",
        "x-pomerium-authorization\r\nx-other: 1",
        "x-unapproved-gateway",
    ] {
        invalid.push(IndexMap::from([(name.to_string(), "v".to_string())]));
    }
    // Duplicate case variants, empty value, CRLF value, oversize value.
    invalid.push(IndexMap::from([
        ("X-Pomerium-Authorization".to_string(), "one".to_string()),
        ("x-pomerium-authorization".to_string(), "two".to_string()),
    ]));
    for value in ["", "v\r\nHost: x", &"x".repeat(16385)] {
        invalid.push(IndexMap::from([(
            "x-pomerium-authorization".to_string(),
            value.to_string(),
        )]));
    }
    for headers in invalid {
        let error = SglangOcc3Connector::new(SglangOcc3Config {
            endpoint: endpoint.clone(),
            api_key: None,
            control_headers: Some(headers),
        })
        .expect_err("malformed control headers must refuse");
        assert!(!error.to_string().contains("one"));
        assert!(!error.to_string().contains("two"));
    }
    // Nothing above may open a connection: no HTTP request was built or sent.
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[tokio::test]
async fn stale_capabilities_cannot_enable_unimplemented_native_connector() {
    use pi_coding_agent::core::occ::occ3::Occ3ConnectorRegistry;
    use std::sync::Arc;
    let (mock, _, _) = fixture().await;
    let mut caps = mock.capabilities();
    let fingerprint = caps.model_identity.fingerprint();
    let mut registry = Occ3ConnectorRegistry::empty();
    let legacy = SglangOcc3Connector::new(SglangOcc3Config {
        endpoint: "http://127.0.0.1:1".parse().unwrap(),
        api_key: None,
        control_headers: None,
    })
    .unwrap();
    registry.register(Arc::new(legacy), caps.clone());
    assert!(!registry.supports("sglang-occ3", Occ3Method::KvHandoff, &fingerprint));
    caps.features.clear();
    registry.register(Arc::new(mock), caps);
    assert!(!registry.supports("mock-occ3", Occ3Method::KvHandoff, &fingerprint));
}

#[tokio::test]
async fn negotiation_refuses_different_peer_connector_before_admission() {
    use pi_coding_agent::core::occ::occ3::{
        negotiate_occ3, Occ3CapabilityAdvertisement, Occ3ConnectorRegistry, Occ3Decision,
        Occ3FallbackReason, Occ3NegotiationInput,
    };
    use std::sync::Arc;
    let (mock, _, _) = fixture().await;
    let mut registry = Occ3ConnectorRegistry::empty();
    let caps = mock.capabilities();
    registry.register(Arc::new(mock), caps);
    let mut input = Occ3NegotiationInput {
        enabled: true,
        sender_identity: Some(identity()),
        receiver_identity: Some(identity()),
        peer_occ3: Some(Occ3CapabilityAdvertisement {
            supported: true,
            model_fingerprint: Some(identity().fingerprint()),
            connector_id: Some("unrelated-runtime-connector".into()),
            methods: vec![Occ3Method::KvHandoff],
        }),
        method: Occ3Method::KvHandoff,
        registry,
        max_envelope_chars: 16384,
        control_summary_chars: 32,
    };
    assert!(matches!(
        negotiate_occ3(&input),
        Occ3Decision::Fallback {
            reason: Occ3FallbackReason::NoConnector
        }
    ));
    input.peer_occ3.as_mut().unwrap().connector_id = Some("mock-occ3".into());
    assert!(matches!(negotiate_occ3(&input), Occ3Decision::Occ3 { .. }));
}

#[tokio::test]
async fn duplicate_after_expiry_stays_idempotent_without_text_fallback() {
    let (_, mut request, _) = fixture().await;
    request.envelope.state_handle.expires_at_epoch = 10;
    let envelope = request.envelope;
    let mut inbox = admitted(envelope.clone());
    assert!(!inbox
        .admit(envelope.clone(), "message-1", "sender", 20)
        .unwrap());
    assert_eq!(
        inbox.status("message-1"),
        Some(Occ3DeliveryStatus::Admitted)
    );
    assert!(!permits_text_fallback(inbox.status("message-1").unwrap()));
    assert_eq!(inbox.cancel_pending().len(), 1);
    assert!(!inbox.admit(envelope, "message-1", "sender", 30).unwrap());
    assert_eq!(inbox.status("message-1"), Some(Occ3DeliveryStatus::Failed));
}

#[tokio::test]
async fn mock_source_binding_rejects_message_sender_and_method_substitution() {
    let (connector, request, _) = fixture().await;
    let mut changed = request.clone();
    changed.envelope.message_id = "other-message".into();
    assert!(matches!(
        connector.prepare_inference(changed).await,
        Err(Occ3Error::Unauthorized)
    ));
    let mut changed = request.clone();
    changed.envelope.sender_session_id = "other-sender".into();
    assert!(matches!(
        connector.prepare_inference(changed).await,
        Err(Occ3Error::Unauthorized)
    ));
    let both = MockOcc3Connector::new(
        identity(),
        vec![Occ3Method::KvHandoff, Occ3Method::StatebridgePrefix],
    );
    let captured = both.capture_latent(capture_request()).await.unwrap();
    let mut changed = request;
    changed.envelope.state_handle = captured;
    changed.envelope.latent_method = Occ3Method::StatebridgePrefix;
    assert!(matches!(
        both.prepare_inference(changed).await,
        Err(Occ3Error::Unauthorized)
    ));
}

#[tokio::test]
async fn unpreparable_wire_identifiers_are_rejected_before_admission() {
    let (_, request, _) = fixture().await;
    for invalid_id in [
        "with space".to_string(),
        "nonascii-é".to_string(),
        "h".repeat(257),
    ] {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        let mut envelope = request.envelope.clone();
        envelope.state_handle.handle_id = invalid_id.clone();
        assert!(inbox.admit(envelope, "message-1", "sender", 1).is_err());
        assert!(inbox.status("message-1").is_none());
        let mut envelope = request.envelope.clone();
        envelope.message_id = invalid_id.clone();
        assert!(inbox.admit(envelope, &invalid_id, "sender", 1).is_err());
        assert!(inbox.status(&invalid_id).is_none());
    }
}

/// End-to-end completed-use through the REAL public provider transport: a
/// local fixture server answers the guarded `/chat/completions` request with
/// a genuine `x-optimus-occ3-receipt` header built from the exact attachment
/// the request carried. The lease is produced by the mock connector through
/// the normal `prepare_inference` path, bound to the inbox, dispatched via
/// `stream_simple_openai_completions` (the same public transport production
/// uses), and the receipt verification inside pi-ai transitions the shared
/// lease to `Consumed`. This is receipt verification, not status simulation:
/// no test-only API is exposed and no production surface changes.
#[tokio::test]
async fn consumed_receipt_flows_through_real_provider_transport_to_poll_consumed() {
    use pi_ai::providers::openai_completions::stream_simple_openai_completions;
    use pi_ai::types::{SimpleStreamOptions, StreamOptions};
    use pi_ai::utils::event_stream::AssistantMessageEventStream;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const COMPLETE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"native-fixture-answer\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    // A one-shot local server: it records the request, mirrors a valid
    // completed-use receipt for the exact `occ3` attachment the client
    // sent, and answers with a minimal SSE completion.
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let seen_request = Arc::new(Mutex::new(None::<serde_json::Value>));
    let captured = seen_request.clone();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let body = loop {
            let mut chunk = [0; 4096];
            let count = socket.read(&mut chunk).await.unwrap();
            assert_ne!(count, 0);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                let length: usize = head
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .unwrap()
                    .1
                    .trim()
                    .parse()
                    .unwrap();
                if bytes.len() >= end + 4 + length {
                    break serde_json::from_slice::<serde_json::Value>(
                        &bytes[end + 4..end + 4 + length],
                    )
                    .unwrap();
                }
            }
        };
        *captured.lock().unwrap() = Some(body.clone());
        let receipt = serde_json::json!({
            "attachment": body["occ3"],
            "consumed": true,
            "fallback_used": false,
            "injected_state": true,
        });
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nx-optimus-occ3-receipt: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            receipt,
            COMPLETE.len(),
            COMPLETE,
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    // The receiver model points at the fixture server through the same
    // loopback transport the guarded native path allows.
    let model = pi_ai::types::Model::new(
        "fixture-model",
        "fixture",
        pi_ai::types::API_OPENAI_COMPLETIONS,
        "fixture-provider",
        format!("http://{address}/v1"),
    );

    // Capture, envelope, admission and lease binding exactly as production
    // does: admit -> claim (earliest arrival) -> prepare_inference ->
    // bind_lease while the lease is still Prepared.
    let connector = MockOcc3Connector::new(identity(), vec![Occ3Method::KvHandoff]);
    let handle = connector.capture_latent(capture_request()).await.unwrap();
    let envelope = build_envelope(
        "message-1".into(),
        "sender".into(),
        "receiver".into(),
        identity().fingerprint(),
        connector.runtime_id(),
        "mock-occ3".into(),
        Occ3Method::KvHandoff,
        handle,
        "OCC/1\nTASK=fixture\nKIND=RESULT\nSTATUS=done".into(),
        "occ1".into(),
    )
    .unwrap();
    let request = PrepareInferenceRequest {
        envelope: envelope.clone(),
        request_id: "inference-1".into(),
        receiver_session_id: "receiver".into(),
        receiver_identity: identity(),
        now_epoch: 1,
        target: InferenceStateTarget {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
            base_url: model.base_url.clone(),
        },
    };
    let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(inbox
        .admit(envelope, "message-1", "sender", 1)
        .unwrap());
    let claimed = inbox
        .claim(&["message-1".to_string()], &[], 1)
        .unwrap()
        .expect("earliest pending delivery is claimable");
    assert_eq!(claimed.message_id, "message-1");
    let lease = connector.prepare_inference(request).await.unwrap();
    inbox.bind_lease(&lease, &identity()).unwrap();
    assert_eq!(inbox.status("message-1"), Some(Occ3DeliveryStatus::Delivered));

    // The actual native inference through the public provider transport.
    let context = pi_ai::types::Context::new(
        None,
        vec![pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text("fixture question".into()),
            1,
        ))],
        None,
    );
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some("synthetic-fixture-key".into()),
            timeout_ms: Some(3000.0),
            inference_state: Some(lease.clone()),
            ..Default::default()
        },
        ..Default::default()
    };
    let stream: AssistantMessageEventStream =
        stream_simple_openai_completions(&model, &context, Some(options));
    let output = stream.result().await;
    assert!(
        output.error_message.is_none(),
        "fixture transport must succeed: {:?}",
        output.error_message
    );

    // The fixture answered exactly one guarded request carrying the state.
    let body = seen_request.lock().unwrap().clone().expect("request captured");
    assert!(body.get("occ3").is_some(), "guarded body carries the attachment");

    // Receipt verification inside pi-ai consumed the shared lease.
    assert_eq!(lease.state(), InferenceStateLeaseState::Consumed);
    assert_eq!(
        inbox.record_outcome("message-1").unwrap(),
        Occ3DeliveryStatus::Consumed
    );
    // The completion notification drains exactly once, with the sender id.
    assert_eq!(
        inbox.poll_consumed(),
        vec![("message-1".to_string(), "sender".to_string())]
    );
    assert!(inbox.poll_consumed().is_empty());
    // A completed delivery is never a terminal failure notice.
    assert!(inbox.poll_terminal_failures().is_empty());
    // And a consumed record never re-arms.
    assert!(inbox
        .claim(&["message-1".to_string()], &[], 1)
        .unwrap()
        .is_none());
    server.await.unwrap();
}

/// The receiver's `prepare_inference` control round trip must classify a
/// synchronous proven-non-use refusal as the distinct busy error, so the
/// resolver can release its inbox claim and the next inference request
/// retries the consume cleanly (the live 2026-11-07 receiver wedge).
/// `prepare_inference` awaits exactly one control call before the
/// attachment exists - `sessions/enroll` - so a refusal on that call
/// proves no receiver attachment was created or submitted. On that route
/// today the real classified decline is the saturation gate's `quota`
/// (429: session-table capacity, or the pre-parse 16 KiB body bound);
/// enroll has no `busy` code, which exists only in the registry's
/// reserve_capture/reserve_consume admission declines - the paired 409
/// row keeps the classification's contract for a future control route
/// that reuses it. Every other decline - including a 409 `expired`, a
/// malformed, oversized or empty-code envelope, a non-identity
/// content-encoding, or a 200 carrying an error body on the success path
/// - keeps the generic fail-closed `TransferFailed { "native control
/// binding invalid" }` refusal: only the proven busy classes release a
/// claim. The request line is asserted to be the enroll route so the
/// classification is pinned to `prepare_inference`'s own control round
/// trip, never a publication or retirement call.
#[tokio::test]
async fn sglang_prepare_inference_classifies_pre_dispatch_busy_refusals() {
    use pi_coding_agent::core::occ3_connectors::sglang::{
        NativeExperimentalPermit, SglangOcc3Config, SglangOcc3Connector,
    };
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // One raw control fixture: each accepted connection is answered with
    // the next scripted (status, body) pair, then closed.
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    // (status line, extra headers, body, expected busy) — the classification
    // table including every parser boundary: the exact-256-byte accept, the
    // 257-byte reject, empty/64/65-char codes, a non-identity
    // content-encoding, and a 200 that carries an error body onto the
    // success path. Route coverage today: enroll returns 429 for quota
    // (session capacity or the pre-parse 16 KiB body bound) and can never
    // return `busy` (that is a registry-only decline); the 409-busy row
    // exists so the paired classification keeps its contract if a future
    // control route reuses the registry's busy decline.
    let long_code = "e".repeat(64);
    let over_code = "e".repeat(65);
    let busy_pad = r#"{"error":{"code":  "busy"}}"#;
    assert_eq!(busy_pad.len(), 27);
    let busy_256 = format!("{busy_pad}{}", " ".repeat(256 - busy_pad.len()));
    let busy_257 = format!("{busy_pad}{}", " ".repeat(257 - busy_pad.len()));
    assert_eq!(busy_256.len(), 256);
    assert_eq!(busy_257.len(), 257);
    let scripted: Vec<(&str, &str, String, bool)> = vec![
        ("409 Conflict", "", r#"{"error":{"code":"busy"}}"#.into(), true),
        ("429 Too Many Requests", "", r#"{"error":{"code":"quota"}}"#.into(), true),
        // Exactly 256 bytes is still accepted at the bound.
        ("409 Conflict", "", busy_256, true),
        ("409 Conflict", "", r#"{"error":{"code":"expired"}}"#.into(), false),
        ("503 Service Unavailable", "", r#"{"error":{"code":"unavailable"}}"#.into(), false),
        // One byte over the bound: never classified.
        ("409 Conflict", "", busy_257, false),
        ("409 Conflict", "", r#"{"error":{"code":""}}"#.into(), false),
        // A 64-char code is inside the bound but is not busy; 65 is rejected.
        ("409 Conflict", "", format!(r#"{{"error":{{"code":"{long_code}"}}}}"#), false),
        ("409 Conflict", "", format!(r#"{{"error":{{"code":"{over_code}"}}}}"#), false),
        // A busy code behind a non-identity content-encoding stays generic.
        ("409 Conflict", "Content-Encoding: gzip\r\n", r#"{"error":{"code":"busy"}}"#.into(), false),
        // A 200 carrying an error body fails on the success path, generically.
        ("200 OK", "", r#"{"error":{"code":"busy"}}"#.into(), false),
    ];
    // The expected classification per scripted row, captured before the
    // server task consumes the scripted table by move, so the two tables
    // cannot drift apart.
    let expected_busy: Vec<bool> = scripted.iter().map(|row| row.3).collect();
    let server = tokio::spawn(async move {
        for (status, extra, body, _) in scripted {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            // Read until the full head plus content-length body arrived.
            let length = loop {
                let mut chunk = [0; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                assert_ne!(count, 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                    let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                    let length: usize = head
                        .lines()
                        .filter_map(|line| line.split_once(':'))
                        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                        .unwrap()
                        .1
                        .trim()
                        .parse()
                        .unwrap();
                    if bytes.len() >= end + 4 + length {
                        break length;
                    }
                }
            };
            let _ = length;
            let head = String::from_utf8(bytes[..bytes.windows(4).position(|v| v == b"\r\n\r\n").unwrap()].to_vec()).unwrap();
            let request_line = head.lines().next().unwrap().to_string();
            assert!(
                request_line.contains("/occ3/v1/native/sessions/enroll"),
                "the busy classification must see the enroll control route, saw: {request_line}"
            );
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });

    let endpoint: url::Url = format!("http://{address}").parse().unwrap();
    let mut identity = identity();
    identity.model_id = "fixture-model".into();
    let permit = NativeExperimentalPermit {
        attempt_id: "attempt-fixture-1".into(),
        identity: identity.clone(),
        target: InferenceStateTarget {
            provider: "fixture-provider".into(),
            model_id: identity.model_id.clone(),
            base_url: format!("http://{address}/v1"),
        },
        runtime_id: "fixture-runtime-epoch-001".into(),
        runtime_generation: 7,
        access_domain: "fixture-family".into(),
        expires_at_epoch: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 600,
        source_profile: pi_ai::source_capture::SourceCaptureProfile {
            layer_index: 77,
            hidden_dim: 6144,
            dtype: "bfloat16".into(),
            layout_digest: "b".repeat(64),
            expected_ranks: 8,
        },
        profile_digest: "c".repeat(64),
        max_capture_bytes: 268_435_456,
    };
    let connector = SglangOcc3Connector::new_experimental(
        SglangOcc3Config {
            endpoint,
            api_key: Some("synthetic-fixture-key".into()),
            control_headers: None,
        },
        permit,
    )
    .unwrap();

    let request = || PrepareInferenceRequest {
        envelope: Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: "message-busy-1".into(),
            sender_session_id: "sender".into(),
            receiver_session_id: "receiver".into(),
            model_fingerprint: identity.fingerprint(),
            runtime_id: "fixture-runtime-epoch-001".into(),
            connector_id: "sglang-occ3".into(),
            latent_method: Occ3Method::KvHandoff,
            state_handle: pi_coding_agent::core::occ::occ3::Occ3HandleRef {
                handle_id: "handle-busy-1".into(),
                connector_id: "sglang-occ3".into(),
                model_fingerprint: identity.fingerprint(),
                runtime_generation: 7,
                expires_at_epoch: u64::MAX,
                state_size_bytes: 128,
            },
            control_summary: "busy classification fixture".into(),
            fallback_reference: "occ1".into(),
        },
        request_id: "inference-busy-1".into(),
        receiver_session_id: "receiver".into(),
        receiver_identity: identity.clone(),
        now_epoch: 1,
        target: InferenceStateTarget {
            provider: "fixture-provider".into(),
            model_id: identity.model_id.clone(),
            base_url: format!("http://{address}/v1"),
        },
    };

    // The classification table, in scripted order. The busy rows are the
    // distinct proven-non-use refusal; every other row is pinned to the
    // exact generic fail-closed error, so neither an unrelated error kind
    // nor a false Ok can pass the negative assertions.
    for (index, expect_busy) in expected_busy.into_iter().enumerate() {
        let error = connector
            .prepare_inference(request())
            .await
            .err()
            .expect("every scripted response must decline or fail the success parse");
        if expect_busy {
            assert!(
                matches!(error, Occ3Error::NativeInferenceBusyRefused { .. }),
                "row {index}: expected the busy refusal, saw {error:?}"
            );
        } else {
            match error {
                Occ3Error::TransferFailed { reason } => assert_eq!(
                    reason, "native control binding invalid",
                    "row {index}: expected the generic fail-closed refusal, saw {reason}"
                ),
                other => panic!("row {index}: expected the generic refusal, saw {other:?}"),
            }
        }
    }
    server.await.unwrap();
}
