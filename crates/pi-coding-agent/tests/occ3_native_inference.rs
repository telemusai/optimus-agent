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
    assert!(inbox.claim(&["message-1".into()], 1).unwrap().is_some());
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
        .claim(&["unrelated-message".into()], 1)
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
    assert!(inbox.claim(&["message-1".into()], 1).unwrap().is_some());
    assert!(matches!(
        inbox.claim(&["message-1".into()], 1),
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
    assert!(inbox.claim(&["message-1".into()], 1).is_err());
    let mut restarted = Occ3InferenceInbox::new("receiver".into()).unwrap();
    assert!(
        restarted.claim(&["message-1".into()], 1).unwrap().is_none(),
        "no transcript hydration"
    );
}

#[tokio::test]
async fn concurrent_receiver_admission_is_bounded_before_second_capture_use() {
    let (_, request, _) = fixture().await;
    let mut inbox = admitted(request.envelope.clone());
    let mut second = request.envelope.clone();
    second.message_id = "message-2".into();
    assert!(inbox
        .admit(second.clone(), "message-2", "sender", 1)
        .is_err());
    assert_eq!(inbox.cancel_pending().len(), 1);
    assert!(inbox.admit(second, "message-2", "sender", 1).unwrap());
    assert!(
        inbox.claim(&["message-1".into()], 1).is_err(),
        "cancelled message never silently becomes text"
    );
}

#[tokio::test]
async fn expiry_after_admission_is_recorded_and_not_silent_fallback() {
    let (_, mut request, _) = fixture().await;
    request.envelope.state_handle.expires_at_epoch = 5;
    let mut inbox = admitted(request.envelope);
    assert!(matches!(
        inbox.claim(&["message-1".into()], 5),
        Err(Occ3Error::HandleExpired { .. })
    ));
    assert_eq!(inbox.status("message-1"), Some(Occ3DeliveryStatus::Expired));
    assert!(inbox.claim(&["message-1".into()], 5).is_err());
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
