/// Session teardown is not a runtime restart: all live sessions share this ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unrelated_close_preserves_shared_ready_connector() {
    let _cache_guard = reset_ready_cache().await;
    let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
    let request = fixture_request(&fixture, None, None);
    let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
    let model = fixture_model(&fixture);
    let receiver = crate::core::agent_session::occ3_state::Occ3NativeSession::enroll(
        "cache-receiver", &model, ready.connector.clone()).await.unwrap();
    let unrelated = crate::core::agent_session::occ3_state::Occ3NativeSession::enroll(
        "cache-unrelated", &model, ready.connector.clone()).await.unwrap();
    let before = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
    unrelated.close().await.unwrap();
    let next = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&ready, &next), "closing one session must preserve the runtime-scoped connector");
    assert_eq!(fixture.requests.load(std::sync::atomic::Ordering::SeqCst), before,
        "unrelated teardown must not re-handshake the runtime");
    let sender = crate::core::agent_session::occ3_state::Occ3NativeSession::enroll(
        "cache-sender", &model, next.connector.clone()).await.unwrap();
    // Publication re-enrolls the receiver through the sender's connector.
    // The strict fixture rejects any changed permit for this existing id.
    next.connector.enroll_native_session("cache-receiver").await
        .expect("receiver enrollment remains identical on the sender connector");
    sender.close().await.unwrap();
    receiver.close().await.unwrap();
}

/// A rejected old permit must force only that connector to re-handshake.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_refusal_evicts_only_affected_ready_connector() {
    let _cache_guard = reset_ready_cache().await;
    for failure in [2, 3] {
        let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let other = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        let other_request = fixture_request(&other, None, None);
        let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        let untouched = verify_cached(&other_request, Duration::from_secs(5)).await.unwrap();
        fixture.enrollment_failure.store(failure, std::sync::atomic::Ordering::SeqCst);
        assert!(ready.connector.enroll_native_session("restart-victim").await.is_err(),
            "the old runtime binding must remain fail closed");
        fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
        let replacement = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&ready, &replacement),
            "runtime refusal must evict the affected cached connector (failure={failure})");
        let other_cached = verify_cached(&other_request, Duration::from_secs(5)).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&untouched, &other_cached),
            "unrelated runtime readiness must survive");
        // Exercise eviction directly: a probe cooldown must not mask a late
        // old-instance failure accidentally evicting the replacement.
        invalidate_connector_ready(ready.connector.as_ref());
        let still_new = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&replacement, &still_new));
    }
}

/// Authentication/admission errors and failed probes are not restart evidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unverified_runtime_refusal_preserves_ready_connector() {
    let _cache_guard = reset_ready_cache().await;
    for failure in [1, 4] {
        let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        fixture.enrollment_failure.store(failure, std::sync::atomic::Ordering::SeqCst);
        assert!(ready.connector.enroll_native_session("ordinary-refusal").await.is_err());
        fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
        let cached = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&ready, &cached),
            "unverified refusal must not split live connectors");
    }
}

/// Repeated failures must not fan out into repeated control-plane probes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_refusal_probe_has_connector_cooldown() {
    let _cache_guard = reset_ready_cache().await;
    let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
    let request = fixture_request(&fixture, None, None);
    let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
    fixture.enrollment_failure.store(4, std::sync::atomic::Ordering::SeqCst);
    let before = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
    for _ in 0..3 {
        assert!(ready.connector.enroll_native_session("probe-cooldown").await.is_err());
    }
    let calls = fixture.requests.load(std::sync::atomic::Ordering::SeqCst) - before;
    assert_eq!(calls, 5, "three refusals plus only one pair of identity probes");
}

mod wrapped_ready_tests {
    use super::*;
    use crate::core::occ::occ3::connector::*;
    use crate::core::occ::occ3::types::*;

    struct ForwardingConnector(std::sync::Arc<dyn Occ3Connector>);
    #[async_trait::async_trait]
    impl Occ3Connector for ForwardingConnector {
        fn connector_id(&self) -> &str { self.0.connector_id() }
        fn readiness_instance_id(&self) -> Option<&str> { self.0.readiness_instance_id() }
        fn protocol_version(&self) -> u32 { self.0.protocol_version() }
        async fn enroll_native_session(&self, id: &str) -> Result<NativeSessionEnrollment, Occ3Error> {
            self.0.enroll_native_session(id).await
        }
        async fn health(&self) -> Result<(), Occ3Error> { self.0.health().await }
        async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error> { self.0.discover_capabilities().await }
        async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> { self.0.model_identity().await }
        async fn capture_latent(&self, r: CaptureRequest) -> Result<Occ3HandleRef, Occ3Error> { self.0.capture_latent(r).await }
        async fn deliver_latent(&self, h: &Occ3HandleRef) -> Result<(), Occ3Error> { self.0.deliver_latent(h).await }
        async fn consume_latent(&self, r: ConsumeRequest) -> Result<Occ3TransferMetrics, Occ3Error> { self.0.consume_latent(r).await }
        async fn acknowledge_delivery(&self, h: &Occ3HandleRef) -> Result<(), Occ3Error> { self.0.acknowledge_delivery(h).await }
        async fn release_handle(&self, h: &Occ3HandleRef) -> Result<(), Occ3Error> { self.0.release_handle(h).await }
        async fn shutdown(&self) -> Result<(), Occ3Error> { self.0.shutdown().await }
    }

    /// The failing inner instance must evict its outer cached wrapper too.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runtime_refusal_evicts_wrapped_connector() {
        let _cache_guard = reset_ready_cache().await;
        let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        let wrapped = std::sync::Arc::new(Occ3Ready {
            connector: std::sync::Arc::new(ForwardingConnector(ready.connector.clone())),
            capabilities: ready.capabilities.clone(), method: ready.method,
            method_id: ready.method_id.clone(), artifact_digest: ready.artifact_digest.clone(),
            expires_at_epoch: ready.expires_at_epoch, connector_id: ready.connector_id.clone(),
        });
        assert!(ready.connector.readiness_instance_id().is_some(), "native readiness has instance identity");
        assert_eq!(wrapped.connector.readiness_instance_id(), ready.connector.readiness_instance_id(),
            "wrapper MUST forward the exact inner instance identity");
        let config = configured_connector(&request).unwrap();
        ready_cache().lock().unwrap().insert(cache_key(&request, &config), wrapped.clone());
        fixture.enrollment_failure.store(2, std::sync::atomic::Ordering::SeqCst);
        assert!(wrapped.connector.enroll_native_session("wrapped-victim").await.is_err());
        fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
        let fresh = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&wrapped, &fresh), "inner failure evicts wrapper readiness");
    }
}

/// Busy, service failure, and malformed payloads carry no authority evidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_authority_refusals_do_not_probe_runtime() {
    let _cache_guard = reset_ready_cache().await;
    for failure in [5, 6, 7] {
        let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        fixture.enrollment_failure.store(failure, std::sync::atomic::Ordering::SeqCst);
        let before = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
        assert!(ready.connector.enroll_native_session("not-authority").await.is_err());
        assert_eq!(fixture.requests.load(std::sync::atomic::Ordering::SeqCst) - before, 1,
            "non-authority refusal must not probe (failure={failure})");
        fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
        let cached = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&ready, &cached));
    }
}

/// A healthy unchanged probe cannot mask a restart immediately afterward.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unchanged_probe_does_not_suppress_subsequent_restart() {
    let _cache_guard = reset_ready_cache().await;
    let fixture = serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
    let request = fixture_request(&fixture, None, None);
    let ready = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
    fixture.enrollment_failure.store(1, std::sync::atomic::Ordering::SeqCst);
    assert!(ready.connector.enroll_native_session("unchanged-refusal").await.is_err());
    fixture.enrollment_failure.store(3, std::sync::atomic::Ordering::SeqCst);
    assert!(ready.connector.enroll_native_session("immediate-restart").await.is_err());
    fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
    let replacement = verify_cached(&request, Duration::from_secs(5)).await.unwrap();
    assert!(!std::sync::Arc::ptr_eq(&ready, &replacement), "verified unchanged probe must not impose cooldown");
}
