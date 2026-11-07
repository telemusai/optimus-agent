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
        // Late failure from the old connector cannot remove its replacement.
        fixture.enrollment_failure.store(failure, std::sync::atomic::Ordering::SeqCst);
        assert!(ready.connector.enroll_native_session("late-old-permit").await.is_err());
        fixture.enrollment_failure.store(0, std::sync::atomic::Ordering::SeqCst);
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
