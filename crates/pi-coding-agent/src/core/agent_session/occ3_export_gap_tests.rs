//! GAP-3 follow-up regression tests (receiver-tests-audit.md, section 7
//! item 3, export discipline): the `Occ3NativeSession` negative-control
//! export path in `occ3_state.rs` around resolve/arm/invalidate ordering.
//! The inline `occ3_state.rs` tests prove the armed export fails closed
//! until a receipt is captured (`armed_export_fails_closed_until_receipt_is_captured`)
//! and after `invalidate` (`export_fails_closed_after_invalidate`); these
//! tests pin the still-uncovered ordering discipline:
//!
//! - after `resolve_capture` but BEFORE any receipt lands, the export is
//!   None on every attempt and the audit keeps
//!   `negative_control_armed == true` / `negative_control_exported == false`
//!   (a fail-closed export never consumes the one-shot armed flag),
//! - a second `arm_source` after the first resolve returns an error
//!   (one-shot arming) and the audit still projects exactly one armed
//!   capture, and
//! - `invalidate()` keeps the export None across the boundary (attempts on
//!   both sides) and every re-arm path refuses afterwards.
//!
//! The fixture duplicates the private inline-test connector from
//! `occ3_state.rs`; everything answers locally, offline.
use crate::core::agent_session::occ3_state::{NativeSourceBudget, Occ3NativeSession};
use crate::core::occ::occ3::connector::{
    CaptureRequest, ConsumeRequest, NativeSessionEnrollment, Occ3Connector,
};
use crate::core::occ::occ3::types::{
    Occ3ConnectorCapabilities, Occ3Error, Occ3HandleRef, Occ3Method, Occ3ModelIdentity,
    Occ3TransferMetrics,
};
use pi_agent_core::types::InferenceStateRequest;
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{
    SourceCaptureIntent, SourceCaptureLease, SourceCaptureLimits, SourceCaptureProfile,
    SourceCaptureState, SOURCE_CAPTURE_SCHEMA, SOURCE_SPAN_POLICY,
};
use pi_ai::types::{Model, API_OPENAI_COMPLETIONS};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fixture_identity() -> Occ3ModelIdentity {
    Occ3ModelIdentity {
        model_id: "fixture-model".into(),
        architecture: "fixture-arch".into(),
        checkpoint_revision: "fixture-revision".into(),
        weight_fingerprint: "fixture-weights".into(),
        tokenizer_fingerprint: "fixture-tokenizer".into(),
        chat_template_fingerprint: "fixture-template".into(),
        quantisation: "fixture-bf16".into(),
        tensor_geometry: "fixture-d8".into(),
        kv_layout: "fixture-tp8".into(),
        latent_method: METHOD.into(),
        runtime: "fixture-runtime".into(),
    }
}

fn fixture_model() -> Model {
    Model::new(
        "fixture-model",
        "fixture-model",
        API_OPENAI_COMPLETIONS,
        "unit-fixture",
        "http://127.0.0.1:40000/v1",
    )
}

struct FixtureConnector {
    identity: Occ3ModelIdentity,
    profile: SourceCaptureProfile,
}

impl FixtureConnector {
    fn new() -> Self {
        Self {
            identity: fixture_identity(),
            profile: SourceCaptureProfile {
                layer_index: 3,
                hidden_dim: 8,
                dtype: "bfloat16".into(),
                layout_digest: "1".repeat(64),
                expected_ranks: 8,
            },
        }
    }
}

#[async_trait::async_trait]
impl Occ3Connector for FixtureConnector {
    fn connector_id(&self) -> &str {
        "unit-fixture-occ3"
    }
    fn protocol_version(&self) -> u32 {
        1
    }
    fn supports_native_inference(&self) -> bool {
        true
    }
    async fn enroll_native_session(
        &self,
        session_id: &str,
    ) -> Result<NativeSessionEnrollment, Occ3Error> {
        Ok(NativeSessionEnrollment {
            session_id: session_id.into(),
            principal_digest: "unit-fixture-principal".into(),
            identity: self.identity.clone(),
            runtime_id: "unit-fixture-runtime".into(),
            runtime_generation: 7,
            access_domain: "unit-fixture-domain".into(),
            expires_at_epoch: now() + 600,
            source_profile: self.profile.clone(),
            profile_digest: "2".repeat(64),
        })
    }
    async fn prepare_source_capture(
        &self,
        session_id: &str,
        target: InferenceStateTarget,
        retained_count_limit: u32,
        max_capture_bytes: u64,
        limits: SourceCaptureLimits,
        _arming: Option<pi_ai::source_capture::SourceSpanArming>,
    ) -> Result<SourceCaptureLease, Occ3Error> {
        SourceCaptureLease::new(
            target,
            SourceCaptureIntent {
                schema: SOURCE_CAPTURE_SCHEMA.into(),
                capture_id: format!("unit-capture-{}", uuid::Uuid::new_v4()),
                source_session_id: session_id.into(),
                source_request_id: format!("unit-source-{}", uuid::Uuid::new_v4()),
                turn_nonce: format!("unit-turn-{}", uuid::Uuid::new_v4()),
                model_fingerprint: self.identity.fingerprint(),
                runtime_id: "unit-fixture-runtime".into(),
                runtime_generation: 7,
                method: self.identity.latent_method.clone(),
                retained_count_limit,
                max_capture_bytes,
                expires_at_epoch: now() + 600,
                span_policy: SOURCE_SPAN_POLICY.into(),
                capture_mode: String::new(),
                label_anchor_alpha: None,
            },
            self.profile.clone(),
            limits,
        )
        .map_err(|_| Occ3Error::TransferFailed {
            reason: "unit-fixture".into(),
        })
    }
    async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error> {
        Ok(Occ3ConnectorCapabilities {
            protocol_version: 1,
            connector_version: "unit-fixture".into(),
            model_identity: self.identity.clone(),
            methods: vec![Occ3Method::StatebridgePrefix],
            tensor_formats: vec!["unit-fixture".into()],
            max_state_bytes: 1024,
            state_ttl_seconds: 60,
            max_concurrent_handles: 2,
            cache_isolation: "unit-fixture".into(),
            features: std::collections::BTreeMap::from([(
                "native_inference_attachment".into(),
                crate::core::occ::occ3::inference::NATIVE_INFERENCE_CONTRACT.into(),
            )]),
        })
    }
    async fn health(&self) -> Result<(), Occ3Error> {
        Ok(())
    }
    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> {
        Ok(self.identity.clone())
    }
    async fn capture_latent(
        &self,
        _request: CaptureRequest,
    ) -> Result<Occ3HandleRef, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn deliver_latent(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn consume_latent(
        &self,
        _request: ConsumeRequest,
    ) -> Result<Occ3TransferMetrics, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn acknowledge_delivery(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn release_handle(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn shutdown(&self) -> Result<(), Occ3Error> {
        Ok(())
    }
}

fn budget() -> NativeSourceBudget {
    NativeSourceBudget {
        retained_count_limit: 8,
        max_capture_bytes: 1024,
        deadline: Instant::now() + Duration::from_secs(60),
        max_response_bytes: 4096,
    }
}

async fn enrolled() -> Arc<Occ3NativeSession> {
    Occ3NativeSession::enroll(
        "gap3-export-session",
        &fixture_model(),
        Arc::new(FixtureConnector::new()),
    )
    .await
    .expect("fixture enrollment")
}

/// Drives the real resolve hook once so the slot holds a Prepared lease.
async fn resolve_capture(session: &Arc<Occ3NativeSession>) {
    let hooks = session.source_hooks();
    let lease = (hooks.resolve)(InferenceStateRequest {
        model: fixture_model(),
        context: Default::default(),
        messages: Vec::new(),
        newly_supplied_messages: Vec::new(),
        signal: None,
    })
    .await
    .expect("fixture resolve")
    .expect("capture lease");
    assert_eq!(lease.state(), SourceCaptureState::Prepared);
}

/// After `resolve_capture` but BEFORE any receipt lands, the armed export
/// fails closed on EVERY attempt and never consumes the one-shot armed
/// flag: `export_negative_control` stays None while the audit keeps
/// `negative_control_armed == true` and `negative_control_exported == false`.
#[tokio::test]
async fn export_fails_closed_repeatedly_before_any_receipt_lands() {
    let session = enrolled().await;
    session.arm_source(budget()).expect("arm source");
    session.arm_negative_control().expect("arm the negative control");
    resolve_capture(&session).await;
    for attempt in 1..=3 {
        assert!(
            session.export_negative_control().is_none(),
            "export attempt {attempt} must fail closed before a receipt lands"
        );
        let audit = session.audit();
        assert_eq!(
            audit["source"]["state"],
            json!("Prepared"),
            "attempt {attempt}: the capture stays un-receipted: {audit}"
        );
        assert!(
            audit["source"]["receipt"].is_null(),
            "attempt {attempt}: no receipt may be projected yet: {audit}"
        );
        assert_eq!(
            audit["source"]["negative_control_armed"],
            json!(true),
            "attempt {attempt}: a fail-closed export must not consume the armed flag: {audit}"
        );
        assert_eq!(
            audit["source"]["negative_control_exported"],
            json!(false),
            "attempt {attempt}: a fail-closed export must not mark the one-shot export: {audit}"
        );
    }
    // The audit projection stays hash-only: no raw handle key appears.
    assert!(!session.audit().to_string().contains("source_handle"));
}

/// One-shot arming: a second `arm_source` after the first resolve returns
/// an error, and the audit still shows exactly one armed capture - the
/// first lease, untouched, still `Prepared`.
#[tokio::test]
async fn second_arm_source_after_the_first_resolve_is_refused_with_one_capture_armed() {
    let session = enrolled().await;
    session.arm_source(budget()).expect("the first arm must succeed");
    resolve_capture(&session).await;
    let before = session.audit();
    assert_eq!(before["source"]["state"], json!("Prepared"));
    let first_capture_id = before["source"]["capture_id"].clone();
    let second = session.arm_source(budget());
    assert!(
        second.is_err(),
        "a second arm_source after the first resolve must be refused: {second:?}"
    );
    let after = session.audit();
    assert!(
        after["source"].is_object(),
        "the audit must project exactly one source slot: {after}"
    );
    assert!(after["source"]["capture_id"].is_string());
    assert_eq!(
        after["source"]["capture_id"], first_capture_id,
        "the refused arm must not replace or add a capture: {after}"
    );
    assert_eq!(after["source"]["state"], json!("Prepared"));
    assert_eq!(after["source"]["negative_control_armed"], json!(false));
    assert_eq!(after["source"]["negative_control_exported"], json!(false));
}

/// The invalidate boundary through the export path ordering: the export is
/// already None before `invalidate()` (no receipt yet), stays None after
/// it, the armed capture is cancelled, and every re-arm path refuses - a
/// mirror of `export_fails_closed_after_invalidate` with export attempts
/// placed on both sides of the invalidate.
#[tokio::test]
async fn export_stays_none_across_invalidate_and_every_re_arm_refuses() {
    let session = enrolled().await;
    session.arm_source(budget()).expect("arm source");
    session.arm_negative_control().expect("arm the negative control");
    resolve_capture(&session).await;
    // Before the boundary: fail-closed because no receipt has landed.
    assert!(
        session.export_negative_control().is_none(),
        "the export must fail closed before invalidate (no receipt yet)"
    );
    session.invalidate();
    // After the boundary: fail-closed because the permit is closed.
    assert!(
        session.export_negative_control().is_none(),
        "the export must stay None after invalidate"
    );
    let audit = session.audit();
    assert_eq!(
        audit["source"]["state"],
        json!("Cancelled"),
        "invalidate must cancel the armed capture: {audit}"
    );
    assert_eq!(
        audit["source"]["negative_control_armed"],
        json!(true),
        "the armed flag must survive invalidation fail-closed: {audit}"
    );
    assert_eq!(
        audit["source"]["negative_control_exported"],
        json!(false),
        "no fail-closed export may mark the one-shot export: {audit}"
    );
    // Re-arming refuses on both paths after invalidation.
    assert!(
        session.arm_source(budget()).is_err(),
        "re-arming the source after invalidate must refuse"
    );
    assert!(
        session.arm_negative_control().is_err(),
        "re-arming the negative control after invalidate must refuse"
    );
    // And the export stays closed after the refused re-arms.
    assert!(
        session.export_negative_control().is_none(),
        "the export must stay closed after the refused re-arms"
    );
}
