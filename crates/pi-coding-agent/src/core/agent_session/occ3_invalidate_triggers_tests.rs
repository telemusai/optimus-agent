//! GAP-3 regression tests (receiver-tests-audit.md, section 7 item 3): the
//! `AgentSession`-level triggers that invalidate the installed OCC-3 native
//! receiver/source state. Every trigger below is production wiring in
//! `agent_session.rs`; until these tests none of them was exercised with a
//! real `Occ3NativeSession` installed on a real `AgentSession`:
//!
//! - `clear_queue`                       -> `state.invalidate()`
//! - `clear_queued_agent_messages`       -> `state.invalidate()`
//! - `request_abort`/`request_abort_inner`-> `state.invalidate()`
//! - `abort`/`abort_inner`               -> `state.close().await`
//! - `set_model`                         -> `state.close().await?`
//! - `dispose_async`/`dispose_async_once_body` -> `state.close().await`
//! - `dispose`                           -> `state.invalidate()`
//!
//! The assertions are the receiver-side effects the audit names: the inbox
//! admission is invalidated (`Occ3InferenceInbox::invalidate_all`, an admitted
//! record becomes `Failed`), the source capture lease is cancelled, and the
//! native state stays fail-closed afterwards. Everything runs offline: the
//! fixture connector answers locally and no provider is contacted.
use super::*;
use crate::core::agent_messages::{
    AgentFamilyCatalogEntry, AgentSessionMessageController, AgentSessionMessageEndpoint,
    AgentSessionMessageReceipt, AgentSessionMessageSendInput, AgentFamilyRosterResult,
    AuthorizedNativeHandoff, PreparedNativeHandoff,
};
use crate::core::agent_session::occ3_state::{
    NativeSourceBudget, Occ3NativeSession,
};
use crate::core::occ::occ3::connector::{
    NativePublication, NativePublicationRequest, NativeRetirement, NativeSessionEnrollment,
    Occ3Connector,
};
use crate::core::occ::occ3::types::{
    Occ3ConnectorCapabilities, Occ3DeliveryStatus, Occ3Envelope, Occ3Error, Occ3HandleRef,
    Occ3Method, Occ3ModelIdentity,
};
use pi_agent_core::types::InferenceStateRequest;
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{
    SourceCaptureIntent, SourceCaptureLease, SourceCaptureLimits, SourceCaptureProfile,
    SourceCaptureState, SOURCE_CAPTURE_SCHEMA, SOURCE_SPAN_POLICY,
};
use pi_ai::types::BoxFuture;
use std::time::Duration;

const GAP_CONNECTOR_ID: &str = "gap3-occ3-connector";
const GAP_METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";

/// The model the fixture connector attests; `API_OPENAI_COMPLETIONS` is the
/// only API `Occ3NativeSession::enroll` accepts.
fn fixture_model() -> Model {
    Model::new(
        "fixture-model",
        "fixture-model",
        pi_ai::types::API_OPENAI_COMPLETIONS,
        "unit-fixture",
        "http://127.0.0.1:40000/v1",
    )
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
        latent_method: GAP_METHOD.into(),
        runtime: "fixture-runtime".into(),
    }
}

/// Local offline connector: enrollment, capabilities and source-capture
/// preparation only. No latent I/O exists in this fixture.
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
        GAP_CONNECTOR_ID
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
            principal_digest: "gap3-fixture-principal".into(),
            identity: self.identity.clone(),
            runtime_id: "gap3-fixture-runtime".into(),
            runtime_generation: 7,
            access_domain: "gap3-fixture-domain".into(),
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
    ) -> Result<SourceCaptureLease, Occ3Error> {
        SourceCaptureLease::new(
            target,
            SourceCaptureIntent {
                schema: SOURCE_CAPTURE_SCHEMA.into(),
                capture_id: format!("gap3-capture-{}", uuid::Uuid::new_v4()),
                source_session_id: session_id.into(),
                source_request_id: format!("gap3-source-{}", uuid::Uuid::new_v4()),
                turn_nonce: format!("gap3-turn-{}", uuid::Uuid::new_v4()),
                model_fingerprint: self.identity.fingerprint(),
                runtime_id: "gap3-fixture-runtime".into(),
                runtime_generation: 7,
                method: self.identity.latent_method.clone(),
                retained_count_limit,
                max_capture_bytes,
                expires_at_epoch: now() + 600,
                span_policy: SOURCE_SPAN_POLICY.into(),
            },
            self.profile.clone(),
            limits,
        )
        .map_err(|_| Occ3Error::TransferFailed {
            reason: "gap3-fixture".into(),
        })
    }
    async fn retire_native_source(
        &self,
        _source: &SourceCaptureLease,
    ) -> Result<NativeRetirement, Occ3Error> {
        Ok(NativeRetirement {
            revoked: true,
            gpu_freed: true,
        })
    }
    async fn health(&self) -> Result<(), Occ3Error> {
        Ok(())
    }
    async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error> {
        Ok(Occ3ConnectorCapabilities {
            protocol_version: 1,
            connector_version: "gap3-fixture".into(),
            model_identity: self.identity.clone(),
            methods: vec![Occ3Method::StatebridgePrefix],
            tensor_formats: vec!["gap3-fixture".into()],
            max_state_bytes: 1024,
            state_ttl_seconds: 60,
            max_concurrent_handles: 2,
            cache_isolation: "gap3-fixture".into(),
            features: std::collections::BTreeMap::from([(
                "native_inference_attachment".into(),
                crate::core::occ::occ3::inference::NATIVE_INFERENCE_CONTRACT.into(),
            )]),
        })
    }
    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> {
        Ok(self.identity.clone())
    }
    async fn capture_latent(
        &self,
        _request: crate::core::occ::occ3::connector::CaptureRequest,
    ) -> Result<Occ3HandleRef, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn deliver_latent(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn consume_latent(
        &self,
        _request: crate::core::occ::occ3::connector::ConsumeRequest,
    ) -> Result<crate::core::occ::occ3::types::Occ3TransferMetrics, Occ3Error> {
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

/// Inert controller double: the invalidation triggers under test never call
/// back into the controller, so it only has to satisfy the trait.
struct InertController;

impl AgentSessionMessageController for InertController {
    fn roster(&self) -> BoxFuture<Result<AgentFamilyRosterResult, String>> {
        Box::pin(async { Ok(AgentFamilyRosterResult::default()) })
    }
    fn await_pending_child_publication(&self, _selector: String)
        -> BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn send_agent_message(
        &self,
        _input: AgentSessionMessageSendInput,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        Box::pin(async { Err("gap3 fixture controller is inert".into()) })
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn budget() -> NativeSourceBudget {
    NativeSourceBudget {
        retained_count_limit: 8,
        max_capture_bytes: 1024,
        deadline: std::time::Instant::now() + Duration::from_secs(60),
        max_response_bytes: 4096,
    }
}

struct NativeFixture {
    session: Arc<AgentSession>,
    native: Arc<Occ3NativeSession>,
}

/// A real `AgentSession` (the shared in-memory fixture) with a real
/// `Occ3NativeSession` installed through the production
/// `install_experimental_native_occ3` seam. The fixture session's faux model
/// is swapped for the openai-completions fixture model the connector attests,
/// exactly like `retained_stop_tests::pair` swaps `model.api`.
async fn native_fixture() -> NativeFixture {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    {
        let mut state = session.agent.state();
        state.model = fixture_model();
        session.agent.set_state(state);
    }
    let native = Occ3NativeSession::enroll(
        &session.session_id(),
        &fixture_model(),
        Arc::new(FixtureConnector::new()),
    )
    .await
    .expect("fixture enrollment");
    session
        .install_experimental_native_occ3(
            native.clone(),
            Arc::new(InertController) as Arc<dyn AgentSessionMessageController>,
        )
        .expect("install the native session on the agent session");
    assert!(
        session.experimental_native_occ3().is_some(),
        "the installed permit must be observable on the session"
    );
    NativeFixture { session, native }
}

/// Admit one envelope into the receiver inbox through the real
/// `Occ3NativeSession::admit` path (the only authenticated admission entry
/// production has). Returns the admitted message id.
fn admit_envelope(native: &Arc<Occ3NativeSession>) -> String {
    let audit = native.audit();
    let receiver_session_id = audit["session_id"].as_str().expect("session id").to_string();
    let model_fingerprint = audit["model_fingerprint"].as_str().expect("fingerprint").to_string();
    let runtime_id = audit["runtime_id"].as_str().expect("runtime id").to_string();
    let runtime_generation = audit["runtime_generation"].as_u64().expect("generation");
    let authorization = AuthorizedNativeHandoff::from_catalog(
        &AgentFamilyCatalogEntry {
            id: "gap3-source".into(),
            name: Some("gap3-source".into()),
            depth: 0.,
            ..AgentFamilyCatalogEntry::default()
        },
        &AgentFamilyCatalogEntry {
            id: receiver_session_id.clone(),
            name: Some("gap3-receiver".into()),
            depth: 1.,
            parent_session_id: Some("gap3-source".into()),
            ..AgentFamilyCatalogEntry::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: "gap3-source-active".into(),
            session_id: "gap3-source".into(),
            session_name: Some("gap3-source".into()),
            ..AgentSessionMessageEndpoint::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: "gap3-receiver-active".into(),
            session_id: receiver_session_id.clone(),
            session_name: Some("gap3-receiver".into()),
            ..AgentSessionMessageEndpoint::default()
        },
        "child",
    )
    .expect("fixture authorization");
    let publication = NativePublication {
        publication_id: format!("gap3-publication-{}", uuid::Uuid::new_v4()),
        binding_digest: "3".repeat(64),
        envelope: Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: authorization.message_id().to_string(),
            sender_session_id: "gap3-source".into(),
            receiver_session_id: receiver_session_id,
            model_fingerprint: model_fingerprint.clone(),
            runtime_id: runtime_id,
            connector_id: GAP_CONNECTOR_ID.into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: format!("gap3-handle-{}", uuid::Uuid::new_v4()),
                connector_id: GAP_CONNECTOR_ID.into(),
                model_fingerprint: model_fingerprint.clone(),
                runtime_generation,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary:
                "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.".into(),
            fallback_reference: "occ1".into(),
        },
    };
    let prepared =
        PreparedNativeHandoff::new(authorization.clone(), publication).expect("fixture prepared handoff");
    let admitted = native
        .admit(&prepared)
        .expect("the native session must still be live for a fresh admission");
    assert!(
        admitted,
        "the first admission must be a genuinely new inbox record"
    );
    authorization.message_id().to_string()
}

/// The same admission after the native state was closed: `Occ3NativeSession::
/// admit` must fail closed instead of journaling a second record.
fn admit_after_close_errs(native: &Arc<Occ3NativeSession>) {
    let audit = native.audit();
    let receiver_session_id = audit["session_id"].as_str().expect("session id").to_string();
    let model_fingerprint = audit["model_fingerprint"].as_str().expect("fingerprint").to_string();
    let runtime_id = audit["runtime_id"].as_str().expect("runtime id").to_string();
    let runtime_generation = audit["runtime_generation"].as_u64().expect("generation");
    let authorization = AuthorizedNativeHandoff::from_catalog(
        &AgentFamilyCatalogEntry {
            id: "gap3-source".into(),
            depth: 0.,
            ..AgentFamilyCatalogEntry::default()
        },
        &AgentFamilyCatalogEntry {
            id: receiver_session_id.clone(),
            depth: 1.,
            parent_session_id: Some("gap3-source".into()),
            ..AgentFamilyCatalogEntry::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: "gap3-source-active".into(),
            session_id: "gap3-source".into(),
            ..AgentSessionMessageEndpoint::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: "gap3-receiver-active".into(),
            session_id: receiver_session_id.clone(),
            ..AgentSessionMessageEndpoint::default()
        },
        "child",
    )
    .expect("fixture authorization");
    let publication = NativePublication {
        publication_id: format!("gap3-publication-{}", uuid::Uuid::new_v4()),
        binding_digest: "3".repeat(64),
        envelope: Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: authorization.message_id().to_string(),
            sender_session_id: "gap3-source".into(),
            receiver_session_id: receiver_session_id,
            model_fingerprint,
            runtime_id,
            connector_id: GAP_CONNECTOR_ID.into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: format!("gap3-handle-{}", uuid::Uuid::new_v4()),
                connector_id: GAP_CONNECTOR_ID.into(),
                model_fingerprint: audit["model_fingerprint"].as_str().unwrap().into(),
                runtime_generation,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary:
                "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.".into(),
            fallback_reference: "occ1".into(),
        },
    };
    let prepared =
        PreparedNativeHandoff::new(authorization, publication).expect("fixture prepared handoff");
    assert!(
        native.admit(&prepared).is_err(),
        "a closed native session must refuse further admissions"
    );
}

/// Arm the source capture and drive the real resolve hook once, keeping the
/// lease so the test can observe its cancellation.
async fn armed_lease(native: &Arc<Occ3NativeSession>) -> SourceCaptureLease {
    native.arm_source(budget()).expect("arm the source budget");
    let hooks = native.source_hooks();
    (hooks.resolve)(InferenceStateRequest {
        model: fixture_model(),
        context: Default::default(),
        messages: Vec::new(),
        signal: None,
    })
    .await
    .expect("fixture resolve")
    .expect("capture lease")
}

/// The receiver-side invalidation contract: the inbox admission is revoked
/// (`Failed`, never reusable), the source lease is cancelled and the state is
/// fail-closed for any further native use.
fn assert_native_invalidated(native: &Arc<Occ3NativeSession>, message_id: &str, lease: &SourceCaptureLease) {
    let audit = native.audit();
    let record = audit["receiver"]
        .as_array()
        .expect("receiver audit")
        .iter()
        .find(|record| record["message_id"].as_str() == Some(message_id))
        .expect("the admitted record must stay journaled");
    assert_eq!(
        record["status"], serde_json::json!("FAILED"),
        "invalidate_all must revoke the admission: {audit}"
    );
    assert_eq!(
        lease.state(),
        SourceCaptureState::Cancelled,
        "the source capture lease must be cancelled"
    );
    assert!(
        native.arm_source(budget()).is_err(),
        "the native state must stay fail-closed after invalidation"
    );
}

/// `clear_queue` (agent_session.rs:12506) must invalidate the installed native
/// state: the receiver inbox admission is revoked and the source lease is
/// cancelled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_queue_invalidates_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    assert_eq!(lease.state(), SourceCaptureState::Prepared);
    fixture.session.clear_queue();
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// `clear_queued_agent_messages` (agent_session.rs:12582) must invalidate the
/// installed native state as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_queued_agent_messages_invalidates_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    fixture.session.clear_queued_agent_messages();
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// `request_abort` (agent_session.rs:13320 -> request_abort_inner:13325) must
/// invalidate the installed native state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_abort_invalidates_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    fixture.session.request_abort();
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// `abort` (agent_session.rs:13403 -> abort_inner:13416) must CLOSE the
/// installed native state: invalidation plus the best-effort runtime
/// retirement. A second admission attempt must fail closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_closes_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    tokio::time::timeout(Duration::from_secs(30), fixture.session.abort())
        .await
        .expect("abort must settle")
        .expect("abort on the idle fixture session");
    assert_native_invalidated(&fixture.native, &message_id, &lease);
    admit_after_close_errs(&fixture.native);
}

/// `set_model` (agent_session.rs:13556) must close the installed native state
/// BEFORE the model switch proceeds; the closed state is what makes any
/// later native use fail closed under the new model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_model_closes_the_installed_native_receiver_state_before_the_switch() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    fixture
        .session
        .set_model(fixture_model(), ModelSelectOptions::default())
        .await
        .expect("set_model must succeed after closing the native state");
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// `dispose` (agent_session.rs:8429) must invalidate the installed native
/// state; the synchronous teardown cannot await runtime retirement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispose_invalidates_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    fixture.session.dispose();
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// `dispose_async` (agent_session.rs:8123 -> dispose_async_once_body:8352)
/// must close the installed native state before the rest of teardown.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispose_async_closes_the_installed_native_receiver_state() {
    let fixture = native_fixture().await;
    let message_id = admit_envelope(&fixture.native);
    let lease = armed_lease(&fixture.native).await;
    tokio::time::timeout(Duration::from_secs(30), fixture.session.dispose_async(Some(false)))
        .await
        .expect("dispose_async must settle");
    assert_native_invalidated(&fixture.native, &message_id, &lease);
}

/// The delivery status of the revoked admission must never permit a text
/// fallback: `FAILED` is outside `permits_text_fallback`.
#[test]
fn revoked_native_admission_never_permits_text_fallback() {
    assert!(!crate::core::occ::occ3::inference::permits_text_fallback(
        Occ3DeliveryStatus::Failed
    ));
}
