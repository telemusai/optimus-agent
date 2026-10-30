//! GAP-2 follow-up regression tests (receiver-tests-audit.md, section 7
//! item 2): the production `DaemonAgentMessageController` OCC-3 native
//! wiring paths `daemon_occ3_native_wiring_tests.rs` leaves uncovered:
//!
//! - a send whose RECEIVER permit was invalidated between
//!   `authorize_native_handoff` and `send_prepared_native_handoff` must
//!   refuse with the invalidation error and never admit the envelope into
//!   the receiver inbox, and
//! - an authorize future that is alive when agent messaging pauses must
//!   still refuse: the pause gate is evaluated when the future runs, not
//!   captured when the controller call creates the future.
//!
//! The fixture duplicates `daemon_occ3_native_wiring_tests.rs`
//! `NativeHandoffFixture` exactly (that module is private and existing
//! files must not be edited): a REAL parent/child pair of `AgentSession`s
//! behind the REAL `AgentSessionDaemonAdapter` in a REAL `AgentDaemon`,
//! each native permit installed through the production
//! `install_experimental_native_occ3` seam, and the controller under test
//! built by the daemon's own factory. Everything is offline: the connector
//! answers locally and the scripted provider never leaves the process.
use super::*;
use crate::core::agent_messages::{
    AgentFamilyRosterResult, AgentSessionMessageController, AgentSessionMessageReceipt,
    AgentSessionMessageSendInput, AuthorizedNativeHandoff, PreparedNativeHandoff,
};
use crate::core::agent_session::AgentSession;
use crate::core::agent_session::occ3_state::{
    NativeSourceBudget, Occ3NativeSession, NATIVE_HANDOFF_TOOL,
};
use crate::core::agent_session_runtime::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionDaemonAdapter, AgentSessionRuntime,
    AgentSessionRuntimeMetadata as CoreRuntimeMetadata, CreateAgentSessionRuntimeFactory,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use crate::core::agent_session_services::AgentSessionCreationOptions as CoreCreationOptions;
use crate::core::occ::occ3::connector::{
    CaptureRequest, ConsumeRequest, NativePublication, NativePublicationRequest, NativeRetirement,
    NativeSessionEnrollment, Occ3Connector,
};
use crate::core::occ::occ3::inference::{
    PrepareInferenceRequest, NATIVE_INFERENCE_CONTRACT,
};
use crate::core::occ::occ3::types::{
    Occ3ConnectorCapabilities, Occ3Envelope, Occ3Error, Occ3HandleRef, Occ3Method,
    Occ3ModelIdentity, Occ3TransferMetrics,
};
use crate::core::resource_loader::DefaultResourceLoaderOptions;
use crate::core::settings_manager::SettingsManager;
use crate::core::auth_storage::AuthStorage;
use crate::core::model_registry::ModelRegistry;
use crate::core::session_manager::SessionManager;
use pi_agent_core::types::InferenceStateRequest;
use pi_ai::inference_attachment::{InferenceAttachment, InferenceStateLease, InferenceStateTarget};
use pi_ai::source_capture::{
    SourceCaptureIntent, SourceCaptureLease, SourceCaptureLimits, SourceCaptureProfile,
    SOURCE_CAPTURE_SCHEMA, SOURCE_SPAN_POLICY,
};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, TextContent, API_OPENAI_COMPLETIONS,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use std::time::Duration;

const GAP_CONNECTOR_ID: &str = "gap2-occ3-connector";
const GAP_METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fixture_model() -> pi_ai::types::Model {
    pi_ai::types::Model::new(
        "fixture-model",
        "fixture-model",
        API_OPENAI_COMPLETIONS,
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

/// The invalidation error an invalidated `Occ3NativeSession` surfaces
/// through its fail-closed `check()`: the exact string
/// `occ3_state.rs::failure()` returns. `admit` and every arming path
/// propagate it verbatim, so the send under test must refuse with it.
const NATIVE_PERMIT_INVALIDATED_ERROR: &str =
    "Experimental native handoff unavailable or uncertain; replay is forbidden";

/// Offline local connector: enrollment, capabilities, source-capture
/// preparation, publication and inference-lease preparation. No latent I/O.
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
            principal_digest: "gap2-fixture-principal".into(),
            identity: self.identity.clone(),
            runtime_id: "gap2-fixture-runtime".into(),
            runtime_generation: 7,
            access_domain: "gap2-fixture-domain".into(),
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
                capture_id: format!("gap2-capture-{}", uuid::Uuid::new_v4()),
                source_session_id: session_id.into(),
                source_request_id: format!("gap2-source-{}", uuid::Uuid::new_v4()),
                turn_nonce: format!("gap2-turn-{}", uuid::Uuid::new_v4()),
                model_fingerprint: self.identity.fingerprint(),
                runtime_id: "gap2-fixture-runtime".into(),
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
            reason: "gap2-fixture".into(),
        })
    }
    async fn publish_native_source(
        &self,
        request: NativePublicationRequest,
    ) -> Result<NativePublication, Occ3Error> {
        let intent = request.source.intent();
        Ok(NativePublication {
            publication_id: request.publication_id,
            binding_digest: "4".repeat(64),
            envelope: Occ3Envelope {
                schema: "optimus.occ3.v1".into(),
                protocol_version: 1,
                message_id: request.message_id,
                sender_session_id: intent.source_session_id.clone(),
                receiver_session_id: request.receiver_session_id,
                model_fingerprint: self.identity.fingerprint(),
                runtime_id: "gap2-fixture-runtime".into(),
                connector_id: GAP_CONNECTOR_ID.into(),
                latent_method: Occ3Method::StatebridgePrefix,
                state_handle: Occ3HandleRef {
                    handle_id: format!("gap2-handle-{}", uuid::Uuid::new_v4()),
                    connector_id: GAP_CONNECTOR_ID.into(),
                    model_fingerprint: self.identity.fingerprint(),
                    runtime_generation: 7,
                    expires_at_epoch: now() + 600,
                    state_size_bytes: 128,
                },
                control_summary:
                    "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.".into(),
                fallback_reference: "occ1".into(),
            },
        })
    }
    async fn prepare_inference(
        &self,
        request: PrepareInferenceRequest,
    ) -> Result<InferenceStateLease, Occ3Error> {
        let envelope = request.envelope;
        let attachment = InferenceAttachment {
            protocol_version: 1,
            request_id: request.request_id,
            message_id: envelope.message_id,
            connector_id: envelope.connector_id.clone(),
            receiver_session_id: envelope.receiver_session_id.clone(),
            model_fingerprint: envelope.model_fingerprint.clone(),
            runtime_id: envelope.runtime_id.clone(),
            runtime_generation: envelope.state_handle.runtime_generation,
            state_handle: envelope.state_handle.handle_id.clone(),
            expires_at_epoch: envelope.state_handle.expires_at_epoch,
            method: self.identity.latent_method.clone(),
        };
        InferenceStateLease::new(request.target, attachment).map_err(|_| {
            Occ3Error::TransferFailed {
                reason: "gap2-fixture-lease".into(),
            }
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
            connector_version: "gap2-fixture".into(),
            model_identity: self.identity.clone(),
            methods: vec![Occ3Method::StatebridgePrefix],
            tensor_formats: vec!["gap2-fixture".into()],
            max_state_bytes: 1024,
            state_ttl_seconds: 60,
            max_concurrent_handles: 2,
            cache_isolation: "gap2-fixture".into(),
            features: std::collections::BTreeMap::from([(
                "native_inference_attachment".into(),
                NATIVE_INFERENCE_CONTRACT.into(),
            )]),
        })
    }
    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> {
        Ok(self.identity.clone())
    }
    async fn capture_latent(&self, _request: CaptureRequest) -> Result<Occ3HandleRef, Occ3Error> {
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

/// Controller double for sessions whose handoff tool is never driven by these
/// tests (child): inert, like the isolated harness doubles.
struct InertController;

impl AgentSessionMessageController for InertController {
    fn roster(&self) -> pi_ai::types::BoxFuture<Result<AgentFamilyRosterResult, String>> {
        Box::pin(async { Ok(AgentFamilyRosterResult::default()) })
    }
    fn await_pending_child_publication(
        &self,
        _selector: String,
    ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn send_agent_message(
        &self,
        _input: AgentSessionMessageSendInput,
    ) -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        Box::pin(async { Err("gap2 fixture controller is inert".into()) })
    }
}

/// The completed provider turn the scripted session returns. The child is a
/// depth-1 session, so its text carries the terminal RLM marker
/// (`queue_stop_tests::install_provider` precedent).
fn scripted_completion(model: &pi_ai::types::Model) -> AssistantMessage {
    AssistantMessage {
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        content: vec![ContentBlock::Text(TextContent::new(
            "done\nRLM_CHILD_STATUS: complete",
        ))],
        stop_reason: "stop".into(),
        timestamp: now_millis() as i64,
        ..Default::default()
    }
}

/// A provider that completes the turn immediately.
fn install_scripted_provider(session: &Arc<AgentSession>) {
    session.agent.set_stream_fn(Arc::new(|model, _, _| {
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            let message = scripted_completion(&model);
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason.clone(),
                message,
            });
            stream
        })
    }));
}

type StateSlot = Arc<StdMutex<Option<Arc<StdMutex<ActiveSessionState>>>>>;

fn state_slot() -> StateSlot {
    Arc::new(StdMutex::new(None))
}

fn slot_reader(
    slot: &StateSlot,
) -> Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync> {
    let slot = Arc::clone(slot);
    Arc::new(move || slot.lock().expect("state slot poisoned").clone())
}

/// `AuthorizedNativeHandoff` has no `Debug` projection (its fields stay
/// private), so authorize refusals unwrap through this helper.
fn authorize_refusal(
    result: Result<AuthorizedNativeHandoff, String>,
    context: &str,
) -> String {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

/// One real session over the in-memory stack, registered in the daemon behind
/// the real adapter (`daemon_parity_tests::ParityFixture` shape).
struct ResidentSession {
    session: Arc<AgentSession>,
    state: Arc<StdMutex<ActiveSessionState>>,
    native: Option<Arc<Occ3NativeSession>>,
    controller: Option<Arc<dyn AgentSessionMessageController>>,
}

struct NativeHandoffFixture {
    _directory: tempfile::TempDir,
    daemon: Arc<AgentDaemon>,
    connector: Arc<FixtureConnector>,
    parent: ResidentSession,
    child: ResidentSession,
}

impl NativeHandoffFixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().expect("temp dir");
        let cwd = directory.path().join("workspace").to_string_lossy().into_owned();
        let agent_dir = directory
            .path()
            .join("agent")
            .to_string_lossy()
            .into_owned();
        std::fs::create_dir_all(&cwd).expect("workspace dir");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let socket_path = directory
            .path()
            .join("native-handoff-gap.sock")
            .to_string_lossy()
            .into_owned();
        let daemon = AgentDaemon::new(
            socket_path.clone(),
            DaemonModeOptions {
                socket_path: Some(socket_path),
                default_session_config: AgentSessionRuntimeConfig {
                    cwd: Some(cwd.clone()),
                    agent_dir: Some(agent_dir.clone()),
                    ..AgentSessionRuntimeConfig::default()
                },
                create_runtime: Arc::new(|_| {
                    Box::pin(async { Err("gap2 fixture does not create runtimes".to_string()) })
                }),
                worker: None,
            },
        );
        let connector = Arc::new(FixtureConnector::new());

        let parent = Self::add_resident_session(
            &daemon,
            &connector,
            &cwd,
            &agent_dir,
            0,
            None,
            true,
        )
        .await;
        let child = Self::add_resident_session(
            &daemon,
            &connector,
            &cwd,
            &agent_dir,
            1,
            Some(parent.session.session_id()),
            true,
        )
        .await;
        Self {
            _directory: directory,
            daemon,
            connector,
            parent,
            child,
        }
    }

    /// Build one session, optionally enroll+install its native permit, and
    /// register it behind the real daemon adapter with the real metadata
    /// projection (`add_runtime`).
    async fn add_resident_session(
        daemon: &Arc<AgentDaemon>,
        connector: &Arc<FixtureConnector>,
        cwd: &str,
        agent_dir: &str,
        rlm_depth: i64,
        parent_session_id: Option<String>,
        with_native: bool,
    ) -> ResidentSession {
        let settings = Arc::new(StdMutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));
        let session_manager = Arc::new(StdMutex::new(
            SessionManager::in_memory(Some(cwd), Some(agent_dir)).expect("session manager"),
        ));
        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            Default::default(),
            None,
        )));
        let model_registry = Arc::new(StdMutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(Default::default(), None),
        )));
        auth_storage
            .lock()
            .await
            .set_runtime_api_key("unit-fixture", "gap2-fixture-key");
        model_registry
            .lock()
            .expect("model registry poisoned")
            .set_runtime_api_key("unit-fixture", "gap2-fixture-key");
        let services = create_agent_session_services(CreateAgentSessionServicesOptions {
            cwd: cwd.to_string(),
            agent_dir: Some(agent_dir.to_string()),
            auth_storage: Some(Arc::clone(&auth_storage)),
            settings_manager: Some(Arc::clone(&settings)),
            model_registry: Some(Arc::clone(&model_registry)),
            extension_flag_values: None,
            no_builtin_herdr_reporter: Some(true),
            telemetry_disabled: Some(true),
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: cwd.to_string(),
                agent_dir: agent_dir.to_string(),
                settings_manager: Some(Arc::clone(&settings)),
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            }),
        })
        .await
        .expect("session services");
        let services = Arc::new(services);
        let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
            services: Arc::clone(&services),
            session_manager: Arc::clone(&session_manager),
            session_start_event: None,
            creation: CoreCreationOptions {
                model: Some(fixture_model()),
                no_tools: Some("all".to_string()),
                prewarm_ipython_kernel: Some(false),
                telemetry_disabled: Some(true),
                rlm_depth: Some(rlm_depth),
                rlm_parent_agent: parent_session_id
                    .as_ref()
                    .map(|_| "gap2-parent-fixture".to_string()),
                semantic_parent_session_id: parent_session_id.clone(),
                ..Default::default()
            },
        })
        .await
        .expect("agent session");
        let session = created.session;
        install_scripted_provider(&session);

        // The controller this session's handoff tool would use. The parent's
        // controller is the production wiring under test; the other residents
        // only need a controller so the installation seam accepts them.
        let slot = state_slot();
        let controller = daemon.create_agent_message_controller(slot_reader(&slot));

        let native = if with_native {
            let native = Occ3NativeSession::enroll(
                &session.session_id(),
                &fixture_model(),
                Arc::clone(connector) as Arc<dyn Occ3Connector>,
            )
            .await
            .expect("fixture enrollment");
            session
                .install_experimental_native_occ3(
                    native.clone(),
                    if rlm_depth > 0 {
                        Arc::new(InertController) as Arc<dyn AgentSessionMessageController>
                    } else {
                        controller.clone()
                    },
                )
                .expect("install the native permit");
            Some(native)
        } else {
            None
        };

        let refuse: CreateAgentSessionRuntimeFactory = Arc::new(|_| {
            Box::pin(async { Err("gap2 fixture does not create runtimes".to_string()) })
        });
        let core_metadata = if rlm_depth > 0 {
            CoreRuntimeMetadata {
                kind: "subagent".to_string(),
                created_at: now_millis(),
                parent_session_id: parent_session_id.clone(),
                ..CoreRuntimeMetadata::default()
            }
        } else {
            CoreRuntimeMetadata::top_level()
        };
        let runtime = AgentSessionRuntime::new(
            Arc::clone(&session),
            services,
            refuse,
            Vec::new(),
            None,
            None,
            core_metadata,
            None,
        );
        let adapter: Arc<dyn DaemonSession> =
            Arc::new(AgentSessionDaemonAdapter::new(runtime));
        let handle_metadata = AgentSessionRuntimeMetadata {
            kind: Some(if rlm_depth > 0 {
                "subagent".to_string()
            } else {
                "top-level".to_string()
            }),
            parent_session_id: if rlm_depth > 0 {
                parent_session_id
            } else {
                None
            },
            ..AgentSessionRuntimeMetadata::default()
        };
        let state = daemon
            .add_runtime(
                AgentSessionRuntimeHandle {
                    session: adapter,
                    metadata: handle_metadata,
                    model_fallback_message: None,
                    new_session: None,
                    switch_session: None,
                    fork: None,
                    import_from_jsonl: None,
                },
                None,
                None,
                None,
            )
            .await
            .expect("daemon registers the runtime");
        *slot.lock().expect("state slot poisoned") = Some(Arc::clone(&state));
        ResidentSession {
            session,
            state,
            native,
            controller: Some(controller),
        }
    }

    fn active_id(state: &Arc<StdMutex<ActiveSessionState>>) -> String {
        state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone()
    }

/// The production flow the occ3_handoff tool drives: authorize through the
    /// controller, resolve the armed capture lease, publish through the
    /// connector, then bind the prepared handoff.
    async fn prepared_handoff(&self) -> Prepared {
        let authorized = self
            .parent
            .controller
            .clone()
            .unwrap()
            .authorize_native_handoff(
                "child".to_string(),
                Self::active_id(&self.child.state),
            )
            .await
            .expect("authorize the native handoff");
        let parent_native = self.parent.native.as_ref().expect("parent permit");
        parent_native
            .arm_source(NativeSourceBudget {
                retained_count_limit: 8,
                max_capture_bytes: 1024,
                deadline: std::time::Instant::now() + Duration::from_secs(60),
                max_response_bytes: 4096,
            })
            .expect("arm the source budget");
        let hooks = parent_native.source_hooks();
        let lease = (hooks.resolve)(InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
            signal: None,
        })
        .await
        .expect("fixture resolve")
        .expect("capture lease");
        let publication = self
            .connector
            .publish_native_source(NativePublicationRequest {
                source: lease,
                receiver_session_id: authorized.target_session_id().to_string(),
                message_id: authorized.message_id().to_string(),
                publication_id: format!("gap2-publication-{}", uuid::Uuid::new_v4()),
                tool_call_id: "call-gap2-fixture".to_string(),
                tool_name: NATIVE_HANDOFF_TOOL.to_string(),
                tool_arguments: serde_json::json!({
                    "receiver_role": "child",
                    "receiver_name": "gap2-child-fixture"
                }),
            })
            .await
            .expect("fixture publication");
        let prepared = PreparedNativeHandoff::new(authorized, publication)
            .expect("fixture prepared handoff");
        Prepared { prepared }
    }
}

struct Prepared {
    prepared: crate::core::agent_messages::PreparedNativeHandoff,
}

impl Prepared {
    fn message_id(&self) -> String {
        self.prepared.authorization().message_id().to_string()
    }
}

/// GAP-2 follow-up (a): the send path must re-validate the RECEIVER permit
/// at send time. A prepared handoff whose receiver permit was invalidated
/// between authorize and send refuses with the invalidation error - the
/// permit is still installed, so the refusal comes from the closed permit's
/// own fail-closed `admit` gate - and the receiver inbox journals nothing
/// for that message id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_send_refuses_a_receiver_permit_invalidated_between_authorize_and_send() {
    let fixture = NativeHandoffFixture::new().await;
    // authorize + arm + resolve + publish all happen while the child permit
    // is live; the prepared handoff is fully bound to the live pair.
    let prepared = fixture.prepared_handoff().await;
    let message_id = prepared.message_id();
    // Between authorize and send: the receiver's native permit is
    // invalidated (the same closed flag the AgentSession abort/queue-clear
    // triggers set in agent_session.rs).
    fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .invalidate();
    let error = tokio::time::timeout(
        Duration::from_secs(30),
        fixture
            .parent
            .controller
            .clone()
            .unwrap()
            .send_prepared_native_handoff(prepared.prepared),
    )
    .await
    .expect("the send must settle")
    .expect_err("an invalidated receiver permit must refuse the send");
    assert_eq!(error, NATIVE_PERMIT_INVALIDATED_ERROR);
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    let records = audit["receiver"]
        .as_array()
        .expect("receiver audit")
        .iter()
        .filter(|record| record["message_id"].as_str() == Some(message_id.as_str()))
        .count();
    assert_eq!(
        records, 0,
        "the refused send must never admit the envelope into the receiver inbox: {audit}"
    );
}

/// GAP-2 follow-up (b): the pause gate is evaluated when the authorize
/// future is driven, not captured when the controller call creates it. The
/// authorize body (`daemon_mode.rs` `authorize_native_handoff`) runs
/// synchronously from its gate read to `from_catalog` with no interior
/// await, so the observable "during the authorize future" window is the
/// future's lifetime between creation and first poll: a pause landing in
/// that window must still refuse, and the gate must be live afterwards -
/// unpausing lets a fresh authorize through.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_authorize_refuses_a_pause_toggled_during_the_authorize_future() {
    let fixture = NativeHandoffFixture::new().await;
    assert!(
        !fixture.daemon.agent_messages_paused.load(Ordering::SeqCst),
        "fixture precondition: messaging starts unpaused"
    );
    let controller = fixture.parent.controller.clone().unwrap();
    let child_active = NativeHandoffFixture::active_id(&fixture.child.state);
    // The authorize future is created while messaging is unpaused...
    let pending =
        controller.authorize_native_handoff("child".to_string(), child_active.clone());
    // ...and while that future is alive - created, not yet driven to
    // completion - agent messaging pauses.
    fixture.daemon.agent_messages_paused.store(true, Ordering::SeqCst);
    let error = authorize_refusal(
        pending.await,
        "an authorize alive when messaging paused must still refuse",
    );
    assert_eq!(error, "Agent messaging is paused".to_string());
    // The gate is live, not latched: unpausing lets a fresh authorize
    // through, proving the refusal above came from the gate read at
    // drive time rather than a fixture artifact.
    fixture.daemon.agent_messages_paused.store(false, Ordering::SeqCst);
    controller
        .authorize_native_handoff("child".to_string(), child_active)
        .await
        .expect("a fresh authorize must succeed once messaging is unpaused");
}
