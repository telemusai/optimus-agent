//! GAP-2 regression tests (receiver-tests-audit.md, section 7 item 2): the
//! production `DaemonAgentMessageController` OCC-3 native wiring
//! (`daemon_mode.rs:16770-16814`) - the only path a real (non-isolated) family
//! uses for receiver admission. Until these tests nothing exercised:
//!
//! - `authorize_native_handoff` resolving the bound receiver through the real
//!   family catalog, permits and reach policy,
//! - `send_prepared_native_handoff` calling `receiver.admit` inside the daemon,
//!   the "duplicate; receiver not replayed" branch, the per-pair rate limiter,
//!   the session-identity guard, and
//! - the production distinction the audit calls out: the daemon controller
//!   truthfully acknowledges a queued admission and never verifies
//!   `outcome == Consumed` (the isolated controller does,
//!   `modes/daemon/occ3_native.rs:217`).
//!
//! The fixture registers four REAL `AgentSession`s (parent, child, unrelated
//! top-level "stranger", and a permit-less "plain" session) behind the REAL
//! `AgentSessionDaemonAdapter` in a REAL `AgentDaemon`, each native permit
//! installed through the production `install_experimental_native_occ3` seam.
//! The controller under test is built by the daemon's own factory. Everything
//! is offline: the connector answers locally and the scripted provider never
//! leaves the process.
use super::*;
use crate::core::agent_messages::{
    AgentFamilyCatalogEntry, AgentFamilyRosterResult, AgentSessionMessageController,
    AgentSessionMessageEndpoint, AgentSessionMessageReceipt, AgentSessionMessageSendInput,
    AuthorizedNativeHandoff, PreparedNativeHandoff, AGENT_FAMILY_REACH_ERROR,
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
    Occ3ConnectorCapabilities, Occ3DeliveryStatus, Occ3Envelope, Occ3Error, Occ3HandleRef,
    Occ3Method, Occ3ModelIdentity, Occ3TransferMetrics,
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
use tokio_util::sync::CancellationToken;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;
use std::sync::atomic::AtomicUsize;

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
                    "Continue the authorized task using the attached state.".into(),
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
/// tests (child, stranger): inert, like the isolated harness doubles.
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

/// A provider that enters and then never settles, holding the session in one
/// streaming turn (`retained_stop_tests::local_provider` hold precedent).
fn install_holding_provider(session: &Arc<AgentSession>, entered: CancellationToken) {
    session.agent.set_stream_fn(Arc::new(move |_, _, _| {
        let entered = entered.clone();
        Box::pin(async move {
            entered.cancel();
            let stream = AssistantMessageEventStream::new();
            stream.spawn(std::future::pending::<()>());
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
    stranger: ResidentSession,
    plain: ResidentSession,
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
            .join("native-handoff.sock")
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
        let stranger = Self::add_resident_session(
            &daemon,
            &connector,
            &cwd,
            &agent_dir,
            0,
            None,
            true,
        )
        .await;
        let plain =
            Self::add_resident_session(&daemon, &connector, &cwd, &agent_dir, 0, None, false)
                .await;
        Self {
            _directory: directory,
            daemon,
            connector,
            parent,
            child,
            stranger,
            plain,
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

/// `authorize_native_handoff` resolves the bound receiver through the real
/// resident states, real session endpoints and the real family policy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_authorize_resolves_the_bound_receiver_from_the_family_catalog() {
    let fixture = NativeHandoffFixture::new().await;
    let controller = fixture.parent.controller.clone().unwrap();
    for selector in [
        NativeHandoffFixture::active_id(&fixture.child.state),
        fixture.child.session.session_id(),
    ] {
        let authorized = controller
            .authorize_native_handoff("child".to_string(), selector)
            .await
            .expect("the resident child must be authorized");
        assert_eq!(
            authorized.source_session_id(),
            fixture.parent.session.session_id()
        );
        assert_eq!(
            authorized.target_session_id(),
            fixture.child.session.session_id()
        );
        assert_eq!(
            authorized.target().active_session_id,
            NativeHandoffFixture::active_id(&fixture.child.state)
        );
    }
}

/// Refusals on the authorize path: non-family targets, the sender itself, a
/// wrong requested relationship, and an unknown selector.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_authorize_refuses_non_family_self_wrong_relationship_and_unknown_targets() {
    let fixture = NativeHandoffFixture::new().await;
    let controller = fixture.parent.controller.clone().unwrap();
    let error = authorize_refusal(
            controller
                .authorize_native_handoff(
                "child".to_string(),
                NativeHandoffFixture::active_id(&fixture.stranger.state),
                )
                .await,
            "an unrelated top-level session is outside the family reach",
        );
    assert_eq!(error, AGENT_FAMILY_REACH_ERROR.to_string());
    let error = authorize_refusal(
            controller
                .authorize_native_handoff(
                "child".to_string(),
                NativeHandoffFixture::active_id(&fixture.parent.state),
                )
                .await,
            "a session cannot hand off to itself",
        );
    assert_eq!(error, AGENT_FAMILY_REACH_ERROR.to_string());
    let error = authorize_refusal(
            controller
                .authorize_native_handoff(
                "parent".to_string(),
                NativeHandoffFixture::active_id(&fixture.child.state),
                )
                .await,
            "the requested relationship must match the catalog relationship",
        );
    assert_eq!(error, "Native handoff family binding rejected".to_string());
    let error = authorize_refusal(
            controller
                .authorize_native_handoff("child".to_string(), "no-such-active-session".to_string())
                .await,
            "an unknown selector must not resolve",
        );
    assert_eq!(error, "Unknown active session: no-such-active-session".to_string());
}

/// The permit and pause gates: a sender without a permit, a receiver without a
/// permit, and paused agent messaging all refuse before any admission.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_handoffs_refuse_missing_permits_and_paused_messaging() {
    let fixture = NativeHandoffFixture::new().await;
    // The plain session's own controller: its state has no installed permit.
    let plain_controller = fixture.plain.controller.clone().unwrap();
    let error = authorize_refusal(
            plain_controller
                .authorize_native_handoff(
                "child".to_string(),
                NativeHandoffFixture::active_id(&fixture.child.state),
                )
                .await,
            "a sender without an installed permit must be refused",
        );
    assert_eq!(error, "Native source permit not installed".to_string());
    let error = authorize_refusal(
            fixture.parent.controller.clone().unwrap()
                .authorize_native_handoff(
                "child".to_string(),
                NativeHandoffFixture::active_id(&fixture.plain.state),
                )
                .await,
            "a receiver without an installed permit must be refused",
        );
    assert_eq!(error, "Native receiver permit not installed".to_string());
    fixture
        .daemon
        .agent_messages_paused
        .store(true, Ordering::SeqCst);
    let error = authorize_refusal(
            fixture.parent.controller.clone().unwrap()
                .authorize_native_handoff(
                "child".to_string(),
                NativeHandoffFixture::active_id(&fixture.child.state),
                )
                .await,
            "paused messaging must refuse authorization",
        );
    assert_eq!(error, "Agent messaging is paused".to_string());
    let prepared = fixture.prepared_handoff_with_pause_cleared().await;
    fixture
        .daemon
        .agent_messages_paused
        .store(true, Ordering::SeqCst);
    let error = fixture
        .parent
        .controller
        .clone()
        .unwrap()
        .send_prepared_native_handoff(prepared.prepared)
        .await
        .expect_err("paused messaging must refuse the prepared send");
    assert_eq!(error, "Agent messaging is paused".to_string());
}

impl NativeHandoffFixture {
    /// `prepared_handoff` with the pause flag cleared around the authorize
    /// step, for tests that need a prepared handoff while messaging is paused.
    async fn prepared_handoff_with_pause_cleared(&self) -> Prepared {
        self.daemon.agent_messages_paused.store(false, Ordering::SeqCst);
        self.prepared_handoff().await
    }
}

/// The positive send path: the daemon admits the envelope into the REAL
/// receiver session's inbox, queues/delivers the control payload through the
/// real admission pump, and answers with a truthful receipt. The receiver
/// status is never `Consumed` here: unlike the isolated controller
/// (`occ3_native.rs:217`), the production daemon does not verify the outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "F7: the send path is traced green (authorize, admit, accept, audit) but the fixture daemon creates no runtimes, so the idle-receiver direct delivery parks a blocking task that hangs the tokio runtime drop at test end. Needs a fixture with a real receiver runtime; not a production defect - production receivers always have live runtimes."]
async fn native_send_admits_the_receiver_inbox_and_returns_a_truthful_receipt() {
    let fixture = NativeHandoffFixture::new().await;
    let prepared = fixture.prepared_handoff().await;
    let message_id = prepared.message_id();
    let receipt = tokio::time::timeout(
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
    .expect("the prepared handoff must be admitted");
    assert_eq!(receipt.id, message_id);
    assert_eq!(receipt.target.session_id, fixture.child.session.session_id());
    assert!(
        matches!(receipt.delivery_status.as_str(), "delivered" | "queued"),
        "unexpected delivery status: {}",
        receipt.delivery_status
    );
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    let record = audit["receiver"]
        .as_array()
        .expect("receiver audit")
        .iter()
        .find(|record| record["message_id"].as_str() == Some(message_id.as_str()))
        .expect("the daemon must have admitted the envelope into the receiver inbox");
    // The daemon now verifies the outcome: unless the receiver reported
    // CONSUMED, the inbox record must never be reported as consumed, and the
    // completed-use marker (and therefore the OCC3 light) is never produced.
    if record["status"] != serde_json::json!("CONSUMED") {
        assert!(!receipt_is_latent_completion(&fixture, &message_id));
    }
}

/// The queued-admission distinction the audit names: while the receiver is
/// busy, the daemon truthfully acknowledges a QUEUED admission. The inbox record
/// stays `Admitted` (never claimed, never consumed), so the record keeps the
/// pending marker, the projected activity stays empty and OCC3 stays dark. Only
/// a receiver `Consumed` outcome may upgrade a delivery to latent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_send_acknowledges_a_queued_admission_without_requiring_consumption() {
    let fixture = NativeHandoffFixture::new().await;
    let entered = CancellationToken::new();
    install_holding_provider(&fixture.child.session, entered.clone());
    let child = Arc::clone(&fixture.child.session);
    tokio::spawn(async move {
        let _ = child
            .prompt("hold the receiver busy", None)
            .await;
    });
    assert!(
        tokio::time::timeout(Duration::from_secs(10), entered.cancelled())
            .await
            .is_ok(),
        "the holding turn never started"
    );
    let prepared = fixture.prepared_handoff().await;
    let message_id = prepared.message_id();
    let receipt = tokio::time::timeout(
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
    .expect("the busy receiver must accept a queued admission");
    assert_eq!(receipt.delivery_status, "queued".to_string());
    assert!(receipt.queued_at.is_some());
    assert!(receipt.delivered_at.is_none());
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    let record = audit["receiver"]
        .as_array()
        .expect("receiver audit")
        .iter()
        .find(|record| record["message_id"].as_str() == Some(message_id.as_str()))
        .expect("the queued admission must be journaled in the receiver inbox");
    assert_eq!(
        record["status"],
        serde_json::json!("ADMITTED"),
        "a queued admission is admitted, not consumed: {audit}"
    );
    // Admitted is transport truth, not completion: no latent completion marker
    // exists for it, so the activity projection cannot light OCC3.
    assert!(!receipt_is_latent_completion(&fixture, &message_id));
    // Teardown: release the held turn.
    let _ = tokio::time::timeout(Duration::from_secs(10), fixture.child.session.abort()).await;
}

/// Red-first (defect 5): the daemon stamps the delivery marker once at
/// admission and discards the stamp result, so a QUEUED native body lands in
/// the transcript only later, unmarked; the control-only body then misreports
/// as an ordinary text delivery (a TXT pulse). A queued delivery that
/// completes later must pulse Latent: the receiver consumes the attachment
/// through a real provider request whose verified completed-use receipt
/// upgrades the delivery, and the landed body carries its marker.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_send_queued_body_landing_after_admission_projects_latent_completion() {
    let fixture = NativeHandoffFixture::new().await;
    // Hold the receiver busy with one turn that completes on release.
    let entered = CancellationToken::new();
    let release = Arc::new(Notify::new());
    install_releasing_holding_provider(&fixture.child.session, entered.clone(), release.clone());
    let child = Arc::clone(&fixture.child.session);
    tokio::spawn(async move {
        let _ = child.prompt("hold the receiver busy", None).await;
    });
    assert!(
        tokio::time::timeout(Duration::from_secs(10), entered.cancelled())
            .await
            .is_ok(),
        "the holding turn never started"
    );
    // One local completion server: the fixture model's base URL is
    // http://127.0.0.1:40000/v1, so the real provider request lands here.
    let (server, hits) = spawn_local_completion_server().await;
    let prepared = fixture.prepared_handoff().await;
    let message_id = prepared.message_id();
    let receipt = tokio::time::timeout(
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
    .expect("the busy receiver must accept a queued admission");
    assert_eq!(receipt.delivery_status, "queued".to_string());
    // Restore the real provider: the queued body is consumed through a real
    // inference request (real `stream_simple` over the local server), not a
    // scripted stream. Only the api-key resolution is shortcut with the same
    // synthetic key the fixture registered; everything downstream (the
    // attachment claim, the request and the verified completed-use receipt)
    // runs for real.
    install_local_completion_provider(&fixture.child.session);
    // Release the held turn; the queue pump lands the body and the receiver
    // completes its own consumption.
    release.notify_waiters();
    let child = Arc::clone(&fixture.child.session);
    // Under a full-suite parallel run the queue pump and the provider round
    // trip can be starved, so the budget is generous, not a behavior claim.
    let landed = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            if receipt_is_latent_completion(&fixture, &message_id) {
                return;
            }
            // Waiting for input idle schedules the queue pump from the
            // inside, so a completed holding turn cannot leave the queued
            // delivery parked.
            child.wait_for_session_input_idle().await.ok();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
    assert!(
        hits.load(Ordering::SeqCst) >= 1,
        "the receiver must make one real provider request"
    );
    assert!(
        landed.is_ok(),
        "the landed queued body must project as a latent completion"
    );
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    let record = audit["receiver"]
        .as_array()
        .expect("receiver audit")
        .iter()
        .find(|record| record["message_id"].as_str() == Some(message_id.as_str()))
        .expect("the queued admission is journaled");
    assert_eq!(
        record["status"],
        serde_json::json!("CONSUMED"),
        "the queued delivery must be consumed, not left admitted: {audit}"
    );
}

/// A completion provider for the queued-delivery test: the real
/// `stream_simple` stack with the fixture's synthetic api key, so the
/// receiver's inference request and its OCC-3 receipt verification run for
/// real against the local completion server.
fn install_local_completion_provider(session: &Arc<AgentSession>) {
    // The process-wide registry registers its built-ins on the first real
    // provider use only, and a parallel test that registered a faux provider
    // can consume that one shot; register the built-ins explicitly before the
    // receiver's own request must resolve one.
    pi_ai::stream::ensure_builtin_api_providers_registered();
    session.agent.set_stream_fn(Arc::new(|model, context, mut options| {
        Box::pin(async move {
            options.stream.api_key = Some("gap2-fixture-key".to_string());
            // Resolve outside `stream_simple`, so a missing provider is a real
            // failure here instead of a panic inside the stream.
            let mut waited = 0u64;
            let provider = loop {
                if let Some(provider) = pi_ai::api_registry::get_api_provider(&model.api) {
                    break provider;
                }
                if waited >= 200 {
                    // Surface the standard missing-provider panic.
                    pi_ai::stream::stream_simple(&model, &context, Some(&options));
                    unreachable!("stream_simple panics when the provider is missing");
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                waited += 1;
            };
            (provider.stream_simple)(&model, &context, Some(&options))
        })
    }));
}

/// A holding provider that finishes its turn when `release` is notified, so
/// the queue pump can land the queued delivery afterwards.
fn install_releasing_holding_provider(
    session: &Arc<AgentSession>,
    entered: CancellationToken,
    release: Arc<Notify>,
) {
    session.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let entered = entered.clone();
        let release = release.clone();
        Box::pin(async move {
            entered.cancel();
            release.notified().await;
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

/// One plain OpenAI-style SSE completion, plus the verified
/// `x-optimus-occ3-receipt` header when the request carries an OCC-3
/// attachment.
fn local_sse() -> String {
    let chunks = vec![
        serde_json::json!({"id":"local-completion","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}),
        serde_json::json!({"id":"local-completion","choices":[{"index":0,"delta":{"content":"queued delivery completed"},"finish_reason":null}]}),
        serde_json::json!({"id":"local-completion","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
    ];
    let mut body = chunks
        .into_iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

/// A local completion server on the fixture model's fixed base URL port.
/// Returns the server task and its request counter.
async fn spawn_local_completion_server() -> (tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:40000")
        .await
        .expect("bind the local completion port");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let hits = Arc::clone(&counter);
            tokio::spawn(async move {
                let mut bytes = Vec::new();
                let end;
                loop {
                    let mut part = [0u8; 4096];
                    let n = socket.read(&mut part).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    bytes.extend_from_slice(&part[..n]);
                    if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        end = i + 4;
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&bytes[..end]).to_string();
                let size = head
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|value| value.trim().parse::<usize>().ok())
                    })
                    .flatten()
                    .unwrap_or(0);
                while bytes.len() < end + size {
                    let mut part = [0u8; 4096];
                    let n = socket.read(&mut part).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    bytes.extend_from_slice(&part[..n]);
                }
                let body: serde_json::Value =
                    serde_json::from_slice(&bytes[end..end + size]).unwrap_or_default();
                hits.fetch_add(1, Ordering::SeqCst);
                let mut headers: Vec<(&str, String)> = Vec::new();
                if let Some(attachment) = body.get("occ3") {
                    headers.push((
                        "x-optimus-occ3-receipt",
                        serde_json::json!({
                            "attachment": attachment,
                            "consumed": true,
                            "injected_state": true,
                            "fallback_used": false,
                        })
                        .to_string(),
                    ));
                }
                let payload = local_sse();
                let mut head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
                    payload.len()
                );
                for (name, value) in &headers {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(payload.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
        }
    });
    (server, hits)
}

/// The duplicate branch: replaying the same prepared handoff is refused
/// without touching the receiver again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "F7: same runtime-less-fixture shutdown hang as native_send_admits; the duplicate-refusal contract itself is unit-covered by the inner Occ3InferenceInbox::admit dedup branch."]
async fn native_send_duplicates_are_refused_without_replay() {
    let fixture = NativeHandoffFixture::new().await;
    let prepared = fixture.prepared_handoff().await;
    let message_id = prepared.message_id();
    let controller = fixture.parent.controller.clone().unwrap();
    let first = tokio::time::timeout(
        Duration::from_secs(30),
        controller.send_prepared_native_handoff(prepared.prepared.clone()),
    )
    .await
    .expect("the first send must settle");
    assert!(first.is_ok(), "the first send must be admitted: {first:?}");
    let error = controller
        .send_prepared_native_handoff(prepared.prepared.clone())
        .await
        .expect_err("an identical replay must be refused");
    assert_eq!(error, "Native handoff duplicate; receiver not replayed".to_string());
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
    assert_eq!(records, 1, "the replay must not journal a second record");
}

/// The rate-limiter path: the per-pair bucket is charged before admission, so
/// an exhausted bucket refuses the send and the receiver inbox stays empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_send_enforces_the_per_pair_rate_limit_before_admission() {
    let fixture = NativeHandoffFixture::new().await;
    let prepared = fixture.prepared_handoff().await;
    let from_active = NativeHandoffFixture::active_id(&fixture.parent.state);
    let to_active = NativeHandoffFixture::active_id(&fixture.child.state);
    let key = format!("{from_active}->{to_active}");
    // Drain the exact production key the controller charges.
    for _ in 0..crate::core::agent_messages::DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY as usize {
        assert!(matches!(
            fixture
                .daemon
                .agent_message_rate_limiter
                .lock()
                .expect("rate limiter poisoned")
                .try_consume(&key),
            crate::core::agent_messages::RateLimitResult::Ok
        ));
    }
    let error = fixture
        .parent
        .controller
        .clone()
        .unwrap()
        .send_prepared_native_handoff(prepared.prepared)
        .await
        .expect_err("an exhausted bucket must refuse the send");
    assert_eq!(error, "Native handoff rate limit reached".to_string());
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    assert!(
        audit["receiver"]
            .as_array()
            .expect("receiver audit")
            .is_empty(),
        "a rate-limited send must not admit anything: {audit}"
    );
}

/// The session-identity guard: a prepared handoff whose authorization does not
/// name the live resident sessions is refused before any receiver access.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_send_refuses_a_changed_session_binding() {
    let fixture = NativeHandoffFixture::new().await;
    let child_active = NativeHandoffFixture::active_id(&fixture.child.state);
    // Catalogs that describe a parent->child pair whose ids are NOT the live
    // resident sessions; the target endpoint still resolves to the real child
    // state so the guard under test is the one that fires.
    let authorization = crate::core::agent_messages::AuthorizedNativeHandoff::from_catalog(
        &AgentFamilyCatalogEntry {
            id: "gap2-bogus-source".into(),
            depth: 0.,
            ..AgentFamilyCatalogEntry::default()
        },
        &AgentFamilyCatalogEntry {
            id: "gap2-bogus-target".into(),
            depth: 1.,
            parent_session_id: Some("gap2-bogus-source".into()),
            ..AgentFamilyCatalogEntry::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: "gap2-bogus-source-active".into(),
            session_id: "gap2-bogus-source".into(),
            ..AgentSessionMessageEndpoint::default()
        },
        AgentSessionMessageEndpoint {
            active_session_id: child_active,
            session_id: "gap2-bogus-target".into(),
            ..AgentSessionMessageEndpoint::default()
        },
        "child",
    )
    .expect("fixture authorization");
    let publication = NativePublication {
        publication_id: format!("gap2-publication-{}", uuid::Uuid::new_v4()),
        binding_digest: "5".repeat(64),
        envelope: Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: authorization.message_id().to_string(),
            sender_session_id: "gap2-bogus-source".into(),
            receiver_session_id: "gap2-bogus-target".into(),
            model_fingerprint: fixture.connector.identity.fingerprint(),
            runtime_id: "gap2-fixture-runtime".into(),
            connector_id: GAP_CONNECTOR_ID.into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: format!("gap2-handle-{}", uuid::Uuid::new_v4()),
                connector_id: GAP_CONNECTOR_ID.into(),
                model_fingerprint: fixture.connector.identity.fingerprint(),
                runtime_generation: 7,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary:
                "Continue the authorized task using the attached state.".into(),
            fallback_reference: "occ1".into(),
        },
    };
    let prepared = crate::core::agent_messages::PreparedNativeHandoff::new(
        authorization,
        publication,
    )
    .expect("fixture prepared handoff");
    let error = fixture
        .parent
        .controller
        .clone()
        .unwrap()
        .send_prepared_native_handoff(prepared)
        .await
        .expect_err("a stale session binding must be refused");
    assert_eq!(error, "Native handoff session changed".to_string());
    let audit = fixture
        .child
        .native
        .as_ref()
        .expect("child permit")
        .audit();
    assert!(
        audit["receiver"]
            .as_array()
            .expect("receiver audit")
            .is_empty(),
        "the refused send must not have admitted anything: {audit}"
    );
}

/// Whether the receiver transcript carries a completed-use latent marker for
/// this message id. A queued or uncertain admission must never have one.
fn receipt_is_latent_completion(fixture: &NativeHandoffFixture, message_id: &str) -> bool {
    let activity = crate::core::occ::activity::from_admitted_messages(
        &fixture.child.session.messages(),
        &fixture.child.session.session_id(),
        "g",
    );
    activity.deliveries.iter().any(|delivery| {
        delivery.id == message_id
            && delivery.kind == crate::core::occ::activity::CommunicationKind::Latent
    })
}
