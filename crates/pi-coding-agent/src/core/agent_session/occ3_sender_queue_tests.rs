//! Sender-queue integration through the production `install_experimental_native_occ3`
//! seam: enqueue-after-completed-handoff, head-ownership of the arm, notice
//! journaling, and the blocked-artifact guarantee. Offline: the fixture
//! connector answers locally, no provider is contacted, no cargo network.
use super::*;
use crate::core::agent_messages::{
    AgentFamilyCatalogEntry, AgentSessionMessageController, AgentSessionMessageEndpoint,
    AgentSessionMessageReceipt, AgentSessionMessageSendInput, AgentFamilyRosterResult,
    AuthorizedNativeHandoff,
};
use crate::core::agent_session::occ3_state::{
    Occ3NativeSession, MAX_NATIVE_CAPTURE_RETRIES,
};
use crate::core::occ::occ3::connector::{
    NativePublication, NativePublicationRequest, NativeRetirement, NativeSessionEnrollment,
    Occ3Connector,
};
use crate::core::occ::occ3::types::{
    Occ3ConnectorCapabilities, Occ3Error, Occ3HandleRef,
    Occ3Method, Occ3ModelIdentity, Occ3TransferMetrics,
};
use pi_ai::types::BoxFuture;
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{
    SourceCaptureIntent, SourceCaptureLease, SourceCaptureLimits, SourceCaptureProfile,
    SOURCE_CAPTURE_SCHEMA, SOURCE_SPAN_POLICY,
};

const METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";

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
        pi_ai::types::API_OPENAI_COMPLETIONS,
        "unit-fixture",
        "http://127.0.0.1:40000/v1",
    )
}

struct QueueFixtureConnector;

#[async_trait::async_trait]
impl Occ3Connector for QueueFixtureConnector {
    fn connector_id(&self) -> &str { "queue-fixture-occ3" }
    fn protocol_version(&self) -> u32 { 1 }
    fn supports_native_inference(&self) -> bool { true }
    async fn enroll_native_session(&self, session_id: &str) -> Result<NativeSessionEnrollment, Occ3Error> {
        Ok(NativeSessionEnrollment {
            session_id: session_id.into(),
            principal_digest: "queue-fixture-principal".into(),
            identity: fixture_identity(),
            runtime_id: "queue-fixture-runtime".into(),
            runtime_generation: 11,
            access_domain: "queue-fixture-domain".into(),
            expires_at_epoch: occ3_state_now() + 600,
            source_profile: SourceCaptureProfile {
                layer_index: 3,
                hidden_dim: 8,
                dtype: "bfloat16".into(),
                layout_digest: "1".repeat(64),
                expected_ranks: 8,
            },
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
                capture_id: format!("queue-capture-{}", uuid::Uuid::new_v4()),
                source_session_id: session_id.into(),
                source_request_id: format!("queue-source-{}", uuid::Uuid::new_v4()),
                turn_nonce: format!("queue-turn-{}", uuid::Uuid::new_v4()),
                model_fingerprint: fixture_identity().fingerprint(),
                runtime_id: "queue-fixture-runtime".into(),
                runtime_generation: 11,
                method: METHOD.into(),
                retained_count_limit,
                max_capture_bytes,
                expires_at_epoch: occ3_state_now() + 600,
                span_policy: SOURCE_SPAN_POLICY.into(),
                capture_mode: String::new(),
                label_anchor_alpha: None,
            },
            SourceCaptureProfile {
                layer_index: 3,
                hidden_dim: 8,
                dtype: "bfloat16".into(),
                layout_digest: "1".repeat(64),
                expected_ranks: 8,
            },
            limits,
        )
        .map_err(|_| Occ3Error::TransferFailed { reason: "queue-fixture".into() })
    }
    async fn publish_native_source(
        &self,
        _request: NativePublicationRequest,
    ) -> Result<NativePublication, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported { connector_id: "queue-fixture-occ3".into() })
    }
    async fn retire_native_source(
        &self,
        _source: &SourceCaptureLease,
    ) -> Result<NativeRetirement, Occ3Error> {
        Ok(NativeRetirement { revoked: true, gpu_freed: true })
    }
    async fn health(&self) -> Result<(), Occ3Error> { Ok(()) }
    async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error> {
        Ok(Occ3ConnectorCapabilities {
            protocol_version: 1,
            connector_version: "queue-fixture".into(),
            model_identity: fixture_identity(),
            methods: vec![Occ3Method::StatebridgePrefix],
            tensor_formats: vec!["queue-fixture".into()],
            max_state_bytes: 1024,
            state_ttl_seconds: 60,
            max_concurrent_handles: 1,
            cache_isolation: "queue-fixture".into(),
            // The production enroll gate resolves support through the
            // registry, which demands the exact native-inference-attachment
            // contract feature (registry.rs `supports`). The fixture must
            // advertise it like every other wiring fixture does.
            features: std::collections::BTreeMap::from([(
                "native_inference_attachment".into(),
                crate::core::occ::occ3::inference::NATIVE_INFERENCE_CONTRACT.into(),
            )]),
        })
    }
    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> {
        Ok(fixture_identity())
    }
    async fn capture_latent(&self, _request: crate::core::occ::occ3::connector::CaptureRequest) -> Result<Occ3HandleRef, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn deliver_latent(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn prepare_inference(
        &self,
        _request: crate::core::occ::occ3::inference::PrepareInferenceRequest,
    ) -> Result<pi_ai::inference_attachment::InferenceStateLease, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn consume_latent(&self, _request: crate::core::occ::occ3::connector::ConsumeRequest) -> Result<Occ3TransferMetrics, Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn acknowledge_delivery(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn release_handle(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(Occ3Error::Disabled)
    }
    async fn shutdown(&self) -> Result<(), Occ3Error> { Ok(()) }
}

fn occ3_state_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

struct PermittingController;

impl AgentSessionMessageController for PermittingController {
    fn authorize_native_handoff(&self, role: String, name: String)
        -> BoxFuture<Result<AuthorizedNativeHandoff, String>> {
        Box::pin(async move {
            AuthorizedNativeHandoff::from_catalog(
                &AgentFamilyCatalogEntry { id: "unit-session".into(), depth: 0., ..Default::default() },
                &AgentFamilyCatalogEntry {
                    id: name.clone(),
                    depth: 1.,
                    parent_session_id: Some("unit-session".into()),
                    ..Default::default()
                },
                AgentSessionMessageEndpoint {
                    session_id: "unit-session".into(),
                    active_session_id: "unit-session".into(),
                    ..Default::default()
                },
                AgentSessionMessageEndpoint {
                    session_id: name.clone(),
                    active_session_id: name.clone(),
                    ..Default::default()
                },
                &role,
            )
        })
    }
    fn roster(&self) -> BoxFuture<Result<AgentFamilyRosterResult, String>> {
        Box::pin(async { Err("unused".into()) })
    }
    fn await_pending_child_publication(&self, _: String) -> BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn send_agent_message(
        &self,
        _: AgentSessionMessageSendInput,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        Box::pin(async { Err("ordinary messages must not be used".into()) })
    }
}

async fn enrolled_session() -> Arc<Occ3NativeSession> {
    Occ3NativeSession::enroll(
        "unit-session",
        &fixture_model(),
        Arc::new(QueueFixtureConnector),
    )
    .await
    .expect("queue fixture enrollment")
}

/// Writes one queue artifact for a test-owned session file.
///
/// The derived artifact path walks to `dirname(dirname(session_file))` +
/// `session-artifacts/<basename>`; tests therefore nest the session file
/// one level inside their tempdir so the artifact root is test-private
/// (a shared root made the whole module order-dependent: every persist
/// and manual write landed on one global file). The parent directories
/// do not exist on the first write, and a path-shape regression would
/// silently reintroduce cross-test sharing, so both are handled - and
/// guarded - here once for every test that stages an artifact by hand.
fn write_queue_artifact(session_file: &str, artifact: &serde_json::Value) {
    let path = crate::core::agent_session::occ3_sender_queue::sender_queue_path(session_file);
    let root = std::path::Path::new(session_file)
        .parent()
        .and_then(|parent| parent.parent())
        .expect("the test session file must sit one level inside its tempdir");
    assert!(
        path.starts_with(root),
        "queue artifact {path:?} escapes its test directory {root:?}"
    );
    std::fs::create_dir_all(path.parent().expect("artifact parent"))
        .expect("create the test artifact parent");
    std::fs::write(&path, artifact.to_string()).expect("write the test queue artifact");
}

/// The queued (pending) head owns the next arm: a different-receiver
/// call while the head waits QUEUES BEHIND it (F4) - it never arms and
/// never touches the capture permit; the head keeps its arm right.
#[tokio::test]
async fn queued_head_owns_the_next_arm() {
    let session = enrolled_session().await;
    let notice = session
        .sender_queue_enqueue("child", "head-receiver")
        .expect("enqueue head");
    session.record_queue_notice(notice);
    let tool = session.handoff_tool(Arc::new(PermittingController));
    let result = (tool.execute)(
        "tool-1".into(),
        serde_json::json!({"receiver_role":"child","receiver_name":"other-receiver"}),
        None,
        None,
    )
    .await
    .expect("a different receiver queues behind the pending head");
    assert_eq!(result.details["status"], "queued");
    assert_eq!(result.details["queue_position"], 2);
    // The pending head is untouched and still owns the next arm: it
    // stays the pending head, and the different-receiver call never
    // armed the source slot (no lease, no capture).
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].receiver_name, "head-receiver");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending
    );
    let audit = session.audit();
    assert!(audit["source"].is_null(), "the call must not arm or capture");
}

/// The persisted artifact is preserved byte-for-byte when its schema is
/// unknown, the queue opens blocked, and enqueue refuses.
#[tokio::test]
async fn unknown_schema_blocks_and_preserves_artifact() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("occ3-sender-queue.json");
    let contents = br#"{"schema":"optimus.occ3-sender-queue.v999","entries":[]}"#;
    std::fs::write(&path, contents).unwrap();
    session
        .bind_chat_preferences(Some(dir.path().join("nested/session.jsonl").to_string_lossy().into_owned()))
        .expect("bind preferences");
    // session.jsonl does not exist; the artifact path derives from its
    // basename, so point the loader at the artifact we wrote.
    let artifact = crate::core::agent_session::occ3_sender_queue::sender_queue_path(
        &dir.path().join("nested/session.jsonl").to_string_lossy(),
    );
    std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    std::fs::write(&artifact, contents).unwrap();
    let blocked_notice = session
        .load_sender_queue(Some(&dir.path().join("nested/session.jsonl").to_string_lossy()))
        .expect("load returns a notice, not an error");
    assert!(blocked_notice.is_some(), "blocked load surfaces a notice");
    assert_eq!(std::fs::read(&artifact).unwrap(), contents.to_vec(), "file preserved");
    // Enqueue refuses while blocked.
    assert!(session.sender_queue_enqueue("child", "x").is_err());
    let (entries, blocked) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 0);
    assert!(blocked.is_some());
    assert_eq!(blocked.unwrap().as_str(), "unsupported_schema");
}

/// F1 restart split, ARMED case: pre-publication failure with budget
/// remaining => PENDING for a fresh explicit arm; history recorded.
#[tokio::test]
async fn restart_armed_entry_fresh_arms_within_budget() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    session.sender_queue_enqueue("child", "pre-publication").expect("enqueue");
    session.sender_queue_mark_armed(Some("agentmsg_armed"), None).expect("mark armed");
    session.sender_queue_enqueue("child", "waiting").expect("enqueue tail");
    session.load_sender_queue(Some(&session_file)).expect("reload");
    let (entries, blocked) = session.sender_queue_snapshot().expect("snapshot");
    assert!(blocked.is_none());
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending
    );
    assert_eq!(entries[0].reason.as_deref(), Some("restart_pre_publication"));
    assert_eq!(entries[0].attempts, 1);
    assert!(entries[0].message_ids.iter().any(|(label, id)| label == "restart_ambiguous" && id == "agentmsg_armed"));
    assert_eq!(entries[1].state, crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending);
}

/// F1 restart split, PUBLISHED/ADMITTED case: a possibly-consumed
/// publication resolves to UNCERTAIN - visible, explicit, never
/// auto-recaptured, auto-fallen-back or TTL-expired.
#[tokio::test]
async fn restart_admitted_entry_becomes_uncertain() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    session.sender_queue_enqueue("child", "admitted-one").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_admitted",
        )
        .expect("progressed");
    session.sender_queue_enqueue("child", "waiting").expect("enqueue tail");
    session.load_sender_queue(Some(&session_file)).expect("reload");
    let (entries, blocked) = session.sender_queue_snapshot().expect("snapshot");
    assert!(blocked.is_none());
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Uncertain
    );
    assert_eq!(entries[0].reason.as_deref(), Some("restart_uncertain_publication"));
    assert_eq!(entries[0].attempts, 1);
    // TTL sweep skips uncertain entries even past expiry.
    let notices = session.sender_queue_sweep_expired().expect("sweep");
    assert!(notices.is_empty());
    // Capture-failure journaling leaves the uncertain entry untouched
    // (it is not armed).
    assert!(session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("journal call")
        .is_none());
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot after");
    assert_eq!(entries.len(), 2, "uncertain entry untouched");
}

/// Restart expiry is never a silent drop: an entry that expired while the
/// session was down is journaled as a terminal `expired` notice at load
/// (the same drain the live TTL sweep uses), so the sender and the
/// parent-side roster reading the durable `occ3_queue_notice` records
/// learn of the loss on resume. Goes through the production
/// `load_sender_queue` path with an actual expired artifact on disk.
#[tokio::test]
async fn restart_expired_pending_surfaces_durable_expired_notice() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // One pending entry whose TTL lapsed long before the restart.
    let artifact = serde_json::json!({
        "schema": "optimus.occ3-sender-queue.v1",
        "entries": [{
            "queueId": "occ3q_disk_expired",
            "receiverRole": "child",
            "receiverName": "gone",
            "enqueuedAtEpoch": 1,
            "expiresAtEpoch": 2,
            "state": "pending",
            "attempts": 0,
        }],
    });
    write_queue_artifact(&session_file, &artifact);
    // No fallback aggregate (nothing survived the restart), but the expiry
    // IS journaled for the durable drain.
    let aggregate = session.load_sender_queue(Some(&session_file)).expect("load");
    assert!(aggregate.is_none());
    let notices = session.take_pending_queue_notices();
    assert_eq!(notices.len(), 1, "the expired entry is reported, not silently dropped");
    assert_eq!(notices[0].queue_id, "occ3q_disk_expired");
    assert_eq!(notices[0].state, "expired");
    assert_eq!(notices[0].reason.as_deref(), Some("ttl_expired"));
    // The durable record payload (the listener writes `details()` verbatim).
    let details = notices[0].details();
    assert_eq!(details["queueId"], "occ3q_disk_expired");
    assert_eq!(details["state"], "expired");
    assert_eq!(details["reason"], "ttl_expired");
    assert!(notices[0].refusal.is_none());
    assert!(details.get("refused").is_none());
    // The queue itself no longer carries the dead entry.
    let (entries, blocked) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty());
    assert!(blocked.is_none());
}

/// REGRESSION (atomic transition+outbox): a TTL sweep's expired notice
/// must be in the persisted artifact immediately after the sweep - a
/// crash between the sweep and the drain loses nothing. Asserts on the
/// artifact BYTES, not in-memory state.
#[tokio::test]
async fn sweep_expired_notice_is_durable_before_any_drain() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // One pending entry that is already expired.
    let now = crate::core::agent_session::occ3_sender_queue::now_epoch();
    let artifact = serde_json::json!({
        "schema": "optimus.occ3-sender-queue.v1",
        "entries": [{
            "queueId": "occ3q_sweep_crash",
            "receiverRole": "child",
            "receiverName": "gone",
            "enqueuedAtEpoch": now,
            "expiresAtEpoch": now - 1,
            "state": "pending",
            "attempts": 0,
        }],
    });
    write_queue_artifact(&session_file, &artifact);
    // LOAD is itself the restart expiry surface: the loader drops the
    // expired pending entry and journals its terminal `expired` notice
    // into the loaded queue's durable outbox + drain journal BEFORE the
    // load is stored (one atomic persist) - no live sweep is involved.
    // (The live-sweep crash boundary has its own regression below.)
    session.load_sender_queue(Some(&session_file)).expect("load");
    // The load journaled the expiry without any drain.
    let load_expired = session.take_pending_queue_notices();
    assert_eq!(
        load_expired.len(),
        1,
        "load journals the expired entry's terminal notice before any drain"
    );
    assert_eq!(load_expired[0].state, "expired");
    assert_eq!(load_expired[0].queue_id, "occ3q_sweep_crash");
    // The live sweep on the now-clean queue is a legitimate no-op.
    let expired = session.sender_queue_sweep_expired().expect("sweep");
    assert_eq!(expired.len(), 0);
    // The persisted bytes carry the expired notice in the outbox.
    let bytes = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    assert!(
        bytes.contains("\"state\":\"expired\"") && bytes.contains("ttl_expired"),
        "the expiry notice is durable before any drain"
    );
    // A fresh session reload replays it from the artifact (same stable id).
    let restarted = enrolled_session().await;
    restarted.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    restarted.load_sender_queue(Some(&session_file)).expect("reload");
    let replay = restarted.take_pending_queue_notices();
    assert_eq!(replay.len(), 1, "the crash lost nothing: the notice replays");
    assert_eq!(replay[0].state, "expired");
}

/// LIVE-sweep crash boundary: an entry that expires AFTER the load is
/// swept by the listener/timer path, the sweep's journal is persisted
/// atomically with the queue rewrite, and a crash before ANY drain
/// loses nothing - the notice replays from the artifact on restart.
/// No sleeps, no global clock mutation: the artifact carries an
/// expiry the current clock already satisfies.
#[tokio::test]
async fn live_sweep_expiry_journal_survives_a_crash_before_drain() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // One UNEXPIRED pending entry: the load keeps it as live work.
    let now = crate::core::agent_session::occ3_sender_queue::now_epoch();
    let artifact = serde_json::json!({
        "schema": "optimus.occ3-sender-queue.v1",
        "entries": [{
            "queueId": "occ3q_live_sweep",
            "receiverRole": "child",
            "receiverName": "late",
            "enqueuedAtEpoch": now,
            "expiresAtEpoch": now + 4,
            "state": "pending",
            "attempts": 0,
        }],
    });
    write_queue_artifact(&session_file, &artifact);
    session.load_sender_queue(Some(&session_file)).expect("load");
    // Nothing expires at load time: no notices yet.
    assert!(
        session.take_pending_queue_notices().is_empty(),
        "unexpired entry loads silently"
    );
    // Advance the sweep clock beyond the entry's expiry through the
    // existing API - the entry becomes expired relative to the sweep
    // instant, exactly as the 30s timer would find it.
    let expired = session
        .sender_queue_sweep_expired_at(now + 5)
        .expect("live sweep");
    assert_eq!(expired.len(), 1, "the live sweep expires the entry");
    assert_eq!(expired[0].state, "expired");
    assert_eq!(expired[0].queue_id, "occ3q_live_sweep");
    // Crash boundary: NO drain, NO ack - the persisted bytes alone
    // must carry the expired notice in the outbox.
    let bytes = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    assert!(
        bytes.contains("\"state\":\"expired\"") && bytes.contains("ttl_expired"),
        "the live sweep's expired notice is durable before any drain"
    );
    // Restart: the notice replays from the artifact (same stable id).
    let restarted = enrolled_session().await;
    restarted.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    restarted.load_sender_queue(Some(&session_file)).expect("reload");
    let replay = restarted.take_pending_queue_notices();
    assert_eq!(replay.len(), 1, "the crash lost nothing: the notice replays");
    assert_eq!(replay[0].state, "expired");
    assert_eq!(replay[0].queue_id, "occ3q_live_sweep");
}

/// REGRESSION (load clobber): expiry notices recorded at load must
/// survive a SUBSEQUENT transition's persist (the old code recorded
/// them into the previous in-memory queue and the next persist of the
/// loaded queue clobbered them). Asserts on the artifact bytes.
#[tokio::test]
async fn load_expiry_notices_survive_subsequent_persists() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    let now = crate::core::agent_session::occ3_sender_queue::now_epoch();
    let artifact = serde_json::json!({
        "schema": "optimus.occ3-sender-queue.v1",
        "entries": [{
            "queueId": "occ3q_load_clobber",
            "receiverRole": "child",
            "receiverName": "gone",
            "enqueuedAtEpoch": now,
            "expiresAtEpoch": now - 1,
            "state": "pending",
            "attempts": 0,
        }],
    });
    write_queue_artifact(&session_file, &artifact);
    session.load_sender_queue(Some(&session_file)).expect("load");
    let expiry_notices = session.take_pending_queue_notices();
    assert_eq!(expiry_notices.len(), 1);
    let notice_id = expiry_notices[0].notice_id.clone();
    // A later transition persists the loaded queue again; the expiry
    // record must still be in the artifact bytes.
    session.sender_queue_enqueue("child", "after-load").expect("enqueue");
    let bytes = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    assert!(bytes.contains(&notice_id), "the expiry record survived the later persist");
    // And a reload still replays it (both legs unacked).
    let restarted = enrolled_session().await;
    restarted.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    restarted.load_sender_queue(Some(&session_file)).expect("reload");
    assert!(
        restarted
            .take_pending_queue_notices()
            .iter()
            .any(|record| record.notice_id == notice_id),
        "the expiry record replays after the later persist and reload"
    );
}

/// F1 (same-process retry): a drained record whose parent leg fails
/// (default controller refuses) is re-selected from the durable outbox
/// by `requeue_unacked_queue_notices` - the next drain cycle retries it
/// WITHOUT a process restart. A subsequent ack clears it.
#[tokio::test]
async fn failed_parent_leg_is_retried_in_the_same_process() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    session.sender_queue_enqueue("child", "retry-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure")
        .expect("fallback notice");
    // Drain 1: the worklist yields the journaled records.
    let first = session.take_pending_queue_notices();
    assert!(first.len() >= 1);
    // The transcript leg acks; the parent leg FAILS (refuses).
    for record in &first {
        session.sender_queue_notice_ack_transcript(&record.notice_id);
    }
    let outbox = session.sender_queue_outbox_snapshot();
    assert!(
        outbox.iter().all(|record| record["parentDone"] == false),
        "parent leg unacked"
    );
    // The drain cycle's requeue step (the exact call the listener/timer
    // make after delivering): the unacked records re-enter the worklist.
    session.requeue_unacked_queue_notices();
    let requeued = session.take_pending_queue_notices();
    assert_eq!(
        requeued.len(),
        outbox.len(),
        "every unacked record is re-selected in the same process"
    );
    // Recovery in the same process: the parent leg now accepts.
    for record in &requeued {
        session.sender_queue_notice_ack_parent(&record.notice_id);
    }
    assert!(session.sender_queue_outbox_snapshot().is_empty());
    // Nothing left to requeue.
    session.requeue_unacked_queue_notices();
    assert!(session.take_pending_queue_notices().is_empty());
    // No duplicate requeue either: acked records never return.
    session.requeue_unacked_queue_notices();
    assert!(session.take_pending_queue_notices().is_empty());
}

/// F2 (persist failure surfaced, never silent success): an enqueue whose
/// artifact persist fails returns Err (no successful-return transition
/// with unsaved state), and the journaled record still sits in the
/// worklist + durable outbox for the retry cycle.
#[tokio::test]
async fn persist_failure_surfaces_and_keeps_the_record_retryable() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    // A FILE where the queue artifact's parent directory should be:
    // create_dir_all fails, so the atomic persist fails deterministically.
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"not a directory").unwrap();
    let session_file = blocker.join("sub").join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // The enqueue transition applies in memory, journals the notice, and
    // the persist FAILS -> Err surfaced (never a silent success).
    let error = session
        .sender_queue_enqueue("child", "unsaved")
        .expect_err("persist failure surfaced");
    assert!(!error.is_empty());
    // The record is still retryable: worklist + outbox hold it.
    let worklist = session.take_pending_queue_notices();
    assert!(
        worklist.iter().any(|record| record.state == "pending"),
        "the notice is journaled despite the persist failure"
    );
    let outbox = session.sender_queue_outbox_snapshot();
    assert!(
        outbox
            .iter()
            .any(|record| record.get("state").and_then(serde_json::Value::as_str) == Some("pending"))
    );
    // And the requeue keeps it selectable for the next drain cycle.
    session.requeue_unacked_queue_notices();
    assert!(
        session
            .take_pending_queue_notices()
            .iter()
            .any(|record| record.state == "pending")
    );
}

/// REGRESSION (double recording): a transition records its notice into
/// the durable outbox EXACTLY once (the wrapper journals atomically; the
/// old code journaled again at the caller).
#[tokio::test]
async fn transitions_record_outbox_entries_exactly_once() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // Enqueue -> armed -> terminal capture failure: three transitions.
    session.sender_queue_enqueue("child", "once-only").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure");
    // Exactly one outbox record per journaled notice - no duplicates
    // from the (now removed) caller-side journal.
    let outbox = session.sender_queue_outbox_snapshot();
    assert_eq!(outbox.len(), 3, "enqueue + armed + fallback: one record each");
    let ids: std::collections::HashSet<&str> = outbox
        .iter()
        .filter_map(|record| record["noticeId"].as_str())
        .collect();
    assert_eq!(ids.len(), 3, "all records carry distinct stable ids");
    // The persisted bytes agree.
    let bytes = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    for id in ids {
        assert!(bytes.contains(id));
    }
}

/// The durable outbox: a recorded notice persists atomically with the
/// queue, survives restart, replays (same stable noticeId) until BOTH
/// deliveries are acknowledged, and is removed only after the last ack.
/// Simulates the crash boundaries: queue saved + transcript written but
/// parent not yet acked (record replays, no duplicate transcript work),
/// then parent ack (record gone, no further replay).
#[tokio::test]
async fn durable_outbox_survives_restart_and_dedups_by_notice_id() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // Three transitions journal three records atomically; the fallback
    // demotion (terminal capture failure) is the subject of this test.
    session.sender_queue_enqueue("child", "outbox-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure")
        .expect("fallback notice");
    // Crash boundary 1: the queue (and its outbox) were persisted by the
    // wrappers themselves; nothing has been drained or acknowledged.
    let first = session.take_pending_queue_notices();
    assert_eq!(first.len(), 3, "one record per transition, journaled by the wrapper");
    let fallback = first
        .iter()
        .find(|record| record.state == "fallback_pending")
        .expect("the fallback demotion was journaled");
    let notice_id = fallback.notice_id.clone();
    assert!(notice_id.starts_with("occ3n_"), "stable durable id");
    let outbox = session.sender_queue_outbox_snapshot();
    assert_eq!(outbox.len(), 3);
    let subject = outbox
        .iter()
        .find(|record| record["noticeId"].as_str() == Some(notice_id.as_str()))
        .expect("the fallback record is in the outbox");
    assert_eq!(subject["transcriptDone"], false);
    // The durable artifact carries the outbox record.
    let artifact = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    assert!(artifact.contains(&notice_id), "the outbox record is persisted with the queue");
    // The sender transcript delivery is acknowledged (durable write done).
    session.sender_queue_notice_ack_transcript(&notice_id);
    // Crash boundary 2: transcript written and acked, parent NOT acked.
    // Restart: the pending record replays with the SAME noticeId.
    let restarted = enrolled_session().await;
    restarted.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    let aggregate = restarted.load_sender_queue(Some(&session_file)).expect("reload");
    let replay = restarted.take_pending_queue_notices();
    // The restart surface for a fallback_pending entry is the designed
    // AGGREGATE notice: the model is prompted for the ordinary
    // re-emission (occ3_state.rs load path).
    let aggregate = aggregate.expect("fallback_pending surfaces its aggregate notice");
    assert_eq!(aggregate.state, "fallback_pending");
    // Replay journal: the original unacked record (same stable id) AND
    // the load-journaled aggregate - consumers dedup by noticeId.
    assert_eq!(
        replay.len(),
        4,
        "the unacked records replay alongside the load aggregate (3 + 1)"
    );
    assert!(
        replay.iter().any(|record| record.notice_id == notice_id),
        "same stable id for consumer dedup"
    );
    let outbox = restarted.sender_queue_outbox_snapshot();
    let subject = outbox
        .iter()
        .find(|record| record.get("noticeId").and_then(serde_json::Value::as_str) == Some(&notice_id))
        .expect("the original record survived the restart");
    assert_eq!(subject["transcriptDone"], true, "the acked leg is not redone");
    assert_eq!(subject["parentDone"], false);
    // Parent delivery accepted: the ORIGINAL record is removed; the
    // aggregate stays until its own legs are acknowledged.
    restarted.sender_queue_notice_ack_parent(&notice_id);
    let remaining = restarted.sender_queue_outbox_snapshot();
    assert!(
        !remaining
            .iter()
            .any(|record| record.get("noticeId").and_then(serde_json::Value::as_str) == Some(&notice_id)),
        "the fully acked record leaves the outbox"
    );
    let artifact = std::fs::read_to_string(
        crate::core::agent_session::occ3_sender_queue::sender_queue_path(&session_file),
    )
    .unwrap();
    assert!(!artifact.contains(&notice_id), "fully delivered records leave the artifact");
    // Crash boundary 3: fully acked before restart -> no replay of the
    // original record. (The aggregate is a separate, still-unacked
    // record by design - its own delivery legs govern it.)
    let settled = enrolled_session().await;
    settled.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    settled.load_sender_queue(Some(&session_file)).expect("reload 2");
    assert!(
        settled
            .take_pending_queue_notices()
            .iter()
            .all(|record| record.notice_id != notice_id),
        "no replay after both acks"
    );
    assert!(
        !settled
            .sender_queue_outbox_snapshot()
            .iter()
            .any(|record| record.get("noticeId").and_then(serde_json::Value::as_str) == Some(&notice_id))
    );
}

/// An undeliverable parent leg (a controller without a parent surface
/// refuses by default) keeps the record pending, bounded, and visibly
/// overflow-flagged - never silently dropped and never silently cleared.
#[tokio::test]
async fn undelivered_parent_leg_stays_pending_bounded_and_visible() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // Refusals journal durable records naturally: fill the queue, then
    // keep attempting enqueues past the outbox cap.
    let mut accepted = 0usize;
    loop {
        match session.sender_queue_enqueue("child", "filler") {
            Ok(_) => accepted += 1,
            Err(_) => break,
        }
        assert!(accepted <= 64, "the entry ceiling must be bounded");
    }
    assert!(accepted >= 1);
    let cap = crate::core::agent_session::occ3_sender_queue::OCC3_SENDER_QUEUE_OUTBOX_CAP;
    for _ in 0..(cap + 2) {
        assert_eq!(session.sender_queue_enqueue("child", "refused").unwrap_err(), "queue_full");
    }
    // The outbox is capped, and the overflow flag is visible in the
    // snapshot details (the F5 visible-fail state).
    assert_eq!(session.sender_queue_outbox_snapshot().len(), cap);
    let details = session.sender_queue_snapshot_details().expect("details");
    assert_eq!(details["noticeOverflow"], true);
    // The pending records stay: draining the journal yields the capped
    // records; none of them is silently dropped.
    let drained = session.take_pending_queue_notices();
    assert!(drained.len() >= accepted, "accepted enqueues plus refusals were journaled (bounded by the journal cap)");
    assert!(drained.iter().all(|record| record.parent_done == false));
}

/// A refused enqueue is reported durably: one content-free queue-level
/// refusal notice is journaled (`queue_full` when the ceiling is hit,
/// `queue_blocked` while the artifact is blocked), so the parent-side
/// roster learns the handoff was refused and why. The refusal record
/// carries an empty queue id, so it can never project as a queued entry.
#[tokio::test]
async fn refused_enqueue_journals_a_queue_level_notice() {
    let session = enrolled_session().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    // Fill the queue to its effective ceiling: the next enqueue is refused.
    let mut enqueued = 0usize;
    loop {
        match session.sender_queue_enqueue("child", "filler") {
            Ok(_) => enqueued += 1,
            Err(reason) => {
                assert_eq!(reason, "queue_full");
                break;
            }
        }
        assert!(enqueued <= 64, "the entry ceiling must be bounded");
    }
    assert!(enqueued >= 1);
    // The wrapper journals BOTH accepted enqueues and the refusal; only
    // the refusal record is queue-level.
    let refusal: Vec<_> = session
        .take_pending_queue_notices()
        .into_iter()
        .filter(|record| record.state == "refused")
        .collect();
    assert_eq!(refusal.len(), 1);
    assert_eq!(refusal[0].refusal.as_deref(), Some("queue_full"));
    // The durable payload: whitelisted `refused` value; the empty queue
    // id means the record never projects as a queue entry on the read
    // side (bounded_text rejects empty ids).
    let details = refusal[0].details();
    assert_eq!(details["refused"], "queue_full");
    assert_eq!(details["queueId"], "");
    // Blocked path: an unknown-schema artifact blocks the queue; the next
    // enqueue is refused queue_blocked with the same record shape.
    write_queue_artifact(
        &session_file,
        &serde_json::json!({"schema": "optimus.occ3-sender-queue.v99", "entries": []}),
    );
    session.load_sender_queue(Some(&session_file)).expect("reload");
    assert_eq!(
        session.sender_queue_enqueue("child", "blocked").unwrap_err(),
        "queue_blocked"
    );
    let blocked_refusal: Vec<_> = session
        .take_pending_queue_notices()
        .into_iter()
        .filter(|record| record.state == "refused")
        .collect();
    assert_eq!(blocked_refusal.len(), 1);
    assert_eq!(blocked_refusal[0].refusal.as_deref(), Some("queue_blocked"));
    assert_eq!(blocked_refusal[0].details()["refused"], "queue_blocked");
    assert_eq!(blocked_refusal[0].details()["queueId"], "");
}

#[tokio::test]
async fn delivery_events_advance_the_queue() {
    let session = enrolled_session().await;
    // CYCLE 1 (proven completion): only a PENDING HEAD may arm, so the
    // cycle runs enqueue -> arm -> admit -> consume(+digest) and the
    // entry pops. Every transition is asserted so the setup can never
    // silently no-op.
    session.sender_queue_enqueue("child", "first").expect("enqueue");
    let armed = session
        .sender_queue_mark_armed(None, None)
        .expect("armed transition");
    assert!(armed.is_some(), "the head actually armed");
    let progressed = session
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_first",
        )
        .expect("progressed transition");
    assert!(progressed.is_some(), "the head actually admitted");
    // Proven completion (consumed + digest) pops as completed.
    let notices = session
        .sender_queue_record_delivery_event("agentmsg_first", true, Some("sha256"))
        .expect("consumed event");
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].state, "completed");
    assert_eq!(notices[0].message_id.as_deref(), Some("agentmsg_first"));
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty(), "the completed entry popped");
    // CYCLE 2 (digestless): a fresh head is armed and admitted, then a
    // consumed report WITHOUT a receipt digest cannot prove completion:
    // the entry resolves to `uncertain` - visible and explicit, never
    // auto-advanced (F5).
    session.sender_queue_enqueue("child", "digestless").expect("enqueue");
    let armed = session
        .sender_queue_mark_armed(None, None)
        .expect("armed transition");
    assert!(armed.is_some(), "the fresh head actually armed");
    let progressed = session
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_digestless",
        )
        .expect("progressed transition");
    assert!(progressed.is_some(), "the fresh head actually admitted");
    let notices = session
        .sender_queue_record_delivery_event("agentmsg_digestless", true, None)
        .expect("uncertain event");
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].state, "uncertain");
    // The uncertain entry STAYS: visible, never auto-advanced.
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].receiver_name, "digestless");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Uncertain
    );
}

/// Capture-failure journaling never eats the PR235 budget from the queue
/// side: non-terminal failures return the entry to pending.
#[tokio::test]
async fn capture_failure_returns_entry_to_pending_without_touching_budget() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "retry-me").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(false, "watchdog_abort")
        .expect("capture failure journal")
        .expect("notice present");
    assert_eq!(notice.state, "pending");
    assert_eq!(notice.attempts, 1);
    assert_eq!(session.failed_capture_attempts(), 0, "queue never consumes the PR235 budget");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries[0].state, crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending);
}

/// Terminal native death demotes the armed entry to fallback and pops it.
#[tokio::test]
async fn terminal_capture_failure_falls_back() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "doomed").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure")
        .expect("notice present");
    assert_eq!(notice.state, "fallback_pending");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1, "entry stays for the ordinary re-emission");
    let popped = session
        .sender_queue_acknowledge_fallback(&notice.queue_id, notice.dispatch_token.as_deref())
        .expect("acknowledge")
        .expect("terminal fallback notice");
    assert_eq!(popped.state, "fallback");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot after ack");
    assert!(entries.is_empty());
}

/// The audit projection includes the queue without any handle-like keys.
#[tokio::test]
async fn audit_projects_the_sender_queue_content_free() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "audited").expect("enqueue");
    let audit = session.audit();
    assert!(audit["sender_queue"].is_array());
    let text = audit.to_string();
    assert!(!text.contains("source_handle"));
    assert!(text.contains("audited"));
}

/// MAX_NATIVE_CAPTURE_RETRIES is unchanged by the queue work (PR235 pin).
#[test]
fn pr235_retry_budget_constant_untouched() {
    assert_eq!(MAX_NATIVE_CAPTURE_RETRIES, 3);
}

/// The snapshot-form details carry the full queue including blocked state.
#[tokio::test]
async fn snapshot_details_carry_the_full_queue() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "snap-a").expect("enqueue a");
    session.sender_queue_enqueue("child", "snap-b").expect("enqueue b");
    let details = session.sender_queue_snapshot_details().expect("snapshot details");
    assert_eq!(details["snapshot"], true);
    let entries = details["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["queueId"].as_str().unwrap().starts_with("occ3q_"), true);
    assert_eq!(entries[0]["receiver"], "child:snap-a");
    assert_eq!(entries[0]["state"], "pending");
    assert_eq!(entries[0]["position"], 1);
    assert_eq!(entries[1]["position"], 2);
    assert!(entries[0].get("messageId").is_none(), "message id absent until minted");
}

/// F5 back-to-back scenario through the production tool path: the first
/// receiver arms (busy head), a second call for a DIFFERENT receiver queues
/// at position 2 instead of erroring, and the armed capture is untouched.
#[tokio::test]
async fn busy_head_new_receiver_queues_at_position_two() {
    let session = enrolled_session().await;
    // Head A: armed through the real tool (first call arms).
    let tool = session.handoff_tool(Arc::new(PermittingController));
    // Bounded: a regression that re-parks on the capture permit (or any
    // await without a waker) fails here instead of hanging the suite.
    let work = async {
        let result = (tool.execute)(
            "call-a1".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"alpha"}),
            None,
            None,
        )
        .await
        .expect("first call arms");
        assert_eq!(result.details["status"], "capture_armed");
        // F2 regression: the initial request created its stable head
        // entry BEFORE the arm - exactly one entry, alpha, ARMED, with
        // the authorized message id. (Pre-fix the arm left the queue
        // empty while holding the process permit, so the next call
        // deadlocked on the acquire.)
        let (entries, _) = session.sender_queue_snapshot().expect("snapshot after arm");
        assert_eq!(entries.len(), 1, "initial request enqueued its head entry");
        assert_eq!(entries[0].receiver_name, "alpha");
        assert_eq!(
            entries[0].state,
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Armed
        );
        assert!(entries[0].message_id.is_some(), "armed entry carries the message id");
        // Second call for a DIFFERENT receiver while the head is armed: queue.
        let result = (tool.execute)(
            "call-b1".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"beta"}),
            None,
            None,
        )
        .await
        .expect("busy head must enqueue a new receiver, not error");
        assert_eq!(result.details["status"], "queued");
        assert_eq!(result.details["queue_position"], 2);
        // The armed capture is preserved: the head is still alpha/armed.
        let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].receiver_name, "alpha");
        assert_eq!(
            entries[0].state,
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Armed
        );
        assert_eq!(entries[1].receiver_name, "beta");
        assert_eq!(
            entries[1].state,
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending
        );
        // A third receiver queues at position 3.
        let result = (tool.execute)(
            "call-c1".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"gamma"}),
            None,
            None,
        )
        .await
        .expect("busy head must enqueue the third receiver");
        assert_eq!(result.details["queue_position"], 3);
        let (entries, _) = session.sender_queue_snapshot().expect("final snapshot");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2].receiver_name, "gamma");
    };
    tokio::time::timeout(std::time::Duration::from_secs(10), work)
        .await
        .expect("bounded: no permit-park regression");
}

/// F2 regression, dedicated: the INITIAL request on an empty queue
/// creates its stable head entry BEFORE the arm, so the arm transitions
/// a real entry (Armed with the authorized message id) instead of
/// leaving the queue empty while the process permit is held.
#[tokio::test]
async fn initial_request_creates_the_head_entry_before_the_arm() {
    let session = enrolled_session().await;
    let tool = session.handoff_tool(Arc::new(PermittingController));
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        (tool.execute)(
            "call-first".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"first-receiver"}),
            None,
            None,
        ),
    )
    .await
    .expect("bounded: a fresh arm must not park on the permit")
    .expect("first call arms");
    assert_eq!(result.details["status"], "capture_armed");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1, "the initial request enqueued its entry");
    assert_eq!(entries[0].receiver_name, "first-receiver");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Armed
    );
    assert!(entries[0].message_id.is_some(), "armed entry carries the message id");
}

/// F3 regression: a same-receiver COMPLETING call must NOT reacquire the
/// process capture permit - the active capture already holds it (default
/// pool = 1; reacquiring deadlocks the two-step flow). Both completing
/// stages stay bounded: the double-arm is refused before any lease
/// exists, and with the lease minted through the REAL resolve hook the
/// tool-binding validation refuses. Never a permit park; the armed
/// entry survives both refusals.
#[tokio::test]
async fn same_receiver_completion_reuses_the_active_permit() {
    let session = enrolled_session().await;
    let tool = session.handoff_tool(Arc::new(PermittingController));
    let first = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        (tool.execute)(
            "call-arm".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"completion-target"}),
            None,
            None,
        ),
    )
    .await
    .expect("bounded arm")
    .expect("first call arms");
    assert_eq!(first.details["status"], "capture_armed");
    // (a) Completing call with the armed budget but no captured lease:
    // a prompt refusal (the double-arm is rejected), never a permit park.
    let second = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        (tool.execute)(
            "call-complete".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"completion-target"}),
            None,
            None,
        ),
    )
    .await
    .expect("bounded: the completing call must not park on the permit it already holds")
    .expect_err("no captured lease yet: the double-arm is refused");
    let text = format!("{second}");
    assert!(text.contains("replay is forbidden"), "named failure: {text}");
    // The armed entry survives the refused completion.
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].receiver_name, "completion-target");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Armed
    );
    // (b) Mint the lease through the REAL resolve hook (budget -> lease),
    // then complete again: still bounded, still no park - the uncaptured
    // lease fails the tool-binding validation inside the publish path.
    let hooks = session.source_hooks();
    let request = pi_agent_core::types::InferenceStateRequest {
        model: fixture_model(),
        context: Default::default(),
        messages: Vec::new(),
        newly_supplied_messages: Vec::new(),
        signal: None,
    };
    let lease = (hooks.resolve)(request)
        .await
        .expect("resolve hook")
        .expect("lease minted from the armed budget");
    assert!(matches!(
        lease.state(),
        pi_ai::source_capture::SourceCaptureState::Prepared
    ));
    let third = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        (tool.execute)(
            "call-publish".into(),
            serde_json::json!({"receiver_role":"child","receiver_name":"completion-target"}),
            None,
            None,
        ),
    )
    .await
    .expect("bounded: completing with a lease must not park on the permit")
    .expect_err("uncaptured lease: the tool-binding validation refuses");
    let text = format!("{third}");
    assert!(text.contains("occ3_tool_validate_tool"), "named refusal: {text}");
    // The armed entry is still intact after both refused completions.
    let (entries, _) = session.sender_queue_snapshot().expect("final snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Armed
    );
}

/// F4: a genuinely new request while a PENDING head waits never errors -
/// it enqueues behind the head and reports its position.
#[tokio::test]
async fn pending_head_new_request_enqueues_behind() {
    let session = enrolled_session().await;
    let notice = session
        .sender_queue_enqueue("child", "head-receiver")
        .expect("enqueue head");
    session.record_queue_notice(notice);
    let tool = session.handoff_tool(Arc::new(PermittingController));
    let result = (tool.execute)(
        "new-request".into(),
        serde_json::json!({"receiver_role":"child","receiver_name":"other-receiver"}),
        None,
        None,
    )
    .await
    .expect("a new request enqueues, never errors");
    assert_eq!(result.details["status"], "queued");
    assert_eq!(result.details["queue_position"], 2);
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].receiver_name, "head-receiver");
    assert_eq!(entries[0].state, crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending);
}

/// F1: the ordinary-send acknowledgment hook. A delivered receipt
/// acknowledges the matching fallback_pending entry; a non-delivered
/// receipt reports honestly and keeps it.
#[tokio::test]
async fn ordinary_send_receipt_acknowledges_fallback_entry() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "fallback-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure")
        .expect("notice");
    assert_eq!(notice.state, "fallback_pending");
    // Unrelated same-receiver message outside the fallback turn: NO token
    // exists to pick up, so it acknowledges nothing (turn-scoped binding).
    assert!(session
        .sender_queue_note_ordinary_send(None, Some("delivered"), None)
        .expect("note")
        .is_empty());
    assert!(session.active_fallback_token_for("fallback-target").is_none());
    // The dedicated fallback turn is marked by the host listener; only
    // then does the token exist, and only for the matching receiver.
    session.issue_fallback_prompt(&notice.queue_id);
    assert!(session.note_fallback_prompt_message(&notice.queue_id));
    session.begin_fallback_turn(&notice.queue_id);
    assert!(session.active_fallback_token_for("other-receiver").is_none());
    let token = session
        .active_fallback_token_for("child:fallback-target")
        .expect("token present during the fallback turn");
    // Non-accepted receipt: honest report, entry kept.
    let kept = session
        .sender_queue_note_ordinary_send(Some(&token), Some("failed"), None)
        .expect("note");
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].state, "fallback_pending");
    assert_eq!(kept[0].reason.as_deref(), Some("ordinary_delivery:failed"));
    // Accepted receipt with the SAME token: terminal fallback pop.
    let acknowledged = session
        .sender_queue_note_ordinary_send(Some(&token), Some("delivered"), Some("agentmsg_reemit"))
        .expect("note");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].state, "fallback");
    assert_eq!(acknowledged[0].reason.as_deref(), Some("ordinary_delivery:delivered"));
    // The terminal notice carries the ACTUAL re-emission message id.
    assert_eq!(acknowledged[0].message_id.as_deref(), Some("agentmsg_reemit"));
    // F3: a queued receipt is acknowledged with the queued transport
    // status recorded honestly, distinct from delivered.
    session.sender_queue_enqueue("child", "queued-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal")
        .expect("notice");
    session.issue_fallback_prompt(&notice.queue_id);
    assert!(session.note_fallback_prompt_message(&notice.queue_id));
    session.begin_fallback_turn(&notice.queue_id);
    let token_q = session
        .active_fallback_token_for("child:queued-target")
        .expect("token");
    let queued_notice = session
        .sender_queue_note_ordinary_send(Some(&token_q), Some("queued"), Some("agentmsg_reemit_q"))
        .expect("note");
    assert_eq!(queued_notice.len(), 1);
    assert_eq!(queued_notice[0].state, "fallback");
    assert_eq!(queued_notice[0].reason.as_deref(), Some("ordinary_delivery:queued"));
    session.end_fallback_turn();
    assert!(session.active_fallback_token_for("child:fallback-target").is_none());
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty());
}


#[tokio::test]
async fn hydrated_receiver_receipt_closes_fallback_with_honest_status() {
    // Live-incident shape (2026-11-09, occ3q_ed361452): the ordinary
    // re-emission for a dead/non-resident receiver is accepted only after
    // the daemon hydrates its saved session artifacts. The receipt carries
    // `delivered_to_hydrated`, and the queue must close the entry on it
    // (nothing better is possible for a non-resident receiver) while
    // recording the distinction - never a clean `delivered`.
    let session = enrolled_session().await;
    session.sender_queue_enqueue("sibling", "dead-receiver").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("terminal capture failure")
        .expect("notice");
    session.issue_fallback_prompt(&notice.queue_id);
    assert!(session.note_fallback_prompt_message(&notice.queue_id));
    session.begin_fallback_turn(&notice.queue_id);
    let token = session
        .active_fallback_token_for("sibling:dead-receiver")
        .expect("token present during the fallback turn");
    let acknowledged = session
        .sender_queue_note_ordinary_send(
            Some(&token),
            Some(crate::core::agent_session::occ3_sender_queue::OCC3_FALLBACK_HYDRATED_STATUS),
            Some("agentmsg_zombie"),
        )
        .expect("note");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].state, "fallback");
    assert_eq!(
        acknowledged[0].reason.as_deref(),
        Some("ordinary_delivery:delivered_to_hydrated")
    );
    assert_eq!(acknowledged[0].message_id.as_deref(), Some("agentmsg_zombie"));
    session.end_fallback_turn();
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty());
}

/// F6: routing registry proof. A delivery event for a message sent by a
/// DIFFERENT session routes to THAT sender's queue, never to the
/// receiver's own queue. Two distinct sessions with distinct ids: the
/// sender's entry pops on its consumption event while the receiver's own
/// decoy entry is untouched.
#[tokio::test]
async fn delivery_events_route_to_the_sender_session_queue() {
    let sender = Occ3NativeSession::enroll(
        "sender-route",
        &fixture_model(),
        Arc::new(QueueFixtureConnector),
    )
    .await
    .expect("sender route fixture");
    let receiver = Occ3NativeSession::enroll(
        "receiver-route",
        &fixture_model(),
        Arc::new(QueueFixtureConnector),
    )
    .await
    .expect("receiver route fixture");
    sender.sender_queue_enqueue("child", "target-route").expect("enqueue");
    sender.sender_queue_mark_armed(None, None).expect("armed");
    sender
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_route",
        )
        .expect("progressed");
    receiver.sender_queue_enqueue("child", "decoy-route").expect("decoy");
    receiver.sender_queue_mark_armed(None, None).expect("decoy armed");
    receiver
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_route",
        )
        .expect("decoy progressed");
    // Drain BOTH worklists: the setup transitions (enqueue/arm/admit on
    // each side) journaled their own notices through the atomic wrapper
    // - routing evidence below counts only the ROUTED event's notices.
    let _setup = sender.take_pending_queue_notices();
    let _setup = receiver.take_pending_queue_notices();
    // Route the SENDER's message through the exact production dispatcher
    // path (the same fn the receiver's restamp listener calls with the
    // sender session id from its poll), WITH the receipt digest a real
    // consumed marker carries: the sender's queue pops, the receiver's
    // own decoy entry stays.
    crate::core::agent_session::occ3_state::route_delivery_event_to_sender(
        "sender-route",
        "agentmsg_route",
        true,
        Some("sha256_route_digest"),
    )
    .await;
    // The notice was journaled ON THE SENDER (not the receiver).
    let sender_notices = sender.take_pending_queue_notices();
    assert_eq!(sender_notices.len(), 1, "sender journals its completion notice");
    assert_eq!(sender_notices[0].state, "completed");
    assert!(receiver.take_pending_queue_notices().is_empty(), "receiver journals nothing for the sender's event");
    let (entries, _) = sender.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty(), "sender's entry popped");
    let (decoy, _) = receiver.sender_queue_snapshot().expect("snapshot");
    assert_eq!(decoy.len(), 1, "receiver's own queue untouched by the sender's event");
}

/// A consumed event without proof must never authorize an ordinary fallback.
#[tokio::test]
async fn digestless_consumed_delivery_routes_to_uncertain_without_fallback() {
    let sender = Occ3NativeSession::enroll(
        "sender-route-digestless",
        &fixture_model(),
        Arc::new(QueueFixtureConnector),
    )
    .await
    .expect("sender fixture");
    sender.sender_queue_enqueue("child", "digestless-target").expect("enqueue");
    sender.sender_queue_mark_armed(None, None).expect("armed");
    sender.sender_queue_mark_progressed(
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
        "agentmsg_route_digestless",
    ).expect("admitted");
    let _setup = sender.take_pending_queue_notices();
    crate::core::agent_session::occ3_state::route_delivery_event_to_sender(
        "sender-route-digestless", "agentmsg_route_digestless", true, None,
    )
    .await;
    let notices = sender.take_pending_queue_notices();
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].state, "uncertain");
    assert!(sender.active_fallback_token_for("child:digestless-target").is_none(), "no fallback authority");
    let (entries, _) = sender.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Uncertain);
    assert!(sender.sender_queue_sweep_expired().expect("sweep").is_empty());
    assert!(sender.sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("failure").is_none());
}

/// F2: an unrelated ordinary message to the same receiver cannot
/// acknowledge the fallback entry (token gate); a duplicate ack with the
/// same token is idempotent; two entries correlate separately.
#[tokio::test]
async fn fallback_ack_correlation_is_token_gated() {
    // Two SIMULTANEOUS fallback_pending entries are only reachable live
    // sequentially (after A falls back, A blocks B's arm until
    // acknowledged): the head-only arming serialization. The production
    // shape with two at once is the RESTART artifact (both demoted on
    // disk), so the fixture builds exactly that - and proves both
    // tokens survive the reload before any acknowledgement runs.
    let now = crate::core::agent_session::occ3_sender_queue::now_epoch();
    let artifact = serde_json::json!({
        "schema": "optimus.occ3-sender-queue.v1",
        "entries": [
            {
                "queueId": "occ3q_corr_a",
                "receiverRole": "child",
                "receiverName": "corr-a",
                "enqueuedAtEpoch": now,
                "expiresAtEpoch": now + 600,
                "state": "fallback_pending",
                "attempts": 3,
                "reason": "native_path_deactivated",
                "fallbackDispatchToken": "occ3dt_corr_a_token",
            },
            {
                "queueId": "occ3q_corr_b",
                "receiverRole": "child",
                "receiverName": "corr-b",
                "enqueuedAtEpoch": now,
                "expiresAtEpoch": now + 600,
                "state": "fallback_pending",
                "attempts": 3,
                "reason": "native_path_deactivated",
                "fallbackDispatchToken": "occ3dt_corr_b_token",
            },
        ],
    });
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    write_queue_artifact(&session_file, &artifact);
    let session = enrolled_session().await;
    session.bind_chat_preferences(Some(session_file.clone())).expect("bind");
    session.load_sender_queue(Some(&session_file)).expect("load");
    // Both fallback_pending entries (and their persisted tokens) survive
    // the reload - the queue state is the durable truth.
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 2, "both fallback_pending entries reloaded");
    assert!(entries.iter().all(|entry| entry.fallback_dispatch_token.is_some()));
    assert!(
        entries
            .iter()
            .any(|entry| entry.fallback_dispatch_token.as_deref() == Some("occ3dt_corr_a_token")),
        "entry A's persisted token survived the reload"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry.fallback_dispatch_token.as_deref() == Some("occ3dt_corr_b_token")),
        "entry B's persisted token survived the reload"
    );
    // The load journal drains before the turn/token assertions below.
    let _load_notices = session.take_pending_queue_notices();
    // Unrelated same-receiver message WITHOUT a token: nothing acknowledged.
    assert!(session
        .sender_queue_note_ordinary_send(None, Some("delivered"), None)
        .expect("note")
        .is_empty());
    // Turn-scoped: the active fallback turn yields exactly ONE entry's
    // token, only for the matching receiver. The other entry is untouched.
    // A turn bound to entry A yields a token only for A's receiver; B's
    // receiver gets none (entry-authoritative correlation).
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    session.issue_fallback_prompt(&entries[0].queue_id);
    assert!(session.note_fallback_prompt_message(&entries[0].queue_id));
    session.begin_fallback_turn(&entries[0].queue_id);
    let token_a = session
        .active_fallback_token_for("child:corr-a")
        .expect("token a");
    assert!(session.active_fallback_token_for("child:corr-b").is_none());
    // A turn bound to entry B (after A's turn ends) yields B's token.
    let token_b = {
        session.end_fallback_turn();
        session.issue_fallback_prompt(&entries[1].queue_id);
        assert!(session.note_fallback_prompt_message(&entries[1].queue_id));
        session.begin_fallback_turn(&entries[1].queue_id);
        session
            .active_fallback_token_for("child:corr-b")
            .expect("token b")
    };
    session.end_fallback_turn();
    assert_ne!(token_a, token_b, "entries carry distinct tokens");
    // Acknowledge entry A via its token: only A pops.
    let acknowledged = session
        .sender_queue_note_ordinary_send(Some(&token_a), Some("delivered"), Some("agentmsg_a"))
        .expect("ack a");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].receiver, "child:corr-a");
    // Duplicate ack with the same token: idempotent no-op.
    assert!(session
        .sender_queue_note_ordinary_send(Some(&token_a), Some("delivered"), Some("agentmsg_a"))
        .expect("duplicate ack")
        .is_empty());
    // Entry B still pending and acknowledges separately with its own token.
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].receiver_name, "corr-b");
    let acknowledged = session
        .sender_queue_note_ordinary_send(Some(&token_b), Some("queued"), Some("agentmsg_b"))
        .expect("ack b");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].receiver, "child:corr-b");
    assert_eq!(acknowledged[0].reason.as_deref(), Some("ordinary_delivery:queued"));
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty());
}

/// F4 production send-hook regression: the ordinary-send acknowledgment runs
/// inside the REAL `AgentSession::send_agent_message` path (the production
/// hook), against a real installed `Occ3NativeSession`. A pending fallback
/// entry survives an unrelated same-receiver ordinary send; only the send
/// made during the entry's dedicated fallback turn closes it, with that
/// send's ACTUAL message id and delivery status.
struct DeliveringController {
    session_id: String,
}

impl crate::core::agent_messages::AgentSessionMessageController for DeliveringController {
    fn roster(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<crate::core::agent_messages::AgentFamilyRosterResult, String>,
                > + Send,
        >,
    > {
        Box::pin(async { Ok(crate::core::agent_messages::AgentFamilyRosterResult::default()) })
    }
    fn await_pending_child_publication(
        &self,
        _: String,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send>,
    > {
        Box::pin(async { Ok(None) })
    }
    fn send_agent_message(
        &self,
        input: crate::core::agent_messages::AgentSessionMessageSendInput,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        crate::core::agent_messages::AgentSessionMessageReceipt,
                        String,
                    >,
                > + Send,
        >,
    > {
        let session_id = self.session_id.clone();
        Box::pin(async move {
            Ok(crate::core::agent_messages::AgentSessionMessageReceipt {
                id: format!("agentmsg_{}", uuid::Uuid::new_v4()),
                source: session_id,
                target: crate::core::agent_messages::AgentSessionMessageEndpoint {
                    session_id: input.target.clone(),
                    ..Default::default()
                },
                message: input.message,
                delivery_status: "delivered".into(),
                delivered_at: Some("now".into()),
                ..Default::default()
            })
        })
    }
}

/// A real `AgentSession` (real agent handle, real send path) with a real
/// installed `Occ3NativeSession` and a controller that DELIVERS ordinary
/// messages with unique receipt ids.
async fn sending_session_fixture() -> (Arc<AgentSession>, Arc<Occ3NativeSession>) {
    let scratch = std::env::temp_dir().canonicalize().unwrap();
    let root = tempfile::Builder::new()
        .prefix("occ3-f4-")
        .tempdir_in(scratch)
        .unwrap();
    let cwd_path = root.path().join("workspace");
    let agent_dir_path = root.path().join("agent");
    std::fs::create_dir_all(&cwd_path).unwrap();
    std::fs::create_dir_all(&agent_dir_path).unwrap();
    let cwd = cwd_path.to_string_lossy().to_string();
    let agent_dir = agent_dir_path.to_string_lossy().to_string();
    let session_manager = Arc::new(Mutex::new(
        crate::core::session_manager::SessionManager::create(&cwd, Some(&agent_dir)).unwrap(),
    ));
    let session_id = session_manager
        .lock()
        .unwrap()
        .get_session_id();
    // The REAL agent handle (supports native state hooks).
    let agent = pi_agent_core::agent::Agent::new(pi_agent_core::agent::AgentOptions {
        session_id: Some(session_id.clone()),
        ..Default::default()
    });
    {
        let mut state = agent.state();
        state.model = fixture_model();
        agent.set_state(state);
    }
    let settings = Arc::new(Mutex::new(
        crate::core::settings_manager::SettingsManager::in_memory(
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
        ),
    ));
    let loader = Arc::new(crate::core::resource_loader::DefaultResourceLoader::new(
        crate::core::resource_loader::DefaultResourceLoaderOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            settings_manager: Some(settings.clone()),
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        },
    ));
    let controller: Arc<DeliveringController> = Arc::new(DeliveringController {
        session_id: session_id.clone(),
    });
    let session = AgentSession::new(AgentSessionConfig {
        // `AgentHandle` is implemented for `Arc<Agent>`; the config's
        // `Arc<dyn AgentHandle>` therefore wraps the agent's own Arc.
        agent: Arc::new(agent),
        session_manager,
        settings_manager: settings,
        service_tier_preference: None,
        cwd: cwd.clone(),
        agent_dir: Some(agent_dir.clone()),
        scoped_models: None,
        resource_loader: loader,
        custom_tools: None,
        model_registry: Arc::new(Mutex::new(
            crate::core::model_registry::ModelRegistry::in_memory(
                crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
            ),
        )),
        initial_active_tool_names: None,
        allowed_tool_names: None,
        include_goals: Some(false),
        agent_message_controller: Some(controller.clone()),
        agent_observe_controller: None,
        include_compact_skill: Some(false),
        rlm_heartbeat_controller: None,
        mcp_manager: None,
        base_tools_override: Some(Vec::new()),
        extension_runner_ref: None,
        session_start_event: None,
        rlm_depth: Some(0),
        rlm_max_depth: Some(2),
        rlm_session_dir: None,
        rlm_parent_node_id: None,
        rlm_parent_agent: None,
        semantic_parent_session_id: None,
        semantic_spawned_by_request_id: None,
        subagent_runtime_host: None,
        autonomous: None,
        prewarm_ipython_kernel: Some(false),
        auto_refine_reviewer: None,
        serialized_refine: None,
        initial_goal: None,
    })
    .unwrap();
    let native = Occ3NativeSession::enroll(
        &session.session_id(),
        &fixture_model(),
        Arc::new(QueueFixtureConnector),
    )
    .await
    .expect("fixture enrollment");
    session
        .install_experimental_native_occ3(
            native.clone(),
            controller as Arc<dyn crate::core::agent_messages::AgentSessionMessageController>,
        )
        .expect("install the native session");
    (session, native)
}

#[tokio::test]
async fn production_send_hook_correlates_only_the_fallback_turn() {
    let (session, native) = sending_session_fixture().await;
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("nested/session.jsonl").to_string_lossy().into_owned();
    native
        .bind_chat_preferences(Some(session_file.clone()))
        .expect("bind");
    // A pending fallback entry for receiver child:fallback-child.
    native.sender_queue_enqueue("child", "fallback-child").expect("enqueue");
    native.sender_queue_mark_armed(None, None).expect("armed");
    let notice = native
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("capture failure")
        .expect("notice");
    assert_eq!(notice.state, "fallback_pending");
    let queue_id = notice.queue_id.clone();
    native.record_queue_notice(notice);
    // Drain the journal so later asserts see only new notices.
    assert!(!native.take_pending_queue_notices().is_empty());

    // (a) UNRELATED ordinary send to the SAME receiver, OUTSIDE the
    // fallback turn (a user-driven message): accepted and delivered, but
    // the fallback entry is untouched - no token exists to acknowledge it.
    let unrelated = session
        .send_agent_message(
            "child:fallback-child".into(),
            "an unrelated user-driven message to the same receiver".into(),
            None,
        )
        .await
        .expect("the unrelated send is accepted");
    assert_eq!(unrelated.delivery_status, "delivered");
    let (entries, _) = native.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1, "the fallback entry survives the unrelated send");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
    );
    assert!(
        native.take_pending_queue_notices().is_empty(),
        "no notice was journaled for the unrelated send"
    );

    // (b) The DEDICATED fallback turn (the real host sequence: the
    // listener issues the prompt, the prompt's MessageEnd consumes the
    // issue and arms the turn): only this send carries the token.
    native.issue_fallback_prompt(&queue_id);
    assert!(native.note_fallback_prompt_message(&queue_id));
    native.begin_fallback_turn(&queue_id);
    let reemission = session
        .send_agent_message(
            "child:fallback-child".into(),
            "the scheduled fallback re-emission".into(),
            None,
        )
        .await
        .expect("the fallback re-emission is accepted");
    assert_eq!(reemission.delivery_status, "delivered");
    native.end_fallback_turn();
    // The entry closed: terminal fallback, with the ACTUAL re-emission's
    // message id and delivery status.
    let (entries, _) = native.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty(), "the entry is terminal after the fallback turn send");
    let notices = native.take_pending_queue_notices();
    let terminal = notices
        .iter()
        .find(|notice| notice.state == "fallback")
        .expect("the terminal fallback notice was journaled");
    assert_eq!(terminal.queue_id, queue_id);
    assert_eq!(terminal.message_id.as_deref(), Some(reemission.id.as_str()));
    assert_eq!(terminal.reason.as_deref(), Some("ordinary_delivery:delivered"));

    // (c) After the fallback turn ended, a NEW fallback entry for the same
    // receiver again survives an ordinary send (no token persists past the
    // turn boundary).
    native.sender_queue_enqueue("child", "fallback-child").expect("enqueue");
    native.sender_queue_mark_armed(None, None).expect("armed");
    let notice = native
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("capture failure")
        .expect("notice");
    assert_eq!(notice.state, "fallback_pending");
    let _ = native.take_pending_queue_notices();
    let after = session
        .send_agent_message(
            "child:fallback-child".into(),
            "a later ordinary message after the turn ended".into(),
            None,
        )
        .await
        .expect("the later send is accepted");
    assert_eq!(after.delivery_status, "delivered");
    let (entries, _) = native.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1, "no token persists past the turn boundary");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
    );
}

/// F3: an uncertain entry resolves ONLY explicitly. Manual re-emit returns
/// it to fallback_pending with a fresh token (the human accepted the
/// duplication risk) and it then closes only through the token-gated
/// path; manual cancel abandons it as terminal failed. No automatic path
/// ever touches it, and the token never serializes into notice details.
#[tokio::test]
async fn uncertain_entries_resolve_only_explicitly() {
    let session = enrolled_session().await;
    // Entry A: consumed without a receipt digest => uncertain.
    session.sender_queue_enqueue("child", "uncertain-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_admitted",
        )
        .expect("progressed");
    let notices = session
        .sender_queue_record_delivery_event("agentmsg_admitted", true, None)
        .expect("uncertain event");
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].state, "uncertain");
    let details = notices[0].details().to_string();
    assert!(!details.contains("dispatchToken"), "no token in details: {details}");
    // TTL sweep and capture-failure journaling never touch it.
    assert!(session.sender_queue_sweep_expired().expect("sweep").is_empty());
    assert!(session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("journal call")
        .is_none());
    // Manual re-emit: fallback_pending with a FRESH token, never a
    // re-send by itself.
    let reemit = session
        .sender_queue_resolve_uncertain(&notices[0].queue_id, true)
        .expect("manual re-emit")
        .expect("notice");
    assert_eq!(reemit.state, "fallback_pending");
    assert_eq!(reemit.reason.as_deref(), Some("manual_reemit"));
    assert!(reemit.dispatch_token.is_some(), "fresh token minted");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
    );
    // The re-emitted entry closes only through the token-gated path.
    session.issue_fallback_prompt(&entries[0].queue_id);
    assert!(session.note_fallback_prompt_message(&entries[0].queue_id));
    session.begin_fallback_turn(&entries[0].queue_id);
    let token = session
        .active_fallback_token_for("child:uncertain-target")
        .expect("token");
    let acknowledged = session
        .sender_queue_note_ordinary_send(Some(&token), Some("delivered"), Some("agentmsg_reemit"))
        .expect("ack");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].state, "fallback");
    assert_eq!(acknowledged[0].message_id.as_deref(), Some("agentmsg_reemit"));
    session.end_fallback_turn();
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert!(entries.is_empty());

    // Entry B: another uncertain entry, cancelled manually.
    session.sender_queue_enqueue("child", "uncertain-two").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    session
        .sender_queue_mark_progressed(
            crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "agentmsg_two",
        )
        .expect("progressed");
    let notices = session
        .sender_queue_record_delivery_event("agentmsg_two", true, None)
        .expect("uncertain event");
    assert_eq!(notices[0].state, "uncertain");
    // Refusals: unknown id, and a non-uncertain entry.
    assert!(session.sender_queue_resolve_uncertain("no-such-entry", true).is_err());
    session.sender_queue_enqueue("child", "plain-pending").expect("enqueue");
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    let plain_id = entries
        .iter()
        .find(|entry| entry.state
            == crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending)
        .map(|entry| entry.queue_id.clone())
        .expect("pending entry");
    assert!(session.sender_queue_resolve_uncertain(&plain_id, true).is_err());
    // Manual cancel: terminal failed, no re-send.
    let cancelled = session
        .sender_queue_resolve_uncertain(&notices[0].queue_id, false)
        .expect("manual cancel")
        .expect("notice");
    assert_eq!(cancelled.state, "failed");
    assert_eq!(cancelled.reason.as_deref(), Some("manual_cancel"));
    let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
    assert_eq!(entries.len(), 1, "only the plain pending entry remains");
    assert_eq!(
        entries[0].state,
        crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending
    );
}

/// F1: the fallback-turn marker is host-issued and one-shot. A MessageEnd
/// that the host never issued (spoofed/replayed), a stale re-use of a
/// consumed issue, and a marker for a closed entry all arm NOTHING.
#[tokio::test]
async fn fallback_turn_marker_is_host_issued_and_one_shot() {
    let session = enrolled_session().await;
    session.sender_queue_enqueue("child", "spoof-target").expect("enqueue");
    session.sender_queue_mark_armed(None, None).expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("capture failure")
        .expect("notice");
    assert_eq!(notice.state, "fallback_pending");
    let queue_id = notice.queue_id.clone();

    // (1) No issue: a spoofed occ3_fallback_prompt MessageEnd for a
    // well-formed, still-pending entry matches nothing.
    assert!(!session.note_fallback_prompt_message(&queue_id));
    // Even if a rogue caller then forced the marker, the token grant for
    // any receiver would... (the grant itself is what matters:) with no
    // armed turn there is no token:
    assert!(session.active_fallback_token_for("child:spoof-target").is_none());

    // (2) Host issues the prompt; its MessageEnd consumes the issue once.
    session.issue_fallback_prompt(&queue_id);
    assert!(session.note_fallback_prompt_message(&queue_id));
    session.begin_fallback_turn(&queue_id);
    let token = session
        .active_fallback_token_for("child:spoof-target")
        .expect("token during the armed turn");
    // (3) A SECOND MessageEnd replaying the same marker (stale) is
    // rejected: the issue was consumed one-shot.
    assert!(!session.note_fallback_prompt_message(&queue_id));
    session.end_fallback_turn();

    // (4) A marker for an entry that has since closed arms nothing: close
    // the entry first (token-gated acknowledge), then try a fresh issue.
    session.issue_fallback_prompt(&queue_id);
    let acknowledged = session
        .sender_queue_note_ordinary_send(Some(&token), Some("delivered"), Some("agentmsg_x"))
        .expect("ack");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].state, "fallback");
    // The issue for the closed entry is now stale: note rejects (entry no
    // longer fallback_pending), and no turn is armed.
    assert!(!session.note_fallback_prompt_message(&queue_id));
    assert!(session.active_fallback_token_for("child:spoof-target").is_none());
}

/// F1: the token grant accepts the AUTHENTICATED receiver session id
/// captured at arm time, and rejects a same-name different-receiver send.
#[tokio::test]
async fn token_grant_uses_the_authenticated_receiver_id() {
    let session = enrolled_session().await;
    // Arm captures the authenticated target session id from the
    // authorization (here supplied directly by the host-side test).
    session.sender_queue_enqueue("child", "id-target").expect("enqueue");
    session
        .sender_queue_mark_armed(None, Some("sess-auth-receiver-1"))
        .expect("armed");
    let notice = session
        .sender_queue_capture_failed(true, "native_path_deactivated")
        .expect("capture failure")
        .expect("notice");
    assert_eq!(notice.state, "fallback_pending");
    session.issue_fallback_prompt(&notice.queue_id);
    assert!(session.note_fallback_prompt_message(&notice.queue_id));
    session.begin_fallback_turn(&notice.queue_id);
    // A DIFFERENT agent that merely shares the display name ("id-target"
    // under a different role, or a bare name form) gets no token.
    assert!(session.active_fallback_token_for("sibling:id-target").is_none());
    assert!(session.active_fallback_token_for("id-target").is_none());
    // The AUTHENTICATED session id grants the token.
    let token = session
        .active_fallback_token_for("sess-auth-receiver-1")
        .expect("token via authenticated id");
    // And so does the entry's own selector.
    assert_eq!(
        session.active_fallback_token_for("child:id-target").as_deref(),
        Some(token.as_str())
    );
    session.end_fallback_turn();
}

/// Historical busy/arm records may drain after admission. They remain audit
/// records, never sender prompts, and replayed worklist clones do nothing.
#[tokio::test]
async fn admitted_queue_drain_does_not_prompt_or_replay_busy_history() {
    let (session, native) = sending_session_fixture().await;
    let controller: Arc<dyn AgentSessionMessageController> = Arc::new(PermittingController);
    let _drain = native.queue_notice_drain.lock().await;
    native.sender_queue_enqueue("child", "receiver").unwrap();
    native.sender_queue_mark_armed(Some("old"), None).unwrap();
    native.sender_queue_source_busy(pi_ai::source_capture::SourceBusyRefusal {
        refused_at_epoch: occ3_state_now(), retry_after_secs: Some(1),
    }).unwrap();
    let gate = Arc::new(Mutex::new(occ3_sender_queue::HeadPromptGate::default()));
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate.clone()).await;
    assert_eq!(session.get_follow_up_messages().len(), 1);
    native.sender_queue_mark_progressed(occ3_sender_queue::Occ3SenderQueueState::Admitted, "sent").unwrap();
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate).await;
    assert!(session.get_follow_up_messages().is_empty(), "admission cancels queued retry prompt");
    let records = native.take_pending_queue_notices();
    assert_eq!(records.len(), 4);
    for record in &records {
        AgentSession::deliver_queue_notice_record(&session, &native, &controller, record.clone()).await;
    }
    assert!(session.get_follow_up_messages().is_empty(), "historical notices cannot drive turns");
    assert!(native.sender_queue_outbox_snapshot().is_empty());
    let projected = crate::core::occ::queue::from_queue_notices(&session.messages(), &session.session_id(), "test");
    assert_eq!(projected.pending_total, 0, "live roster sees admission without a model turn");
    let before = session.session_manager.lock().unwrap().get_entries().len();
    for record in records {
        AgentSession::deliver_queue_notice_record(&session, &native, &controller, record).await;
    }
    assert_eq!(session.session_manager.lock().unwrap().get_entries().len(), before);
    assert!(native.sender_queue_mark_armed(None, None).unwrap().is_none());
    assert!(!native.sender_queue_source_busy(pi_ai::source_capture::SourceBusyRefusal {
        refused_at_epoch: occ3_state_now(), retry_after_secs: None,
    }).unwrap());
    assert!(native.take_pending_queue_notices().is_empty());
    drop(_drain);
    session.dispose_async(Some(false)).await;
}

/// Exercise the production idle-timer body, without waiting for wall time.
#[tokio::test]
async fn idle_receiver_busy_tick_prompts_once_and_admission_cancels_it() {
    let (session, native) = sending_session_fixture().await;
    let controller: Arc<dyn AgentSessionMessageController> = Arc::new(PermittingController);
    let gate = Arc::new(Mutex::new(occ3_sender_queue::HeadPromptGate::default()));
    native.sender_queue_enqueue("child", "receiver").unwrap();
    native.sender_queue_mark_armed(Some("idle-busy"), None).unwrap();
    native.sender_queue_mark_progressed(occ3_sender_queue::Occ3SenderQueueState::Admitted, "idle-busy").unwrap();
    occ3_state::route_receiver_busy_to_sender(&native.session_id(), "idle-busy");
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller.clone(), gate.clone()).await;
    let prompts = session.get_follow_up_messages();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("was busy and did not use"));
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller.clone(), gate.clone()).await;
    assert_eq!(session.get_follow_up_messages(), prompts);
    native.sender_queue_mark_progressed(occ3_sender_queue::Occ3SenderQueueState::Admitted, "idle-retry").unwrap();
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller.clone(), gate.clone()).await;
    assert!(session.get_follow_up_messages().is_empty());
    // Old receiver outcome and another idle tick cannot revive the retry.
    occ3_state::route_receiver_busy_to_sender(&native.session_id(), "idle-busy");
    native.requeue_unacked_queue_notices();
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller, gate).await;
    assert!(session.get_follow_up_messages().is_empty());
    session.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn idle_receiver_failure_tick_schedules_one_fallback_prompt() {
    let (session, native) = sending_session_fixture().await;
    let controller: Arc<dyn AgentSessionMessageController> = Arc::new(PermittingController);
    let gate = Arc::new(Mutex::new(occ3_sender_queue::HeadPromptGate::default()));
    native.sender_queue_enqueue("child", "receiver").unwrap();
    native.sender_queue_mark_armed(Some("idle-failure"), None).unwrap();
    native.sender_queue_mark_progressed(occ3_sender_queue::Occ3SenderQueueState::Admitted, "idle-failure").unwrap();
    occ3_state::route_delivery_event_to_sender(&native.session_id(), "idle-failure", false, None).await;
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller.clone(), gate.clone()).await;
    let prompts = session.get_follow_up_messages();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("fell back to ordinary messaging"));
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller, gate).await;
    assert_eq!(session.get_follow_up_messages(), prompts);
    session.dispose_async(Some(false)).await;
}

/// A model may arm the pending head before its queued prompt is consumed.
/// Returning to pending after a capture failure is a new prompting occasion.
#[tokio::test]
async fn cancelled_head_prompt_can_prompt_once_after_nonterminal_capture_failure() {
    let (session, native) = sending_session_fixture().await;
    let gate = Arc::new(Mutex::new(occ3_sender_queue::HeadPromptGate::default()));
    let controller: Arc<dyn AgentSessionMessageController> = Arc::new(PermittingController);
    let _drain = native.queue_notice_drain.lock().await;
    native.sender_queue_enqueue("child", "receiver").unwrap();
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate.clone()).await;
    assert_eq!(session.get_follow_up_messages().len(), 1);
    native.sender_queue_mark_armed(None, None).unwrap();
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate.clone()).await;
    assert!(session.get_follow_up_messages().is_empty());
    native.sender_queue_capture_failed(false, "capture_failed").unwrap();
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate.clone()).await;
    let prompts = session.get_follow_up_messages();
    assert_eq!(prompts.len(), 1);
    AgentSession::drive_occ3_head_prompt(session.clone(), native.clone(), gate.clone()).await;
    assert_eq!(session.get_follow_up_messages(), prompts);
    drop(_drain);
    AgentSession::sender_queue_idle_tick(session.clone(), native.clone(), controller, gate).await;
    assert_eq!(session.get_follow_up_messages(), prompts, "idle tick cannot duplicate the retry");
    session.dispose_async(Some(false)).await;
}
