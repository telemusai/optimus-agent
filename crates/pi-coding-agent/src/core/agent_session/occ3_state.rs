//! Explicit process-local source/receiver state for isolated native handoff.
//! No settings, durable lease recovery, helper generation or kernel provenance
//! inference. Ordinary sessions have no instance and retain their normal path.
use crate::core::agent_messages::{
    AgentSessionMessageController, Occ3DeliveryDetails, PreparedNativeHandoff,
    AGENT_MESSAGE_CUSTOM_TYPE,
};
use crate::core::occ::occ3::connector::{
    NativePublicationDispatch, NativePublicationRequest, NativeSessionEnrollment, Occ3Connector,
};
use crate::core::occ::occ3::inference::{Occ3InferenceInbox, PrepareInferenceRequest};
use crate::core::occ::occ3::registry::Occ3ConnectorRegistry;
use crate::core::occ::occ3::types::{Occ3DeliveryStatus, Occ3Error};
use pi_agent_core::types::ContentBlock;
use pi_agent_core::types::{
    AgentMessage, AgentTool, AgentToolResult, CustomAgentMessage, ResolveInferenceState,
    SourceCaptureHooks,
};
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{
    SourceCaptureLease, SourceCaptureLimits, SourceCaptureReceipt, SourceCaptureState,
    SourceSpanArming,
};
use super::occ3_sender_queue::{
    Occ3QueueNotice, Occ3SenderQueue, Occ3SenderQueueBlocked, Occ3SenderQueueLoad,
    OCC3_SENDER_QUEUE_MAX_ENTRIES,
};
use pi_ai::types::Model;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const NATIVE_HANDOFF_TOOL: &str = "occ3_handoff";

/// Failed armed captures tolerated before the native path deactivates for
/// the session (recovery, not a brick). Each failure consumes one retry;
/// the model re-arms explicitly by calling the handoff tool again between
/// attempts, so capture stays one-shot per attempt and is never replayed.
pub const MAX_NATIVE_CAPTURE_RETRIES: u32 = 3;
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn target(model: &Model) -> InferenceStateTarget {
    InferenceStateTarget {
        provider: model.provider.clone(),
        model_id: model.id.clone(),
        base_url: model.base_url.clone(),
    }
}
fn failure() -> String {
    "Experimental native handoff unavailable or uncertain; replay is forbidden".into()
}

/// Content-free diagnostic for a receiver-side fail-closed refusal.
/// `reason` is a fixed label naming the exit, and `error_kind` is the
/// connector error's variant name when the refusal came from the
/// connector. It never logs envelope content, facts, keys or server text,
/// and the turn still returns the same generic failure. Exception-safe
/// like `receiver_busy_claim_released`.
fn receiver_refusal_diagnostic(message_id: Option<&str>, reason: &'static str, error_kind: Option<&'static str>) {
    let _ = std::panic::catch_unwind(|| {
        let mut fields = serde_json::Map::from_iter([("reason".into(), Value::from(reason))]);
        if let Some(message_id) = message_id {
            fields.insert("message_id".into(), Value::from(message_id));
        }
        if let Some(error_kind) = error_kind {
            fields.insert("error_kind".into(), Value::from(error_kind));
        }
        pi_ai::log::get_logger("occ3.receiver").warn("receiver_refused", Some(fields));
    });
}

/// The variant name of a connector error, for content-free diagnostics.
fn occ3_error_kind(error: &Occ3Error) -> &'static str {
    match error {
        Occ3Error::IncompatibleModel { .. } => "incompatible_model",
        Occ3Error::Unauthorized => "unauthorized",
        Occ3Error::ConnectorUnreachable { .. } => "connector_unreachable",
        Occ3Error::ConnectorTimeout { .. } => "connector_timeout",
        Occ3Error::HandleExpired { .. } => "handle_expired",
        Occ3Error::HandleAlreadyConsumed { .. } => "handle_already_consumed",
        Occ3Error::RuntimeGenerationChanged { .. } => "runtime_generation_changed",
        Occ3Error::TransferFailed { .. } => "transfer_failed",
        Occ3Error::NativeInferenceUnsupported { .. } => "native_inference_unsupported",
        Occ3Error::NativeInferenceBusyRefused { .. } => "native_inference_busy_refused",
        Occ3Error::InferenceOutcomeUncertain { .. } => "inference_outcome_uncertain",
        Occ3Error::Disabled => "disabled",
        Occ3Error::UnknownConnector { .. } => "unknown_connector",
    }
}

/// Content-free diagnostic for a receiver-busy claim retirement, in the
/// connector-diagnostics style: names the retired message id, the refusing
/// connector, whether the claim actually retired, and whether a live
/// sender absorbed the route - never any envelope content, so live
/// diagnosis of a busy-retired turn needs no source archaeology. The log
/// name `receiver_busy_claim_released` and its `released` key deliberately
/// keep the pre-rename wording for grep stability across builds.
/// Exception-safe like `native_binding_diagnostic`: a broken sink can
/// never turn a retired claim into a failed turn.
fn receiver_busy_claim_released(
    message_id: &str,
    connector_id: &str,
    released: bool,
    sender_routed: bool,
) {
    let _ = std::panic::catch_unwind(|| {
        pi_ai::log::get_logger("occ3.receiver").warn(
            "receiver_busy_claim_released",
            Some(serde_json::Map::from_iter([
                ("message_id".into(), Value::from(message_id)),
                ("connector_id".into(), Value::from(connector_id)),
                ("released".into(), Value::from(released)),
                ("sender_routed".into(), Value::from(sender_routed)),
            ])),
        );
    });
}

/// Process-local registry of live native sessions BY SENDER SESSION ID, so
/// a receiver's consumption dispatcher can route delivery events to the
/// queue of the session that SENT the message - never to the receiver's own
/// queue. Entries are weak: a dropped session simply fails the upgrade.
fn sender_session_registry(
) -> &'static Mutex<std::collections::HashMap<String, Vec<std::sync::Weak<Occ3NativeSession>>>> {
    static REGISTRY: std::sync::OnceLock<
        Mutex<std::collections::HashMap<String, Vec<std::sync::Weak<Occ3NativeSession>>>>,
    > = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Registers one live native session under its sender session id. Called by
/// [`Occ3NativeSession::enroll`].
fn register_sender_session(session: &Arc<Occ3NativeSession>) {
    if let Ok(mut registry) = sender_session_registry().lock() {
        registry
            .entry(session.session_id().to_string())
            .or_default()
            .retain(|weak| weak.upgrade().is_some());
        registry
            .entry(session.session_id().to_string())
            .or_default()
            .push(Arc::downgrade(session));
    }
}

/// Removes one session's registration (terminal teardown). Only THIS
/// instance's weak ref is dropped, so two live sessions sharing an id
/// (test fixtures) never unregister each other.
fn unregister_sender_session(session: &Occ3NativeSession) {
    let Some(self_weak) = session.self_registration.get() else {
        return;
    };
    if let Ok(mut registry) = sender_session_registry().lock() {
        if let Some(weak_list) = registry.get_mut(session.session_id()) {
            weak_list.retain(|weak| {
                !std::sync::Weak::ptr_eq(weak, self_weak)
            });
            // Prune the key itself once the last registration is gone, so
            // a long-lived daemon does not accumulate empty Vec entries.
            if weak_list.is_empty() {
                registry.remove(session.session_id());
            }
        }
    }
}

/// Routes one delivery event to the sender's queue by sender session id and
/// journals the resulting notices on the SENDER (no-op when the sender is
/// gone or has no matching in-flight entry). `pub(crate)` so the routing
/// proof test can drive the exact production path.
pub(crate) async fn route_delivery_event_to_sender(
    sender_session_id: &str,
    message_id: &str,
    consumed: bool,
    receipt_sha256: Option<&str>,
) {
    let senders = {
        let Ok(registry) = sender_session_registry().lock() else {
            return;
        };
        registry
            .get(sender_session_id)
            .map(|weak_list| {
                weak_list
                    .iter()
                    .filter_map(std::sync::Weak::upgrade)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    // Preserve consumption even without a receipt digest: the queue marks
    // that report uncertain, never proven unused or eligible for fallback.
    let mut proven_completion = false;
    for sender in senders {
        if let Ok(notices) =
            sender.sender_queue_record_delivery_event(message_id, consumed, receipt_sha256)
        {
            if !notices.is_empty() {
                // The notices belong to the SENDER's transcript, not the
                // receiver's. The wrapper already recorded each into the
                // sender's durable outbox under the same lock as the queue
                // transition (one atomic persist) - nothing to journal
                // here.
                if consumed && receipt_sha256.is_some() {
                    proven_completion = true;
                }
                break;
            }
        }
    }
    if proven_completion {
        // Proven completion frees the server reservation immediately: the
        // capture behind this message was consumed, so holding its bytes
        // until the sender's next arm (or its session teardown) lets two
        // finished one-shot senders wedge the server's whole live-byte
        // budget until expiry. Best-effort, guarded by the slot's own
        // published message id so a sender already re-armed for a later
        // queue entry is untouched.
        let senders = {
            let Ok(registry) = sender_session_registry().lock() else { return; };
            registry
                .get(sender_session_id)
                .map(|weak_list| {
                    weak_list
                        .iter()
                        .filter_map(std::sync::Weak::upgrade)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        for sender in senders {
            sender.retire_capture_for_completed(message_id).await;
        }
    }
}
/// Routes one proven-non-use receiver-busy refusal to the sender's queue
/// by sender session id: the published capture was proven unused, so the
/// sender re-captures with bounded backoff instead of falling straight
/// back to OCC-1. Returns `true` when a live sender absorbed the busy
/// signal (its queue took the bounded busy retry or exhausted it to
/// fallback); `false` when no live sender, a closed sender or no matching
/// in-flight entry took it, so the caller can record that the facts need
/// explicit attention.
pub(crate) fn route_receiver_busy_to_sender(sender_session_id: &str, message_id: &str) -> bool {
    let senders = {
        let Ok(registry) = sender_session_registry().lock() else {
            return false;
        };
        registry
            .get(sender_session_id)
            .map(|weak_list| {
                weak_list
                    .iter()
                    .filter_map(std::sync::Weak::upgrade)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    for sender in senders {
        if let Ok(notices) = sender.sender_queue_record_receiver_busy(message_id) {
            if !notices.is_empty() {
                return true;
            }
        }
    }
    false
}
#[derive(Clone)]
pub struct NativeSourceBudget {
    pub retained_count_limit: u32,
    pub max_capture_bytes: u64,
    pub deadline: Instant,
    pub max_response_bytes: usize,
}
struct SourceSlot {
    budget: Option<NativeSourceBudget>,
    /// Target fixed by the first, capture-arming tool call in normal sessions.
    armed_receiver: Option<String>,
    /// Failed capture attempts observed through the tombstone path. Each
    /// failed armed capture counts exactly once (the first resolve after the
    /// failure); the slot resets for one re-arm per attempt until this
    /// exceeds MAX_NATIVE_CAPTURE_RETRIES, after which the native path
    /// deactivates for recovery and later turns run the ordinary path.
    failed_attempts: u32,
    /// Retained content-only negotiation for the armed capture: the wire and
    /// the completed-hook contract switch to the no-tool mode for this arm.
    retained: bool,
    resolved: bool,
    lease: Option<SourceCaptureLease>,
    tool_claimed: bool,
    /// One-shot orchestrator publication claim for the retained mode.
    publication_claimed: bool,
    handoff_admitted: bool,
    published_message: Option<(String, String)>,
    negative_control_armed: bool,
    negative_control_exported: bool,
    /// Armed-capture concurrency permit held for the capture lifetime
    /// (default 1 per process). Acquired at arm, released on every
    /// terminal slot outcome. Never held across the queue lock.
    capture_permit: Option<tokio::sync::OwnedSemaphorePermit>,
}
/// The only sanctioned retention reason for an exported negative control:
/// the lease is retained solely so the after-restart phase can prove
/// server-side staleness (the one-shot 409 state probe).
pub const NEGATIVE_CONTROL_RETENTION_REASON: &str = "planned_restart";
/// Root-authorized SEAM1 export of one retained negative-control capture.
/// Carries the RAW source handle and the full bound receipt client-side for
/// the validator's `negative_control_retained` block and the after-phase
/// 409 probe, which reuses the RETURNED value (never a re-export). It is
/// reachable only through [`Occ3NativeSession::export_negative_control`],
/// which requires the explicit pre-capture
/// [`Occ3NativeSession::arm_negative_control`] opt-in; ordinary agent
/// sessions can never obtain it. Not `Clone`; the `Debug` projection hides
/// the handle and the receipt so accidental logging stays hash-only.
pub struct NegativeControlExport {
    pub raw_source_handle: String,
    pub receipt: SourceCaptureReceipt,
    pub runtime_epoch: String,
    pub runtime_generation: u64,
    pub expires_at_epoch: u64,
    pub retention_reason: &'static str,
}
impl std::fmt::Debug for NegativeControlExport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NegativeControlExport")
            .field(
                "raw_source_handle_sha256",
                &pi_ai::source_capture::sha256(self.raw_source_handle.as_bytes()),
            )
            .field("runtime_epoch", &self.runtime_epoch)
            .field("runtime_generation", &self.runtime_generation)
            .field("expires_at_epoch", &self.expires_at_epoch)
            .field("retention_reason", &self.retention_reason)
            .finish_non_exhaustive()
    }
}
/// Owned by one live session and never serialized into its transcript.
pub struct Occ3NativeSession {
    enrollment: NativeSessionEnrollment,
    target: InferenceStateTarget,
    connector: Arc<dyn Occ3Connector>,
    registry: Occ3ConnectorRegistry,
    inbox: Mutex<Occ3InferenceInbox>,
    /// Per-session sender FIFO for queued native handoffs. The slot stays
    /// single-armed; the queue decides which enqueued handoff owns the
    /// next arm. Never serialized into the transcript; its artifact is the
    /// bounded, content-free `occ3-sender-queue.json` projection.
    queue: Mutex<Occ3SenderQueue>,
    /// Serializes event and timer notice drains, including their async parent leg.
    pub(super) queue_notice_drain: tokio::sync::Mutex<()>,
    /// Session-file path bound at install time; `None` in isolated tests
    /// (no persistence, no notices persisted to disk).
    session_file: std::sync::OnceLock<Option<String>>,
    /// Sender-transcript notices waiting for the host's drain (TurnEnd/
    /// AgentEnd). Journal only; never the wire, never handles.
    pending_queue_notices:
        Mutex<Vec<super::occ3_sender_queue::Occ3DurableQueueNotice>>,
    /// The ACTIVE fallback re-emission turn: the queue id of the
    /// fallback entry whose dedicated prompt drove the current turn. Set at
    /// the fallback prompt's MessageEnd, cleared at TurnEnd. Only an
    /// ordinary send DURING this window for the entry's own resolved
    /// receiver may pick up that entry's dispatch token; every other send
    /// has none. The token itself never leaves this struct's memory.
    active_fallback_turn: Mutex<Option<String>>,
    /// Host-issued fallback prompts awaiting their MessageEnd, by queue id
    /// (bounded at 8). Only the HOST listener records an issue when it
    /// queues a fallback re-emission prompt, and each issue is consumed
    /// one-shot by the matching MessageEnd. A `occ3_fallback_prompt`
    /// MessageEnd that the host never queued (a spoofed, stale or replayed
    /// marker) matches nothing and cannot arm the fallback turn.
    pending_fallback_prompt_issues: Mutex<Vec<String>>,
    source: Mutex<SourceSlot>,
    chat_preferences: std::sync::OnceLock<Option<String>>,
    closed: std::sync::atomic::AtomicBool,
    /// Test-only bias added to the wall clock for this session's enrollment
    /// lifetime checks, so expiry-driven recovery is testable offline without
    /// sleeping. Production always reads zero; see [`Self::session_now`].
    #[cfg(test)]
    test_clock_bias: std::sync::atomic::AtomicI64,
    /// Host-side recovery callback run exactly once when this session's
    /// native path is terminally deactivated (retry budget exhausted, or
    /// invalidate from teardown). The host clears its agent hooks and
    /// removes the handoff tool so later turns take the ordinary
    /// inference path instead of failing on a closed native session.
    tombstone_cleanup: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// This instance's own weak registration, so terminal teardown removes
    /// exactly this instance and never a sibling sharing the session id.
    self_registration: std::sync::OnceLock<std::sync::Weak<Occ3NativeSession>>,
    /// Unsubscribe for the delivery-marker restamp listener installed by
    /// [`Self::install_delivery_restamp`]; dropped by [`Self::invalidate`].
    restamp_unsubscribe: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Unsubscribe for the host's sender-queue notice listener; dropped by
    /// [`Self::invalidate`] and [`Self::deactivate`] so no notice drains
    /// for dead state.
    queue_listener_unsubscribe: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}
impl Occ3NativeSession {
    pub async fn enroll(
        session_id: &str,
        model: &Model,
        connector: Arc<dyn Occ3Connector>,
    ) -> Result<Arc<Self>, String> {
        if model.api != pi_ai::types::API_OPENAI_COMPLETIONS {
            return Err(failure());
        }
        let enrollment = connector
            .enroll_native_session(session_id)
            .await
            .map_err(|_| failure())?;
        if enrollment.identity.model_id != model.id
            || enrollment.session_id != session_id
            || enrollment.expires_at_epoch <= now()
        {
            return Err(failure());
        }
        let capabilities = connector
            .discover_capabilities()
            .await
            .map_err(|_| failure())?;
        if capabilities.model_identity != enrollment.identity {
            return Err(failure());
        }
        let mut registry = Occ3ConnectorRegistry::empty();
        registry.register(connector.clone(), capabilities);
        // The family the registry must support is the family the enrollment's
        // full dotted method id names; two methods of one family must not
        // collide with a hard-coded single method check.
        let family = crate::core::occ3_connectors::method_family(&enrollment.identity.latent_method)
            .ok_or_else(failure)?;
        if !registry.supports(
            connector.connector_id(),
            family,
            &enrollment.identity.fingerprint(),
        ) {
            return Err(failure());
        }
        let session = Arc::new(Self {
            enrollment,
            target: target(model),
            connector,
            registry,
            restamp_unsubscribe: Mutex::new(None),
            queue_listener_unsubscribe: Mutex::new(None),
            inbox: Mutex::new(Occ3InferenceInbox::new(session_id.into()).map_err(|_| failure())?),
            queue: Mutex::new(Occ3SenderQueue::from_configured()),
            queue_notice_drain: tokio::sync::Mutex::new(()),
            session_file: std::sync::OnceLock::new(),
            pending_queue_notices: Mutex::new(Vec::new()),
            active_fallback_turn: Mutex::new(None),
            pending_fallback_prompt_issues: Mutex::new(Vec::new()),
            source: Mutex::new(SourceSlot {
                budget: None,
                armed_receiver: None,
                retained: false,
                resolved: false,
                lease: None,
                tool_claimed: false,
                publication_claimed: false,
                handoff_admitted: false,
                published_message: None,
                negative_control_armed: false,
                negative_control_exported: false,
                failed_attempts: 0,
                capture_permit: None,
            }),
            closed: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            test_clock_bias: std::sync::atomic::AtomicI64::new(0),
            chat_preferences: std::sync::OnceLock::new(),
            tombstone_cleanup: Mutex::new(None),
            self_registration: std::sync::OnceLock::new(),
        });
        session
            .self_registration
            .set(Arc::downgrade(&session))
            .map_err(|_| failure())?;
        register_sender_session(&session);
        Ok(session)
    }
    pub fn session_id(&self) -> &str {
        &self.enrollment.session_id
    }

    pub(crate) fn bind_chat_preferences(&self, session_file: Option<String>) -> Result<(), String> {
        self.session_file.set(session_file.clone()).map_err(|_| failure())?;
        self.chat_preferences.set(session_file).map_err(|_| failure())
    }

    /// Normal sessions have no pre-armed validation harness. The first authorized
    /// handoff call arms exactly the next response; it publishes nothing.
    fn arm_handoff_if_needed(
        &self,
        receiver: &str,
        capture_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> Result<bool, String> {
        self.check()?;
        self.check_native_allowed()?;
        let mut slot = self.source.lock().map_err(|_| failure())?;
        if slot.armed_receiver.as_deref().is_some_and(|id| id != receiver) {
            return Err("Native handoff receiver changed after capture was armed".into());
        }
        if let Some(lease) = &slot.lease {
            // Terminal captures retain cleanup metadata, never reuse authority.
            if !matches!(lease.state(), SourceCaptureState::Prepared | SourceCaptureState::Captured) {
                return Err("Native capture ended and is awaiting reset; nothing was sent again. On the next turn, explicitly re-arm the handoff after failure accounting; do not retry this capture in the same turn. [occ3_capture_terminal_pending_reset]".into());
            }
            return Ok(false);
        }
        if slot.resolved || slot.budget.is_some() {
            return Err(failure());
        }
        let caps = self.registry.capabilities(self.connector.connector_id()).ok_or_else(failure)?;
        let bytes_per_element = match self.enrollment.source_profile.dtype.as_str() {
            "bfloat16" | "float16" => 2u64,
            "float32" => 4,
            _ => return Err(failure()),
        };
        let bytes_per_row = u64::from(self.enrollment.source_profile.hidden_dim)
            .checked_mul(bytes_per_element).filter(|n| *n > 0).ok_or_else(failure)?;
        // This budget covers staged buffers on every tensor-parallel rank,
        // not just the final retained rows. GLM's 128-row v2 capture needs
        // about 75 MiB across eight ranks; a logical-row-only cap is refused
        // before inference. Stay within both the verified and local limits.
        let max_capture_bytes = caps.max_state_bytes.min(128 * 1024 * 1024);
        let retained_count_limit = (max_capture_bytes / bytes_per_row).min(128) as u32;
        let seconds = self.enrollment.expires_at_epoch.saturating_sub(self.session_now()).min(600);
        if retained_count_limit == 0 || seconds == 0 {
            return Err(failure());
        }
        if slot.capture_permit.is_none() && capture_permit.is_some() {
            slot.capture_permit = capture_permit;
        }
        slot.budget = Some(NativeSourceBudget {
            retained_count_limit,
            max_capture_bytes,
            deadline: Instant::now() + std::time::Duration::from_secs(seconds),
            max_response_bytes: 8 * 1024 * 1024,
        });
        slot.armed_receiver = Some(receiver.to_string());
        Ok(true)
    }
    pub fn arm_source(&self, budget: NativeSourceBudget) -> Result<(), String> {
        self.arm(budget, false)
    }

    /// Arm the next source request for retained content-only capture: the
    /// captured turn must carry a real eligible content span with ZERO tool
    /// bindings, and publication is dispatched later by the authorised
    /// family controller through [`Self::publish_retained`] - never by a
    /// same-turn tool call.
    pub fn arm_source_retained(&self, budget: NativeSourceBudget) -> Result<(), String> {
        self.arm(budget, true)
    }

    fn arm(&self, budget: NativeSourceBudget, retained: bool) -> Result<(), String> {
        self.check()?;
        if budget.deadline <= Instant::now()
            || budget.retained_count_limit == 0
            || budget.retained_count_limit > 128
        {
            return Err(failure());
        }
        let mut slot = self.source.lock().map_err(|_| failure())?;
        if slot.resolved || slot.budget.is_some() || slot.lease.is_some() {
            return Err(failure());
        }
        // The armed-capture concurrency ceiling also binds the orchestrator
        // and test arming path: a non-blocking permit take records the
        // holder without blocking sync code. The enforced serialization is
        // the tool path's async acquire; when this sync take finds none free
        // the arm still proceeds (test/isolated paths must not deadlock)
        // while the process-wide ceiling keeps its default-of-one guarantee
        // for production tool captures.
        let permit = crate::core::occ3_connectors::activation::try_capture_permit();
        if slot.capture_permit.is_none() && permit.is_some() {
            slot.capture_permit = permit;
        }
        slot.retained = retained;
        slot.budget = Some(budget);
        Ok(())
    }
    /// Isolated-validation-only opt-in (SEAM1). Marks the ALREADY armed
    /// source capture as a planned-restart negative control BEFORE the
    /// capture starts. One-shot and pre-capture only: it fails once the
    /// capture has started (resolved or leased) or when already armed, so an
    /// ordinary completed capture can never be reclassified for export.
    /// Ordinary agent sessions never call this.
    pub fn arm_negative_control(&self) -> Result<(), String> {
        self.check()?;
        let mut slot = self.source.lock().map_err(|_| failure())?;
        if slot.resolved
            || slot.lease.is_some()
            || slot.negative_control_armed
            || slot.budget.is_none()
        {
            return Err(failure());
        }
        slot.negative_control_armed = true;
        Ok(())
    }
    /// Root-authorized SEAM1 accessor: exports the retained negative
    /// control's RAW source handle plus the full bound receipt, exactly
    /// once. Fail-closed: requires the pre-capture `arm_negative_control`
    /// opt-in, a Captured lease that was never tool-claimed, published or
    /// admitted, and a live session. The hash-only `audit()` projection
    /// is unchanged; this is the sole raw-value exit, and the after-phase
    /// 409 probe reuses the RETURNED value (a second call returns None).
    pub fn export_negative_control(&self) -> Option<NegativeControlExport> {
        self.check().ok()?;
        let mut slot = self.source.lock().ok()?;
        if !slot.negative_control_armed
            || slot.negative_control_exported
            || slot.tool_claimed
            || slot.handoff_admitted
            || slot.published_message.is_some()
        {
            return None;
        }
        let receipt = slot.lease.as_ref()?.receipt()?;
        slot.negative_control_exported = true;
        Some(NegativeControlExport {
            raw_source_handle: receipt.source_handle.clone(),
            expires_at_epoch: receipt.expires_at_epoch,
            receipt: receipt.clone(),
            runtime_epoch: self.enrollment.runtime_id.clone(),
            runtime_generation: self.enrollment.runtime_generation,
            retention_reason: NEGATIVE_CONTROL_RETENTION_REASON,
        })
    }
    fn check_native_allowed(&self) -> Result<(), String> {
        if let Some(Some(file)) = self.chat_preferences.get() {
            let preference = crate::core::occ::session_settings::resolve(file)?;
            // With no chat override the already-verified global opt-in applies.
            // Explicit chat/ancestor preferences are read on each native action.
            if preference.source_session_id.is_some() && !preference.occ3_enabled {
                return Err("OCC-3 is disabled for this chat; use /occ 3 on with a supported model".into());
            }
        }
        Ok(())
    }
    /// Wall clock for this session's enrollment lifetime. Tests may bias it
    /// past the enrollment TTL without sleeping; production always reads the
    /// real clock. Both expiry predicates (`check`, `is_dead`) and the arm
    /// budget consult this, never a stale cached instant.
    fn session_now(&self) -> u64 {
        #[cfg(test)]
        {
            let bias = self.test_clock_bias.load(std::sync::atomic::Ordering::Relaxed);
            if bias != 0 {
                return now().saturating_add_signed(bias);
            }
        }
        now()
    }
    /// Test-only: bias this session's clock one second past the enrollment
    /// TTL so expiry-driven recovery runs offline without sleeping.
    #[cfg(test)]
    pub(crate) fn force_expire_enrollment_for_test(&self) {
        let remaining = self.enrollment.expires_at_epoch.saturating_sub(now());
        self.test_clock_bias
            .store(remaining as i64 + 1, std::sync::atomic::Ordering::Relaxed);
    }
    /// Terminal but replaceable: an explicitly closed session, or one whose
    /// enrollment aged out with its permit (server max_ttl_seconds, bounded
    /// to one hour). The install seam uses this to let a fresh activation
    /// replace the dead state instead of wedging long-lived sessions.
    pub fn is_dead(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::Acquire)
            || self.enrollment.expires_at_epoch <= self.session_now()
    }
    fn check(&self) -> Result<(), String> {
        if self.closed.load(std::sync::atomic::Ordering::Acquire)
            || self.enrollment.expires_at_epoch <= self.session_now()
        {
            Err(failure())
        } else {
            Ok(())
        }
    }
    fn check_model(&self, model: &Model) -> Result<(), String> {
        // An enrollment that merely aged out with its permit is
        // terminal-recoverable, not a hard failure: deactivate so the
        // registered cleanup clears the host hooks and the handoff tool,
        // then let THIS request take the ordinary inference path. Nothing
        // was captured or claimed yet - the inbox was invalidated and the
        // sender queue's own entry epochs retire the rest. This is the
        // live wedge fixed: an expired enrollment used to fail every turn
        // with `model_check_failed` because the hooks were never cleared.
        // A model switch stays a loud identity break, and an explicitly
        // closed session still fails closed for in-flight hook calls.
        if !self.closed.load(std::sync::atomic::Ordering::Acquire)
            && self.enrollment.expires_at_epoch <= self.session_now()
        {
            self.deactivate();
            return Ok(());
        }
        self.check()?;
        if model.api != pi_ai::types::API_OPENAI_COMPLETIONS || target(model) != self.target {
            self.invalidate();
            return Err(failure());
        }
        Ok(())
    }
    pub fn source_hooks(self: &Arc<Self>) -> SourceCaptureHooks {
        let resolve = self.clone();
        let complete = self.clone();
        SourceCaptureHooks {
            resolve: Arc::new(move |request| {
                let this = resolve.clone();
                Box::pin(async move {
                    this.check_model(&request.model)
                        .map_err(anyhow::Error::msg)?;
                    let mut tombstone = false;
                    let (budget, retained) = {
                        let mut slot =
                            this.source.lock().map_err(|_| anyhow::anyhow!(failure()))?;
                        match slot.budget.take() {
                            Some(budget) => {
                                slot.resolved = true;
                                if let Err(error) = this.check_native_allowed() {
                                    // No capture was attempted: disabling policy releases
                                    // capacity without consuming the capture-failure budget.
                                    Self::reset_slot_for_fresh_arm(&mut slot);
                                    // Never hold the source guard across queue journaling
                                    // or persistence. No capture attempt was dispatched.
                                    drop(slot);
                                    // The in-memory transition already happened if
                                    // persistence failed; preserve actionable policy guidance.
                                    let _ = this.sender_queue_capture_policy_denied();
                                    return Err(anyhow::Error::msg(error));
                                }
                                (Some(budget), slot.retained)
                            }
                            None => {
                                // Only a completed source + definitively admitted
                                // handoff may enter its normal tool continuation.
                                // Missing budget after a failure is a tombstone,
                                // never permission to retry without capture.
                                tombstone = slot.resolved
                                    && (!slot.handoff_admitted
                                        || !slot.lease.as_ref().is_some_and(|lease| {
                                            lease.state() == SourceCaptureState::Captured
                                        }));
                                (None, slot.retained)
                            }
                        }
                    };
                    if tombstone {
                        // A failed armed capture must not brick the session:
                        // count the failure, reset the slot so the model can
                        // re-arm explicitly (the handoff tool re-arms and
                        // returns capture_armed again), and let THIS turn run
                        // uncaptured. Capture itself stays one-shot and is
                        // never replayed silently. After MAX_NATIVE_CAPTURE_RETRIES
                        // the native path deactivates for the session (the
                        // registered cleanup clears the host hooks and removes
                        // the handoff tool) and later turns take the ordinary
                        // inference path.
                        //
                        // A pre-admission busy refusal is NOT a capture
                        // failure. The decision keys only on the lease's own
                        // refusal record (Cancelled + busy_refusal, set by
                        // pi-ai on a recognised 503 busy code), never on the
                        // queue head's reason text. The refused lease has no
                        // receipt and was never published, so the slot reset
                        // below just drops it (already cancelled); it is never
                        // retired or replayed. The armed queue head then takes
                        // one retry from its separate busy budget; with no
                        // armed head the outcome counts as an ordinary
                        // failure so it stays bounded.
                        if this.absorb_source_busy_refusal() {
                            return Ok(None);
                        }
                        // Preserve terminal receipt metadata until cleanup is attempted.
                        // A failed cleanup is best-effort, as in close(); it never
                        // restores publication authority or silently retries capture.
                        let failed_lease = this.source.lock()
                            .map_err(|_| anyhow::anyhow!(failure()))?.lease.clone();
                        if let Some(lease) = failed_lease {
                            if lease.retirement_receipt().is_some() {
                                let _ = tokio::time::timeout(
                                    std::time::Duration::from_millis(250),
                                    this.connector.retire_native_source(&lease),
                                ).await;
                            }
                        }
                        let exhausted = {
                            let mut slot =
                                this.source.lock().map_err(|_| anyhow::anyhow!(failure()))?;
                            slot.failed_attempts = slot.failed_attempts.saturating_add(1);
                            let exhausted = slot.failed_attempts > MAX_NATIVE_CAPTURE_RETRIES;
                            Self::reset_slot_for_fresh_arm(&mut slot);
                            exhausted
                        };
                        if exhausted {
                            this.deactivate();
                        }
                        // Sender-queue journal: this capture failure belongs to
                        // the armed queue head. Non-terminal: the entry returns
                        // to pending for the model's explicit re-arm inside the
                        // PR235 budget (that budget was consumed above; the
                        // queue adds no budget of its own). Terminal: the entry
                        // falls back to ordinary OCC-1 re-emission.
                        // The wrapper journals the notice into the
                        // durable outbox atomically with the transition.
                        let _ = this.sender_queue_capture_failed(exhausted, "capture_failed");
                        return Ok(None);
                    }
                    let Some(budget) = budget else { return Ok(None); };
                    // Preparation owns capacity across its await; a failed or
                    // cancelled preparation must not leave a permit in the slot.
                    let capture_permit = this.source.lock()
                        .map_err(|_| anyhow::anyhow!(failure()))?.capture_permit.take();
                    let lease = this
                        .connector
                        .prepare_source_capture(
                            this.session_id(),
                            this.target.clone(),
                            budget.retained_count_limit,
                            budget.max_capture_bytes,
                            SourceCaptureLimits {
                                deadline: budget.deadline,
                                max_response_bytes: budget.max_response_bytes,
                            },
                            retained.then_some(SourceSpanArming::RETAINED_CONTENT_ONLY),
                        )
                        .await
                        .map_err(|_| anyhow::anyhow!(failure()))?;
                    let mut slot = this.source.lock().map_err(|_| anyhow::anyhow!(failure()))?;
                    this.check().map_err(anyhow::Error::msg)?;
                    slot.lease = Some(lease.clone());
                    slot.capture_permit = capture_permit;
                    Ok(Some(lease))
                })
            }),
            completed: Arc::new(move |completion| {
                let this = complete.clone();
                Box::pin(async move {
                    // Failed completion checks are terminal for this capture.
                    // Return capacity to the slot only when publication is valid.
                    let mut capture_permit = this.source.lock()
                        .map_err(|_| anyhow::anyhow!(failure()))?.capture_permit.take();
                    let result = (|| -> anyhow::Result<()> {
                        this.check_model(&completion.model)
                            .map_err(anyhow::Error::msg)?;
                        // Each refusal names its check: live TS14 evidence showed
                        // one generic message is not diagnosable from evidence.
                        let bindings = completion.lease.tool_bindings();
                        if completion.lease.state() != SourceCaptureState::Captured {
                            return Err(anyhow::anyhow!(format!("{} [occ3_completed_state_not_captured]", failure())));
                        }
                        if completion.lease.is_retained_content_only() {
                            // Retained content-only mode: the captured turn must
                            // be exactly a content turn. Any tool call forbids the
                            // mode; the empty binding is the valid shape.
                            if !bindings.is_empty() {
                                return Err(anyhow::anyhow!(format!(
                                    "{} [occ3_completed_mode_tool_forbidden]",
                                    failure()
                                )));
                            }
                        } else {
                            // Legacy same-turn protocol: exactly one occ3_handoff
                            // binding from this same captured turn.
                            if bindings.len() != 1 {
                                return Err(anyhow::anyhow!(format!("{} [occ3_completed_binding_count]", failure())));
                            }
                            if bindings[0].tool_name != NATIVE_HANDOFF_TOOL {
                                return Err(anyhow::anyhow!(format!("{} [occ3_completed_tool_name]", failure())));
                            }
                        }
                        let mut slot = this
                            .source
                            .lock()
                            .map_err(|_| anyhow::anyhow!(format!("{} [occ3_source_lock]", failure())))?;
                        let expected = slot
                            .lease
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!(format!("{} [occ3_source_slot_missing]", failure())))?;
                        if expected.intent() != completion.lease.intent() {
                            return Err(anyhow::anyhow!(format!("{} [occ3_completed_intent_mismatch]", failure())));
                        }
                        if slot.tool_claimed {
                            return Err(anyhow::anyhow!(format!("{} [occ3_completed_tool_claimed]", failure())));
                        }
                        this.check().map_err(anyhow::Error::msg)?;
                        slot.capture_permit = capture_permit.take();
                        Ok(())
                    })();
                    if result.is_err() {
                        // Keep the receipt for close/tombstone cleanup, but revoke
                        // capture authority before any subsequent handoff can run.
                        if let Ok(mut slot) = this.source.lock() {
                            if let Some(lease) = &slot.lease {
                                lease.cancel();
                            }
                            slot.budget = None;
                            slot.resolved = true;
                        }
                    }
                    // Revoke failed authority before releasing capacity to another session.
                    drop(capture_permit);
                    result
                })
            }),
        }
    }
    pub fn receiver_resolver(self: &Arc<Self>) -> ResolveInferenceState {
        let this = self.clone();
        Arc::new(move |request| {
            let this = this.clone();
            Box::pin(async move {
                this.check_model(&request.model).map_err(|error| {
                    receiver_refusal_diagnostic(None, "model_check_failed", None);
                    anyhow::Error::msg(error)
                })?;
                let ids: Vec<String> = request
                    .messages
                    .iter()
                    .filter_map(|message| {
                        if let AgentMessage::Custom(CustomAgentMessage::Custom {
                            custom_type,
                            details,
                            ..
                        }) = message
                        {
                            if custom_type == AGENT_MESSAGE_CUSTOM_TYPE {
                                return details.as_ref()?.get("id")?.as_str().map(str::to_string);
                            }
                        }
                        None
                    })
                    .collect();
                // Fresh-input provenance: the ids of the agent-session
                // records THIS inference request newly supplied (the
                // loop-owned prompt/steering/follow-up lanes), extracted
                // with the same rule as `ids` above. A resubmitted expired
                // delivery refuses loudly; history-only presence does not.
                let fresh_ids: Vec<String> = request
                    .newly_supplied_messages
                    .iter()
                    .filter_map(|message| {
                        if let AgentMessage::Custom(CustomAgentMessage::Custom {
                            custom_type,
                            details,
                            ..
                        }) = message
                        {
                            if custom_type == AGENT_MESSAGE_CUSTOM_TYPE {
                                return details.as_ref()?.get("id")?.as_str().map(str::to_string);
                            }
                        }
                        None
                    })
                    .collect();
                let envelope = this
                    .inbox
                    .lock()
                    .map_err(|_| {
                        receiver_refusal_diagnostic(None, "inbox_lock_poisoned", None);
                        anyhow::anyhow!(failure())
                    })?
                    .claim(&ids, &fresh_ids, now())
                    .map_err(|_| {
                        receiver_refusal_diagnostic(None, "inbox_claim_refused", None);
                        anyhow::anyhow!(failure())
                    })?;
                let Some(envelope) = envelope else {
                    return Ok(None);
                };
                // The record is now claimed/`Delivered`: the deterministic
                // checks below (native allowed, session check, registry
                // supports) intentionally fail closed with the record left
                // claimed - the same stuck-claimed shape the busy arm
                // retires. Those refusals are deterministic
                // configuration/identity failures where a silent ordinary
                // turn would hide a broken setup, so they stay loud.
                if let Err(error) = this.check_native_allowed() {
                    receiver_refusal_diagnostic(Some(&envelope.message_id), "native_not_allowed", None);
                    return Err(anyhow::Error::msg(error));
                }
                if let Err(error) = this.check() {
                    receiver_refusal_diagnostic(Some(&envelope.message_id), "session_check_failed", None);
                    return Err(anyhow::Error::msg(error));
                }
                if !this.registry.supports(
                    this.connector.connector_id(),
                    envelope.latent_method,
                    &envelope.model_fingerprint,
                ) {
                    receiver_refusal_diagnostic(Some(&envelope.message_id), "registry_unsupported", None);
                    return Err(anyhow::anyhow!(failure()));
                }
                let claimed_message_id = envelope.message_id.clone();
                // Routed to the sender when a busy refusal retires this
                // claim; captured before the envelope moves into the
                // prepare request.
                let sender_session_id = envelope.sender_session_id.clone();
                let lease = match this
                    .connector
                    .prepare_inference(PrepareInferenceRequest {
                        envelope,
                        request_id: format!("consume-{}", uuid::Uuid::new_v4()),
                        receiver_session_id: this.session_id().into(),
                        receiver_identity: this.enrollment.identity.clone(),
                        target: this.target.clone(),
                        now_epoch: now(),
                    })
                    .await
                {
                    Ok(lease) => lease,
                    Err(Occ3Error::NativeInferenceBusyRefused { connector_id }) => {
                        // A busy refusal of the enroll control call that
                        // `prepare_inference` awaits before building this
                        // consume's attachment. The attachment was therefore
                        // never created nor submitted to the provider, so the
                        // claim is provably undispatched and this delivery is
                        // RETIRED as a terminal busy Failed (pre-notified, so
                        // the TurnEnd poll never re-routes it as an ordinary
                        // receiver failure that would spend an OCC-1
                        // fallback). The SENDER is routed the same
                        // proven-non-use busy signal and re-arms a FRESH
                        // capture; making this record unclaimable is what
                        // keeps the two retry paths exclusive - a re-claim
                        // here could consume this delivery while the sender's
                        // re-send is already in flight and deliver the same
                        // facts twice. This turn proceeds without the native
                        // attachment. The refusal proves nothing about the
                        // server's own state (the worker may have written its
                        // session table before a readiness decline), so
                        // nothing is marked consumed and no second delivery
                        // is ever issued. Mirrors the source-side busy
                        // absorption: busy is never a counted failure and
                        // never a replay permit.
                        let released = match this.inbox.lock() {
                            Ok(mut inbox) => inbox.retire_busy_claim(&claimed_message_id),
                            // A poisoned inbox lock keeps the fail-closed
                            // error, but the one event that must be loud is
                            // the retirement attempt itself: emit the
                            // diagnostic with released=false so the sender's
                            // waiting Admitted entry is not unobservable.
                            // No sender route here: the record was not
                            // retired, so a sender re-send could race a
                            // later claim of this handle.
                            Err(_) => {
                                receiver_busy_claim_released(
                                    &claimed_message_id,
                                    &connector_id,
                                    false,
                                    false,
                                );
                                return Err(anyhow::anyhow!(failure()));
                            }
                        };
                        // Route first so the diagnostic reports whether a
                        // live sender actually absorbed the busy signal: a
                        // retired delivery whose route found no sender
                        // needs explicit attention, not a silent success.
                        let sender_routed = route_receiver_busy_to_sender(
                            &sender_session_id,
                            &claimed_message_id,
                        );
                        receiver_busy_claim_released(
                            &claimed_message_id,
                            &connector_id,
                            released,
                            sender_routed,
                        );
                        if released {
                            // Liveness: this receiver's busy turn proceeds
                            // WITHOUT the attachment, and the control-only
                            // transcript body never carries the source
                            // facts. The retired record is never re-claimable,
                            // so the sender's busy-routed re-arm (above) is
                            // the sole retry path for these facts.
                            return Ok(None);
                        }
                        return Err(anyhow::anyhow!(failure()));
                    }
                    Err(error) => {
                        receiver_refusal_diagnostic(
                            Some(&claimed_message_id),
                            "prepare_inference_refused",
                            Some(occ3_error_kind(&error)),
                        );
                        return Err(anyhow::anyhow!(failure()));
                    }
                };
                this.inbox
                    .lock()
                    .map_err(|_| {
                        receiver_refusal_diagnostic(Some(&claimed_message_id), "inbox_lock_poisoned", None);
                        anyhow::anyhow!(failure())
                    })?
                    .bind_lease(&lease, &this.enrollment.identity)
                    .map_err(|_| {
                        receiver_refusal_diagnostic(Some(&claimed_message_id), "bind_lease_refused", None);
                        anyhow::anyhow!(failure())
                    })?;
                Ok(Some(lease))
            })
        })
    }
    /// Called only from the controller's authenticated acceptance path. Returns
    /// false for duplicate/tombstone; it must not schedule another receiver run.
    pub fn admit(&self, prepared: &PreparedNativeHandoff) -> Result<bool, String> {
        self.check()?;
        self.check_native_allowed()?;
        let e = prepared.envelope();
        if e.receiver_session_id != self.session_id()
            || e.model_fingerprint != self.enrollment.identity.fingerprint()
            || e.runtime_id != self.enrollment.runtime_id
            || e.state_handle.runtime_generation != self.enrollment.runtime_generation
            || e.connector_id != self.connector.connector_id()
        {
            return Err(failure());
        }
        self.inbox
            .lock()
            .map_err(|_| failure())?
            .admit(
                e.clone(),
                prepared.authorization().message_id(),
                prepared.authorization().source_session_id(),
                now(),
            )
            .map_err(|_| failure())
    }
    pub fn outcome(&self, message_id: &str) -> Result<Occ3DeliveryStatus, String> {
        self.inbox
            .lock()
            .map_err(|_| failure())?
            .record_outcome(message_id)
            .map_err(|_| failure())
    }

    /// Content-free digest of this receiver's runtime identity. The raw runtime
    /// id, access domain and model identity never leave this function.
    fn runtime_identity_sha256(&self) -> String {
        pi_ai::source_capture::sha256(
            format!(
                "occ3.runtime.v1\n{}\n{}\n{}",
                self.enrollment.runtime_id,
                self.enrollment.runtime_generation,
                self.enrollment.identity.fingerprint()
            )
            .as_bytes(),
        )
    }

    /// Digest of the completed-use receipt bound to one admitted message. Only the
    /// bound lease can supply the request id and the consumed state, and the
    /// source handle enters this digest already hashed by [`Occ3InferenceInbox::audit`].
    fn completed_use_receipt_sha256(&self, message_id: &str) -> Result<Option<String>, String> {
        let audit = self.inbox.lock().map_err(|_| failure())?.audit();
        let Some(record) = audit
            .into_iter()
            .find(|record| record["message_id"].as_str() == Some(message_id))
        else {
            return Ok(None);
        };
        if record["lease_state"].as_str() != Some("Consumed") {
            return Ok(None);
        }
        let Some(request_id) = record["request_id"].as_str() else {
            return Ok(None);
        };
        let Some(handle_sha256) = record["handle_sha256"].as_str() else {
            return Ok(None);
        };
        Ok(Some(pi_ai::source_capture::sha256(
            format!(
                "occ3.completed_use.v1\n{}\n{}\n{}\n{}\n{}",
                message_id,
                request_id,
                handle_sha256,
                record["runtime_generation"],
                record["model_fingerprint"].as_str().unwrap_or_default()
            )
            .as_bytes(),
        )))
    }

    /// Recorded delivery status. Unlike [`Self::outcome`] this also answers for an
    /// admission that has no bound lease yet; only a bound lease can report
    /// `Consumed`, so an unclaimed admission can never look consumed here.
    pub fn delivery_status(&self, message_id: &str) -> Result<Occ3DeliveryStatus, String> {
        let mut inbox = self.inbox.lock().map_err(|_| failure())?;
        let Some(recorded) = inbox.status(message_id) else {
            return Err(failure());
        };
        Ok(inbox.record_outcome(message_id).unwrap_or(recorded))
    }

    /// The marker that matches this delivery's current receipt state: a completed
    /// `latent_native` marker only when the receiver proved completed use, and a
    /// pulse-free `pending` marker for capture-only, admitted, delivered, failed,
    /// expired and uncertain deliveries.
    pub fn delivery_marker(&self, message_id: &str) -> Result<Occ3DeliveryDetails, String> {
        let method_id = &self.enrollment.identity.latent_method;
        let generation = self.enrollment.runtime_generation;
        let identity = self.runtime_identity_sha256();
        if self.delivery_status(message_id)? != Occ3DeliveryStatus::Consumed {
            return Occ3DeliveryDetails::pending(method_id, generation, &identity);
        }
        let receipt = self
            .completed_use_receipt_sha256(message_id)?
            .ok_or_else(failure)?;
        Occ3DeliveryDetails::latent_native(method_id, generation, &identity, &receipt)
    }

    /// Rewrites the already-admitted transcript record to the marker its receipt
    /// state allows. Same record, same message id, no second record; a repeated
    /// call is idempotent.
    pub fn stamp_delivery_marker(
        &self,
        agent: &dyn crate::core::agent_session::AgentHandle,
        message_id: &str,
    ) -> Result<Occ3DeliveryDetails, String> {
        let marker = self.delivery_marker(message_id)?;
        let message_id = message_id.to_string();
        let applied = marker.clone();
        agent.update_state(Box::new(move |state| {
            crate::core::agent_messages::stamp_occ3_delivery(
                &mut state.messages,
                &message_id,
                &applied,
            );
        }));
        Ok(marker)
    }

    /// Admitted message ids, in inbox order.
    fn admitted_message_ids(&self) -> Vec<String> {
        self.inbox
            .lock()
            .map(|inbox| {
                inbox
                    .audit()
                    .into_iter()
                    .filter_map(|record| {
                        record
                            .get("message_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether the receiver admitted this control message id.
    fn is_admitted(&self, message_id: &str) -> bool {
        self.inbox
            .lock()
            .map(|inbox| inbox.status(message_id).is_some())
            .unwrap_or(false)
    }

    /// The stable id of an agent-messaging control custom message, if `message`
    /// is one.
    fn control_message_id(message: &AgentMessage) -> Option<String> {
        let AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type, details, ..
        }) = message
        else {
            return None;
        };
        if custom_type != AGENT_MESSAGE_CUSTOM_TYPE {
            return None;
        }
        details
            .as_ref()
            .and_then(|details| details.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// Subscribe a delivery-marker restamp on the agent's own event stream.
    ///
    /// The controller stamps the marker once at admission, but a QUEUED body
    /// lands in the transcript only later, unmarked; a body that lands after
    /// its stamp (or that was stamped before its lease was consumed) must be
    /// restamped from the delivery record, so the content-free activity
    /// projection classifies it by its receipt state instead of
    /// misreporting the control-only body as ordinary text. When an admitted
    /// body lands (message end) it is restamped pending; at every turn and
    /// agent end all admitted ids are restamped, which upgrades a delivery to
    /// `latent_native` exactly when its completed-use receipt exists.
    pub fn install_delivery_restamp(
        self: &Arc<Self>,
        agent: Arc<dyn crate::core::agent_session::AgentHandle>,
    ) -> Result<(), String> {
        {
            let guard = self
                .restamp_unsubscribe
                .lock()
                .map_err(|_| failure())?;
            if guard.is_some() {
                return Ok(());
            }
        }
        let receiver = self.clone();
        let agent_handle = Arc::clone(&agent);
        let unsubscribe = agent.subscribe(Arc::new(
            move |event: pi_agent_core::types::AgentEvent,
                  _signal: Option<tokio_util::sync::CancellationToken>| {
                let receiver = receiver.clone();
                let agent = Arc::clone(&agent_handle);
                Box::pin(async move {
                    let ids: Vec<String> = match &event {
                        pi_agent_core::types::AgentEvent::MessageEnd { message } => {
                            let Some(id) = Self::control_message_id(message) else {
                                return;
                            };
                            // Only bodies this receiver admitted are its
                            // markers to rewrite.
                            if !receiver.is_admitted(&id) {
                                return;
                            }
                            vec![id]
                        }
                        pi_agent_core::types::AgentEvent::TurnEnd { .. }
                        | pi_agent_core::types::AgentEvent::AgentEnd { .. } => {
                            // Single-owner consumption dispatcher: drain the
                            // receiver inbox's exactly-once polls (Consumed
                            // pairs and terminal Failed/Expired triples) and
                            // route each event to the SENDER's queue by the
                            // sender session id the poll carries - never to
                            // this receiver's own queue. The polls are
                            // disjoint; a record can never appear twice.
                            // Collect both polls under the inbox lock, then
                            // drop the guard before any sender routing:
                            // sender-queue routing persists queue files, and
                            // no sender IO may run under the inbox lock.
                            let (consumed, terminal) = if let Ok(mut inbox) =
                                receiver.inbox.lock()
                            {
                                let consumed = inbox.poll_consumed();
                                let terminal = inbox
                                    .poll_terminal_failures()
                                    .into_iter()
                                    .map(|(message_id, sender_session_id, _status)| {
                                        let refused_busy = inbox.refused_busy(&message_id);
                                        (message_id, sender_session_id, refused_busy)
                                    })
                                    .collect::<Vec<_>>();
                                (consumed, terminal)
                            } else {
                                (Vec::new(), Vec::new())
                            };
                            for (message_id, sender_session_id, refused_busy) in terminal {
                                // A pre-event busy refusal proved the
                                // capture unused: ask the sender for a
                                // fresh capture (bounded backoff). Any
                                // other non-use outcome falls back.
                                if refused_busy {
                                    // A false here is unobservable today;
                                    // the bound-lease busy path predates
                                    // the retirement diagnostic and is
                                    // left as-is deliberately.
                                    let _sender_absorbed =
                                        route_receiver_busy_to_sender(
                                            &sender_session_id,
                                            &message_id,
                                        );
                                } else {
                                    route_delivery_event_to_sender(
                                        &sender_session_id,
                                        &message_id,
                                        false,
                                        None,
                                    )
                                    .await;
                                }
                            }
                            // Receipt lookup takes the inbox lock itself. Drop
                            // the poll guard before lookup or sender routing.
                            for (message_id, sender_session_id) in consumed {
                                let receipt = receiver
                                    .completed_use_receipt_sha256(&message_id)
                                    .ok()
                                    .flatten();
                                // A missing proof remains consumed-but-uncertain,
                                // never a proven non-use eligible for fallback.
                                route_delivery_event_to_sender(
                                    &sender_session_id,
                                    &message_id,
                                    true,
                                    receipt.as_deref(),
                                )
                                .await;
                            }
                            let ids = receiver.admitted_message_ids();
                            if ids.is_empty() {
                                return;
                            }
                            ids
                        }
                        _ => return,
                    };
                    for id in ids {
                        // Idempotent: the marker always follows the delivery
                        // record's state; a closed receiver refuses stamps.
                        let _ = receiver.stamp_delivery_marker(&*agent, &id);
                    }
                })
            },
        ));
        let mut guard = self
            .restamp_unsubscribe
            .lock()
            .map_err(|_| failure())?;
        *guard = Some(Box::from(unsubscribe));
        Ok(())
    }
    /// The fallback short-circuit for a repeated `occ3_handoff` call. Returns
    /// a result only when the queue head is `fallback_pending` for exactly
    /// this receiver. Read-only: no arm, no enqueue, no notice. The host's
    /// fallback prompt (sent after this run) carries the one-shot dispatch
    /// gate for the ordinary delivery.
    fn fallback_pending_result(&self, role: &str, name: &str) -> Option<AgentToolResult> {
        let queue = self.queue.lock().ok()?;
        let head = queue.head()?;
        if head.receiver_role != role
            || head.receiver_name != name
            || head.state != super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
        {
            return None;
        }
        let cause = match head.reason.as_deref() {
            Some(super::occ3_sender_queue::OCC3_SOURCE_BUSY_EXHAUSTED_REASON) => format!(
                " after {} busy retries (the server was busy and captured nothing)",
                head.busy_retries
            ),
            Some("receiver_busy_exhausted") => format!(
                " after {} busy retries (the receiver was busy and did not use the capture)",
                head.busy_retries
            ),
            _ => String::new(),
        };
        let queue_id = head.queue_id.clone();
        // Exactly one OCC-1 request per entry: only the host's fallback
        // turn carries the dispatch token that completes the entry, so an
        // agent_message sent at any other time would leave the entry
        // pending and the fallback prompt would ask a second time. Outside
        // that turn the model is told to wait; inside it, to send now.
        let in_fallback_turn = self
            .active_fallback_turn
            .lock()
            .ok()
            .is_some_and(|active| active.as_deref() == Some(queue_id.as_str()));
        let instruction = if in_fallback_turn {
            "Send the facts now, once, as an ordinary agent_message to that receiver."
        } else {
            "Do not send the facts now. A separate fallback prompt will ask you to send them once as an ordinary agent_message to that receiver; send them only then."
        };
        Some(AgentToolResult::new(
            vec![ContentBlock::text(format!(
                "Native handoff to {role}:{name} fell back to ordinary messaging{cause}; nothing was armed or queued. {instruction} \
Do not call occ3_handoff again for these facts."
            ))],
            json!({"status":"fallback_pending","delivery_completed":false,"queue_id":queue_id,"busy_retries":head.busy_retries,"send_now":in_fallback_turn}),
        ))
    }

    /// The "already delivered" short-circuit for a repeated `occ3_handoff`
    /// call. Returns a result only when the slot holds an admitted handoff
    /// AND the queue head is the admitted/published entry for exactly this
    /// receiver. Read-only: no re-arm, no lease validation, no notice.
    fn already_admitted_result(&self, role: &str, name: &str) -> Option<AgentToolResult> {
        let admitted = self.source.lock().ok().is_some_and(|slot| slot.handoff_admitted);
        if !admitted {
            return None;
        }
        let queue = self.queue.lock().ok()?;
        let head = queue.head()?;
        if head.receiver_role != role
            || head.receiver_name != name
            || !matches!(
                head.state,
                super::occ3_sender_queue::Occ3SenderQueueState::Admitted
                    | super::occ3_sender_queue::Occ3SenderQueueState::Published
            )
        {
            return None;
        }
        let message_id = head.message_id.clone();
        let queue_id = head.queue_id.clone();
        Some(AgentToolResult::new(
            vec![ContentBlock::text(format!(
                "Native handoff for {role}:{name} was already admitted{}; nothing was sent again. Do not repeat this handoff.",
                message_id.as_deref().map(|id| format!(" ({id})")).unwrap_or_default()
            ))],
            json!({"status":"already_delivered","delivery_completed":true,"queue_id":queue_id,"message_id":message_id}),
        ))
    }

    /// Readies a receiver-busy re-capture for the pending head addressed by
    /// this call. No-op unless the head is `pending` for exactly this
    /// receiver with a scheduled busy retry. The previous capture was
    /// published and then refused unused, so it is retired (never resent)
    /// and the slot is reset for a fresh arm. Then waits out the remaining
    /// backoff, at most [`receiver_busy_backoff_secs`] of the last retry.
    ///
    /// Also serves source-busy re-arms (head reason `source_busy_retry`):
    /// nothing was admitted then, so there is no spent capture to retire.
    /// The wait is capped at [`OCC3_BUSY_BACKOFF_MAX_SECS`] and stops with a
    /// failure when `signal` is cancelled, so no arm follows a cancel. It
    /// runs BEFORE the capture permit is acquired, so no permit is held
    /// while waiting.
    async fn prepare_receiver_busy_recapture(
        self: &Arc<Self>,
        role: &str,
        name: &str,
        signal: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<(), String> {
        let not_before = {
            let queue = self.queue.lock().map_err(|_| failure())?;
            let Some(head) = queue.head() else { return Ok(()) };
            if head.receiver_role != role
                || head.receiver_name != name
                || head.state != super::occ3_sender_queue::Occ3SenderQueueState::Pending
                || head.busy_retries == 0
            {
                return Ok(());
            }
            head.retry_not_before_epoch
        };
        let spent_capture = self
            .source
            .lock()
            .map(|slot| {
                slot.handoff_admitted
                    && slot
                        .lease
                        .as_ref()
                        .is_some_and(|lease| lease.state() == SourceCaptureState::Captured)
            })
            .unwrap_or(false);
        if spent_capture {
            self.complete_handoff_slot().await?;
        }
        if let Some(not_before) = not_before {
            let cap = super::occ3_sender_queue::OCC3_BUSY_BACKOFF_MAX_SECS;
            let wait = not_before.saturating_sub(now()).min(cap);
            if wait > 0 {
                let sleep = tokio::time::sleep(std::time::Duration::from_secs(wait));
                match signal {
                    Some(signal) => {
                        tokio::select! {
                            _ = sleep => {}
                            _ = signal.cancelled() => return Err(failure()),
                        }
                    }
                    None => sleep.await,
                }
            }
        }
        Ok(())
    }

    pub fn handoff_tool(
        self: &Arc<Self>,
        controller: Arc<dyn AgentSessionMessageController>,
    ) -> AgentTool {
        let this = self.clone();
        AgentTool {
            name:NATIVE_HANDOFF_TOOL.into(),label:"Experimental native state handoff".into(),
            description:"Transfer native state to one compatible authorized family agent. On an unarmed session the first call returns capture_armed without sending anything. Then emit the concise source facts as assistant content and call this tool once more for the same receiver, with no other tool calls in that response. Never put source facts in arguments. After admission do not repeat the handoff. If the server or receiver is busy, nothing is delivered or used; call this tool again to re-arm a fresh capture after a short wait (up to 3 busy retries, not counted as failures), then emit the facts again; after that the handoff falls back to ordinary messaging. A failed capture can be retried by calling this tool again (up to 3 retries); after that native handoffs are disabled for this session and ordinary messages continue to work.".into(),
            parameters:json!({"type":"object","additionalProperties":false,"properties":{"receiver_role":{"type":"string","enum":["parent","sibling","child"]},"receiver_name":{"type":"string"}},"required":["receiver_role","receiver_name"]}),
            prepare_arguments:None,execution_mode:None,
            execute:Arc::new(move |tool_id,args,signal,_update| {
                let this=this.clone(); let controller=controller.clone(); Box::pin(async move {
                    this.check().map_err(anyhow::Error::msg)?;
                    this.check_native_allowed().map_err(anyhow::Error::msg)?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { return Err(anyhow::anyhow!(failure())); }
                    let object=args.as_object().ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_args_not_object]", failure())))?;
                    if object.len()!=2 { return Err(anyhow::anyhow!(format!("{} [occ3_tool_args_len]", failure()))); }
                    let role=object.get("receiver_role").and_then(Value::as_str).ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_role_missing]", failure())))?;
                    let name=object.get("receiver_name").and_then(Value::as_str).filter(|v|!v.is_empty()&&v.len()<=128).ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_name_missing]", failure())))?;
                    let authorized=controller.authorize_native_handoff(role.into(),name.into()).await.map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_authorize: {e}]", failure())))?;
                    if authorized.source_session_id()!=this.session_id() { return Err(anyhow::anyhow!(format!("{} [occ3_tool_sender_mismatch]", failure()))); }
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { return Err(anyhow::anyhow!(failure())); }
                    // An admitted handoff is terminal for this call path: a
                    // repeat for the same receiver must neither re-arm nor
                    // re-validate the completed capture against a newer
                    // response (that comparison can only fail provenance).
                    if let Some(result) = this.already_admitted_result(role, name) {
                        return Ok(result);
                    }
                    // Receiver-busy re-capture: the head went back to
                    // `pending` after the receiver refused the published
                    // capture before using it. Retire that spent capture so a
                    // FRESH arm is legal, then honour the bounded backoff.
                    this.prepare_receiver_busy_recapture(role, name, signal.as_ref()).await.map_err(anyhow::Error::msg)?;
                    // A same-receiver call while the head is falling back to
                    // ordinary messaging (for example the session retry
                    // re-ran a context that ended in capture_armed just as
                    // the busy budget ran out) must not queue a second
                    // native handoff of the same facts. Report the fallback
                    // truthfully; nothing is armed or enqueued.
                    if let Some(result) = this.fallback_pending_result(role, name) {
                        return Ok(result);
                    }
                    let queue_head = {
                        let queue = this.queue.lock().map_err(|_| anyhow::anyhow!(failure()))?;
                        queue.head().map(|head| {
                            (
                                head.receiver_role.clone(),
                                head.receiver_name.clone(),
                                head.state.is_in_flight()
                                    || head.state == super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending,
                            )
                        })
                    };
                    // F4: any genuinely new request (different receiver OR a
                    // repeat of a busy receiver) queues behind the current
                    // work; the immutable current capture is preserved and
                    // the queue position is reported. Only the head's own
                    // pending entry continues the two-step arm/publish
                    // completion below.
                    if let Some((head_role, head_name, _head_busy)) = &queue_head {
                        // Same receiver on an armed head = the two-step
                        // completion's second call (the capture is armed for
                        // exactly this receiver); it falls through to the
                        // publish path. Everything else queues.
                        let completing_armed_head = *head_role == role
                            && *head_name == name
                            && this.source.lock().map(|slot| slot.budget.is_some() || slot.lease.is_some()).unwrap_or(false);
                        let same_pending_head = *head_role == role
                            && *head_name == name
                            && !*_head_busy;
                        if !same_pending_head && !completing_armed_head {
                            let notice = this
                                .sender_queue_enqueue(role, name)
                                .map_err(|e| anyhow::anyhow!(format!("{} [occ3_tool_enqueue: {e}]", failure())))?;
                            let (queued_id, queued_position) = (notice.queue_id.clone(), notice.position);
                            // The wrapper journals the notice into the
                            // durable outbox atomically with the enqueue.
                            let head_label = format!("{head_role}:{head_name}");
                            return Ok(AgentToolResult::new(vec![ContentBlock::text(format!(
                                "Native handoff queued for {role}:{name}. The current entry {head_label} finishes first; position {queued_position}. You will be prompted to emit fresh source facts when this entry reaches the queue head."
                            ))], json!({"status":"queued","queue_id":queued_id,"queue_position":queued_position})));
                        }
                    }
                    // Completed-handoff continuation: the slot holds a finished
                    // handoff. Complete it (retire + reset) and enqueue this new
                    // receiver instead of failing; the model gets the queue
                    // advance prompt when the previous entry's notice drains.
                    {
                        let queue_empty = this.queue.lock().map(|queue| queue.is_empty()).unwrap_or(false);
                        let slot_completed = this.source.lock().map(|slot| {
                            slot.handoff_admitted
                                && slot.lease.as_ref().is_some_and(|lease| {
                                    lease.state() == SourceCaptureState::Captured
                                })
                        }).unwrap_or(false);
                        if slot_completed && queue_empty {
                            this.complete_handoff_slot().await.map_err(anyhow::Error::msg)?;
                            let notice = this.sender_queue_enqueue(role, name).map_err(|e| anyhow::anyhow!(format!("{} [occ3_tool_enqueue: {e}]", failure())))?;
                            let (queued_id, queued_position) = (notice.queue_id.clone(), notice.position);
                            // The wrapper journals the notice into the
                            // durable outbox atomically with the enqueue.
                            let queue_len = this.queue.lock().map(|queue| queue.len()).unwrap_or(0);
                            return Ok(AgentToolResult::new(vec![ContentBlock::text(format!(
                                "Native handoff queued for {}:{}. {} handoff(s) now queued; you will be prompted to emit fresh source facts when this entry reaches the queue head.",
                                role, name, queue_len
                            ))], json!({"status":"queued","queue_id":queued_id,"queue_position":queued_position})));
                        }
                    }
                    // The INITIAL request (no queue head at all) creates its
                    // stable head entry BEFORE the arm: every later receiver
                    // sees a busy head and queues behind it at position 2,
                    // 3, ... instead of falling through to the permit
                    // acquire. The wrapper journals the notice into the
                    // durable outbox atomically with the enqueue.
                    let slot_holds_active_capture = this
                        .source
                        .lock()
                        .map(|slot| {
                            slot.lease.is_some()
                                || (slot.armed_receiver.as_deref()
                                    == Some(authorized.target_session_id())
                                    && slot.budget.is_some())
                        })
                        .unwrap_or(false);
                    if queue_head.is_none() && !slot_holds_active_capture {
                        this.sender_queue_enqueue(role, name)
                            .map_err(|e| anyhow::anyhow!(format!("{} [occ3_tool_enqueue_head: {e}]", failure())))?;
                    }
                    // Permit acquisition ONLY for a fresh arm. A completing
                    // same-receiver call reuses the permit its own arm
                    // stored in the slot: reacquiring here would wait on
                    // the very pool permit the active capture already
                    // holds (default pool = 1) and deadlock the two-step
                    // flow. The publish path still validates the exact
                    // tool binding; nothing is replayed silently.
                    let capture_permit = if slot_holds_active_capture {
                        None
                    } else {
                        // Waiting does not arm or replay anything. A stuck holder
                        // must not prevent cancellation or monopolize this turn.
                        // The default 90s covers a sibling sender's whole
                        // arm->publish window, which is dominated by that
                        // model's facts turn (measured ~40s live); 30s cut it
                        // off and forced a noisy re-arm cycle. The wait runs
                        // between model turns, so no inference stream stalls.
                        let wait = tokio::time::timeout(
                            crate::core::occ3_connectors::activation::capture_capacity_wait(),
                            crate::core::occ3_connectors::activation::acquire_capture_permit(),
                        );
                        let cancelled = async {
                            match signal.as_ref() {
                                Some(signal) => signal.cancelled().await,
                                None => std::future::pending::<()>().await,
                            }
                        };
                        let permit = tokio::select! {
                            biased;
                            _ = cancelled => return Err(anyhow::anyhow!(
                                "Native capture capacity wait cancelled; nothing was armed, sent, or replayed [occ3_capture_capacity_cancelled]"
                            )),
                            result = wait => result.map_err(|_| anyhow::anyhow!(
                                "Native capture capacity wait timed out; nothing was armed, sent, or replayed [occ3_capture_capacity_timeout]"
                            ))?,
                        };
                        Some(permit)
                    };
                    // Busy re-arm context for the truthful tool text: read
                    // before mark_armed clears the busy reason.
                    // Keyed on the reason, not on busy_retries alone: an
                    // ordinary failure after a busy re-arm clears the reason
                    // but keeps the count, and must not read as busy.
                    let busy_rearm = this.queue.lock().ok().and_then(|queue| {
                        use super::occ3_sender_queue::{
                            Occ3SenderQueueState, OCC3_RECEIVER_BUSY_RETRY_REASON, OCC3_SOURCE_BUSY_RETRY_REASON,
                        };
                        queue.head().and_then(|head| {
                            if head.state != Occ3SenderQueueState::Pending || head.busy_retries == 0 {
                                return None;
                            }
                            match head.reason.as_deref() {
                                Some(OCC3_SOURCE_BUSY_RETRY_REASON) => Some((head.busy_retries, true)),
                                Some(OCC3_RECEIVER_BUSY_RETRY_REASON) => Some((head.busy_retries, false)),
                                _ => None,
                            }
                        })
                    });
                    if this.arm_handoff_if_needed(authorized.target_session_id(), capture_permit).map_err(anyhow::Error::msg)? {
                        let attempts = this.failed_capture_attempts();
                        // Queue journal: the head entry (if any) is armed.
                        // The wrapper journals the notice into the
                        // durable outbox atomically with the transition.
                        let _ = this
                            .sender_queue_mark_armed(
                                Some(authorized.message_id()),
                                Some(authorized.target_session_id()),
                            )
                            .ok()
                            .flatten();
                        // "busy retries do not count as failures" below holds
                        // on every path reachable from this tool: a busy
                        // refusal is counted as a failure only when no queue
                        // head was armed, and the tool always arms the head.
                        let busy_note = match busy_rearm {
                            Some((retry, true)) => format!(
                                " (busy retry {retry} of {}: the server was busy and captured nothing; the backoff wait has elapsed, so emit the facts again)",
                                super::occ3_sender_queue::OCC3_RECEIVER_BUSY_MAX_RETRIES
                            ),
                            Some((retry, false)) => format!(
                                " (busy retry {retry} of {}: the receiver was busy and did not use the earlier capture; the backoff wait has elapsed, so emit the facts again)",
                                super::occ3_sender_queue::OCC3_RECEIVER_BUSY_MAX_RETRIES
                            ),
                            None => String::new(),
                        };
                        let retry_note = if attempts > 0 {
                            format!(
                                " (retry attempt {} of {} after {} failed capture{})",
                                attempts,
                                MAX_NATIVE_CAPTURE_RETRIES,
                                attempts,
                                if attempts == 1 { "" } else { "s" }
                            )
                        } else {
                            String::new()
                        };
                        return Ok(AgentToolResult::new(vec![ContentBlock::text(format!(
                            "Native source capture armed{busy_note}{retry_note}; nothing has been sent. In your next response emit only the concise source facts as assistant content, then call occ3_handoff once with the same receiver_role and receiver_name. Do not call other tools or include source facts in tool arguments. If the server is busy, call this tool again to re-arm after a short wait (up to {} busy retries, which do not count as failures); after that the handoff falls back to ordinary messaging. If the captured turn fails for another reason you can re-arm up to {} times before native handoffs are disabled for this session; ordinary messages always keep working.",
                            super::occ3_sender_queue::OCC3_RECEIVER_BUSY_MAX_RETRIES,
                            MAX_NATIVE_CAPTURE_RETRIES
                        ))], json!({"status":"capture_armed","delivery_completed":false,"capture_attempts_failed":attempts,"busy_retries":busy_rearm.map(|(retry, _)| retry).unwrap_or(0)})));
                    }
                    // Publication owns the permit now. Local ownership releases
                    // it on failure or cancellation without resetting the spent slot.
                    let (source, publication_permit)={let mut slot=this.source.lock().map_err(|_|anyhow::anyhow!(format!("{} [occ3_tool_source_lock]", failure())))?;
                        let permit = slot.capture_permit.take();
                        let source=slot.lease.clone().ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_source_lease_missing]", failure())))?;
                        source.validate_tool(&tool_id,NATIVE_HANDOFF_TOOL,&args).map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_validate_tool: {e}]", failure())))?;
                        if slot.tool_claimed {return Err(anyhow::anyhow!(format!("{} [occ3_tool_already_claimed]", failure())));} slot.tool_claimed=true; (source, permit)};
                    this.check().map_err(anyhow::Error::msg)?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(failure())); }
                    let expected_message_id=authorized.message_id().to_string();
                    let expected_receiver_id=authorized.target_session_id().to_string();
                    let publication=this.connector.publish_native_source(NativePublicationRequest {source:source.clone(),
                        receiver_session_id:authorized.target_session_id().into(),message_id:authorized.message_id().into(),
                        publication_id:format!("publication-{}",uuid::Uuid::new_v4()),dispatch:NativePublicationDispatch::ModelTool{tool_call_id:tool_id,tool_name:NATIVE_HANDOFF_TOOL.into(),tool_arguments:args}}).await;
                    // Publication spends capture capacity; receiver admission and
                    // consumption may finish much later and do not own this slot.
                    drop(publication_permit);
                    let publication = publication
                        .map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_publish: {e}]", failure())))?;
                    this.source.lock().map_err(|_|anyhow::anyhow!(format!("{} [occ3_tool_publish_lock]", failure())))?.published_message=Some((publication.envelope.message_id.clone(),publication.binding_digest.clone()));
                    this.check().map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_check_publish: {e}]", failure())))?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(format!("{} [occ3_tool_signal_cancel]", failure()))); }
                    let prepared=PreparedNativeHandoff::new(authorized,publication).map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_prepared: {e}]", failure())))?;
                    let receipt=match controller.send_prepared_native_handoff(prepared).await {
                        Ok(receipt)=>receipt,
                        Err(error)=>{
                            // The connector publication succeeded but the authenticated
                            // send failed: this capture cannot be republished (the
                            // serving side pins one publication per capture), so the
                            // queued entry falls back to ordinary OCC-1 re-emission
                            // instead of burning a retry on a dead capture.
                            // The wrapper journals the notice into the
                            // durable outbox atomically with the transition.
                            let _ = this.sender_queue_capture_failed(true,"send_failed_after_publish");
                            return Err(anyhow::anyhow!(format!("{} [occ3_tool_send: {error}]", failure())));
                        }
                    };
                    if receipt.id!=expected_message_id || receipt.target.session_id!=expected_receiver_id
                        || !matches!(receipt.delivery_status.as_str(),"delivered"|"queued")
                    { return Err(anyhow::anyhow!(format!("{} [occ3_tool_receipt_mismatch]", failure()))); }
                    this.check().map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_check_final: {e}]", failure())))?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(failure())); }
                    // Admission is not consumption proof. The isolated native
                    // controller separately requires Consumed; the real daemon
                    // may truthfully acknowledge a queued admission only.
                    this.source.lock().map_err(|_|anyhow::anyhow!(failure()))?.handoff_admitted=true;
                    // Queue journal: the head entry reached the admitted state.
                    // The wrapper journals the notice into the durable
                    // outbox atomically with the transition.
                    let _ = this.sender_queue_mark_progressed(
                        super::occ3_sender_queue::Occ3SenderQueueState::Admitted,
                        &receipt.id,
                    ).ok().flatten();
                    Ok(AgentToolResult::new(vec![ContentBlock::text(format!("Native handoff admitted: {}",receipt.id))],
                        json!({"message_id":receipt.id,"delivery_status":receipt.delivery_status})))
                })
            }),
        }
    }

    /// Deterministic orchestrator publication of a retained content-only
    /// capture. The recipient must already be requested and permitted by
    /// the normal agent workflow: authorization goes through the same
    /// authenticated family controller the handoff tool uses, and the
    /// retained capture is published exactly once under its original
    /// capture-turn lineage. There is no model tool dispatch in this mode.
    pub async fn publish_retained(
        self: &Arc<Self>,
        controller: Arc<dyn AgentSessionMessageController>,
        relationship: &str,
        selector: &str,
    ) -> Result<crate::core::agent_messages::AgentSessionMessageReceipt, String> {
        self.check()?;
        self.check_native_allowed()?;
        let (source, publication_permit) = {
            let mut slot = self
                .source
                .lock()
                .map_err(|_| format!("{} [occ3_publish_source_lock]", failure()))?;
            let source = slot
                .lease
                .clone()
                .ok_or_else(|| format!("{} [occ3_publish_lease_missing]", failure()))?;
            if source.state() != SourceCaptureState::Captured {
                return Err(format!("{} [occ3_publish_state_not_captured]", failure()));
            }
            if !source.is_retained_content_only() {
                return Err(format!("{} [occ3_publish_mode_not_retained]", failure()));
            }
            if slot.tool_claimed {
                return Err(format!("{} [occ3_publish_tool_claimed]", failure()));
            }
            if slot.publication_claimed {
                return Err(format!("{} [occ3_publish_already_claimed]", failure()));
            }
            // One-shot claim BEFORE any await: concurrent callers cannot
            // publish the same retained capture twice.
            slot.publication_claimed = true;
            (source, slot.capture_permit.take())
        };
        let authorized = controller
            .authorize_native_handoff(relationship.into(), selector.into())
            .await
            .map_err(|e| format!("{} [occ3_publish_authorize: {e}]", failure()))?;
        if authorized.source_session_id() != self.session_id() {
            return Err(format!("{} [occ3_publish_sender_mismatch]", failure()));
        }
        self.check()?;
        let expected_message_id = authorized.message_id().to_string();
        let expected_receiver_id = authorized.target_session_id().to_string();
        let publication = self
            .connector
            .publish_native_source(NativePublicationRequest {
                source: source.clone(),
                receiver_session_id: authorized.target_session_id().into(),
                message_id: authorized.message_id().into(),
                publication_id: format!("publication-{}", uuid::Uuid::new_v4()),
                dispatch: NativePublicationDispatch::Orchestrator,
            })
            .await;
        // Retained publication has the same capacity lifetime as tool publication.
        drop(publication_permit);
        let publication = publication
            .map_err(|e| format!("{} [occ3_publish_publish: {e}]", failure()))?;
        self.source
            .lock()
            .map_err(|_| format!("{} [occ3_publish_lock]", failure()))?
            .published_message =
            Some((publication.envelope.message_id.clone(), publication.binding_digest.clone()));
        let prepared = PreparedNativeHandoff::new(authorized, publication)
            .map_err(|e| format!("{} [occ3_publish_prepared: {e}]", failure()))?;
        let receipt = controller
            .send_prepared_native_handoff(prepared)
            .await
            .map_err(|e| format!("{} [occ3_publish_send: {e}]", failure()))?;
        if receipt.id != expected_message_id
            || receipt.target.session_id != expected_receiver_id
            || !matches!(receipt.delivery_status.as_str(), "delivered" | "queued")
        {
            return Err(format!("{} [occ3_publish_receipt_mismatch]", failure()));
        }
        self.check()?;
        self.source
            .lock()
            .map_err(|_| failure())?
            .handoff_admitted = true;
        Ok(receipt)
    }

    /// Span-coverage evidence for the experiment-arm trials (OCC-3 campaign
    /// section 5: TOKEN_FULL / TOKEN_SELECTED). Projects the captured
    /// receipt's `occ3.span.v1` block plus the retained-window integers a
    /// driver-side coverage assertion needs. Hash-only discipline is kept:
    /// no raw handle leaves this session. `None` before a receipt lands or
    /// when the capture ran under a v1 intent (no span selection exists).
    pub fn span_evidence(&self) -> Option<Value> {
        let slot = self.source.lock().ok()?;
        let lease = slot.lease.as_ref()?;
        let receipt = lease.retirement_receipt()?;
        let span = receipt.span.as_ref()?;
        Some(json!({
            "span_schema": span.span_schema,
            "span_policy": span.span_policy,
            "span_marker_profile": span.span_marker_profile,
            "span_marker_schema": span.span_marker_schema,
            "span_reasoning_terminator": span.span_reasoning_terminator,
            "span_turn_end_attested": span.span_turn_end_attested,
            "span_thinking_count": span.span_thinking_count,
            "span_content_count": span.span_content_count,
            "span_tool_call_count": span.span_tool_call_count,
            "span_tool_arguments_count": span.span_tool_arguments_count,
            "span_control_count": span.span_control_count,
            "span_unknown_count": span.span_unknown_count,
            "span_window_start_position": span.span_window_start_position,
            "span_window_end_position": span.span_window_end_position,
            "span_contiguous": span.span_contiguous,
            "span_contains_final_content": span.span_contains_final_content,
            "span_contains_tool_call": span.span_contains_tool_call,
            "span_message_truncated": span.span_message_truncated,
            "accepted_span_start": receipt.accepted_span_start,
            "accepted_span_end": receipt.accepted_span_end,
            "retained_count": receipt.retained_count,
            "accepted_sequence_digest": receipt.accepted_sequence_digest,
            "label_anchor_alpha": receipt.label_anchor_alpha,
            "output_sha256": receipt.output_sha256,
        }))
    }

    /// Evidence projection only. Opaque handles are hashed, never persisted.
    pub fn audit(&self) -> Value {
        let slot = self.source.lock().expect("native source state poisoned");
        let receipt = slot
            .lease
            .as_ref()
            .and_then(SourceCaptureLease::retirement_receipt);
        json!({"session_id":self.session_id(),"runtime_id":self.enrollment.runtime_id,
            "runtime_generation":self.enrollment.runtime_generation,"model_fingerprint":self.enrollment.identity.fingerprint(),
            "source":slot.lease.as_ref().map(|lease|json!({
                "capture_id":lease.intent().capture_id,"source_request_id":lease.intent().source_request_id,"turn_nonce":lease.intent().turn_nonce,
                "capture_mode":lease.capture_mode(),
                "intent_digest":lease.intent().digest().ok(),"state":format!("{:?}",lease.state()),
                "receipt":receipt.as_ref().map(|r|json!({"native_rid":r.native_rid,"response_id":r.response_id,
                    "capture_commit_id":r.capture_commit_id,"handle_sha256":pi_ai::source_capture::sha256(r.source_handle.as_bytes()),
                    "output_sha256":r.output_sha256,"retained_count":r.retained_count,
                    "accepted_span_start":r.accepted_span_start,"accepted_span_end":r.accepted_span_end,"logical_prefix_bytes":r.byte_count,
                    "ready_ranks":r.ready_ranks,"expected_ranks":r.expected_ranks,"layout_digest":r.layout_digest})),
                "tool_claimed":slot.tool_claimed,"handoff_admitted":slot.handoff_admitted,"published_message":slot.published_message,
                "retained":slot.retained,"publication_claimed":slot.publication_claimed,
                "negative_control_armed":slot.negative_control_armed,"negative_control_exported":slot.negative_control_exported,
            })),
            "receiver":self.inbox.lock().expect("native receiver state poisoned").audit(),
            "sender_queue":self.queue.lock().expect("native sender queue poisoned").snapshot_vec().iter().map(|entry|json!({
                "queue_id":entry.queue_id,"message_id":entry.message_id,"receiver":entry.receiver_selector(),
                "state":entry.state.as_str(),"attempts":entry.attempts,
                "enqueued_at_epoch":entry.enqueued_at_epoch,"expires_at_epoch":entry.expires_at_epoch})).collect::<Vec<_>>()})
    }

    pub fn invalidate(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.invalidate_all();
        }
        if let Ok(mut source) = self.source.lock() {
            source.budget = None;
            source.capture_permit = None;
            if let Some(lease) = &source.lease {
                lease.cancel();
            }
        }
        // Readiness and publication authority belong to the shared runtime
        // connector, not this session. Evicting it here splits later senders
        // from live receivers' enrollment and publication records. Only an
        // explicit runtime invalidation may clear the process-wide cache.
        // No restamp may run for dead receiver state.
        if let Ok(mut guard) = self.restamp_unsubscribe.lock() {
            if let Some(unsubscribe) = guard.take() {
                unsubscribe();
            }
        }
        // No queue notice may drain for dead state.
        if let Ok(mut guard) = self.queue_listener_unsubscribe.lock() {
            if let Some(unsubscribe) = guard.take() {
                unsubscribe();
            }
        }
        unregister_sender_session(self);
        self.run_tombstone_cleanup();
    }
    /// Runs the host-side recovery callback exactly once. Called on every
    /// terminal deactivation path so the host's native hooks and handoff
    /// tool never outlive the native session they were installed for.
    fn run_tombstone_cleanup(&self) {
        let cleanup = self
            .tombstone_cleanup
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some(cleanup) = cleanup {
            cleanup();
        }
    }
    /// Recoverable deactivation for a terminally failed source capture
    /// (retry budget exhausted). Like [`Self::invalidate`] it preserves
    /// the process-wide ready cache: the connector handshake stays
    /// valid for other sessions and later re-activation. It marks this
    /// session closed, cancels any residual lease, and runs the registered
    /// tombstone cleanup so the host clears its native hooks and removes
    /// the handoff tool. Subsequent turns of the host session then take
    /// the ordinary inference path instead of failing forever.
    pub fn deactivate(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.invalidate_all();
        }
        if let Ok(mut source) = self.source.lock() {
            source.budget = None;
            source.capture_permit = None;
            if let Some(lease) = &source.lease {
                lease.cancel();
            }
        }
        if let Ok(mut guard) = self.restamp_unsubscribe.lock() {
            if let Some(unsubscribe) = guard.take() {
                unsubscribe();
            }
        }
        if let Ok(mut guard) = self.queue_listener_unsubscribe.lock() {
            if let Some(unsubscribe) = guard.take() {
                unsubscribe();
            }
        }
        unregister_sender_session(self);
        self.run_tombstone_cleanup();
    }
    /// Stores the host's queue-notice listener unsubscribe so teardown can
    /// stop the drain. Called once by the install hook.
    pub fn set_queue_listener_unsubscribe(&self, unsubscribe: Box<dyn Fn() + Send + Sync>) {
        if let Ok(mut guard) = self.queue_listener_unsubscribe.lock() {
            if guard.is_none() {
                *guard = Some(unsubscribe);
            }
        }
    }
    /// Registers the host-side recovery callback run exactly once when this
    /// session's native path is terminally deactivated. The host uses it to
    /// clear its agent hooks and remove the handoff tool.
    pub fn set_tombstone_cleanup(&self, cleanup: Box<dyn Fn() + Send + Sync>) {
        if let Ok(mut slot) = self.tombstone_cleanup.lock() {
            *slot = Some(cleanup);
        }
    }
    /// Failed armed captures observed so far. Exposed for the handoff
    /// tool's retry notice and host diagnostics.
    pub fn failed_capture_attempts(&self) -> u32 {
        self.source
            .lock()
            .map(|slot| slot.failed_attempts)
            .unwrap_or(0)
    }

    // ------------------------------------------------------------------
    // Sender FIFO queue (queued native handoffs)
    // ------------------------------------------------------------------

    /// The session file bound at enrollment (for queue artifact paths).
    pub fn bound_session_file(&self) -> Option<String> {
        self.session_file.get().cloned().flatten()
    }
    /// Journals transition notices into the durable outbox AND the drain
    /// worklist while the caller still holds the queue lock, so the
    /// transition and its notices persist in ONE atomic artifact write.
    /// Lock order (queue -> pending worklist) is the only order used.
    fn journal_notices_locked(
        &self,
        queue: &mut Occ3SenderQueue,
        notices: impl IntoIterator<Item = Occ3QueueNotice>,
    ) {
        for notice in notices {
            if let Some(record) = queue.outbox_record(&notice) {
                if let Ok(mut pending) = self.pending_queue_notices.lock() {
                    if pending.len() < 64 {
                        pending.push(record);
                    }
                }
            }
        }
    }

    /// Queue snapshot (arrival order) plus its blocked reason, if any.
    pub fn sender_queue_snapshot(
        &self,
    ) -> Result<(Vec<super::occ3_sender_queue::Occ3SenderQueueEntry>, Option<Occ3SenderQueueBlocked>), String> {
        let queue = self.queue.lock().map_err(|_| failure())?;
        Ok((queue.snapshot_vec(), queue.blocked().cloned()))
    }

    /// Enqueues one queued handoff. Returns the enqueue notice, or a
    /// content-free refusal reason (`queue_full` / `queue_blocked`).
    pub fn sender_queue_enqueue(&self, receiver_role: &str, receiver_name: &str) -> Result<Occ3QueueNotice, String> {
        self.check()?;
        // One lock, one atomic queue+outbox persist for every outcome -
        // success AND both refusal classes. A refused enqueue is still
        // reported durably: one content-free queue-level refusal notice
        // is journaled, so the parent-side roster (reading the durable
        // `occ3_queue_notice` records) learns the handoff was refused and
        // why - the in-turn tool error alone is not durable and never
        // reaches the parent.
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let refused = if queue.blocked().is_some() {
            Some("queue_blocked")
        } else if queue.len() >= OCC3_SENDER_QUEUE_MAX_ENTRIES {
            Some("queue_full")
        } else {
            None
        };
        if let Some(refused) = refused {
            self.journal_notices_locked(
                &mut queue,
                [super::occ3_sender_queue::queue_refusal_notice(refused, now())],
            );
            // A blocked artifact is never overwritten (byte-for-byte
            // preserved by design), so a queue_blocked refusal cannot be
            // outbox-persisted: the record stays in the memory worklist
            // and the drain still delivers it this run. Every other
            // refusal persists atomically with the queue.
            if queue.blocked().is_none() {
                self.persist_sender_queue(&queue)?;
            }
            return Err(refused.into());
        }
        let notice = queue
            .enqueue(receiver_role, receiver_name, now())
            .ok_or_else(|| "queue_enqueue_refused".to_string())?;
        self.journal_notices_locked(&mut queue, [notice.clone()]);
        self.persist_sender_queue(&queue)?;
        Ok(notice)
    }

    /// Marks the queue head armed (called when the armed handoff's tool call
    /// authorized a receiver that matches the head entry).
    pub(crate) fn sender_queue_mark_armed(
        &self,
        message_id: Option<&str>,
        receiver_session_id: Option<&str>,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.mark_armed(message_id, receiver_session_id, now());
        self.journal_notices_locked(&mut queue, notice.clone());
        self.persist_sender_queue(&queue)?;
        Ok(notice)
    }

    /// Records the daemon-minted message id and the publish/admit
    /// progression for the in-flight head.
    pub(crate) fn sender_queue_mark_progressed(
        &self,
        state: super::occ3_sender_queue::Occ3SenderQueueState,
        message_id: &str,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        if state == super::occ3_sender_queue::Occ3SenderQueueState::Admitted {
            // Defensive cleanup for admission paths that did not pass through
            // publication (normal publication already released this permit).
            // Lock order is source -> queue; never retain either lock across
            // delivery routing, and drop source before acquiring queue here.
            let mut source = self.source.lock().map_err(|_| failure())?;
            if source.handoff_admitted {
                source.capture_permit = None;
            }
        }
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.mark_progressed(state, message_id, now());
        self.journal_notices_locked(&mut queue, notice.clone());
        self.persist_sender_queue(&queue)?;
        Ok(notice)
    }

    /// Receiver-side consumption/failure events for this sender's messages.
    pub(crate) fn sender_queue_record_delivery_event(
        &self,
        message_id: &str,
        consumed: bool,
        receipt_sha256: Option<&str>,
    ) -> Result<Vec<Occ3QueueNotice>, String> {
        self.check()?;
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notices = queue.record_delivery_event(message_id, consumed, receipt_sha256, now());
        self.journal_notices_locked(&mut queue, notices.clone());
        if !notices.is_empty() {
            self.persist_sender_queue(&queue)?;
        }
        Ok(notices)
    }

    /// Retires the completed capture bound to a published message id.
    /// No-op when the slot has no lease, the lease belongs to a different
    /// (later) publication, or the handoff was not admitted: every skip is
    /// a state where retirement would be wrong, not an error to surface.
    /// Called from the async delivery-event routing path on every live
    /// sender registered for the sender session id; the message-id guard
    /// makes it a no-op on every sender except the one that published it.
    pub(crate) async fn retire_capture_for_completed(&self, message_id: &str) {
        let lease = {
            let Ok(slot) = self.source.lock() else { return };
            let bound = slot
                .published_message
                .as_ref()
                .is_some_and(|(id, _)| id == message_id);
            if !bound || !slot.handoff_admitted {
                return;
            }
            match slot.lease.as_ref() {
                Some(lease) if lease.state() == SourceCaptureState::Captured => lease.clone(),
                _ => return,
            }
        };
        if lease.retirement_receipt().is_some() {
            let _ = self.connector.retire_native_source(&lease).await;
        } else {
            lease.cancel();
        }
        if let Ok(mut slot) = self.source.lock() {
            // Same reset the sender's own completion path applies, so a
            // skipped failure here only delays cleanup to the next arm,
            // exactly like today.
            Self::reset_slot_after_completed_handoff(&mut slot);
        }
    }

    /// Receiver-busy refusal for one of this sender's in-flight messages:
    /// the entry returns to `pending` with a backoff for a fresh capture, or
    /// falls back to OCC-1 once the busy retry budget is spent.
    pub(crate) fn sender_queue_record_receiver_busy(
        &self,
        message_id: &str,
    ) -> Result<Vec<Occ3QueueNotice>, String> {
        self.check()?;
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notices = queue.record_receiver_busy(message_id, now());
        self.journal_notices_locked(&mut queue, notices.clone());
        if !notices.is_empty() {
            self.persist_sender_queue(&queue)?;
        }
        Ok(notices)
    }

    /// A pre-admission busy refusal of the armed head's source capture.
    /// Returns `true` when an armed head absorbed it (busy retry scheduled
    /// or busy budget exhausted to OCC-1 fallback); `false` when there was
    /// no armed head, so the caller counts an ordinary capture failure.
    /// Never touches the `failed_attempts` capture-failure budget.
    pub(crate) fn sender_queue_source_busy(
        &self,
        refusal: pi_ai::source_capture::SourceBusyRefusal,
    ) -> Result<bool, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let Some(notice) =
            queue.record_source_busy(refusal.retry_after_secs, refusal.refused_at_epoch, now())
        else {
            return Ok(false);
        };
        self.journal_notices_locked(&mut queue, Some(notice));
        // The in-memory transition already happened, so the refusal is
        // absorbed even if persisting fails; reporting false here would
        // also spend the capture-failure budget for the same refusal.
        // A later queue write persists the same state.
        let _ = self.persist_sender_queue(&queue);
        Ok(true)
    }

    /// Absorbs a pre-admission source busy refusal left in the slot: the
    /// slot is resolved, not admitted, and its lease is `Cancelled` with a
    /// busy refusal record (set only by pi-ai on a recognised 503 busy
    /// code). The armed queue head takes one busy retry (or falls back once
    /// the busy budget is spent) and the slot resets for a fresh arm; the
    /// refused lease was never published, so it is dropped, never retired
    /// or replayed. Returns false when the slot holds no such refusal or no
    /// queue head is armed; the resolve hook then counts an ordinary capture
    /// failure. Terminal captures release their capacity even without a busy
    /// refusal to absorb. Called from the resolve hook's
    /// tombstone path and, eagerly, at AgentEnd so the outcome is recorded
    /// even when no automatic provider retry follows.
    pub(crate) fn absorb_source_busy_refusal(&self) -> bool {
        // Check and reset under ONE slot lock so a concurrent resolve cannot
        // see the refused lease after the queue absorbed it and count the
        // same refusal as a capture failure. Lock order is source -> queue,
        // the same order the audit snapshot uses.
        let Ok(mut slot) = self.source.lock() else {
            return false;
        };
        if !slot.resolved || slot.handoff_admitted {
            return false;
        }
        // AgentEnd also visits ordinary failed captures. They still count
        // against the normal retry budget on the next resolve, but must not
        // retain process-wide capacity until that next turn. Busy retries also
        // release capacity deliberately: other sessions may capture during the
        // backoff, and the next explicit arm must acquire a fresh permit.
        if Self::capture_capacity_releasable(slot.lease.as_ref().map(|lease| lease.state())) {
            slot.capture_permit = None;
        }
        let Some(refusal) = slot.lease.as_ref().and_then(|lease| {
            (lease.state() == SourceCaptureState::Cancelled).then(|| lease.busy_refusal()).flatten()
        }) else {
            return false;
        };
        if !self.sender_queue_source_busy(refusal).unwrap_or(false) {
            return false;
        }
        Self::reset_slot_for_fresh_arm(&mut slot);
        true
    }

    /// Terminal leases cannot retain capture capacity between turns.
    fn capture_capacity_releasable(state: Option<SourceCaptureState>) -> bool {
        !matches!(state, Some(SourceCaptureState::Prepared | SourceCaptureState::Submitted | SourceCaptureState::Captured))
    }

    /// Slot reset after a completed (admitted, retired) handoff. Also
    /// releases the receiver binding: before this, a queued handoff to a
    /// different receiver in the same session was refused with "receiver
    /// changed after capture was armed" (pre-existing). The binding stays
    /// while a budget is still armed, because that budget belongs to the
    /// bound receiver.
    fn reset_slot_after_completed_handoff(slot: &mut SourceSlot) {
        slot.resolved = false;
        slot.lease = None;
        slot.tool_claimed = false;
        slot.publication_claimed = false;
        slot.handoff_admitted = false;
        slot.published_message = None;
        slot.capture_permit = None;
        if slot.budget.is_none() {
            slot.armed_receiver = None;
        }
    }

    /// Resets the capture slot after a capture ended without delivery
    /// (busy refusal or ordinary failure) so the next arm is fresh. Clears
    /// the receiver binding too: no lease or budget remains for it.
    fn reset_slot_for_fresh_arm(slot: &mut SourceSlot) {
        slot.resolved = false;
        slot.lease = None;
        slot.budget = None;
        slot.capture_permit = None;
        slot.armed_receiver = None;
    }

    /// Requeues a policy-denied arm without spending a capture attempt.
    /// Queue transition, outbox journal, and persistence share the queue lock.
    fn sender_queue_capture_policy_denied(&self) -> Result<Option<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.capture_policy_denied(now());
        self.journal_notices_locked(&mut queue, notice.clone());
        self.persist_sender_queue(&queue)?;
        Ok(notice)
    }

    /// A capture failure for the armed head. `terminal` marks the native
    /// path's death for this session (PR235 budget exhausted); the queue
    /// demotes the entry to OCC-1 fallback in that case only. Non-terminal
    /// failures return the entry to `pending` for the model's explicit
    /// re-arm and NEVER touch the PR235 `failed_attempts` budget here -
    /// the resolve-hook tombstone path owns that counter.
    pub(crate) fn sender_queue_capture_failed(
        &self,
        terminal: bool,
        reason: &str,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.capture_failed(terminal, reason, now());
        self.journal_notices_locked(&mut queue, notice.clone());
        self.persist_sender_queue(&queue)?;
        Ok(notice)
    }

    /// Expired pending entries drop to fallback notices.
    pub(crate) fn sender_queue_sweep_expired(&self) -> Result<Vec<Occ3QueueNotice>, String> {
        self.sweep_expired_at_clock(now())
    }

    /// The single live-sweep path (lock, sweep, atomic queue+outbox
    /// journal, one persist) parameterized only by the clock: production
    /// delegates with the current epoch; the tests drive the SAME body
    /// with the explicit clock a 30s timer tick would hold, so the
    /// exercised persistence path is exactly the production one.
    fn sweep_expired_at_clock(&self, now_epoch: u64) -> Result<Vec<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notices = queue.sweep_expired(now_epoch);
        self.journal_notices_locked(&mut queue, notices.clone());
        if !notices.is_empty() {
            self.persist_sender_queue(&queue)?;
        }
        Ok(notices)
    }

    /// Test-only explicit-clock entry into the SHARED sweep body above:
    /// no duplicated implementation, no global clock mutation. Production
    /// never calls this.
    #[cfg(test)]
    pub(crate) fn sender_queue_sweep_expired_at(
        &self,
        now_epoch: u64,
    ) -> Result<Vec<Occ3QueueNotice>, String> {
        self.sweep_expired_at_clock(now_epoch)
    }

    /// Acknowledges the ordinary OCC-1 re-emission of a fallback_pending
    /// entry (receipt delivered|queued). ONLY this emits the terminal
    /// `fallback` notice and pops the entry.
    pub fn sender_queue_acknowledge_fallback(
        &self,
        queue_id: &str,
        dispatch_token: Option<&str>,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.acknowledge_fallback(queue_id, dispatch_token, now());
        self.journal_notices_locked(&mut queue, notice.clone());
        if notice.is_some() {
            self.persist_sender_queue(&queue)?;
        }
        Ok(notice)
    }

    /// F3: EXPLICIT manual resolution of an uncertain entry (never
    /// automatic; uncertain entries stay visible until a human-driven
    /// host action decides). `reemit=true` re-arms the ordinary
    /// re-emission path with a fresh dispatch token (the human accepts
    /// the duplication risk); `reemit=false` abandons the handoff as
    /// terminal `failed` (`manual_cancel`) with no re-send.
    pub fn sender_queue_resolve_uncertain(
        &self,
        queue_id: &str,
        reemit: bool,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        let notice = queue.resolve_uncertain(queue_id, reemit, now())?;
        self.journal_notices_locked(&mut queue, notice.clone());
        if notice.is_some() {
            self.persist_sender_queue(&queue)?;
        }
        Ok(notice)
    }

    /// Records that the HOST queued a fallback re-emission prompt for this
    /// entry. Called only by the session's own queue listener; the issue is
    /// consumed one-shot by the prompt's MessageEnd.
    pub fn issue_fallback_prompt(&self, queue_id: &str) {
        if let Ok(mut issues) = self.pending_fallback_prompt_issues.lock() {
            if !issues.iter().any(|pending| pending == queue_id) {
                issues.push(queue_id.to_string());
                let overflow = issues.len().saturating_sub(8);
                issues.drain(0..overflow);
            }
        }
    }

    /// The fallback prompt's MessageEnd: consumes one outstanding host
    /// issue. Returns true ONLY when the host actually queued this entry's
    /// fallback prompt, the issue is still outstanding, and the entry is
    /// still `fallback_pending` - a spoofed, stale or replayed
    /// `occ3_fallback_prompt` message (or a marker for an entry that has
    /// since closed) matches nothing and arms nothing.
    pub fn note_fallback_prompt_message(&self, queue_id: &str) -> bool {
        let Ok(mut issues) = self.pending_fallback_prompt_issues.lock() else {
            return false;
        };
        let Some(position) = issues.iter().position(|pending| pending == queue_id) else {
            return false;
        };
        let still_pending = {
            let Ok(queue) = self.queue.lock() else {
                return false;
            };
            queue.snapshot_vec().iter().any(|entry| {
                entry.queue_id == queue_id
                    && entry.state == super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
            })
        };
        if still_pending {
            issues.remove(position);
        }
        still_pending
    }

    /// Marks the ACTIVE fallback re-emission turn for one entry. Called by
    /// the host listener ONLY after [`Self::note_fallback_prompt_message`]
    /// accepted the host-issued marker, with the entry's own queue id (the
    /// receiver authority is re-read from the entry at send time, never
    /// from this marker).
    pub fn begin_fallback_turn(&self, queue_id: &str) {
        if let Ok(mut active) = self.active_fallback_turn.lock() {
            *active = Some(queue_id.to_string());
        }
    }

    /// Clears the active fallback turn. Called at TurnEnd.
    pub fn end_fallback_turn(&self) {
        if let Ok(mut active) = self.active_fallback_turn.lock() {
            *active = None;
        }
    }

    /// The dispatch token for the ACTIVE fallback turn's entry, but only
    /// when the send target matches that entry's receiver AND the entry is
    /// still fallback_pending. This is the ONLY token source: unrelated
    /// ordinary sends (any other turn, or a different receiver) get None.
    pub fn active_fallback_token_for(&self, target: &str) -> Option<String> {
        let queue_id = self.active_fallback_turn.lock().ok()?.clone()?;
        let queue = self.queue.lock().ok()?;
        let entry = queue
            .snapshot_vec()
            .into_iter()
            .find(|entry| entry.queue_id == queue_id)?;
        if entry.state != super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending {
            return None;
        }
        // The send target must resolve to the ENTRY's own identity: the
        // persisted receiver selector, or the AUTHENTICATED receiver
        // session id captured at arm time from the authorization (never
        // model-supplied). Any other receiver - including a different
        // agent that merely shares the display name - yields no token.
        let selector_matches = target == entry.receiver_selector();
        let session_id_matches = entry
            .receiver_session_id
            .as_deref()
            .is_some_and(|session_id| target == session_id);
        if !selector_matches && !session_id_matches {
            return None;
        }
        entry.fallback_dispatch_token.clone()
    }

    /// Ordinary-send acknowledgment hook (production fallback close): the
    /// host calls this after this session's ordinary `agent_message.send`
    /// completed. Matches the OLDEST fallback_pending entry whose receiver
    /// matches the send target (by name or `role:name`), and:
    /// * receipt `delivered`|`queued` (ordinary delivery accepted) =>
    ///   terminal `fallback` notice carrying the honest status;
    /// * anything else => a `fallback_pending` notice carrying the honest
    ///   status (no false terminal; the entry stays for retry/expiry).
    /// Returns the notice(s) to journal.
    pub fn sender_queue_note_ordinary_send(
        &self,
        dispatch_token: Option<&str>,
        receipt_status: Option<&str>,
        receipt_message_id: Option<&str>,
    ) -> Result<Vec<Occ3QueueNotice>, String> {
        let mut queue = self.queue.lock().map_err(|_| failure())?;
        // F2: correlation is by the trusted dispatch token only - an
        // unrelated ordinary message to the same receiver can never
        // acknowledge this entry. The token was issued when the entry
        // entered fallback_pending and returned to the host with the
        // fallback_pending notice; the host passes it back with the
        // receipt.
        let Some(token) = dispatch_token else {
            return Ok(Vec::new());
        };
        let snapshot = queue.snapshot_vec();
        let Some(index) = snapshot.iter().position(|entry| {
            entry.state == super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
                && entry.fallback_dispatch_token.as_deref() == Some(token)
        }) else {
            return Ok(Vec::new());
        };
        let accepted = matches!(receipt_status, Some("delivered") | Some("queued"));
        let status_note = format!(
            "ordinary_delivery:{}",
            receipt_status.unwrap_or("unknown")
        );
        if accepted {
            let queue_id = snapshot[index].queue_id.clone();
            match queue.acknowledge_fallback(&queue_id, Some(token), now()) {
                Some(mut notice) => {
                    notice.reason = Some(status_note);
                    // The terminal notice carries the ACTUAL ordinary
                    // re-emission's message id (the receipt's), never the
                    // retired capture attempt's id.
                    if let Some(id) = receipt_message_id {
                        notice.message_id = Some(id.to_string());
                    }
                    // One lock, one atomic queue+outbox persist. A failed
                    // persist is SURFACED (Err) - never a successful
                    // return with unsaved state; the record stays in the
                    // worklist and the durable-retry requeue heals it.
                    self.journal_notices_locked(&mut queue, [notice.clone()]);
                    self.persist_sender_queue(&queue)?;
                    Ok(vec![notice])
                }
                None => Ok(Vec::new()),
            }
        } else {
            // Honest non-terminal report: the ordinary send did not land.
            // F3: the transport status stays distinct in the reason; the
            // entry remains fallback_pending with the same live token.
            let entry = &snapshot[index];
            let notice = Occ3QueueNotice {
                queue_id: entry.queue_id.clone(),
                message_id: entry.message_id.clone(),
                receiver: entry.receiver_selector(),
                state: "fallback_pending",
                attempts: entry.attempts,
                position: (index + 1) as u32,
                timestamp: now(),
                reason: Some(status_note),
                dispatch_token: entry.fallback_dispatch_token.clone(),
                refusal: None,
            };
            self.journal_notices_locked(&mut queue, [notice.clone()]);
            self.persist_sender_queue(&queue)?;
            Ok(vec![notice])
        }
    }

    /// The snapshot-form notice details for the current queue (one record
    /// carrying the full state; the visibility projection's base).
    pub fn sender_queue_snapshot_details(&self) -> Result<Value, String> {
        let queue = self.queue.lock().map_err(|_| failure())?;
        Ok(queue.snapshot_details())
    }

    /// Journals one notice for the host's next drain AND records it into
    /// the durable outbox: the notice is persisted atomically with the
    /// queue (same artifact file), carries a stable `occ3n_` id for
    /// consumer dedup, and is removed only after BOTH deliveries (the
    /// sender-transcript record and the parent notification) are
    /// acknowledged. A crash anywhere before the acks replays the record
    /// from the outbox on restart - a notice is never silently dropped.
    /// Bounded: at the outbox cap the record is refused a slot and the
    /// persisted overflow flag flips (visible fail, surfaced in
    /// `sender_queue_snapshot_details`), never a silent drop.
    /// Must NOT be called while holding the queue mutex.
    pub(crate) fn record_queue_notice(&self, notice: Occ3QueueNotice) {
        if let Ok(mut queue) = self.queue.lock() {
            let recorded = queue.outbox_record(&notice);
            if recorded.is_some() || queue.outbox_overflow() {
                let _ = self.persist_sender_queue(&queue);
            }
            if let Some(record) = recorded {
                if let Ok(mut pending) = self.pending_queue_notices.lock() {
                    if pending.len() < 64 {
                        pending.push(record);
                    }
                }
            }
        }
    }

    /// Drains journaled notices for the host's next drain. Each record
    /// still lives in the durable outbox until both deliveries are
    /// acknowledged via the ack methods below - the journal is only the
    /// fast-path worklist.
    pub fn take_pending_queue_notices(
        &self,
    ) -> Vec<super::occ3_sender_queue::Occ3DurableQueueNotice> {
        self.pending_queue_notices
            .lock()
            .map(|mut pending| std::mem::take(&mut *pending))
            .unwrap_or_default()
    }

    /// The durable outbox, not a cached worklist clone, owns acknowledgment.
    pub(super) fn current_queue_notice(
        &self,
        notice_id: &str,
    ) -> Option<super::occ3_sender_queue::Occ3DurableQueueNotice> {
        self.queue.lock().ok()?.outbox_pending().into_iter()
            .find(|record| record.notice_id == notice_id)
    }

    /// Re-selects every outbox record not yet acknowledged by BOTH
    /// deliveries into the drain worklist (dedup by stable notice id, so
    /// a record is never queued twice). Called at the END of every drain
    /// cycle: a transient transcript-write or parent-notification failure
    /// (or a skipped worklist push) is healed by the NEXT cycle - the
    /// next turn end or the 30s idle timer - in the SAME process, with
    /// no daemon restart. The durable outbox is the source of truth; the
    /// worklist is only its cache.
    pub fn requeue_unacked_queue_notices(&self) {
        let pending_records = self
            .queue
            .lock()
            .map(|queue| queue.outbox_pending())
            .unwrap_or_default();
        if pending_records.is_empty() {
            return;
        }
        if let Ok(mut pending) = self.pending_queue_notices.lock() {
            for record in pending_records {
                if !pending
                    .iter()
                    .any(|queued: &super::occ3_sender_queue::Occ3DurableQueueNotice| {
                        queued.notice_id == record.notice_id
                    })
                {
                    if pending.len() < 64 {
                        pending.push(record);
                    }
                }
            }
        }
        // A poisoned worklist lock is not a loss: the records stay in the
        // durable outbox and the next cycle re-selects them here again.
    }

    /// Silently retires this sender's whole queue when its owner session
    /// is deleted: no entry stays live, no outbox record or worklist item
    /// survives, so no later drain (turn end or idle timer) can notify
    /// the parent about a session that no longer exists. Persists the
    /// emptied queue so a restart replays nothing either.
    pub fn retire_sender_queue_silently(&self) -> (usize, usize) {
        let removed = match self.queue.lock() {
            Ok(mut queue) => {
                let removed = queue.retire_all_silently();
                let _ = self.persist_sender_queue(&queue);
                removed
            }
            Err(_) => (0, 0),
        };
        if let Ok(mut pending) = self.pending_queue_notices.lock() {
            pending.clear();
        }
        removed
    }

    /// Acknowledges the sender-transcript delivery of one durable notice
    /// (the custom message was appended and flushed). Persists the outbox
    /// state; the record is removed once the parent delivery is also
    /// acknowledged.
    pub fn sender_queue_notice_ack_transcript(&self, notice_id: &str) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.outbox_ack_transcript(notice_id);
            let _ = self.persist_sender_queue(&queue);
        }
    }

    /// Acknowledges the parent notification of one durable notice.
    /// Persists the outbox state; the record is removed when both
    /// deliveries are acknowledged.
    pub fn sender_queue_notice_ack_parent(&self, notice_id: &str) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.outbox_ack_parent(notice_id);
            let _ = self.persist_sender_queue(&queue);
        }
    }

    /// Content-free projection of the durable notice outbox (pending
    /// notices with delivery flags) for read-only visibility surfaces.
    /// Never contains dispatch tokens, handles or receipts.
    pub fn sender_queue_outbox_snapshot(&self) -> Vec<Value> {
        self.queue
            .lock()
            .map(|queue| queue.outbox_snapshot())
            .unwrap_or_default()
    }

    /// Loads the persisted queue artifact. A blocked load preserves the
    /// file untouched and returns the blocked notice for the host to emit;
    /// the queue refuses all enqueue/arm actions while blocked.
    pub fn load_sender_queue(
        self: &Arc<Self>,
        session_file: Option<&str>,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let Some(session_file) = session_file else { return Ok(None) };
        let path = super::occ3_sender_queue::sender_queue_path(session_file);
        let (mut queue, outcome) = Occ3SenderQueue::load(&path, std::env::var("OCC3_SENDER_QUEUE_TTL_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(super::occ3_sender_queue::OCC3_SENDER_QUEUE_TTL_SECS));
        let notice = match &outcome {
            Occ3SenderQueueLoad::Fresh => None,
            Occ3SenderQueueLoad::Loaded(entries, expired_notices) => {
                // Restart expiry is never a silent drop: each entry that
                // expired on disk is recorded into the LOADED queue's
                // durable outbox BEFORE the load is stored, so the expiry
                // notices persist atomically with the new queue state
                // (never into the previous in-memory queue, never
                // clobbered by the load's own persist). The sender - and
                // the parent-side roster reading the durable
                // `occ3_queue_notice` records - learn of the loss on
                // resume, exactly like a live TTL sweep would have
                // reported it.
                for expired in expired_notices {
                    queue.outbox_record(expired);
                }
                if entries.is_empty() {
                    None
                } else {
                // Restart demotions already applied inside `load`: in-flight
                // captures were marked ambiguously failed (attempts+1, no
                // replay); entries with remaining PR235 budget returned to
                // `pending` for a fresh explicit arm, exhausted ones to
                // `fallback_pending`; expired entries dropped as terminal
                // `expired` (with their own outbox records above).
                // Surface one aggregate fallback_pending notice per
                // exhausted entry so the model is prompted for ordinary
                // re-emission (fresh-arm entries are prompted by the advance
                // prompt path instead).
                let now = now();
                let mut last = None;
                for (index, entry) in entries.iter().enumerate() {
                    if entry.state
                        == super::occ3_sender_queue::Occ3SenderQueueState::FallbackPending
                    {
                        last = Some(Occ3QueueNotice {
                            queue_id: entry.queue_id.clone(),
                            message_id: entry.message_id.clone(),
                            receiver: entry.receiver_selector(),
                            state: "fallback_pending",
                            attempts: entry.attempts,
                            position: (index + 1) as u32,
                            timestamp: now,
                            reason: entry.reason.clone(),
                            dispatch_token: entry.fallback_dispatch_token.clone(),
                            refusal: None,
                        });
                    }
                }
                // The aggregate rides the same atomic load persist.
                if let Some(last) = &last {
                    queue.outbox_record(last);
                }
                last
                }
            }
            Occ3SenderQueueLoad::Blocked(blocked) => Some(Occ3QueueNotice {
                queue_id: "blocked".into(),
                message_id: None,
                receiver: "unknown".into(),
                state: "failed",
                attempts: 0,
                position: 0,
                timestamp: now(),
                refusal: None,
                reason: Some(match blocked {
                    Occ3SenderQueueBlocked::UnsupportedSchema { found } => {
                        format!("queue_artifact_unsupported_schema:{found}")
                    }
                    Occ3SenderQueueBlocked::Unreadable { .. } => {
                        "queue_artifact_unreadable".into()
                    }
                    Occ3SenderQueueBlocked::UnreadableIo { .. } => {
                        "queue_artifact_unreadable_io".into()
                    }
                }),
                dispatch_token: None,
            }),
        };
        // One atomic persist: the loaded queue state AND its outbox
        // (expiry + aggregate records) in a single artifact write. A
        // blocked load preserves the file untouched, and a fresh load has
        // nothing to record, so neither persists.
        if queue.blocked().is_none() && matches!(outcome, Occ3SenderQueueLoad::Loaded(..)) {
            self.persist_sender_queue(&queue)?;
        }
        *self.queue.lock().map_err(|_| failure())? = queue;
        // Restart replay: durable outbox records that were never fully
        // acknowledged (either delivery) re-enter the drain journal, so a
        // crash between the queue rewrite and the transcript flush (or
        // before the parent ack) replays the notice instead of dropping
        // it. Consumers dedup by the stable noticeId.
        for record in self
            .queue
            .lock()
            .map(|queue| queue.outbox_pending())
            .unwrap_or_default()
        {
            if let Ok(mut pending) = self.pending_queue_notices.lock() {
                if pending.len() < 64 {
                    pending.push(record);
                }
            }
        }
        Ok(notice)
    }

    /// Persists the queue artifact, returning persistence failures to callers.
    /// Best-effort callers ignore them; the in-memory queue stays authoritative.
    fn persist_sender_queue(&self, queue: &Occ3SenderQueue) -> Result<(), String> {
        let Some(session_file) = self.session_file.get().cloned().flatten() else {
            return Ok(());
        };
        let path = super::occ3_sender_queue::sender_queue_path(&session_file);
        queue.persist(&path)
    }

    /// Slot reset after a COMPLETED handoff: retires the lease through the
    /// connector retirement path and clears the per-handoff slot fields so
    /// the next arm is legal. Only valid on a slot whose handoff was
    /// admitted with a Captured lease; never touches the tombstone retry
    /// path or the retry budget. Used by the sender queue when an entry
    /// reaches a terminal state.
    pub async fn complete_handoff_slot(self: &Arc<Self>) -> Result<(), String> {
        self.check()?;
        let lease = {
            let slot = self.source.lock().map_err(|_| failure())?;
            let lease = slot.lease.clone().ok_or_else(|| format!("{} [occ3_slot_no_lease]", failure()))?;
            if !slot.handoff_admitted
                || lease.state() != SourceCaptureState::Captured
            {
                return Err(format!("{} [occ3_slot_not_complete]", failure()));
            }
            lease
        };
        // Retire the completed capture server-side first (best-effort on
        // failure: the capture is spent either way - a failed retirement
        // never blocks the slot reset, mirroring `close()` semantics).
        if lease.retirement_receipt().is_some() {
            let _ = self.connector.retire_native_source(&lease).await;
        } else {
            lease.cancel();
        }
        let mut slot = self.source.lock().map_err(|_| failure())?;
        Self::reset_slot_after_completed_handoff(&mut slot);
        Ok(())
    }

    /// Revocation stays in memory; best-effort runtime cleanup is awaited by the
    /// isolated controller. It never restarts the model or changes settings.
    pub async fn close(&self) -> Result<(), String> {
        self.invalidate();
        let source = self.source.lock().map_err(|_| failure())?.lease.clone();
        // Copy-only facts so the closure stays unwind-safe; the lease
        // itself never crosses the boundary.
        let had_lease = source.is_some();
        let retirable = source
            .as_ref()
            .is_some_and(|lease| lease.retirement_receipt().is_some());
        let _ = std::panic::catch_unwind(|| {
            // Content-free visibility for teardown leaks: without this,
            // a close that never reaches the server is indistinguishable
            // from a session that never closed.
            pi_ai::log::get_logger("occ3.sender").warn(
                "native_session_closed",
                Some(serde_json::Map::from_iter([
                    ("had_lease".into(), Value::from(had_lease)),
                    ("retirable".into(), Value::from(retirable)),
                ])),
            );
        });
        if let Some(source) = source {
            if source.retirement_receipt().is_some() {
                self.connector
                    .retire_native_source(&source)
                    .await
                    .map_err(|_| failure())?;
            } else {
                source.cancel();
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_messages::{
        AgentFamilyCatalogEntry, AgentSessionMessageEndpoint, AuthorizedNativeHandoff,
        OCC3_DELIVERY_PENDING,
    };
    use crate::core::occ::occ3::connector::{
        CaptureRequest, ConsumeRequest, NativePublication,
    };
    use crate::core::occ::occ3::types::{
        Occ3ConnectorCapabilities, Occ3Envelope, Occ3Error, Occ3HandleRef, Occ3Method,
        Occ3ModelIdentity, Occ3TransferMetrics,
    };
    use pi_agent_core::types::{CustomMessageContent, InferenceStateRequest};
    use pi_ai::inference_attachment::{InferenceAttachment, InferenceStateLease};
    use pi_ai::source_capture::{
        SourceCaptureIntent, SourceCaptureProfile, SOURCE_CAPTURE_SCHEMA, SOURCE_SPAN_POLICY,
    };
    use std::time::Duration;

    const METHOD: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
    const RAW_HANDLE: &str = "unit-fixture-raw-source-handle";

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

    struct FixtureConnector {
        identity: Occ3ModelIdentity,
        profile: SourceCaptureProfile,
        /// Scripted `prepare_inference` refusals, returned in order before
        /// the real lease path (the mock connector's fault-table style).
        /// Used to reproduce a live pre-dispatch refusal that provably
        /// dispatched nothing.
        prepare_refusals: Arc<Mutex<Vec<Occ3Error>>>,
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
                prepare_refusals: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// Script the next `prepare_inference` calls to fail with these
        /// errors, in order, before the real lease path runs.
        fn refuse_prepare_with(&self, errors: &[Occ3Error]) {
            *self.prepare_refusals.lock().unwrap() = errors.to_vec();
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
        async fn prepare_inference(
            &self,
            request: PrepareInferenceRequest,
        ) -> Result<InferenceStateLease, Occ3Error> {
            let refusal = {
                let mut scripted = self.prepare_refusals.lock().unwrap();
                (!scripted.is_empty()).then(|| scripted.remove(0))
            };
            if let Some(error) = refusal {
                return Err(error);
            }
            // Real-attachment fixture: binds the exact envelope the receiver
            // inbox claimed, mirroring the daemon wiring fixture, so the
            // receiver resolver's bind path is exercised end to end.
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
                    reason: "unit-fixture-inference-lease".into(),
                }
            })
        }
        async fn prepare_source_capture(
            &self,
            session_id: &str,
            target: InferenceStateTarget,
            retained_count_limit: u32,
            max_capture_bytes: u64,
            limits: SourceCaptureLimits,
            _arming: Option<SourceSpanArming>,
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
            "unit-session",
            &fixture_model(),
            Arc::new(FixtureConnector::new()),
        )
        .await
        .expect("fixture enrollment")
    }

    struct ArmingController(bool);
    impl AgentSessionMessageController for ArmingController {
        fn authorize_native_handoff(&self, role: String, name: String)
            -> pi_ai::types::BoxFuture<Result<AuthorizedNativeHandoff, String>> {
            let allowed = self.0;
            Box::pin(async move {
                if !allowed || role != "child" || name.is_empty() || name.len() > 128 {
                    return Err("Native receiver permit not installed".into());
                }
                AuthorizedNativeHandoff::from_catalog(
                    &AgentFamilyCatalogEntry { id: "unit-session".into(), depth: 0., ..Default::default() },
                    &AgentFamilyCatalogEntry { id: "receiver".into(), depth: 1., parent_session_id: Some("unit-session".into()), ..Default::default() },
                    AgentSessionMessageEndpoint { session_id: "unit-session".into(), active_session_id: "unit-session".into(), ..Default::default() },
                    AgentSessionMessageEndpoint { session_id: "receiver".into(), active_session_id: "receiver".into(), ..Default::default() },
                    &role,
                )
            })
        }
        fn roster(&self) -> pi_ai::types::BoxFuture<Result<crate::core::agent_messages::AgentFamilyRosterResult, String>> {
            Box::pin(async { Err("unused".into()) })
        }
        fn await_pending_child_publication(&self, _: String) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
            Box::pin(async { Ok(None) })
        }
        fn send_agent_message(&self, _: crate::core::agent_messages::AgentSessionMessageSendInput)
            -> pi_ai::types::BoxFuture<Result<crate::core::agent_messages::AgentSessionMessageReceipt, String>> {
            Box::pin(async { Err("ordinary messages must not be used".into()) })
        }
    }

    #[tokio::test]
    async fn normal_handoff_arms_once_only_after_receiver_authorization() {
        for allowed in [false, true] {
            let session = enrolled().await;
            let tool = session.handoff_tool(Arc::new(ArmingController(allowed)));
            let result = (tool.execute)("prepare-call".into(), json!({"receiver_role":"child","receiver_name":"receiver"}), None, None).await;
            assert_eq!(result.is_ok(), allowed);
            if let Ok(result) = result {
                assert_eq!(result.details["status"], "capture_armed");
                assert_eq!(result.details["delivery_completed"], false);
            }
            let slot = session.source.lock().unwrap();
            assert_eq!(slot.budget.is_some(), allowed);
            assert!(slot.lease.is_none());
            assert!(!slot.handoff_admitted);
            if let Some(budget) = &slot.budget {
                assert_eq!(budget.max_capture_bytes, 1024);
                assert_eq!(budget.retained_count_limit, 64);
            }
        }
    }

    /// Pins defensive admission cleanup with a deterministic one-slot pool.
    /// The retained-state provider test covers the real publication path.
    #[tokio::test]
    async fn occ3_admitted_sender_releases_capture_slot_for_another_session() {
        let first = enrolled().await;
        let second = enrolled().await;
        // A private one-slot pool makes the process-wide ownership invariant
        // deterministic without contending with unrelated parallel tests.
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = pool.clone().acquire_owned().await.unwrap();
        assert!(first.arm_handoff_if_needed("receiver", Some(permit)).unwrap());
        assert!(pool.clone().try_acquire_owned().is_err(), "armed capture stays exclusive");
        // Resolve the armed turn; the completing second call reuses its
        // lease and never reacquires the capacity held by that lease.
        resolve_capture(&first).await;
        assert!(!first.arm_handoff_if_needed("receiver", None).unwrap());
        first.sender_queue_enqueue("child", "receiver").unwrap();
        first.sender_queue_mark_armed(Some("admitted-slot"), Some("receiver")).unwrap();
        first.source.lock().unwrap().handoff_admitted = true;
        first.sender_queue_mark_progressed(
            super::super::occ3_sender_queue::Occ3SenderQueueState::Admitted,
            "admitted-slot",
        ).unwrap();
        let next = pool.clone().try_acquire_owned()
            .expect("admitted sender must release capture capacity without teardown or another tool call");
        assert!(second.arm_handoff_if_needed("another-receiver", Some(next)).unwrap());
        assert!(first.source.lock().unwrap().handoff_admitted);
    }

    /// Failed completion cannot leave a reusable lease after releasing capacity.
    #[tokio::test]
    async fn occ3_failed_completion_tombstones_capture_before_releasing_capacity() {
        let session = enrolled().await;
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = pool.clone().acquire_owned().await.unwrap();
        assert!(session.arm_handoff_if_needed("receiver", Some(permit)).unwrap());
        resolve_capture(&session).await;
        let lease = session.source.lock().unwrap().lease.clone().unwrap();
        let hooks = session.source_hooks();
        assert!((hooks.completed)(pi_agent_core::types::SourceCaptureCompletion {
            model: fixture_model(),
            lease,
            assistant_message: Default::default(),
        }).await.is_err(), "prepared lease cannot complete");
        assert_eq!(pool.available_permits(), 1, "failed capture releases capacity");
        {
            let slot = session.source.lock().unwrap();
            assert_eq!(slot.lease.as_ref().unwrap().state(), SourceCaptureState::Cancelled,
                "failed completion must leave a terminal lease, not reusable capture authority");
            assert!(slot.budget.is_none());
            assert!(slot.resolved, "failure accounting remains pending");
        }
        let terminal_error = session.arm_handoff_if_needed("receiver", None)
            .expect_err("no reuse before accounting");
        assert!(terminal_error.contains("[occ3_capture_terminal_pending_reset]"));
        assert!(terminal_error.contains("next turn"));
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1, "count each failure once");
        let next = pool.clone().try_acquire_owned().unwrap();
        assert!(session.arm_handoff_if_needed("receiver", Some(next)).unwrap());
        assert_eq!(pool.available_permits(), 0, "explicit fresh arm holds capacity");
    }

    #[tokio::test]
    async fn repeated_handoff_after_admission_reports_already_delivered() {
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        session
            .sender_queue_mark_armed(Some("agentmsg_admitted"), Some("receiver"))
            .unwrap();
        session
            .sender_queue_mark_progressed(
                super::super::occ3_sender_queue::Occ3SenderQueueState::Admitted,
                "agentmsg_admitted",
            )
            .unwrap();
        session.source.lock().unwrap().handoff_admitted = true;
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let result = (tool.execute)(
            "repeat-call".into(),
            json!({"receiver_role":"child","receiver_name":"receiver"}),
            None,
            None,
        )
        .await
        .expect("an admitted entry answers a repeat without error");
        assert_eq!(result.details["status"], "already_delivered");
        assert_eq!(result.details["message_id"], "agentmsg_admitted");
        let slot = session.source.lock().unwrap();
        assert!(slot.budget.is_none(), "no re-arm after admission");
        assert!(slot.handoff_admitted);
        drop(slot);
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries.len(), 1, "no new queue entry for the repeat");
        assert_eq!(
            entries[0].state,
            super::super::occ3_sender_queue::Occ3SenderQueueState::Admitted
        );
    }

    #[tokio::test]
    async fn failed_capture_resets_for_retry_and_reports_attempts() {
        let session = enrolled().await;
        // Arm, then simulate the armed turn's failure: the resolve hook
        // took the budget and resolved the slot, but no admitted capture
        // survived (stalled stream -> cancelled lease).
        assert!(session.arm_source(budget()).is_ok());
        {
            let mut slot = session.source.lock().unwrap();
            let spent = slot.budget.take();
            assert!(spent.is_some());
            slot.resolved = true;
        }
        let hooks = session.source_hooks();
        let request = || InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
            newly_supplied_messages: Vec::new(),
            signal: None,
        };
        // First tombstone: counted as failure 1, slot resets for a re-arm,
        // and the turn is allowed to run uncaptured instead of failing.
        let resolved = (hooks.resolve)(request()).await;
        assert!(resolved.is_ok(), "retry must not fail the turn");
        assert!(resolved.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
        {
            let slot = session.source.lock().unwrap();
            assert!(!slot.resolved, "slot resets for the next arm");
            assert!(slot.lease.is_none());
            assert!(slot.budget.is_none());
        }
        // The model can explicitly re-arm through the handoff tool.
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let result = (tool.execute)("retry-call".into(), json!({"receiver_role":"child","receiver_name":"receiver"}), None, None).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.details["status"], "capture_armed");
        assert_eq!(result.details["capture_attempts_failed"], 1);
        let text = result
            .content
            .iter()
            .filter_map(|block| match block {
                pi_agent_core::types::ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<String>();
        assert!(text.contains("retry attempt 1 of"), "retry note surfaced: {text}");
    }

    #[tokio::test]
    async fn failed_captures_deactivate_the_native_path_after_the_retry_budget() {
        let session = enrolled().await;
        let hooks = session.source_hooks();
        let request = || InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
            newly_supplied_messages: Vec::new(),
            signal: None,
        };
        for expected_attempts in 1..=(MAX_NATIVE_CAPTURE_RETRIES + 1) {
            // Simulate each armed turn failing after resolve took the budget.
            assert!(session.arm_source(budget()).is_ok());
            {
                let mut slot = session.source.lock().unwrap();
                assert!(slot.budget.take().is_some());
                slot.resolved = true;
            }
            let resolved = (hooks.resolve)(request()).await;
            assert!(resolved.is_ok(), "no retry step may fail the turn");
            assert!(resolved.unwrap().is_none());
            assert_eq!(session.failed_capture_attempts(), expected_attempts);
        }
        // Budget exhausted: the session deactivated for recovery. Even the
        // direct arm path refuses the closed session.
        assert!(session.arm_source(budget()).is_err());
    }

    fn hook_request() -> InferenceStateRequest {
        InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
            newly_supplied_messages: Vec::new(),
            signal: None,
        }
    }

    /// Drive one armed head through a real resolve (fresh lease), then end
    /// that capture either as a pre-admission busy refusal or as an ordinary
    /// failure (the agent loop's request guard cancels the lease), and run
    /// the next resolve, which is the tombstone decision.
    async fn armed_capture_outcome(session: &Arc<Occ3NativeSession>, busy: Option<Option<u64>>) {
        assert!(session.arm_handoff_if_needed("receiver", None).expect("arm"));
        session.sender_queue_mark_armed(None, Some("receiver")).expect("queue armed");
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(hook_request()).await.unwrap().expect("fresh lease");
        match busy {
            Some(retry_after_secs) => assert!(lease.cancel_unsent_as_busy(
                pi_ai::source_capture::SourceBusyRefusal { refused_at_epoch: now(), retry_after_secs }
            )),
            None => {
                lease.cancel();
            }
        }
        let next = (hooks.resolve)(hook_request()).await;
        assert!(next.is_ok(), "no outcome may fail the turn");
        assert!(next.unwrap().is_none(), "the retried turn runs uncaptured");
        let slot = session.source.lock().unwrap();
        assert!(slot.lease.is_none() && slot.budget.is_none() && !slot.resolved);
    }

    /// A source busy refusal keeps the capture-failure budget, schedules a
    /// bounded busy retry, and the next handoff call waits out the backoff
    /// and arms a FRESH capture. A later ordinary failure still counts.
    #[tokio::test]
    async fn source_busy_refusal_rearms_without_spending_the_failure_budget() {
        use super::super::occ3_sender_queue::{Occ3SenderQueueState, OCC3_SOURCE_BUSY_RETRY_REASON};
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        armed_capture_outcome(&session, Some(Some(1))).await;
        assert_eq!(session.failed_capture_attempts(), 0, "busy is not a capture failure");
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
        assert_eq!(entries[0].busy_retries, 1);
        assert_eq!(entries[0].attempts, 0);
        assert_eq!(entries[0].reason.as_deref(), Some(OCC3_SOURCE_BUSY_RETRY_REASON));
        assert!(entries[0].retry_not_before_epoch.is_some());
        // Cancellation during the backoff fails the call; nothing is armed.
        let cancelled = tokio_util::sync::CancellationToken::new();
        cancelled.cancel();
        assert!(session
            .prepare_receiver_busy_recapture("child", "receiver", Some(&cancelled))
            .await
            .is_err());
        assert!(session.source.lock().unwrap().budget.is_none(), "no arm after a cancel");
        // The tool's prepare + arm steps, called directly so the test does
        // not wait on the process-wide capture permit.
        let started = std::time::Instant::now();
        session
            .prepare_receiver_busy_recapture("child", "receiver", None)
            .await
            .expect("busy re-arm preparation");
        assert!(session.arm_handoff_if_needed("receiver", None).expect("fresh arm"));
        session.sender_queue_mark_armed(None, Some("receiver")).unwrap();
        // Retry-After 1 anchored at the refusal: at most ~1 s of wait.
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
        {
            let slot = session.source.lock().unwrap();
            assert!(slot.budget.is_some() && slot.lease.is_none(), "fresh budget, nothing reused");
        }
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].state, Occ3SenderQueueState::Armed);
        assert_eq!(entries[0].reason, None, "the busy reason does not outlive the re-arm");
        assert_eq!(entries[0].retry_not_before_epoch, None);
        // The fresh capture now fails for an ordinary reason: the failure
        // budget counts it and the busy count is unchanged.
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(hook_request()).await.unwrap().expect("lease");
        lease.cancel();
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].busy_retries, 1);
        assert_eq!(entries[0].attempts, 1);
    }

    /// After the busy budget the head falls back to ordinary OCC-1
    /// re-emission; the native path stays active and no failure is spent.
    #[tokio::test]
    async fn source_busy_exhausts_to_fallback_and_keeps_native_active() {
        use super::super::occ3_sender_queue::{Occ3SenderQueueState, OCC3_RECEIVER_BUSY_MAX_RETRIES};
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        for retry in 1..=OCC3_RECEIVER_BUSY_MAX_RETRIES {
            armed_capture_outcome(&session, Some(None)).await;
            let (entries, _) = session.sender_queue_snapshot().unwrap();
            assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
            assert_eq!(entries[0].busy_retries, retry);
            // Skip the real backoff: it is covered above and in the queue tests.
            session.queue.lock().unwrap().clear_head_backoff_for_test();
        }
        armed_capture_outcome(&session, Some(None)).await;
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].state, Occ3SenderQueueState::FallbackPending);
        assert_eq!(entries[0].reason.as_deref(), Some("source_busy_exhausted"));
        assert_eq!(entries[0].attempts, 0, "busy exhaustion is not a capture failure");
        assert_eq!(session.failed_capture_attempts(), 0);
        assert!(session.check().is_ok(), "busy never deactivates the native path");
        // The session retry re-runs a context ending in capture_armed, so the
        // model calls occ3_handoff again for the same receiver. That call
        // must not enqueue a duplicate (review round 3 MAJOR-1).
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let result = (tool.execute)(
            "after-exhaustion".into(),
            json!({"receiver_role":"child","receiver_name":"receiver"}),
            None,
            None,
        )
        .await
        .expect("truthful non-error result");
        assert_eq!(result.details["status"], "fallback_pending");
        assert_eq!(result.details["busy_retries"], OCC3_RECEIVER_BUSY_MAX_RETRIES);
        let text = result
            .content
            .iter()
            .filter_map(|block| match block {
                pi_agent_core::types::ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<String>();
        assert!(text.contains("fell back to ordinary messaging after 3 busy retries"), "{text}");
        assert!(text.contains("Do not send the facts now"), "{text}");
        assert!(text.contains("fallback prompt will ask you"), "{text}");
        assert!(text.contains("Do not call occ3_handoff again"), "{text}");
        assert_eq!(result.details["send_now"], false);
        // Inside the host's fallback turn the same call says send now, and
        // that turn is the only one whose send carries the dispatch token.
        let queue_id = session.sender_queue_snapshot().unwrap().0[0].queue_id.clone();
        session.begin_fallback_turn(&queue_id);
        let inside = (tool.execute)(
            "in-fallback-turn".into(),
            json!({"receiver_role":"child","receiver_name":"receiver"}),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(inside.details["send_now"], true);
        assert!(session.active_fallback_token_for("child:receiver").is_some());
        session.end_fallback_turn();
        // Outside that turn no send can complete the entry.
        assert!(session.active_fallback_token_for("child:receiver").is_none());
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries.len(), 1, "no duplicate entry for the same receiver");
        assert_eq!(entries[0].state, Occ3SenderQueueState::FallbackPending);
        {
            let slot = session.source.lock().unwrap();
            assert!(slot.budget.is_none() && slot.lease.is_none(), "nothing armed");
        }
        // The head prompt is the fallback prompt, never a native re-capture.
        let prompt = super::super::occ3_sender_queue::head_prompt(&entries[0], entries.len()).unwrap();
        assert!(prompt.contains("fell back to ordinary messaging"), "{prompt}");
        assert!(!prompt.contains("call occ3_handoff"), "{prompt}");
        // After the token-gated ack pops the head the queue is empty, so no
        // advance prompt asks for a native handoff of the same facts.
        let token = session.queue.lock().unwrap().head().unwrap().fallback_dispatch_token.clone();
        let queue_id = entries[0].queue_id.clone();
        assert!(session
            .queue
            .lock()
            .unwrap()
            .acknowledge_fallback(&queue_id, token.as_deref(), now())
            .is_some());
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert!(entries.is_empty());
    }

    /// When no automatic provider retry follows a source busy refusal, the
    /// AgentEnd hook records it so the head prompt still goes out; a later
    /// request then runs normally without counting a failure. Also pins
    /// Retry-After flowing from the lease record into the queue backoff.
    #[tokio::test]
    async fn source_busy_is_recorded_at_agent_end_without_a_provider_retry() {
        use super::super::occ3_sender_queue::{Occ3SenderQueueState, OCC3_SOURCE_BUSY_RETRY_REASON};
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        assert!(session.arm_handoff_if_needed("receiver", None).expect("arm"));
        session.sender_queue_mark_armed(None, Some("receiver")).unwrap();
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(hook_request()).await.unwrap().expect("fresh lease");
        let refused_at = now();
        assert!(lease.cancel_unsent_as_busy(pi_ai::source_capture::SourceBusyRefusal {
            refused_at_epoch: refused_at,
            retry_after_secs: Some(7),
        }));
        // No second resolve: the run simply ends (retry disabled or vetoed).
        assert!(session.absorb_source_busy_refusal());
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
        assert_eq!(entries[0].reason.as_deref(), Some(OCC3_SOURCE_BUSY_RETRY_REASON));
        assert_eq!(entries[0].busy_retries, 1);
        assert_eq!(entries[0].retry_not_before_epoch, Some(refused_at + 7), "Retry-After reaches the queue");
        let prompt = super::super::occ3_sender_queue::head_prompt(&entries[0], entries.len()).unwrap();
        assert!(prompt.contains("busy retry 1 of 3"), "{prompt}");
        // Idempotent: the slot was reset, so a second call changes nothing.
        assert!(!session.absorb_source_busy_refusal());
        assert_eq!(session.sender_queue_snapshot().unwrap().0[0].busy_retries, 1);
        // A later (user-prompted) request runs uncaptured; no failure counted.
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 0);
    }

    /// The AgentEnd hook leaves ordinary failures to the resolve hook.
    #[tokio::test]
    async fn agent_end_absorb_ignores_ordinary_failures() {
        let session = enrolled().await;
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = pool.clone().acquire_owned().await.unwrap();
        session.sender_queue_enqueue("child", "receiver").unwrap();
        assert!(session.arm_handoff_if_needed("receiver", Some(permit)).expect("arm"));
        session.sender_queue_mark_armed(None, Some("receiver")).unwrap();
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(hook_request()).await.unwrap().expect("fresh lease");
        lease.cancel();
        assert!(!session.absorb_source_busy_refusal());
        assert_eq!(pool.available_permits(), 1, "terminal failure releases capacity immediately");
        assert!(session.source.lock().unwrap().lease.is_some(), "retry accounting remains untouched");
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
    }

    /// With no armed queue head a busy refusal counts as an ordinary
    /// capture failure, so the attempt stays bounded.
    #[tokio::test]
    async fn source_busy_without_an_armed_head_counts_as_a_failure() {
        let session = enrolled().await;
        assert!(session.arm_source(budget()).is_ok());
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(hook_request()).await.unwrap().expect("lease");
        assert!(lease.cancel_unsent_as_busy(pi_ai::source_capture::SourceBusyRefusal {
            refused_at_epoch: now(),
            retry_after_secs: None,
        }));
        assert!((hooks.resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
    }

    /// Busy, re-arm, then an ordinary failure, then re-arm: the text is
    /// the ordinary retry text and names no busy cause (review MAJOR-1).
    #[tokio::test]
    async fn ordinary_failure_after_a_busy_rearm_is_not_reported_as_busy() {
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        armed_capture_outcome(&session, Some(None)).await;
        session.queue.lock().unwrap().clear_head_backoff_for_test();
        // Re-arm (directly, avoiding the process-wide permit), then fail.
        session.prepare_receiver_busy_recapture("child", "receiver", None).await.unwrap();
        session.source.lock().unwrap().budget = None;
        armed_capture_outcome(&session, None).await;
        assert_eq!(session.failed_capture_attempts(), 1);
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].busy_retries, 1);
        assert_eq!(entries[0].reason, None);
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let result = (tool.execute)(
            "rearm".into(),
            json!({"receiver_role":"child","receiver_name":"receiver"}),
            None,
            None,
        )
        .await
        .expect("re-arm");
        assert_eq!(result.details["status"], "capture_armed");
        assert_eq!(result.details["busy_retries"], 0);
        let text = result
            .content
            .iter()
            .filter_map(|block| match block {
                pi_agent_core::types::ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<String>();
        assert!(!text.contains("busy retry"), "{text}");
        assert!(!text.contains("was busy"), "{text}");
        assert!(text.contains("retry attempt 1 of"), "{text}");
    }

    /// The tool re-arm after a source busy refusal reports the busy retry
    /// truthfully. A second call in the same uncaptured window keeps the
    /// existing double-arm refusal and spends neither budget.
    #[tokio::test]
    async fn busy_rearm_text_is_truthful_and_a_double_arm_spends_nothing() {
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        armed_capture_outcome(&session, Some(None)).await;
        session.queue.lock().unwrap().clear_head_backoff_for_test();
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let args = json!({"receiver_role":"child","receiver_name":"receiver"});
        let first = (tool.execute)("rearm".into(), args.clone(), None, None).await.expect("re-arm");
        assert_eq!(first.details["status"], "capture_armed");
        assert_eq!(first.details["busy_retries"], 1);
        let text = |result: &AgentToolResult| {
            result
                .content
                .iter()
                .filter_map(|block| match block {
                    pi_agent_core::types::ContentBlock::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<String>()
        };
        assert!(text(&first).contains("busy retry 1 of 3"), "{}", text(&first));
        assert!(!text(&first).contains("failed capture"), "{}", text(&first));
        // A second call before any request resolved the fresh capture keeps
        // the existing double-arm refusal: it spends neither budget.
        assert!((tool.execute)("dup".into(), args, None, None).await.is_err());
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].busy_retries, 1, "no extra busy retry");
        assert_eq!(session.failed_capture_attempts(), 0);
    }

    /// Pre-existing bug: after a completed handoff to A, a queued handoff
    /// to B in the same session must be able to arm. While A's capture is
    /// armed, B is still refused.
    #[tokio::test]
    async fn completed_handoff_releases_the_receiver_binding() {
        let session = enrolled().await;
        assert!(session.arm_handoff_if_needed("receiver-a", None).unwrap());
        assert!(session.arm_handoff_if_needed("receiver-b", None).is_err(), "A armed: B refused");
        resolve_capture(&session).await;
        assert!(session.arm_handoff_if_needed("receiver-b", None).is_err(), "A capturing: B refused");
        {
            // The state complete_handoff_slot resets after retiring A's
            // admitted Captured lease (a Captured lease cannot be built
            // without a real receipt, so the reset helper is driven directly).
            let mut slot = session.source.lock().unwrap();
            slot.handoff_admitted = true;
            Occ3NativeSession::reset_slot_after_completed_handoff(&mut slot);
            assert!(slot.armed_receiver.is_none());
        }
        assert!(session.arm_handoff_if_needed("receiver-b", None).expect("B arms after A completed"));
        assert!(session.arm_handoff_if_needed("receiver-a", None).is_err(), "B armed: A refused");
        // A live budget keeps its binding through the completed-handoff reset.
        let mut slot = session.source.lock().unwrap();
        Occ3NativeSession::reset_slot_after_completed_handoff(&mut slot);
        assert_eq!(slot.armed_receiver.as_deref(), Some("receiver-b"));
    }


    /// Proven completion may only retire the capture bound to THAT message:
    /// an unrelated completion, or a slot whose lease never reached the
    /// Captured state, must leave the slot untouched. (A Captured lease
    /// cannot be built without a real receipt - see
    /// `complete_handoff_slot_refuses_unfinished_states` - so the positive
    /// retirement path is covered by the live parallel-handoff canary.)
    #[tokio::test]
    async fn proven_completion_retirement_guards_the_slot_state() {
        let session = enrolled().await;
        {
            // A different publication still bound: not this message's
            // capture, so nothing may be retired or reset.
            let mut slot = session.source.lock().unwrap();
            slot.handoff_admitted = true;
            slot.published_message = Some(("other-message".into(), "digest".into()));
        }
        session.retire_capture_for_completed("agentmsg-done").await;
        {
            let slot = session.source.lock().unwrap();
            assert!(slot.handoff_admitted, "unbound completion changes nothing");
            assert_eq!(
                slot.published_message.as_ref().unwrap().0,
                "other-message"
            );
        }
        {
            // Bound to this message, but the slot never held a Captured
            // lease: retirement would be wrong, so the slot stays.
            let mut slot = session.source.lock().unwrap();
            slot.published_message = Some(("agentmsg-done".into(), "digest".into()));
        }
        session.retire_capture_for_completed("agentmsg-done").await;
        {
            let slot = session.source.lock().unwrap();
            assert!(slot.handoff_admitted, "a not-yet-captured slot is never retired");
            assert!(slot.published_message.is_some());
        }
    }

    /// A busy refusal or ordinary failure frees the binding too, so the
    /// next queue head (another receiver) can arm.
    #[tokio::test]
    async fn failed_or_busy_capture_releases_the_receiver_binding() {
        let session = enrolled().await;
        armed_capture_outcome(&session, None).await;
        assert!(session.source.lock().unwrap().armed_receiver.is_none());
        assert!(session.arm_handoff_if_needed("other", None).expect("other arms"));
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        armed_capture_outcome(&session, Some(None)).await;
        assert!(session.source.lock().unwrap().armed_receiver.is_none());
    }

    #[tokio::test]
    async fn normal_handoff_capture_cannot_change_receiver_or_rearm() {
        let session = enrolled().await;
        assert!(session.arm_handoff_if_needed("receiver", None).unwrap());
        assert!(session.arm_handoff_if_needed("receiver", None).is_err());
        assert!(session.arm_handoff_if_needed("other", None).is_err());
        resolve_capture(&session).await;
        assert!(!session.arm_handoff_if_needed("receiver", None).unwrap());
        assert!(session.arm_handoff_if_needed("other", None).is_err());
        session.invalidate();
        assert!(session.arm_handoff_if_needed("receiver", None).is_err());
    }

    #[tokio::test]
    async fn occ3_policy_denial_requeues_unattempted_armed_head_without_spending_attempts() {
        use super::super::occ3_sender_queue::Occ3SenderQueueState;
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let file = sessions.join("session.jsonl");
        std::fs::write(&file, "{\"type\":\"session\",\"rlmDepth\":0}\n").unwrap();
        let session = enrolled().await;
        session.bind_chat_preferences(Some(file.to_string_lossy().into_owned())).unwrap();
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let args = json!({"receiver_role":"child","receiver_name":"receiver"});
        let armed = (tool.execute)("policy-arm".into(), args.clone(), None, None).await.unwrap();
        assert_eq!(armed.details["status"], "capture_armed");
        let before = session.sender_queue_snapshot().unwrap().0;
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].state, Occ3SenderQueueState::Armed);
        let queue_id = before[0].queue_id.clone();
        let _setup = session.take_pending_queue_notices();
        crate::core::occ::session_settings::save_occ3(file.to_str().unwrap(), true, false).unwrap();
        assert!((session.source_hooks().resolve)(hook_request()).await.is_err());
        let pending = session.sender_queue_snapshot().unwrap().0;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].state, Occ3SenderQueueState::Pending);
        assert_eq!(pending[0].queue_id, queue_id);
        assert_eq!(pending[0].attempts, 0);
        assert_eq!(session.failed_capture_attempts(), 0);
        let notices = session.take_pending_queue_notices();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "pending");
        assert_eq!(notices[0].reason.as_deref(), Some("capture_policy_denied"));
        assert!((tool.execute)("policy-still-disabled".into(), args.clone(), None, None).await.is_err());
        assert_eq!(session.sender_queue_snapshot().unwrap().0[0].state, Occ3SenderQueueState::Pending);
        assert!(session.source.lock().unwrap().budget.is_none());
        crate::core::occ::session_settings::save_occ3(file.to_str().unwrap(), true, true).unwrap();
        let rearmed = (tool.execute)("policy-rearm".into(), args, None, None).await.unwrap();
        assert_eq!(rearmed.details["status"], "capture_armed");
        let after = session.sender_queue_snapshot().unwrap().0;
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].queue_id, queue_id);
        assert_eq!(after[0].state, Occ3SenderQueueState::Armed);
        assert_eq!(after[0].reason, None, "re-arm must not retain policy denial");
        let rearm_notices = session.take_pending_queue_notices();
        assert_eq!(rearm_notices.len(), 1);
        assert_eq!(rearm_notices[0].state, "armed");
        assert_eq!(rearm_notices[0].reason, None);
        assert_eq!(after[0].attempts, 0);
        assert_eq!(session.failed_capture_attempts(), 0);
    }

    #[tokio::test]
    async fn occ3_policy_denial_preserves_actionable_error_when_queue_persist_fails() {
        use super::super::occ3_sender_queue::{sender_queue_path, Occ3SenderQueueState};
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let file = sessions.join("session.jsonl");
        std::fs::write(&file, "{\"type\":\"session\",\"rlmDepth\":0}\n").unwrap();
        let session = enrolled().await;
        session.bind_chat_preferences(Some(file.to_string_lossy().into_owned())).unwrap();
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let armed = (tool.execute)("policy-persist-arm".into(),
            json!({"receiver_role":"child","receiver_name":"receiver"}), None, None).await.unwrap();
        assert_eq!(armed.details["status"], "capture_armed");
        crate::core::occ::session_settings::save_occ3(file.to_str().unwrap(), true, false).unwrap();
        // Block only the queue artifact replacement, leaving occ.json readable.
        let queue_path = sender_queue_path(file.to_str().unwrap());
        std::fs::remove_file(&queue_path).unwrap();
        std::fs::create_dir(&queue_path).unwrap();
        assert!(session.check_native_allowed().unwrap_err().contains("/occ 3 on"));
        let error = (session.source_hooks().resolve)(hook_request()).await.unwrap_err().to_string();
        assert!(queue_path.is_dir(), "queue persist must have failed");
        assert!(error.contains("disabled for this chat"), "{error}");
        assert!(error.contains("/occ 3 on"), "{error}");
        let entries = session.sender_queue_snapshot().unwrap().0;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
        assert_eq!(entries[0].attempts, 0);
        assert_eq!(session.failed_capture_attempts(), 0);
        assert!(session.source.lock().unwrap().capture_permit.is_none());
    }

    #[tokio::test]
    async fn occ3_submitted_capture_cancellation_releases_capacity_at_agent_end() {
        let session = enrolled().await;
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        assert!(session.arm_handoff_if_needed("receiver", Some(pool.clone().acquire_owned().await.unwrap())).unwrap());
        resolve_capture(&session).await;
        let lease = session.source.lock().unwrap().lease.clone().unwrap();
        assert_eq!(lease.state(), SourceCaptureState::Prepared);
        assert!(lease.cancel_submitted_for_test());
        assert_eq!(lease.state(), SourceCaptureState::Uncertain);
        assert!(lease.receipt().is_none());
        assert!(lease.retirement_receipt().is_none());
        assert!(!lease.cancel_submitted_for_test(), "cannot replay a terminal lease");
        assert_eq!(pool.available_permits(), 0);
        assert!(!session.absorb_source_busy_refusal(), "uncertain is not a busy refusal");
        assert_eq!(pool.available_permits(), 1, "AgentEnd releases abandoned submitted capture");
        assert_eq!(session.failed_capture_attempts(), 0);
        assert!(session.arm_handoff_if_needed("receiver", None).is_err());
        assert!((session.source_hooks().resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 1);
        assert!(session.arm_handoff_if_needed("receiver", Some(pool.clone().try_acquire_owned().unwrap())).unwrap());
        assert_eq!(pool.available_permits(), 0);
    }

    #[test]
    fn occ3_terminal_capture_capacity_covers_every_lease_state() {
        use SourceCaptureState::*;
        assert!(Occ3NativeSession::capture_capacity_releasable(None));
        for state in [Prepared, Submitted, Captured] {
            assert!(!Occ3NativeSession::capture_capacity_releasable(Some(state)), "active {state:?}");
        }
        // SourceCaptureLease::cancel maps Submitted to Uncertain. Its submit
        // method is pi-ai-private, so test the exact state predicate here;
        // the Cancelled lifecycle test covers actual slot/permit release.
        for state in [Cancelled, Uncertain, Retired] {
            assert!(Occ3NativeSession::capture_capacity_releasable(Some(state)), "terminal {state:?}");
        }
    }

    #[tokio::test]
    async fn occ3_policy_disabled_before_capture_does_not_spend_failure_budget() {
        let dir = tempfile::tempdir().unwrap();
        // Preferences live beside the sessions directory, so nest the session
        // file to keep this test's policy artifact inside its own temp root.
        let sessions = dir.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let file = sessions.join("session.jsonl");
        std::fs::write(&file, "{\"type\":\"session\",\"rlmDepth\":0}\n").unwrap();
        let session = enrolled().await;
        session.bind_chat_preferences(Some(file.to_string_lossy().into_owned())).unwrap();
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        assert!(session.arm_handoff_if_needed("receiver", Some(pool.clone().acquire_owned().await.unwrap())).unwrap());
        crate::core::occ::session_settings::save_occ3(file.to_str().unwrap(), true, false).unwrap();
        assert!((session.source_hooks().resolve)(hook_request()).await.is_err());
        assert_eq!(pool.available_permits(), 1);
        assert!((session.source_hooks().resolve)(hook_request()).await.unwrap().is_none());
        assert_eq!(session.failed_capture_attempts(), 0, "policy denial is not capture failure");
        let slot = session.source.lock().unwrap();
        assert!(!slot.resolved);
        assert!(slot.budget.is_none() && slot.lease.is_none());
        assert!(slot.armed_receiver.is_none());
    }

    /// The live wedge: an enrollment that aged out with its permit (server
    /// max_ttl_seconds, at most one hour) must not fail every later turn.
    /// Both hook entry points deactivate the native session (terminal
    /// recovery clears the host hooks and the handoff tool) and let the
    /// request run on the ordinary inference path. Before the fix,
    /// `check_model` returned early from `check()` without any recovery,
    /// so the installed receiver resolver failed every inference request
    /// with `model_check_failed` forever.
    #[tokio::test]
    async fn expired_enrollment_deactivates_instead_of_failing_turns() {
        let request = || InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
            newly_supplied_messages: Vec::new(),
            signal: None,
        };
        // Receiver side: the resolver runs the turn ordinary instead of
        // refusing it, and the session is terminally recovered.
        let receiver = enrolled().await;
        receiver.force_expire_enrollment_for_test();
        assert!(receiver.is_dead());
        assert!(
            (receiver.receiver_resolver())(request()).await.unwrap().is_none(),
            "an expired enrollment must fall back to ordinary inference"
        );
        assert!(receiver.check().is_err(), "terminal recovery closed the native session");
        // Source side: an expired enrollment arms no capture and the turn
        // also runs uncaptured instead of erroring.
        let sender = enrolled().await;
        sender.force_expire_enrollment_for_test();
        assert!(
            (sender.source_hooks().resolve)(request()).await.unwrap().is_none(),
            "an expired enrollment must not fail the source turn"
        );
        assert!(sender.is_dead());
        // A live session is unaffected and a model switch stays a loud
        // identity break.
        let live = enrolled().await;
        assert!(!live.is_dead());
        assert!(live.check_model(&fixture_model()).is_ok());
        let mut other_model = fixture_model();
        other_model.id = "other-model".into();
        assert!(live.check_model(&other_model).is_err());
    }

    #[tokio::test]
    async fn native_actions_respect_dynamic_chat_and_parent_preferences() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let parent = sessions.join("parent.jsonl");
        let child = sessions.join("child.jsonl");
        std::fs::write(&parent, "{\"type\":\"session\",\"rlmDepth\":0}\n").unwrap();
        std::fs::write(&child, format!("{}\n", json!({"type":"session","rlmDepth":1,"parentSession":parent}))).unwrap();
        let session = enrolled().await;
        session.bind_chat_preferences(Some(child.to_string_lossy().into_owned())).unwrap();
        assert!(session.check_native_allowed().is_ok());
        let parent = parent.to_str().unwrap();
        crate::core::occ::session_settings::save_occ3(parent, true, false).unwrap();
        assert!(session.check_native_allowed().unwrap_err().contains("disabled for this chat"));
        assert!(session.check().is_ok(), "ordinary inference must remain available");
        let request = || InferenceStateRequest {
            model: fixture_model(), context: Default::default(), messages: Vec::new(),
            newly_supplied_messages: Vec::new(), signal: None,
        };
        assert!((session.source_hooks().resolve)(request()).await.unwrap().is_none());
        assert!((session.receiver_resolver())(request()).await.unwrap().is_none());
        crate::core::occ::session_settings::save_occ3(parent, true, true).unwrap();
        assert!(session.check_native_allowed().is_ok());
        std::fs::write(crate::core::occ::session_settings::preference_path(parent), "invalid").unwrap();
        assert!(session.check_native_allowed().is_err());
    }

    /// Receiver-side fresh-input provenance through the REAL resolver: an
    /// expired delivery resubmitted by the current turn refuses loudly, the
    /// same record present only as retained history never refuses and never
    /// parks the queue, and a live delivery alongside that history still
    /// claims and binds. X is admitted with expiry epoch 2 at admission
    /// epoch 1, so it is already dead relative to the real clock the
    /// resolver reads: the first unrelated turn sweeps the discovery loop
    /// and journals X expired SILENTLY (X absent from that turn's context).
    #[tokio::test]
    async fn receiver_resolver_expired_resubmission_vs_retained_history() {
        let session = enrolled().await;
        let fingerprint = fixture_identity().fingerprint();
        let envelope = |message_id: &str, sender: &str, expires_at_epoch: u64| Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: message_id.into(),
            sender_session_id: sender.into(),
            receiver_session_id: "unit-session".into(),
            model_fingerprint: fingerprint.clone(),
            runtime_id: "unit-fixture-runtime".into(),
            connector_id: "unit-fixture-occ3".into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: format!("handle-{message_id}"),
                connector_id: "unit-fixture-occ3".into(),
                model_fingerprint: fingerprint.clone(),
                runtime_generation: 7,
                expires_at_epoch,
                state_size_bytes: 128,
            },
            control_summary: "expired-resend-proof".into(),
            fallback_reference: "occ1".into(),
        };
        let marker = |id: &str| {
            AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
                content: CustomMessageContent::Text(String::new()),
                display: false,
                details: Some(json!({"id": id})),
                timestamp: 0,
            })
        };
        {
            let mut inbox = session.inbox.lock().unwrap();
            assert!(inbox
                .admit(
                    envelope("message-x", "sender-x", 2),
                    "message-x",
                    "sender-x",
                    1
                )
                .unwrap());
            assert!(inbox
                .admit(
                    envelope("message-y", "sender-y", now() + 600),
                    "message-y",
                    "sender-y",
                    1
                )
                .unwrap());
        }
        let resolver = session.receiver_resolver();
        let request = |ids: &[&str], fresh: &[&str]| InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: ids.iter().map(|id| marker(id)).collect(),
            newly_supplied_messages: fresh.iter().map(|id| marker(id)).collect(),
            signal: None,
        };
        // 1) Unrelated turn: X is absent from the context, so the discovery
        //    loop journals it expired silently and the turn proceeds with no
        //    attachment.
        assert!(
            (resolver)(request(&[], &[])).await.unwrap().is_none(),
            "silent expiry never blocks the unrelated turn"
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("message-x"),
            Some(Occ3DeliveryStatus::Expired)
        );
        // 2) Fresh resubmission of the expired delivery refuses loudly
        //    instead of silently falling back to an ordinary text turn.
        assert!(
            (resolver)(request(&["message-x"], &["message-x"]))
                .await
                .is_err(),
            "a resubmitted expired delivery must refuse"
        );
        // 3) The same expired record as retained history (both retained
        //    copies in the context, none freshly supplied) never refuses.
        assert!(
            (resolver)(request(&["message-x", "message-x"], &[]))
                .await
                .unwrap()
                .is_none(),
            "history-only expired records never block the turn"
        );
        // 4) A fresh live Y alongside that history claims and binds Y; the
        //    expired X record is never consumed by it.
        let lease = (resolver)(request(&["message-x", "message-y"], &["message-y"]))
            .await
            .expect("the live claim must not fail")
            .expect("the live delivery must bind");
        drop(lease);
        assert_eq!(
            session.inbox.lock().unwrap().status("message-y"),
            Some(Occ3DeliveryStatus::Delivered)
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("message-x"),
            Some(Occ3DeliveryStatus::Expired),
            "the expired record is never consumed by the live claim"
        );
    }

    /// The receiver-side stuck-inbox-claim wedge (live 2026-11-07 parallel
    /// handoff pair q1): `claim` marks the earliest admitted record
    /// claimed/`Delivered`, then a busy-refused `prepare_inference` must
    /// retire that claim terminally instead of parking the record claimed
    /// forever. The busy turn proceeds as an ordinary turn without the
    /// attachment, the delivery retires as a terminal busy failure (never
    /// re-claimable: the sender's fresh-capture retry is the sole path for
    /// these facts, so the two retries can never double-deliver), and a
    /// distinct content-free notice - including whether a live sender
    /// absorbed the route - names the retirement so live diagnosis needs
    /// no source archaeology.
    #[tokio::test]
    async fn receiver_resolver_retires_claim_on_busy_refusal_and_routes_sender_retry() {
        struct ResetSink;
        impl Drop for ResetSink {
            fn drop(&mut self) {
                pi_ai::log::set_log_sink(None);
            }
        }
        let _sink_guard = pi_ai::log::test_sink_guard();
        let _reset = ResetSink;
        let entries = Arc::new(Mutex::new(Vec::new()));
        let captured = entries.clone();
        let me = std::thread::current().id();
        pi_ai::log::set_log_sink(Some(Box::new(move |entry| {
            if std::thread::current().id() == me
                && entry["msg"].as_str() == Some("receiver_busy_claim_released")
            {
                captured.lock().unwrap().push(entry.clone());
            }
        })));
        let connector = FixtureConnector::new();
        connector.refuse_prepare_with(&[Occ3Error::NativeInferenceBusyRefused {
            connector_id: "unit-fixture-occ3".into(),
        }]);
        let session = Occ3NativeSession::enroll(
            "unit-session",
            &fixture_model(),
            Arc::new(connector),
        )
        .await
        .expect("fixture enrollment");
        // The live sender of this delivery, so the retire route has a real
        // queue to notify (liveness: the retired record never surfaces
        // through terminal-failure polls, and an in-flight Admitted entry
        // never TTL-expires).
        let sender = Occ3NativeSession::enroll(
            "sender-busy",
            &fixture_model(),
            Arc::new(FixtureConnector::new()),
        )
        .await
        .expect("fixture sender enrollment");
        sender.sender_queue_enqueue("child", "receiver").unwrap();
        sender
            .sender_queue_mark_armed(Some("busy-1"), Some("receiver"))
            .unwrap();
        sender
            .sender_queue_mark_progressed(
                crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Admitted,
                "busy-1",
            )
            .unwrap();
        let _sender_setup = sender.take_pending_queue_notices();
        let fingerprint = fixture_identity().fingerprint();
        let envelope = Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: "busy-1".into(),
            sender_session_id: "sender-busy".into(),
            receiver_session_id: "unit-session".into(),
            model_fingerprint: fingerprint.clone(),
            runtime_id: "unit-fixture-runtime".into(),
            connector_id: "unit-fixture-occ3".into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: "handle-busy-1".into(),
                connector_id: "unit-fixture-occ3".into(),
                model_fingerprint: fingerprint.clone(),
                runtime_generation: 7,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary: "busy-release-proof".into(),
            fallback_reference: "occ1".into(),
        };
        {
            let mut inbox = session.inbox.lock().unwrap();
            assert!(inbox.admit(envelope, "busy-1", "sender-busy", 1).unwrap());
        }
        let resolver = session.receiver_resolver();
        let request = |ids: &[&str], fresh: &[&str]| InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: ids
                .iter()
                .map(|id| {
                    AgentMessage::Custom(CustomAgentMessage::Custom {
                        custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
                        content: CustomMessageContent::Text(String::new()),
                        display: false,
                        details: Some(json!({"id": id})),
                        timestamp: 0,
                    })
                })
                .collect(),
            newly_supplied_messages: fresh
                .iter()
                .map(|id| {
                    AgentMessage::Custom(CustomAgentMessage::Custom {
                        custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
                        content: CustomMessageContent::Text(String::new()),
                        display: false,
                        details: Some(json!({"id": id})),
                        timestamp: 0,
                    })
                })
                .collect(),
            signal: None,
        };
        // 1) The busy turn: the claim is retired and the turn proceeds
        //    without the attachment - never the generic uncertain error
        //    with the record stuck claimed.
        assert!(
            (resolver)(request(&["busy-1"], &["busy-1"]))
                .await
                .unwrap()
                .is_none(),
            "a proven-undispatched busy refusal must retire the claim and run the turn ordinary"
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("busy-1"),
            Some(Occ3DeliveryStatus::Failed),
            "the busy-retired delivery retires as a terminal busy failure: it must not stay claimable while the sender re-arms"
        );
        // The distinct content-free notice names the retirement exactly
        // once, including that a live sender absorbed the route.
        {
            let values = entries.lock().unwrap();
            assert_eq!(values.len(), 1, "exactly one busy-retirement notice");
            assert_eq!(values[0]["message_id"], "busy-1");
            assert_eq!(values[0]["connector_id"], "unit-fixture-occ3");
            assert_eq!(values[0]["released"], serde_json::json!(true));
            assert_eq!(values[0]["sender_routed"], serde_json::json!(true));
            assert!(
                !values[0].to_string().contains("busy-release-proof"),
                "the notice never carries envelope content"
            );
        }
        // Liveness: the retire route must ALSO notify the sender. The receiver's
        // control-only transcript body carries no source facts and the
        // retired record never reaches a terminal-failure poll, so without
        // this route the sender's in-flight Admitted entry would sit
        // unexpired forever and the facts would be silently lost. The same
        // proven-non-use route the TurnEnd dispatcher uses for refused
        // leases schedules one bounded busy retry on the sender.
        {
            let notices = sender.take_pending_queue_notices();
            assert_eq!(notices.len(), 1, "exactly one sender busy-retry notice");
            assert_eq!(notices[0].state, "pending");
            assert_eq!(notices[0].message_id.as_deref(), Some("busy-1"));
            let (entries_snapshot, _) = sender.sender_queue_snapshot().unwrap();
            assert_eq!(
                entries_snapshot[0].state,
                crate::core::agent_session::occ3_sender_queue::Occ3SenderQueueState::Pending
            );
            assert_eq!(entries_snapshot[0].busy_retries, 1);
            assert!(entries_snapshot[0].retry_not_before_epoch.is_some());
        }
        // 2) The retired record is never re-claimed: a later inference
        //    request finds no claimable delivery, so the receiver cannot
        //    consume this handle while the sender's busy-retry re-arm is
        //    already in flight (the duplicate-delivery guard).
        assert!(
            (resolver)(request(&["busy-1"], &["busy-1"]))
                .await
                .unwrap()
                .is_none(),
            "a busy-retired delivery must never be claimable again"
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("busy-1"),
            Some(Occ3DeliveryStatus::Failed)
        );
        // The pre-notified terminal marker means the TurnEnd poll never
        // re-routes this record as an ordinary receiver failure.
        assert!(session.inbox.lock().unwrap().poll_terminal_failures().is_empty());
    }

    /// The orphan case of the busy retirement: the sender is gone (no live
    /// native session under the envelope's sender id), so the route finds
    /// nobody. The turn must STILL proceed ordinary (the retirement itself
    /// is not a failure), the record retires terminal, and the diagnostic
    /// carries `released == true, sender_routed == false` so the lost facts
    /// are visible instead of silent.
    #[tokio::test]
    async fn receiver_resolver_busy_retirement_without_a_live_sender_is_visible() {
        let _sink_guard = pi_ai::log::test_sink_guard();
        struct ResetSink;
        impl Drop for ResetSink {
            fn drop(&mut self) {
                pi_ai::log::set_log_sink(None);
            }
        }
        let _reset = ResetSink;
        let entries = Arc::new(Mutex::new(Vec::new()));
        let captured = entries.clone();
        let me = std::thread::current().id();
        pi_ai::log::set_log_sink(Some(Box::new(move |entry| {
            if std::thread::current().id() == me
                && entry["msg"].as_str() == Some("receiver_busy_claim_released")
            {
                captured.lock().unwrap().push(entry.clone());
            }
        })));
        // No sender session is enrolled for "sender-orphan": the route
        // must report false and nothing may panic or wedge.
        let connector = FixtureConnector::new();
        connector.refuse_prepare_with(&[Occ3Error::NativeInferenceBusyRefused {
            connector_id: "unit-fixture-occ3".into(),
        }]);
        let session = Occ3NativeSession::enroll(
            "unit-session-orphan",
            &fixture_model(),
            Arc::new(connector),
        )
        .await
        .expect("fixture enrollment");
        let fingerprint = fixture_identity().fingerprint();
        let envelope = Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: "busy-orphan-1".into(),
            sender_session_id: "sender-orphan".into(),
            receiver_session_id: "unit-session-orphan".into(),
            model_fingerprint: fingerprint.clone(),
            runtime_id: "unit-fixture-runtime".into(),
            connector_id: "unit-fixture-occ3".into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: "handle-orphan-1".into(),
                connector_id: "unit-fixture-occ3".into(),
                model_fingerprint: fingerprint,
                runtime_generation: 7,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary: "busy-orphan-proof".into(),
            fallback_reference: "occ1".into(),
        };
        {
            let mut inbox = session.inbox.lock().unwrap();
            assert!(inbox
                .admit(envelope, "busy-orphan-1", "sender-orphan", 1)
                .unwrap());
        }
        let resolver = session.receiver_resolver();
        let request = || InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: vec![AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
                content: CustomMessageContent::Text(String::new()),
                display: false,
                details: Some(json!({"id": "busy-orphan-1"})),
                timestamp: 0,
            })],
            newly_supplied_messages: Vec::new(),
            signal: None,
        };
        assert!(
            (resolver)(request()).await.unwrap().is_none(),
            "an orphaned busy retirement must still run the turn ordinary"
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("busy-orphan-1"),
            Some(Occ3DeliveryStatus::Failed)
        );
        {
            // Scoped so the guard drops before the second resolve: the log
            // sink locks the same mutex.
            let values = entries.lock().unwrap();
            assert_eq!(values.len(), 1, "exactly one busy-retirement notice");
            assert_eq!(values[0]["message_id"], "busy-orphan-1");
            assert_eq!(values[0]["connector_id"], "unit-fixture-occ3");
            assert_eq!(values[0]["released"], serde_json::json!(true));
            assert_eq!(values[0]["sender_routed"], serde_json::json!(false));
            assert!(
                !values[0].to_string().contains("busy-orphan-proof"),
                "the notice never carries envelope content"
            );
        }
        // The orphan diagnostic is the only record of the lost facts, so
        // the retirement is pinned like the sibling's: never re-claimable,
        // and no terminal poll ever re-routes it.
        assert!(
            (resolver)(request()).await.unwrap().is_none(),
            "a busy-retired delivery must never be claimable again"
        );
        assert!(session
            .inbox
            .lock()
            .unwrap()
            .poll_terminal_failures()
            .is_empty());
    }

    /// Fail-closed stays fail-closed: a NON-busy `prepare_inference`
    /// failure keeps today's exact generic uncertain error and the record
    /// stays claimed, so nothing is ever replayed and the error string the
    /// live wedge investigations pinned stays byte-identical.
    #[tokio::test]
    async fn receiver_resolver_keeps_fail_closed_error_when_prepare_is_not_busy() {
        struct ResetSink;
        impl Drop for ResetSink {
            fn drop(&mut self) {
                pi_ai::log::set_log_sink(None);
            }
        }
        let _sink_guard = pi_ai::log::test_sink_guard();
        let _reset = ResetSink;
        let entries = Arc::new(Mutex::new(Vec::new()));
        let captured = entries.clone();
        let me = std::thread::current().id();
        pi_ai::log::set_log_sink(Some(Box::new(move |entry| {
            if std::thread::current().id() == me
                && entry["msg"].as_str() == Some("receiver_refused")
            {
                captured.lock().unwrap().push(entry.clone());
            }
        })));
        let connector = FixtureConnector::new();
        connector.refuse_prepare_with(&[Occ3Error::TransferFailed {
            reason: "unit-fixture-prepare".into(),
        }]);
        let session = Occ3NativeSession::enroll(
            "unit-session",
            &fixture_model(),
            Arc::new(connector),
        )
        .await
        .expect("fixture enrollment");
        let fingerprint = fixture_identity().fingerprint();
        let envelope = Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: "uncertain-1".into(),
            sender_session_id: "sender-uncertain".into(),
            receiver_session_id: "unit-session".into(),
            model_fingerprint: fingerprint.clone(),
            runtime_id: "unit-fixture-runtime".into(),
            connector_id: "unit-fixture-occ3".into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: "handle-uncertain-1".into(),
                connector_id: "unit-fixture-occ3".into(),
                model_fingerprint: fingerprint,
                runtime_generation: 7,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary: "uncertain-proof".into(),
            fallback_reference: "occ1".into(),
        };
        {
            let mut inbox = session.inbox.lock().unwrap();
            assert!(inbox.admit(envelope, "uncertain-1", "sender-uncertain", 1).unwrap());
        }
        let resolver = session.receiver_resolver();
        let request = || InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: vec![AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
                content: CustomMessageContent::Text(String::new()),
                display: false,
                details: Some(json!({"id": "uncertain-1"})),
                timestamp: 0,
            })],
            newly_supplied_messages: Vec::new(),
            signal: None,
        };
        // The first turn fails with today's exact generic string and the
        // record stays claimed...
        assert_eq!(
            (resolver)(request()).await.unwrap_err().to_string(),
            failure(),
            "a non-busy prepare failure keeps the fail-closed error unchanged"
        );
        assert_eq!(
            session.inbox.lock().unwrap().status("uncertain-1"),
            Some(Occ3DeliveryStatus::Delivered)
        );
        // The refusal is now visible: one content-free diagnostic naming the
        // exit and the connector error kind, never the envelope or reason text.
        {
            let values = entries.lock().unwrap();
            assert_eq!(values.len(), 1, "exactly one receiver refusal diagnostic");
            assert_eq!(values[0]["reason"], "prepare_inference_refused");
            assert_eq!(values[0]["error_kind"], "transfer_failed");
            assert_eq!(values[0]["message_id"], "uncertain-1");
            let text = values[0].to_string();
            assert!(!text.contains("uncertain-proof"), "never envelope content");
            assert!(!text.contains("unit-fixture-prepare"), "never connector reason text");
        }
        // ...so the next turn still refuses on the claimed record instead of
        // silently re-arming it: replay stays forbidden.
        assert_eq!(
            (resolver)(request()).await.unwrap_err().to_string(),
            failure(),
            "the claimed record must keep refusing until a terminal outcome"
        );
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

    /// The authenticated native family edge the controller authorizes before it
    /// can build a prepared handoff.
    fn authorized_handoff() -> AuthorizedNativeHandoff {
        AuthorizedNativeHandoff::from_catalog(
            &AgentFamilyCatalogEntry {
                id: "unit-source".into(),
                depth: 0.,
                ..Default::default()
            },
            &AgentFamilyCatalogEntry {
                id: "unit-session".into(),
                depth: 1.,
                parent_session_id: Some("unit-source".into()),
                ..Default::default()
            },
            AgentSessionMessageEndpoint {
                active_session_id: "unit-source-active".into(),
                session_id: "unit-source".into(),
                ..Default::default()
            },
            AgentSessionMessageEndpoint {
                active_session_id: "unit-session-active".into(),
                session_id: "unit-session".into(),
                ..Default::default()
            },
            "child",
        )
        .expect("fixture authorization")
    }

    fn prepared_handoff(authorization: AuthorizedNativeHandoff) -> PreparedNativeHandoff {
        let envelope = Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: authorization.message_id().to_string(),
            sender_session_id: "unit-source".into(),
            receiver_session_id: "unit-session".into(),
            model_fingerprint: fixture_identity().fingerprint(),
            runtime_id: "unit-fixture-runtime".into(),
            connector_id: "unit-fixture-occ3".into(),
            latent_method: Occ3Method::StatebridgePrefix,
            state_handle: Occ3HandleRef {
                handle_id: "unit-fixture-handle".into(),
                connector_id: "unit-fixture-occ3".into(),
                model_fingerprint: fixture_identity().fingerprint(),
                runtime_generation: 7,
                expires_at_epoch: now() + 600,
                state_size_bytes: 128,
            },
            control_summary: "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.".into(),
            fallback_reference: "occ1".into(),
        };
        PreparedNativeHandoff::new(
            authorization,
            NativePublication {
                publication_id: "unit-fixture-publication".into(),
                binding_digest: "4".repeat(64),
                envelope,
            },
        )
        .expect("fixture prepared handoff")
    }

    /// The latent marker exists only behind the receiver's completed-use receipt:
    /// an admitted or failed delivery records a pulse-free `pending` marker.
    #[tokio::test]
    async fn delivery_marker_requires_completed_use() {
        let session = enrolled().await;
        let prepared = prepared_handoff(authorized_handoff());
        let message_id = prepared.authorization().message_id().to_string();
        assert!(session.admit(&prepared).expect("fixture admit"));
        // A bare admission binds no lease, so it has no completed-use receipt.
        assert!(session.outcome(&message_id).is_err());
        let admitted = session.delivery_marker(&message_id).expect("marker");
        assert_eq!(admitted.kind, OCC3_DELIVERY_PENDING);
        assert!(!admitted.completed_use);
        assert!(admitted.receipt_sha256.is_none());
        assert!(!admitted.is_completed_native_latent());
        assert_eq!(admitted.runtime_generation, 7);
        assert_eq!(admitted.method_id, METHOD);
        // The identity digest is a digest: neither the raw runtime id nor the
        // session id may appear in the recorded marker.
        assert_eq!(admitted.runtime_identity_sha256.len(), 64);
        assert!(!admitted.runtime_identity_sha256.contains("unit"));
        // A failed or expired delivery stays pending: no completion, no pulse.
        session.invalidate();
        let failed = session.delivery_marker(&message_id).expect("marker");
        assert_eq!(failed.kind, OCC3_DELIVERY_PENDING);
        assert!(!failed.is_completed_native_latent());
        // An unknown message id has no marker at all.
        assert!(session.delivery_marker("no-such-message").is_err());
    }

    /// A receiver-busy refusal returns the head to `pending`, and the next
    /// handoff call for that receiver waits out the backoff and arms a FRESH
    /// capture (new budget, no lease reused) instead of falling back.
    #[tokio::test]
    async fn receiver_busy_refusal_rearms_a_fresh_capture_after_backoff() {
        use super::super::occ3_sender_queue::Occ3SenderQueueState;
        let session = enrolled().await;
        session.sender_queue_enqueue("child", "receiver").unwrap();
        session
            .sender_queue_mark_armed(Some("agentmsg_busy"), Some("receiver"))
            .unwrap();
        session
            .sender_queue_mark_progressed(Occ3SenderQueueState::Admitted, "agentmsg_busy")
            .unwrap();
        let notices = session.sender_queue_record_receiver_busy("agentmsg_busy").unwrap();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "pending");
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].busy_retries, 1);
        assert!(entries[0].retry_not_before_epoch.is_some());
        // The tool runs these two steps after authorization. They are
        // called directly so the test does not wait on the process-wide
        // capture permit, which parallel tests can hold.
        let started = std::time::Instant::now();
        session
            .prepare_receiver_busy_recapture("child", "receiver", None)
            .await
            .expect("busy re-capture preparation");
        assert!(
            session.arm_handoff_if_needed("receiver", None).expect("fresh arm"),
            "a busy-refused head re-arms"
        );
        session.sender_queue_mark_armed(None, Some("receiver")).expect("queue armed");
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(900),
            "the re-arm honours the backoff"
        );
        let slot = session.source.lock().unwrap();
        assert!(slot.budget.is_some(), "fresh budget armed");
        assert!(slot.lease.is_none(), "no earlier capture reused");
        assert!(!slot.handoff_admitted);
        drop(slot);
        let (entries, _) = session.sender_queue_snapshot().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].state, Occ3SenderQueueState::Armed);
        assert_eq!(entries[0].retry_not_before_epoch, None);
        assert_eq!(entries[0].busy_retries, 1, "the retry count persists");
        assert_eq!(session.failed_capture_attempts(), 0, "busy never spends the capture budget");
    }

    /// The receiver-side dispatcher routes a busy refusal to the sender's
    /// queue by sender session id; an unknown message id changes nothing.
    #[tokio::test]
    async fn receiver_busy_routes_to_the_sender_queue() {
        use super::super::occ3_sender_queue::Occ3SenderQueueState;
        let sender = enrolled().await;
        sender.sender_queue_enqueue("child", "receiver").unwrap();
        sender
            .sender_queue_mark_armed(Some("agentmsg_route_busy"), Some("receiver"))
            .unwrap();
        sender
            .sender_queue_mark_progressed(Occ3SenderQueueState::Admitted, "agentmsg_route_busy")
            .unwrap();
        let _setup = sender.take_pending_queue_notices();
        route_receiver_busy_to_sender("unit-session", "agentmsg_unknown");
        assert!(sender.take_pending_queue_notices().is_empty());
        route_receiver_busy_to_sender("unit-session", "agentmsg_route_busy");
        let notices = sender.take_pending_queue_notices();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "pending");
        let (entries, _) = sender.sender_queue_snapshot().unwrap();
        assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
        assert_eq!(entries[0].busy_retries, 1);
    }

    /// `complete_handoff_slot` refuses a slot that is not a completed
    /// handoff (admitted Captured lease): a Prepared lease keeps the
    /// fail-closed contract even with the flags set.
    #[tokio::test]
    async fn complete_handoff_slot_refuses_unfinished_states() {
        let session = enrolled().await;
        // Nothing armed: no lease at all.
        assert!(session.complete_handoff_slot().await.is_err());
        session.arm_source(budget()).expect("arm source");
        resolve_capture(&session).await;
        {
            let mut slot = session.source.lock().unwrap();
            slot.resolved = true;
            slot.tool_claimed = true;
            slot.handoff_admitted = true;
        }
        // The lease is Prepared, not Captured: refuse.
        assert!(session.complete_handoff_slot().await.is_err());
        {
            let slot = session.source.lock().unwrap();
            assert!(slot.lease.is_some(), "refusal preserves the lease");
            assert!(slot.handoff_admitted, "refusal preserves the admission");
        }
    }

    /// F4: a different receiver's request while a pending head waits
    /// enqueues behind it (never an error); the head keeps its position.
    #[tokio::test]
    async fn queued_head_new_request_enqueues_behind() {
        let session = enrolled().await;
        let notice = session
            .sender_queue_enqueue("child", "head-receiver")
            .expect("enqueue head");
        session.record_queue_notice(notice);
        let tool = session.handoff_tool(Arc::new(ArmingController(true)));
        let result = (tool.execute)(
            "new-request".into(),
            json!({"receiver_role":"child","receiver_name":"other-receiver"}),
            None,
            None,
        )
        .await
        .expect("a new request enqueues");
        assert_eq!(result.details["status"], "queued");
        assert_eq!(result.details["queue_position"], 2);
        let (entries, _) = session.sender_queue_snapshot().expect("snapshot");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].receiver_name, "head-receiver");
    }

    #[tokio::test]
    async fn negative_control_arm_requires_armed_budget_and_is_one_shot() {
        let session = enrolled().await;
        // Without an armed budget the opt-in cannot attach to any capture.
        assert!(session.arm_negative_control().is_err());
        session.arm_source(budget()).expect("arm source");
        assert!(session.arm_negative_control().is_ok());
        // One-shot: the second opt-in fails.
        assert!(session.arm_negative_control().is_err());
    }

    #[tokio::test]
    async fn negative_control_arm_fails_closed_once_capture_started() {
        let session = enrolled().await;
        session.arm_source(budget()).expect("arm source");
        resolve_capture(&session).await;
        // An ordinary started/completed capture can never be reclassified.
        assert!(session.arm_negative_control().is_err());
        assert!(session.export_negative_control().is_none());
        let audit = session.audit();
        assert_eq!(audit["source"]["state"], json!("Prepared"));
        assert_eq!(audit["source"]["negative_control_armed"], json!(false));
    }

    #[tokio::test]
    async fn armed_export_fails_closed_until_receipt_is_captured() {
        let session = enrolled().await;
        session.arm_source(budget()).expect("arm source");
        session.arm_negative_control().expect("arm negative control");
        resolve_capture(&session).await;
        // Armed but not Captured: the raw handle stays inaccessible.
        assert!(session.export_negative_control().is_none());
        let audit = session.audit();
        assert_eq!(audit["source"]["negative_control_armed"], json!(true));
        assert_eq!(audit["source"]["negative_control_exported"], json!(false));
        // The audit projection stays hash-only: no raw handle key appears.
        assert!(!audit.to_string().contains("source_handle"));
    }

    #[tokio::test]
    async fn export_fails_closed_after_invalidate() {
        let session = enrolled().await;
        session.arm_source(budget()).expect("arm source");
        session.arm_negative_control().expect("arm negative control");
        resolve_capture(&session).await;
        session.invalidate();
        assert!(session.export_negative_control().is_none());
        // Arming stays one-shot even after invalidation.
        assert!(session.arm_negative_control().is_err());
    }

    #[test]
    fn export_debug_projection_hides_raw_handle_and_receipt() {
        let receipt = SourceCaptureReceipt {
            schema: "occ3.source.v1".into(),
            intent_digest: "a".repeat(64),
            source_handle: RAW_HANDLE.into(),
            native_rid: "unit-native-rid".into(),
            response_id: "unit-response-id".into(),
            capture_commit_id: "unit-commit".into(),
            accepted_span_start: 0,
            accepted_span_end: 8,
            retained_count: 8,
            predictor_position_offset: -1,
            representation: "final_layer_prenorm_predictor".into(),
            layer_index: 3,
            hidden_dim: 8,
            dtype: "bfloat16".into(),
            layout_digest: "1".repeat(64),
            ready_ranks: 8,
            expected_ranks: 8,
            accepted_sequence_digest: "c".repeat(64),
            label_anchor_alpha: Default::default(),
            output_sha256: "d".repeat(64),
            expires_at_epoch: now() + 600,
            byte_count: 128,
            span: None,
        };
        let export = NegativeControlExport {
            raw_source_handle: RAW_HANDLE.into(),
            receipt,
            runtime_epoch: "unit-fixture-runtime".into(),
            runtime_generation: 7,
            expires_at_epoch: now() + 600,
            retention_reason: NEGATIVE_CONTROL_RETENTION_REASON,
        };
        let debug = format!("{export:?}");
        assert!(!debug.contains(RAW_HANDLE));
        assert!(!debug.contains("unit-native-rid"));
        assert!(debug.contains(&pi_ai::source_capture::sha256(RAW_HANDLE.as_bytes())));
        assert_eq!(export.retention_reason, "planned_restart");
    }
}
