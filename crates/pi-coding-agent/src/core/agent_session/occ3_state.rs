//! Explicit process-local source/receiver state for isolated native handoff.
//! No settings, durable lease recovery, helper generation or kernel provenance
//! inference. Ordinary sessions have no instance and retain their normal path.
use crate::core::agent_messages::{
    AgentSessionMessageController, Occ3DeliveryDetails, PreparedNativeHandoff,
    AGENT_MESSAGE_CUSTOM_TYPE,
};
use crate::core::occ::occ3::connector::{
    NativePublicationRequest, NativeSessionEnrollment, Occ3Connector,
};
use crate::core::occ::occ3::inference::{Occ3InferenceInbox, PrepareInferenceRequest};
use crate::core::occ::occ3::registry::Occ3ConnectorRegistry;
use crate::core::occ::occ3::types::Occ3DeliveryStatus;
use pi_agent_core::types::ContentBlock;
use pi_agent_core::types::{
    AgentMessage, AgentTool, AgentToolResult, CustomAgentMessage, ResolveInferenceState,
    SourceCaptureHooks,
};
use pi_ai::inference_attachment::InferenceStateTarget;
use pi_ai::source_capture::{
    SourceCaptureLease, SourceCaptureLimits, SourceCaptureReceipt, SourceCaptureState,
};
use pi_ai::types::Model;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const NATIVE_HANDOFF_TOOL: &str = "occ3_handoff";
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
#[derive(Clone)]
pub struct NativeSourceBudget {
    pub retained_count_limit: u32,
    pub max_capture_bytes: u64,
    pub deadline: Instant,
    pub max_response_bytes: usize,
}
struct SourceSlot {
    budget: Option<NativeSourceBudget>,
    resolved: bool,
    lease: Option<SourceCaptureLease>,
    tool_claimed: bool,
    handoff_admitted: bool,
    published_message: Option<(String, String)>,
    negative_control_armed: bool,
    negative_control_exported: bool,
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
    source: Mutex<SourceSlot>,
    closed: std::sync::atomic::AtomicBool,
    /// Unsubscribe for the delivery-marker restamp listener installed by
    /// [`Self::install_delivery_restamp`]; dropped by [`Self::invalidate`].
    restamp_unsubscribe: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
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
        Ok(Arc::new(Self {
            enrollment,
            target: target(model),
            connector,
            registry,
            restamp_unsubscribe: Mutex::new(None),
            inbox: Mutex::new(Occ3InferenceInbox::new(session_id.into()).map_err(|_| failure())?),
            source: Mutex::new(SourceSlot {
                budget: None,
                resolved: false,
                lease: None,
                tool_claimed: false,
                handoff_admitted: false,
                published_message: None,
                negative_control_armed: false,
                negative_control_exported: false,
            }),
            closed: std::sync::atomic::AtomicBool::new(false),
        }))
    }
    pub fn session_id(&self) -> &str {
        &self.enrollment.session_id
    }
    pub fn arm_source(&self, budget: NativeSourceBudget) -> Result<(), String> {
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
    fn check(&self) -> Result<(), String> {
        if self.closed.load(std::sync::atomic::Ordering::Acquire)
            || self.enrollment.expires_at_epoch <= now()
        {
            Err(failure())
        } else {
            Ok(())
        }
    }
    fn check_model(&self, model: &Model) -> Result<(), String> {
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
                    let budget = {
                        let mut slot =
                            this.source.lock().map_err(|_| anyhow::anyhow!(failure()))?;
                        let Some(budget) = slot.budget.take() else {
                            // Only a completed source + definitively admitted
                            // handoff may enter its normal tool continuation.
                            // Missing budget after a failure is a tombstone,
                            // never permission to retry without capture.
                            if slot.resolved
                                && (!slot.handoff_admitted
                                    || !slot.lease.as_ref().is_some_and(|lease| {
                                        lease.state() == SourceCaptureState::Captured
                                    }))
                            {
                                return Err(anyhow::anyhow!(failure()));
                            }
                            return Ok(None);
                        };
                        slot.resolved = true;
                        budget
                    };
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
                        )
                        .await
                        .map_err(|_| anyhow::anyhow!(failure()))?;
                    this.source
                        .lock()
                        .map_err(|_| anyhow::anyhow!(failure()))?
                        .lease = Some(lease.clone());
                    Ok(Some(lease))
                })
            }),
            completed: Arc::new(move |completion| {
                let this = complete.clone();
                Box::pin(async move {
                    this.check_model(&completion.model)
                        .map_err(anyhow::Error::msg)?;
                    // Each refusal names its check: live TS14 evidence showed
                    // one generic message is not diagnosable from evidence.
                    let bindings = completion.lease.tool_bindings();
                    if completion.lease.state() != SourceCaptureState::Captured {
                        return Err(anyhow::anyhow!(format!("{} [occ3_completed_state_not_captured]", failure())));
                    }
                    if bindings.len() != 1 {
                        return Err(anyhow::anyhow!(format!("{} [occ3_completed_binding_count]", failure())));
                    }
                    if bindings[0].tool_name != NATIVE_HANDOFF_TOOL {
                        return Err(anyhow::anyhow!(format!("{} [occ3_completed_tool_name]", failure())));
                    }
                    let slot = this
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
                    Ok(())
                })
            }),
        }
    }
    pub fn receiver_resolver(self: &Arc<Self>) -> ResolveInferenceState {
        let this = self.clone();
        Arc::new(move |request| {
            let this = this.clone();
            Box::pin(async move {
                this.check_model(&request.model)
                    .map_err(anyhow::Error::msg)?;
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
                let envelope = this
                    .inbox
                    .lock()
                    .map_err(|_| anyhow::anyhow!(failure()))?
                    .claim(&ids, now())
                    .map_err(|_| anyhow::anyhow!(failure()))?;
                let Some(envelope) = envelope else {
                    return Ok(None);
                };
                this.check().map_err(anyhow::Error::msg)?;
                if !this.registry.supports(
                    this.connector.connector_id(),
                    envelope.latent_method,
                    &envelope.model_fingerprint,
                ) {
                    return Err(anyhow::anyhow!(failure()));
                }
                let lease = this
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
                    .map_err(|_| anyhow::anyhow!(failure()))?;
                this.inbox
                    .lock()
                    .map_err(|_| anyhow::anyhow!(failure()))?
                    .bind_lease(&lease, &this.enrollment.identity)
                    .map_err(|_| anyhow::anyhow!(failure()))?;
                Ok(Some(lease))
            })
        })
    }
    /// Called only from the controller's authenticated acceptance path. Returns
    /// false for duplicate/tombstone; it must not schedule another receiver run.
    pub fn admit(&self, prepared: &PreparedNativeHandoff) -> Result<bool, String> {
        self.check()?;
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
    pub fn handoff_tool(
        self: &Arc<Self>,
        controller: Arc<dyn AgentSessionMessageController>,
    ) -> AgentTool {
        let this = self.clone();
        AgentTool {
            name:NATIVE_HANDOFF_TOOL.into(),label:"Experimental native state handoff".into(),
            description:"Transfer the captured source state to one authorized family agent. Do not include source facts in arguments.".into(),
            parameters:json!({"type":"object","additionalProperties":false,"properties":{"receiver_role":{"type":"string","enum":["parent","sibling","child"]},"receiver_name":{"type":"string"}},"required":["receiver_role","receiver_name"]}),
            prepare_arguments:None,execution_mode:None,
            execute:Arc::new(move |tool_id,args,signal,_update| {
                let this=this.clone(); let controller=controller.clone(); Box::pin(async move {
                    this.check().map_err(anyhow::Error::msg)?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { return Err(anyhow::anyhow!(failure())); }
                    let object=args.as_object().ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_args_not_object]", failure())))?;
                    if object.len()!=2 { return Err(anyhow::anyhow!(format!("{} [occ3_tool_args_len]", failure()))); }
                    let role=object.get("receiver_role").and_then(Value::as_str).ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_role_missing]", failure())))?;
                    let name=object.get("receiver_name").and_then(Value::as_str).filter(|v|!v.is_empty()&&v.len()<=128).ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_name_missing]", failure())))?;
                    let source={let mut slot=this.source.lock().map_err(|_|anyhow::anyhow!(format!("{} [occ3_tool_source_lock]", failure())))?;
                        let source=slot.lease.clone().ok_or_else(||anyhow::anyhow!(format!("{} [occ3_tool_source_lease_missing]", failure())))?;
                        source.validate_tool(&tool_id,NATIVE_HANDOFF_TOOL,&args).map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_validate_tool: {e}]", failure())))?;
                        if slot.tool_claimed {return Err(anyhow::anyhow!(format!("{} [occ3_tool_already_claimed]", failure())));} slot.tool_claimed=true; source};
                    // Family authorization precedes publication and creates the
                    // real transport ID only after actual source completion.
                    let authorized=controller.authorize_native_handoff(role.into(),name.into()).await.map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_authorize: {e}]", failure())))?;
                    if authorized.source_session_id()!=this.session_id() { return Err(anyhow::anyhow!(format!("{} [occ3_tool_sender_mismatch]", failure()))); }
                    this.check().map_err(anyhow::Error::msg)?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(failure())); }
                    let expected_message_id=authorized.message_id().to_string();
                    let expected_receiver_id=authorized.target_session_id().to_string();
                    let publication=this.connector.publish_native_source(NativePublicationRequest {source:source.clone(),
                        receiver_session_id:authorized.target_session_id().into(),message_id:authorized.message_id().into(),
                        publication_id:format!("publication-{}",uuid::Uuid::new_v4()),tool_call_id:tool_id,tool_name:NATIVE_HANDOFF_TOOL.into(),tool_arguments:args}).await
                        .map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_publish: {e}]", failure())))?;
                    this.source.lock().map_err(|_|anyhow::anyhow!(format!("{} [occ3_tool_publish_lock]", failure())))?.published_message=Some((publication.envelope.message_id.clone(),publication.binding_digest.clone()));
                    this.check().map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_check_publish: {e}]", failure())))?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(format!("{} [occ3_tool_signal_cancel]", failure()))); }
                    let prepared=PreparedNativeHandoff::new(authorized,publication).map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_prepared: {e}]", failure())))?;
                    let receipt=controller.send_prepared_native_handoff(prepared).await.map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_send: {e}]", failure())))?;
                    if receipt.id!=expected_message_id || receipt.target.session_id!=expected_receiver_id
                        || !matches!(receipt.delivery_status.as_str(),"delivered"|"queued")
                    { return Err(anyhow::anyhow!(format!("{} [occ3_tool_receipt_mismatch]", failure()))); }
                    this.check().map_err(|e|anyhow::anyhow!(format!("{} [occ3_tool_check_final: {e}]", failure())))?;
                    if signal.as_ref().is_some_and(|s|s.is_cancelled()) { source.cancel(); return Err(anyhow::anyhow!(failure())); }
                    // Admission is not consumption proof. The isolated native
                    // controller separately requires Consumed; the real daemon
                    // may truthfully acknowledge a queued admission only.
                    this.source.lock().map_err(|_|anyhow::anyhow!(failure()))?.handoff_admitted=true;
                    Ok(AgentToolResult::new(vec![ContentBlock::text(format!("Native handoff admitted: {}",receipt.id))],
                        json!({"message_id":receipt.id,"delivery_status":receipt.delivery_status})))
                })
            }),
        }
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
                "intent_digest":lease.intent().digest().ok(),"state":format!("{:?}",lease.state()),
                "receipt":receipt.as_ref().map(|r|json!({"native_rid":r.native_rid,"response_id":r.response_id,
                    "capture_commit_id":r.capture_commit_id,"handle_sha256":pi_ai::source_capture::sha256(r.source_handle.as_bytes()),
                    "output_sha256":r.output_sha256,"retained_count":r.retained_count,
                    "accepted_span_start":r.accepted_span_start,"accepted_span_end":r.accepted_span_end,"logical_prefix_bytes":r.byte_count,
                    "ready_ranks":r.ready_ranks,"expected_ranks":r.expected_ranks,"layout_digest":r.layout_digest})),
                "tool_claimed":slot.tool_claimed,"handoff_admitted":slot.handoff_admitted,"published_message":slot.published_message,
                "negative_control_armed":slot.negative_control_armed,"negative_control_exported":slot.negative_control_exported,
            })),
            "receiver":self.inbox.lock().expect("native receiver state poisoned").audit()})
    }

    pub fn invalidate(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.invalidate_all();
        }
        if let Ok(mut source) = self.source.lock() {
            source.budget = None;
            if let Some(lease) = &source.lease {
                lease.cancel();
            }
        }
        // The ready cache is a cache of THIS connector's live handshake:
        // once the session state is invalid, a cached ready must not be
        // served to the next activation as if nothing had changed.
        crate::core::occ3_connectors::activation::invalidate_ready_cache();
        // No restamp may run for dead receiver state.
        if let Ok(mut guard) = self.restamp_unsubscribe.lock() {
            if let Some(unsubscribe) = guard.take() {
                unsubscribe();
            }
        }
    }
    /// Revocation stays in memory; best-effort runtime cleanup is awaited by the
    /// isolated controller. It never restarts the model or changes settings.
    pub async fn close(&self) -> Result<(), String> {
        self.invalidate();
        let source = self.source.lock().map_err(|_| failure())?.lease.clone();
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
    use pi_agent_core::types::InferenceStateRequest;
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

    /// Drives the real resolve hook once so the slot holds a Prepared lease.
    async fn resolve_capture(session: &Arc<Occ3NativeSession>) {
        let hooks = session.source_hooks();
        let lease = (hooks.resolve)(InferenceStateRequest {
            model: fixture_model(),
            context: Default::default(),
            messages: Vec::new(),
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
            output_sha256: "d".repeat(64),
            expires_at_epoch: now() + 600,
            byte_count: 128,
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
