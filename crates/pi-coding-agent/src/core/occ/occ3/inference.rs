//! Process-local receiver admission and typed native-inference bindings.
//!
//! No transcript hydration: an opaque envelope alone never rearms a consumed
//! attachment after restart. The embedding host must admit it through its
//! authenticated message route and claim it only when that message is in the
//! actual inference context. This module performs no I/O or clock reads.

use std::collections::BTreeMap;

use pi_ai::inference_attachment::{InferenceAttachment, InferenceStateTarget};

use super::types::{
    Occ3DeliveryStatus, Occ3Envelope, Occ3Error, Occ3Method, Occ3ModelIdentity,
    OCC3_MAX_SUMMARY_CHARS,
};

pub const NATIVE_INFERENCE_CONTRACT: &str = "optimus.occ3.inference.v1";
const MAX_RECEIPT_RECORDS: usize = 1024;

/// Runtime identity freshly verified by the connector, not by an envelope.
#[derive(Debug, Clone)]
pub struct VerifiedInferenceRuntime {
    pub model_identity: Occ3ModelIdentity,
    pub runtime_id: String,
    pub generation: u64,
    pub methods: Vec<Occ3Method>,
}

/// Request-bound input to a connector. Target identity must come from the
/// selected native receiver model, never from the sender's text.
#[derive(Debug, Clone)]
pub struct PrepareInferenceRequest {
    pub envelope: Occ3Envelope,
    pub request_id: String,
    pub receiver_session_id: String,
    pub receiver_identity: Occ3ModelIdentity,
    pub target: InferenceStateTarget,
    pub now_epoch: u64,
}

fn invalid(reason: &str) -> Occ3Error {
    Occ3Error::TransferFailed {
        reason: reason.to_string(),
    }
}

fn bounded_id(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max.min(256)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-/".contains(&byte))
}

pub(crate) fn complete_identity(identity: &Occ3ModelIdentity) -> bool {
    // Informational provenance fields may be honestly empty: the sglang
    // extension degrades an unknown checkpoint provenance and a chat
    // template kept outside tokenizer_config.json to "" (occ3_sglang/
    // identity.py). Every other identity field still requires a
    // non-empty, bounded, non-placeholder attested value.
    let bounded = |value: &str| {
        value.len() <= 4096
            && !value.chars().any(char::is_control)
            && !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "unknown" | "unverified" | "none"
            )
    };
    let informational = [
        &identity.checkpoint_revision,
        &identity.chat_template_fingerprint,
    ];
    let hard = [
        &identity.model_id,
        &identity.architecture,
        &identity.weight_fingerprint,
        &identity.tokenizer_fingerprint,
        &identity.quantisation,
        &identity.tensor_geometry,
        &identity.kv_layout,
        &identity.latent_method,
        &identity.runtime,
    ];
    informational
        .into_iter()
        .all(|value| bounded(value))
        && hard
            .into_iter()
            .all(|value| bounded(value) && !value.trim().is_empty())
}

fn validate_envelope(envelope: &Occ3Envelope, now_epoch: u64) -> Result<(), Occ3Error> {
    if envelope.schema != "optimus.occ3.v1" || envelope.protocol_version != 1 {
        return Err(invalid("unsupported inference envelope"));
    }
    if !bounded_id(&envelope.message_id, 128)
        || !bounded_id(&envelope.sender_session_id, 128)
        || !bounded_id(&envelope.receiver_session_id, 128)
        || !bounded_id(&envelope.connector_id, 128)
        || !bounded_id(&envelope.runtime_id, 256)
        || !bounded_id(&envelope.model_fingerprint, 4096)
        || !bounded_id(&envelope.state_handle.handle_id, 256)
        || envelope.control_summary.chars().count() > OCC3_MAX_SUMMARY_CHARS
        || !matches!(
            envelope.fallback_reference.as_str(),
            "occ1" | "occ2" | "text"
        )
    {
        return Err(invalid("invalid inference envelope bounds"));
    }
    if envelope.connector_id != envelope.state_handle.connector_id
        || envelope.model_fingerprint != envelope.state_handle.model_fingerprint
    {
        return Err(invalid("inconsistent inference handle binding"));
    }
    if envelope.state_handle.expires_at_epoch <= now_epoch {
        return Err(Occ3Error::HandleExpired {
            handle_id: envelope.state_handle.handle_id.clone(),
        });
    }
    Ok(())
}

/// Validate all supplied identities before preparing provider metadata. A short
/// fingerprint is only a consistency check; the complete receiver and runtime
/// model identities must compare equal.
pub fn inference_attachment(
    request: &PrepareInferenceRequest,
    connector_id: &str,
    runtime: &VerifiedInferenceRuntime,
) -> Result<InferenceAttachment, Occ3Error> {
    let envelope = &request.envelope;
    validate_envelope(envelope, request.now_epoch)?;
    if request.receiver_session_id != envelope.receiver_session_id {
        return Err(Occ3Error::Unauthorized);
    }
    if connector_id != envelope.connector_id {
        return Err(Occ3Error::UnknownConnector {
            connector_id: envelope.connector_id.clone(),
        });
    }
    if !complete_identity(&request.receiver_identity)
        || !complete_identity(&runtime.model_identity)
        || request.receiver_identity != runtime.model_identity
        || request.target.model_id != runtime.model_identity.model_id
        || envelope.model_fingerprint != runtime.model_identity.fingerprint()
    {
        return Err(Occ3Error::IncompatibleModel {
            sender_fingerprint: envelope.model_fingerprint.clone(),
            receiver_fingerprint: runtime.model_identity.fingerprint(),
        });
    }
    if envelope.state_handle.runtime_generation != runtime.generation {
        return Err(Occ3Error::RuntimeGenerationChanged {
            expected: envelope.state_handle.runtime_generation,
            current: runtime.generation,
        });
    }
    if envelope.runtime_id != runtime.runtime_id {
        return Err(invalid("inference runtime access domain mismatch"));
    }
    if !runtime.methods.contains(&envelope.latent_method) {
        return Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: connector_id.to_string(),
        });
    }
    if !bounded_id(&request.request_id, 128) {
        return Err(invalid("invalid native inference request id"));
    }
    Ok(InferenceAttachment {
        protocol_version: 1,
        request_id: request.request_id.clone(),
        message_id: envelope.message_id.clone(),
        connector_id: envelope.connector_id.clone(),
        receiver_session_id: request.receiver_session_id.clone(),
        model_fingerprint: envelope.model_fingerprint.clone(),
        runtime_id: runtime.runtime_id.clone(),
        runtime_generation: runtime.generation,
        state_handle: envelope.state_handle.handle_id.clone(),
        expires_at_epoch: envelope.state_handle.expires_at_epoch,
        method: runtime.model_identity.latent_method.clone(),
    })
}

#[derive(Debug, Clone)]
struct Admission {
    envelope: Occ3Envelope,
    claimed: bool,
    bound_lease: Option<pi_ai::inference_attachment::InferenceStateLease>,
    status: Occ3DeliveryStatus,
    /// Monotonic arrival order assigned by `admit`.
    admission_seq: u64,
    /// Whether `poll_consumed` already reported this record's completion.
    consumed_notified: bool,
    /// Whether `poll_terminal_failures` already reported this record's
    /// terminal non-use outcome.
    terminal_notified: bool,
}

/// Bounded session-local inbox. Only the authenticated host calls `admit`.
/// It has no serde implementation and cannot be restored from a transcript.
/// Multiple pending admissions queue in arrival order (`admission_seq`):
/// exactly one lease binds per native inference request, because the
/// resolver claims the earliest admitted message still present in the
/// receiver's inference context (context order, which the message
/// transport keeps in arrival order). A second outstanding admission is
/// therefore never attached-and-lost; it stays `Admitted` until its own
/// claim, or a terminal invalidation. Consumption stays one-at-a-time and
/// is journaled per message id.
#[derive(Debug)]
pub struct Occ3InferenceInbox {
    receiver_session_id: String,
    records: BTreeMap<String, Admission>,
    admissions: u64,
}

impl Occ3InferenceInbox {
    pub fn new(receiver_session_id: String) -> Result<Self, Occ3Error> {
        if !bounded_id(&receiver_session_id, 128) {
            return Err(Occ3Error::Unauthorized);
        }
        Ok(Self {
            receiver_session_id,
            records: BTreeMap::new(),
            admissions: 0,
        })
    }

    /// Returns false for an identical duplicate, including after expiry. Never
    /// replay duplicates or conflicting existing IDs through a text fallback.
    /// Only rejection of a genuinely new ID is a pre-admission fallback boundary.
    ///
    /// Pending admissions queue: a genuinely new id is admitted even while
    /// another delivery is outstanding (pending or actively consuming), so
    /// multiple senders may queue on one receiver. The one-lease-per-request
    /// invariant is not weakened here; it is enforced by `claim`, which
    /// attaches at most one state per native inference. The bound is the
    /// total record count; the message transport separately bounds pending
    /// agent messages per session.
    pub fn admit(
        &mut self,
        envelope: Occ3Envelope,
        authenticated_message_id: &str,
        authenticated_sender_session_id: &str,
        now_epoch: u64,
    ) -> Result<bool, Occ3Error> {
        if envelope.receiver_session_id != self.receiver_session_id
            || envelope.message_id != authenticated_message_id
            || envelope.sender_session_id != authenticated_sender_session_id
        {
            return Err(Occ3Error::Unauthorized);
        }
        if let Some(record) = self.records.get(&envelope.message_id) {
            return if record.envelope == envelope {
                Ok(false)
            } else {
                Err(Occ3Error::Unauthorized)
            };
        }
        validate_envelope(&envelope, now_epoch)?;
        if self.records.len() >= MAX_RECEIPT_RECORDS {
            return Err(invalid("receiver inference admission capacity"));
        }
        self.admissions = self.admissions.saturating_add(1);
        self.records.insert(
            envelope.message_id.clone(),
            Admission {
                envelope,
                claimed: false,
                bound_lease: None,
                status: Occ3DeliveryStatus::Admitted,
                admission_seq: self.admissions,
                consumed_notified: false,
                terminal_notified: false,
            },
        );
        Ok(true)
    }

    /// Claim the one delivery this native inference may attach, chosen by the
    /// inbox, never by prompt ordering. `included_message_ids` are the ids
    /// extracted from the authenticated messages included in the actual
    /// native inference; arbitrary prompt text is never scanned for handles.
    ///
    /// Ordering and serialization rules:
    /// - Receiver-global single active consumption: while ANY record is
    ///   claimed and not terminal (still attached, or its lease outcome is
    ///   unknown), `claim` refuses loudly with
    ///   [`Occ3Error::InferenceOutcomeUncertain`]. This holds even when that
    ///   record's id is absent from `included_message_ids` (at-most-once:
    ///   a dropped-from-context in-flight delivery must not be shadowed by a
    ///   later claim).
    /// - True arrival FIFO: the only claimable record is the earliest
    ///   still-pending admission by `admission_seq`, independent of the
    ///   context order of `included_message_ids`.
    /// - An absent earliest pending admission never lets a later pending
    ///   admission overtake silently: if the earliest pending id is not in
    ///   `included_message_ids`, nothing is claimed (`Ok(None)`, no
    ///   attachment, no error) and every pending record stays pending until
    ///   its message returns to the context or a terminal invalidation
    ///   (model switch, queue clear) resolves the queue.
    /// - Pending envelope expiry is enforced independently of inclusion: an
    ///   expired pending record is journaled terminal (`Expired`, preserved)
    ///   whether or not its id is in the context, so a dead entry never
    ///   parks the queue. A still-valid earliest entry is never dropped or
    ///   overtaken. When the expired record's id IS in the context the
    ///   expiry is also reported loudly (`HandleExpired`); otherwise it is
    ///   reported only through `poll_terminal_failures`.
    /// - Terminal outcomes (`Consumed`, and `Failed`/`Expired` leases whose
    ///   non-use is proven) are journaled and skipped: they never re-arm and
    ///   never block later pending deliveries.
    pub fn claim(
        &mut self,
        included_message_ids: &[String],
        newly_supplied_ids: &[String],
        now_epoch: u64,
    ) -> Result<Option<Occ3Envelope>, Occ3Error> {
        for record in self.records.values_mut() {
            if !record.claimed {
                continue;
            }
            if let Some(lease) = record.bound_lease.as_ref() {
                record.status = lease_outcome(lease);
            }
            match record.status {
                Occ3DeliveryStatus::Consumed
                | Occ3DeliveryStatus::Failed => continue,
                Occ3DeliveryStatus::Expired => {
                    // A resubmitted expired delivery refuses loudly BEFORE
                    // the terminal skip: `newly_supplied_ids` are the ids of
                    // the records the CURRENT turn newly supplied (the
                    // resolver's fresh-input provenance), never mere history.
                    // A dead entry that is only RETAINED history stays silent
                    // (it never parks the queue); only its re-presentation
                    // refuses. This also catches an entry that expired
                    // SILENTLY on an earlier turn: it is already claimed, so
                    // it never re-enters the discovery loop below - the
                    // fresh-supplied check here is the only refusal point.
                    if newly_supplied_ids.contains(&record.envelope.message_id) {
                        return Err(Occ3Error::HandleExpired {
                            handle_id: record.envelope.state_handle.handle_id.clone(),
                        });
                    }
                    continue;
                }
                _ => {
                    return Err(Occ3Error::InferenceOutcomeUncertain {
                        message_id: record.envelope.message_id.clone(),
                    })
                }
            }
        }
        loop {
            let Some(earliest) = self
                .records
                .values()
                .filter(|record| !record.claimed && record.status == Occ3DeliveryStatus::Admitted)
                .min_by_key(|record| record.admission_seq)
                .map(|record| record.envelope.message_id.clone())
            else {
                return Ok(None);
            };
            let expired = {
                let record = self
                    .records
                    .get(&earliest)
                    .expect("earliest pending record exists");
                record.envelope.state_handle.expires_at_epoch <= now_epoch
            };
            if !expired {
                if !included_message_ids.contains(&earliest) {
                    return Ok(None);
                }
                let record = self
                    .records
                    .get_mut(&earliest)
                    .expect("earliest pending record exists");
                record.claimed = true;
                record.status = Occ3DeliveryStatus::Delivered;
                return Ok(Some(record.envelope.clone()));
            }
            // Journal the expired pending record (preserved, terminal).
            // A dead entry never parks the queue: this call claims
            // nothing, the journal surfaces through
            // poll_terminal_failures, and the next claim proceeds to the
            // live head. When the expired id is in the context the expiry
            // is reported loudly instead.
            let record = self
                .records
                .get_mut(&earliest)
                .expect("earliest pending record exists");
            let handle_id = record.envelope.state_handle.handle_id.clone();
            let in_context = included_message_ids.contains(&earliest);
            record.claimed = true;
            record.status = Occ3DeliveryStatus::Expired;
            if in_context {
                return Err(Occ3Error::HandleExpired { handle_id });
            }
            return Ok(None);
        }
    }

    /// Non-authoritative audit projection; never contains reusable handles.
    /// Records are listed in admission (arrival) order, with `admission_seq`
    /// making that order explicit.
    pub fn audit(&self) -> Vec<serde_json::Value> {
        let mut records: Vec<&Admission> = self.records.values().collect();
        records.sort_by_key(|record| record.admission_seq);
        records.iter().map(|record| {
            let id=&record.envelope.message_id;
            let lease=record.bound_lease.as_ref();
            serde_json::json!({"message_id":id,"receiver_session_id":self.receiver_session_id,
                "admission_seq":record.admission_seq,
                "sender_session_id":record.envelope.sender_session_id,"runtime_id":record.envelope.runtime_id,
                "runtime_generation":record.envelope.state_handle.runtime_generation,"model_fingerprint":record.envelope.model_fingerprint,
                "handle_sha256":pi_ai::source_capture::sha256(record.envelope.state_handle.handle_id.as_bytes()),
                "request_id":lease.map(|v|v.attachment().request_id.as_str()),
                "lease_state":lease.map(|v|format!("{:?}",v.state())),
                "status":lease.map(lease_outcome).unwrap_or(record.status)})
        }).collect()
    }

    /// `(message_id, sender_session_id)` pairs whose delivery reached
    /// `Consumed` since the last poll, in admission (arrival) order. This is
    /// the process-local completion notification primitive for a SINGLE
    /// owner dispatcher on the receiver (the restamp listener or its daemon
    /// replacement): it routes each completion to the sender that queued the
    /// delivery. Competing senders must never poll this directly. It changes
    /// no wire format and never returns envelopes or handles. Each completion
    /// is reported exactly once per admission.
    pub fn poll_consumed(&mut self) -> Vec<(String, String)> {
        let mut records: Vec<&mut Admission> = self.records.values_mut().collect();
        records.sort_by_key(|record| record.admission_seq);
        records
            .into_iter()
            .filter_map(|record| {
                let refreshed = record.bound_lease.as_ref().map(lease_outcome);
                if let Some(refreshed) = refreshed {
                    record.status = refreshed;
                }
                if record.status == Occ3DeliveryStatus::Consumed && !record.consumed_notified {
                    record.consumed_notified = true;
                    Some((
                        record.envelope.message_id.clone(),
                        record.envelope.sender_session_id.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn status(&self, message_id: &str) -> Option<Occ3DeliveryStatus> {
        self.records.get(message_id).map(|record| record.status)
    }

    /// `(message_id, sender_session_id, status)` triples for deliveries
    /// that reached a terminal NON-USE outcome (`Failed` or `Expired`)
    /// since the last poll, in admission (arrival) order. The drain is the
    /// sibling of [`Self::poll_consumed`]: consumed completions are
    /// reported there, proven non-use terminals here, and a record never
    /// appears in both (the statuses are mutually exclusive). Uncertain
    /// consumption is deliberately NOT reported: it stays an explicit
    /// refusal and must never trigger a duplicate fallback delivery.
    /// Lease outcomes are refreshed first. The single dispatcher owner
    /// drains both polls at the same call site (occ3_state.rs restamp
    /// listener); senders never poll either directly.
    pub fn poll_terminal_failures(&mut self) -> Vec<(String, String, Occ3DeliveryStatus)> {
        let mut records: Vec<&mut Admission> = self.records.values_mut().collect();
        records.sort_by_key(|record| record.admission_seq);
        records
            .into_iter()
            .filter_map(|record| {
                let refreshed = record.bound_lease.as_ref().map(lease_outcome);
                if let Some(refreshed) = refreshed {
                    record.status = refreshed;
                }
                if matches!(
                    record.status,
                    Occ3DeliveryStatus::Failed | Occ3DeliveryStatus::Expired
                ) && !record.terminal_notified
                {
                    record.terminal_notified = true;
                    Some((
                        record.envelope.message_id.clone(),
                        record.envelope.sender_session_id.clone(),
                        record.status,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Whether this record's bound lease was refused as receiver-busy
    /// before any stream event (proven non-use). The dispatcher reads it for
    /// the triples `poll_terminal_failures` returns, so a busy refusal can
    /// ask the sender for a fresh capture instead of an OCC-1 fallback.
    pub fn refused_busy(&self, message_id: &str) -> bool {
        self.records
            .get(message_id)
            .and_then(|record| record.bound_lease.as_ref())
            .is_some_and(|lease| lease.refused_busy())
    }

    /// Bind exactly one request after claim and before native dispatch. The
    /// complete identity must come from fresh trusted enrollment. A coarse
    /// envelope method family is never a substitute for the actual version.
    pub fn bind_lease(
        &mut self,
        lease: &pi_ai::inference_attachment::InferenceStateLease,
        verified_identity: &Occ3ModelIdentity,
    ) -> Result<(), Occ3Error> {
        use pi_ai::inference_attachment::InferenceStateLeaseState;
        if !complete_identity(verified_identity)
            || !bounded_id(&verified_identity.latent_method, 256)
        {
            return Err(Occ3Error::Unauthorized);
        }
        let attachment = lease.attachment();
        if attachment.model_fingerprint != verified_identity.fingerprint()
            || lease.target().model_id != verified_identity.model_id
        {
            return Err(Occ3Error::Unauthorized);
        }
        if self.records.values().any(|record| {
            record
                .bound_lease
                .as_ref()
                .is_some_and(|bound| bound.attachment().request_id == attachment.request_id)
        }) {
            return Err(Occ3Error::Unauthorized);
        }
        let record = self
            .records
            .get_mut(&attachment.message_id)
            .ok_or(Occ3Error::Unauthorized)?;
        if !record.claimed
            || record.status != Occ3DeliveryStatus::Delivered
            || record.bound_lease.is_some()
            || lease.state() != InferenceStateLeaseState::Prepared
            || attachment.receiver_session_id != self.receiver_session_id
            || attachment.state_handle != record.envelope.state_handle.handle_id
            || attachment.model_fingerprint != record.envelope.model_fingerprint
            || attachment.connector_id != record.envelope.connector_id
            || attachment.runtime_id != record.envelope.runtime_id
            || attachment.runtime_generation != record.envelope.state_handle.runtime_generation
            || attachment.expires_at_epoch != record.envelope.state_handle.expires_at_epoch
            || attachment.method != verified_identity.latent_method
        {
            return Err(Occ3Error::Unauthorized);
        }
        record.bound_lease = Some(lease.clone());
        Ok(())
    }

    /// Observe the exact lease bound before dispatch, not caller-supplied metrics
    /// or a reconstructed request. This is not semantic acknowledgment.
    pub fn record_outcome(&mut self, message_id: &str) -> Result<Occ3DeliveryStatus, Occ3Error> {
        let record = self
            .records
            .get_mut(message_id)
            .ok_or(Occ3Error::Unauthorized)?;
        let lease = record.bound_lease.as_ref().ok_or(Occ3Error::Unauthorized)?;
        record.status = lease_outcome(lease);
        Ok(record.status)
    }

    /// Synchronously revoke all local use during model switch/abort. A submitted
    /// lease becomes uncertain, never a reusable text-fallback permission.
    pub fn invalidate_all(&mut self) {
        for record in self.records.values_mut() {
            record.claimed = true;
            if let Some(lease) = &record.bound_lease {
                lease.cancel();
                record.status = lease_outcome(lease);
            } else if record.status == Occ3DeliveryStatus::Admitted {
                record.status = Occ3DeliveryStatus::Failed;
            }
        }
    }

    /// Cancel unclaimed admissions. Claimed requests require their provider
    /// lease outcome: aborting a request does not prove it was never accepted.
    pub fn cancel_pending(&mut self) -> Vec<Occ3Envelope> {
        let mut cancelled = Vec::new();
        for record in self.records.values_mut() {
            if !record.claimed && record.status == Occ3DeliveryStatus::Admitted {
                record.claimed = true;
                record.status = Occ3DeliveryStatus::Failed;
                cancelled.push(record.envelope.clone());
            }
        }
        cancelled
    }
}

fn lease_outcome(lease: &pi_ai::inference_attachment::InferenceStateLease) -> Occ3DeliveryStatus {
    use pi_ai::inference_attachment::InferenceStateLeaseState;
    match lease.state() {
        InferenceStateLeaseState::Prepared => Occ3DeliveryStatus::Delivered,
        InferenceStateLeaseState::Cancelled => Occ3DeliveryStatus::Failed,
        InferenceStateLeaseState::Consumed => Occ3DeliveryStatus::Consumed,
        InferenceStateLeaseState::Submitted | InferenceStateLeaseState::Uncertain => {
            Occ3DeliveryStatus::Uncertain
        }
    }
}

/// Once message admission happened, a failed/expired/uncertain operation must
/// be journaled. It must not be silently replayed through another protocol.
pub fn permits_text_fallback(status: Occ3DeliveryStatus) -> bool {
    matches!(
        status,
        Occ3DeliveryStatus::Prepared | Occ3DeliveryStatus::Captured
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> crate::core::occ::occ3::types::Occ3ModelIdentity {
        crate::core::occ::occ3::types::Occ3ModelIdentity {
            model_id: "glm-5.3".into(),
            architecture: "GlmMoeDsaForCausalLM".into(),
            checkpoint_revision: "aca966e4".into(),
            weight_fingerprint: "18ad686accf22269".into(),
            tokenizer_fingerprint: "98b1271574f41abf".into(),
            chat_template_fingerprint: "3740abce".into(),
            quantisation: "fp8".into(),
            tensor_geometry: "hidden_size=6144".into(),
            kv_layout: "tp_size=8".into(),
            latent_method: "kv-radix-channel-v1".into(),
            runtime: "sglang/0.5.21".into(),
        }
    }

    #[test]
    fn complete_identity_accepts_honestly_degraded_informational_fields() {
        // The sglang extension degrades checkpoint_revision and
        // chat_template_fingerprint to "" when the provenance is unknown
        // (local checkpoint path without a hub revision; chat template kept
        // outside tokenizer_config.json). A complete identity may carry
        // exactly those two fields empty.
        let mut degraded = identity();
        degraded.checkpoint_revision = String::new();
        degraded.chat_template_fingerprint = String::new();
        assert!(complete_identity(&degraded));
    }

    #[test]
    fn complete_identity_still_refuses_degraded_hard_fields() {
        let base = identity();
        for mutate in [
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.model_id.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.architecture.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.weight_fingerprint.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.tokenizer_fingerprint.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.quantisation.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.tensor_geometry.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.kv_layout.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.latent_method.clear()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.runtime.clear()
            },
        ] {
            let mut broken = base.clone();
            mutate(&mut broken);
            assert!(!complete_identity(&broken));
        }
    }

    fn unit_envelope(message_id: &str, sender: &str) -> Occ3Envelope {
        use crate::core::occ::occ3::types::Occ3HandleRef;
        use crate::core::occ::occ3::types::Occ3Method;
        Occ3Envelope {
            schema: "optimus.occ3.v1".into(),
            protocol_version: 1,
            message_id: message_id.into(),
            sender_session_id: sender.into(),
            receiver_session_id: "receiver".into(),
            model_fingerprint: "fp-1".into(),
            runtime_id: "runtime-1".into(),
            connector_id: "connector-1".into(),
            latent_method: Occ3Method::KvHandoff,
            state_handle: Occ3HandleRef {
                handle_id: format!("handle-{message_id}"),
                connector_id: "connector-1".into(),
                model_fingerprint: "fp-1".into(),
                runtime_generation: 1,
                expires_at_epoch: u64::MAX,
                state_size_bytes: 1,
            },
            control_summary: "summary".into(),
            fallback_reference: "occ1".into(),
        }
    }

    /// `poll_consumed` reports each Consumed delivery exactly once, in
    /// arrival order, and never reports pending, attached or terminal
    /// non-consumed records. Statuses are set through private access to
    /// simulate verified lease outcomes (the public lease API keeps
    /// consumption verification crate-internal to pi-ai).
    #[test]
    fn poll_consumed_reports_each_completed_delivery_exactly_once_in_arrival_order() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        // Arrival order m3, m1, m2 differs from lexical id order on purpose.
        for (id, sender) in [("m3", "s3"), ("m1", "s1"), ("m2", "s2")] {
            assert!(inbox.admit(unit_envelope(id, sender), id, sender, 1).unwrap());
        }
        // m2: terminal failure; m1 and m3: completed use.
        {
            let record = inbox.records.get_mut("m2").unwrap();
            record.claimed = true;
            record.status = Occ3DeliveryStatus::Failed;
        }
        for id in ["m1", "m3"] {
            let record = inbox.records.get_mut(id).unwrap();
            record.claimed = true;
            record.status = Occ3DeliveryStatus::Consumed;
        }
        // A pending record (m4 admitted later) must not appear either.
        assert!(inbox.admit(unit_envelope("m4", "s4"), "m4", "s4", 1).unwrap());
        assert_eq!(
            inbox.poll_consumed(),
            vec![
                ("m3".to_string(), "s3".to_string()),
                ("m1".to_string(), "s1".to_string()),
            ],
            "consumed (id, sender) pairs in arrival order, exactly the consumed ones"
        );
        assert!(
            inbox.poll_consumed().is_empty(),
            "each completion is reported exactly once"
        );
        // A later consumption is reported on the next poll.
        {
            let record = inbox.records.get_mut("m4").unwrap();
            record.claimed = true;
            record.status = Occ3DeliveryStatus::Consumed;
        }
        assert_eq!(
            inbox.poll_consumed(),
            vec![("m4".to_string(), "s4".to_string())]
        );
        assert!(inbox.poll_consumed().is_empty());
    }

    /// Fix 1 (receiver busy): a bound lease closed as proven non-use
    /// (`Cancelled`, the provider's pre-event busy refusal) resolves to
    /// `Failed`. The next claim skips it, so the session's retry runs as an
    /// ordinary turn without the attachment, and the terminal poll reports
    /// it exactly once so the sender falls back to ordinary OCC-1 delivery.
    #[test]
    fn busy_refused_lease_lets_the_retry_run_ordinary_and_reports_failed_once() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        assert!(inbox.admit(unit_envelope("m1", "s1"), "m1", "s1", 1).unwrap());
        assert!(inbox.claim(&["m1".to_string()], &[], 1).unwrap().is_some());
        let lease = pi_ai::inference_attachment::InferenceStateLease::new(
            pi_ai::inference_attachment::InferenceStateTarget {
                provider: "fixture".into(),
                model_id: "fixture-model".into(),
                base_url: "http://127.0.0.1:1/v1".into(),
            },
            pi_ai::inference_attachment::InferenceAttachment {
                protocol_version: 1,
                request_id: "consume-1".into(),
                message_id: "m1".into(),
                connector_id: "connector-1".into(),
                receiver_session_id: "receiver".into(),
                model_fingerprint: "fp-1".into(),
                runtime_id: "runtime-1".into(),
                runtime_generation: 1,
                state_handle: "handle-m1".into(),
                expires_at_epoch: u64::MAX / 2,
                method: "kv-radix-channel-v1".into(),
            },
        )
        .unwrap();
        lease.cancel();
        inbox.records.get_mut("m1").unwrap().bound_lease = Some(lease);
        // The retry turn: no attachment, no uncertain refusal.
        assert!(inbox.claim(&["m1".to_string()], &["m1".to_string()], 1).unwrap().is_none());
        assert_eq!(
            inbox.poll_terminal_failures(),
            vec![("m1".to_string(), "s1".to_string(), Occ3DeliveryStatus::Failed)]
        );
        assert!(inbox.poll_terminal_failures().is_empty(), "reported exactly once");
        assert!(inbox.poll_consumed().is_empty());
    }

    /// Reversed context order cannot reorder the queue: the earliest pending
    /// admission by ARRIVAL is the only claimable record, even when its id
    /// sorts after a later admission's id (arrival order, not id order).
    #[test]
    fn claim_ignores_context_order_and_picks_the_earliest_arrival() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        assert!(inbox.admit(unit_envelope("m2", "s2"), "m2", "s2", 1).unwrap());
        assert!(inbox.admit(unit_envelope("m1", "s1"), "m1", "s1", 1).unwrap());
        // The audit projection lists arrival order, not id order.
        let audit = inbox.audit();
        let ids: Vec<&str> = audit
            .iter()
            .filter_map(|record| record["message_id"].as_str())
            .collect();
        assert_eq!(ids, vec!["m2", "m1"]);
        let claimed = inbox
            .claim(&["m1".to_string(), "m2".to_string()], &[], 1)
            .unwrap()
            .expect("earliest arrival is claimable");
        assert_eq!(claimed.message_id, "m2");
        assert_eq!(inbox.status("m1"), Some(Occ3DeliveryStatus::Admitted));
    }

    /// An active (attached or unknown-outcome) delivery blocks every later
    /// claim receiver-globally, even when its id is absent from the context.
    #[test]
    fn active_consumption_blocks_later_claims_even_when_absent_from_context() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        assert!(inbox.admit(unit_envelope("m1", "s1"), "m1", "s1", 1).unwrap());
        assert!(inbox.admit(unit_envelope("m2", "s2"), "m2", "s2", 1).unwrap());
        assert!(inbox.claim(&["m1".to_string()], &[], 1).unwrap().is_some());
        assert!(matches!(
            inbox.claim(&["m2".to_string()], &[], 1),
            Err(Occ3Error::InferenceOutcomeUncertain { .. })
        ));
    }

    /// A dead (expired) earliest pending entry never parks the queue: its
    /// expiry is journaled independently of inclusion, the record is
    /// preserved terminal, and the later pending delivery is claimable. The
    /// expired notice flows through `poll_terminal_failures`.
    #[test]
    fn expired_earliest_pending_never_parks_the_queue() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        let mut first = unit_envelope("m1", "s1");
        first.state_handle.expires_at_epoch = 2;
        assert!(inbox.admit(first, "m1", "s1", 1).unwrap());
        assert!(inbox.admit(unit_envelope("m2", "s2"), "m2", "s2", 1).unwrap());
        // The expired earliest is absent from the context; the still-valid
        // later pending admission is present.
        assert!(
            inbox.claim(&["m2".to_string()], &[], 5).unwrap().is_none(),
            "nothing is claimed, but the dead entry must not park the queue"
        );
        // The expired record is journaled terminal and preserved.
        assert_eq!(inbox.status("m1"), Some(Occ3DeliveryStatus::Expired));
        // The terminal-failure drain reports it exactly once, with its
        // sender, while the consumed drain stays empty.
        assert_eq!(
            inbox.poll_terminal_failures(),
            vec![("m1".to_string(), "s1".to_string(), Occ3DeliveryStatus::Expired)]
        );
        assert!(inbox.poll_terminal_failures().is_empty());
        assert!(inbox.poll_consumed().is_empty());
        // The later pending delivery is now the earliest and claimable.
        let claimed = inbox
            .claim(&["m2".to_string()], &[], 5)
            .unwrap()
            .expect("still-valid later pending is claimable after the dead entry");
        assert_eq!(claimed.message_id, "m2");
    }

    /// An absent earliest pending admission never lets a later pending
    /// admission overtake silently: nothing is claimed, nothing is lost.
    #[test]
    fn absent_earliest_pending_does_not_overtake() {
        let mut inbox = Occ3InferenceInbox::new("receiver".into()).unwrap();
        assert!(inbox.admit(unit_envelope("m1", "s1"), "m1", "s1", 1).unwrap());
        assert!(inbox.admit(unit_envelope("m2", "s2"), "m2", "s2", 1).unwrap());
        assert!(
            inbox.claim(&["m2".to_string()], &[], 1).unwrap().is_none(),
            "the later pending admission must not be claimed while the earliest is absent"
        );
        assert_eq!(inbox.status("m1"), Some(Occ3DeliveryStatus::Admitted));
        assert_eq!(inbox.status("m2"), Some(Occ3DeliveryStatus::Admitted));
        // Once the earliest returns to the context it is claimable again.
        assert!(inbox.claim(&["m1".to_string()], &[], 1).unwrap().is_some());
    }

    #[test]
    fn complete_identity_never_accepts_placeholders() {
        let base = identity();
        for mutate in [
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.checkpoint_revision = "unknown".into()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.chat_template_fingerprint = "unverified".into()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.checkpoint_revision = "none".into()
            },
            |identity: &mut crate::core::occ::occ3::types::Occ3ModelIdentity| {
                identity.chat_template_fingerprint = "none".into()
            },
        ] {
            let mut broken = base.clone();
            mutate(&mut broken);
            assert!(!complete_identity(&broken));
        }
    }
}
