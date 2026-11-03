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
}

/// Bounded session-local inbox. Only the authenticated host calls `admit`.
/// It has no serde implementation and cannot be restored from a transcript.
/// Multi-handle composition is unsupported; refuse a second outstanding
/// admission rather than attach one and silently lose the other.
#[derive(Debug)]
pub struct Occ3InferenceInbox {
    receiver_session_id: String,
    records: BTreeMap<String, Admission>,
}

impl Occ3InferenceInbox {
    pub fn new(receiver_session_id: String) -> Result<Self, Occ3Error> {
        if !bounded_id(&receiver_session_id, 128) {
            return Err(Occ3Error::Unauthorized);
        }
        Ok(Self {
            receiver_session_id,
            records: BTreeMap::new(),
        })
    }

    /// Returns false for an identical duplicate, including after expiry. Never
    /// replay duplicates or conflicting existing IDs through a text fallback.
    /// Only rejection of a genuinely new ID is a pre-admission fallback boundary.
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
        if self.records.len() >= MAX_RECEIPT_RECORDS
            || self.records.values().any(|record| {
                matches!(
                    record.status,
                    Occ3DeliveryStatus::Admitted
                        | Occ3DeliveryStatus::Delivered
                        | Occ3DeliveryStatus::Uncertain
                )
            })
        {
            return Err(invalid("receiver inference admission capacity"));
        }
        self.records.insert(
            envelope.message_id.clone(),
            Admission {
                envelope,
                claimed: false,
                bound_lease: None,
                status: Occ3DeliveryStatus::Admitted,
            },
        );
        Ok(true)
    }

    /// Claim only IDs extracted from the authenticated messages included in the
    /// actual native inference. Never scan arbitrary prompt text for handles.
    pub fn claim(
        &mut self,
        included_message_ids: &[String],
        now_epoch: u64,
    ) -> Result<Option<Occ3Envelope>, Occ3Error> {
        for message_id in included_message_ids {
            let Some(record) = self.records.get_mut(message_id) else {
                continue;
            };
            if record.claimed {
                if let Some(lease) = &record.bound_lease {
                    record.status = lease_outcome(lease);
                }
                if record.status == Occ3DeliveryStatus::Consumed {
                    continue;
                }
                return Err(Occ3Error::InferenceOutcomeUncertain {
                    message_id: message_id.clone(),
                });
            }
            if record.status != Occ3DeliveryStatus::Admitted {
                return Err(invalid("inference admission requires explicit recovery"));
            }
            record.claimed = true;
            if record.envelope.state_handle.expires_at_epoch <= now_epoch {
                record.status = Occ3DeliveryStatus::Expired;
                return Err(Occ3Error::HandleExpired {
                    handle_id: record.envelope.state_handle.handle_id.clone(),
                });
            }
            record.status = Occ3DeliveryStatus::Delivered;
            return Ok(Some(record.envelope.clone()));
        }
        Ok(None)
    }

    /// Non-authoritative audit projection; never contains reusable handles.
    pub fn audit(&self) -> Vec<serde_json::Value> {
        self.records.iter().map(|(id,record)| {
            let lease=record.bound_lease.as_ref();
            serde_json::json!({"message_id":id,"receiver_session_id":self.receiver_session_id,
                "sender_session_id":record.envelope.sender_session_id,"runtime_id":record.envelope.runtime_id,
                "runtime_generation":record.envelope.state_handle.runtime_generation,"model_fingerprint":record.envelope.model_fingerprint,
                "handle_sha256":pi_ai::source_capture::sha256(record.envelope.state_handle.handle_id.as_bytes()),
                "request_id":lease.map(|v|v.attachment().request_id.as_str()),
                "lease_state":lease.map(|v|format!("{:?}",v.state())),
                "status":lease.map(lease_outcome).unwrap_or(record.status)})
        }).collect()
    }

    pub fn status(&self, message_id: &str) -> Option<Occ3DeliveryStatus> {
        self.records.get(message_id).map(|record| record.status)
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
