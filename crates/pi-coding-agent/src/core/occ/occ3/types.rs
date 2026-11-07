//! OCC-3 core types: model identity, latent-state handles, envelopes and
//! delivery status.
//!
//! OCC-3 is true latent communication between agents running the SAME
//! EXACT model on a compatible local runtime. The generic bridge in this
//! tree is engine-agnostic: it never sees tensors, GPU pointers or
//! runtime-specific protocols. A [`super::connector::Occ3Connector`]
//! implementation owns everything runtime-specific.
//!
//! Purity rules (same as the rest of the OCC tree): no network, no model
//! APIs, no filesystem access, no clocks. Everything is passed in.

use serde::{Deserialize, Serialize};

/// Default server-side latent-state TTL (15 minutes), mirroring the
/// plugin's default channel TTL.
pub const DEFAULT_CAPTURE_TTL_SECONDS: u64 = 900;
use std::collections::BTreeMap;

/// Canonical identity of a model serving runtime that can host latent
/// states. Two agents may use OCC-3 only when their identities compare
/// equal under the strict same-model rule.
///
/// Identical display names prove nothing. The connector is responsible
/// for verifying these fields against the live runtime, not trusting a
/// JSON declaration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Occ3ModelIdentity {
    /// Canonical provider model id (e.g. `glm-5.3`).
    pub model_id: String,
    /// Architecture identifier (e.g. `GlmMoeDsaForCausalLM`).
    pub architecture: String,
    /// Checkpoint revision (e.g. git sha of the weights).
    pub checkpoint_revision: String,
    /// Fingerprint of the weight files (hash over shard names+sizes at
    /// minimum; the connector decides the exact digest).
    pub weight_fingerprint: String,
    /// Tokenizer identity (e.g. tokenizer hash or revision).
    pub tokenizer_fingerprint: String,
    /// Chat-template identity where the runtime applies one.
    pub chat_template_fingerprint: String,
    /// Quantisation representation (e.g. `fp8-e4m3`).
    pub quantisation: String,
    /// Hidden size / tensor geometry summary, connector-defined.
    pub tensor_geometry: String,
    /// Attention/KV layout summary, connector-defined.
    pub kv_layout: String,
    /// Latent method version the runtime implements (e.g. `statebridge-prefix-v1`).
    pub latent_method: String,
    /// Serving runtime identity (engine name + version).
    pub runtime: String,
}

impl Occ3ModelIdentity {
    /// Stable fingerprint string for logs and capability advertisements.
    /// Contains no secrets: only public identity components.
    pub fn fingerprint(&self) -> String {
        // Canonical JSON, not delimiter concatenation. The exact eleven named
        // fields are shared with the controller's independently attested mapping.
        // This digest is a summary; complete identity equality is still required.
        let value = serde_json::to_value(self).expect("model identity serializes");
        let canonical = pi_ai::source_capture::canonical_json_bytes(&value)
            .expect("model identity contains only JSON strings");
        format!("sha256:{}", pi_ai::source_capture::sha256(&canonical))
    }
}

/// Reference to a latent state held server-side. Opaque by contract:
/// no CUDA pointers, no tensor bytes, no file paths, no credentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occ3HandleRef {
    /// Unguessable handle id issued by the connector.
    pub handle_id: String,
    /// Connector that owns the state.
    pub connector_id: String,
    /// Model identity fingerprint the state is bound to.
    pub model_fingerprint: String,
    /// Monotonic runtime generation; changes invalidate handles.
    pub runtime_generation: u64,
    /// Unix epoch seconds after which the state expires.
    pub expires_at_epoch: u64,
    /// Approximate server-side state size class in bytes.
    pub state_size_bytes: u64,
}

/// Delivery lifecycle of one OCC-3 transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Occ3DeliveryStatus {
    /// Sender requested capture; not yet confirmed.
    Prepared,
    /// Connector confirmed the latent state exists server-side.
    Captured,
    /// Envelope admitted to the receiver's message queue.
    Admitted,
    /// Receiver's connector accepted the handoff authorisation.
    Delivered,
    /// Receiver confirmed the latent state was consumed.
    Consumed,
    /// Receiver acknowledged semantic use of the state.
    Acknowledged,
    /// Terminal failure; fallback already engaged or required.
    Failed,
    /// Expired before consumption.
    Expired,
    /// Outcome unknown (timeout/lost reply); treat as at-most-once.
    Uncertain,
}

/// Supported latent transfer methods for OCC-3 v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Occ3Method {
    /// Continuous latent prefix in input-embedding space
    /// (StateBridge-style alignment).
    StatebridgePrefix,
    /// KV-cache working-memory handoff (LatentMAS-style).
    KvHandoff,
}

impl Occ3Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Occ3Method::StatebridgePrefix => "statebridge-prefix",
            Occ3Method::KvHandoff => "kv-handoff",
        }
    }
}

/// The OCC-3 envelope carried inside an ordinary agent message body.
///
/// The envelope NEVER contains tensors: it references a server-side
/// handle and carries the compact OCC-1 control summary that keeps the
/// message useful even after the handle expires or the receiver falls
/// back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occ3Envelope {
    /// Fixed schema discriminator: `optimus.occ3.v1`.
    pub schema: String,
    /// Protocol version (1).
    pub protocol_version: u32,
    /// Stable message id (matches the agent message id).
    pub message_id: String,
    /// Sender session id.
    pub sender_session_id: String,
    /// Receiver session id (may be a scope id for sibling broadcasts).
    pub receiver_session_id: String,
    /// Model identity fingerprint both sides negotiated.
    pub model_fingerprint: String,
    /// Runtime identity string (engine + version + generation).
    pub runtime_id: String,
    /// Connector id that holds the state.
    pub connector_id: String,
    /// Latent method used for this transfer.
    pub latent_method: Occ3Method,
    /// Opaque server-side state reference.
    pub state_handle: Occ3HandleRef,
    /// Compact OCC-1 control summary; always present so fallback never
    /// loses the task identity. Bounded to `max_summary_chars`.
    pub control_summary: String,
    /// Fallback reference: which protocol handled the control plane.
    pub fallback_reference: String,
}

/// Bounded default for the embedded control summary.
pub const OCC3_MAX_SUMMARY_CHARS: usize = 2048;

/// Error taxonomy for OCC-3 bridge operations. All variants are
/// fallback-classified by [`super::negotiation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Occ3Error {
    /// Model identities do not match (strict same-model rule).
    IncompatibleModel {
        sender_fingerprint: String,
        receiver_fingerprint: String,
    },
    /// The receiver is not authorised for this handle.
    Unauthorized,
    /// The connector rejected the operation as unreachable.
    ConnectorUnreachable { connector_id: String },
    /// The connector timed out.
    ConnectorTimeout {
        connector_id: String,
        timeout_ms: u64,
    },
    /// The handle expired or was evicted server-side.
    HandleExpired { handle_id: String },
    /// The handle was already consumed (duplicate delivery attempt).
    HandleAlreadyConsumed { handle_id: String },
    /// Runtime generation changed; all handles invalidated.
    RuntimeGenerationChanged { expected: u64, current: u64 },
    /// Connector-reported capture/consume failure.
    TransferFailed { reason: String },
    /// The connector cannot attach state to an actual native inference.
    NativeInferenceUnsupported { connector_id: String },
    /// The connector refused to prepare the receiver's native inference as
    /// busy before any dispatch: a synchronous admission-unavailability
    /// decline (the control path's busy/quota gate; on sessions/enroll the
    /// real decline is quota). Proven non-use - nothing was generated and
    /// the attachment was never submitted - so the receiver retires its
    /// inbox claim terminally and routes the busy signal to the sender,
    /// whose fresh-capture re-arm becomes the sole retry path. Never a
    /// replay permit.
    NativeInferenceBusyRefused { connector_id: String },
    /// A native inference may have accepted the state. Never replay as text.
    InferenceOutcomeUncertain { message_id: String },
    /// OCC-3 disabled by configuration or policy.
    Disabled,
    /// No connector registered for the requested id.
    UnknownConnector { connector_id: String },
}

impl std::fmt::Display for Occ3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Occ3Error::IncompatibleModel { .. } => write!(f, "occ3: incompatible model"),
            Occ3Error::Unauthorized => write!(f, "occ3: unauthorized"),
            Occ3Error::ConnectorUnreachable { connector_id } => {
                write!(f, "occ3: connector unreachable: {connector_id}")
            }
            Occ3Error::ConnectorTimeout { connector_id, .. } => {
                write!(f, "occ3: connector timeout: {connector_id}")
            }
            Occ3Error::HandleExpired { handle_id } => {
                write!(f, "occ3: handle expired: {handle_id}")
            }
            Occ3Error::HandleAlreadyConsumed { handle_id } => {
                write!(f, "occ3: handle already consumed: {handle_id}")
            }
            Occ3Error::RuntimeGenerationChanged { .. } => {
                write!(f, "occ3: runtime generation changed")
            }
            Occ3Error::TransferFailed { reason } => write!(f, "occ3: transfer failed: {reason}"),
            Occ3Error::NativeInferenceUnsupported { connector_id } => {
                write!(
                    f,
                    "occ3: native inference attachment unsupported: {connector_id}"
                )
            }
            Occ3Error::NativeInferenceBusyRefused { connector_id } => {
                write!(
                    f,
                    "occ3: native inference busy refused: {connector_id}"
                )
            }
            Occ3Error::InferenceOutcomeUncertain { message_id } => {
                write!(f, "occ3: inference outcome uncertain: {message_id}")
            }
            Occ3Error::Disabled => write!(f, "occ3: disabled"),
            Occ3Error::UnknownConnector { connector_id } => {
                write!(f, "occ3: unknown connector: {connector_id}")
            }
        }
    }
}

impl std::error::Error for Occ3Error {}

/// Capability snapshot a connector reports after a live handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occ3ConnectorCapabilities {
    /// OCC-3 protocol version the connector implements.
    pub protocol_version: u32,
    /// Connector implementation version.
    pub connector_version: String,
    /// Model identity as verified live by the connector.
    pub model_identity: Occ3ModelIdentity,
    /// Supported latent methods.
    pub methods: Vec<Occ3Method>,
    /// Supported tensor transfer formats (connector-defined ids).
    pub tensor_formats: Vec<String>,
    /// Maximum latent-state size in bytes.
    pub max_state_bytes: u64,
    /// Server-side state TTL seconds.
    pub state_ttl_seconds: u64,
    /// Concurrency limit for simultaneous live handles.
    pub max_concurrent_handles: u32,
    /// Cache-isolation status string (connector-verified).
    pub cache_isolation: String,
    /// Free-form feature flags the bridge does not interpret.
    pub features: BTreeMap<String, String>,
}

/// Bounded metrics for one OCC-3 transfer attempt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occ3TransferMetrics {
    /// Latent capture wall time in milliseconds.
    pub capture_ms: u64,
    /// Handle handoff/authorisation time in milliseconds.
    pub handoff_ms: u64,
    /// Receiver-side consume time in milliseconds.
    pub consume_ms: u64,
    /// Server-side state size at capture, bytes.
    pub state_bytes: u64,
    /// Total lifetime of the handle until release/expiry, milliseconds.
    pub state_lifetime_ms: u64,
}

#[cfg(test)]
mod identity_digest_tests {
    use super::*;
    #[test]
    fn identity_digest_matches_python_canonical_ascii_fixture_and_not_old_fnv() {
        let names = [
            "model_id",
            "architecture",
            "checkpoint_revision",
            "weight_fingerprint",
            "tokenizer_fingerprint",
            "chat_template_fingerprint",
            "quantisation",
            "tensor_geometry",
            "kv_layout",
            "latent_method",
            "runtime",
        ];
        let value: serde_json::Map<String, serde_json::Value> = names
            .into_iter()
            .map(|key| (key.into(), serde_json::json!(format!("{key}|é🌍"))))
            .collect();
        let identity: Occ3ModelIdentity = serde_json::from_value(value.into()).unwrap();
        assert_eq!(
            identity.fingerprint(),
            "sha256:07fea040f5765b0d7e13d7783b34a970d5ff806977b64c2dc4a2424964429659"
        );
        let mut left = identity.clone();
        left.model_id = "a|b".into();
        left.architecture = "c".into();
        let mut right = identity.clone();
        right.model_id = "a".into();
        right.architecture = "b|c".into();
        assert_ne!(left.fingerprint(), right.fingerprint());
        for name in names {
            let mut changed = serde_json::to_value(&identity).unwrap();
            changed[name] = serde_json::json!("changed");
            let changed: Occ3ModelIdentity = serde_json::from_value(changed).unwrap();
            assert_ne!(identity.fingerprint(), changed.fingerprint());
        }
    }
}
