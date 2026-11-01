//! The OCC-3 connector contract: the versioned boundary between the pure
//! generic bridge and a runtime-specific latent-state engine.
//!
//! The bridge (this tree) is deterministic and offline-testable. A
//! connector implementation (e.g. the SGLang extension client, living
//! OUTSIDE this tree) performs all I/O. The trait below is the entire
//! contract surface; it never mentions tensors, GPU pointers, CUDA or
//! any specific inference engine.
//!
//! Control-plane only: handles are opaque ids, never payload bytes.

use async_trait::async_trait;
use std::time::Duration;

use super::types::{
    Occ3ConnectorCapabilities, Occ3Error, Occ3HandleRef, Occ3Method, Occ3ModelIdentity,
    Occ3TransferMetrics,
};

/// What a sender asks the connector to prepare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureRequest {
    /// Agent-visible task/message identity for auditing.
    pub message_id: String,
    /// Sender session id.
    pub sender_session_id: String,
    /// The receiver session authorised to consume.
    pub receiver_session_id: String,
    /// Latent method to use.
    pub method: Occ3Method,
    /// Connector-specific capture payload (e.g. prompt context reference).
    /// Small by contract: never tensors.
    pub capture_context: String,
    /// Server-side state TTL requested by the bridge.
    pub ttl: Duration,
}

/// What the receiver asks the connector to do with a delivered handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumeRequest {
    /// Handle to consume.
    pub handle: Occ3HandleRef,
    /// Consuming session (must match the handle's authorised receiver).
    pub receiver_session_id: String,
    /// Where the injected state should land for the receiver's next
    /// inference (connector-specific routing key, small string).
    pub injection_target: String,
}

/// Fresh, explicitly enrolled native runtime information. This is control-plane
/// metadata, not a persisted permission to reuse a GPU lease.
#[derive(Clone, Debug)]
pub struct NativeSessionEnrollment {
    pub session_id: String,
    pub principal_digest: String,
    pub identity: Occ3ModelIdentity,
    pub runtime_id: String,
    pub runtime_generation: u64,
    pub access_domain: String,
    pub expires_at_epoch: u64,
    pub source_profile: pi_ai::source_capture::SourceCaptureProfile,
    pub profile_digest: String,
}

/// A completed actual source turn, never text for a new capture request.
#[derive(Clone)]
pub struct NativePublicationRequest {
    pub source: pi_ai::source_capture::SourceCaptureLease,
    pub receiver_session_id: String,
    pub message_id: String,
    pub publication_id: String,
    pub dispatch: NativePublicationDispatch,
}

/// How a completed capture is dispatched to its authorised receiver. The
/// legacy protocol binds publication to the model's own same-turn
/// `occ3_handoff` tool call; the retained content-only mode is dispatched
/// deterministically by the authenticated family controller instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativePublicationDispatch {
    /// The captured turn carried exactly one `occ3_handoff` tool call: the
    /// publication must be validated against that exact binding.
    ModelTool {
        tool_call_id: String,
        tool_name: String,
        tool_arguments: serde_json::Value,
    },
    /// Retained content-only capture: the turn carried zero tool bindings and
    /// the authorised orchestrator (family controller) dispatches publication.
    Orchestrator,
}

/// Only authenticated publication creates this process-local envelope authority.
#[derive(Clone)]
pub struct NativePublication {
    pub publication_id: String,
    pub binding_digest: String,
    pub envelope: super::types::Occ3Envelope,
}

/// Logical revocation is separate from physical device allocation release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRetirement {
    pub revoked: bool,
    pub gpu_freed: bool,
}

/// The connector contract. Every method is fallible and every failure is
/// fallback-classified by the bridge. Implementations must be cheap to
/// clone (shared state behind an Arc is the expected shape).
#[async_trait]
pub trait Occ3Connector: Send + Sync {
    /// Stable connector id (e.g. `sglang-occ3`).
    fn connector_id(&self) -> &str;

    /// OCC-3 protocol version this connector implements.
    fn protocol_version(&self) -> u32;

    /// This implementation can prepare an actual request-bound native attachment.
    /// Legacy metrics-only connectors must not enter the admission path.
    fn supports_native_inference(&self) -> bool {
        false
    }

    /// Explicit broker enrollment. Default off; no capability flag can enable it.
    async fn enroll_native_session(
        &self,
        _session_id: &str,
    ) -> Result<NativeSessionEnrollment, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: self.connector_id().into(),
        })
    }

    /// Create metadata for the source's actual next native request, not a helper.
    /// `arming` pins the capture negotiation: `None` resolves the legacy
    /// environment default, `Some(SourceSpanArming::RETAINED_CONTENT_ONLY)`
    /// negotiates the retained content-only mode.
    async fn prepare_source_capture(
        &self,
        _session_id: &str,
        _target: pi_ai::inference_attachment::InferenceStateTarget,
        _retained_count_limit: u32,
        _max_capture_bytes: u64,
        _limits: pi_ai::source_capture::SourceCaptureLimits,
        _arming: Option<pi_ai::source_capture::SourceSpanArming>,
    ) -> Result<pi_ai::source_capture::SourceCaptureLease, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: self.connector_id().into(),
        })
    }

    /// Zero-inference publication of an actual completed, parsed source turn.
    async fn publish_native_source(
        &self,
        _request: NativePublicationRequest,
    ) -> Result<NativePublication, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: self.connector_id().into(),
        })
    }

    async fn retire_native_source(
        &self,
        _source: &pi_ai::source_capture::SourceCaptureLease,
    ) -> Result<NativeRetirement, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: self.connector_id().into(),
        })
    }

    /// Live health probe (cheap, no latent state created).
    async fn health(&self) -> Result<(), Occ3Error>;

    /// Discover capabilities as verified live. `Ok(capabilities)` means
    /// the runtime answered and its identity was checked.
    async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error>;

    /// Verified live model identity. Must be checked against the
    /// sender/receiver identities by the bridge, never trusted from
    /// configuration alone.
    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error>;

    /// Capture a latent state server-side and return an opaque handle.
    async fn capture_latent(&self, request: CaptureRequest) -> Result<Occ3HandleRef, Occ3Error>;

    /// Authorise the receiver named in the handle to consume it, after
    /// the bridge has re-verified model-identity compatibility. The
    /// connector re-checks binding (receiver, model fingerprint,
    /// generation, expiry).
    async fn deliver_latent(&self, handle: &Occ3HandleRef) -> Result<(), Occ3Error>;

    /// Prepare a request-bound, one-shot attachment for the receiver's actual
    /// inference. This operation must not run a detached generation. The
    /// provider verifies consumption on that same request before exposing output.
    /// Older connectors fail closed; cached capability flags alone cannot opt in.
    async fn prepare_inference(
        &self,
        _request: super::inference::PrepareInferenceRequest,
    ) -> Result<pi_ai::inference_attachment::InferenceStateLease, Occ3Error> {
        Err(Occ3Error::NativeInferenceUnsupported {
            connector_id: self.connector_id().to_string(),
        })
    }

    /// Legacy connector-only consumption hook. It must never stand in for an
    /// actual native inference: native callers use `prepare_inference` and its
    /// request-bound receipt. Metrics alone do not prove consumption.
    async fn consume_latent(
        &self,
        request: ConsumeRequest,
    ) -> Result<Occ3TransferMetrics, Occ3Error>;

    /// Acknowledge semantic use after the receiver confirmed the
    /// transferred information was applied (bridge-verified).
    async fn acknowledge_delivery(&self, handle: &Occ3HandleRef) -> Result<(), Occ3Error>;

    /// Release server-side resources for a handle immediately.
    async fn release_handle(&self, handle: &Occ3HandleRef) -> Result<(), Occ3Error>;

    /// Shutdown hook: invalidate all handles this connector issued.
    async fn shutdown(&self) -> Result<(), Occ3Error>;
}

impl dyn Occ3Connector {
    /// Connector id as an owned string, for registry keys.
    pub fn connector_id_string(&self) -> String {
        self.connector_id().to_string()
    }
}
