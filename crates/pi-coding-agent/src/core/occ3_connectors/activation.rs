//! Production activation of experimental OCC-3.
//!
//! The bridge, connector, native session, admission inbox and activity
//! projection all exist; what was missing is one place that decides
//! whether a real session may use them, and a named reason when it may
//! not. This module is that place.
//!
//! Four facts stay separate on purpose and are never collapsed:
//!
//! * capability available  - the runtime verified a connector and method;
//! * experimental enabled  - the user opted in twice and the model
//!   declares an OCC-3 route;
//! * native delivery completed - a receiver consumption receipt exists;
//! * semantically verified - a measurement, never a configuration.
//!
//! This module only decides the first two. It never claims the last two.
//!
//! Nothing here infers support from a model name, a provider string, a
//! tokenizer, a hidden dimension or an endpoint shape: eligibility comes
//! from `models.json`'s `occ3` block and from the connector's own live
//! handshake, and it is still subject to the same-exact-model rule inside
//! [`Occ3NativeSession::enroll`].
use std::time::Duration;

use crate::core::occ::occ3::connector::Occ3Connector;
use crate::core::occ::occ3::types::{Occ3ConnectorCapabilities, Occ3Method, Occ3ModelIdentity};
use crate::core::occ3_connectors::method_family;
use crate::core::occ3_connectors::sglang::{
    native_digest, native_id, NativeExperimentalPermit, SglangOcc3Config, SglangOcc3Connector,
};
use url::Url;
use pi_ai::types::{Model, Occ3Validation};

/// Upper bound on the readiness handshake so activation can never stall a
/// session start. A timeout is a decline, not an error path.
pub const ACTIVATION_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Why OCC-3 is not active for this session. Every variant is a clean
/// decline: the session keeps its ordinary text/OCC-1/OCC-2 behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occ3Decline {
    /// No `occ3` block on the model.
    NotConfigured,
    /// The model declares OCC-3 but has it switched off.
    DisabledOnModel,
    /// `agentCommunication.occ3Enabled` is false.
    DisabledByPolicy,
    /// `agentCommunication.occ3Experimental` is false.
    ExperimentalAckRequired,
    /// The model's `occ3.validation` is not `live-verified`.
    NotLiveVerified,
    /// The block does not use `occ3-v1`.
    UnsupportedProtocol,
    /// `sameModelOnly` is false; v1 cannot cross a model boundary.
    SameModelOnlyRequired,
    /// The connector id has no implementation.
    UnknownConnector,
    /// The endpoint is not an acceptable absolute http(s) URL.
    InvalidEndpoint,
    /// The live handshake failed or timed out.
    ConnectorUnavailable,
    /// The runtime does not advertise the required native method.
    MethodUnsupported,
    /// The runtime's verified identity did not bind to this model.
    ModelIdentityMismatch,
    /// The `occ3.method` declaration is not a valid method id.
    InvalidMethodDeclaration,
    /// The `occ3.artifactDigest` declaration is not a sha256 digest.
    InvalidArtifactDigest,
}

impl Occ3Decline {
    /// Stable, value-free identifier for logs, receipts and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Occ3Decline::NotConfigured => "not_configured",
            Occ3Decline::DisabledOnModel => "disabled_on_model",
            Occ3Decline::DisabledByPolicy => "disabled_by_policy",
            Occ3Decline::ExperimentalAckRequired => "experimental_ack_required",
            Occ3Decline::NotLiveVerified => "not_live_verified",
            Occ3Decline::UnsupportedProtocol => "unsupported_protocol",
            Occ3Decline::SameModelOnlyRequired => "same_model_only_required",
            Occ3Decline::UnknownConnector => "unknown_connector",
            Occ3Decline::InvalidEndpoint => "invalid_endpoint",
            Occ3Decline::ConnectorUnavailable => "connector_unavailable",
            Occ3Decline::MethodUnsupported => "method_unsupported",
            Occ3Decline::ModelIdentityMismatch => "model_identity_mismatch",
            Occ3Decline::InvalidMethodDeclaration => "invalid_method_declaration",
            Occ3Decline::InvalidArtifactDigest => "invalid_artifact_digest",
        }
    }

    /// Operator-facing sentence. No endpoint, key or handle is included.
    pub fn message(self) -> &'static str {
        match self {
            Occ3Decline::NotConfigured => "model has no occ3 configuration block",
            Occ3Decline::DisabledOnModel => "model occ3 block is not enabled",
            Occ3Decline::DisabledByPolicy => "agentCommunication.occ3Enabled is false",
            Occ3Decline::ExperimentalAckRequired => {
                "agentCommunication.occ3Experimental is false; OCC-3 semantic reliability is unproven"
            }
            Occ3Decline::NotLiveVerified => {
                "model occ3.validation is not live-verified; run the readiness probe and record its result"
            }
            Occ3Decline::UnsupportedProtocol => "model occ3 block is not occ3-v1",
            Occ3Decline::SameModelOnlyRequired => "occ3-v1 is same-exact-model only",
            Occ3Decline::UnknownConnector => "occ3 connector has no implementation",
            Occ3Decline::InvalidEndpoint => "occ3 endpoint is not an absolute http(s) URL",
            Occ3Decline::ConnectorUnavailable => "occ3 connector readiness handshake failed or timed out",
            Occ3Decline::MethodUnsupported => "runtime does not advertise the required native method",
            Occ3Decline::ModelIdentityMismatch => "runtime verified identity does not bind to this model",
            Occ3Decline::InvalidMethodDeclaration => {
                "occ3 method declaration is not a valid native method id"
            }
            Occ3Decline::InvalidArtifactDigest => {
                "occ3 artifactDigest declaration is not a sha256 digest"
            }
        }
    }
}

impl std::fmt::Display for Occ3Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// What the caller configured. Built from settings and the model definition
/// only; no session state, no clock, no I/O. The full dotted method id and
/// the artifact digest are identity-only declarations from the model's
/// `occ3` block; credentials are host-resolved and never come from
/// `models.json`.
#[derive(Clone)]
pub struct Occ3ActivationRequest {
    pub occ3_enabled: bool,
    pub occ3_experimental: bool,
    pub model: Model,
    pub endpoint_override: Option<String>,
    pub method: Occ3Method,
    /// Full dotted method id the deployment declares. `None` accepts the
    /// method the live handshake verifies.
    pub method_id: Option<String>,
    /// sha256 of the frozen calibration artifact the deployment binds to.
    /// Identity only: the artifact is never carried or loaded here.
    pub artifact_digest: Option<String>,
    api_key: Option<String>,
    control_headers: Option<indexmap::IndexMap<String, String>>,
}

impl std::fmt::Debug for Occ3ActivationRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Occ3ActivationRequest")
            .field("occ3_enabled", &self.occ3_enabled)
            .field("occ3_experimental", &self.occ3_experimental)
            .field("model", &self.model.id)
            .field("endpoint_override", &self.endpoint_override)
            .field("method", &self.method.as_str())
            .field("method_id", &self.method_id)
            .field("artifact_digest", &self.artifact_digest)
            // Credentials are sensitive: render presence, never values.
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "control_headers",
                &self
                    .control_headers
                    .as_ref()
                    .map(|headers| headers.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}

impl Occ3ActivationRequest {
    pub fn new(model: &Model, occ3_enabled: bool, occ3_experimental: bool) -> Self {
        let occ3 = model.occ3.as_ref();
        let method_id = occ3
            .and_then(|occ3| occ3.method.clone())
            .filter(|method| !method.trim().is_empty());
        Self {
            occ3_enabled,
            occ3_experimental,
            model: model.clone(),
            endpoint_override: None,
            // The negotiation family follows the declaration when one is
            // present; `configured_connector` declines a declaration that
            // names no family before this default can ever matter.
            method: method_id
                .as_deref()
                .and_then(method_family)
                .unwrap_or(Occ3Method::StatebridgePrefix),
            method_id,
            artifact_digest: occ3
                .and_then(|occ3| occ3.artifact_digest.clone())
                .filter(|digest| !digest.trim().is_empty()),
            api_key: None,
            control_headers: None,
        }
    }

    pub fn with_endpoint_override(mut self, endpoint: Option<String>) -> Self {
        self.endpoint_override = endpoint;
        self
    }

    pub fn with_method(mut self, method: Occ3Method) -> Self {
        self.method = method;
        self
    }

    /// Host-resolved credentials for the connector control plane: the same
    /// auth resolution the model's ordinary inference uses. Never values
    /// from `models.json`, never logged.
    pub fn with_credentials(
        mut self,
        api_key: Option<String>,
        control_headers: Option<indexmap::IndexMap<String, String>>,
    ) -> Self {
        self.api_key = api_key;
        self.control_headers = control_headers;
        self
    }

    /// The host-resolved control-plane credentials, if any were supplied.
    pub fn credentials(
        &self,
    ) -> (Option<String>, Option<indexmap::IndexMap<String, String>>) {
        (self.api_key.clone(), self.control_headers.clone())
    }
}

/// The configuration-only gate: no network. Returns the connector config a
/// live handshake would need, or the named decline.
pub fn configured_connector(
    request: &Occ3ActivationRequest,
) -> Result<SglangOcc3Config, Occ3Decline> {
    if !request.occ3_enabled {
        return Err(Occ3Decline::DisabledByPolicy);
    }
    // The acknowledgement is checked before the model so the operator sees
    // the policy reason rather than a configuration reason when both are off.
    if !request.occ3_experimental {
        return Err(Occ3Decline::ExperimentalAckRequired);
    }
    let Some(occ3) = request.model.occ3.as_ref() else {
        return Err(Occ3Decline::NotConfigured);
    };
    if !occ3.enabled {
        return Err(Occ3Decline::DisabledOnModel);
    }
    if occ3.protocol != pi_ai::types::OCC3_PROTOCOL_V1
        || occ3.api_version != pi_ai::types::OCC3_API_VERSION_V1
    {
        return Err(Occ3Decline::UnsupportedProtocol);
    }
    if !occ3.same_model_only {
        return Err(Occ3Decline::SameModelOnlyRequired);
    }
    if occ3.validation != Occ3Validation::LiveVerified {
        return Err(Occ3Decline::NotLiveVerified);
    }
    // Identity-only declarations: a method id must be a bounded native id
    // naming a known negotiation family, and an artifact digest must be a
    // sha256 hex digest. Unknown or malformed values decline by name; the
    // artifact itself is never fetched or loaded here.
    if let Some(method) = request.method_id.as_deref() {
        if !native_id(method) || method_family(method).is_none() {
            return Err(Occ3Decline::InvalidMethodDeclaration);
        }
    }
    if let Some(digest) = request.artifact_digest.as_deref() {
        if !native_digest(digest) {
            return Err(Occ3Decline::InvalidArtifactDigest);
        }
    }
    let endpoint = request
        .endpoint_override
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(occ3.endpoint.as_str());
    if !endpoint_is_acceptable(endpoint) {
        return Err(Occ3Decline::InvalidEndpoint);
    }
    if occ3.connector != "sglang" {
        return Err(Occ3Decline::UnknownConnector);
    }
    let endpoint = Url::parse(endpoint).map_err(|_| Occ3Decline::InvalidEndpoint)?;
    // Credentials are supplied by the host from the environment, never from
    // `models.json`, and are never logged.
    let (api_key, control_headers) = request.credentials();
    Ok(SglangOcc3Config {
        endpoint,
        api_key,
        control_headers,
    })
}

/// Absolute http(s) URL, no userinfo, no query, no fragment. The connector
/// repeats this check; doing it here keeps the decline named.
fn endpoint_is_acceptable(endpoint: &str) -> bool {
    let Ok(url) = Url::parse(endpoint) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.host().is_some()
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

/// A verified, ready connector. Holding this is the only way to reach
/// [`enroll`], so an unverified route cannot be enrolled by accident.
pub struct Occ3Ready {
    pub connector: std::sync::Arc<dyn Occ3Connector>,
    pub capabilities: Occ3ConnectorCapabilities,
    pub method: Occ3Method,
    /// Full dotted method id the verified server identity advertised.
    pub method_id: String,
    /// Declared artifact digest the verification bound as identity. The
    /// artifact itself was never fetched or inspected.
    pub artifact_digest: Option<String>,
    /// Server-attested expiry of the minted permit. A ready past this
    /// boundary is not a verification anymore.
    pub expires_at_epoch: u64,
    pub connector_id: String,
}

impl std::fmt::Debug for Occ3Ready {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Occ3Ready")
            .field("connector_id", &self.connector_id)
            .field("method", &self.method.as_str())
            .field("method_id", &self.method_id)
            .field("artifact_digest", &self.artifact_digest)
            .field("expires_at_epoch", &self.expires_at_epoch)
            .field("supported_methods", &self.capabilities.methods)
            .field("protocol_version", &self.capabilities.protocol_version)
            .finish_non_exhaustive()
    }
}

/// Perform the live readiness handshake and prove the runtime is usable.
/// Failures are named declines; nothing here panics or retries.
///
/// The permit the connector needs is minted here from the server's own
/// handshake: the identity fields from `GET /occ3/v1/identity`, the current
/// runtime incarnation from `POST /occ3/v1/native/runtime/identity`, and a
/// probe enrollment the server must accept before discovery. Nothing about
/// the runtime is hardcoded or fabricated locally: a fingerprint the server
/// did not attest, a method the runtime does not serve, or an enrollment it
/// refuses is a named decline, never a made-up identity.
pub async fn verify(
    request: &Occ3ActivationRequest,
    handshake_timeout: Duration,
) -> Result<Occ3Ready, Occ3Decline> {
    let config = configured_connector(request)?;
    match tokio::time::timeout(handshake_timeout, mint_ready(request, config)).await {
        Err(_) => Err(Occ3Decline::ConnectorUnavailable),
        Ok(result) => result,
    }
}

/// One bounded production handshake: identity, runtime incarnation, permit
/// construction, probe enrollment and capability discovery. Every value that
/// names the runtime comes from the server; only the local operation ids
/// (attempt, probe session, access-domain label) are minted here.
async fn mint_ready(
    request: &Occ3ActivationRequest,
    config: SglangOcc3Config,
) -> Result<Occ3Ready, Occ3Decline> {
    // Control-plane probe without a permit: the read-only handshake routes.
    let probe =
        SglangOcc3Connector::new(config.clone()).map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    let identity = probe
        .model_identity()
        .await
        .map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    // The identity the runtime serves must be the model being activated.
    if identity.model_id != request.model.id {
        return Err(Occ3Decline::ModelIdentityMismatch);
    }
    let runtime = probe
        .runtime_identity()
        .await
        .map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    let method_id = runtime.method_id.clone();
    // The full dotted method id names the negotiation family; a runtime
    // serving an unknown family has no negotiation route at all.
    let family = method_family(&method_id).ok_or(Occ3Decline::MethodUnsupported)?;
    // A declared method must be exactly the method the runtime serves.
    if request
        .method_id
        .as_deref()
        .is_some_and(|declared| declared != method_id)
    {
        return Err(Occ3Decline::MethodUnsupported);
    }
    // The native identity is the server's identity fields bound to the native
    // method the runtime serves. The server-attested fingerprint must equal
    // the digest of exactly that identity, so the binding is proved by the
    // server, never assumed locally.
    let permit_identity = Occ3ModelIdentity {
        latent_method: method_id.clone(),
        ..identity
    };
    if permit_identity.fingerprint() != runtime.model_fingerprint {
        return Err(Occ3Decline::ModelIdentityMismatch);
    }
    let expires_at_epoch = now_epoch() + runtime.max_ttl_seconds.min(3600);
    let permit = NativeExperimentalPermit {
        attempt_id: format!("activation-{}", uuid::Uuid::new_v4()),
        identity: permit_identity,
        target: pi_ai::inference_attachment::InferenceStateTarget {
            provider: request.model.provider.clone(),
            model_id: request.model.id.clone(),
            base_url: request.model.base_url.clone(),
        },
        runtime_id: runtime.runtime_id.clone(),
        runtime_generation: runtime.runtime_generation,
        // Deployment-scoping label shared by every enrollment under this
        // permit; derived from the server-attested runtime id, never a secret.
        access_domain: format!("experimental.{}", runtime.runtime_id),
        expires_at_epoch,
        source_profile: pi_ai::source_capture::SourceCaptureProfile {
            layer_index: runtime.layer_index,
            hidden_dim: runtime.hidden_dim,
            dtype: runtime.dtype.clone(),
            layout_digest: runtime.tp_layout_digest.clone(),
            expected_ranks: runtime.expected_ranks,
        },
        profile_digest: runtime.profile_digest.clone(),
        max_capture_bytes: runtime.max_capture_bytes,
    };
    let connector = SglangOcc3Connector::new_experimental(config, permit)
        .map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    // Probe enrollment: the server must accept the minted permit before any
    // capability claim is made.
    let probe_session = format!("activation-probe-{}", uuid::Uuid::new_v4());
    connector
        .enroll_native_session(&probe_session)
        .await
        .map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    let connector: std::sync::Arc<dyn Occ3Connector> = std::sync::Arc::new(connector);
    let capabilities = connector
        .discover_capabilities()
        .await
        .map_err(|_| Occ3Decline::ConnectorUnavailable)?;
    // The advertised identity must be the model we are about to bind to.
    // `Occ3NativeSession::enroll` repeats this against the enrollment; the
    // check here keeps the decline named and prevents a wrong route from
    // being registered process-wide.
    if capabilities.model_identity.model_id != request.model.id {
        return Err(Occ3Decline::ModelIdentityMismatch);
    }
    if !capabilities.methods.contains(&family) {
        return Err(Occ3Decline::MethodUnsupported);
    }
    Ok(Occ3Ready {
        connector_id: connector.connector_id().to_string(),
        connector,
        capabilities,
        method: family,
        method_id,
        artifact_digest: request.artifact_digest.clone(),
        expires_at_epoch,
    })
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Process-wide verified connector cache, keyed by runtime, model, method id
/// and artifact digest. The readiness handshake and the verified identity are
/// properties of the runtime, not of a session, so one successful
/// verification serves every later enrollment with the same identity. A
/// changed endpoint, model, method or artifact digest re-verifies, and two
/// methods used against one runtime keep their verifications side by side
/// instead of evicting each other.
fn ready_cache(
) -> &'static std::sync::Mutex<std::collections::BTreeMap<String, std::sync::Arc<Occ3Ready>>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<String, std::sync::Arc<Occ3Ready>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

fn cache_key(request: &Occ3ActivationRequest, config: &SglangOcc3Config) -> String {
    // Readiness is per runtime, per method and per declared artifact
    // identity: two methods, or two artifact digests, must never share one
    // verification.
    format!(
        "{}|{}|{}|{}",
        config.endpoint,
        request.model.id,
        request.method_id.as_deref().unwrap_or(""),
        request.artifact_digest.as_deref().unwrap_or(""),
    )
}

/// [`verify`] with the process-wide cache. Concurrent callers share one
/// handshake result; a poisoned cache is a miss, never a panic.
pub async fn verify_cached(
    request: &Occ3ActivationRequest,
    handshake_timeout: Duration,
) -> Result<std::sync::Arc<Occ3Ready>, Occ3Decline> {
    let config = configured_connector(request)?;
    let key = cache_key(request, &config);
    if let Some(ready) = ready_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(&key).cloned())
    {
        if ready.expires_at_epoch > now_epoch() {
            return Ok(ready);
        }
        // An expired permit is not a verification: drop that entry so the
        // handshake below replaces it.
        if let Ok(mut cache) = ready_cache().lock() {
            cache.remove(&key);
        }
    }
    let ready = std::sync::Arc::new(verify(request, handshake_timeout).await?);
    if let Ok(mut cache) = ready_cache().lock() {
        cache.insert(key, std::sync::Arc::clone(&ready));
    }
    Ok(ready)
}

/// Drop the cached verification. Called when a runtime restart invalidates
/// every enrollment, so the next activation re-handshakes.
pub fn invalidate_ready_cache() {
    if let Ok(mut cache) = ready_cache().lock() {
        cache.clear();
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Offline loopback control plane for activation tests: the three
    //! production handshake routes with one fixed server-attested identity
    //! and a per-fixture request counter. No GPU, no provider account.
    //!
    //! The routes answer like the deployed extension does: the identity
    //! route serves the legacy bridge fields (whose `latent_method` is the
    //! legacy label), the runtime-identity route attests the native method
    //! id and the fingerprint of the identity bound to that method, and the
    //! enrollment route echoes the client's permit values in its readiness.

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::core::occ::occ3::types::Occ3ModelIdentity;

    pub(crate) const METHOD_A: &str =
        "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
    pub(crate) const METHOD_B: &str = "statebridge.glm53.calibrated-lowrank-procrustes.v1";
    /// The legacy bridge identity label served by `GET /occ3/v1/identity`.
    pub(crate) const LEGACY_LATENT_METHOD: &str = "kv-radix-channel-v1";

    const RUNTIME_ID: &str = "fixture-runtime-epoch-001";
    const RUNTIME_GENERATION: u64 = 7;
    const LAYER_INDEX: u32 = 77;
    const HIDDEN_DIM: u32 = 6144;
    const DTYPE: &str = "bfloat16";
    const EXPECTED_RANKS: u32 = 8;
    const MAX_TTL_SECONDS: u64 = 3600;
    const MAX_CAPTURE_BYTES: u64 = 134_217_728;

    /// A complete, non-placeholder server identity the fixture attests. The
    /// `latent_method` is the NATIVE method id the runtime serves.
    pub(crate) fn fixture_identity(
        model_id: &str,
        latent_method: &str,
    ) -> Occ3ModelIdentity {
        Occ3ModelIdentity {
            model_id: model_id.into(),
            architecture: "GlmMoeDsaForCausalLM".into(),
            checkpoint_revision: "fixture-revision-a1".into(),
            weight_fingerprint: "metadata-fixture-not-weights".into(),
            tokenizer_fingerprint: "fixture-tokenizer".into(),
            chat_template_fingerprint: "fixture-template".into(),
            quantisation: "fixture-bf16".into(),
            tensor_geometry: "fixture-d8".into(),
            kv_layout: "fixture-tp8".into(),
            latent_method: latent_method.into(),
            runtime: "synthetic-local-fixture".into(),
        }
    }

    pub(crate) struct ActivationFixture {
        pub endpoint: url::Url,
        pub requests: Arc<AtomicUsize>,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    }

    impl Drop for ActivationFixture {
        fn drop(&mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
        }
    }

    /// Serve `GET /occ3/v1/identity`, `POST /occ3/v1/native/runtime/identity`
    /// and `POST /occ3/v1/native/sessions/enroll` on a fresh loopback port.
    /// `required_bearer` makes every route refuse a request without that
    /// exact host credential (the production control plane is keyed).
    pub(crate) async fn serve_activation_fixture(
        identity: Occ3ModelIdentity,
        required_bearer: Option<String>,
    ) -> ActivationFixture {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let endpoint = url::Url::parse(&format!(
            "http://{}",
            listener.local_addr().expect("local addr")
        ))
        .expect("endpoint url");
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_outer = Arc::clone(&requests);
        let (shutdown, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            loop {
                let socket = tokio::select! {
                    accepted = listener.accept() => match accepted {
                        Ok((socket, _)) => socket,
                        Err(_) => break,
                    },
                    _ = &mut shutdown_rx => break,
                };
                let identity = identity.clone();
                let required_bearer = required_bearer.clone();
                let requests = Arc::clone(&requests);
                tokio::spawn(async move {
                    serve_connection(socket, identity, required_bearer, requests).await;
                });
            }
        });
        ActivationFixture {
            endpoint,
            requests: requests_outer,
            shutdown: Some(shutdown),
        }
    }

    async fn serve_connection(
        mut socket: tokio::net::TcpStream,
        identity: Occ3ModelIdentity,
        required_bearer: Option<String>,
        requests: Arc<AtomicUsize>,
    ) {
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        let end;
        loop {
            let mut part = [0u8; 4096];
            let Ok(n) = socket.read(&mut part).await else { return };
            if n == 0 {
                return;
            }
            bytes.extend_from_slice(&part[..n]);
            if let Some(i) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                end = i + 4;
                break;
            }
        }
        let head = String::from_utf8_lossy(&bytes[..end]).to_string();
        let content_length = head
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|value| value.trim().parse::<usize>().ok())
            })
            .flatten()
            .unwrap_or(0);
        while bytes.len() < end + content_length {
            let mut part = [0u8; 4096];
            let Ok(n) = socket.read(&mut part).await else { return };
            if n == 0 {
                return;
            }
            bytes.extend_from_slice(&part[..n]);
        }
        let path = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        if let Some(expected) = &required_bearer {
            let authorized = head
                .to_ascii_lowercase()
                .contains(&format!("authorization: bearer {expected}").to_ascii_lowercase());
            if !authorized {
                let body = r#"{"error":{"code":"identity"}}"#;
                let _ = write_response(&mut socket, "401 Unauthorized", body, &[]).await;
                return;
            }
        }
        let body: Option<serde_json::Value> = if content_length == 0 {
            None
        } else {
            serde_json::from_slice(&bytes[end..end + content_length]).ok()
        };
        requests.fetch_add(1, Ordering::SeqCst);
        // The GET identity route serves the legacy bridge fields: the
        // extension's identity.py pins `latent_method` to the legacy label,
        // and the native method id is attested only by the runtime route.
        let mut legacy_fields = serde_json::to_value(&identity).expect("identity serializes");
        legacy_fields["latent_method"] = serde_json::json!(LEGACY_LATENT_METHOD);
        // The runtime fingerprint attests the identity bound to the NATIVE
        // method id, exactly like a pinned deployment profile does.
        let native_fingerprint = identity.fingerprint();
        let readiness = serde_json::json!({
            "runtime_epoch": RUNTIME_ID,
            "runtime_generation": RUNTIME_GENERATION,
            "model_fingerprint": native_fingerprint.as_str(),
            "method_id": identity.latent_method.as_str(),
            "expected_ranks": EXPECTED_RANKS,
            "ready_ranks": EXPECTED_RANKS,
            "tp_layout_digest": "b".repeat(64),
            "hidden_dim": HIDDEN_DIM,
            "layer_index": LAYER_INDEX,
            "dtype": DTYPE,
            "profile_digest": "c".repeat(64),
            "source_capture": true,
            "continuous_prefix": true,
            "terminal_journal": true,
            "max_capture_bytes": MAX_CAPTURE_BYTES,
        });
        let response = match (path.as_str(), body.as_ref()) {
            ("/occ3/v1/identity", _) => serde_json::json!({
                "schema": "optimus.occ3.identity.v1",
                "fields": legacy_fields,
                "fingerprint": legacy_fingerprint(&legacy_fields),
                "field_order": ["model_id","architecture","checkpoint_revision","weight_fingerprint",
                    "tokenizer_fingerprint","chat_template_fingerprint","quantisation",
                    "tensor_geometry","kv_layout","latent_method","runtime"],
            }),
            ("/occ3/v1/native/runtime/identity", Some(_)) => serde_json::json!({
                "schema": "occ3.runtime.identity.v1",
                "runtime_id": RUNTIME_ID,
                "runtime_generation": RUNTIME_GENERATION,
                "model_fingerprint": native_fingerprint.as_str(),
                "method_id": identity.latent_method.as_str(),
                "tp_layout_digest": "b".repeat(64),
                "layer_index": LAYER_INDEX,
                "hidden_dim": HIDDEN_DIM,
                "dtype": DTYPE,
                "expected_ranks": EXPECTED_RANKS,
                "ready_ranks": EXPECTED_RANKS,
                "profile_digest": "c".repeat(64),
                "source_capture": true,
                "continuous_prefix": true,
                "terminal_journal": true,
                "server_time_epoch": crate_now(),
                "max_ttl_seconds": MAX_TTL_SECONDS,
                "max_capture_bytes": MAX_CAPTURE_BYTES,
            }),
            ("/occ3/v1/native/sessions/enroll", Some(body)) => serde_json::json!({
                "schema": "occ3.sessions.enrolled.v1",
                "operation_id": body["operation_id"],
                "session_id": body["session_id"],
                "principal_digest": "a".repeat(64),
                "runtime_id": body["runtime_id"],
                "runtime_generation": body["runtime_generation"],
                "model_fingerprint": body["model_fingerprint"],
                "expires_at_epoch": body["expires_at_epoch"],
                "readiness": readiness,
            }),
            _ => return,
        };
        let payload = serde_json::to_string(&response).expect("fixture response serializes");
        let _ = write_response(&mut socket, "200 OK", &payload, &[]).await;
    }

    /// The legacy bridge fingerprint served by the identity route
    /// (`|`-joined fields, identity.py:197). The activation path never
    /// compares against this value; it exists for wire fidelity.
    fn legacy_fingerprint(fields: &serde_json::Value) -> String {
        const ORDER: [&str; 11] = [
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
        ORDER
            .iter()
            .map(|field| fields[field].as_str().unwrap_or_default())
            .collect::<Vec<_>>()
            .join("|")
    }

    async fn write_response(
        socket: &mut tokio::net::TcpStream,
        status_line: &str,
        body: &str,
        extra: &[(&str, &str)],
    ) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        let mut head = format!(
            "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in extra {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        socket.write_all(head.as_bytes()).await?;
        socket.write_all(body.as_bytes()).await?;
        socket.shutdown().await
    }

    fn crate_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Drop any primed cache entry so tests cannot share readiness, and
    /// hold the returned guard for the whole test: the ready cache is
    /// process-wide and tests run in parallel. Holding the guard keeps one
    /// test's prime/invalidate sequence from racing another's.
    pub(crate) async fn reset_ready_cache() -> tokio::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
            std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
        super::invalidate_ready_cache();
        LOCK.lock().await
    }
}

/// Enroll one live session against a verified connector and install the
/// native permit and handoff tool. The session must not be streaming.
pub async fn enroll(
    session: &std::sync::Arc<crate::core::agent_session::AgentSession>,
    model: &Model,
    ready: &Occ3Ready,
    controller: std::sync::Arc<dyn crate::core::agent_messages::AgentSessionMessageController>,
) -> Result<(), String> {
    if session.is_streaming() {
        return Err("occ3 activation refused: session is streaming".into());
    }
    let state = crate::core::agent_session::occ3_state::Occ3NativeSession::enroll(
        &session.session_id(),
        model,
        ready.connector.clone(),
    )
    .await?;
    session.install_experimental_native_occ3(state, controller)
}

/// Register a verified connector in the process-wide registry so the
/// negotiation ladder can see it.
pub fn register(ready: &Occ3Ready) {
    crate::core::occ3_connectors::with_global_registry(|registry| {
        registry.register(ready.connector.clone(), ready.capabilities.clone());
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{API_OPENAI_COMPLETIONS, Occ3Capability};

    fn model_with(occ3: Option<Occ3Capability>) -> Model {
        let mut model = Model::new(
            "glm-5.3",
            "GLM-5.3",
            API_OPENAI_COMPLETIONS,
            "dgx-sglang",
            "http://127.0.0.1:30001/v1",
        );
        model.occ3 = occ3;
        model
    }

    fn capability() -> Occ3Capability {
        Occ3Capability {
            protocol: pi_ai::types::OCC3_PROTOCOL_V1.into(),
            connector: "sglang".into(),
            endpoint: "http://127.0.0.1:30001".into(),
            api_version: pi_ai::types::OCC3_API_VERSION_V1.into(),
            same_model_only: true,
            enabled: true,
            validation: Occ3Validation::LiveVerified,
            fallback: vec!["occ2".into(), "occ1".into(), "text".into()],
            method: None,
            artifact_digest: None,
        }
    }

    fn request(model: Model) -> Occ3ActivationRequest {
        Occ3ActivationRequest::new(&model, true, true)
    }

    #[test]
    fn a_fully_configured_live_verified_model_produces_a_connector_config() {
        let config = configured_connector(&request(model_with(Some(capability()))))
            .expect("activated");
        assert_eq!(config.endpoint.as_str(), "http://127.0.0.1:30001/");
        assert!(config.api_key.is_none());
        assert!(config.control_headers.is_none());
    }

    #[test]
    fn every_gate_declines_with_its_own_named_reason() {
        // Policy off wins over everything else.
        assert_eq!(
            configured_connector(&Occ3ActivationRequest::new(&model_with(Some(capability())), false, true)).err(),
            Some(Occ3Decline::DisabledByPolicy)
        );
        // Unacknowledged experimental use.
        assert_eq!(
            configured_connector(&Occ3ActivationRequest::new(&model_with(Some(capability())), true, false)).err(),
            Some(Occ3Decline::ExperimentalAckRequired)
        );
        // No occ3 block: a model name alone is never evidence.
        assert_eq!(
            configured_connector(&request(model_with(None))).err(),
            Some(Occ3Decline::NotConfigured)
        );
        let cases: [(fn(&mut Occ3Capability), Occ3Decline); 6] = [
            (|c| c.enabled = false, Occ3Decline::DisabledOnModel),
            (|c| c.protocol = "occ3-v2".into(), Occ3Decline::UnsupportedProtocol),
            (|c| c.api_version = "v2".into(), Occ3Decline::UnsupportedProtocol),
            (|c| c.same_model_only = false, Occ3Decline::SameModelOnlyRequired),
            (|c| c.validation = Occ3Validation::Unverified, Occ3Decline::NotLiveVerified),
            (|c| c.validation = Occ3Validation::DocumentationVerified, Occ3Decline::NotLiveVerified),
        ];
        for (mutate, expected) in cases {
            let mut block = capability();
            mutate(&mut block);
            assert_eq!(configured_connector(&request(model_with(Some(block)))).err(), Some(expected));
        }
        let mut unknown = capability();
        unknown.connector = "vllm".into();
        assert_eq!(
            configured_connector(&request(model_with(Some(unknown)))).err(),
            Some(Occ3Decline::UnknownConnector)
        );
    }

    #[test]
    fn endpoints_that_could_leak_credentials_or_override_the_target_are_rejected() {
        for endpoint in [
            "",
            "127.0.0.1:30001",
            "ftp://127.0.0.1:30001",
            "http://user:secret@127.0.0.1:30001",
            "http://127.0.0.1:30001?token=secret",
            "http://127.0.0.1:30001#frag",
        ] {
            let mut block = capability();
            block.endpoint = endpoint.into();
            let request = request(model_with(Some(block)));
            assert_eq!(
                configured_connector(&request).err(),
                Some(Occ3Decline::InvalidEndpoint),
                "endpoint {endpoint:?} must be rejected"
            );
        }
    }

    #[test]
    fn the_endpoint_override_replaces_the_model_endpoint() {
        let overridden = request(model_with(Some(capability())))
            .with_endpoint_override(Some("  http://127.0.0.1:30002  ".into()));
        let config = configured_connector(&overridden).expect("override accepted");
        assert_eq!(config.endpoint.port(), Some(30002));
        // A blank override is not a value.
        let config = configured_connector(
            &request(model_with(Some(capability()))).with_endpoint_override(Some("   ".into())),
        )
        .expect("blank override falls back");
        assert_eq!(config.endpoint.port(), Some(30001));
    }

    #[test]
    fn the_default_method_is_the_statebridge_prefix_and_it_is_requestable() {
        let request = request(model_with(Some(capability())));
        assert_eq!(request.method, Occ3Method::StatebridgePrefix);
        assert_eq!(configured_connector(&request).is_ok(), true);
        let kv = request.with_method(Occ3Method::KvHandoff);
        assert_eq!(kv.method, Occ3Method::KvHandoff);
    }

    #[tokio::test]
    async fn an_unreachable_runtime_declines_without_erroring_or_hanging() {
        let mut block = capability();
        // Port 1 on loopback refuses immediately; the timeout keeps the
        // worst case bounded even if it did not.
        block.endpoint = "http://127.0.0.1:1".into();
        let request = request(model_with(Some(block)));
        assert_eq!(
            verify(&request, Duration::from_millis(750)).await.err(),
            Some(Occ3Decline::ConnectorUnavailable)
        );
    }

    #[test]
    fn decline_identifiers_are_stable_and_value_free() {
        let all = [
            Occ3Decline::NotConfigured,
            Occ3Decline::DisabledOnModel,
            Occ3Decline::DisabledByPolicy,
            Occ3Decline::ExperimentalAckRequired,
            Occ3Decline::NotLiveVerified,
            Occ3Decline::UnsupportedProtocol,
            Occ3Decline::SameModelOnlyRequired,
            Occ3Decline::UnknownConnector,
            Occ3Decline::InvalidEndpoint,
            Occ3Decline::ConnectorUnavailable,
            Occ3Decline::MethodUnsupported,
            Occ3Decline::ModelIdentityMismatch,
            Occ3Decline::InvalidMethodDeclaration,
            Occ3Decline::InvalidArtifactDigest,
        ];
        let mut seen = std::collections::BTreeSet::new();
        for decline in all {
            assert!(seen.insert(decline.as_str()), "duplicate id for {decline:?}");
            assert!(!decline.as_str().contains(' '));
            assert!(!decline.message().is_empty());
        }
    }

    // --- Live activation against the loopback control-plane fixture ---

    use super::test_support::{
        fixture_identity, reset_ready_cache, serve_activation_fixture, ActivationFixture,
        METHOD_A, METHOD_B,
    };

    const DIGEST_1: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const DIGEST_2: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    /// The fixture model: its inference base URL shares the control-plane
    /// origin, which is what the connector's target binding requires.
    fn fixture_model(fixture: &ActivationFixture) -> Model {
        let mut model = model_with(Some(capability()));
        let block = model.occ3.as_mut().expect("capability block");
        block.endpoint = fixture.endpoint.as_str().into();
        model.base_url = format!("{}/v1", fixture.endpoint.as_str().trim_end_matches('/'));
        model
    }

    fn fixture_request(
        fixture: &ActivationFixture,
        declared_method: Option<&str>,
        declared_digest: Option<&str>,
    ) -> Occ3ActivationRequest {
        let model = fixture_model(fixture);
        let mut request = Occ3ActivationRequest::new(&model, true, true);
        request.method_id = declared_method.map(str::to_string);
        request.artifact_digest = declared_digest.map(str::to_string);
        request
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_activation_mints_a_real_permit_from_the_live_handshake() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        let before = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
        let ready = verify(&request, Duration::from_secs(5))
            .await
            .expect("a serving runtime with a live-verified block must activate");
        // Identity, runtime incarnation, profile and byte cap all come from
        // the server handshake; nothing was fabricated locally.
        assert_eq!(ready.method_id, METHOD_A, "full dotted method id is verified identity");
        assert_eq!(ready.method, Occ3Method::StatebridgePrefix);
        assert_eq!(ready.capabilities.model_identity.model_id, "glm-5.3");
        assert!(
            ready.capabilities.methods.contains(&Occ3Method::StatebridgePrefix),
            "capabilities advertise the family the verified method names"
        );
        let served = fixture.requests.load(std::sync::atomic::Ordering::SeqCst) - before;
        assert!(
            served >= 4,
            "identity + runtime identity + probe enrollment + discovery handshake ({served})"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn verify_declines_a_declared_method_the_verified_runtime_does_not_serve() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        // The runtime verifiably serves METHOD_A; a deployment declaring a
        // different full method id must not activate against it.
        let request = fixture_request(&fixture, Some(METHOD_B), None);
        assert_eq!(
            verify(&request, Duration::from_secs(5)).await.err(),
            Some(Occ3Decline::MethodUnsupported)
        );
        // And a runtime serving a different model than the one bound.
        let fixture =
            serve_activation_fixture(fixture_identity("other-model", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        assert_eq!(
            verify(&request, Duration::from_secs(5)).await.err(),
            Some(Occ3Decline::ModelIdentityMismatch)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_activation_with_a_declared_method_and_artifact_digest_binds_identity_only() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, Some(METHOD_A), Some(DIGEST_1));
        let ready = verify(&request, Duration::from_secs(5))
            .await
            .expect("a matching declaration activates");
        assert_eq!(ready.method_id, METHOD_A);
        assert_eq!(ready.artifact_digest.as_deref(), Some(DIGEST_1));
        // The artifact itself is never fetched: only its declared digest was
        // compared, and no route other than the three handshake routes was
        // touched.
        assert_eq!(
            fixture.requests.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "identity, runtime identity, probe enrollment and discovery only"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_cache_distinguishes_method_and_artifact_digest() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        // Prime: no declaration.
        let plain = fixture_request(&fixture, None, None);
        let ready = verify_cached(&plain, Duration::from_secs(5))
            .await
            .expect("first activation verifies");
        assert_eq!(ready.method_id, METHOD_A);
        let after_prime = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
        // Same request again: the cached ready serves it with no handshake.
        let again = verify_cached(&plain, Duration::from_secs(5))
            .await
            .expect("cache hit");
        assert_eq!(
            fixture.requests.load(std::sync::atomic::Ordering::SeqCst),
            after_prime,
            "an identical activation reuses the verified ready"
        );
        assert!(std::sync::Arc::ptr_eq(&ready, &again));
        // A DIFFERENT declared method must never reuse the first method's
        // ready: it re-handshakes and declines on the verified mismatch.
        let other_method = fixture_request(&fixture, Some(METHOD_B), None);
        assert_eq!(
            verify_cached(&other_method, Duration::from_secs(5)).await.err(),
            Some(Occ3Decline::MethodUnsupported),
            "readiness is per method, never shared across methods"
        );
        assert!(
            fixture.requests.load(std::sync::atomic::Ordering::SeqCst) > after_prime,
            "the mismatch was proven against the live runtime, not the cache"
        );
        // A different artifact digest is a different readiness: both verify,
        // neither reuses the other's ready.
        for (digest, expect_requests_grow) in [(DIGEST_1, true), (DIGEST_2, true)] {
            let declared = fixture_request(&fixture, Some(METHOD_A), Some(digest));
            let before = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
            let ready = verify_cached(&declared, Duration::from_secs(5))
                .await
                .expect("a declared digest verifies against the same runtime");
            assert_eq!(ready.artifact_digest.as_deref(), Some(digest));
            if expect_requests_grow {
                assert!(
                    fixture.requests.load(std::sync::atomic::Ordering::SeqCst) > before,
                    "a changed artifact digest re-verifies"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalidate_ready_cache_forces_a_re_handshake() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let request = fixture_request(&fixture, None, None);
        verify_cached(&request, Duration::from_secs(5))
            .await
            .expect("first verification");
        let served_first = fixture.requests.load(std::sync::atomic::Ordering::SeqCst);
        invalidate_ready_cache();
        verify_cached(&request, Duration::from_secs(5))
            .await
            .expect("re-verification after invalidation");
        assert!(
            fixture.requests.load(std::sync::atomic::Ordering::SeqCst) > served_first,
            "invalidation drops the cached ready, forcing a fresh handshake"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn host_credentials_reach_the_connector_control_plane() {
        let _cache_guard = reset_ready_cache().await;
        let fixture = serve_activation_fixture(
            fixture_identity("glm-5.3", METHOD_A),
            Some("synthetic-host-key".into()),
        )
        .await;
        // The keyed control plane only answers the host-resolved credential;
        // no models.json value is involved.
        let request = fixture_request(&fixture, None, None)
            .with_credentials(Some("synthetic-host-key".into()), None);
        let ready = verify(&request, Duration::from_secs(5))
            .await
            .expect("the host credential authorizes the handshake");
        assert_eq!(ready.method_id, METHOD_A);
        // A wrong credential is refused by the control plane: a named
        // decline, never a hang and never a bypass.
        let request = fixture_request(&fixture, None, None)
            .with_credentials(Some("wrong-host-key".into()), None);
        assert_eq!(
            verify(&request, Duration::from_secs(5)).await.err(),
            Some(Occ3Decline::ConnectorUnavailable)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_runtime_advertising_a_second_method_family_verifies() {
        let _cache_guard = reset_ready_cache().await;
        let fixture = serve_activation_fixture(
            fixture_identity("glm-5.3", "kv-handoff.glm53.final-kv.v1"),
            None,
        )
        .await;
        let request = fixture_request(&fixture, Some("kv-handoff.glm53.final-kv.v1"), None);
        let ready = verify(&request, Duration::from_secs(5))
            .await
            .expect("a kv-handoff runtime activates on its own family");
        assert_eq!(ready.method_id, "kv-handoff.glm53.final-kv.v1");
        assert_eq!(ready.method, Occ3Method::KvHandoff);
        assert!(
            ready.capabilities.methods.contains(&Occ3Method::KvHandoff),
            "capabilities advertise the kv-handoff family, not a hardcoded one"
        );
    }

    #[test]
    fn invalid_method_and_digest_declarations_decline_by_name() {
        let fixture_endpoint = "http://127.0.0.1:30001";
        let mut block = capability();
        block.endpoint = fixture_endpoint.into();
        // A method id with no known negotiation family.
        block.method = Some("vllm-gc.fancy.v1".into());
        assert_eq!(
            configured_connector(&request(model_with(Some(block.clone())))).err(),
            Some(Occ3Decline::InvalidMethodDeclaration)
        );
        // Out-of-bounds identifiers (257 bytes) decline identically.
        block.method = Some(format!("statebridge.{}", "a".repeat(300)));
        assert_eq!(
            configured_connector(&request(model_with(Some(block.clone())))).err(),
            Some(Occ3Decline::InvalidMethodDeclaration)
        );
        block.method = Some(METHOD_A.into());
        // An artifact digest that is not sha256 hex declines by name.
        block.artifact_digest = Some("not-a-digest".into());
        assert_eq!(
            configured_connector(&request(model_with(Some(block.clone())))).err(),
            Some(Occ3Decline::InvalidArtifactDigest)
        );
        block.artifact_digest = Some(DIGEST_1.into());
        let config = configured_connector(&request(model_with(Some(block))))
            .expect("a valid method and digest declaration activates");
        assert_eq!(config.endpoint.as_str(), "http://127.0.0.1:30001/");
    }

    #[test]
    fn activation_request_debug_never_renders_credentials() {
        let mut block = capability();
        block.method = Some(METHOD_A.into());
        let mut headers = indexmap::IndexMap::new();
        headers.insert("x-pomerium-authorization".to_string(), "secret-gateway".into());
        let request = request(model_with(Some(block)))
            .with_credentials(Some("synthetic-secret-key".into()), Some(headers));
        let rendered = format!("{request:?}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(!rendered.contains("synthetic-secret-key"), "{rendered}");
        assert!(!rendered.contains("secret-gateway"), "{rendered}");
        assert!(rendered.contains(METHOD_A), "{rendered}");
    }

    #[test]
    fn method_families_derive_from_full_dotted_ids() {
        assert_eq!(
            super::method_family(METHOD_A),
            Some(Occ3Method::StatebridgePrefix)
        );
        assert_eq!(
            super::method_family("kv-handoff.glm53.final-kv.v1"),
            Some(Occ3Method::KvHandoff)
        );
        assert_eq!(super::method_family("vllm-gc.fancy.v1"), None);
        assert_eq!(super::method_family(""), None);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalidating_session_state_drops_the_cached_ready() {
        let _cache_guard = reset_ready_cache().await;
        let fixture =
            serve_activation_fixture(fixture_identity("glm-5.3", METHOD_A), None).await;
        let model = fixture_model(&fixture);
        let request = fixture_request(&fixture, None, None);
        let primed = verify_cached(&request, Duration::from_secs(5))
            .await
            .expect("first activation verifies and primes the cache");
        let cached = verify_cached(&request, Duration::from_secs(5))
            .await
            .expect("second activation is a cache hit");
        assert!(
            std::sync::Arc::ptr_eq(&primed, &cached),
            "the same ready serves repeated activations"
        );
        // A real native session enrolled against the verified connector is
        // the session lifecycle the daemon owns; invalidating its state must
        // retire the process-wide ready, so the next activation re-handshakes
        // instead of reusing a ready whose session state no longer exists.
        let session = crate::core::agent_session::occ3_state::Occ3NativeSession::enroll(
            "session-x",
            &model,
            primed.connector.clone(),
        )
        .await
        .expect("session enrolls against the verified connector");
        session.invalidate();
        let reverified = verify_cached(&request, Duration::from_secs(5))
            .await
            .expect("the next activation re-verifies");
        assert!(
            !std::sync::Arc::ptr_eq(&cached, &reverified),
            "invalidate() must drop the cached ready"
        );
    }
}
