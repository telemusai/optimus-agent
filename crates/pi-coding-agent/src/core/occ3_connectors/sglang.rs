//! SGLang OCC-3 connector diagnostics and fail-closed native boundary.
//!
//! The installed plugin's source-prompt KV routes are an experimental
//! text-plus-radix baseline. They do not implement the receiver's actual native
//! chat/tool inference. In particular, a detached `/recv` generation is not a
//! native attachment. Until that adapter and its request-bound receipt exist,
//! discovery and all state operations explicitly report unsupported. Normal
//! OpenAI-compatible inference remains independent of this connector.

use std::time::Duration;

use async_trait::async_trait;
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use url::Url;

use crate::core::occ::occ3::connector::{CaptureRequest, ConsumeRequest, Occ3Connector};
use crate::core::occ::occ3::types::{
    Occ3ConnectorCapabilities, Occ3Error, Occ3HandleRef, Occ3Method, Occ3ModelIdentity,
    Occ3TransferMetrics,
};

pub const CONNECTOR_ID: &str = "sglang-occ3";

const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
const OP_TIMEOUT: Duration = Duration::from_secs(90);

/// Construction parameters (from a model's `occ3` block + environment).
#[derive(Clone)]
pub struct SglangOcc3Config {
    /// Base URL of the SGLang server (e.g. `http://127.0.0.1:30001`).
    pub endpoint: Url,
    /// Bearer token; read from the environment by the host, never from
    /// `models.json`.
    pub api_key: Option<String>,
    /// In-memory already-resolved control-plane header values (never
    /// unresolved references). Sensitive: values must never reach Debug,
    /// errors, reports, audits or logs. Validated against the fixed allowlist
    /// in [`SglangOcc3Connector::new`] before any HTTP request is built.
    pub control_headers: Option<indexmap::IndexMap<String, String>>,
}

/// The only control header names this connector may attach, matched
/// case-insensitively. Every other name — including `authorization`, `host`,
/// `proxy-authorization`, `content-type`, `accept-encoding`, unapproved custom
/// names and malformed/CRLF names — is rejected before any HTTP request.
const CONTROL_HEADER_ALLOWLIST: [&str; 1] = ["x-pomerium-authorization"];
const CONTROL_HEADER_VALUE_MAX_BYTES: usize = 16384;

/// Whether `name` is an allowlisted control header name (case-insensitive).
pub fn control_header_name_is_allowed(name: &str) -> bool {
    CONTROL_HEADER_ALLOWLIST.contains(&name.to_ascii_lowercase().as_str())
}

/// Validated before the HTTP client is built, so every rejection happens
/// before any HTTP request. Header values are sensitive and never appear in
/// the (value-free) rejection error.
fn control_headers_valid(headers: Option<&indexmap::IndexMap<String, String>>) -> bool {
    let Some(headers) = headers else {
        return true;
    };
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (name, value) in headers {
        if !control_header_name_is_allowed(name)
            || !seen.insert(name.to_ascii_lowercase())
            || value.is_empty()
            || value.len() > CONTROL_HEADER_VALUE_MAX_BYTES
            || !value.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
        {
            return false;
        }
    }
    true
}

impl std::fmt::Debug for SglangOcc3Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut endpoint = self.endpoint.clone();
        let _ = endpoint.set_username("");
        let _ = endpoint.set_password(None);
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        f.debug_struct("SglangOcc3Config")
            .field("endpoint", &endpoint)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            // Names only: resolved control header values are sensitive and
            // are never rendered.
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

fn native_unsupported() -> Occ3Error {
    Occ3Error::NativeInferenceUnsupported {
        connector_id: CONNECTOR_ID.to_string(),
    }
}

/// The live connector. Cheap to clone behind an `Arc` (reqwest keeps an
/// internal connection pool).
#[derive(Debug)]
pub struct SglangOcc3Connector {
    client: reqwest::Client,
    config: SglangOcc3Config,
    native: Option<NativeExperiment>,
}

impl SglangOcc3Connector {
    pub fn new(config: SglangOcc3Config) -> Result<Self, Occ3Error> {
        if !config.endpoint.username().is_empty()
            || config.endpoint.password().is_some()
            || config.endpoint.query().is_some()
            || config.endpoint.fragment().is_some()
        {
            return Err(Occ3Error::TransferFailed {
                reason: "invalid connector endpoint".into(),
            });
        }
        // Unapproved, malformed, duplicated-case or non-printable control
        // headers refuse here, before the HTTP client exists: no request can
        // ever be built from them.
        if !control_headers_valid(config.control_headers.as_ref()) {
            return Err(Occ3Error::TransferFailed {
                reason: "invalid connector control headers".into(),
            });
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .no_gzip()
            .no_brotli()
            .no_zstd()
            .no_deflate()
            .timeout(OP_TIMEOUT)
            .build()
            .map_err(|e| Occ3Error::TransferFailed {
                reason: format!("http client build failed: {e}"),
            })?;
        Ok(Self {
            client,
            config,
            native: None,
        })
    }

    fn request_builder(
        &self,
        method: reqwest::Method,
        path: &str,
        timeout: Duration,
    ) -> Result<reqwest::RequestBuilder, Occ3Error> {
        let url = self
            .config
            .endpoint
            .join(path)
            .unwrap_or_else(|_| self.config.endpoint.clone());
        let mut builder = self
            .client
            .request(method, url)
            .timeout(timeout)
            .header("content-type", "application/json");
        if let Some(headers) = &self.config.control_headers {
            // Construction-time allowlisting means transport-owned headers
            // (content-type, accept-encoding, authorization) can never be
            // overridden from here. Values are validated printable ASCII at
            // construction, so parsing cannot fail; a parse error is still
            // a fixed, value-free failure before any HTTP request.
            for (name, value) in headers {
                let mut header_value = reqwest::header::HeaderValue::from_str(value)
                    .map_err(|_| Occ3Error::TransferFailed {
                        reason: "invalid connector control header value".into(),
                    })?;
                header_value.set_sensitive(true);
                builder = builder.header(name.as_str(), header_value);
            }
        }
        if let Some(key) = &self.config.api_key {
            Ok(builder.bearer_auth(key))
        } else {
            Ok(builder)
        }
    }

    async fn get_json(&self, path: &str, timeout: Duration) -> Result<Value, Occ3Error> {
        let response = self
            .request_builder(reqwest::Method::GET, path, timeout)?
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .map_err(|_| Occ3Error::TransferFailed {
                reason: "invalid connector JSON response".into(),
            })?;
        if !status.is_success() {
            return Err(http_status_error(status, &body));
        }
        Ok(body)
    }

    /// Current-incarnation runtime readiness from the live control plane.
    /// The production activation path mints its experimental permit from
    /// this handshake; it never fabricates an identity, a profile or a
    /// live-verified record.
    pub async fn runtime_identity(&self) -> Result<NativeRuntimeIdentity, Occ3Error> {
        let body = serde_json::json!({"schema":"occ3.runtime.identity.request.v1"});
        let identity: NativeRuntimeIdentity = self
            .control("/occ3/v1/native/runtime/identity", &body)
            .await?;
        if !identity.is_self_consistent() {
            return Err(native_invalid());
        }
        Ok(identity)
    }
}

/// An explicit process-local experimental permit. Never deserialized from
/// models.json and never promoted to the production LiveVerified setting.
/// The controller supplies the complete independently attested model identity
/// and the exact runtime profile from its isolated validation descriptor.
#[derive(Clone)]
pub struct NativeExperimentalPermit {
    pub attempt_id: String,
    pub identity: Occ3ModelIdentity,
    pub target: pi_ai::inference_attachment::InferenceStateTarget,
    pub runtime_id: String,
    pub runtime_generation: u64,
    pub access_domain: String,
    pub expires_at_epoch: u64,
    pub source_profile: pi_ai::source_capture::SourceCaptureProfile,
    pub profile_digest: String,
    pub max_capture_bytes: u64,
}
impl std::fmt::Debug for NativeExperimentalPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NativeExperimentalPermit(<private>)")
    }
}

#[derive(Default)]
struct NativeRecords {
    sessions: std::collections::BTreeMap<
        String,
        crate::core::occ::occ3::connector::NativeSessionEnrollment,
    >,
    publications: std::collections::BTreeMap<String, PublicationRecord>,
    // Retain the key before the first HTTP side effect. Logical close and
    // shutdown may both refresh retirement, never mint a second operation.
    retirements: std::collections::BTreeMap<String, RetirementRecord>,
    stopped: bool,
}
struct RetirementRecord {
    operation_id: String,
    source_handle: String,
    source_session_id: String,
}
struct PublicationRecord {
    request: crate::core::occ::occ3::connector::NativePublicationRequest,
    binding_digest: String,
    publication: Option<crate::core::occ::occ3::connector::NativePublication>,
    claimed: bool,
}
struct NativeExperiment {
    permit: NativeExperimentalPermit,
    records: std::sync::Mutex<NativeRecords>,
}
impl std::fmt::Debug for NativeExperiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NativeExperiment(<private>)")
    }
}
/// Current-incarnation runtime readiness from the live control plane
/// (`POST /occ3/v1/native/runtime/identity`). Every field is
/// server-attested; production activation mints its experimental permit
/// from this handshake instead of fabricating an identity or a profile.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRuntimeIdentity {
    pub schema: String,
    pub runtime_id: String,
    pub runtime_generation: u64,
    pub model_fingerprint: String,
    pub method_id: String,
    pub tp_layout_digest: String,
    pub layer_index: u32,
    pub hidden_dim: u32,
    pub dtype: String,
    pub expected_ranks: u32,
    pub ready_ranks: u32,
    pub profile_digest: String,
    pub source_capture: bool,
    pub continuous_prefix: bool,
    pub terminal_journal: bool,
    pub server_time_epoch: u64,
    pub max_ttl_seconds: u64,
    pub max_capture_bytes: u64,
}

impl NativeRuntimeIdentity {
    /// The server-attested readiness must be a complete, self-consistent
    /// incarnation: schema pinned, identifiers and digests well-formed,
    /// ranks complete, capture features on, and a bounded live TTL/cap.
    /// Only the shape is checked; identity binding is the caller's
    /// exact-equality check.
    pub fn is_self_consistent(&self) -> bool {
        self.schema == "occ3.runtime.identity.v1"
            && native_id(&self.runtime_id)
            && native_id(&self.method_id)
            && native_digest(&self.profile_digest)
            && native_digest(&self.tp_layout_digest)
            && self.expected_ranks > 0
            && self.ready_ranks == self.expected_ranks
            && self.hidden_dim > 0
            && matches!(self.dtype.as_str(), "bfloat16" | "float16" | "float32")
            && self.source_capture
            && self.continuous_prefix
            && self.terminal_journal
            && self.max_ttl_seconds > 0
            && self.max_ttl_seconds <= 86400
            && self.max_capture_bytes > 0
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeReadiness {
    runtime_epoch: String,
    runtime_generation: u64,
    model_fingerprint: String,
    method_id: String,
    expected_ranks: u32,
    ready_ranks: u32,
    tp_layout_digest: String,
    hidden_dim: u32,
    layer_index: u32,
    dtype: String,
    profile_digest: String,
    source_capture: bool,
    continuous_prefix: bool,
    terminal_journal: bool,
    /// Server-advertised capture wire cap (attachment_contract
    /// MAX_CAPTURE_BYTES). Absent on servers built before it was advertised;
    /// present it is authoritative and gates the permit's byte budget.
    max_capture_bytes: Option<u64>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EnrollmentResponse {
    schema: String,
    operation_id: String,
    session_id: String,
    principal_digest: String,
    runtime_id: String,
    runtime_generation: u64,
    model_fingerprint: String,
    expires_at_epoch: u64,
    readiness: NativeReadiness,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationResponse {
    schema: String,
    publication_id: String,
    state: String,
    #[serde(default)]
    binding_digest: Option<String>,
    #[serde(default)]
    source_handle: Option<String>,
    #[serde(default)]
    expires_at_epoch: Option<u64>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RetirementResponse {
    schema: String,
    operation_id: String,
    source_handle: String,
    revoked: bool,
    gpu_freed: bool,
    state: String,
}
fn native_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
pub(crate) fn native_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|v| v.is_ascii_alphanumeric() || b"._:-/".contains(&v))
}
pub(crate) fn native_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
fn native_invalid() -> Occ3Error {
    Occ3Error::TransferFailed {
        reason: "native control binding invalid".into(),
    }
}
/// The negotiation family the permit's full dotted method id names. The
/// envelope, the bind lease and the capability advertisement all derive
/// it from the same server-attested method id, never from a constant.
fn permit_method(native: &NativeExperiment) -> Result<Occ3Method, Occ3Error> {
    crate::core::occ3_connectors::method_family(&native.permit.identity.latent_method)
        .ok_or_else(native_invalid)
}
fn native_uncertain(message_id: &str) -> Occ3Error {
    Occ3Error::InferenceOutcomeUncertain {
        message_id: message_id.into(),
    }
}

impl SglangOcc3Connector {
    pub fn new_experimental(
        config: SglangOcc3Config,
        permit: NativeExperimentalPermit,
    ) -> Result<Self, Occ3Error> {
        let now = native_now();
        let target_url = Url::parse(&permit.target.base_url).map_err(|_| native_invalid())?;
        let safe_endpoint = config.endpoint.scheme() == "https"
            || (config.endpoint.scheme() == "http"
                && config.endpoint.host_str().is_some_and(|h| {
                    h == "localhost"
                        || h.parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                }));
        if !safe_endpoint
            || !crate::core::occ::occ3::inference::complete_identity(&permit.identity)
            || !native_id(&permit.target.provider)
            || !matches!(
                permit.source_profile.dtype.as_str(),
                "bfloat16" | "float16" | "float32"
            )
            || !native_id(&permit.attempt_id)
            || !native_id(&permit.runtime_id)
            || !native_id(&permit.access_domain)
            || permit.expires_at_epoch <= now
            || permit.expires_at_epoch > now.saturating_add(3600)
            || permit.max_capture_bytes == 0
            || !native_digest(&permit.profile_digest)
            || !native_digest(&permit.source_profile.layout_digest)
            || permit.source_profile.expected_ranks == 0
            || permit.source_profile.hidden_dim == 0
            || permit.target.model_id != permit.identity.model_id
            || target_url.origin() != config.endpoint.origin()
            || !target_url.username().is_empty()
            || target_url.password().is_some()
            || target_url.query().is_some()
            || target_url.fragment().is_some()
            || !native_id(&permit.identity.latent_method)
        {
            return Err(native_invalid());
        }
        let mut result = Self::new(config)?;
        result.native = Some(NativeExperiment {
            permit,
            records: std::sync::Mutex::new(NativeRecords::default()),
        });
        Ok(result)
    }
    fn experiment(&self) -> Result<&NativeExperiment, Occ3Error> {
        let native = self.native.as_ref().ok_or_else(native_unsupported)?;
        if native.permit.expires_at_epoch <= native_now()
            || native.records.lock().map_err(|_| native_invalid())?.stopped
        {
            return Err(native_unsupported());
        }
        Ok(native)
    }
    async fn control<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<T, Occ3Error> {
        let bytes =
            pi_ai::source_capture::canonical_json_bytes(body).map_err(|_| native_invalid())?;
        if bytes.len() > 16384 {
            return Err(native_invalid());
        }
        let mut response = self
            .request_builder(reqwest::Method::POST, path, HEALTH_TIMEOUT)?
            .header("accept-encoding", "identity")
            .body(bytes)
            .send()
            .await
            .map_err(|_| native_invalid())?;
        if !response.status().is_success()
            || response
                .headers()
                .get("content-encoding")
                .is_some_and(|v| v != "identity")
        {
            return Err(native_invalid());
        }
        let mut received = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| native_invalid())? {
            if received.len().saturating_add(chunk.len()) > 16384 {
                return Err(native_invalid());
            }
            received.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&received).map_err(|_| native_invalid())
    }
    fn check_enrollment_response(
        permit: &NativeExperimentalPermit,
        response: EnrollmentResponse,
        operation_id: &str,
        session_id: &str,
    ) -> Result<crate::core::occ::occ3::connector::NativeSessionEnrollment, Occ3Error> {
        use crate::core::occ::occ3::connector::NativeSessionEnrollment;
        let fingerprint = permit.identity.fingerprint();
        let ready = &response.readiness;
        if response.schema != "occ3.sessions.enrolled.v1"
            || response.operation_id != operation_id
            || response.session_id != session_id
            || !native_digest(&response.principal_digest)
            || response.runtime_id != permit.runtime_id
            || response.runtime_generation != permit.runtime_generation
            || response.model_fingerprint != fingerprint
            || response.expires_at_epoch != permit.expires_at_epoch
            || ready.runtime_epoch != permit.runtime_id
            || ready.runtime_generation != permit.runtime_generation
            || ready.model_fingerprint != fingerprint
            || ready.method_id != permit.identity.latent_method
            || ready.expected_ranks != permit.source_profile.expected_ranks
            || ready.ready_ranks != ready.expected_ranks
            || ready.tp_layout_digest != permit.source_profile.layout_digest
            || ready.hidden_dim != permit.source_profile.hidden_dim
            || ready.layer_index != permit.source_profile.layer_index
            || ready.dtype != permit.source_profile.dtype
            || ready.profile_digest != permit.profile_digest
            || !ready.source_capture
            || !ready.continuous_prefix
            || !ready.terminal_journal
        {
            return Err(native_invalid());
        }
        if let Some(wire_cap) = ready.max_capture_bytes {
            if permit.max_capture_bytes > wire_cap {
                // A permit above the server's advertised capture wire cap can
                // only fail mid-turn with an opaque invalid_attachment 400;
                // refuse it at enrollment with both numbers named instead.
                return Err(Occ3Error::TransferFailed {
                    reason: format!(
                        "occ3 capture byte cap {} exceeds server wire cap {}",
                        permit.max_capture_bytes, wire_cap
                    ),
                });
            }
        }
        let result = NativeSessionEnrollment {
            session_id: session_id.into(),
            principal_digest: response.principal_digest,
            identity: permit.identity.clone(),
            runtime_id: permit.runtime_id.clone(),
            runtime_generation: permit.runtime_generation,
            access_domain: permit.access_domain.clone(),
            expires_at_epoch: permit.expires_at_epoch,
            source_profile: permit.source_profile.clone(),
            profile_digest: permit.profile_digest.clone(),
        };
        Ok(result)
    }
    async fn enroll(
        &self,
        session_id: &str,
    ) -> Result<crate::core::occ::occ3::connector::NativeSessionEnrollment, Occ3Error> {
        let native = self.experiment()?;
        let permit = &native.permit;
        if !native_id(session_id) || session_id.len() > 128 {
            return Err(native_invalid());
        }
        let operation_id = format!("enroll-{}", uuid::Uuid::new_v4());
        let fingerprint = permit.identity.fingerprint();
        let body = serde_json::json!({"schema":"occ3.sessions.enroll.v1", "operation_id":operation_id,
            "session_id":session_id,"model_fingerprint":fingerprint,"runtime_id":permit.runtime_id,
            "runtime_generation":permit.runtime_generation,"access_domain":permit.access_domain,
            "expires_at_epoch":permit.expires_at_epoch});
        let response: EnrollmentResponse = self
            .control("/occ3/v1/native/sessions/enroll", &body)
            .await?;
        let result = Self::check_enrollment_response(permit, response, &operation_id, session_id)?;
        let mut records = native.records.lock().map_err(|_| native_invalid())?;
        if records.stopped
            || records
                .sessions
                .values()
                .any(|old| old.principal_digest != result.principal_digest)
            || (!records.sessions.contains_key(session_id) && records.sessions.len() >= 64)
        {
            return Err(native_invalid());
        }
        records.sessions.insert(session_id.into(), result.clone());
        Ok(result)
    }
    fn check_publication_response(
        response: &PublicationResponse,
        publication_id: &str,
        binding_digest: &str,
        handle: &str,
        expiry: u64,
        require_published: bool,
    ) -> Result<(), Occ3Error> {
        if response.schema != "occ3.publication.v1"
            || response.publication_id != publication_id
            || response.binding_digest.as_deref() != Some(binding_digest)
            || response.source_handle.as_deref() != Some(handle)
            || response.expires_at_epoch != Some(expiry)
            || !(if require_published {
                response.state == "published"
            } else {
                matches!(
                    response.state.as_str(),
                    "published" | "leased" | "injected" | "terminal" | "retired" | "expired"
                )
            })
        {
            return Err(native_uncertain(publication_id));
        }
        Ok(())
    }
}

fn map_reqwest_error(error: reqwest::Error) -> Occ3Error {
    if error.is_timeout() {
        Occ3Error::ConnectorTimeout {
            connector_id: CONNECTOR_ID.to_string(),
            timeout_ms: OP_TIMEOUT.as_millis() as u64,
        }
    } else if error.is_connect() {
        Occ3Error::ConnectorUnreachable {
            connector_id: CONNECTOR_ID.to_string(),
        }
    } else {
        Occ3Error::TransferFailed {
            reason: "connector request failed".into(),
        }
    }
}

fn http_status_error(status: reqwest::StatusCode, _body: &Value) -> Occ3Error {
    match status.as_u16() {
        401 | 403 => Occ3Error::Unauthorized,
        404 | 410 => Occ3Error::HandleExpired {
            handle_id: "unavailable".into(),
        },
        502 | 503 | 504 => Occ3Error::ConnectorUnreachable {
            connector_id: CONNECTOR_ID.to_string(),
        },
        code => Occ3Error::TransferFailed {
            reason: format!("connector HTTP status {code}"),
        },
    }
}

/// Legacy channel-name helper, retained for diagnostics only. This is NOT an
/// opaque state handle and must not be used by native inference attachments.
pub fn channel_id_for(message_id: &str) -> String {
    let cleaned: String = message_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "occ3-msg".to_string()
    } else {
        cleaned
    }
}

fn parse_identity(fields: &Value) -> Result<Occ3ModelIdentity, Occ3Error> {
    let str_field = |name: &str| -> Result<String, Occ3Error> {
        fields
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| {
                !value.trim().is_empty()
                    && value.len() <= 4096
                    && !value.chars().any(char::is_control)
                    && !matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "unknown" | "unverified"
                    )
            })
            .map(str::to_string)
            .ok_or_else(|| Occ3Error::TransferFailed {
                reason: format!("identity field missing: {name}"),
            })
    };
    Ok(Occ3ModelIdentity {
        model_id: str_field("model_id")?,
        architecture: str_field("architecture")?,
        checkpoint_revision: str_field("checkpoint_revision")?,
        weight_fingerprint: str_field("weight_fingerprint")?,
        tokenizer_fingerprint: str_field("tokenizer_fingerprint")?,
        chat_template_fingerprint: str_field("chat_template_fingerprint")?,
        quantisation: str_field("quantisation")?,
        tensor_geometry: str_field("tensor_geometry")?,
        kv_layout: str_field("kv_layout")?,
        latent_method: str_field("latent_method")?,
        runtime: str_field("runtime")?,
    })
}

#[cfg(test)]
fn parse_method(value: &str) -> Result<Occ3Method, Occ3Error> {
    match value {
        "kv-handoff" => Ok(Occ3Method::KvHandoff),
        "statebridge-prefix" => Ok(Occ3Method::StatebridgePrefix),
        other => Err(Occ3Error::TransferFailed {
            reason: format!("unknown occ3 method: {other}"),
        }),
    }
}

#[async_trait]
impl Occ3Connector for SglangOcc3Connector {
    fn connector_id(&self) -> &str {
        CONNECTOR_ID
    }

    fn protocol_version(&self) -> u32 {
        1
    }

    fn supports_native_inference(&self) -> bool {
        self.experiment().is_ok()
    }

    async fn enroll_native_session(
        &self,
        session_id: &str,
    ) -> Result<crate::core::occ::occ3::connector::NativeSessionEnrollment, Occ3Error> {
        self.enroll(session_id).await
    }

    async fn prepare_source_capture(
        &self,
        session_id: &str,
        target: pi_ai::inference_attachment::InferenceStateTarget,
        retained_count_limit: u32,
        max_capture_bytes: u64,
        limits: pi_ai::source_capture::SourceCaptureLimits,
    ) -> Result<pi_ai::source_capture::SourceCaptureLease, Occ3Error> {
        use pi_ai::source_capture::*;
        let enrolled = self.enroll(session_id).await?;
        let permit = &self.experiment()?.permit;
        if target != permit.target || max_capture_bytes > permit.max_capture_bytes {
            return Err(native_invalid());
        }
        SourceCaptureLease::new(
            target,
            SourceCaptureIntent {
                schema: SOURCE_CAPTURE_SCHEMA.into(),
                capture_id: format!("capture-{}", uuid::Uuid::new_v4()),
                source_session_id: session_id.into(),
                source_request_id: format!("source-{}", uuid::Uuid::new_v4()),
                turn_nonce: format!("turn-{}", uuid::Uuid::new_v4()),
                model_fingerprint: enrolled.identity.fingerprint(),
                runtime_id: enrolled.runtime_id,
                runtime_generation: enrolled.runtime_generation,
                method: enrolled.identity.latent_method,
                retained_count_limit,
                max_capture_bytes,
                expires_at_epoch: enrolled.expires_at_epoch,
                span_policy: SOURCE_SPAN_POLICY.into(),
            },
            enrolled.source_profile,
            limits,
        )
        .map_err(|_| native_invalid())
    }

    async fn publish_native_source(
        &self,
        request: crate::core::occ::occ3::connector::NativePublicationRequest,
    ) -> Result<crate::core::occ::occ3::connector::NativePublication, Occ3Error> {
        use crate::core::occ::occ3::connector::NativePublication;
        use crate::core::occ::occ3::types::Occ3Envelope;
        let native = self.experiment()?;
        let intent = request.source.intent();
        let receipt = request.source.receipt().ok_or_else(native_invalid)?;
        if !native_id(&request.message_id)
            || request.message_id.len() > 128
            || !native_id(&request.publication_id)
            || request.tool_name != "occ3_handoff"
            || request.source.tool_bindings().len() != 1
            || request
                .source
                .validate_tool(
                    &request.tool_call_id,
                    &request.tool_name,
                    &request.tool_arguments,
                )
                .is_err()
            || intent.model_fingerprint != native.permit.identity.fingerprint()
            || intent.runtime_id != native.permit.runtime_id
            || intent.runtime_generation != native.permit.runtime_generation
            || intent.method != native.permit.identity.latent_method
            || request.source.target() != &native.permit.target
            || receipt.expires_at_epoch <= native_now()
        {
            return Err(native_invalid());
        }
        // Refresh both memberships/profile before the first publication side effect.
        let source = self.enroll(&intent.source_session_id).await?;
        let receiver = self.enroll(&request.receiver_session_id).await?;
        if source.principal_digest != receiver.principal_digest
            || source.access_domain != receiver.access_domain
            || source.session_id == receiver.session_id
            || receipt.expires_at_epoch > receiver.expires_at_epoch
        {
            return Err(Occ3Error::Unauthorized);
        }
        let body = serde_json::json!({"schema":"occ3.publish.v1","publication_id":request.publication_id,
            "source_handle":receipt.source_handle,"source_session_id":source.session_id,
            "receiver_session_id":receiver.session_id,"message_id":request.message_id,
            "source_turn_nonce":intent.turn_nonce,"tool_call_id":request.tool_call_id,
            "expires_at_epoch":receipt.expires_at_epoch});
        let mut binding = body.clone();
        binding
            .as_object_mut()
            .ok_or_else(native_invalid)?
            .remove("schema");
        binding["principal_digest"] = serde_json::json!(source.principal_digest);
        let digest = pi_ai::source_capture::sha256(
            &pi_ai::source_capture::canonical_json_bytes(&binding).map_err(|_| native_invalid())?,
        );
        {
            let mut records = native.records.lock().map_err(|_| native_invalid())?;
            if let Some(old) = records.publications.get(&request.publication_id) {
                // Public retries never rerun publish or allocate a new consume.
                if old.binding_digest != digest {
                    return Err(Occ3Error::Unauthorized);
                }
                return Err(native_uncertain(&request.message_id));
            }
            if records.publications.len() >= 256
                || records
                    .publications
                    .values()
                    .any(|old| old.request.source.intent().capture_id == intent.capture_id)
            {
                return Err(native_invalid());
            }
            records.publications.insert(
                request.publication_id.clone(),
                PublicationRecord {
                    request: request.clone(),
                    binding_digest: digest.clone(),
                    publication: None,
                    claimed: false,
                },
            );
        }
        let reply: Result<PublicationResponse, _> =
            self.control("/occ3/v1/native/publish", &body).await;
        let response = match reply {
            Ok(value) => value,
            Err(_) => {
                // One status read under exactly the same key. Absence/pending is
                // not proof a delayed first request can never commit: no replay.
                let status = serde_json::json!({"schema":"occ3.publication.status.v1","publication_id":request.publication_id});
                self.control("/occ3/v1/native/publication/status", &status)
                    .await
                    .map_err(|_| native_uncertain(&request.message_id))?
            }
        };
        Self::check_publication_response(
            &response,
            &request.publication_id,
            &digest,
            &receipt.source_handle,
            receipt.expires_at_epoch,
            true,
        )?;
        let publication = NativePublication {
            publication_id: request.publication_id.clone(),
            binding_digest: digest,
            envelope: Occ3Envelope {
                schema: "optimus.occ3.v1".into(),
                protocol_version: 1,
                message_id: request.message_id.clone(),
                sender_session_id: source.session_id,
                receiver_session_id: receiver.session_id,
                model_fingerprint: native.permit.identity.fingerprint(),
                runtime_id: native.permit.runtime_id.clone(),
                connector_id: CONNECTOR_ID.into(),
                latent_method: permit_method(native)?,
                state_handle: Occ3HandleRef {
                    handle_id: receipt.source_handle,
                    connector_id: CONNECTOR_ID.into(),
                    model_fingerprint: native.permit.identity.fingerprint(),
                    runtime_generation: native.permit.runtime_generation,
                    expires_at_epoch: receipt.expires_at_epoch,
                    state_size_bytes: receipt.byte_count,
                },
                // Fixed control instruction: source data and tool arguments are
                // never copied into the successful receiver message.
                control_summary: "Continue the authorized task using the attached state.".into(),
                fallback_reference: "text".into(),
            },
        };
        let mut records = native.records.lock().map_err(|_| native_invalid())?;
        if records.stopped {
            return Err(native_uncertain(&request.message_id));
        }
        records
            .publications
            .get_mut(&request.publication_id)
            .ok_or_else(native_invalid)?
            .publication = Some(publication.clone());
        Ok(publication)
    }

    async fn prepare_inference(
        &self,
        request: crate::core::occ::occ3::inference::PrepareInferenceRequest,
    ) -> Result<pi_ai::inference_attachment::InferenceStateLease, Occ3Error> {
        use crate::core::occ::occ3::inference::{inference_attachment, VerifiedInferenceRuntime};
        let native = self.experiment()?;
        if request.target != native.permit.target
            || request.receiver_identity != native.permit.identity
        {
            return Err(native_invalid());
        }
        self.enroll(&request.receiver_session_id).await?;
        let mut records = native.records.lock().map_err(|_| native_invalid())?;
        let record = records
            .publications
            .values_mut()
            .find(|record| {
                record
                    .publication
                    .as_ref()
                    .is_some_and(|p| p.envelope == request.envelope)
            })
            .ok_or(Occ3Error::Unauthorized)?;
        if record.claimed {
            return Err(native_uncertain(&request.envelope.message_id));
        }
        let attachment = inference_attachment(
            &request,
            CONNECTOR_ID,
            &VerifiedInferenceRuntime {
                model_identity: native.permit.identity.clone(),
                runtime_id: native.permit.runtime_id.clone(),
                generation: native.permit.runtime_generation,
                methods: vec![permit_method(native)?],
            },
        )?;
        let lease =
            pi_ai::inference_attachment::InferenceStateLease::new(request.target, attachment)
                .map_err(|_| native_invalid())?;
        record.claimed = true;
        Ok(lease)
    }

    async fn retire_native_source(
        &self,
        source: &pi_ai::source_capture::SourceCaptureLease,
    ) -> Result<crate::core::occ::occ3::connector::NativeRetirement, Occ3Error> {
        let native = self.native.as_ref().ok_or_else(native_unsupported)?;
        let intent = source.intent();
        let receipt = source.retirement_receipt().ok_or_else(native_invalid)?;
        if intent.runtime_id != native.permit.runtime_id
            || intent.runtime_generation != native.permit.runtime_generation
            || intent.model_fingerprint != native.permit.identity.fingerprint()
        {
            return Err(native_invalid());
        }
        source.cancel();
        let operation_id = {
            let mut records = native.records.lock().map_err(|_| native_invalid())?;
            if let Some(retirement) = records.retirements.get(&intent.capture_id) {
                if retirement.source_handle != receipt.source_handle
                    || retirement.source_session_id != intent.source_session_id
                {
                    return Err(native_invalid());
                }
                retirement.operation_id.clone()
            } else {
                if records.retirements.len() >= 256 {
                    return Err(native_invalid());
                }
                let operation_id = format!("retire-{}", uuid::Uuid::new_v4());
                records.retirements.insert(
                    intent.capture_id.clone(),
                    RetirementRecord {
                        operation_id: operation_id.clone(),
                        source_handle: receipt.source_handle.clone(),
                        source_session_id: intent.source_session_id.clone(),
                    },
                );
                operation_id
            }
        };
        let response: RetirementResponse = self.control("/occ3/v1/native/retire", &serde_json::json!({
            "schema":"occ3.retire.v1","operation_id":operation_id,"source_handle":receipt.source_handle,
            "source_session_id":intent.source_session_id})).await?;
        if response.schema != "occ3.retired.v1"
            || response.operation_id != operation_id
            || response.source_handle != receipt.source_handle
            || !response.revoked
            || !matches!(response.state.as_str(), "retired" | "expired")
        {
            return Err(native_invalid());
        }
        Ok(crate::core::occ::occ3::connector::NativeRetirement {
            revoked: response.revoked,
            gpu_freed: response.gpu_freed,
        })
    }

    async fn health(&self) -> Result<(), Occ3Error> {
        self.get_json("/occ3/v1/status", HEALTH_TIMEOUT)
            .await
            .map(|_| ())
    }

    async fn discover_capabilities(&self) -> Result<Occ3ConnectorCapabilities, Occ3Error> {
        let native = self.experiment()?;
        let session_id = native
            .records
            .lock()
            .map_err(|_| native_invalid())?
            .sessions
            .keys()
            .next()
            .cloned()
            .ok_or_else(native_unsupported)?;
        self.enroll(&session_id).await?;
        Ok(Occ3ConnectorCapabilities {
            protocol_version: 1,
            connector_version: "experimental-native-v1".into(),
            model_identity: native.permit.identity.clone(),
            methods: vec![permit_method(native)?],
            tensor_formats: vec![native.permit.source_profile.dtype.clone()],
            max_state_bytes: native.permit.max_capture_bytes,
            state_ttl_seconds: native.permit.expires_at_epoch.saturating_sub(native_now()),
            max_concurrent_handles: 1,
            cache_isolation: "experimental-private-worker-profile".into(),
            features: std::collections::BTreeMap::from([
                (
                    "native_inference_attachment".into(),
                    crate::core::occ::occ3::inference::NATIVE_INFERENCE_CONTRACT.into(),
                ),
                (
                    "validation".into(),
                    "isolated-experimental-permit-not-live-verified".into(),
                ),
            ]),
        })
    }

    async fn model_identity(&self) -> Result<Occ3ModelIdentity, Occ3Error> {
        let body = self.get_json("/occ3/v1/identity", HEALTH_TIMEOUT).await?;
        let fields = body
            .get("fields")
            .ok_or_else(|| Occ3Error::TransferFailed {
                reason: "identity response missing fields".to_string(),
            })?;
        parse_identity(fields)
    }

    async fn capture_latent(&self, _request: CaptureRequest) -> Result<Occ3HandleRef, Occ3Error> {
        // Do not send capture_context as plaintext to the legacy /send route.
        Err(native_unsupported())
    }

    async fn deliver_latent(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(native_unsupported())
    }

    async fn consume_latent(
        &self,
        _request: ConsumeRequest,
    ) -> Result<Occ3TransferMetrics, Occ3Error> {
        // A successful detached /recv answer is not use by the native receiver.
        Err(native_unsupported())
    }

    async fn acknowledge_delivery(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(native_unsupported())
    }

    async fn release_handle(&self, _handle: &Occ3HandleRef) -> Result<(), Occ3Error> {
        Err(native_unsupported())
    }

    async fn shutdown(&self) -> Result<(), Occ3Error> {
        if let Some(native) = &self.native {
            let sources: Vec<_> = {
                let mut records = native.records.lock().map_err(|_| native_invalid())?;
                records.stopped = true;
                records
                    .publications
                    .values()
                    .map(|r| r.request.source.clone())
                    .collect()
            };
            let mut failed = false;
            for source in sources {
                if self.retire_native_source(&source).await.is_err() {
                    failed = true;
                }
            }
            if failed {
                return Err(native_invalid());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TS12 live regression: the deployed server (PR #200 baseline) carries
    // `max_capture_bytes` in the enrollment readiness payload. Captured
    // verbatim from the live candidate on 2026-10-26T14:12Z (enroll 200).
    #[test]
    fn enrollment_readiness_parses_the_live_payload_with_the_wire_cap() {
        let payload = r#"{"schema":"occ3.sessions.enrolled.v1","operation_id":"enroll-diag-5381b3304ba2",
"session_id":"probe-diag-ee52133b7153","principal_digest":"07e7990bcc87abb833e71344940cb9452a7972ea0c93a0f8d3499f750e1c70e8",
"runtime_id":"36f66c428aa5421a91bf56d97f5a127c","runtime_generation":1,
"model_fingerprint":"sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732","expires_at_epoch":1793024970,
"readiness":{"runtime_epoch":"36f66c428aa5421a91bf56d97f5a127c","runtime_generation":1,
"model_fingerprint":"sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732",
"method_id":"statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1","expected_ranks":8,"ready_ranks":8,
"tp_layout_digest":"a55bf3e06038c4683c5ddeffcb947c46e962f2184291de6d55a68b04290fa77d","hidden_dim":6144,
"layer_index":77,"dtype":"bfloat16","profile_digest":"77f74ffcd6e49a576ebfbe73ee1dfd016c83446f6675e105afd377cdb68d9b4e",
"source_capture":true,"continuous_prefix":true,"terminal_journal":true,"max_capture_bytes":134217728}}"#;
        let response: EnrollmentResponse = serde_json::from_str(payload).expect("live payload must parse");
        assert_eq!(response.readiness.max_capture_bytes, Some(134_217_728));
        assert_eq!(response.readiness.ready_ranks, 8);
        assert!(response.readiness.source_capture);
        // Servers built before the cap was advertised still parse (None).
        let pre_cap = payload.replace(",\"max_capture_bytes\":134217728", "");
        let legacy: EnrollmentResponse = serde_json::from_str(&pre_cap).expect("legacy payload must parse");
        assert_eq!(legacy.readiness.max_capture_bytes, None);
    }


    #[test]
    fn channel_ids_are_sanitized_to_the_plugin_grammar() {
        assert_eq!(channel_id_for("agentmsg_fc9f34a3"), "agentmsg_fc9f34a3");
        assert_eq!(channel_id_for("msg with spaces!"), "msg-with-spaces-");
        assert_eq!(channel_id_for(""), "occ3-msg");
        let long = "a".repeat(100);
        assert_eq!(channel_id_for(&long).len(), 64);
    }

    #[test]
    fn identity_fields_map_one_to_one() {
        let fields = json!({
            "model_id": "glm-5.3",
            "architecture": "GlmMoeDsaForCausalLM",
            "checkpoint_revision": "aca966e4",
            "weight_fingerprint": "abc123",
            "tokenizer_fingerprint": "tok",
            "chat_template_fingerprint": "chat",
            "quantisation": "fp8-e4m3",
            "tensor_geometry": "hidden_size=6144",
            "kv_layout": "tp_size=8",
            "latent_method": "kv-radix-channel-v1",
            "runtime": "sglang/0.5.21",
        });
        let identity = parse_identity(&fields).unwrap();
        assert_eq!(identity.model_id, "glm-5.3");
        assert_eq!(identity.latent_method, "kv-radix-channel-v1");
        // Fingerprint is a stable digest: identical inputs, identical output,
        // and it changes when any field changes.
        let fingerprint_a = identity.fingerprint();
        let fingerprint_b = parse_identity(&fields).unwrap().fingerprint();
        assert_eq!(fingerprint_a, fingerprint_b);
        let mut changed = fields.clone();
        changed["kv_layout"] = json!("tp_size=4");
        assert_ne!(
            fingerprint_a,
            parse_identity(&changed).unwrap().fingerprint()
        );
        assert!(parse_identity(&json!({})).is_err());
    }

    #[test]
    fn methods_parse_and_reject() {
        assert!(matches!(
            parse_method("kv-handoff").unwrap(),
            Occ3Method::KvHandoff
        ));
        assert!(matches!(
            parse_method("statebridge-prefix").unwrap(),
            Occ3Method::StatebridgePrefix
        ));
        assert!(parse_method("nonsense").is_err());
    }

    #[test]
    fn http_status_errors_map_to_connector_semantics() {
        let unauthorized = http_status_error(
            reqwest::StatusCode::UNAUTHORIZED,
            &json!({"detail": "Unauthorized"}),
        );
        assert!(matches!(unauthorized, Occ3Error::Unauthorized));
        let not_found = http_status_error(
            reqwest::StatusCode::NOT_FOUND,
            &json!({"detail": "unknown channel: ghost"}),
        );
        assert!(matches!(not_found, Occ3Error::HandleExpired { .. }));
        let bad_gateway = http_status_error(reqwest::StatusCode::BAD_GATEWAY, &json!({}));
        assert!(matches!(
            bad_gateway,
            Occ3Error::ConnectorUnreachable { .. }
        ));
    }

    #[test]
    fn backend_derived_enrollment_binds_outer_and_nested_epoch_generation() {
        // Generated by actual Python control ASGI + WorkerRuntime.health with
        // simulated ranks/clock. Metadata conformance only, not GPU readiness.
        // Campaign native-enrollment-backend-fixture.json SHA256:
        // 4f16078bc6ec924a3129ae575837ccd554df01474520d2d6b838b239f0d875fa
        let response: serde_json::Value = serde_json::from_str(r#"{"expires_at_epoch":2000000100,"model_fingerprint":"sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732","operation_id":"fixture-operation","principal_digest":"fa5d3c05888c3bcae60ffab01b635d68af5f149f503d5f9ca0053bc412ef768c","readiness":{"continuous_prefix":true,"dtype":"bfloat16","expected_ranks":8,"hidden_dim":6144,"layer_index":77,"method_id":"statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1","model_fingerprint":"sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732","profile_digest":"4ca1b87ffc3fe81c72134ef87673a9940f14bee34da410f535fef3e39f70a16f","ready_ranks":8,"runtime_epoch":"fixture-boot-epoch-001","runtime_generation":7,"source_capture":true,"terminal_journal":true,"tp_layout_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"runtime_generation":7,"runtime_id":"fixture-boot-epoch-001","schema":"occ3.sessions.enrolled.v1","session_id":"fixture-source"}"#).unwrap();
        let identity: Occ3ModelIdentity = serde_json::from_str(r#"{"architecture":"GlmMoeDsaForCausalLM","chat_template_fingerprint":"sha256:3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c","checkpoint_revision":"aca966e4e02791568aa6a4ced368624b3d897f42","kv_layout":"declared-launch:tp8:kv-fp8_e4m3:continuous-prefix","latent_method":"statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1","model_id":"glm-5.3","quantisation":"fp8","runtime":"sglang:0.5.21:e00930c5489053f26d86b179cee0d087f846acbb","tensor_geometry":"layers78:hidden6144:prenorm-layer77","tokenizer_fingerprint":"sha256:08d2801689875e6819f26919b3e07ecf42693dd58e0527be477de286fb8ef532","weight_fingerprint":"metadata-manifest-sha256:493a065ae286f3ca2077d36e8d0fd0bb931d7443d6dab309e7e2cf288c32d23e"}"#).unwrap();
        assert_eq!(
            identity.fingerprint(),
            "sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732"
        );
        let permit = NativeExperimentalPermit {
            attempt_id: "fixture-attempt".into(),
            identity,
            target: pi_ai::inference_attachment::InferenceStateTarget {
                provider: "fixture-provider".into(),
                model_id: "glm-5.3".into(),
                base_url: "http://127.0.0.1:1/v1".into(),
            },
            runtime_id: "fixture-boot-epoch-001".into(),
            runtime_generation: 7,
            access_domain: "fixture-family".into(),
            expires_at_epoch: 2000000100,
            source_profile: pi_ai::source_capture::SourceCaptureProfile {
                layer_index: 77,
                hidden_dim: 6144,
                dtype: "bfloat16".into(),
                layout_digest: "b".repeat(64),
                expected_ranks: 8,
            },
            profile_digest: "4ca1b87ffc3fe81c72134ef87673a9940f14bee34da410f535fef3e39f70a16f"
                .into(),
            max_capture_bytes: 268435456,
        };
        let validate = |value: serde_json::Value| {
            SglangOcc3Connector::check_enrollment_response(
                &permit,
                serde_json::from_value(value).unwrap(),
                "fixture-operation",
                "fixture-source",
            )
        };
        let actual = validate(response.clone()).unwrap();
        assert_eq!(actual.runtime_id, "fixture-boot-epoch-001");
        assert_eq!(actual.runtime_generation, 7);
        assert_ne!(
            actual.runtime_id, actual.identity.runtime,
            "engine/version is not boot epoch"
        );
        for path in ["/runtime_id", "/readiness/runtime_epoch"] {
            let mut changed = response.clone();
            *changed.pointer_mut(path).unwrap() = json!("fixture-boot-epoch-restarted");
            assert!(validate(changed).is_err(), "{path}");
        }
        for path in ["/runtime_generation", "/readiness/runtime_generation"] {
            let mut changed = response.clone();
            *changed.pointer_mut(path).unwrap() = json!(8);
            assert!(validate(changed).is_err(), "{path}");
        }
        // Live TS11 regression: the server advertises its capture wire cap in
        // readiness; a permit above it must be refused at enrollment with both
        // numbers named instead of failing mid-turn with an opaque 400.
        let mut capped = response.clone();
        capped["readiness"]["max_capture_bytes"] = json!(134217728u64);
        let refusal = validate(capped).unwrap_err();
        assert!(
            format!("{refusal:?}").contains("exceeds server wire cap"),
            "named refusal carrying both numbers, got {refusal:?}"
        );
        let mut equal = response.clone();
        equal["readiness"]["max_capture_bytes"] = json!(268435456u64);
        assert!(validate(equal).is_ok(), "equal cap accepted");
        let mut changed = response;
        changed["readiness"]["runtime_epoch"] = json!(permit.identity.runtime);
        assert!(
            validate(changed).is_err(),
            "deployment/model runtime label is never an epoch alias"
        );
    }

    #[test]
    fn diagnostics_never_echo_synthetic_secrets() {
        let secret = "synthetic-private-key";
        let mut control_headers = indexmap::IndexMap::new();
        control_headers.insert("x-pomerium-authorization".to_string(), secret.into());
        let config = SglangOcc3Config {
            endpoint: Url::parse(&format!("http://user:{secret}@127.0.0.1:1/?key={secret}"))
                .unwrap(),
            api_key: Some(secret.into()),
            control_headers: Some(control_headers),
        };
        assert!(!format!("{config:?}").contains(secret));
        assert!(SglangOcc3Connector::new(config).is_err());
        let error = http_status_error(reqwest::StatusCode::BAD_REQUEST, &json!({"detail": secret}));
        assert!(!error.to_string().contains(secret));
    }

    #[test]
    fn control_headers_accept_only_the_gateway_name_and_redact_values() {
        let secret = "synthetic-gateway-control-value";
        let mut control_headers = indexmap::IndexMap::new();
        control_headers.insert("X-Pomerium-Authorization".to_string(), secret.into());
        let config = SglangOcc3Config {
            endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
            api_key: None,
            control_headers: Some(control_headers),
        };
        let debug = format!("{config:?}");
        assert!(debug
            .to_ascii_lowercase()
            .contains("x-pomerium-authorization"));
        assert!(!debug.contains(secret), "values are never rendered");
        let connector = SglangOcc3Connector::new(config).unwrap();
        assert!(!format!("{connector:?}").contains(secret));
        // An empty optional map is harmless.
        SglangOcc3Connector::new(SglangOcc3Config {
            endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
            api_key: None,
            control_headers: Some(indexmap::IndexMap::new()),
        })
        .unwrap();
        SglangOcc3Connector::new(SglangOcc3Config {
            endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
            api_key: None,
            control_headers: None,
        })
        .unwrap();
    }

    #[test]
    fn unapproved_control_header_names_refuse_before_any_http() {
        let value = "synthetic-control-header-value";
        for name in [
            "authorization",
            "Authorization",
            "AUTHORIZATION",
            "host",
            "proxy-authorization",
            "Proxy-Authorization",
            "content-type",
            "accept-encoding",
            "x-pomerium-authorization ",
            " x-pomerium-authorization",
            "x-pomerium-authorization\r\nx-injected: 1",
            "x-other-gateway",
            "",
        ] {
            let mut control_headers = indexmap::IndexMap::new();
            control_headers.insert(name.to_string(), value.to_string());
            let config = SglangOcc3Config {
                endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
                api_key: None,
                control_headers: Some(control_headers),
            };
            let error =
                SglangOcc3Connector::new(config).expect_err("unapproved name must refuse");
            assert!(!error.to_string().contains(value), "{name:?}");
        }
    }

    #[test]
    fn control_header_values_are_sensitive_and_never_rendered_in_debug() {
        use reqwest::header::HeaderName;
        let secret = "synthetic-gateway-control-value";
        let mut control_headers = indexmap::IndexMap::new();
        control_headers.insert("x-pomerium-authorization".to_string(), secret.into());
        let connector = SglangOcc3Connector::new(SglangOcc3Config {
            endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
            api_key: Some("synthetic-fixture-key".into()),
            control_headers: Some(control_headers),
        })
        .unwrap();
        let builder = connector
            .request_builder(
                reqwest::Method::POST,
                "/occ3/v1/native/sessions/enroll",
                HEALTH_TIMEOUT,
            )
            .unwrap();
        let request = builder.build().unwrap();
        let name = HeaderName::from_bytes(b"x-pomerium-authorization").unwrap();
        let value = request.headers().get(&name).unwrap();
        // Raw on-wire bytes are exactly the configured value.
        assert_eq!(value.as_bytes(), secret.as_bytes());
        // The value is marked sensitive: HeaderMap and Request Debug show the
        // redaction marker, never the value.
        assert!(value.is_sensitive());
        let header_debug = format!("{:?}", request.headers());
        assert!(header_debug.contains("Sensitive"), "marker present");
        assert!(!header_debug.contains(secret), "value absent from HeaderMap Debug");
        let request_debug = format!("{:?}", request);
        assert!(request_debug.contains("Sensitive"));
        assert!(!request_debug.contains(secret), "value absent from Request Debug");
        // Header precedence: bearer auth stays present alongside the control header.
        assert!(request
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .is_some());
        assert!(request
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_some());
    }

    #[test]
    fn malformed_duplicate_case_and_oversize_control_headers_refuse() {
        let mut invalid: Vec<indexmap::IndexMap<String, String>> = Vec::new();
        for value in ["", "value\r\nwith-crlf", "value\u{7}with-bell", "v\u{80}obs-text"] {
            let mut headers = indexmap::IndexMap::new();
            headers.insert("x-pomerium-authorization".to_string(), value.to_string());
            invalid.push(headers);
        }
        let mut oversize = indexmap::IndexMap::new();
        oversize.insert(
            "x-pomerium-authorization".to_string(),
            "x".repeat(16385),
        );
        invalid.push(oversize);
        let mut duplicated = indexmap::IndexMap::new();
        duplicated.insert("X-Pomerium-Authorization".to_string(), "one".to_string());
        duplicated.insert("x-pomerium-authorization".to_string(), "two".to_string());
        invalid.push(duplicated);
        for headers in invalid {
            let config = SglangOcc3Config {
                endpoint: Url::parse("http://127.0.0.1:1").unwrap(),
                api_key: None,
                control_headers: Some(headers),
            };
            let error = SglangOcc3Connector::new(config)
                .expect_err("malformed control header must refuse");
            assert!(!error.to_string().contains("one"));
            assert!(!error.to_string().contains("two"));
        }
    }
}
