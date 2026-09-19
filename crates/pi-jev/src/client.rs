//! SystemOne client: HTTP transport, retry policy, bounded payloads and the two
//! `SystemOne` implementations (DESIGN.md sections 2, 4, 8).
//!
//! Invariants enforced here:
//! - `Off` performs no client construction, no DNS, no socket and no task spawn. The cheapest
//!   path is `DisabledSystemOne`, which returns skips without touching the transport.
//! - Modes are explicit; the client returns data and never applies recommendations.
//! - The credential never enters a URL, a log line, an error, an argv or the retry counters.
//! - The response is validated in full (types.rs) before any answer is used. Failures are skips.
//! - Every request is bounded: max payload bytes, max question count, per-attempt timeout,
//!   finite retries, honoring `retry-after` on 429/529.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;

use parking_lot::Mutex;
use serde_json::Value;

use crate::credential::SecretString;
use crate::error::{sanitize_detail, sanitize_url, JevError};
use crate::types::{
    validate_request_shape, validate_response, Answer, BoxFuture, DecisionBundle, DecisionCategory,
    DecisionOutcome, DecisionRecord, JevMode, SystemOne, SystemOneRequest, SystemOneResponse, Transport,
    DEFAULT_MODEL, MAX_QUESTIONS_PER_REQUEST, SYSTEMONE_PATH,
};

/// Production base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// Per-attempt timeout. The comparison lane has its own deadline; this bounds one socket call.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// Maximum retries after the initial attempt (finite).
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// First backoff delay; doubled per attempt (documented policy, see `backoff_policy_line`).
pub const DEFAULT_BACKOFF_INITIAL: Duration = Duration::from_millis(500);

/// Backoff ceiling.
pub const DEFAULT_BACKOFF_MAX: Duration = Duration::from_secs(8);

/// Default payload cap for the serialized request body.
pub const DEFAULT_MAX_PAYLOAD_BYTES: usize = 256 * 1024;

/// Default hard cap on the response body read from the socket (defence against huge payloads).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

/// HTTP statuses that are retried when the budget allows.
pub const RETRYABLE_STATUSES: [u16; 6] = [408, 425, 429, 500, 502, 503];

/// `529 Overloaded` is documented by the API and is retried like 429.
pub const STATUS_OVERLOADED: u16 = 529;

/// Documented backoff policy, quoted in status output and in the lane report.
pub fn backoff_policy_line() -> &'static str {
    "exponential backoff from 500ms doubling to an 8s ceiling, finite retries, honoring Retry-After on 429/529"
}

/// Opaque local request id. Used for correlation records; carries no user content.
///
/// Opaque and local by design: an investigator can correlate a record without a copy of the
/// prompt, the state, or the transcript.
pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Current UTC time as RFC 3339 (millisecond precision), for record start/terminal timestamps.
pub fn utc_now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Current UTC time in epoch milliseconds.
pub fn utc_now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Bounds and caps for one client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevLimits {
    /// Base URL; the `SYSTEMONE_PATH` is appended.
    pub base_url: String,
    /// Per-attempt timeout.
    pub timeout: Duration,
    /// Retries after the first attempt.
    pub max_retries: u32,
    /// First backoff delay.
    pub backoff_initial: Duration,
    /// Backoff ceiling.
    pub backoff_max: Duration,
    /// Max serialized request bytes.
    pub max_payload_bytes: usize,
    /// Max response body bytes read from the socket.
    pub max_response_bytes: usize,
    /// Max questions per request.
    pub max_questions: usize,
}

impl Default for JevLimits {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            backoff_initial: DEFAULT_BACKOFF_INITIAL,
            backoff_max: DEFAULT_BACKOFF_MAX,
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_questions: MAX_QUESTIONS_PER_REQUEST,
        }
    }
}

impl JevLimits {
    /// Endpoint URL. The credential never appears here.
    pub fn endpoint(&self) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), SYSTEMONE_PATH)
    }

    /// Rejects limits that would make the client unbounded.
    pub fn validate(&self) -> Result<(), JevError> {
        let url = url::Url::parse(&self.endpoint())
            .map_err(|_| JevError::config("base url is not a valid absolute url"))?;
        if url.scheme() != "https" && url.scheme() != "http" {
            return Err(JevError::config("base url scheme must be http or https"));
        }
        if self.timeout.is_zero() {
            return Err(JevError::config("timeout must be greater than zero"));
        }
        if self.max_payload_bytes == 0 {
            return Err(JevError::config("max payload bytes must be greater than zero"));
        }
        if self.max_response_bytes == 0 {
            return Err(JevError::config("max response bytes must be greater than zero"));
        }
        if self.max_questions == 0 || self.max_questions > MAX_QUESTIONS_PER_REQUEST {
            return Err(JevError::config(format!(
                "max questions must be between 1 and {MAX_QUESTIONS_PER_REQUEST}"
            )));
        }
        Ok(())
    }
}

/// Counters for `/jev status`. Never contains a credential or prompt content.
#[derive(Debug, Default)]
pub struct JevStats {
    pub attempts: AtomicU64,
    pub successes: AtomicU64,
    pub failures: AtomicU64,
    pub retries: AtomicU64,
    pub rate_limited: AtomicU64,
    pub timeouts: AtomicU64,
    pub malformed: AtomicU64,
    pub validation_skips: AtomicU64,
    pub payload_rejections: AtomicU64,
    pub in_flight: AtomicU64,
    /// Epoch milliseconds of the last successful call.
    pub last_success_ms: AtomicU64,
    /// Latency of the last successful call.
    pub last_latency_ms: AtomicU64,
}

/// Point-in-time copy of the counters (safe to render from the UI thread).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JevStatsSnapshot {
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub retries: u64,
    pub rate_limited: u64,
    pub timeouts: u64,
    pub malformed: u64,
    pub validation_skips: u64,
    pub payload_rejections: u64,
    pub in_flight: u64,
    pub last_success_ms: u64,
    pub last_latency_ms: u64,
}

impl JevStats {
    /// Snapshot of every counter.
    pub fn snapshot(&self) -> JevStatsSnapshot {
        JevStatsSnapshot {
            attempts: self.attempts.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            rate_limited: self.rate_limited.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            malformed: self.malformed.load(Ordering::Relaxed),
            validation_skips: self.validation_skips.load(Ordering::Relaxed),
            payload_rejections: self.payload_rejections.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            last_success_ms: self.last_success_ms.load(Ordering::Relaxed),
            last_latency_ms: self.last_latency_ms.load(Ordering::Relaxed),
        }
    }
}

/// What to do after one failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    /// Try again after the given delay.
    RetryAfter(Duration),
    /// Stop; the error is terminal.
    Stop,
}

fn is_retryable_status(status: u16) -> bool {
    RETRYABLE_STATUSES.contains(&status) || status == STATUS_OVERLOADED
}

/// Documented retry decision for one attempt.
///
/// Policy: retry only while attempts remain; honor the server `retry-after` hint when present
/// (never shortened to `backoff_max`); otherwise exponential backoff starting at `backoff_initial`,
/// doubling per attempt, capped at `backoff_max`. Timeouts and connection failures are retried
/// the same way. Validation failures and 4xx (except 408/425/429/529) are terminal.
pub fn retry_decision(
    error: &JevError,
    attempt: u32,
    limits: &JevLimits,
) -> RetryDecision {
    if attempt >= limits.max_retries {
        return RetryDecision::Stop;
    }
    let retryable = match error {
        JevError::HttpStatus { status, .. } => is_retryable_status(*status),
        JevError::Timeout { .. } | JevError::Connection { .. } => true,
        JevError::MalformedResponse { .. } => false,
        _ => false,
    };
    if !retryable {
        return RetryDecision::Stop;
    }
    let hint = error.retry_after();
    let delay = match hint {
        Some(hint) if hint >= MAX_RETRY_AFTER => return RetryDecision::Stop,
        Some(hint) => hint,
        None => {
            let factor = 1u32 << attempt.min(16);
            limits.backoff_initial.saturating_mul(factor).min(limits.backoff_max)
        }
    };
    RetryDecision::RetryAfter(delay)
}

/// Largest accepted `retry-after` hint. Anything above this is clamped, so a hostile or broken
/// header value can neither panic the conversion nor stall the caller. A clamped hint is
/// terminal rather than retried early.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(3600);

/// Reads a `retry-after` header in seconds or HTTP-date form.
///
/// Invalid values are ignored. Very large delays are clamped to `MAX_RETRY_AFTER`
/// and cause the retry policy to stop.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    let seconds = match trimmed.parse::<f64>() {
        Ok(seconds) => seconds,
        Err(_) => {
            let date = chrono::DateTime::parse_from_rfc2822(trimmed).ok()?;
            (date.with_timezone(&chrono::Utc) - chrono::Utc::now())
                .to_std().unwrap_or_default().as_secs_f64()
        }
    };
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    // `Duration::from_secs_f64` panics above its representable range, so clamp before converting.
    let bounded = seconds.min(MAX_RETRY_AFTER.as_secs_f64());
    Some(Duration::from_secs_f64(bounded))
}

/// Production HTTP transport. Only this type touches the network.
///
/// The transport owns the credential because `Transport::post` (DESIGN.md 3.1) carries no key
/// parameter. The key is used for exactly one purpose: the `Authorization` header of each
/// request. It never enters a URL, an error, a log line or a counter. `Debug` prints a
/// redaction marker (via `SecretString`).
pub struct JevHttpTransport {
    client: reqwest::Client,
    credential: SecretString,
    limits: JevLimits,
}

impl std::fmt::Debug for JevHttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevHttpTransport")
            .field("endpoint", &sanitize_url(&self.limits.endpoint()))
            .field("credential", &"<redacted>")
            .finish()
    }
}

impl JevHttpTransport {
    /// Builds a transport with an explicit redirect-limited reqwest client and the credential
    /// used only for the `Authorization` header.
    ///
    /// Redirects are refused so the `Authorization` header can never be replayed to another host.
    pub fn new(limits: JevLimits, credential: SecretString) -> Result<Self, JevError> {
        limits.validate()?;
        if credential.expose().trim().is_empty() {
            return Err(JevError::MissingCredential);
        }
        let client = reqwest::Client::builder()
            .connect_timeout(limits.timeout)
            .timeout(limits.timeout)
            .pool_max_idle_per_host(2)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| JevError::config(format!("cannot build http client: {}", error_kind(&error))))?;
        Ok(Self {
            client,
            credential,
            limits,
        })
    }

    /// Fingerprint of the credential in use, for version-change detection.
    pub fn credential_fingerprint(&self) -> String {
        self.credential.key_fingerprint()
    }

    /// Limits in use.
    pub fn limits(&self) -> &JevLimits {
        &self.limits
    }

    /// The endpoint that will be called (sanitized, no query string).
    pub fn endpoint(&self) -> String {
        sanitize_url(&self.limits.endpoint())
    }
}

fn error_kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_request() {
        "request"
    } else if error.is_body() {
        "body"
    } else if error.is_decode() {
        "decode"
    } else {
        "other"
    }
}

impl Transport for JevHttpTransport {
    fn post(
        &self,
        request: &SystemOneRequest,
        timeout: Duration,
    ) -> BoxFuture<Result<SystemOneResponse, JevError>> {
        let client = self.client.clone();
        let credential = self.credential.clone();
        let request = request.clone();
        let limits = self.limits.clone();
        Box::pin(async move {
            let url = limits.endpoint();
            let body = serde_json::to_vec(&request).map_err(|_| JevError::validation("request is not serializable"))?;
            // Safety net: the client already refuses an oversized payload before calling here.
            if body.len() > limits.max_payload_bytes {
                return Err(JevError::PayloadTooLarge {
                    limit: limits.max_payload_bytes,
                    actual: body.len(),
                });
            }
            // The key is placed in the header only; the URL stays unparameterized.
            let response = client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(reqwest::header::ACCEPT, "application/json")
                .bearer_auth(credential.expose())
                .body(body)
                .timeout(timeout)
                .send()
                .await
                .map_err(|error| match error_kind(&error) {
                    "timeout" => JevError::Timeout {
                        detail: sanitize_detail("request exceeded its timeout"),
                    },
                    kind => JevError::Connection {
                        // The raw reqwest error can carry a URL with a query string; never store it.
                        detail: sanitize_detail(&format!("transport failure: {kind}")),
                    },
                })?;

            let status = response.status();
            if !status.is_success() {
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(parse_retry_after);
                // The error body may echo the key; never include it. Status code is enough.
                return Err(JevError::HttpStatus {
                    status: status.as_u16(),
                    detail: sanitize_detail(&format!("systemone returned http {}", status.as_u16())),
                    retry_after,
                });
            }

            let mut response = response;
            let mut buffer: Vec<u8> = Vec::new();
            loop {
                let chunk = response.chunk().await.map_err(|error| {
                    let kind = error_kind(&error);
                    if kind == "timeout" {
                        JevError::Timeout {
                            detail: sanitize_detail("response body exceeded its timeout"),
                        }
                    } else {
                        JevError::Connection {
                            detail: sanitize_detail(&format!("response body read failed: {kind}")),
                        }
                    }
                })?;
                let Some(chunk) = chunk else {
                    break;
                };
                if buffer.len() + chunk.len() > limits.max_response_bytes {
                    return Err(JevError::MalformedResponse {
                        detail: format!(
                            "response body exceeds {} bytes",
                            limits.max_response_bytes
                        ),
                    });
                }
                buffer.extend_from_slice(&chunk);
            }

            let parsed = parse_systemone_body(&buffer)?;
            Ok(parsed)
        })
    }
}

fn classify_kind(error: &serde_json::Error) -> &'static str {
    use serde_json::error::Category;
    match error.classify() {
        Category::Io => "io",
        Category::Syntax => "syntax",
        Category::Data => "data",
        Category::Eof => "eof",
    }
}

/// Parses a raw SystemOne response body with the PRODUCTION parser.
///
/// Used by the HTTP transport and by `MockJevTransport`, so a mock scripted with a raw body
/// exercises the same parsing path a real response would.
pub fn parse_systemone_body(bytes: &[u8]) -> Result<SystemOneResponse, JevError> {
    serde_json::from_slice::<SystemOneResponse>(bytes).map_err(|error| {
        JevError::MalformedResponse {
            detail: sanitize_detail(&format!(
                "response is not a valid systemone payload: {}",
                classify_kind(&error)
            )),
        }
    })
}

/// `SystemOne` implementation for `Compare`.
///
/// The key is held as `SecretString`, so `Debug`/`Display` formatting of this struct cannot
/// leak it. URLs and errors are sanitized. Every failure becomes a skip.
pub struct JevSystemOne {
    transport: Arc<dyn Transport>,
    credential: SecretString,
    mode: JevMode,
    limits: JevLimits,
    stats: Arc<JevStats>,
}

impl std::fmt::Debug for JevSystemOne {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevSystemOne")
            .field("mode", &self.mode)
            .field("endpoint", &sanitize_url(&self.limits.endpoint()))
            .field("credential", &"<redacted>")
            .finish()
    }
}

impl JevSystemOne {
    /// Builds a client for an operative mode: `Compare` (record only) or `Active` (the host may
    /// apply accepted answers through `crate::active`). `Off` is refused before any network object
    /// is created.
    ///
    /// The returned client is observational only. It cannot spawn, delete, pause, resume, steer or
    /// re-route a subagent in any mode, and no configuration flag can grant it those abilities.
    pub fn new(
        mode: JevMode,
        credential: SecretString,
        transport: Arc<dyn Transport>,
        limits: JevLimits,
        stats: Arc<JevStats>,
    ) -> Result<Self, JevError> {
        match mode {
            JevMode::Off => return Err(JevError::ModeOff),
            JevMode::Compare | JevMode::Active | JevMode::CompareAndActive => {}
        }
        if credential.expose().trim().is_empty() {
            return Err(JevError::MissingCredential);
        }
        limits.validate()?;
        Ok(Self {
            transport,
            credential,
            mode,
            limits,
            stats,
        })
    }

    /// Builds a client and its production HTTP transport for an operative mode.
    ///
    /// The mode is checked FIRST, so an `Off` mode does not even construct a network client. The credential goes to the transport (header only) and stays here for
    /// the change-detection fingerprint.
    pub fn with_http(
        mode: JevMode,
        credential: SecretString,
        limits: JevLimits,
        stats: Arc<JevStats>,
    ) -> Result<Self, JevError> {
        match mode {
            JevMode::Off => return Err(JevError::ModeOff),
            JevMode::Compare | JevMode::Active | JevMode::CompareAndActive => {}
        }
        let transport = JevHttpTransport::new(limits.clone(), credential.clone())?;
        Self::new(mode, credential, Arc::new(transport), limits, stats)
    }

    /// Effective mode.
    pub fn effective_mode(&self) -> JevMode {
        self.mode
    }

    /// Limits in use.
    pub fn limits(&self) -> &JevLimits {
        &self.limits
    }

    /// Counters.
    pub fn stats(&self) -> Arc<JevStats> {
        Arc::clone(&self.stats)
    }

    /// Endpoint for status output (sanitized).
    pub fn endpoint(&self) -> String {
        sanitize_url(&self.limits.endpoint())
    }

    /// One-way fingerprint of the credential, for change detection without the secret.
    pub fn credential_fingerprint(&self) -> String {
        self.credential.key_fingerprint()
    }

    /// Model configured for requests.
    pub fn model(&self) -> &'static str {
        DEFAULT_MODEL
    }
}

impl SystemOne for JevSystemOne {
    fn mode(&self) -> JevMode {
        self.mode
    }

    fn decide(&self, bundle: DecisionBundle) -> BoxFuture<DecisionOutcome> {
        let transport = Arc::clone(&self.transport);
        let credential = self.credential.clone();
        let limits = self.limits.clone();
        let stats = Arc::clone(&self.stats);
        let mode = self.mode;
        Box::pin(async move {
            decide_with(&transport, &credential, mode, &limits, &stats, bundle).await
        })
    }
}

/// Runs one Compare evaluation: validate the bundle, send with retries, validate every answer.
///
/// This is the single code path shared by both production and tests (the injected transport is
/// the only difference), so a passing test exercises the production flow.
///
/// `credential` is inspected only for presence: the transport owns the `Authorization` header.
pub async fn decide_with(
    transport: &Arc<dyn Transport>,
    credential: &SecretString,
    mode: JevMode,
    limits: &JevLimits,
    stats: &Arc<JevStats>,
    bundle: DecisionBundle,
) -> DecisionOutcome {
    if mode == JevMode::Off {
        return DecisionOutcome::skipped_all("mode_off");
    }
    if credential.expose().trim().is_empty() {
        return DecisionOutcome::skipped_all("missing_credential");
    }

    // One conversion site, shared with lane B's `DecisionBundle::to_request`.
    let request = bundle.to_request();
    if let Err(error) = validate_request_shape(&request) {
        stats.validation_skips.fetch_add(1, Ordering::Relaxed);
        return DecisionOutcome {
            records: Vec::new(),
            // `JevError::kind()` is a static reason code; no error text reaches the record.
            skips: vec![(String::new(), error.kind())],
            response_model: None,
            usage: Default::default(),
            applied: false,
            attempts: 0,
        };
    }

    // Bounded payload is checked BEFORE any transport call, so an oversized request is never sent.
    match serde_json::to_vec(&request) {
        Ok(body) if body.len() > limits.max_payload_bytes => {
            stats.payload_rejections.fetch_add(1, Ordering::Relaxed);
            return DecisionOutcome {
                records: Vec::new(),
                skips: vec![(String::new(), "payload_too_large")],
                response_model: None,
                usage: Default::default(),
                applied: false,
                attempts: 0,
            };
        }
        Ok(_) => {}
        Err(_) => {
            stats.validation_skips.fetch_add(1, Ordering::Relaxed);
            return DecisionOutcome {
                records: Vec::new(),
                skips: vec![(String::new(), "validation")],
                response_model: None,
                usage: Default::default(),
                applied: false,
                attempts: 0,
            };
        }
    }

    let outcome = send_with_retries(transport, limits, stats, &request).await;
    match outcome {
        Ok((response, _latency_ms, client_attempts)) => {
            let validation = validate_response(&request, &response);
            if !validation.skipped.is_empty() {
                stats
                    .validation_skips
                    .fetch_add(validation.skipped.len() as u64, Ordering::Relaxed);
            }
            let mut records = Vec::new();
            for (question_id, answer) in &validation.accepted {
                let category = bundle
                    .question_categories
                    .get(question_id)
                    .copied()
                    .unwrap_or(DecisionCategory::TaskClassification);
                records.push(DecisionRecord {
                    question_id: question_id.clone(),
                    category,
                    answer: answer.clone(),
                    response_model: validation.response_model.clone(),
                    requested_model: request.model.clone(),
                    applied: false,
                });
            }
            records.sort_by(|left, right| left.question_id.cmp(&right.question_id));
            // Latency is already stored in the stats snapshot for `/jev status`.
            DecisionOutcome {
                records,
                skips: validation.skip_reasons(),
                response_model: validation.response_model.clone(),
                usage: response.usage,
                applied: false,
                attempts: client_attempts,
            }
        }
        Err(attempted) => {
            DecisionOutcome {
                records: Vec::new(),
                skips: vec![(String::new(), attempted.error.kind())],
                response_model: None,
                usage: Default::default(),
                applied: false,
                attempts: attempted.attempts,
            }
        }
    }
}

/// A terminal transport error plus the number of attempts actually made.
/// Retries are client-owned, so the attempt count is part of the result.
struct AttemptedError {
    error: JevError,
    attempts: u32,
}

struct InFlightAttempt(Arc<JevStats>);

impl InFlightAttempt {
    fn new(stats: &Arc<JevStats>) -> Self {
        stats.in_flight.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(stats))
    }
}

impl Drop for InFlightAttempt {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Sends one request with the documented retry policy. Returns the response,
/// latency and attempts, or a terminal error with the attempts made.
async fn send_with_retries(
    transport: &Arc<dyn Transport>,
    limits: &JevLimits,
    stats: &Arc<JevStats>,
    request: &SystemOneRequest,
) -> Result<(SystemOneResponse, u64, u32), AttemptedError> {
    let mut attempt: u32 = 0;
    let started = Instant::now();
    // Total budget for one logical call: the per-attempt deadline times the attempt count.
    let budget = limits.timeout.saturating_mul(limits.max_retries.saturating_add(1));
    loop {
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            stats.failures.fetch_add(1, Ordering::Relaxed);
            return Err(AttemptedError {
                error: JevError::Timeout { detail: "request budget exhausted".to_string() },
                attempts: attempt,
            });
        }
        stats.attempts.fetch_add(1, Ordering::Relaxed);
        let in_flight = InFlightAttempt::new(stats);
        let attempt_started = Instant::now();
        let timeout = limits.timeout.min(remaining);
        let result = tokio::time::timeout(timeout, transport.post(request, timeout))
            .await.unwrap_or_else(|_| Err(JevError::Timeout {
                detail: "request exceeded its timeout".to_string(),
            }));
        drop(in_flight);
        match result {
            Ok(response) => {
                stats.successes.fetch_add(1, Ordering::Relaxed);
                let latency_ms = attempt_started.elapsed().as_millis() as u64;
                stats.last_success_ms.store(utc_now_ms(), Ordering::Relaxed);
                stats.last_latency_ms.store(latency_ms, Ordering::Relaxed);
                return Ok((response, latency_ms, attempt + 1));
            }
            Err(error) => {
                // Counted here (client level) so counters stay meaningful for every transport.
                count_error_class(stats, &error);
                match retry_decision(&error, attempt, limits) {
                    RetryDecision::RetryAfter(delay) => {
                        if started.elapsed().saturating_add(delay) >= budget {
                            stats.failures.fetch_add(1, Ordering::Relaxed);
                            return Err(AttemptedError {
                                error,
                                attempts: attempt + 1,
                            });
                        }
                        stats.retries.fetch_add(1, Ordering::Relaxed);
                        attempt += 1;
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                    }
                    RetryDecision::Stop => {
                        stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(AttemptedError {
                            error,
                            attempts: attempt + 1,
                        });
                    }
                }
            }
        }
    }
}

/// Increments the counter that matches the error class. Never records error text.
fn count_error_class(stats: &Arc<JevStats>, error: &JevError) {
    match error {
        JevError::Timeout { .. } => {
            stats.timeouts.fetch_add(1, Ordering::Relaxed);
        }
        JevError::MalformedResponse { .. } => {
            stats.malformed.fetch_add(1, Ordering::Relaxed);
        }
        JevError::HttpStatus { status, .. } if *status == 429 || *status == STATUS_OVERLOADED => {
            stats.rate_limited.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// `SystemOne` implementation for a client that performs no evaluation and no I/O.
///
/// Holds a mode and nothing else: no transport, no credential, no counters, no runtime, and no
/// child-control capability. Calling `decide` returns a skip without performing any I/O.
#[derive(Debug, Clone, Copy)]
pub struct DisabledSystemOne {
    mode: JevMode,
}

impl Default for DisabledSystemOne {
    fn default() -> Self {
        Self { mode: JevMode::Off }
    }
}

impl DisabledSystemOne {
    /// Disabled instance for a mode that evaluates nothing (Off).
    ///
    /// `Compare` and `Active` map to `Off`: a disabled client must never report that evaluation
    /// is enabled, because the UI and the status line read this mode back.
    pub fn new(mode: JevMode) -> Self {
        Self {
            mode: if matches!(mode, JevMode::Compare | JevMode::Active | JevMode::CompareAndActive) {
                JevMode::Off
            } else {
                mode
            },
        }
    }
}

impl SystemOne for DisabledSystemOne {
    fn mode(&self) -> JevMode {
        self.mode
    }

    fn decide(&self, _bundle: DecisionBundle) -> BoxFuture<DecisionOutcome> {
        // A disabled client only ever reports Off.
        let reason = "mode_off";
        Box::pin(async move { DecisionOutcome::skipped_all(reason) })
    }
}

/// Builds the correct `SystemOne` implementation for a resolved mode, without constructing a
/// client when the mode does not enable Compare.
///
/// This is the only constructor the wiring layer needs. It returns a value that can produce
/// `DecisionRecord`s (data) and nothing else: there is no code path, flag or confidence value
/// that yields a subagent handle, a message sender, a model switch or a budget change. See
/// `types::FORBIDDEN_SUBAGENT_CAPABILITIES`.
pub fn build_system_one(
    mode: JevMode,
    credential: Option<SecretString>,
    limits: JevLimits,
    stats: Arc<JevStats>,
) -> Result<Arc<dyn SystemOne>, JevError> {
    match mode {
        // Both operative modes need a credential; without one the caller gets
        // an explicit refusal rather than a silently disabled client.
        JevMode::Compare | JevMode::Active | JevMode::CompareAndActive => {
            let Some(credential) = credential else {
                return Err(JevError::MissingCredential);
            };
            Ok(Arc::new(JevSystemOne::with_http(mode, credential, limits, stats)?))
        }
        other => Ok(Arc::new(DisabledSystemOne::new(other))),
    }
}

/// Convenience for lane B: turn a `Value` question map into a bundle.
pub fn bundle_with_questions(
    session_id: impl Into<String>,
    turn: u64,
    stage: impl Into<String>,
    state: Value,
    model: impl Into<String>,
    questions: BTreeMap<String, crate::types::QuestionSpec>,
) -> DecisionBundle {
    let question_categories = questions
        .keys()
        .map(|id| {
            let prefix = id.split('.').next().unwrap_or(id);
            let category = DecisionCategory::parse(prefix).unwrap_or(DecisionCategory::TaskClassification);
            (id.clone(), category)
        })
        .collect();
    DecisionBundle {
        session_id: session_id.into(),
        turn,
        stage: stage.into(),
        state,
        model: model.into(),
        questions,
        question_categories,
    }
}

/// Aggregate view over accepted answers for simple callers.
pub fn accepted_answers(outcome: &DecisionOutcome) -> BTreeMap<String, Answer> {
    outcome
        .records
        .iter()
        .map(|record| (record.question_id.clone(), record.answer.clone()))
        .collect()
}

/// Shared slot for the live client, so the UI lane can swap versions without stale results.
///
/// A generation counter makes late results from an older version detectable: a result tagged
/// with an old generation must be discarded rather than applied.
#[derive(Default)]
pub struct ClientHandle {
    inner: Mutex<ClientSlot>,
}

impl std::fmt::Debug for ClientHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // One lock acquisition: `parking_lot::Mutex` is not reentrant, and the client itself is
        // not formattable, so only the generation and a presence flag are reported.
        let slot = self.inner.lock();
        f.debug_struct("ClientHandle")
            .field("generation", &slot.generation)
            .field("installed", &slot.client.is_some())
            .finish()
    }
}

#[derive(Default)]
struct ClientSlot {
    client: Option<Arc<dyn SystemOne>>,
    generation: u64,
}

impl ClientHandle {
    /// Empty handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs a client and bumps the generation. Returns the new generation.
    pub fn install(&self, client: Arc<dyn SystemOne>) -> u64 {
        let mut slot = self.inner.lock();
        slot.client = Some(client);
        slot.generation += 1;
        slot.generation
    }

    /// Removes the client and bumps the generation, so in-flight results are stale.
    pub fn clear(&self) -> u64 {
        let mut slot = self.inner.lock();
        slot.client = None;
        slot.generation += 1;
        slot.generation
    }

    /// Current generation.
    pub fn generation(&self) -> u64 {
        self.inner.lock().generation
    }

    /// Clones the installed client.
    pub fn current(&self) -> Option<Arc<dyn SystemOne>> {
        self.inner.lock().client.clone()
    }

    /// True when a result tagged `generation` is still current.
    pub fn is_current(&self, generation: u64) -> bool {
        self.inner.lock().generation == generation
    }
}
