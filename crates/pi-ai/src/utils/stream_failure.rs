//! Port of packages/ai/src/utils/stream-failure.ts
//!
//! Shared classification and reporting for provider stream failures, so no
//! provider collapses a specific cause (refusal, safety filter, overload, ...)
//! into a generic string before it is logged and persisted.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::log::get_logger;
use crate::types::AssistantMessage;
use crate::utils::diagnostics::{
    append_assistant_message_diagnostic, now_millis, AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
};

pub type StreamFailureKind = &'static str;

pub const KIND_REFUSAL: StreamFailureKind = "refusal";
pub const KIND_SAFETY: StreamFailureKind = "safety";
pub const KIND_OVERLOADED: StreamFailureKind = "overloaded";
pub const KIND_RATE_LIMIT: StreamFailureKind = "rate_limit";
pub const KIND_SERVER_ERROR: StreamFailureKind = "server_error";
pub const KIND_AUTH: StreamFailureKind = "auth";
pub const KIND_PERMISSION: StreamFailureKind = "permission";
pub const KIND_INVALID_REQUEST: StreamFailureKind = "invalid_request";
pub const KIND_MALFORMED_RESPONSE: StreamFailureKind = "malformed_response";
pub const KIND_REQUEST_INTERRUPTED: StreamFailureKind = "request_interrupted";
pub const KIND_UNKNOWN: StreamFailureKind = "unknown";
/// Locally detected stall of the agent loop's provider stream (no events within
/// the event-gap window, or the overall stream deadline elapsed). Never produced
/// by `classify_stream_failure`: the agent loop records it directly.
pub const KIND_STREAM_STALL: StreamFailureKind = "stream_stall";
/// Locally detected length-capped completion converted into a retryable failure
/// by `PRIME_AGENT_RETRY_TRUNCATED_RESPONSE`. Never produced by
/// `classify_stream_failure`: the agent loop records it directly.
pub const KIND_TRUNCATED_RESPONSE: StreamFailureKind = "truncated_response";

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StreamFailureInfo {
    pub kind: String,
    /// Provider's own error/stop identifier, e.g. "overloaded_error" or "SAFETY".
    #[serde(rename = "providerErrorType", skip_serializing_if = "Option::is_none")]
    pub provider_error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<i64>,
    #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Server-requested wait before retrying (Retry-After header or reset info),
    /// in milliseconds.
    #[serde(rename = "retryAfterMs", skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<f64>,
    /// Truncated raw provider payload for post-mortems.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// `class StreamFailureError extends Error`
#[derive(Debug, Clone, PartialEq)]
pub struct StreamFailureError {
    pub message: String,
    pub name: &'static str,
    pub info: StreamFailureInfo,
}

impl StreamFailureError {
    pub fn new(message: impl Into<String>, info: StreamFailureInfo) -> Self {
        Self {
            message: message.into(),
            name: "StreamFailureError",
            info,
        }
    }
}

impl std::fmt::Display for StreamFailureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for StreamFailureError {}

pub const KIND_MESSAGES: [(&str, &str); 11] = [
    ("refusal", "Model refused to respond"),
    ("safety", "Response blocked by provider safety filters"),
    ("overloaded", "Provider overloaded"),
    ("rate_limit", "Provider rate limit exceeded"),
    ("server_error", "Provider server error"),
    ("auth", "Provider authentication failed"),
    (
        "permission",
        "Provider denied access to the requested resource",
    ),
    ("invalid_request", "Provider rejected the request"),
    (
        "malformed_response",
        "Provider returned a malformed response",
    ),
    ("unknown", "Provider stream failed"),
    ("request_interrupted", "Request interrupted; not automatically replayed"),
];

fn kind_message(kind: &str) -> &'static str {
    KIND_MESSAGES
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, message)| *message)
        .unwrap_or("Provider stream failed")
}

/// Build a user-facing message like
/// "Provider overloaded (overloaded_error, 529) [request_id: req_abc]".
pub fn stream_failure_message(info: &StreamFailureInfo, detail: Option<&str>) -> String {
    let qualifiers = [
        info.provider_error_type.clone(),
        info.status.map(|status| status.to_string()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(", ");
    let mut message = kind_message(&info.kind).to_string();
    if !qualifiers.is_empty() {
        message.push_str(&format!(" ({})", qualifiers));
    }
    if let Some(detail) = detail {
        message.push_str(&format!(": {}", detail));
    }
    if let Some(request_id) = info.request_id.as_ref() {
        message.push_str(&format!(" [request_id: {}]", request_id));
    }
    message
}

fn regexes() -> &'static Vec<(&'static str, Regex)> {
    static REGEXES: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    REGEXES.get_or_init(|| {
        vec![
            (
                "safety",
                Regex::new(r"sensitive|safety|prohibited_content|blocklist|spii|recitation|content.?filter|guardrail|flagged")
                    .expect("safety regex"),
            ),
            (
                "rate_limit",
                Regex::new(r"rate_limit|usage_limit|usage_not_included|throttl").expect("rate limit regex"),
            ),
            ("auth", Regex::new(r"authentication|unauthorized").expect("auth regex")),
            (
                "permission",
                Regex::new(r"permission|forbidden|access.?denied").expect("permission regex"),
            ),
            ("malformed", Regex::new(r"malformed").expect("malformed regex")),
        ]
    })
}

fn matches(pattern: &str, text: &str) -> bool {
    regexes()
        .iter()
        .find(|(name, _)| *name == pattern)
        .map(|(_, regex)| regex.is_match(text))
        .unwrap_or(false)
}

pub fn classify_stream_failure(
    provider_error_type: Option<&str>,
    status: Option<i64>,
) -> StreamFailureKind {
    let type_ = provider_error_type.unwrap_or_default().to_lowercase();
    // Local transport reports this only once sending begins: delivery is uncertain,
    // even if no text has arrived. A new request could duplicate remote work.
    if type_ == "responses_request_interrupted" {
        return KIND_REQUEST_INTERRUPTED;
    }
    if type_ == "refusal" {
        return KIND_REFUSAL;
    }
    if matches("safety", &type_) {
        return KIND_SAFETY;
    }
    if type_.contains("overloaded") || status == Some(529) {
        return KIND_OVERLOADED;
    }
    // usage_not_included is Codex's plan-entitlement rejection, not bad credentials.
    if matches("rate_limit", &type_) || status == Some(429) {
        return KIND_RATE_LIMIT;
    }
    // Permission/403 shapes are entitlement or policy denials, not bad credentials:
    // never auth-stale.
    if matches("auth", &type_) || status == Some(401) {
        return KIND_AUTH;
    }
    if matches("permission", &type_) || status == Some(403) {
        return KIND_PERMISSION;
    }
    if type_.contains("invalid_request")
        || type_.contains("not_found_error")
        || status == Some(400)
        || status == Some(404)
    {
        return KIND_INVALID_REQUEST;
    }
    if type_.contains("malformed") {
        return KIND_MALFORMED_RESPONSE;
    }
    if type_.contains("api_error")
        || type_.contains("server_error")
        || type_.contains("unavailable")
        || status.map(|status| status >= 500).unwrap_or(false)
    {
        return KIND_SERVER_ERROR;
    }
    KIND_UNKNOWN
}

/// Failure for a stream that terminated with a provider stop/finish reason that
/// maps to "error" (e.g. Anthropic "refusal", Gemini "SAFETY").
pub fn stream_failure_from_stop_reason(
    raw_stop_reason: Option<&str>,
    extra_request_id: Option<&str>,
) -> StreamFailureError {
    let mut info = StreamFailureInfo {
        kind: raw_stop_reason
            .map(|raw| classify_stream_failure(Some(raw), None).to_string())
            .unwrap_or_else(|| KIND_UNKNOWN.to_string()),
        provider_error_type: raw_stop_reason.map(str::to_string),
        status: None,
        request_id: extra_request_id.map(str::to_string),
        retry_after_ms: None,
        raw: None,
    };
    if info.kind == KIND_UNKNOWN && matches("malformed", raw_stop_reason.unwrap_or_default()) {
        info.kind = KIND_MALFORMED_RESPONSE.to_string();
    }
    let message = match raw_stop_reason {
        Some(_) => stream_failure_message(&info, None),
        None => {
            stream_failure_message(&info, Some("stream ended with an error and no stop reason"))
        }
    };
    StreamFailureError::new(message, info)
}

pub const MAX_RAW_LENGTH: usize = 2000;

pub fn truncate_raw_payload(raw: &str) -> String {
    if raw.chars().count() > MAX_RAW_LENGTH {
        let truncated: String = raw.chars().take(MAX_RAW_LENGTH).collect();
        format!("{}\u{2026}", truncated)
    } else {
        raw.to_string()
    }
}

/// The Rust counterpart of a thrown provider error: either a `StreamFailureError`
/// or an opaque `Value` describing an SDK-style error object.
pub enum ThrownStreamError<'a> {
    Failure(&'a StreamFailureError),
    Error(&'a (dyn std::error::Error + 'static)),
    Value(&'a Value),
    Message(&'a str),
}

/// `extractStreamFailureParts(error)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamFailureParts {
    pub info: StreamFailureInfo,
    pub detail: Option<String>,
}

pub fn extract_stream_failure_parts(error: &ThrownStreamError<'_>) -> StreamFailureParts {
    match error {
        ThrownStreamError::Failure(failure) => StreamFailureParts {
            info: failure.info.clone(),
            detail: None,
        },
        ThrownStreamError::Value(value) => extract_parts_from_value(value),
        ThrownStreamError::Message(message) => StreamFailureParts {
            info: StreamFailureInfo {
                kind: KIND_UNKNOWN.to_string(),
                ..Default::default()
            },
            detail: Some((*message).to_string()),
        },
        ThrownStreamError::Error(err) => {
            let message = err.to_string();
            let mut kind = classify_stream_failure(Some(&message), None);
            // A plain Rust Error has no provider type or HTTP status. Like a
            // JavaScript Error, its message alone cannot invalidate credentials.
            if kind == KIND_AUTH || kind == KIND_PERMISSION {
                kind = KIND_UNKNOWN;
            }
            let info = StreamFailureInfo {
                kind: kind.to_string(),
                ..Default::default()
            };
            StreamFailureParts { info, detail: None }
        }
    }
}

fn extract_parts_from_value(value: &Value) -> StreamFailureParts {
    let Some(object) = value.as_object() else {
        return StreamFailureParts {
            info: StreamFailureInfo {
                kind: KIND_UNKNOWN.to_string(),
                ..Default::default()
            },
            detail: None,
        };
    };

    let status = object
        .get("status")
        .and_then(Value::as_i64)
        .or_else(|| object.get("statusCode").and_then(Value::as_i64));

    // Error bodies come nested differently per SDK: Anthropic/OpenAI expose
    // `error.error = {type|code, message}` (sometimes doubly nested).
    let mut body = object.get("error");
    if let Some(inner) = body.and_then(|body| body.get("error")) {
        if inner.is_object() {
            body = Some(inner);
        }
    }
    let body_type = body
        .and_then(|body| body.get("type").or_else(|| body.get("code")))
        .and_then(Value::as_str);
    let body_message = body
        .and_then(|body| body.get("message"))
        .and_then(Value::as_str);
    let provider_error_type = body_type
        .map(str::to_string)
        .or_else(|| {
            object
                .get("code")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            object
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| *name != "Error" && *name != "StreamFailureError")
                .map(str::to_string)
        });

    let headers = object.get("headers");
    let header_request_id =
        header_value(headers, "request-id").or_else(|| header_value(headers, "x-request-id"));
    let raw_request_id = object
        .get("requestID")
        .or_else(|| object.get("request_id"))
        .or_else(|| {
            object
                .get("$metadata")
                .and_then(|metadata| metadata.get("requestId"))
        })
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(header_request_id);
    let retry_after_ms = object
        .get("retryAfterMs")
        .and_then(Value::as_f64)
        .filter(|value| *value >= 0.0)
        .or_else(|| parse_retry_after_ms(headers));

    let message = object
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut kind =
        classify_stream_failure(provider_error_type.as_deref().or(Some(message)), status);
    // Message text is too weak for these verdicts: without a structured type, only
    // the status decides.
    if (kind == KIND_AUTH || kind == KIND_PERMISSION) && provider_error_type.is_none() {
        kind = classify_stream_failure(None, status);
    }

    StreamFailureParts {
        info: StreamFailureInfo {
            kind: kind.to_string(),
            provider_error_type,
            status,
            request_id: raw_request_id,
            retry_after_ms,
            raw: None,
        },
        detail: body_message.map(str::to_string),
    }
}

fn header_value(headers: Option<&Value>, name: &str) -> Option<String> {
    let headers = headers?;
    if !headers.is_object() {
        return None;
    }
    // Record-shaped headers must match case-insensitively, like real Headers.
    headers.as_object()?.iter().find_map(|(key, value)| {
        if key.to_lowercase() == name {
            value.as_str().map(str::to_string)
        } else {
            None
        }
    })
}

/// Parse Retry-After / Retry-After-Ms headers into a millisecond wait.
pub fn parse_retry_after_ms(headers: Option<&Value>) -> Option<f64> {
    if let Some(raw_ms) = header_value(headers, "retry-after-ms") {
        if let Ok(ms) = raw_ms.parse::<f64>() {
            if ms.is_finite() && ms >= 0.0 {
                return Some(ms);
            }
        }
    }
    let raw = header_value(headers, "retry-after")?;
    if let Ok(seconds) = raw.parse::<f64>() {
        if seconds.is_finite() && seconds >= 0.0 {
            return Some(seconds * 1000.0);
        }
    }
    let date = parse_http_date_ms(&raw)?;
    Some((date - now_millis() as f64).max(0.0))
}

/// `Date.parse(raw)` for RFC 7231 HTTP dates.
pub fn parse_http_date_ms(raw: &str) -> Option<f64> {
    let formats = [
        "%a, %d %b %Y %H:%M:%S GMT",
        "%A, %d-%b-%y %H:%M:%S GMT",
        "%a %b %e %H:%M:%S %Y",
    ];
    for format in formats {
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(raw.trim(), format) {
            return Some(parsed.and_utc().timestamp_millis() as f64);
        }
    }
    None
}

/// Best-effort extraction of structured failure info from any thrown value.
pub fn extract_stream_failure_info(error: &ThrownStreamError<'_>) -> StreamFailureInfo {
    extract_stream_failure_parts(error).info
}

/// User-facing message for a thrown stream error.
pub fn format_stream_failure_message(error: &ThrownStreamError<'_>) -> String {
    if let ThrownStreamError::Failure(failure) = error {
        return failure.message.clone();
    }
    let parts = extract_stream_failure_parts(error);
    if parts.info.kind == KIND_UNKNOWN {
        return match error {
            ThrownStreamError::Error(err) => err.to_string(),
            ThrownStreamError::Message(message) => (*message).to_string(),
            ThrownStreamError::Value(value) => value.to_string(),
            ThrownStreamError::Failure(failure) => failure.message.clone(),
        };
    }
    stream_failure_message(&parts.info, parts.detail.as_deref())
}

/// The TS `recordStreamFailure` persists `error: extractDiagnosticError(error)`
/// on the `provider_stream_failure` diagnostic (stream-failure.ts:255),
/// which diagnostics.ts:17-30 resolves to `error.name` / `error.message` for an
/// `Error` and to `{name: "ThrownValue", message: formatThrownValue(error)}` for
/// anything else. `ThrownStreamError` is the Rust counterpart of the thrown
/// value, so map it back onto that shape instead of persisting a fake `null`.
fn diagnostic_error_from_thrown(error: &ThrownStreamError<'_>) -> DiagnosticErrorInfo {
    use crate::utils::diagnostics::{extract_diagnostic_error, ThrownValue};
    match error {
        // `class StreamFailureError extends Error`, so diagnostics.ts:18-24 reads
        // `error.name` / `error.message` back off the instance.
        ThrownStreamError::Failure(failure) => DiagnosticErrorInfo {
            name: Some(failure.name.to_string()),
            message: if failure.message.is_empty() {
                failure.name.to_string()
            } else {
                failure.message.clone()
            },
            stack: None,
            code: None,
        },
        ThrownStreamError::Error(err) => extract_diagnostic_error(&ThrownValue::Error(*err)),
        ThrownStreamError::Message(message) => extract_diagnostic_error(&ThrownValue::Text(message)),
        // A JSON-shaped SDK error object is an `Error` instance in the TS, which is
        // why diagnostics.ts:18-24 prefers its own `name`/`message` fields.
        ThrownStreamError::Value(value) => {
            let name = value.get("name").and_then(Value::as_str);
            let message = value
                .get("message")
                .and_then(Value::as_str)
                .filter(|message| !message.is_empty())
                .or(name);
            match (name, message) {
                (None, _) | (_, None) => extract_diagnostic_error(&ThrownValue::Json(value)),
                (Some(name), Some(message)) => DiagnosticErrorInfo {
                    name: Some(name.to_string()),
                    message: message.to_string(),
                    stack: None,
                    code: None,
                },
            }
        }
    }
}

/// Record a terminal stream failure on the message (structured diagnostic that
/// persists to session JSONL) and emit one structured log line. No-op for
/// user-initiated aborts.
pub fn record_stream_failure(
    model: &crate::types::Model,
    output: &mut AssistantMessage,
    error: &ThrownStreamError<'_>,
) {
    if output.stop_reason != "error" {
        return;
    }
    let info = extract_stream_failure_info(error);
    let mut details = Map::new();
    details.insert("kind".to_string(), Value::String(info.kind.clone()));
    if let Some(provider_error_type) = info.provider_error_type.clone() {
        details.insert(
            "providerErrorType".to_string(),
            Value::String(provider_error_type),
        );
    }
    if let Some(status) = info.status {
        details.insert("status".to_string(), Value::Number(status.into()));
    }
    if let Some(request_id) = info.request_id.clone() {
        details.insert("requestId".to_string(), Value::String(request_id));
    }
    if let Some(retry_after_ms) = info.retry_after_ms {
        if let Some(number) = serde_json::Number::from_f64(retry_after_ms) {
            details.insert("retryAfterMs".to_string(), Value::Number(number));
        }
    }

    let diagnostic = AssistantMessageDiagnostic {
        type_: "provider_stream_failure".to_string(),
        timestamp: now_millis(),
        error: Some(diagnostic_error_from_thrown(error)),
        details: Some(details),
    };
    append_assistant_message_diagnostic(output, diagnostic);

    let raw_message = match error {
        ThrownStreamError::Failure(failure) => failure.message.clone(),
        ThrownStreamError::Error(err) => err.to_string(),
        ThrownStreamError::Message(message) => (*message).to_string(),
        ThrownStreamError::Value(value) => value.to_string(),
    };
    let mut fields = Map::new();
    fields.insert(
        "provider".to_string(),
        Value::String(model.provider.clone()),
    );
    fields.insert("model".to_string(), Value::String(model.id.clone()));
    fields.insert("api".to_string(), Value::String(model.api.clone()));
    fields.insert("kind".to_string(), Value::String(info.kind.clone()));
    if let Some(provider_error_type) = info.provider_error_type.clone() {
        fields.insert(
            "providerErrorType".to_string(),
            Value::String(provider_error_type),
        );
    }
    if let Some(status) = info.status {
        fields.insert("status".to_string(), Value::Number(status.into()));
    }
    if let Some(request_id) = info.request_id.clone() {
        fields.insert("requestId".to_string(), Value::String(request_id));
    }
    fields.insert(
        "message".to_string(),
        output
            .error_message
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    // errorMessage is user-facing and concise; keep the raw cause for debugging.
    if Some(raw_message.as_str()) != output.error_message.as_deref() {
        fields.insert(
            "cause".to_string(),
            Value::String(truncate_raw_payload(&raw_message)),
        );
    }
    get_logger("ai.provider").error("provider stream failure", Some(fields));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classification_order_matches_typescript() {
        assert_eq!(classify_stream_failure(Some("responses_request_interrupted"), None), KIND_REQUEST_INTERRUPTED);
        assert_eq!(classify_stream_failure(Some("refusal"), None), KIND_REFUSAL);
        assert_eq!(classify_stream_failure(Some("SAFETY"), None), KIND_SAFETY);
        assert_eq!(
            classify_stream_failure(Some("content_filter"), None),
            KIND_SAFETY
        );
        assert_eq!(
            classify_stream_failure(Some("overloaded_error"), None),
            KIND_OVERLOADED
        );
        assert_eq!(classify_stream_failure(None, Some(529)), KIND_OVERLOADED);
        assert_eq!(
            classify_stream_failure(Some("usage_not_included"), None),
            KIND_RATE_LIMIT
        );
        assert_eq!(classify_stream_failure(None, Some(429)), KIND_RATE_LIMIT);
        assert_eq!(
            classify_stream_failure(Some("authentication_error"), None),
            KIND_AUTH
        );
        assert_eq!(classify_stream_failure(None, Some(401)), KIND_AUTH);
        assert_eq!(
            classify_stream_failure(Some("permission_error"), None),
            KIND_PERMISSION
        );
        assert_eq!(classify_stream_failure(None, Some(403)), KIND_PERMISSION);
        assert_eq!(
            classify_stream_failure(Some("invalid_request_error"), None),
            KIND_INVALID_REQUEST
        );
        assert_eq!(
            classify_stream_failure(None, Some(404)),
            KIND_INVALID_REQUEST
        );
        assert_eq!(
            classify_stream_failure(Some("malformed_json"), None),
            KIND_MALFORMED_RESPONSE
        );
        assert_eq!(
            classify_stream_failure(Some("api_error"), None),
            KIND_SERVER_ERROR
        );
        assert_eq!(classify_stream_failure(None, Some(500)), KIND_SERVER_ERROR);
        assert_eq!(classify_stream_failure(None, None), KIND_UNKNOWN);
    }

    #[test]
    fn message_format_matches_typescript() {
        let info = StreamFailureInfo {
            kind: KIND_OVERLOADED.to_string(),
            provider_error_type: Some("overloaded_error".to_string()),
            status: Some(529),
            request_id: Some("req_abc".to_string()),
            retry_after_ms: None,
            raw: None,
        };
        assert_eq!(
            stream_failure_message(&info, None),
            "Provider overloaded (overloaded_error, 529) [request_id: req_abc]"
        );
        assert_eq!(
            stream_failure_message(&info, Some("busy")),
            "Provider overloaded (overloaded_error, 529): busy [request_id: req_abc]"
        );

        let plain = StreamFailureInfo {
            kind: KIND_UNKNOWN.to_string(),
            ..Default::default()
        };
        assert_eq!(
            stream_failure_message(&plain, None),
            "Provider stream failed"
        );
    }

    #[test]
    fn stop_reason_failure_keeps_raw_reason() {
        let failure = stream_failure_from_stop_reason(Some("refusal"), Some("req_1"));
        assert_eq!(failure.info.kind, KIND_REFUSAL);
        assert_eq!(failure.info.provider_error_type.as_deref(), Some("refusal"));
        assert_eq!(failure.info.request_id.as_deref(), Some("req_1"));
        assert_eq!(
            failure.message,
            "Model refused to respond (refusal) [request_id: req_1]"
        );

        let malformed = stream_failure_from_stop_reason(Some("malformed_json"), None);
        assert_eq!(malformed.info.kind, KIND_MALFORMED_RESPONSE);

        let missing = stream_failure_from_stop_reason(None, None);
        assert_eq!(missing.info.kind, KIND_UNKNOWN);
        assert_eq!(
            missing.message,
            "Provider stream failed: stream ended with an error and no stop reason"
        );
    }

    #[test]
    fn truncate_raw_payload_appends_ellipsis() {
        assert_eq!(truncate_raw_payload("short"), "short");
        let long = "x".repeat(2500);
        let truncated = truncate_raw_payload(&long);
        assert_eq!(truncated.chars().count(), MAX_RAW_LENGTH + 1);
        assert!(truncated.ends_with('\u{2026}'));
    }

    #[test]
    fn extracts_nested_error_bodies_and_headers() {
        let error = json!({
            "name": "APIError",
            "status": 529,
            "headers": {"x-request-id": "req_header"},
            "error": {"error": {"type": "overloaded_error", "message": "Overloaded"}}
        });
        let parts = extract_stream_failure_parts(&ThrownStreamError::Value(&error));
        assert_eq!(parts.info.kind, KIND_OVERLOADED);
        assert_eq!(
            parts.info.provider_error_type.as_deref(),
            Some("overloaded_error")
        );
        assert_eq!(parts.info.status, Some(529));
        assert_eq!(parts.info.request_id.as_deref(), Some("req_header"));
        assert_eq!(parts.detail.as_deref(), Some("Overloaded"));
    }

    #[test]
    fn auth_verdict_requires_a_structured_type() {
        // Message text is too weak: without a structured type the status decides.
        let error = json!({"message": "authentication failed"});
        let parts = extract_stream_failure_parts(&ThrownStreamError::Value(&error));
        assert_eq!(
            parts.info.kind, KIND_UNKNOWN,
            "message text cannot invalidate credentials"
        );
        let plain = std::io::Error::other("authentication failed");
        let parts = extract_stream_failure_parts(&ThrownStreamError::Error(&plain));
        assert_eq!(parts.info.kind, KIND_UNKNOWN);
        assert!(parts.info.provider_error_type.is_none());
        assert_eq!(
            format_stream_failure_message(&ThrownStreamError::Error(&plain)),
            "authentication failed"
        );

        let error = json!({"message": "please authenticate", "status": 400});
        let parts = extract_stream_failure_parts(&ThrownStreamError::Value(&error));
        assert_eq!(parts.info.kind, KIND_INVALID_REQUEST);
    }

    #[test]
    fn retry_after_parsing() {
        let headers = json!({"retry-after-ms": "1500"});
        assert_eq!(parse_retry_after_ms(Some(&headers)), Some(1500.0));
        let headers = json!({"retry-after": "2"});
        assert_eq!(parse_retry_after_ms(Some(&headers)), Some(2000.0));
        let headers = json!({"retry-after": "not-a-date"});
        assert_eq!(parse_retry_after_ms(Some(&headers)), None);
        assert_eq!(parse_retry_after_ms(None), None);
    }

    #[test]
    fn format_stream_failure_message_passes_unknown_errors_through() {
        let error = json!({"message": "weird failure"});
        assert_eq!(
            format_stream_failure_message(&ThrownStreamError::Value(&error)),
            "{\"message\":\"weird failure\"}"
        );
        let failure = StreamFailureError::new(
            "Model refused to respond (refusal)",
            StreamFailureInfo {
                kind: KIND_REFUSAL.to_string(),
                provider_error_type: Some("refusal".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(
            format_stream_failure_message(&ThrownStreamError::Failure(&failure)),
            "Model refused to respond (refusal)"
        );
    }

    #[test]
    fn record_stream_failure_only_for_error_stop_reason() {
        let model = crate::types::Model::new(
            "m",
            "M",
            "openai-responses",
            "openai",
            "https://example.test",
        );
        let mut message = AssistantMessage::default();
        message.stop_reason = "stop".to_string();
        record_stream_failure(&model, &mut message, &ThrownStreamError::Message("boom"));
        assert!(message.diagnostics.is_none());

        message.stop_reason = "error".to_string();
        message.error_message = Some("Provider overloaded".to_string());
        let error = json!({"status": 529, "error": {"type": "overloaded_error"}});
        record_stream_failure(&model, &mut message, &ThrownStreamError::Value(&error));
        let diagnostics = message.diagnostics.expect("diagnostic appended");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].type_, "provider_stream_failure");
        assert_eq!(
            diagnostics[0]
                .details
                .as_ref()
                .unwrap()
                .get("kind")
                .and_then(Value::as_str),
            Some("overloaded")
        );
    }
}
