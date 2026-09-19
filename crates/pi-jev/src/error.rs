//! Error type and redaction helpers for the isolated Jev / TypeSafe client.
//!
//! Rules enforced here (DESIGN.md section 2, JEV_BRIEF.md boundary 6):
//! - No credential ever appears in an error, a log line or a status string.
//! - URLs are sanitized before they are stored: no query string, no fragment, no userinfo.
//! - Detail strings are built only from sanitized input (`sanitize_detail`).

use std::time::Duration;

/// Marker written in place of any value that is withheld.
pub const REDACTED: &str = "<redacted>";

/// Upper bound for any detail string that reaches a log or a status line.
pub const MAX_DETAIL_CHARS: usize = 512;

/// Token runs at or above this length are treated as credential material.
const SECRET_LIKE_MIN_LEN: usize = 24;

/// Header names whose value must never be copied into a log or an error.
const SENSITIVE_HEADERS: [&str; 7] = [
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "api_key",
    "access_token",
    "auth_token",
];

fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+' | '/' | '=' | '.')
}

/// Replaces the value part of an `Authorization`-style header with `<redacted>`.
pub fn redact_authorization(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    for scheme in ["bearer", "basic", "token"] {
        if lower.starts_with(scheme) {
            let rest = &trimmed[scheme.len()..];
            if rest.is_empty() || rest.starts_with([' ', '\t']) {
                // The original casing is kept; only the value is replaced.
                return format!("{} {REDACTED}", &trimmed[..scheme.len()]);
            }
        }
    }
    if trimmed.is_empty() {
        String::new()
    } else {
        REDACTED.to_string()
    }
}

/// Strips the query string, the fragment and any userinfo from a URL.
pub fn sanitize_url(raw: &str) -> String {
    let without_fragment = match raw.find('#') {
        Some(index) => &raw[..index],
        None => raw,
    };
    let without_query = match without_fragment.find('?') {
        Some(index) => &without_fragment[..index],
        None => without_fragment,
    };
    let trimmed = without_query.trim();
    // `scheme://user:pass@host` -> `scheme://host`
    if let Some(scheme_end) = trimmed.find("://") {
        let authority_start = scheme_end + 3;
        if let Some(at) = trimmed[authority_start..].find('@') {
            let mut out = String::with_capacity(trimmed.len());
            out.push_str(&trimmed[..authority_start]);
            out.push_str(&trimmed[authority_start + at + 1..]);
            return out;
        }
    }
    trimmed.to_string()
}

fn matches_at(lower: &[char], index: usize, needle: &str) -> bool {
    let needle_chars: Vec<char> = needle.chars().collect();
    if index + needle_chars.len() > lower.len() {
        return false;
    }
    lower[index..index + needle_chars.len()] == needle_chars[..]
}

/// Redacts `Authorization`-style header values, Bearer tokens and secret-shaped token runs,
/// then truncates the result to `MAX_DETAIL_CHARS`.
///
/// This is defence in depth: the client never formats a credential into an error at all.
pub fn sanitize_detail(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let lower: Vec<char> = chars.iter().map(|c| c.to_ascii_lowercase()).collect();
    let mut out = String::with_capacity(raw.len().min(MAX_DETAIL_CHARS));
    let mut index = 0usize;

    while index < chars.len() {
        // `authorization: <value>` / `authorization=<value>` / `x-api-key: <value>`.
        if let Some(name) = SENSITIVE_HEADERS.iter().find(|name| matches_at(&lower, index, name)) {
            let after_name = index + name.chars().count();
            let mut cursor = after_name;
            while cursor < chars.len() && matches!(chars[cursor], ' ' | '\t') {
                cursor += 1;
            }
            let separator = chars.get(cursor).copied().filter(|c| *c == ':' || *c == '=');
            let value_confirmed = separator.is_some() || cursor > after_name;
            if value_confirmed {
                out.extend(chars[index..after_name].iter());
                if let Some(sep) = separator {
                    out.push(sep);
                    cursor += 1;
                } else {
                    out.push(' ');
                }
                out.push_str(REDACTED);
                // The value runs to the end of the line.
                while cursor < chars.len() && chars[cursor] != '\n' && chars[cursor] != '\r' {
                    cursor += 1;
                }
                index = cursor;
                continue;
            }
        }
        // Standalone `bearer <token>`.
        if matches_at(&lower, index, "bearer") {
            let after = index + "bearer".len();
            if chars.get(after).is_some_and(|c| matches!(c, ' ' | '\t')) {
                out.push_str(&chars[index..after].iter().collect::<String>());
                out.push(' ');
                out.push_str(REDACTED);
                let mut cursor = after;
                while cursor < chars.len() && matches!(chars[cursor], ' ' | '\t') {
                    cursor += 1;
                }
                while cursor < chars.len() && !chars[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                index = cursor;
                continue;
            }
        }
        // A long token run is withheld even when no header name precedes it.
        if is_token_char(chars[index]) {
            let start = index;
            let mut end = index;
            while end < chars.len() && is_token_char(chars[end]) {
                end += 1;
            }
            if end - start >= SECRET_LIKE_MIN_LEN {
                out.push_str(REDACTED);
            } else {
                out.extend(chars[start..end].iter());
            }
            index = end;
            continue;
        }
        out.push(chars[index]);
        index += 1;
    }

    if out.chars().count() > MAX_DETAIL_CHARS {
        let truncated: String = out.chars().take(MAX_DETAIL_CHARS).collect();
        format!("{truncated}...")
    } else {
        out
    }
}

/// Error type for every Jev / TypeSafe client path.
///
/// `detail` fields must already be sanitized (`sanitize_detail`); they never carry a
/// credential, a raw prompt or a full response body.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JevError {
    /// Shadow work was requested while the effective mode does not enable Compare.
    #[error("jev mode off: shadow evaluation is disabled")]
    ModeOff,
    /// No key is configured anywhere. Key presence is never a mode.
    #[error("jev credential missing: no saved credential and no TYPESAFE_API_KEY/JEV_API_KEY")]
    MissingCredential,
    /// The credential store cannot be used; the client fails closed instead of using plaintext.
    #[error("jev credential store unavailable: {reason}")]
    Unavailable { reason: String },
    /// Credential store operation failed.
    #[error("jev credential store error: {detail}")]
    CredentialStore { detail: String },
    /// A key id failed validation.
    #[error("jev invalid credential key id")]
    InvalidKeyId,
    /// Configuration problem (bad endpoint, bad limits).
    #[error("jev config error: {detail}")]
    Config { detail: String },
    /// HTTP status error; `detail` is a sanitized, bounded snippet.
    #[error("jev http status {status}: {detail}")]
    HttpStatus {
        status: u16,
        detail: String,
        retry_after: Option<Duration>,
    },
    /// The per-attempt deadline elapsed.
    #[error("jev request timeout: {detail}")]
    Timeout { detail: String },
    /// The connection could not be established or read.
    #[error("jev connection error: {detail}")]
    Connection { detail: String },
    /// The response body was not a valid SystemOne payload; nothing is fabricated from it.
    #[error("jev malformed response: {detail}")]
    MalformedResponse { detail: String },
    /// Request-shape or answer-validation failure that is not a per-answer skip.
    #[error("jev validation error: {detail}")]
    Validation { detail: String },
    /// Request body is larger than the configured cap; the request is not sent.
    #[error("jev payload too large: {actual} bytes exceeds limit {limit}")]
    PayloadTooLarge { limit: usize, actual: usize },
    /// The caller cancelled the request.
    #[error("jev request cancelled")]
    Cancelled,
    /// A request that would let Jev influence a child agent or the primary model.
    ///
    /// Permanent boundary (DESIGN.md 11/12): no mode, confidence or configuration can enable it.
    #[error("jev is forbidden from controlling subagents: {capability}")]
    SubagentControlForbidden { capability: &'static str },
    /// Any other internal failure.
    #[error("jev internal error: {detail}")]
    Internal { detail: String },
}

impl JevError {
    /// Stable machine-readable kind for logs and metrics. Never contains data.
    pub fn kind(&self) -> &'static str {
        match self {
            JevError::ModeOff => "mode_off",
            JevError::MissingCredential => "missing_credential",
            JevError::Unavailable { .. } => "credential_store_unavailable",
            JevError::CredentialStore { .. } => "credential_store_error",
            JevError::InvalidKeyId => "invalid_key_id",
            JevError::Config { .. } => "config",
            JevError::HttpStatus { .. } => "http_status",
            JevError::Timeout { .. } => "timeout",
            JevError::Connection { .. } => "connection",
            JevError::MalformedResponse { .. } => "malformed_response",
            JevError::Validation { .. } => "validation",
            JevError::PayloadTooLarge { .. } => "payload_too_large",
            JevError::Cancelled => "cancelled",
            JevError::SubagentControlForbidden { .. } => "subagent_control_forbidden",
            JevError::Internal { .. } => "internal",
        }
    }

    /// HTTP status when the failure came from a response.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            JevError::HttpStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Server-provided backoff hint, when the response carried one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            JevError::HttpStatus { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// True when the failure is a transport failure (recorded as a skip).
    pub fn is_transport_failure(&self) -> bool {
        matches!(
            self,
            JevError::HttpStatus { .. }
                | JevError::Timeout { .. }
                | JevError::Connection { .. }
                | JevError::MalformedResponse { .. }
        )
    }

    /// True when the client must not touch the network (Off semantics).
    pub fn is_mode_refusal(&self) -> bool {
        matches!(
            self,
            JevError::ModeOff | JevError::SubagentControlForbidden { .. }
        )
    }

    /// True when the refusal is the permanent no-subagent-control boundary.
    pub fn is_subagent_control_refusal(&self) -> bool {
        matches!(self, JevError::SubagentControlForbidden { .. })
    }

    /// A sanitized, bounded one-line description for logs and status.
    pub fn log_line(&self) -> String {
        sanitize_detail(&self.to_string())
    }

    /// Constructs a validation error with a sanitized detail.
    pub fn validation(detail: impl AsRef<str>) -> Self {
        JevError::Validation {
            detail: sanitize_detail(detail.as_ref()),
        }
    }

    /// Constructs a config error with a sanitized detail.
    pub fn config(detail: impl AsRef<str>) -> Self {
        JevError::Config {
            detail: sanitize_detail(detail.as_ref()),
        }
    }

    /// Constructs a malformed-response error with a sanitized detail.
    pub fn malformed(detail: impl AsRef<str>) -> Self {
        JevError::MalformedResponse {
            detail: sanitize_detail(detail.as_ref()),
        }
    }
}
