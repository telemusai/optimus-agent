pub mod diagnostics;
pub mod event_stream;
pub mod hash;
pub mod headers;
pub mod json_parse;
pub mod oauth;
pub mod overflow;
pub mod sanitize_unicode;
pub(crate) mod sse_frames;
pub mod stream_failure;
pub mod typebox_helpers;
pub mod validation;
pub mod typed_json;

/// `Date.now()` - Unix timestamp in milliseconds.
pub fn now_ms() -> i64 {
    crate::utils::diagnostics::now_millis()
}
