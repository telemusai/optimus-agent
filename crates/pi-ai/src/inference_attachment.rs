//! Process-local, request-bound attachment of opaque native inference state.
//!
//! The receipt is a runtime-use assertion, not semantic acknowledgment. A server
//! must issue it only after its native runtime has proved use of this exact state.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub const INFERENCE_RECEIPT_HEADER: &str = "x-optimus-occ3-receipt";
pub const MAX_INFERENCE_RECEIPT_BYTES: usize = 2048;

/// Keepalive stream extension (opt-in). A client that cannot tolerate the
/// fully-buffered armed-turn contract sets the request header below; a server
/// that accepts the offer answers with the deferred-receipt marker and moves
/// the receipt from the response headers into a terminal SSE event.
pub const OCC3_KEEPALIVE_REQUEST_HEADER: &str = "x-optimus-occ3-keepalive";
pub const OCC3_KEEPALIVE_REQUEST_MODE: &str = "keepalive-v1";
pub const OCC3_DEFERRED_RECEIPT_HEADER: &str = "x-optimus-occ3-deferred-receipt";
pub const OCC3_DEFERRED_RECEIPT_MODE: &str = "deferred-v1";
/// Envelope schema of the terminal receipt event (`occ3.terminal.v1`).
pub const OCC3_TERMINAL_SCHEMA: &str = "occ3.terminal.v1";
pub const OCC3_TERMINAL_KIND_RECEIVER: &str = "receiver";

/// Exact dispatch target. This is local metadata, never part of `occ3` on the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceStateTarget {
    pub provider: String,
    pub model_id: String,
    pub base_url: String,
}

/// Bounded identifiers only: no source text, tensors, or authentication keys.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceAttachment {
    pub protocol_version: u32,
    pub request_id: String,
    pub message_id: String,
    pub connector_id: String,
    pub receiver_session_id: String,
    pub model_fingerprint: String,
    pub runtime_id: String,
    pub runtime_generation: u64,
    pub state_handle: String,
    pub expires_at_epoch: u64,
    pub method: String,
}

impl std::fmt::Debug for InferenceAttachment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceAttachment")
            .field("protocol_version", &self.protocol_version)
            .finish_non_exhaustive()
    }
}

/// All clones share this lifecycle. Only `Prepared` can be submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceStateLeaseState {
    Prepared,
    Submitted,
    Consumed,
    Uncertain,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InferenceAttachmentError {
    #[error("Inference attachment has an invalid {0} field.")]
    InvalidField(&'static str),
    #[error("Inference attachment protocol version is unsupported.")]
    UnsupportedVersion,
    #[error("Inference attachment target must be HTTPS or loopback HTTP without credentials, query, or fragment.")]
    InvalidEndpoint,
    #[error("Inference attachment has expired.")]
    Expired,
    #[error("Inference attachment clock is unavailable.")]
    ClockUnavailable,
    #[error("Inference attachment target {0} does not match this request.")]
    TargetMismatch(&'static str),
    #[error("Inference attachment is no longer prepared; replay is forbidden.")]
    NotPrepared,
    #[error("Inference attachment submission is no longer active.")]
    NotSubmitted,
    #[error(
        "Inference attachment requires an unchanged target model and a streaming object payload."
    )]
    InvalidPayload,
    #[error("The occ3 payload field is reserved for a typed inference attachment.")]
    ReservedPayloadField,
    #[error("Inference attachment cannot fit the receipt header limit.")]
    ReceiptTooLarge,
    #[error("Inference attachment receipt is missing.")]
    MissingReceipt,
    #[error("Inference attachment receipt is invalid or exceeds its size limit.")]
    InvalidReceipt,
    #[error("Inference attachment receipt does not prove use of the exact attached state.")]
    ReceiptMismatch,
}

struct LeaseInner {
    target: InferenceStateTarget,
    attachment: InferenceAttachment,
    state: Mutex<InferenceStateLeaseState>,
    cancellation: CancellationToken,
}

/// A capability issued by a connector for one actual inference request.
///
/// This type deliberately has no serde implementation. Cloning preserves the
/// one-shot fence; reconstructing a lease from a past request is not a retry.
#[derive(Clone)]
pub struct InferenceStateLease(Arc<LeaseInner>);

impl std::fmt::Debug for InferenceStateLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceStateLease")
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl InferenceStateLease {
    pub fn new(
        target: InferenceStateTarget,
        attachment: InferenceAttachment,
    ) -> Result<Self, InferenceAttachmentError> {
        Self::new_at(target, attachment, epoch_now()?)
    }

    fn new_at(
        target: InferenceStateTarget,
        attachment: InferenceAttachment,
        now: u64,
    ) -> Result<Self, InferenceAttachmentError> {
        validate_identifier("provider", &target.provider)?;
        validate_identifier("model_id", &target.model_id)?;
        validate_endpoint(&target.base_url)?;
        if attachment.protocol_version != 1 {
            return Err(InferenceAttachmentError::UnsupportedVersion);
        }
        for (field, value) in [
            ("request_id", &attachment.request_id),
            ("message_id", &attachment.message_id),
            ("connector_id", &attachment.connector_id),
            ("receiver_session_id", &attachment.receiver_session_id),
            ("model_fingerprint", &attachment.model_fingerprint),
            ("runtime_id", &attachment.runtime_id),
            ("state_handle", &attachment.state_handle),
            ("method", &attachment.method),
        ] {
            validate_identifier(field, value)?;
        }
        if attachment.expires_at_epoch <= now {
            return Err(InferenceAttachmentError::Expired);
        }
        validate_receipt_size(&attachment)?;
        Ok(Self(Arc::new(LeaseInner {
            target,
            attachment,
            state: Mutex::new(InferenceStateLeaseState::Prepared),
            cancellation: CancellationToken::new(),
        })))
    }

    pub fn target(&self) -> &InferenceStateTarget {
        &self.0.target
    }

    pub fn attachment(&self) -> &InferenceAttachment {
        &self.0.attachment
    }

    pub fn state(&self) -> InferenceStateLeaseState {
        *self.0.state.lock()
    }

    /// Close this lease without rearming it. Cancellation after submission is uncertain.
    pub fn cancel(&self) -> InferenceStateLeaseState {
        let mut state = self.0.state.lock();
        *state = match *state {
            InferenceStateLeaseState::Prepared => InferenceStateLeaseState::Cancelled,
            InferenceStateLeaseState::Submitted => InferenceStateLeaseState::Uncertain,
            terminal => terminal,
        };
        if *state != InferenceStateLeaseState::Consumed {
            self.0.cancellation.cancel();
        }
        *state
    }

    pub(crate) async fn cancelled(&self) {
        self.0.cancellation.cancelled().await;
    }

    pub(crate) fn validate_target(
        &self,
        provider: &str,
        model_id: &str,
        base_url: &str,
    ) -> Result<(), InferenceAttachmentError> {
        self.validate_target_at(provider, model_id, base_url, epoch_now()?)
    }

    fn validate_target_at(
        &self,
        provider: &str,
        model_id: &str,
        base_url: &str,
        now: u64,
    ) -> Result<(), InferenceAttachmentError> {
        if self.state() != InferenceStateLeaseState::Prepared {
            return Err(InferenceAttachmentError::NotPrepared);
        }
        for (field, actual, expected) in [
            ("provider", provider, self.0.target.provider.as_str()),
            ("model_id", model_id, self.0.target.model_id.as_str()),
            ("base_url", base_url, self.0.target.base_url.as_str()),
        ] {
            if actual != expected {
                return Err(InferenceAttachmentError::TargetMismatch(field));
            }
        }
        if self.0.attachment.expires_at_epoch <= now {
            return Err(InferenceAttachmentError::Expired);
        }
        Ok(())
    }

    /// Call only after the exact request has been built, immediately before HTTP send.
    pub(crate) fn submit(&self) -> Result<InferenceSubmission, InferenceAttachmentError> {
        let mut state = self.0.state.lock();
        if *state != InferenceStateLeaseState::Prepared {
            return Err(InferenceAttachmentError::NotPrepared);
        }
        if self.0.attachment.expires_at_epoch <= epoch_now()? {
            return Err(InferenceAttachmentError::Expired);
        }
        validate_receipt_size(&self.0.attachment)?;
        *state = InferenceStateLeaseState::Submitted;
        Ok(InferenceSubmission {
            lease: self.clone(),
            receipt_verified: false,
            deferred: false,
        })
    }
}

#[derive(Debug)]
pub(crate) struct InferenceSubmission {
    lease: InferenceStateLease,
    receipt_verified: bool,
    /// The server accepted the keepalive offer: the receipt arrives as the
    /// terminal `optimus.occ3.receipt` SSE event instead of response headers.
    deferred: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceReceipt {
    attachment: InferenceAttachment,
    consumed: bool,
    fallback_used: bool,
    injected_state: bool,
}

impl InferenceSubmission {
    pub(crate) fn ensure_active(&self) -> Result<(), InferenceAttachmentError> {
        if self.lease.state() != InferenceStateLeaseState::Submitted {
            return Err(InferenceAttachmentError::NotSubmitted);
        }
        Ok(())
    }

    /// No caller may expose response content before this succeeds.
    pub(crate) fn verify_receipt(
        &mut self,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<(), InferenceAttachmentError> {
        self.ensure_active()?;
        let mut values = headers.get_all(INFERENCE_RECEIPT_HEADER).iter();
        let value = values
            .next()
            .ok_or(InferenceAttachmentError::MissingReceipt)?;
        if values.next().is_some() || value.as_bytes().len() > MAX_INFERENCE_RECEIPT_BYTES {
            return Err(InferenceAttachmentError::InvalidReceipt);
        }
        let receipt: InferenceReceipt = serde_json::from_slice(value.as_bytes())
            .map_err(|_| InferenceAttachmentError::InvalidReceipt)?;
        if receipt.attachment != *self.lease.attachment()
            || !receipt.consumed
            || receipt.fallback_used
            || !receipt.injected_state
        {
            return Err(InferenceAttachmentError::ReceiptMismatch);
        }
        self.receipt_verified = true;
        Ok(())
    }

    /// Keepalive offer handling at response headers. `Ok(false)` means the
    /// server kept the buffered contract, so the caller must fall back to
    /// `verify_receipt`; `Ok(true)` defers the receipt to the terminal event.
    pub(crate) fn defer_receipt(
        &mut self,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<bool, InferenceAttachmentError> {
        self.ensure_active()?;
        let mut values = headers.get_all(OCC3_DEFERRED_RECEIPT_HEADER).iter();
        match values.next() {
            None => Ok(false),
            Some(value) => {
                if values.next().is_some() || value.as_bytes() != OCC3_DEFERRED_RECEIPT_MODE.as_bytes() {
                    return Err(InferenceAttachmentError::InvalidReceipt);
                }
                self.deferred = true;
                Ok(true)
            }
        }
    }

    /// Whether this submission waits for the terminal receipt event.
    pub(crate) fn is_deferred(&self) -> bool {
        self.deferred
    }

    /// Verify the receipt carried by the terminal SSE envelope. The invariants
    /// are exactly `verify_receipt`'s; only the transport differs.
    pub(crate) fn verify_receipt_value(
        &mut self,
        envelope: &serde_json::Value,
    ) -> Result<(), InferenceAttachmentError> {
        self.ensure_active()?;
        if !self.deferred
            || envelope.get("schema").and_then(serde_json::Value::as_str) != Some(OCC3_TERMINAL_SCHEMA)
            || envelope.get("kind").and_then(serde_json::Value::as_str) != Some(OCC3_TERMINAL_KIND_RECEIVER)
        {
            return Err(InferenceAttachmentError::InvalidReceipt);
        }
        let receipt_value = envelope
            .get("receipt")
            .cloned()
            .ok_or(InferenceAttachmentError::InvalidReceipt)?;
        if serde_json::to_vec(&receipt_value).map(|bytes| bytes.len()).unwrap_or(usize::MAX)
            > MAX_INFERENCE_RECEIPT_BYTES
        {
            return Err(InferenceAttachmentError::InvalidReceipt);
        }
        let receipt: InferenceReceipt = serde_json::from_value(receipt_value)
            .map_err(|_| InferenceAttachmentError::InvalidReceipt)?;
        let response_id = envelope
            .get("response_id")
            .and_then(serde_json::Value::as_str)
            .ok_or(InferenceAttachmentError::InvalidReceipt)?;
        if response_id.is_empty() || response_id.len() > 256 {
            return Err(InferenceAttachmentError::InvalidReceipt);
        }
        if receipt.attachment != *self.lease.attachment()
            || !receipt.consumed
            || receipt.fallback_used
            || !receipt.injected_state
        {
            return Err(InferenceAttachmentError::ReceiptMismatch);
        }
        self.receipt_verified = true;
        Ok(())
    }

    /// Only a successful terminal stream plus its exact receipt consumes the lease.
    pub(crate) fn complete(self) -> Result<(), InferenceAttachmentError> {
        if !self.receipt_verified {
            return Err(InferenceAttachmentError::MissingReceipt);
        }
        let mut state = self.lease.0.state.lock();
        if *state != InferenceStateLeaseState::Submitted {
            return Err(InferenceAttachmentError::NotSubmitted);
        }
        *state = InferenceStateLeaseState::Consumed;
        Ok(())
    }
}

impl Drop for InferenceSubmission {
    fn drop(&mut self) {
        let mut state = self.lease.0.state.lock();
        if *state == InferenceStateLeaseState::Submitted {
            *state = InferenceStateLeaseState::Uncertain;
        }
    }
}

fn validate_receipt_size(attachment: &InferenceAttachment) -> Result<(), InferenceAttachmentError> {
    let value = serde_json::json!({
        "attachment": attachment,
        "consumed": true,
        "fallback_used": false,
        "injected_state": true,
    });
    let size = serde_json::to_vec(&value)
        .map_err(|_| InferenceAttachmentError::InvalidReceipt)?
        .len();
    if size > MAX_INFERENCE_RECEIPT_BYTES {
        return Err(InferenceAttachmentError::ReceiptTooLarge);
    }
    Ok(())
}

fn epoch_now() -> Result<u64, InferenceAttachmentError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| InferenceAttachmentError::ClockUnavailable)
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), InferenceAttachmentError> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-/".contains(&b))
    {
        return Err(InferenceAttachmentError::InvalidField(field));
    }
    Ok(())
}

fn validate_endpoint(value: &str) -> Result<(), InferenceAttachmentError> {
    if value.len() > 2048
        || value
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err(InferenceAttachmentError::InvalidEndpoint);
    }
    let url = url::Url::parse(value).map_err(|_| InferenceAttachmentError::InvalidEndpoint)?;
    let loopback = match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(host)) => host == "localhost",
        None => false,
    };
    if !(url.scheme() == "https" || (url.scheme() == "http" && loopback))
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(InferenceAttachmentError::InvalidEndpoint);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn target() -> InferenceStateTarget {
        InferenceStateTarget {
            provider: "fixture-provider".into(),
            model_id: "fixture/model".into(),
            base_url: "http://127.0.0.1:1234/v1".into(),
        }
    }

    fn attachment() -> InferenceAttachment {
        InferenceAttachment {
            protocol_version: 1,
            request_id: "request-1".into(),
            message_id: "message-1".into(),
            connector_id: "fixture-connector".into(),
            receiver_session_id: "receiver-1".into(),
            model_fingerprint: "sha256:fixture-model".into(),
            runtime_id: "runtime-1".into(),
            runtime_generation: 1,
            state_handle: "opaque-state-handle".into(),
            expires_at_epoch: epoch_now().unwrap() + 300,
            method: "native-state-v1".into(),
        }
    }

    fn receipt(attachment: &InferenceAttachment) -> serde_json::Value {
        json!({"attachment":attachment, "consumed":true, "fallback_used":false, "injected_state":true})
    }

    fn headers(value: &serde_json::Value) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(INFERENCE_RECEIPT_HEADER, value.to_string().parse().unwrap());
        headers
    }

    #[test]
    fn validates_identifiers_endpoint_version_and_constructor_expiry() {
        let mut state = attachment();
        state.protocol_version = 2;
        assert_eq!(
            InferenceStateLease::new(target(), state).unwrap_err(),
            InferenceAttachmentError::UnsupportedVersion
        );
        let mut state = attachment();
        state.expires_at_epoch = epoch_now().unwrap();
        assert_eq!(
            InferenceStateLease::new(target(), state).unwrap_err(),
            InferenceAttachmentError::Expired
        );
        for value in ["", "source prompt with spaces", "a\r\nb", &"a".repeat(257)] {
            let mut state = attachment();
            state.state_handle = value.into();
            assert_eq!(
                InferenceStateLease::new(target(), state).unwrap_err(),
                InferenceAttachmentError::InvalidField("state_handle")
            );
        }
        for endpoint in [
            "http://remote.invalid/v1",
            "file:///tmp/x",
            "https://user:key@example.invalid/v1",
            "https://example.invalid/v1?key=secret",
            "https://example.invalid/v1#x",
            "not-a-url",
            "https://example.invalid/\npath",
        ] {
            let mut binding = target();
            binding.base_url = endpoint.into();
            assert_eq!(
                InferenceStateLease::new(binding, attachment()).unwrap_err(),
                InferenceAttachmentError::InvalidEndpoint
            );
        }
        for endpoint in [
            "https://example.invalid/v1",
            "http://[::1]:1234/v1",
            "http://localhost:1234/v1",
        ] {
            let mut binding = target();
            binding.base_url = endpoint.into();
            assert!(InferenceStateLease::new(binding, attachment()).is_ok());
        }
    }

    #[test]
    fn combined_receipt_size_is_refused_before_submission() {
        let mut state = attachment();
        for field in [
            &mut state.request_id,
            &mut state.message_id,
            &mut state.connector_id,
            &mut state.receiver_session_id,
            &mut state.model_fingerprint,
            &mut state.runtime_id,
            &mut state.state_handle,
            &mut state.method,
        ] {
            *field = "a".repeat(256);
        }
        assert!(receipt(&state).to_string().len() > MAX_INFERENCE_RECEIPT_BYTES);
        assert_eq!(
            InferenceStateLease::new(target(), state).unwrap_err(),
            InferenceAttachmentError::ReceiptTooLarge
        );
        assert!(receipt(&attachment()).to_string().len() < MAX_INFERENCE_RECEIPT_BYTES);
    }

    #[test]
    fn exact_target_and_dispatch_expiry_fail_while_prepared() {
        let binding = target();
        let lease = InferenceStateLease::new(binding.clone(), attachment()).unwrap();
        for (provider, model, endpoint, field) in [
            (
                "wrong",
                binding.model_id.as_str(),
                binding.base_url.as_str(),
                "provider",
            ),
            (
                binding.provider.as_str(),
                "wrong",
                binding.base_url.as_str(),
                "model_id",
            ),
            (
                binding.provider.as_str(),
                binding.model_id.as_str(),
                "http://127.0.0.1:1235/v1",
                "base_url",
            ),
            (
                binding.provider.as_str(),
                binding.model_id.as_str(),
                "http://127.0.0.1:1234/v1/",
                "base_url",
            ),
        ] {
            assert_eq!(
                lease.validate_target(provider, model, endpoint),
                Err(InferenceAttachmentError::TargetMismatch(field))
            );
            assert_eq!(lease.state(), InferenceStateLeaseState::Prepared);
        }
        assert_eq!(
            lease.validate_target_at(
                &binding.provider,
                &binding.model_id,
                &binding.base_url,
                lease.attachment().expires_at_epoch
            ),
            Err(InferenceAttachmentError::Expired)
        );
        // Constructor time and actual submit time are independently checked.
        let mut state = attachment();
        state.expires_at_epoch = 2;
        let expired = InferenceStateLease::new_at(binding, state, 1).unwrap();
        assert_eq!(
            expired.submit().unwrap_err(),
            InferenceAttachmentError::Expired
        );
        assert_eq!(expired.state(), InferenceStateLeaseState::Prepared);
    }

    #[test]
    fn clones_share_one_submission_and_only_complete_consumes() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let clone = lease.clone();
        let mut submission = clone.submit().unwrap();
        assert_eq!(lease.state(), InferenceStateLeaseState::Submitted);
        assert_eq!(
            lease.submit().unwrap_err(),
            InferenceAttachmentError::NotPrepared
        );
        submission
            .verify_receipt(&headers(&receipt(lease.attachment())))
            .unwrap();
        assert_eq!(lease.state(), InferenceStateLeaseState::Submitted);
        submission.complete().unwrap();
        assert_eq!(lease.state(), InferenceStateLeaseState::Consumed);
        assert_eq!(
            clone.submit().unwrap_err(),
            InferenceAttachmentError::NotPrepared
        );
        assert_eq!(clone.cancel(), InferenceStateLeaseState::Consumed);
    }

    #[test]
    fn concurrent_clones_have_exactly_one_winner() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let lease = lease.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    lease.submit().is_ok()
                })
            })
            .collect();
        assert_eq!(
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
    }

    #[test]
    fn drop_and_cancel_do_not_rearm_or_claim_consumption() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        assert_eq!(lease.cancel(), InferenceStateLeaseState::Cancelled);
        assert_eq!(lease.cancel(), InferenceStateLeaseState::Cancelled);
        assert_eq!(
            lease.submit().unwrap_err(),
            InferenceAttachmentError::NotPrepared
        );
        for verify in [false, true] {
            let lease = InferenceStateLease::new(target(), attachment()).unwrap();
            let mut submission = lease.submit().unwrap();
            if verify {
                submission
                    .verify_receipt(&headers(&receipt(lease.attachment())))
                    .unwrap();
            }
            drop(submission);
            assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
        }
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let submission = lease.submit().unwrap();
        assert_eq!(lease.cancel(), InferenceStateLeaseState::Uncertain);
        assert_eq!(
            submission.complete(),
            Err(InferenceAttachmentError::MissingReceipt)
        );
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
    }

    #[test]
    fn receipt_requires_every_exact_binding_and_all_three_assertions() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut submission = lease.submit().unwrap();
        let valid = receipt(lease.attachment());
        for field in [
            "request_id",
            "message_id",
            "connector_id",
            "receiver_session_id",
            "model_fingerprint",
            "runtime_id",
            "state_handle",
            "method",
            "runtime_generation",
            "expires_at_epoch",
            "protocol_version",
        ] {
            let mut wrong = valid.clone();
            wrong["attachment"][field] = if wrong["attachment"][field].is_string() {
                json!("other")
            } else {
                json!(0)
            };
            assert_eq!(
                submission.verify_receipt(&headers(&wrong)),
                Err(InferenceAttachmentError::ReceiptMismatch),
                "{field}"
            );
        }
        for field in ["consumed", "fallback_used", "injected_state"] {
            let mut wrong = valid.clone();
            wrong[field] = json!(field == "fallback_used");
            assert_eq!(
                submission.verify_receipt(&headers(&wrong)),
                Err(InferenceAttachmentError::ReceiptMismatch),
                "{field}"
            );
        }
        assert_eq!(lease.state(), InferenceStateLeaseState::Submitted);
        drop(submission);
        assert_eq!(lease.state(), InferenceStateLeaseState::Uncertain);
    }

    #[test]
    fn receipt_rejects_missing_oversized_duplicate_and_unknown_fields() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut submission = lease.submit().unwrap();
        assert_eq!(
            submission.verify_receipt(&reqwest::header::HeaderMap::new()),
            Err(InferenceAttachmentError::MissingReceipt)
        );
        let valid = receipt(lease.attachment());
        let mut duplicate = headers(&valid);
        duplicate.append(INFERENCE_RECEIPT_HEADER, valid.to_string().parse().unwrap());
        assert_eq!(
            submission.verify_receipt(&duplicate),
            Err(InferenceAttachmentError::InvalidReceipt)
        );
        let mut oversized = reqwest::header::HeaderMap::new();
        oversized.insert(
            INFERENCE_RECEIPT_HEADER,
            "a".repeat(MAX_INFERENCE_RECEIPT_BYTES + 1).parse().unwrap(),
        );
        assert_eq!(
            submission.verify_receipt(&oversized),
            Err(InferenceAttachmentError::InvalidReceipt)
        );
        let mut outer = valid.clone();
        outer["source_text"] = json!("forbidden");
        let mut inner = valid.clone();
        inner["attachment"]["unexpected"] = json!(true);
        let mut missing = valid.clone();
        missing.as_object_mut().unwrap().remove("consumed");
        let duplicate_json = format!("{{\"consumed\":true,{}", &valid.to_string()[1..]);
        for invalid in [
            outer.to_string(),
            inner.to_string(),
            missing.to_string(),
            duplicate_json,
            "{".into(),
        ] {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(INFERENCE_RECEIPT_HEADER, invalid.parse().unwrap());
            assert_eq!(
                submission.verify_receipt(&headers),
                Err(InferenceAttachmentError::InvalidReceipt)
            );
        }
    }

    #[test]
    fn options_serde_never_serializes_or_reconstructs_a_lease() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut options = crate::types::StreamOptions::default();
        let normal = serde_json::to_value(&options).unwrap();
        options.inference_state = Some(lease.clone());
        assert_eq!(serde_json::to_value(&options).unwrap(), normal);
        assert_eq!(normal, json!({}));
        let restored: crate::types::StreamOptions =
            serde_json::from_value(json!({"inferenceState":receipt(lease.attachment())})).unwrap();
        assert!(restored.inference_state.is_none());
        assert!(!format!("{lease:?}").contains(&lease.attachment().state_handle));
        assert!(!format!("{:?}", lease.attachment()).contains(&lease.attachment().state_handle));
    }

    #[test]
    fn deferred_receipt_transport_verifies_and_completes() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut submission = lease.submit().unwrap();
        assert!(!submission.is_deferred());
        // No marker: the server kept the buffered contract, so the caller
        // falls back to `verify_receipt`.
        assert!(!submission.defer_receipt(&reqwest::header::HeaderMap::new()).unwrap());
        let mut marker = reqwest::header::HeaderMap::new();
        marker.insert(
            OCC3_DEFERRED_RECEIPT_HEADER,
            OCC3_DEFERRED_RECEIPT_MODE.parse().unwrap(),
        );
        assert!(submission.defer_receipt(&marker).unwrap());
        assert!(submission.is_deferred());
        let envelope = json!({
            "schema": OCC3_TERMINAL_SCHEMA,
            "kind": OCC3_TERMINAL_KIND_RECEIVER,
            "response_id": "resp-1",
            "receipt": receipt(&attachment()),
        });
        submission.verify_receipt_value(&envelope).unwrap();
        submission.complete().unwrap();
    }

    #[test]
    fn deferred_receipt_rejects_wrong_kinds_schemas_and_unguarded_use() {
        let lease = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut submission = lease.submit().unwrap();
        let envelope = json!({
            "schema": OCC3_TERMINAL_SCHEMA,
            "kind": OCC3_TERMINAL_KIND_RECEIVER,
            "response_id": "resp-1",
            "receipt": receipt(&attachment()),
        });
        // Without the defer marker the envelope path is refused.
        assert_eq!(
            submission.verify_receipt_value(&envelope).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        let mut marker = reqwest::header::HeaderMap::new();
        marker.insert(
            OCC3_DEFERRED_RECEIPT_HEADER,
            OCC3_DEFERRED_RECEIPT_MODE.parse().unwrap(),
        );
        assert!(submission.defer_receipt(&marker).unwrap());
        // Wrong kind, wrong schema and an oversized response id are refused.
        let mut wrong_kind = envelope.clone();
        wrong_kind["kind"] = json!("source");
        assert_eq!(
            submission.verify_receipt_value(&wrong_kind).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        let mut wrong_schema = envelope.clone();
        wrong_schema["schema"] = json!("occ3.other.v1");
        assert_eq!(
            submission.verify_receipt_value(&wrong_schema).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        let mut long_id = envelope.clone();
        long_id["response_id"] = json!("a".repeat(257));
        assert_eq!(
            submission.verify_receipt_value(&long_id).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        // A receipt that did not consume the state is a mismatch.
        let mut not_consumed = envelope.clone();
        not_consumed["receipt"]["consumed"] = json!(false);
        assert_eq!(
            submission.verify_receipt_value(&not_consumed).unwrap_err(),
            InferenceAttachmentError::ReceiptMismatch
        );
        // Duplicated or wrong-valued markers are refused.
        let lease_two = InferenceStateLease::new(target(), attachment()).unwrap();
        let mut submission_two = lease_two.submit().unwrap();
        let mut duplicated = reqwest::header::HeaderMap::new();
        duplicated.append(OCC3_DEFERRED_RECEIPT_HEADER, OCC3_DEFERRED_RECEIPT_MODE.parse().unwrap());
        duplicated.append(OCC3_DEFERRED_RECEIPT_HEADER, OCC3_DEFERRED_RECEIPT_MODE.parse().unwrap());
        assert_eq!(
            submission_two.defer_receipt(&duplicated).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        let mut wrong_value = reqwest::header::HeaderMap::new();
        wrong_value.insert(OCC3_DEFERRED_RECEIPT_HEADER, "deferred-v2".parse().unwrap());
        assert_eq!(
            submission_two.defer_receipt(&wrong_value).unwrap_err(),
            InferenceAttachmentError::InvalidReceipt
        );
        // A stream that never delivers the envelope cannot complete.
        assert_eq!(
            submission_two.complete().unwrap_err(),
            InferenceAttachmentError::MissingReceipt
        );
    }
}
