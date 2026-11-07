//! Process-local source enrollment for the actual native provider request.
//!
//! A completed source receipt is not evidence of receiver use or task semantics.
//! Raw tokens/tensors stay on the runtime. A transcript cannot restore this lease.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::inference_attachment::InferenceStateTarget;
use crate::types::{AssistantMessage, ContentBlock};

pub const SOURCE_CAPTURE_SCHEMA: &str = "occ3.capture.v1";
/// Capture-intent schema that admits the content_tail_v2 span family.
pub const SOURCE_CAPTURE_SCHEMA_V2: &str = "occ3.capture.v2";
/// v3 admits exactly the retained content-only capture mode: a no-tool
/// content turn whose publication is dispatched by the authorised
/// orchestrator instead of a same-turn model tool call.
pub const SOURCE_CAPTURE_SCHEMA_V3: &str = "occ3.capture.v3";
/// Legacy (v1/v2) capture publication mode: the captured turn itself must
/// carry exactly one `occ3_handoff` tool call binding.
pub const SOURCE_CAPTURE_MODE_SAME_TURN_TOOL: &str = "same_turn_tool_v1";
/// Retained capture mode: a real eligible content span with zero tool
/// bindings; publication happens later through the authenticated family
/// controller (orchestrator dispatch), never a forged tool binding.
pub const SOURCE_CAPTURE_MODE_CONTENT_ONLY: &str = "content_only_orchestrator_v1";
pub const SOURCE_RECEIPT_SCHEMA: &str = "occ3.source.v1";
/// Source receipt with 21 v1 fields plus the optional label anchor and occ3.span.v1 block.
pub const SOURCE_RECEIPT_SCHEMA_V2: &str = "occ3.source.v2";
pub const SOURCE_RECEIPT_HEADER: &str = "x-optimus-occ3-source";
pub const MAX_SOURCE_RECEIPT_BYTES: usize = 2048;

/// Keepalive stream extension (opt-in). Mirrors the inference-attachment names:
/// the client offers `keepalive-v1`; a server that accepts answers with the
/// deferred-receipt marker and moves the receipt into a terminal SSE event
/// emitted before the unchanged trailing `[DONE]` frame.
pub const OCC3_KEEPALIVE_REQUEST_HEADER: &str = "x-optimus-occ3-keepalive";
pub const OCC3_KEEPALIVE_REQUEST_MODE: &str = "keepalive-v1";
pub const OCC3_DEFERRED_RECEIPT_HEADER: &str = "x-optimus-occ3-deferred-receipt";
pub const OCC3_DEFERRED_RECEIPT_MODE: &str = "deferred-v1";
/// Envelope schema of the terminal receipt event.
pub const OCC3_TERMINAL_SCHEMA: &str = "occ3.terminal.v1";
pub const OCC3_TERMINAL_KIND_SOURCE: &str = "source";
/// The SSE comment frame a keepalive server emits while the inner stream drains.
pub const OCC3_KEEPALIVE_COMMENT: &str = ": occ3-keepalive";
/// The event name of the terminal receipt event.
pub const OCC3_RECEIPT_EVENT: &str = "event: optimus.occ3.receipt";
/// The v1 span policy: the last N accepted rows, tool tail and stop row
/// included. Admitted only under `occ3.capture.v1`. A v1 turn carries no span
/// selection, so the calibration collector never records it.
pub const SOURCE_SPAN_POLICY: &str = "whole_completed_response_tail_v1";
/// The content_tail_v2 family (extension `span_policy` rules R0-R13), admitted
/// only under `occ3.capture.v2`. These are the only policies whose receipt
/// carries a span selection, which calibration-corpus collection requires.
pub const SOURCE_SPAN_POLICIES_V2: [&str; 4] = [
    "content_tail_no_tool_v2",
    "content_tail_v2",
    "content_tail_last64_v2",
    "content_tail_last32_v2",
];
pub const SOURCE_SPAN_POLICY_V2_DEFAULT: &str = "content_tail_v2";
/// Operator override for the armed span policy; unset arms the v2 default.
pub const SOURCE_SPAN_POLICY_ENV: &str = "OCC3_SPAN_POLICY";
pub const SOURCE_SPAN_SCHEMA: &str = "occ3.span.v1";
pub const SOURCE_MARKER_SCHEMA: &str = "occ3.span.markers.v1";
pub const SOURCE_REPRESENTATION: &str = "final_layer_prenorm_predictor";

/// Which capture schema and span policy a source intent arms.
///
/// v2 is the default: it is the only family whose capture produces the span
/// selection the calibration collector consumes (P4 corpus rows). v1 stays
/// available by name for explicit legacy runs; an unknown name is refused with
/// the admitted set rather than falling back to anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceSpanArming {
    pub schema: &'static str,
    pub span_policy: &'static str,
    /// Retained content-only capture negotiation. Empty means the legacy
    /// same-turn tool mode; the wire stays byte-identical for v1/v2 arming.
    pub capture_mode: &'static str,
}

impl SourceSpanArming {
    pub const DEFAULT: SourceSpanArming = SourceSpanArming {
        schema: SOURCE_CAPTURE_SCHEMA_V2,
        span_policy: SOURCE_SPAN_POLICY_V2_DEFAULT,
        capture_mode: "",
    };
    pub const V1: SourceSpanArming = SourceSpanArming {
        schema: SOURCE_CAPTURE_SCHEMA,
        span_policy: SOURCE_SPAN_POLICY,
        capture_mode: "",
    };
    /// The retained content-only capture: a no-tool content turn under the
    /// no-tool span policy, published later by orchestrator dispatch.
    pub const RETAINED_CONTENT_ONLY: SourceSpanArming = SourceSpanArming {
        schema: SOURCE_CAPTURE_SCHEMA_V3,
        span_policy: "content_tail_no_tool_v2",
        capture_mode: SOURCE_CAPTURE_MODE_CONTENT_ONLY,
    };

    /// Whether this arming negotiates the retained content-only mode.
    pub fn is_retained_content_only(&self) -> bool {
        self.capture_mode == SOURCE_CAPTURE_MODE_CONTENT_ONLY
    }

    /// `None` or blank arms the v2 default; a known policy name arms the schema
    /// that admits it; anything else is `InvalidField("span_policy")`.
    pub fn resolve(requested: Option<&str>) -> Result<Self, SourceCaptureError> {
        let Some(name) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(Self::DEFAULT);
        };
        if name == SOURCE_SPAN_POLICY {
            return Ok(Self::V1);
        }
        if name == SOURCE_CAPTURE_MODE_CONTENT_ONLY {
            return Ok(Self::RETAINED_CONTENT_ONLY);
        }
        SOURCE_SPAN_POLICIES_V2
            .iter()
            .find(|policy| **policy == name)
            .map(|policy| Self {
                schema: SOURCE_CAPTURE_SCHEMA_V2,
                span_policy: policy,
                capture_mode: "",
            })
            .ok_or(SourceCaptureError::InvalidField("span_policy"))
    }

    /// Resolve from `OCC3_SPAN_POLICY`.
    pub fn from_env() -> Result<Self, SourceCaptureError> {
        Self::resolve(std::env::var(SOURCE_SPAN_POLICY_ENV).ok().as_deref())
    }

    pub fn is_v2(&self) -> bool {
        self.schema == SOURCE_CAPTURE_SCHEMA_V2
    }

    /// Every admitted policy name, for refusal messages.
    pub fn admitted_names() -> String {
        std::iter::once(SOURCE_SPAN_POLICY)
            .chain(SOURCE_SPAN_POLICIES_V2.iter().copied())
            .collect::<Vec<_>>()
            .join("|")
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCaptureIntent {
    pub schema: String,
    pub capture_id: String,
    pub source_session_id: String,
    pub source_request_id: String,
    pub turn_nonce: String,
    pub model_fingerprint: String,
    pub runtime_id: String,
    pub runtime_generation: u64,
    pub method: String,
    pub retained_count_limit: u32,
    /// Aggregate all-TP live staged and retained bytes, NOT logical receipt bytes.
    pub max_capture_bytes: u64,
    pub expires_at_epoch: u64,
    pub span_policy: String,
    /// Retained content-only negotiation. Absent (never serialized) keeps the
    /// legacy same-turn tool mode and leaves v1/v2 wire bytes identical; only a
    /// v3 intent may carry it, pinned to `content_only_orchestrator_v1`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub capture_mode: String,
    /// None uses the server profile default; explicit zero disables anchoring.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "deserialize_selected_label_anchor_alpha")]
    pub label_anchor_alpha: Option<LabelAnchorAlpha>,
}

// Absence selects the profile default; a supplied null is not a selection.
fn deserialize_selected_label_anchor_alpha<'de, D>(
    deserializer: D,
) -> Result<Option<LabelAnchorAlpha>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    LabelAnchorAlpha::deserialize(deserializer).map(Some)
}

impl std::fmt::Debug for SourceCaptureIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCaptureIntent")
            .finish_non_exhaustive()
    }
}

impl SourceCaptureIntent {
    pub fn validate(&self, now: u64) -> Result<(), SourceCaptureError> {
        // Schema-gated span policy, mirroring the server's CaptureIntent: the
        // v1 policy only under v1, the v2 family only under v2, and the
        // retained content-only mode only under v3 pinned to the no-tool
        // policy and its explicit capture mode.
        let admitted = match self.schema.as_str() {
            SOURCE_CAPTURE_SCHEMA => {
                self.span_policy == SOURCE_SPAN_POLICY && self.capture_mode.is_empty()
            }
            SOURCE_CAPTURE_SCHEMA_V2 => {
                SOURCE_SPAN_POLICIES_V2.contains(&self.span_policy.as_str())
                    && self.capture_mode.is_empty()
            }
            SOURCE_CAPTURE_SCHEMA_V3 => {
                self.span_policy == "content_tail_no_tool_v2"
                    && self.capture_mode == SOURCE_CAPTURE_MODE_CONTENT_ONLY
            }
            _ => false,
        };
        if !admitted {
            return Err(SourceCaptureError::InvalidField("schema_or_scope"));
        }
        for (field, value) in [
            ("capture_id", &self.capture_id),
            ("source_session_id", &self.source_session_id),
            ("source_request_id", &self.source_request_id),
            ("turn_nonce", &self.turn_nonce),
            ("model_fingerprint", &self.model_fingerprint),
            ("runtime_id", &self.runtime_id),
            ("method", &self.method),
        ] {
            identifier(field, value, 256)?;
        }
        if !(1..=128).contains(&self.retained_count_limit) || self.max_capture_bytes == 0 {
            return Err(SourceCaptureError::InvalidField("capture_limits"));
        }
        if self.expires_at_epoch <= now {
            return Err(SourceCaptureError::Expired);
        }
        Ok(())
    }

    /// Sorted-key compact ASCII JSON. All string fields are restricted to ASCII.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SourceCaptureError> {
        let value =
            serde_json::to_value(self).map_err(|_| SourceCaptureError::InvalidField("intent"))?;
        // Only this bounded intent field admits decimals. Keep the generic
        // canonical encoder integer-only for all other protocol documents.
        if self.label_anchor_alpha.is_none() {
            return canonical_json_bytes(&value);
        }
        let fields = value.as_object().ok_or(SourceCaptureError::InvalidField("intent"))?;
        let sorted: BTreeMap<_, _> = fields.iter().collect();
        let mut out = vec![b'{'];
        for (index, (key, value)) in sorted.into_iter().enumerate() {
            if index > 0 { out.push(b','); }
            out.extend(canonical_json_bytes(&serde_json::Value::String(key.clone()))?);
            out.push(b':');
            if key == "label_anchor_alpha" {
                out.extend(serde_json::to_vec(value).map_err(|_| SourceCaptureError::InvalidField("intent"))?);
            } else {
                out.extend(canonical_json_bytes(value)?);
            }
        }
        out.push(b'}');
        Ok(out)
    }

    pub fn digest(&self) -> Result<String, SourceCaptureError> {
        Ok(sha256(&self.canonical_bytes()?))
    }

    /// The negotiated publication mode. Empty means legacy same-turn tool.
    pub fn capture_mode(&self) -> &str {
        if self.capture_mode.is_empty() {
            SOURCE_CAPTURE_MODE_SAME_TURN_TOOL
        } else {
            self.capture_mode.as_str()
        }
    }

    /// Whether this intent negotiated the retained content-only mode.
    pub fn is_retained_content_only(&self) -> bool {
        self.capture_mode == SOURCE_CAPTURE_MODE_CONTENT_ONLY
    }
}

/// The bounded occ3.span.v1 block a v2 receipt carries next to the 21 v1 fields
/// plus the optional label anchor (extension SOURCE-SPAN-SELECTION.md section 6.2).
/// Counts cover the
/// retained window; the aligned span stays `accepted_span_start/_end`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceReceiptSpan {
    pub span_schema: String,
    pub span_policy: String,
    pub span_marker_profile: String,
    pub span_marker_schema: String,
    pub span_reasoning_terminator: String,
    pub span_turn_end_attested: bool,
    pub span_thinking_count: u32,
    pub span_content_count: u32,
    pub span_tool_call_count: u32,
    pub span_tool_arguments_count: u32,
    pub span_control_count: u32,
    pub span_unknown_count: u32,
    pub span_window_start_position: u64,
    pub span_window_end_position: u64,
    pub span_contiguous: bool,
    pub span_contains_final_content: bool,
    pub span_contains_tool_call: bool,
    pub span_message_truncated: bool,
}

impl SourceReceiptSpan {
    /// The server-side `_check_span_block` invariants, plus the binding to the
    /// intent's own policy (the worker commits the descriptor's policy).
    fn validate(
        &self,
        intent: &SourceCaptureIntent,
        receipt: &SourceCaptureReceipt,
    ) -> Result<(), SourceCaptureError> {
        let counts = [
            self.span_thinking_count,
            self.span_content_count,
            self.span_tool_call_count,
            self.span_tool_arguments_count,
            self.span_control_count,
            self.span_unknown_count,
        ];
        let sum: u64 = counts.iter().map(|value| u64::from(*value)).sum();
        let width = self
            .span_window_end_position
            .checked_sub(self.span_window_start_position)
            .ok_or(SourceCaptureError::InvalidReceipt)?;
        // A contiguous window's rows exactly fill its width; a gapped one has
        // strictly fewer rows than its width.
        let counts_fit = if self.span_contiguous {
            sum == width
        } else {
            sum < width
        };
        if self.span_schema != SOURCE_SPAN_SCHEMA
            || self.span_policy != intent.span_policy
            || !SOURCE_SPAN_POLICIES_V2.contains(&self.span_policy.as_str())
            || self.span_marker_schema != SOURCE_MARKER_SCHEMA
            || !matches!(
                self.span_reasoning_terminator.as_str(),
                "thinking_end" | "tool_call_start" | "none"
            )
            || counts.iter().any(|value| *value > 128)
            || width == 0
            || !counts_fit
            || u64::from(receipt.retained_count) > sum
            || (self.span_policy == "content_tail_no_tool_v2" && self.span_contains_tool_call)
            || (!self.span_turn_end_attested && !self.span_message_truncated)
        {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        identifier("span_marker_profile", &self.span_marker_profile, 64)
    }
}

/// Bounded label anchoring coefficient. Old receipts default to disabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct LabelAnchorAlpha(u8);

impl LabelAnchorAlpha {
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl TryFrom<f64> for LabelAnchorAlpha {
    type Error = SourceCaptureError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        [0.0, 0.3, 0.5, 0.7, 1.0]
            .iter()
            .position(|allowed| *allowed == value)
            .map(|index| Self(index as u8))
            .ok_or(SourceCaptureError::InvalidReceipt)
    }
}

impl From<LabelAnchorAlpha> for f64 {
    fn from(value: LabelAnchorAlpha) -> Self {
        [0.0, 0.3, 0.5, 0.7, 1.0][usize::from(value.0)]
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SourceCaptureReceiptWire", into = "SourceCaptureReceiptWire")]
pub struct SourceCaptureReceipt {
    pub schema: String,
    pub intent_digest: String,
    pub source_handle: String,
    pub native_rid: String,
    pub response_id: String,
    pub capture_commit_id: String,
    pub accepted_span_start: u64,
    pub accepted_span_end: u64,
    pub retained_count: u32,
    pub predictor_position_offset: i32,
    pub representation: String,
    pub layer_index: u32,
    pub hidden_dim: u32,
    pub dtype: String,
    pub layout_digest: String,
    pub ready_ranks: u32,
    pub expected_ranks: u32,
    pub accepted_sequence_digest: String,
    pub output_sha256: String,
    pub expires_at_epoch: u64,
    /// One logical aligned [retained_count, hidden_dim] prefix, NOT GPU total.
    pub byte_count: u64,
    /// Present exactly on an `occ3.source.v2` wire (the whole block or none).
    pub span: Option<SourceReceiptSpan>,
    pub label_anchor_alpha: LabelAnchorAlpha,
}

/// The flat wire shape: 21 v1 fields plus the optional label anchor and, on v2,
/// the 18 span fields at the
/// same level. Unknown fields and a partial span block are refused.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCaptureReceiptWire {
    schema: String,
    intent_digest: String,
    source_handle: String,
    native_rid: String,
    response_id: String,
    capture_commit_id: String,
    accepted_span_start: u64,
    accepted_span_end: u64,
    retained_count: u32,
    predictor_position_offset: i32,
    representation: String,
    layer_index: u32,
    hidden_dim: u32,
    dtype: String,
    layout_digest: String,
    ready_ranks: u32,
    expected_ranks: u32,
    accepted_sequence_digest: String,
    output_sha256: String,
    expires_at_epoch: u64,
    byte_count: u64,
    #[serde(default, skip_serializing_if = "LabelAnchorAlpha::is_zero")]
    label_anchor_alpha: LabelAnchorAlpha,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_marker_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_marker_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_reasoning_terminator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_turn_end_attested: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_thinking_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_content_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_tool_call_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_tool_arguments_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_control_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_unknown_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_window_start_position: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_window_end_position: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_contiguous: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_contains_final_content: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_contains_tool_call: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    span_message_truncated: Option<bool>,
}

impl From<SourceCaptureReceipt> for SourceCaptureReceiptWire {
    fn from(receipt: SourceCaptureReceipt) -> Self {
        let span = receipt.span;
        let field = |pick: fn(&SourceReceiptSpan) -> String| span.as_ref().map(pick);
        let flag = |pick: fn(&SourceReceiptSpan) -> bool| span.as_ref().map(pick);
        let count = |pick: fn(&SourceReceiptSpan) -> u32| span.as_ref().map(pick);
        let position = |pick: fn(&SourceReceiptSpan) -> u64| span.as_ref().map(pick);
        Self {
            schema: receipt.schema,
            intent_digest: receipt.intent_digest,
            source_handle: receipt.source_handle,
            native_rid: receipt.native_rid,
            response_id: receipt.response_id,
            capture_commit_id: receipt.capture_commit_id,
            accepted_span_start: receipt.accepted_span_start,
            accepted_span_end: receipt.accepted_span_end,
            retained_count: receipt.retained_count,
            predictor_position_offset: receipt.predictor_position_offset,
            representation: receipt.representation,
            layer_index: receipt.layer_index,
            hidden_dim: receipt.hidden_dim,
            dtype: receipt.dtype,
            layout_digest: receipt.layout_digest,
            ready_ranks: receipt.ready_ranks,
            expected_ranks: receipt.expected_ranks,
            accepted_sequence_digest: receipt.accepted_sequence_digest,
            output_sha256: receipt.output_sha256,
            expires_at_epoch: receipt.expires_at_epoch,
            byte_count: receipt.byte_count,
            label_anchor_alpha: receipt.label_anchor_alpha,
            span_schema: field(|s| s.span_schema.clone()),
            span_policy: field(|s| s.span_policy.clone()),
            span_marker_profile: field(|s| s.span_marker_profile.clone()),
            span_marker_schema: field(|s| s.span_marker_schema.clone()),
            span_reasoning_terminator: field(|s| s.span_reasoning_terminator.clone()),
            span_turn_end_attested: flag(|s| s.span_turn_end_attested),
            span_thinking_count: count(|s| s.span_thinking_count),
            span_content_count: count(|s| s.span_content_count),
            span_tool_call_count: count(|s| s.span_tool_call_count),
            span_tool_arguments_count: count(|s| s.span_tool_arguments_count),
            span_control_count: count(|s| s.span_control_count),
            span_unknown_count: count(|s| s.span_unknown_count),
            span_window_start_position: position(|s| s.span_window_start_position),
            span_window_end_position: position(|s| s.span_window_end_position),
            span_contiguous: flag(|s| s.span_contiguous),
            span_contains_final_content: flag(|s| s.span_contains_final_content),
            span_contains_tool_call: flag(|s| s.span_contains_tool_call),
            span_message_truncated: flag(|s| s.span_message_truncated),
        }
    }
}

impl TryFrom<SourceCaptureReceiptWire> for SourceCaptureReceipt {
    type Error = SourceCaptureError;

    fn try_from(wire: SourceCaptureReceiptWire) -> Result<Self, Self::Error> {
        let present = [
            wire.span_schema.is_some(),
            wire.span_policy.is_some(),
            wire.span_marker_profile.is_some(),
            wire.span_marker_schema.is_some(),
            wire.span_reasoning_terminator.is_some(),
            wire.span_turn_end_attested.is_some(),
            wire.span_thinking_count.is_some(),
            wire.span_content_count.is_some(),
            wire.span_tool_call_count.is_some(),
            wire.span_tool_arguments_count.is_some(),
            wire.span_control_count.is_some(),
            wire.span_unknown_count.is_some(),
            wire.span_window_start_position.is_some(),
            wire.span_window_end_position.is_some(),
            wire.span_contiguous.is_some(),
            wire.span_contains_final_content.is_some(),
            wire.span_contains_tool_call.is_some(),
            wire.span_message_truncated.is_some(),
        ];
        let span = if present.iter().all(|flag| !flag) {
            None
        } else if present.iter().all(|flag| *flag) {
            Some(SourceReceiptSpan {
                span_schema: wire.span_schema.unwrap_or_default(),
                span_policy: wire.span_policy.unwrap_or_default(),
                span_marker_profile: wire.span_marker_profile.unwrap_or_default(),
                span_marker_schema: wire.span_marker_schema.unwrap_or_default(),
                span_reasoning_terminator: wire.span_reasoning_terminator.unwrap_or_default(),
                span_turn_end_attested: wire.span_turn_end_attested.unwrap_or_default(),
                span_thinking_count: wire.span_thinking_count.unwrap_or_default(),
                span_content_count: wire.span_content_count.unwrap_or_default(),
                span_tool_call_count: wire.span_tool_call_count.unwrap_or_default(),
                span_tool_arguments_count: wire.span_tool_arguments_count.unwrap_or_default(),
                span_control_count: wire.span_control_count.unwrap_or_default(),
                span_unknown_count: wire.span_unknown_count.unwrap_or_default(),
                span_window_start_position: wire.span_window_start_position.unwrap_or_default(),
                span_window_end_position: wire.span_window_end_position.unwrap_or_default(),
                span_contiguous: wire.span_contiguous.unwrap_or_default(),
                span_contains_final_content: wire.span_contains_final_content.unwrap_or_default(),
                span_contains_tool_call: wire.span_contains_tool_call.unwrap_or_default(),
                span_message_truncated: wire.span_message_truncated.unwrap_or_default(),
            })
        } else {
            // A partial block is neither wire; refuse rather than guess.
            return Err(SourceCaptureError::InvalidReceipt);
        };
        Ok(Self {
            schema: wire.schema,
            intent_digest: wire.intent_digest,
            source_handle: wire.source_handle,
            native_rid: wire.native_rid,
            response_id: wire.response_id,
            capture_commit_id: wire.capture_commit_id,
            accepted_span_start: wire.accepted_span_start,
            accepted_span_end: wire.accepted_span_end,
            retained_count: wire.retained_count,
            predictor_position_offset: wire.predictor_position_offset,
            representation: wire.representation,
            layer_index: wire.layer_index,
            hidden_dim: wire.hidden_dim,
            dtype: wire.dtype,
            layout_digest: wire.layout_digest,
            ready_ranks: wire.ready_ranks,
            expected_ranks: wire.expected_ranks,
            accepted_sequence_digest: wire.accepted_sequence_digest,
            output_sha256: wire.output_sha256,
            expires_at_epoch: wire.expires_at_epoch,
            byte_count: wire.byte_count,
            label_anchor_alpha: wire.label_anchor_alpha,
            span,
        })
    }
}

impl std::fmt::Debug for SourceCaptureReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCaptureReceipt")
            .finish_non_exhaustive()
    }
}

/// Fresh connector-verified expected shape. It is not public request authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCaptureProfile {
    pub layer_index: u32,
    pub hidden_dim: u32,
    pub dtype: String,
    pub layout_digest: String,
    pub expected_ranks: u32,
}

/// Local full-operation limits; never serialized into ordinary provider options.
#[derive(Clone, Copy, Debug)]
pub struct SourceCaptureLimits {
    pub deadline: Instant,
    pub max_response_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceCaptureState {
    Prepared,
    Submitted,
    Captured,
    Cancelled,
    Uncertain,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SourceCaptureError {
    #[error("Source capture has an invalid {0} field.")]
    InvalidField(&'static str),
    #[error("Source capture target does not match this request.")]
    TargetMismatch,
    #[error("Source capture endpoint is unsafe.")]
    InvalidEndpoint,
    #[error("Source capture has expired or exceeded its deadline.")]
    Expired,
    #[error("Source capture clock is unavailable.")]
    ClockUnavailable,
    #[error("Source capture is no longer prepared; replay is forbidden.")]
    NotPrepared,
    #[error("Source capture submission is no longer active.")]
    NotSubmitted,
    #[error("Source capture receipt is missing, invalid, or mismatched.")]
    InvalidReceipt,
    #[error("Source capture body or terminal provenance is invalid.")]
    InvalidCompletion,
    #[error("Source capture output exceeds its reserved bound.")]
    ResponseTooLarge,
    #[error(
        "Source capture payload fields are reserved; capture and consumption cannot be combined."
    )]
    InvalidPayload,
    #[error("Source capture tool provenance does not match the completed response.")]
    InvalidToolBinding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceToolBinding {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments_sha256: String,
}

struct SourceInner {
    target: InferenceStateTarget,
    intent: SourceCaptureIntent,
    profile: SourceCaptureProfile,
    limits: SourceCaptureLimits,
    state: Mutex<SourceRecord>,
    cancellation: CancellationToken,
}
struct SourceRecord {
    state: SourceCaptureState,
    receipt: Option<SourceCaptureReceipt>,
    tools: Vec<SourceToolBinding>,
    busy_refusal: Option<SourceBusyRefusal>,
}

/// A pre-event busy refusal of an armed source capture: the server refused
/// admission before generating anything, so nothing was captured. Recorded on
/// the (now `Cancelled`) lease for the session's busy re-arm budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBusyRefusal {
    /// Epoch second of the refusal.
    pub refused_at_epoch: u64,
    /// Bounded server `Retry-After` in seconds, when it was a valid value.
    pub retry_after_secs: Option<u64>,
}

#[derive(Clone)]
pub struct SourceCaptureLease(Arc<SourceInner>);
impl std::fmt::Debug for SourceCaptureLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCaptureLease")
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl SourceCaptureLease {
    pub fn new(
        target: InferenceStateTarget,
        intent: SourceCaptureIntent,
        profile: SourceCaptureProfile,
        limits: SourceCaptureLimits,
    ) -> Result<Self, SourceCaptureError> {
        intent.validate(epoch_now()?)?;
        identifier("provider", &target.provider, 256)?;
        identifier("model_id", &target.model_id, 256)?;
        endpoint(&target.base_url)?;
        digest_field(&profile.layout_digest)?;
        if profile.hidden_dim == 0
            || profile.expected_ranks == 0
            || element_bytes(&profile.dtype).is_none()
        {
            return Err(SourceCaptureError::InvalidField("profile"));
        }
        if limits.deadline <= Instant::now()
            || limits.max_response_bytes == 0
            || limits.max_response_bytes > 64 * 1024 * 1024
        {
            return Err(SourceCaptureError::InvalidField("response_limits"));
        }
        Ok(Self(Arc::new(SourceInner {
            target,
            intent,
            profile,
            limits,
            state: Mutex::new(SourceRecord {
                state: SourceCaptureState::Prepared,
                receipt: None,
                tools: Vec::new(),
                busy_refusal: None,
            }),
            cancellation: CancellationToken::new(),
        })))
    }
    pub fn target(&self) -> &InferenceStateTarget {
        &self.0.target
    }
    pub fn intent(&self) -> &SourceCaptureIntent {
        &self.0.intent
    }
    pub fn profile(&self) -> &SourceCaptureProfile {
        &self.0.profile
    }
    pub fn limits(&self) -> SourceCaptureLimits {
        self.0.limits
    }
    pub fn state(&self) -> SourceCaptureState {
        self.0.state.lock().state
    }
    /// The recorded pre-event busy refusal, if the server refused this
    /// capture as busy before admission (the lease is then `Cancelled`).
    pub fn busy_refusal(&self) -> Option<SourceBusyRefusal> {
        let record = self.0.state.lock();
        (record.state == SourceCaptureState::Cancelled)
            .then_some(record.busy_refusal)
            .flatten()
    }
    /// Available only after the exact native response and parser completed.
    pub fn receipt(&self) -> Option<SourceCaptureReceipt> {
        let record = self.0.state.lock();
        (record.state == SourceCaptureState::Captured)
            .then(|| record.receipt.clone())
            .flatten()
    }
    /// Cleanup-only metadata remains readable after local retirement. This does
    /// not restore publication authority; validate_tool still requires Captured.
    pub fn retirement_receipt(&self) -> Option<SourceCaptureReceipt> {
        self.0.state.lock().receipt.clone()
    }
    /// The negotiated capture/publication mode of this lease's intent.
    pub fn capture_mode(&self) -> &str {
        self.0.intent.capture_mode()
    }

    /// Whether this lease was armed for retained content-only capture.
    pub fn is_retained_content_only(&self) -> bool {
        self.0.intent.is_retained_content_only()
    }

    pub fn tool_bindings(&self) -> Vec<SourceToolBinding> {
        let record = self.0.state.lock();
        if record.state == SourceCaptureState::Captured {
            record.tools.clone()
        } else {
            Vec::new()
        }
    }
    pub fn validate_tool(
        &self,
        tool_id: &str,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Result<SourceToolBinding, SourceCaptureError> {
        let expected = SourceToolBinding {
            tool_call_id: tool_id.into(),
            tool_name: name.into(),
            arguments_sha256: value_digest(arguments)?,
        };
        let record = self.0.state.lock();
        if record.state != SourceCaptureState::Captured || !record.tools.contains(&expected) {
            return Err(SourceCaptureError::InvalidToolBinding);
        }
        Ok(expected)
    }
    /// Test seam for downstream crates: closes a still-`Prepared` (never
    /// sent) lease as `Cancelled` with a busy refusal record, the same end
    /// state a real pre-admission busy refusal produces. `Prepared ->
    /// Cancelled` is already a public transition ([`Self::cancel`]), so this
    /// cannot turn a submitted or uncertain lease into proven non-use. The
    /// only effect a non-test caller could get is to have its own unsent
    /// capture counted as a busy retry instead of a capture failure, which
    /// is still bounded by the queue busy budget. Kept public (not behind a
    /// cargo feature) so test builds share the workspace's pi-ai artifact.
    #[doc(hidden)]
    pub fn cancel_unsent_as_busy(&self, refusal: SourceBusyRefusal) -> bool {
        let mut record = self.0.state.lock();
        if record.state != SourceCaptureState::Prepared {
            return false;
        }
        record.state = SourceCaptureState::Cancelled;
        record.busy_refusal = Some(refusal);
        self.0.cancellation.cancel();
        true
    }
    /// Revokes local authority; callers separately retire the runtime resource.
    pub fn cancel(&self) -> SourceCaptureState {
        let mut record = self.0.state.lock();
        record.state = match record.state {
            SourceCaptureState::Prepared => SourceCaptureState::Cancelled,
            SourceCaptureState::Submitted => SourceCaptureState::Uncertain,
            SourceCaptureState::Captured => SourceCaptureState::Retired,
            other => other,
        };
        self.0.cancellation.cancel();
        record.state
    }
    pub(crate) async fn cancelled(&self) {
        self.0.cancellation.cancelled().await;
    }
    pub(crate) fn validate_target(
        &self,
        provider: &str,
        model: &str,
        base_url: &str,
    ) -> Result<(), SourceCaptureError> {
        if self.state() != SourceCaptureState::Prepared {
            return Err(SourceCaptureError::NotPrepared);
        }
        if provider != self.target().provider
            || model != self.target().model_id
            || base_url != self.target().base_url
        {
            return Err(SourceCaptureError::TargetMismatch);
        }
        self.intent().validate(epoch_now()?)?;
        if self.limits().deadline <= Instant::now() {
            return Err(SourceCaptureError::Expired);
        }
        Ok(())
    }
    pub(crate) fn submit(&self) -> Result<SourceSubmission, SourceCaptureError> {
        let mut record = self.0.state.lock();
        if record.state != SourceCaptureState::Prepared {
            return Err(SourceCaptureError::NotPrepared);
        }
        self.intent().validate(epoch_now()?)?;
        if self.limits().deadline <= Instant::now() {
            return Err(SourceCaptureError::Expired);
        }
        record.state = SourceCaptureState::Submitted;
        Ok(SourceSubmission {
            lease: self.clone(),
            receipt: None,
            body_verified: false,
            deferred: false,
            resend_wait: None,
        })
    }
}

#[derive(Debug)]
pub(crate) struct SourceSubmission {
    lease: SourceCaptureLease,
    receipt: Option<SourceCaptureReceipt>,
    body_verified: bool,
    /// The server accepted the keepalive offer: the receipt arrives as the
    /// terminal `optimus.occ3.receipt` SSE event instead of response headers.
    deferred: bool,
    /// Set only while waiting out `Retry-After` before a same-request resend:
    /// the last response was a proven pre-use busy refusal and nothing is in
    /// flight. Holds that refusal's Retry-After. If the submission is dropped
    /// in this window (the outer stream guard cancels the body on the signal
    /// or the deadline), it closes as that busy refusal, not `Uncertain`. An
    /// explicit `lease.cancel()` keeps its own conservative `Uncertain`.
    resend_wait: Option<Option<u64>>,
}
impl SourceSubmission {
    pub(crate) fn ensure_active(&self) -> Result<(), SourceCaptureError> {
        if self.lease.state() != SourceCaptureState::Submitted {
            return Err(SourceCaptureError::NotSubmitted);
        }
        if self.lease.limits().deadline <= Instant::now()
            || self.lease.intent().expires_at_epoch <= epoch_now()?
        {
            return Err(SourceCaptureError::Expired);
        }
        Ok(())
    }
    pub(crate) fn verify_header(
        &mut self,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<(), SourceCaptureError> {
        self.ensure_active()?;
        let mut values = headers.get_all(SOURCE_RECEIPT_HEADER).iter();
        let header = values.next().ok_or(SourceCaptureError::InvalidReceipt)?;
        if values.next().is_some() || header.as_bytes().len() > MAX_SOURCE_RECEIPT_BYTES {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        if headers
            .get_all(reqwest::header::CONTENT_ENCODING)
            .iter()
            .any(|v| v.as_bytes() != b"identity")
        {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        let receipt: SourceCaptureReceipt = serde_json::from_slice(header.as_bytes())
            .map_err(|_| SourceCaptureError::InvalidReceipt)?;
        validate_receipt(
            &receipt,
            self.lease.intent(),
            self.lease.profile(),
            epoch_now()?,
        )?;
        self.receipt = Some(receipt);
        Ok(())
    }
    pub(crate) fn expected_response_id(&self) -> Result<&str, SourceCaptureError> {
        self.receipt
            .as_ref()
            .map(|r| r.response_id.as_str())
            .ok_or(SourceCaptureError::InvalidReceipt)
    }
    /// Keepalive offer handling at response headers. `Ok(false)` means the
    /// server kept the buffered contract, so the caller must fall back to
    /// `verify_header`; `Ok(true)` defers the receipt to the terminal event.
    pub(crate) fn defer_header(
        &mut self,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<bool, SourceCaptureError> {
        self.ensure_active()?;
        if headers
            .get_all(reqwest::header::CONTENT_ENCODING)
            .iter()
            .any(|v| v.as_bytes() != b"identity")
        {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        let mut values = headers.get_all(OCC3_DEFERRED_RECEIPT_HEADER).iter();
        match values.next() {
            None => Ok(false),
            Some(value) => {
                if values.next().is_some() || value.as_bytes() != OCC3_DEFERRED_RECEIPT_MODE.as_bytes() {
                    return Err(SourceCaptureError::InvalidReceipt);
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
    /// are exactly `verify_header`'s; only the transport differs.
    pub(crate) fn verify_deferred_receipt(
        &mut self,
        envelope: &serde_json::Value,
    ) -> Result<(), SourceCaptureError> {
        self.ensure_active()?;
        if !self.deferred
            || envelope.get("schema").and_then(serde_json::Value::as_str) != Some(OCC3_TERMINAL_SCHEMA)
            || envelope.get("kind").and_then(serde_json::Value::as_str) != Some(OCC3_TERMINAL_KIND_SOURCE)
        {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        let receipt_value = envelope
            .get("receipt")
            .cloned()
            .ok_or(SourceCaptureError::InvalidReceipt)?;
        if serde_json::to_vec(&receipt_value).map(|bytes| bytes.len()).unwrap_or(usize::MAX)
            > MAX_SOURCE_RECEIPT_BYTES
        {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        let receipt: SourceCaptureReceipt = serde_json::from_value(receipt_value)
            .map_err(|_| SourceCaptureError::InvalidReceipt)?;
        if envelope.get("response_id").and_then(serde_json::Value::as_str) != Some(receipt.response_id.as_str()) {
            return Err(SourceCaptureError::InvalidReceipt);
        }
        validate_receipt(
            &receipt,
            self.lease.intent(),
            self.lease.profile(),
            epoch_now()?,
        )?;
        self.receipt = Some(receipt);
        Ok(())
    }

    pub(crate) fn verify_body(&mut self, bytes: &[u8]) -> Result<(), SourceCaptureError> {
        self.ensure_active()?;
        if bytes.len() > self.lease.limits().max_response_bytes {
            return Err(SourceCaptureError::ResponseTooLarge);
        }
        let receipt = self
            .receipt
            .as_ref()
            .ok_or(SourceCaptureError::InvalidReceipt)?;
        if sha256(bytes) != receipt.output_sha256 {
            return Err(SourceCaptureError::InvalidCompletion);
        }
        self.body_verified = true;
        Ok(())
    }
    pub(crate) fn complete(self, output: &AssistantMessage) -> Result<(), SourceCaptureError> {
        self.ensure_active()?;
        let receipt = self
            .receipt
            .as_ref()
            .ok_or(SourceCaptureError::InvalidReceipt)?;
        if !self.body_verified
            || output.response_id.as_deref() != Some(receipt.response_id.as_str())
            || !matches!(output.stop_reason.as_str(), "stop" | "length" | "toolUse")
        {
            return Err(SourceCaptureError::InvalidCompletion);
        }
        let mut tools = Vec::new();
        for block in &output.content {
            if let ContentBlock::ToolCall(tool) = block {
                identifier("tool_call_id", &tool.id, 256)?;
                identifier("tool_name", &tool.name, 256)?;
                if tools
                    .iter()
                    .any(|prior: &SourceToolBinding| prior.tool_call_id == tool.id)
                {
                    return Err(SourceCaptureError::InvalidToolBinding);
                }
                tools.push(SourceToolBinding {
                    tool_call_id: tool.id.clone(),
                    tool_name: tool.name.clone(),
                    arguments_sha256: value_digest(&serde_json::Value::Object(
                        tool.arguments.clone(),
                    ))?,
                });
            }
        }
        let mut record = self.lease.0.state.lock();
        if record.state != SourceCaptureState::Submitted {
            return Err(SourceCaptureError::NotSubmitted);
        }
        record.receipt = Some(receipt.clone());
        record.tools = tools;
        record.state = SourceCaptureState::Captured;
        Ok(())
    }
}
impl SourceSubmission {
    /// Whether an identical resend after `delay` still fits this lease: the
    /// submission is still `Submitted` and both the local deadline and the
    /// intent expiry lie strictly after the delay PLUS `headroom`, the time
    /// the resent request may need to be admitted and produce its first
    /// event. Without that headroom a resend started just before the
    /// deadline would expire in flight as `Uncertain`, although refusing at
    /// once is proven non-use. Called only after a classified pre-admission
    /// busy refusal, before any stream event.
    pub(crate) fn resend_window_allows(&self, delay: std::time::Duration, headroom: std::time::Duration) -> bool {
        if self.lease.state() != SourceCaptureState::Submitted {
            return false;
        }
        let Some(needed) = delay.checked_add(headroom) else {
            return false;
        };
        let Some(resume) = Instant::now().checked_add(needed) else {
            return false;
        };
        let Ok(now) = epoch_now() else {
            return false;
        };
        resume < self.lease.limits().deadline
            && now.saturating_add(needed.as_secs()) < self.lease.intent().expires_at_epoch
    }
    /// The server refused the armed capture as busy BEFORE any stream event:
    /// nothing was generated or captured. That is proven non-use, so the
    /// lease closes as `Cancelled` (never `Uncertain`, never reusable). The
    /// cancellation token is left for the request guard, as on the receiver
    /// path, so the provider's biased select keeps this bounded refusal.
    pub(crate) fn refuse_before_admission(self, retry_after_secs: Option<u64>) {
        let mut record = self.lease.0.state.lock();
        if record.state == SourceCaptureState::Submitted {
            record.state = SourceCaptureState::Cancelled;
            record.busy_refusal = Some(SourceBusyRefusal {
                refused_at_epoch: epoch_now().unwrap_or(0),
                retry_after_secs,
            });
        }
    }
}
impl SourceSubmission {
    /// Enter (`Some(retry_after)`) or leave (`None`) the Retry-After wait
    /// before a same-request resend. See `resend_wait`.
    pub(crate) fn set_resend_wait(&mut self, retry_after_secs: Option<Option<u64>>) {
        self.resend_wait = retry_after_secs;
    }
}
impl Drop for SourceSubmission {
    fn drop(&mut self) {
        let mut record = self.lease.0.state.lock();
        if record.state == SourceCaptureState::Submitted {
            match self.resend_wait {
                // Dropped while waiting to resend: nothing is in flight and the
                // last refusal was proven pre-use, so close as that refusal.
                Some(retry_after_secs) => {
                    record.state = SourceCaptureState::Cancelled;
                    record.busy_refusal = Some(SourceBusyRefusal {
                        refused_at_epoch: epoch_now().unwrap_or(0),
                        retry_after_secs,
                    });
                }
                None => record.state = SourceCaptureState::Uncertain,
            }
        }
    }
}

pub fn validate_receipt(
    receipt: &SourceCaptureReceipt,
    intent: &SourceCaptureIntent,
    profile: &SourceCaptureProfile,
    now: u64,
) -> Result<(), SourceCaptureError> {
    intent.validate(now)?;
    // The receipt schema follows the intent schema: a v1 intent is answered by
    // the 21 v1 fields plus the optional label anchor, a v2 intent by that wire
    // with the span block.
    let schema_matches = match (&receipt.span, intent.schema.as_str()) {
        (None, SOURCE_CAPTURE_SCHEMA) => receipt.schema == SOURCE_RECEIPT_SCHEMA,
        (Some(_), SOURCE_CAPTURE_SCHEMA_V2) => receipt.schema == SOURCE_RECEIPT_SCHEMA_V2,
        // The retained content-only capture reuses the v2 receipt wire: the
        // mode lives in the intent (and its digest), never the receipt.
        (Some(_), SOURCE_CAPTURE_SCHEMA_V3) => receipt.schema == SOURCE_RECEIPT_SCHEMA_V2,
        _ => false,
    };
    if intent.label_anchor_alpha.is_some_and(|alpha| alpha != receipt.label_anchor_alpha) {
        return Err(SourceCaptureError::InvalidReceipt);
    }
    if !schema_matches
        || receipt.representation != SOURCE_REPRESENTATION
        || receipt.predictor_position_offset != -1
        || receipt.intent_digest != intent.digest()?
    {
        return Err(SourceCaptureError::InvalidReceipt);
    }
    identifier("source_handle", &receipt.source_handle, 128)?;
    for (field, value) in [
        ("native_rid", &receipt.native_rid),
        ("response_id", &receipt.response_id),
        ("capture_commit_id", &receipt.capture_commit_id),
    ] {
        identifier(field, value, 96)?;
    }
    for digest in [
        &receipt.intent_digest,
        &receipt.layout_digest,
        &receipt.accepted_sequence_digest,
        &receipt.output_sha256,
    ] {
        digest_field(digest)?;
    }
    let width = receipt
        .accepted_span_end
        .checked_sub(receipt.accepted_span_start)
        .ok_or(SourceCaptureError::InvalidReceipt)?;
    let element = element_bytes(&receipt.dtype).ok_or(SourceCaptureError::InvalidReceipt)?;
    let logical_bytes = u64::from(receipt.retained_count)
        .checked_mul(u64::from(receipt.hidden_dim))
        .and_then(|v| v.checked_mul(element))
        .ok_or(SourceCaptureError::InvalidReceipt)?;
    // The retained content-only capture reuses the v2 receipt wire, so its
    // stricter retention identity lives here: every aligned span row of a v3
    // capture is retained (the final content row is never dropped for a stop
    // row - the canary's final-character loss). The legacy v1/v2 arms keep
    // the historical upper bound.
    let retained_fits = if intent.is_retained_content_only() {
        u64::from(receipt.retained_count) == width
    } else {
        u64::from(receipt.retained_count) <= width
    };
    if width == 0
        || receipt.retained_count == 0
        || receipt.retained_count > intent.retained_count_limit
        || !retained_fits
        || logical_bytes == 0
        || receipt.byte_count != logical_bytes
        || receipt.byte_count > intent.max_capture_bytes
        || receipt.hidden_dim != profile.hidden_dim
        || receipt.layer_index != profile.layer_index
        || receipt.dtype != profile.dtype
        || receipt.layout_digest != profile.layout_digest
        || receipt.expected_ranks != profile.expected_ranks
        || receipt.expected_ranks == 0
        || receipt.ready_ranks != receipt.expected_ranks
        || receipt.expires_at_epoch > intent.expires_at_epoch
        || receipt.expires_at_epoch <= now
    {
        return Err(SourceCaptureError::InvalidReceipt);
    }
    if let Some(span) = &receipt.span {
        span.validate(intent, receipt)?;
    }
    Ok(())
}

/// Shared protocol canonical JSON: sorted object keys, compact ASCII, exact integers.
/// Typed protocol deserialization rejects duplicate/unknown keys before this call.
pub fn canonical_json_bytes(value: &serde_json::Value) -> Result<Vec<u8>, SourceCaptureError> {
    fn string(value: &str, out: &mut String) -> Result<(), SourceCaptureError> {
        let escaped = serde_json::to_string(value)
            .map_err(|_| SourceCaptureError::InvalidField("canonical_json"))?;
        for ch in escaped.chars() {
            if ch.is_ascii() {
                out.push(ch);
            } else {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
        Ok(())
    }
    fn encode(value: &serde_json::Value, out: &mut String) -> Result<(), SourceCaptureError> {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            serde_json::Value::Number(v) => {
                if v.as_u64().is_none() && v.as_i64().is_none() {
                    return Err(SourceCaptureError::InvalidField("canonical_integer"));
                }
                out.push_str(&v.to_string());
            }
            serde_json::Value::String(v) => string(v, out)?,
            serde_json::Value::Array(values) => {
                out.push('[');
                for (i, v) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    encode(v, out)?;
                }
                out.push(']');
            }
            serde_json::Value::Object(values) => {
                out.push('{');
                let sorted: BTreeMap<_, _> = values.iter().collect();
                for (i, (key, value)) in sorted.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    string(key, out)?;
                    out.push(':');
                    encode(value, out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    let mut out = String::new();
    encode(value, &mut out)?;
    Ok(out.into_bytes())
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn value_digest(value: &serde_json::Value) -> Result<String, SourceCaptureError> {
    fn sorted(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let keys: std::collections::BTreeSet<_> = map.keys().collect();
                serde_json::Value::Object(
                    keys.into_iter()
                        .map(|key| (key.clone(), sorted(&map[key])))
                        .collect(),
                )
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(sorted).collect())
            }
            other => other.clone(),
        }
    }
    serde_json::to_vec(&sorted(value))
        .map(|bytes| sha256(&bytes))
        .map_err(|_| SourceCaptureError::InvalidToolBinding)
}
fn identifier(field: &'static str, value: &str, maximum: usize) -> Result<(), SourceCaptureError> {
    if value.is_empty()
        || value.len() > maximum
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-/".contains(&b))
    {
        return Err(SourceCaptureError::InvalidField(field));
    }
    Ok(())
}
fn digest_field(value: &str) -> Result<(), SourceCaptureError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(SourceCaptureError::InvalidField("digest"));
    }
    Ok(())
}
fn element_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "bfloat16" | "float16" => Some(2),
        "float32" => Some(4),
        _ => None,
    }
}
fn epoch_now() -> Result<u64, SourceCaptureError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_secs())
        .map_err(|_| SourceCaptureError::ClockUnavailable)
}
fn endpoint(value: &str) -> Result<(), SourceCaptureError> {
    if value.len() > 2048
        || value
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err(SourceCaptureError::InvalidEndpoint);
    }
    let url = url::Url::parse(value).map_err(|_| SourceCaptureError::InvalidEndpoint)?;
    let loopback = match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(h)) => h == "localhost",
        None => false,
    };
    if !(url.scheme() == "https" || (url.scheme() == "http" && loopback))
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(SourceCaptureError::InvalidEndpoint);
    }
    Ok(())
}

#[cfg(test)]
mod span_arming_tests {
    //! Offline shapes only: the v2 arming resolver, the schema-gated intent
    //! and the occ3.source.v2 receipt wire. No provider, no GPU.
    use super::*;

    fn now() -> u64 {
        epoch_now().unwrap()
    }

    fn intent(arming: SourceSpanArming) -> SourceCaptureIntent {
        SourceCaptureIntent {
            schema: arming.schema.into(),
            capture_id: "capture-1".into(),
            source_session_id: "source-1".into(),
            source_request_id: "request-1".into(),
            turn_nonce: "turn-1".into(),
            model_fingerprint: "a".repeat(64),
            runtime_id: "runtime-1".into(),
            runtime_generation: 7,
            method: "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1".into(),
            retained_count_limit: 128,
            max_capture_bytes: 1 << 20,
            expires_at_epoch: now() + 300,
            span_policy: arming.span_policy.into(),
            capture_mode: arming.capture_mode.into(),
            label_anchor_alpha: None,
        }
    }

    fn profile() -> SourceCaptureProfile {
        SourceCaptureProfile {
            layer_index: 77,
            hidden_dim: 6144,
            dtype: "bfloat16".into(),
            layout_digest: "b".repeat(64),
            expected_ranks: 8,
        }
    }

    /// A GLM-5.3-shaped v1 wire: 8 of 10 accepted rows, [8, 6144] bfloat16.
    fn v1_wire(intent: &SourceCaptureIntent) -> serde_json::Value {
        serde_json::json!({
            "schema": SOURCE_RECEIPT_SCHEMA,
            "intent_digest": intent.digest().unwrap(),
            "source_handle": "owner-only-handle",
            "native_rid": "native-1",
            "response_id": "native-1",
            "capture_commit_id": "commit-1",
            "accepted_span_start": 1,
            "accepted_span_end": 9,
            "retained_count": 8,
            "predictor_position_offset": -1,
            "representation": SOURCE_REPRESENTATION,
            "layer_index": 77,
            "hidden_dim": 6144,
            "dtype": "bfloat16",
            "layout_digest": "b".repeat(64),
            "ready_ranks": 8,
            "expected_ranks": 8,
            "accepted_sequence_digest": "c".repeat(64),
            "output_sha256": "d".repeat(64),
            "expires_at_epoch": intent.expires_at_epoch,
            "byte_count": 8u64 * 6144 * 2,
        })
    }

    /// The same capture answered on the v2 wire: a 10-row contiguous window
    /// (thinking-end row, 8 content rows, the role-token stop row).
    fn v2_wire(intent: &SourceCaptureIntent) -> serde_json::Value {
        let mut wire = v1_wire(intent);
        wire["schema"] = serde_json::json!(SOURCE_RECEIPT_SCHEMA_V2);
        let span = serde_json::json!({
            "span_schema": SOURCE_SPAN_SCHEMA,
            "span_policy": intent.span_policy,
            "span_marker_profile": "glm45_glm47_xml_kv.v1",
            "span_marker_schema": SOURCE_MARKER_SCHEMA,
            "span_reasoning_terminator": "thinking_end",
            "span_turn_end_attested": true,
            "span_thinking_count": 0,
            "span_content_count": 8,
            "span_tool_call_count": 0,
            "span_tool_arguments_count": 0,
            "span_control_count": 2,
            "span_unknown_count": 0,
            "span_window_start_position": 640,
            "span_window_end_position": 650,
            "span_contiguous": true,
            "span_contains_final_content": true,
            "span_contains_tool_call": false,
            "span_message_truncated": false,
        });
        for (key, value) in span.as_object().unwrap() {
            wire[key.as_str()] = value.clone();
        }
        wire
    }

    fn parse(wire: &serde_json::Value) -> Result<SourceCaptureReceipt, serde_json::Error> {
        serde_json::from_value(wire.clone())
    }

    #[test]
    fn label_anchor_receipt_defaults_and_bounded_roundtrip() {
        let intent = intent(SourceSpanArming::resolve(None).unwrap());
        for mut wire in [v1_wire(&intent), v2_wire(&intent)] {
            assert_eq!(f64::from(parse(&wire).unwrap().label_anchor_alpha), 0.0);
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0] {
                wire["label_anchor_alpha"] = serde_json::json!(alpha);
                let receipt = parse(&wire).unwrap();
                assert_eq!(f64::from(receipt.label_anchor_alpha), alpha);
                assert_eq!(
                    serde_json::to_value(receipt)
                        .unwrap()
                        .get("label_anchor_alpha")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                    alpha
                );
            }
            for invalid in [
                serde_json::json!(true),
                serde_json::json!(null),
                serde_json::json!("0.3"),
                serde_json::json!(0.1),
                serde_json::json!(-0.3),
                serde_json::json!(1.1),
            ] {
                wire["label_anchor_alpha"] = invalid;
                assert!(parse(&wire).is_err());
            }
        }
    }

    #[test]
    fn v2_is_the_default_arming_and_unknown_names_refuse() {
        assert_eq!(SourceSpanArming::resolve(None).unwrap(), SourceSpanArming::DEFAULT);
        assert_eq!(SourceSpanArming::resolve(Some("  ")).unwrap(), SourceSpanArming::DEFAULT);
        assert!(SourceSpanArming::DEFAULT.is_v2());
        assert_eq!(SourceSpanArming::DEFAULT.span_policy, "content_tail_v2");
        assert_eq!(SourceSpanArming::DEFAULT.schema, "occ3.capture.v2");
        let v1 = SourceSpanArming::resolve(Some(SOURCE_SPAN_POLICY)).unwrap();
        assert_eq!(v1, SourceSpanArming::V1);
        assert!(!v1.is_v2() && v1.schema == "occ3.capture.v1");
        for policy in SOURCE_SPAN_POLICIES_V2 {
            let armed = SourceSpanArming::resolve(Some(policy)).unwrap();
            assert_eq!((armed.schema, armed.span_policy), ("occ3.capture.v2", policy));
        }
        assert!(matches!(
            SourceSpanArming::resolve(Some("last_accepted")),
            Err(SourceCaptureError::InvalidField("span_policy"))
        ));
        assert!(SourceSpanArming::admitted_names().contains("content_tail_no_tool_v2"));
    }

    #[test]
    fn retained_content_only_arming_is_explicitly_negotiated() {
        let retained = SourceSpanArming::RETAINED_CONTENT_ONLY;
        assert!(retained.is_retained_content_only());
        assert_eq!(retained.schema, "occ3.capture.v3");
        assert_eq!(retained.span_policy, "content_tail_no_tool_v2");
        assert_eq!(retained.capture_mode, SOURCE_CAPTURE_MODE_CONTENT_ONLY);
        // The mode name also arms through the same resolver as the policies.
        assert_eq!(
            SourceSpanArming::resolve(Some(SOURCE_CAPTURE_MODE_CONTENT_ONLY)).unwrap(),
            retained
        );
        // Legacy arming never carries the mode.
        assert!(!SourceSpanArming::DEFAULT.is_retained_content_only());
        assert!(!SourceSpanArming::V1.is_retained_content_only());
        // A v3 intent validates only with the pinned policy and mode.
        let mut v3 = intent(SourceSpanArming::RETAINED_CONTENT_ONLY);
        assert!(v3.validate(now()).is_ok());
        assert!(v3.is_retained_content_only());
        assert_eq!(v3.capture_mode(), SOURCE_CAPTURE_MODE_CONTENT_ONLY);
        v3.span_policy = "content_tail_v2".into();
        assert!(matches!(
            v3.validate(now()),
            Err(SourceCaptureError::InvalidField("schema_or_scope"))
        ));
        v3.span_policy = "content_tail_no_tool_v2".into();
        v3.capture_mode = "model_tool_v2".into();
        assert!(matches!(
            v3.validate(now()),
            Err(SourceCaptureError::InvalidField("schema_or_scope"))
        ));
        // The mode never leaks onto a legacy v1/v2 intent wire: the legacy
        // intent validates clean, and crossing the mode onto it is refused.
        // (The old shape demanded the legacy intent itself be refused, which
        // contradicted intent_admits_only_schema_matched_policies once the
        // v2 default became content_tail_v2.)
        let legacy = intent(SourceSpanArming::DEFAULT);
        assert!(legacy.capture_mode.is_empty());
        let wire = serde_json::to_value(&legacy).unwrap();
        assert!(wire.get("capture_mode").is_none());
        assert!(legacy.validate(now()).is_ok());
        let mut crossed = legacy.clone();
        crossed.capture_mode = SOURCE_CAPTURE_MODE_CONTENT_ONLY.into();
        assert!(matches!(
            crossed.validate(now()),
            Err(SourceCaptureError::InvalidField("schema_or_scope"))
        ));
    }

    #[test]
    fn intent_admits_only_schema_matched_policies() {
        assert!(intent(SourceSpanArming::DEFAULT).validate(now()).is_ok());
        assert!(intent(SourceSpanArming::V1).validate(now()).is_ok());
        let mut crossed = intent(SourceSpanArming::V1);
        crossed.span_policy = "content_tail_v2".into();
        assert!(matches!(
            crossed.validate(now()),
            Err(SourceCaptureError::InvalidField("schema_or_scope"))
        ));
        let mut crossed = intent(SourceSpanArming::DEFAULT);
        crossed.span_policy = SOURCE_SPAN_POLICY.into();
        assert!(crossed.validate(now()).is_err());
        let mut unknown = intent(SourceSpanArming::DEFAULT);
        unknown.schema = "occ3.capture.v3".into();
        assert!(unknown.validate(now()).is_err());
    }

    #[test]
    fn v2_receipt_round_trips_flat_and_validates_against_a_v2_intent() {
        let v2 = intent(SourceSpanArming::DEFAULT);
        let receipt = parse(&v2_wire(&v2)).expect("v2 wire parses");
        let span = receipt.span.as_ref().expect("span block present");
        assert_eq!(span.span_policy, "content_tail_v2");
        assert_eq!(span.span_content_count, 8);
        assert!(validate_receipt(&receipt, &v2, &profile(), now()).is_ok());
        // Serialization is the same flat 39-key wire, not a nested object.
        let back = serde_json::to_value(&receipt).unwrap();
        assert_eq!(back.as_object().unwrap().len(), 39);
        assert!(back.get("span").is_none());
        assert_eq!(back, v2_wire(&v2));
        // A v1 receipt serializes as exactly the 21 v1 keys.
        let v1 = intent(SourceSpanArming::V1);
        let bare = parse(&v1_wire(&v1)).unwrap();
        assert!(bare.span.is_none());
        assert_eq!(serde_json::to_value(&bare).unwrap().as_object().unwrap().len(), 21);
        assert!(validate_receipt(&bare, &v1, &profile(), now()).is_ok());
    }

    #[test]
    fn receipt_schema_must_follow_the_intent_schema() {
        let v2 = intent(SourceSpanArming::DEFAULT);
        let v1 = intent(SourceSpanArming::V1);
        // A v2 intent answered with the bare v1 wire (the pre-fix live shape
        // once a v2-armed client met an older server) is refused.
        let mut bare = v1_wire(&v2);
        let receipt = parse(&bare).unwrap();
        assert!(matches!(
            validate_receipt(&receipt, &v2, &profile(), now()),
            Err(SourceCaptureError::InvalidReceipt)
        ));
        // The schema string alone cannot promote a bare wire to v2.
        bare["schema"] = serde_json::json!(SOURCE_RECEIPT_SCHEMA_V2);
        let receipt = parse(&bare).unwrap();
        assert!(validate_receipt(&receipt, &v2, &profile(), now()).is_err());
        // A v1 intent answered with a span block is refused.
        let receipt = parse(&v2_wire(&v1)).unwrap();
        assert!(validate_receipt(&receipt, &v1, &profile(), now()).is_err());
        // A partial span block is neither wire.
        let mut partial = v2_wire(&v2);
        partial.as_object_mut().unwrap().remove("span_unknown_count");
        assert!(parse(&partial).is_err());
        // Unknown keys stay refused on both wires.
        let mut extra = v2_wire(&v2);
        extra["span_extra"] = serde_json::json!(1);
        assert!(parse(&extra).is_err());
    }

    #[test]
    fn span_block_invariants_mirror_the_server() {
        let v2 = intent(SourceSpanArming::DEFAULT);
        let check = |mutate: fn(&mut serde_json::Value)| {
            let mut wire = v2_wire(&v2);
            mutate(&mut wire);
            let receipt = parse(&wire).unwrap();
            validate_receipt(&receipt, &v2, &profile(), now())
        };
        assert!(check(|_| {}).is_ok());
        // Counts must fill a contiguous window exactly.
        assert!(check(|w| w["span_content_count"] = serde_json::json!(7)).is_err());
        // A gapped window has strictly fewer rows than its width.
        assert!(check(|w| w["span_contiguous"] = serde_json::json!(false)).is_err());
        assert!(check(|w| {
            w["span_contiguous"] = serde_json::json!(false);
            w["span_window_end_position"] = serde_json::json!(651);
        })
        .is_ok());
        // The aligned span cannot exceed the window rows.
        assert!(check(|w| w["retained_count"] = serde_json::json!(11)).is_err());
        // The block binds the intent's own policy and the reviewed schemas.
        assert!(check(|w| w["span_policy"] = serde_json::json!("content_tail_last64_v2")).is_err());
        assert!(check(|w| w["span_schema"] = serde_json::json!("occ3.span.v2")).is_err());
        assert!(check(|w| w["span_marker_schema"] = serde_json::json!("x")).is_err());
        assert!(check(|w| w["span_reasoning_terminator"] = serde_json::json!("eos")).is_err());
        assert!(check(|w| w["span_marker_profile"] = serde_json::json!("")).is_err());
        assert!(check(|w| w["span_thinking_count"] = serde_json::json!(129)).is_err());
        // An unattested turn end is only honest on a truncated message.
        assert!(check(|w| w["span_turn_end_attested"] = serde_json::json!(false)).is_err());
        assert!(check(|w| {
            w["span_turn_end_attested"] = serde_json::json!(false);
            w["span_message_truncated"] = serde_json::json!(true);
        })
        .is_ok());
        // The no-tool policy cannot carry tool syntax.
        let no_tool = SourceSpanArming::resolve(Some("content_tail_no_tool_v2")).unwrap();
        let no_tool_intent = intent(no_tool);
        let mut wire = v2_wire(&no_tool_intent);
        wire["span_contains_tool_call"] = serde_json::json!(true);
        let receipt = parse(&wire).unwrap();
        assert!(validate_receipt(&receipt, &no_tool_intent, &profile(), now()).is_err());
    }

    #[test]
    fn retained_content_only_receipt_requires_full_width_retention() {
        // Live canary icalib-canary-20261101T1330Z: the value's final character
        // lived in the turn's LAST row - a stop-set role token the legacy
        // last-row stop heuristic classified as the terminator, so the aligned
        // span covered 33 of the 34 content rows and the receiver reproduced
        // the 40-char value missing exactly its final character. The v3
        // retained content-only arm now requires every aligned span row to be
        // retained; the legacy v1/v2 arms keep the historical upper bound.
        let v3 = intent(SourceSpanArming::RETAINED_CONTENT_ONLY);
        let receipt = parse(&v2_wire(&v3)).unwrap();
        assert!(validate_receipt(&receipt, &v3, &profile(), now()).is_ok());
        // A retention narrower than the aligned span width is refused.
        let mut narrow = v2_wire(&v3);
        narrow["retained_count"] = serde_json::json!(7);
        narrow["byte_count"] = serde_json::json!(7u64 * 6144 * 2);
        let receipt = parse(&narrow).unwrap();
        assert!(matches!(
            validate_receipt(&receipt, &v3, &profile(), now()),
            Err(SourceCaptureError::InvalidReceipt)
        ));
        // The legacy v2 arm keeps the upper-bound semantics.
        let v2 = intent(SourceSpanArming::DEFAULT);
        let mut legacy = v2_wire(&v2);
        legacy["retained_count"] = serde_json::json!(7);
        legacy["byte_count"] = serde_json::json!(7u64 * 6144 * 2);
        let receipt = parse(&legacy).unwrap();
        assert!(validate_receipt(&receipt, &v2, &profile(), now()).is_ok());
    }
}
