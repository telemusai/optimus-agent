//! Port of packages/coding-agent/src/core/agent-messages.ts
//!
//! Type mapping notes:
//! - `CustomMessage<T>` is carried by `pi_agent_core::types::CustomAgentMessage::Custom`;
//!   `AgentSessionMessage` is therefore a typed view over that variant plus the
//!   validation helpers that the TypeScript declaration provides.
//! - `Map`/`Set` iteration order is observable in the roster output, so ordered
//!   `Vec` collections are used where the TypeScript sorts explicitly.
//! - `randomUUID()` -> `uuid::Uuid::new_v4()`.
//! - `AbortSignal` -> `tokio_util::sync::CancellationToken`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent};
use serde_json::{Map, Value};

use crate::core::messages::{ASYNC_BASH_COMPLETION_CUSTOM_TYPE, HEARTBEAT_PROMPT_CUSTOM_TYPE};
use crate::core::occ::contract::{AgentCommMessage, ContractError};
use crate::core::occ::dialect::{DialectCard, DialectRegistry, OccCapability, CODEBOOK_PREFIX_LEN};
use crate::core::occ::selection::{
    select_with_policy, DialectEvaluationRecord, SelectionContext, SelectionDecision,
    SelectionPolicy, SelectionReason,
};
use crate::core::occ::occ1::{Occ1Message, OCC1_HEADER};
use crate::core::occ::occ2::{self, OCC2_FALLBACK_SENTINEL, OCC2_HEADER};

/// Port of `agentFamilyRelationship`'s use of `canonicalSessionPath`: the pure
/// path canonicalisation from core/session-lease.ts, kept private here so this
/// module does not depend on another slice's file.
fn canonical_session_path(session_path: &str) -> String {
    let resolved = std::path::Path::new(session_path);
    let resolved = if resolved.is_absolute() {
        resolved.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(resolved),
            Err(_) => resolved.to_path_buf(),
        }
    };
    if let Ok(real) = std::fs::canonicalize(&resolved) {
        return real.to_string_lossy().to_string();
    }
    match resolved
        .parent()
        .and_then(|parent| std::fs::canonicalize(parent).ok())
    {
        Some(real_parent) => real_parent
            .join(resolved.file_name().unwrap_or_default())
            .to_string_lossy()
            .to_string(),
        None => resolved.to_string_lossy().to_string(),
    }
}

pub const AGENT_MESSAGE_CUSTOM_TYPE: &str = "agent_message";
pub const AGENT_MESSAGE_SKILL_NAME: &str = "agent-message";
pub const AGENT_MESSAGE_IMPORT_NAME: &str = "agent_message";
pub const AGENT_MESSAGE_SOURCE: &str = "agent_message";
pub const AGENT_MESSAGE_RECEIVED_PREVIEW_LABEL: &str = "Agent message received";
pub const DEFAULT_AGENT_MESSAGE_MAX_CHARS: usize = 16_384;
pub const DEFAULT_AGENT_MESSAGE_MAX_PENDING_PER_SESSION: usize = 20;
pub const DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY: f64 = 3.0;
pub const DEFAULT_AGENT_MESSAGE_RATE_LIMIT_REFILL_MS: f64 = 1000.0;

/// Legacy daemon wire input accepted and ignored for compatibility.
pub type AgentSessionMessageDeliveryMode = String;
pub type AgentSessionMessageDeliveryStatus = String;
pub type AgentSessionMessageRuntimeKind = String;
pub type AgentFamilyStatus = String;
pub type AgentFamilyRelationship = String;

pub const DELIVERY_MODE_AUTO: &str = "auto";
pub const DELIVERY_MODE_STEER: &str = "steer";
pub const DELIVERY_MODE_FOLLOW_UP: &str = "follow_up";

pub const DELIVERY_STATUS_DELIVERED: &str = "delivered";
pub const DELIVERY_STATUS_QUEUED: &str = "queued";

pub const RUNTIME_KIND_TOP_LEVEL: &str = "top-level";
pub const RUNTIME_KIND_SUBAGENT: &str = "subagent";

pub const FAMILY_STATUS_RUNNING: &str = "running";
pub const FAMILY_STATUS_IDLE: &str = "idle";
pub const FAMILY_STATUS_INACTIVE: &str = "inactive";

pub const FAMILY_RELATIONSHIP_PARENT: &str = "parent";
pub const FAMILY_RELATIONSHIP_SIBLING: &str = "sibling";
pub const FAMILY_RELATIONSHIP_CHILD: &str = "child";

pub const AGENT_FAMILY_REACH_ERROR: &str =
    "Agent reach is limited to parent, siblings, and children";

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageEndpoint {
    pub active_session_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<AgentSessionMessageRuntimeKind>,
}

/// `AgentSessionMessageSender extends Partial<AgentSessionMessageEndpoint>`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageSender {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<AgentSessionMessageRuntimeKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

pub type AgentMessageDirection = String;

pub const AGENT_MESSAGE_DIRECTION_RECEIVED: &str = "received";
pub const AGENT_MESSAGE_DIRECTION_SENT: &str = "sent";

fn trimmed(value: Option<&String>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Format the directional role/name segment shared by received and sent agent-message UI.
pub fn format_agent_message_participant(
    direction: &AgentMessageDirection,
    role: Option<&AgentFamilyRelationship>,
    endpoint: Option<&AgentSessionMessageSender>,
) -> String {
    let empty = AgentSessionMessageSender::default();
    let normalized_endpoint = endpoint.unwrap_or(&empty);
    let name_or_id = trimmed(normalized_endpoint.session_name.as_ref())
        .or_else(|| trimmed(normalized_endpoint.active_session_id.as_ref()))
        .or_else(|| trimmed(normalized_endpoint.client_id.as_ref()))
        .or_else(|| trimmed(normalized_endpoint.session_id.as_ref()))
        .unwrap_or("unknown");
    let participant = match role {
        Some(role) => format!("{role} {name_or_id}"),
        None => name_or_id.to_string(),
    };
    let direction_word = if direction == AGENT_MESSAGE_DIRECTION_RECEIVED {
        "from"
    } else {
        "to"
    };
    format!("{direction_word} {participant}")
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageAgentSummary {
    pub active_session_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<AgentSessionMessageRuntimeKind>,
    pub cwd: String,
    pub is_streaming: bool,
    pub unfinished_action_count: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_child_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_depth: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<AgentFamilyStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_child_registry_status: Option<String>,
    /// OCC-2 dialect capability of this agent (OCC-2 integration design,
    /// section 4). Absent means the agent reported nothing: negotiation
    /// then yields no dialect and sends stay OCC-1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occ_capability: Option<OccCapability>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageListResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<AgentSessionMessageEndpoint>,
    pub agents: Vec<AgentSessionMessageAgentSummary>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentFamilyCatalogEntry {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub depth: f64,
    pub status: AgentFamilyStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replied_since_task: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_path: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentFamilyRosterEntry {
    pub relationship: AgentFamilyRelationship,
    pub name: String,
    pub id: String,
    pub depth: f64,
    pub status: AgentFamilyStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replied_since_task: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentFamilyRosterCurrent {
    pub name: String,
    pub id: String,
    pub depth: f64,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentFamilyRosterResult {
    pub current: AgentFamilyRosterCurrent,
    pub entries: Vec<AgentFamilyRosterEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionNameScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
    pub depth: f64,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionNameAvailabilityInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
    pub depth: f64,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_session_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessagePayload {
    pub id: String,
    pub source: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<AgentSessionMessageSender>,
    /// Sender relationship from the receiver's point of view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_relationship: Option<AgentFamilyRelationship>,
    pub target: AgentSessionMessageEndpoint,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageDetails {
    pub id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<AgentSessionMessageSender>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_relationship: Option<AgentFamilyRelationship>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<AgentSessionMessageEndpoint>,
    /// How the OCC-2 delivery consumed this body, recorded once at
    /// delivery so telemetry never re-decodes. Absent on pre-OCC-2
    /// records and plain bodies without an OCC/2 header involvement.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub occ2_delivery: Option<Occ2DeliveryDetails>,
}

/// The OCC-3 native delivery marker, recorded under `details.occ3Delivery` of one
/// agent-session record: the only evidence that a delivered native handoff reached
/// the receiver's completed-use receipt.
/// Content-free by construction: a bounded method id, the runtime generation,
/// and two digests. No handle, no receipt blob, no tensor bytes, no body text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Occ3DeliveryDetails {
    /// [`OCC3_DELIVERY_PENDING`] or [`OCC3_DELIVERY_LATENT_NATIVE`].
    pub kind: String,
    /// The connector's advertised native method id.
    pub method_id: String,
    /// Monotonic runtime generation the marker was minted against.
    pub runtime_generation: u64,
    /// Digest of the receiver runtime identity (never the identity itself).
    pub runtime_identity_sha256: String,
    /// Digest of the completed-use receipt; present only once consumed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub receipt_sha256: Option<String>,
    /// True only for a receipt that proved completed use.
    #[serde(default)]
    pub completed_use: bool,
}

/// Recorded before the receiver's completed-use receipt exists: no pulse.
pub const OCC3_DELIVERY_PENDING: &str = "pending";
/// Recorded once, and only once, the receiver proved completed use.
pub const OCC3_DELIVERY_LATENT_NATIVE: &str = "latent_native";

const MAX_OCC3_NATIVE_ID_CHARS: usize = 128;

fn bounded_occ3_native_id(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_OCC3_NATIVE_ID_CHARS
        && !value.chars().any(char::is_control)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

impl Occ3DeliveryDetails {
    fn marker(
        kind: &str,
        method_id: &str,
        runtime_generation: u64,
        runtime_identity_sha256: &str,
        receipt_sha256: Option<&str>,
    ) -> Result<Self, String> {
        if !bounded_occ3_native_id(method_id)
            || !is_sha256_hex(runtime_identity_sha256)
            || receipt_sha256.is_some_and(|digest| !is_sha256_hex(digest))
        {
            return Err("OCC-3 delivery marker rejected".into());
        }
        Ok(Self {
            kind: kind.to_string(),
            method_id: method_id.to_string(),
            runtime_generation,
            runtime_identity_sha256: runtime_identity_sha256.to_string(),
            receipt_sha256: receipt_sha256.map(str::to_string),
            completed_use: kind == OCC3_DELIVERY_LATENT_NATIVE,
        })
    }

    /// Admitted but not consumed: recorded so the control-only body is never
    /// classified as text, and never projected as a pulse.
    pub fn pending(
        method_id: &str,
        runtime_generation: u64,
        runtime_identity_sha256: &str,
    ) -> Result<Self, String> {
        Self::marker(
            OCC3_DELIVERY_PENDING,
            method_id,
            runtime_generation,
            runtime_identity_sha256,
            None,
        )
    }

    /// The receiver's completed-use receipt exists for this exact message id.
    pub fn latent_native(
        method_id: &str,
        runtime_generation: u64,
        runtime_identity_sha256: &str,
        receipt_sha256: &str,
    ) -> Result<Self, String> {
        Self::marker(
            OCC3_DELIVERY_LATENT_NATIVE,
            method_id,
            runtime_generation,
            runtime_identity_sha256,
            Some(receipt_sha256),
        )
    }

    /// The recorded marker of one delivery, when the record carries one.
    pub fn from_details(details: &Value) -> Option<Self> {
        serde_json::from_value(details.get("occ3Delivery")?.clone()).ok()
    }

    /// True only for a marker that names a completed consumption. The label, the
    /// completed-use bit and the receipt digest must all agree, so a record that
    /// merely claims `latent_native` in text is never admitted.
    pub fn is_completed_native_latent(&self) -> bool {
        self.kind == OCC3_DELIVERY_LATENT_NATIVE
            && self.completed_use
            && bounded_occ3_native_id(&self.method_id)
            && is_sha256_hex(&self.runtime_identity_sha256)
            && self
                .receipt_sha256
                .as_deref()
                .is_some_and(is_sha256_hex)
    }
}

/// Writes the marker onto the already-recorded delivery `message_id`, keeping the
/// same record, message id and content. Returns false when no record matches.
fn write_occ3_delivery(details: &mut Value, marker: &Occ3DeliveryDetails) -> bool {
    let Some(object) = details.as_object_mut() else {
        return false;
    };
    let Ok(value) = serde_json::to_value(marker) else {
        return false;
    };
    object.insert("occ3Delivery".to_string(), value);
    true
}

/// Stamps one not-yet-delivered agent-session record.
pub fn stamp_occ3_delivery_record(
    record: &mut CustomAgentMessage,
    marker: &Occ3DeliveryDetails,
) -> Result<(), String> {
    let CustomAgentMessage::Custom {
        custom_type,
        details,
        ..
    } = record
    else {
        return Err("OCC-3 delivery marker requires a custom message".into());
    };
    if custom_type != AGENT_MESSAGE_CUSTOM_TYPE {
        return Err("OCC-3 delivery marker requires an agent message".into());
    }
    let details = details.get_or_insert_with(|| Value::Object(Map::new()));
    write_occ3_delivery(details, marker)
        .then_some(())
        .ok_or_else(|| "OCC-3 delivery marker rejected".to_string())
}

/// Rewrites one already-admitted delivery in the live transcript. The record, its
/// message id and its content are unchanged, so a repeated stamp is idempotent and
/// the activity projection still de-duplicates by that id.
pub fn stamp_occ3_delivery(
    messages: &mut [AgentMessage],
    message_id: &str,
    marker: &Occ3DeliveryDetails,
) -> bool {
    for message in messages.iter_mut().rev() {
        let AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type,
            details: Some(details),
            ..
        }) = message
        else {
            continue;
        };
        if custom_type != AGENT_MESSAGE_CUSTOM_TYPE
            || details.get("id").and_then(Value::as_str) != Some(message_id)
        {
            continue;
        }
        return write_occ3_delivery(details, marker);
    }
    false
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageReceipt {
    pub id: String,
    pub source: String,
    pub target: AgentSessionMessageEndpoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<AgentSessionMessageSender>,
    pub message: String,
    // Not named "status": the kernel host bridge envelope reserves that key.
    pub delivery_status: AgentSessionMessageDeliveryStatus,
    /// Present only for delivered messages: when the target context received it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivered_at: Option<String>,
    /// Present only for queued messages: when it was placed behind current work.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queued_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery_mode: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageSendInput {
    pub target: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receiver_role: Option<AgentFamilyRelationship>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionMessageSafetyStatus {
    pub paused: bool,
    pub max_message_chars: f64,
    pub max_pending_per_session: f64,
    pub rate_limit_capacity: f64,
    pub rate_limit_refill_ms: f64,
}

/// Structural reservation key for sibling-scoped session names. JSON encoding keeps
/// parent paths and names containing delimiter characters from colliding into one key;
/// the worker- and supervisor-side reservation maps must never diverge in encoding.
pub fn session_name_reservation_key(input: &AgentSessionNameScope, name: &str) -> String {
    let (parent_type, parent_value) = if input.depth == 0.0 {
        ("root", String::new())
    } else if let Some(parent_session_path) = &input.parent_session_path {
        ("path", canonical_session_path(parent_session_path))
    } else if let Some(parent_session_id) = &input.parent_session_id {
        ("id", parent_session_id.clone())
    } else {
        ("root", String::new())
    };
    // Reservation keys compare serialized bytes across hosts. JavaScript emits
    // integral depths without a decimal suffix (including negative zero).
    let depth = if input.depth.fract() == 0.0 && input.depth.abs() <= 9_007_199_254_740_991.0 {
        serde_json::json!(input.depth as i64)
    } else {
        serde_json::json!(input.depth)
    };
    serde_json::to_string(&serde_json::json!([
        depth,
        parent_type,
        parent_value,
        name
    ]))
    .unwrap_or_default()
}

pub fn format_agent_session_name_unavailable(name: &str, depth: f64) -> String {
    format!(
        "Agent name \"{name}\" is unavailable: an agent of that name already exists at depth {depth} under this parent"
    )
}

pub fn assert_agent_session_name_available(
    catalog: &[AgentFamilyCatalogEntry],
    input: &AgentSessionNameAvailabilityInput,
) -> Result<(), String> {
    let input_scope = AgentSessionNameScope {
        parent_session_id: input.parent_session_id.clone(),
        parent_session_path: input.parent_session_path.clone(),
        depth: input.depth,
    };
    let conflict = catalog.iter().any(|entry| {
        Some(entry.id.as_str()) != input.ignore_session_id.as_deref()
            && entry.name.as_deref() == Some(input.name.as_str())
            && entry.depth == input.depth
            && same_agent_session_name_parent(&catalog_entry_scope(entry), &input_scope, catalog)
    });
    if conflict {
        return Err(format_agent_session_name_unavailable(
            &input.name,
            input.depth,
        ));
    }
    Ok(())
}

fn catalog_entry_scope(entry: &AgentFamilyCatalogEntry) -> AgentSessionNameScope {
    AgentSessionNameScope {
        parent_session_id: entry.parent_session_id.clone(),
        parent_session_path: entry.parent_session_path.clone(),
        depth: entry.depth,
    }
}

pub fn build_agent_family_roster(
    current: &AgentFamilyCatalogEntry,
    catalog: &[AgentFamilyCatalogEntry],
) -> AgentFamilyRosterResult {
    let parent = catalog
        .iter()
        .find(|entry| is_agent_family_parent(entry, current));
    let mut siblings: Vec<&AgentFamilyCatalogEntry> = catalog
        .iter()
        .filter(|entry| {
            entry.id != current.id
                && entry.depth == current.depth
                && same_agent_family_parent(
                    &catalog_entry_scope(entry),
                    &catalog_entry_scope(current),
                    catalog,
                )
        })
        .collect();
    let mut children: Vec<&AgentFamilyCatalogEntry> = catalog
        .iter()
        .filter(|entry| {
            entry.depth == current.depth + 1.0 && is_agent_family_parent(current, entry)
        })
        .collect();

    let name_of =
        |entry: &AgentFamilyCatalogEntry| entry.name.clone().unwrap_or_else(|| entry.id.clone());
    // `localeCompare` on the default locale orders plain identifiers like Rust's
    // byte comparison for the ASCII session names this catalogue contains.
    siblings.sort_by_key(|entry| name_of(entry));
    children.sort_by_key(|entry| name_of(entry));

    let row = |relationship: &str, entry: &AgentFamilyCatalogEntry| AgentFamilyRosterEntry {
        relationship: relationship.to_string(),
        name: name_of(entry),
        id: entry.id.clone(),
        depth: entry.depth,
        status: entry.status.clone(),
        replied_since_task: if relationship == FAMILY_RELATIONSHIP_CHILD {
            entry.replied_since_task
        } else {
            None
        },
    };

    let mut entries: Vec<AgentFamilyRosterEntry> = Vec::new();
    if let Some(parent) = parent {
        entries.push(row(FAMILY_RELATIONSHIP_PARENT, parent));
    }
    for entry in siblings {
        entries.push(row(FAMILY_RELATIONSHIP_SIBLING, entry));
    }
    for entry in children {
        entries.push(row(FAMILY_RELATIONSHIP_CHILD, entry));
    }

    AgentFamilyRosterResult {
        current: AgentFamilyRosterCurrent {
            name: name_of(current),
            id: current.id.clone(),
            depth: current.depth,
        },
        entries,
    }
}

fn same_agent_session_name_parent(
    left: &AgentSessionNameScope,
    right: &AgentSessionNameScope,
    catalog: &[AgentFamilyCatalogEntry],
) -> bool {
    if left.depth == 0.0 && right.depth == 0.0 {
        return true;
    }
    same_agent_family_parent(left, right, catalog)
}

fn same_agent_family_parent(
    left: &AgentSessionNameScope,
    right: &AgentSessionNameScope,
    catalog: &[AgentFamilyCatalogEntry],
) -> bool {
    if left.parent_session_path.is_some() && left.parent_session_path == right.parent_session_path {
        return true;
    }
    if left.parent_session_id.is_some() && left.parent_session_id == right.parent_session_id {
        return true;
    }
    let has_catalog_parent_pair =
        |parent_session_id: Option<&String>, parent_session_path: Option<&String>| {
            let (Some(parent_session_id), Some(parent_session_path)) =
                (parent_session_id, parent_session_path)
            else {
                return false;
            };
            catalog.iter().any(|entry| {
                (Some(entry.id.as_str()) == Some(parent_session_id.as_str())
                    && entry.session_path.as_deref() == Some(parent_session_path.as_str()))
                    || (entry.parent_session_id.as_deref() == Some(parent_session_id.as_str())
                        && entry.parent_session_path.as_deref()
                            == Some(parent_session_path.as_str()))
            })
        };
    if has_catalog_parent_pair(
        left.parent_session_id.as_ref(),
        right.parent_session_path.as_ref(),
    ) || has_catalog_parent_pair(
        right.parent_session_id.as_ref(),
        left.parent_session_path.as_ref(),
    ) {
        return true;
    }
    if left.depth == 0.0
        && right.depth == 0.0
        && left.parent_session_path.is_none()
        && right.parent_session_path.is_none()
        && left.parent_session_id.is_none()
        && right.parent_session_id.is_none()
    {
        return true;
    }
    // Unresolved mixed identifiers stay unrelated to avoid false name conflicts across families.
    false
}

fn is_agent_family_parent(
    parent: &AgentFamilyCatalogEntry,
    child: &AgentFamilyCatalogEntry,
) -> bool {
    (child.parent_session_path.is_some() && child.parent_session_path == parent.session_path)
        || (child.parent_session_id.is_some() && child.parent_session_id == Some(parent.id.clone()))
}

/// Pure nuclear-family policy over persisted parent-edge snapshots.
pub fn agent_family_relationship(
    current: &AgentFamilyCatalogEntry,
    target: &AgentFamilyCatalogEntry,
) -> Option<AgentFamilyRelationship> {
    if current.id == target.id {
        return None;
    }
    if is_agent_family_parent(target, current) {
        return Some(FAMILY_RELATIONSHIP_PARENT.to_string());
    }
    if is_agent_family_parent(current, target) {
        return Some(FAMILY_RELATIONSHIP_CHILD.to_string());
    }
    if current.depth == target.depth {
        let pair = [current.clone(), target.clone()];
        if same_agent_family_parent(
            &catalog_entry_scope(current),
            &catalog_entry_scope(target),
            &pair,
        ) {
            return Some(FAMILY_RELATIONSHIP_SIBLING.to_string());
        }
    }
    None
}

pub fn assert_agent_family_reach(
    current: &AgentFamilyCatalogEntry,
    target: &AgentFamilyCatalogEntry,
) -> Result<AgentFamilyRelationship, String> {
    match agent_family_relationship(current, target) {
        Some(relationship) => Ok(relationship),
        None => Err(AGENT_FAMILY_REACH_ERROR.to_string()),
    }
}

pub fn create_agent_session_message_id() -> String {
    format!("agentmsg_{}", uuid::Uuid::new_v4())
}

pub fn normalize_agent_session_message(message: &str, max_chars: usize) -> Result<String, String> {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return Err("Agent session message cannot be empty".to_string());
    }
    let length = trimmed.chars().count();
    if length > max_chars {
        return Err(format!(
            "Agent session message is too long: {length} chars exceeds {max_chars}"
        ));
    }
    Ok(trimmed.to_string())
}

pub fn assert_direct_agent_message_target(target: &str) -> Result<String, String> {
    let normalized = target.trim();
    if normalized.is_empty() {
        return Err("Agent message target cannot be empty".to_string());
    }
    let lower = normalized.to_lowercase();
    if normalized == "*" || lower == "all" || lower == "broadcast" {
        return Err("Broadcast agent messaging is not supported".to_string());
    }
    Ok(normalized.to_string())
}

pub fn assert_agent_message_queue_capacity(
    unfinished_action_count: f64,
    max_pending: usize,
) -> Result<(), String> {
    if unfinished_action_count >= max_pending as f64 {
        return Err(format!(
            "Target session has too many pending messages: {unfinished_action_count} unfinished, limit is {max_pending}"
        ));
    }
    Ok(())
}

/// `parseAgentSessionMessagePromptId`: `\n`-split lines, with the optional
/// `[from ...]` relationship line shifting the fixed offsets by one.
pub fn parse_agent_session_message_prompt_id(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let offset = if lines
        .first()
        .map(|line| line.starts_with("[from "))
        .unwrap_or(false)
    {
        1
    } else {
        0
    };
    if lines.get(offset).copied() != Some("Agent-to-agent message received.") {
        return None;
    }
    if lines.get(offset + 1).copied() != Some(&format!("Source: {AGENT_MESSAGE_SOURCE}")) {
        return None;
    }
    let to_line_index = if lines
        .get(offset + 2)
        .map(|line| line.starts_with("From: "))
        .unwrap_or(false)
    {
        offset + 3
    } else {
        offset + 2
    };
    if !lines
        .get(to_line_index)
        .map(|line| line.starts_with("To: "))
        .unwrap_or(false)
    {
        return None;
    }
    let candidate = lines.get(to_line_index + 1).copied().unwrap_or("");
    let id = candidate.strip_prefix("Message id: ")?;
    if !id.starts_with("agentmsg_") || id.contains('\n') {
        return None;
    }
    Some(id.to_string())
}

pub fn is_agent_session_message_prompt(text: &str) -> bool {
    parse_agent_session_message_prompt_id(text).is_some()
}

// ---------------------------------------------------------------------------
// OCC-2 payload encoding seam (research/integration/
// OCC2-INTEGRATION-DESIGN.md, sections 1-4). OCC levels are interchangeable
// encodings of one typed payload: this seam substitutes only the single
// `AgentSessionMessagePayload.message` string, so sender identity,
// queueing, receipts, rate limits and family scoping are untouched. With
// the feature disabled (the default) every function here is the identity
// on the body and prompt output stays byte-identical.
// ---------------------------------------------------------------------------

/// Send policy for the OCC-2 wrap seam. `enabled: false` (the default)
/// makes [`wrap_outgoing_body`] the identity function.
#[derive(Debug, Clone, PartialEq)]
pub struct Occ2SendPolicy {
    /// Feature flag (`agentCommunication.occ2Enabled`), default OFF.
    pub enabled: bool,
    /// Negotiated (dialect id, version) for this target; `None` keeps OCC-1.
    pub dialect: Option<(String, u32)>,
    /// Dialect cards available for encoding.
    pub registry: DialectRegistry,
    /// Transport bound mirroring `normalize_agent_session_message`: a dual
    /// payload that would exceed it is never sent (R1: no truncation).
    pub max_message_chars: usize,
}

impl Occ2SendPolicy {
    /// The conservative default: OCC-1 only, byte-identical behavior.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            dialect: None,
            registry: DialectRegistry::new(),
            max_message_chars: DEFAULT_AGENT_MESSAGE_MAX_CHARS,
        }
    }

    /// An enabled policy for one negotiated dialect.
    pub fn for_dialect(
        registry: DialectRegistry,
        dialect: (String, u32),
        max_message_chars: usize,
    ) -> Self {
        Self {
            enabled: true,
            dialect: Some(dialect),
            registry,
            max_message_chars,
        }
    }
}

/// Negotiates the OCC-2 dialect for one target (design section 4). A
/// target that reports no capability, a peer with no shared dialect, or a
/// dialect that is not registered yields `None` and the send stays OCC-1.
/// Cards must be evaluated with round-trip fidelity 1.0 unless
/// `allow_unevaluated` is set (the dev flag): this is the tier-2 gate of
/// the fallback policy. Model-pair and task-family matching on evolved
/// cards is a daemon-side follow-up.
pub fn select_occ2_dialect(
    local: &OccCapability,
    peer: Option<&OccCapability>,
    registry: &DialectRegistry,
    allow_unevaluated: bool,
) -> Option<(String, u32)> {
    let peer = peer?;
    let (dialect_id, version) = local.negotiate(peer)?;
    let card = registry.get(&dialect_id, version)?;
    let evaluated = card.metrics.samples > 0 && card.metrics.round_trip_fidelity >= 1.0;
    if evaluated || allow_unevaluated {
        Some((dialect_id, version))
    } else {
        None
    }
}

/// Wraps one outgoing body (design section 2). Called at both send seams
/// after the target is known and before the payload is built:
///
/// 1. Disabled or no negotiated dialect: identity.
/// 2. The body must be a complete OCC-1 contract message; plain English,
///    deltas (excluded in v1, OQ4) and malformed OCC-1 pass through.
/// 3. `occ2::encode_dual` emits the symbolic section, the
///    [`OCC2_FALLBACK_SENTINEL`] line and the full OCC-1 text.
/// 4. Any encode error, or a dual payload that would exceed the transport
///    bound, returns the original body (R1: the message is never lost,
///    altered or truncated).
pub fn wrap_outgoing_body(body: &str, policy: &Occ2SendPolicy) -> String {
    if !policy.enabled {
        return body.to_string();
    }
    let Some((dialect_id, version)) = &policy.dialect else {
        return body.to_string();
    };
    if !body.starts_with(OCC1_HEADER) {
        return body.to_string();
    }
    let Ok(occ1) = Occ1Message::decode(body) else {
        return body.to_string();
    };
    let Ok(message) = AgentCommMessage::from_occ1(&occ1) else {
        return body.to_string();
    };
    let Some(card) = policy.registry.get(dialect_id, *version) else {
        return body.to_string();
    };
    let Ok(dual) = occ2::encode_dual(&message, card) else {
        return body.to_string();
    };
    if dual.trim().chars().count() > policy.max_message_chars {
        return body.to_string();
    }
    dual
}

/// Send policy for the OCC-3 latent seam. Mirrors [`Occ2SendPolicy`]:
/// `enabled: false` (the default) is the identity. When enabled and a
/// captured handle is present, the body is replaced by the OCC-3
/// envelope whose control summary is the (already OCC-encoded) body.
///
/// Capture happens BEFORE this seam (the daemon send handler is async
/// and owns connector I/O); this seam stays pure: it only encodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Occ3SendPolicy {
    /// Feature flag (session `occ3Enabled`), default OFF.
    pub enabled: bool,
    /// A handle captured for this specific message, plus the negotiated
    /// connector/runtime identity. `None` keeps the prior encoding.
    pub handle: Option<crate::core::occ::occ3::types::Occ3HandleRef>,
    /// Connector id for the envelope.
    pub connector_id: String,
    /// Runtime identity string for the envelope.
    pub runtime_id: String,
    /// Negotiated latent method.
    pub method: crate::core::occ::occ3::types::Occ3Method,
    /// Receiver session id (bound into the envelope).
    pub receiver_session_id: String,
    /// Transport bound (R1: never truncate; refuse instead).
    pub max_message_chars: usize,
}

impl Occ3SendPolicy {
    /// Conservative default: no latent substitution, byte-identical.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            handle: None,
            connector_id: String::new(),
            runtime_id: String::new(),
            method: crate::core::occ::occ3::types::Occ3Method::KvHandoff,
            receiver_session_id: String::new(),
            max_message_chars: DEFAULT_AGENT_MESSAGE_MAX_CHARS,
        }
    }
}

/// Apply the OCC-3 seam AFTER the OCC-2 seam: an eligible latent message
/// wraps the OCC-1/OCC-2-encoded control summary inside the envelope.
/// Any error, missing handle or budget breach returns the body
/// unchanged (R1) so the message still delivers.
pub fn wrap_outgoing_body_occ3(body: &str, _policy: &Occ3SendPolicy) -> String {
    // Legacy kernel sends have no authenticated native tool/source provenance.
    // Never fabricate pending IDs or copy the source body into a successful
    // latent envelope. Actual native handoff uses PreparedNativeHandoff.
    body.to_string()
}

/// Splits a dual OCC-2 payload at the fallback sentinel, mirroring the
/// codec's private splitter: (symbolic section, OCC-1 section). Returns
/// `None` when the body carries no sentinel line.
fn split_occ2_dual_body(body: &str) -> Option<(&str, &str)> {
    let mut offset = 0usize;
    for line in body.split('\n') {
        let line_start = offset;
        let line_end = line_start + line.len();
        offset = line_end + 1;
        let bare = line.strip_suffix('\r').unwrap_or(line);
        if bare == OCC2_FALLBACK_SENTINEL {
            let fallback_start = (line_end + 1).min(body.len());
            return Some((&body[..line_start], &body[fallback_start..]));
        }
    }
    None
}

/// Validate before exposing symbolic text; retain the durable dual payload.
///
/// Conservative always-teach render: the listener's context is unknown, so
/// every strict-decode body carries the full codebook legend. The spawn seam
/// relies on this (the child's context starts empty), as do callers without
/// access to the listener's message list. Delivery uses
/// [`agent_session_message_prompt_body_with_codebooks`].
pub fn agent_session_message_prompt_body(message: &str) -> std::borrow::Cow<'_, str> {
    agent_session_message_prompt_body_with_registry_and_codebooks(
        message,
        &DialectRegistry::with_sym_v1(),
        &Occ2CodebooksPresent::new(),
    )
}

// ---------------------------------------------------------------------------
// OCC-2 codebook amortization (efficiency campaign): the full codebook
// legend is a fixed ~600-char block, and injecting it into every delivered
// message grew total prompt tokens +32% on SILO-BENCH while the wire shrank.
// The render below injects the full legend only when the listener's context
// does not already carry the same codebook; later bodies carry a one-line
// reference.
//
// Identity = dialect id + version + the leading `CODEBOOK_PREFIX_LEN` hex
// of the codebook sha256 (the identity the OCC-2 wire header already
// carries), so a re-tuned card re-injects. Presence is derived from the
// listener's live message list, never from a volatile flag: compaction,
// restart and context rebuilds are correct by construction, because
// whatever the scan still finds is what the model still sees.
// ---------------------------------------------------------------------------

/// Identity of one codebook as injected into a listener context.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Occ2CodebookId {
    /// Dialect id (e.g. `sym-v1`).
    pub dialect: String,
    /// Dialect version.
    pub version: u32,
    /// Leading `CODEBOOK_PREFIX_LEN` hex characters of the codebook sha256.
    pub sha256_prefix: String,
}

impl Occ2CodebookId {
    /// The identity of a registered dialect card.
    pub fn of_card(card: &DialectCard) -> Self {
        Self {
            dialect: card.dialect_id.clone(),
            version: card.version,
            sha256_prefix: card.sha256.chars().take(CODEBOOK_PREFIX_LEN).collect(),
        }
    }
}

/// Codebook identities already present in a listener's context.
pub type Occ2CodebooksPresent = BTreeSet<Occ2CodebookId>;

/// The full codebook legend for one dialect card: identity plus the complete
/// slot and subfield mapping and the reading rules. This is the only place
/// the legend text is composed; the spawn seam and the delivery path share
/// it, so both record the same identity scheme.
pub fn occ2_codebook_legend(card: &DialectCard) -> String {
    let identity = Occ2CodebookId::of_card(card);
    let slots = card.symbol_table.slots.iter().map(|(name, symbol)| format!("{symbol}:{name}")).collect::<Vec<_>>().join(", ");
    let subfields = card.symbol_table.subfields.iter().map(|(name, symbol)| format!("{name}:{symbol}")).collect::<Vec<_>>().join(", ");
    format!(
        "OCC-2 {dialect}@{version} codebook (sha256:{prefix}): {slots}. Indexed fields use 1-based indices; suffixes: {subfields}. Values follow '='; evidence references are comma-separated. Reply using OCC-1; transport handles encoding.",
        dialect = identity.dialect,
        version = identity.version,
        prefix = identity.sha256_prefix,
    )
}

/// The one-line reference to a codebook already present in the context.
/// Session-scoped wording ("in this conversation") keeps the line true
/// even when a later re-injection follows a compaction that removed the
/// original legend.
pub fn occ2_codebook_reference(identity: &Occ2CodebookId) -> String {
    format!(
        "OCC-2 {dialect}@{version} codebook (sha256:{prefix}) previously provided in this conversation; decode rules unchanged.",
        dialect = identity.dialect,
        version = identity.version,
        prefix = identity.sha256_prefix,
    )
}

/// Parses the identity from a rendered codebook legend line (exactly what
/// [`occ2_codebook_legend`] emits). Reference lines do not parse: only a
/// full legend counts as a codebook present in the context, so compaction
/// that removes the legend-bearing record forces a re-injection instead of
/// leaving a dangling reference as the only evidence.
pub fn parse_occ2_codebook_legend(line: &str) -> Option<Occ2CodebookId> {
    let rest = line.strip_prefix("OCC-2 ")?;
    let (identity, tail) = rest.split_once(" codebook (sha256:")?;
    let (dialect, version) = identity.split_once('@')?;
    let version: u32 = version.parse().ok()?;
    let prefix: String = tail
        .chars()
        .take_while(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        .collect();
    if prefix.len() != CODEBOOK_PREFIX_LEN || !tail[prefix.len()..].starts_with("):") {
        return None;
    }
    let valid_dialect = !dialect.is_empty()
        && dialect.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && dialect
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !valid_dialect {
        return None;
    }
    Some(Occ2CodebookId {
        dialect: dialect.to_string(),
        version,
        sha256_prefix: prefix,
    })
}

/// The codebook identities present in a listener's message list: every
/// stored agent-message record whose rendered content carries a full
/// legend. Delivered messages and RLM spawn prompts both record their
/// content as `agent_message` custom messages, so one scan covers both
/// seams, and the spawn codebook suppresses the first delivered message's
/// legend (no double injection).
pub fn occ2_codebooks_present(messages: &[AgentMessage]) -> Occ2CodebooksPresent {
    let mut present = Occ2CodebooksPresent::new();
    for message in messages {
        let AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type,
            content: CustomMessageContent::Text(text),
            ..
        }) = message
        else {
            continue;
        };
        if custom_type != AGENT_MESSAGE_CUSTOM_TYPE {
            continue;
        }
        for line in text.lines() {
            if let Some(identity) = parse_occ2_codebook_legend(line) {
                present.insert(identity);
            }
        }
    }
    present
}

/// Resolves the dialect card a body advertises, from its wire header
/// (`OCC/2 D=<dialect>@<version> C=<prefix>`).
fn occ2_body_dialect_card<'a>(
    message: &str,
    registry: &'a DialectRegistry,
) -> Option<&'a DialectCard> {
    let header = message.lines().next()?;
    let rest = header.strip_prefix(OCC2_HEADER)?;
    let mut tokens = rest.split(' ').filter(|token| !token.is_empty());
    let dialect = tokens.next()?.strip_prefix("D=")?;
    let (dialect, version) = dialect.split_once('@')?;
    let version: u32 = version.parse().ok()?;
    registry.get(dialect, version)
}

/// The render core shared by every public variant. One fallback split, the
/// strict decode as the validity gate, then either the full legend (first
/// body of this identity in the listener's context) or the one-line
/// reference. Fallback bodies render the embedded OCC-1 section; anything
/// else passes through unchanged.
fn render_agent_session_message_prompt_body<'a>(
    message: &'a str,
    registry: &DialectRegistry,
    codebooks: &Occ2CodebooksPresent,
) -> std::borrow::Cow<'a, str> {
    use std::borrow::Cow;
    if !message.starts_with(OCC2_HEADER) {
        return Cow::Borrowed(message);
    }
    // One split serves both branches; the strict decode below re-splits
    // internally, but the render needs the original section bytes.
    let split = split_occ2_dual_body(message);
    if occ2::decode(message, registry).is_ok() {
        let symbolic = split
            .map(|(body, _)| body)
            .unwrap_or(message)
            .trim_end_matches('\n');
        // A strict decode resolved this exact card; the lookup cannot miss.
        let Some(card) = occ2_body_dialect_card(message, registry) else {
            return Cow::Owned(symbolic.to_string());
        };
        let identity = Occ2CodebookId::of_card(card);
        let lead = if codebooks.contains(&identity) {
            occ2_codebook_reference(&identity)
        } else {
            occ2_codebook_legend(card)
        };
        return Cow::Owned(format!("{lead}\n\n{symbolic}"));
    }
    // Fallback: consume the embedded OCC-1 section when it parses. This
    // mirrors `occ2::occ1_fallback` on the already-split sections; a body
    // without a sentinel cannot parse as OCC-1 because it starts with the
    // OCC/2 header.
    if let Some((_, fallback)) = split {
        if let Ok(occ1) = Occ1Message::decode(fallback) {
            if AgentCommMessage::from_occ1(&occ1).is_ok() {
                return Cow::Borrowed(fallback);
            }
        }
    }
    Cow::Borrowed(message)
}

/// [`agent_session_message_prompt_body`] with the codebook identities
/// already present in the listener's context (from
/// [`occ2_codebooks_present`]): the first body of an identity carries the
/// full legend, later ones the one-line reference.
pub fn agent_session_message_prompt_body_with_codebooks<'a>(
    message: &'a str,
    codebooks: &Occ2CodebooksPresent,
) -> std::borrow::Cow<'a, str> {
    agent_session_message_prompt_body_with_registry_and_codebooks(
        message,
        &DialectRegistry::with_sym_v1(),
        codebooks,
    )
}

/// The amortized render against an explicit dialect registry, so tests can
/// model evolved dialects alongside the built-in card.
pub fn agent_session_message_prompt_body_with_registry_and_codebooks<'a>(
    message: &'a str,
    registry: &DialectRegistry,
    codebooks: &Occ2CodebooksPresent,
) -> std::borrow::Cow<'a, str> {
    render_agent_session_message_prompt_body(message, registry, codebooks)
}

/// Incidents per (sender, listener) pair before OCC-2 is disabled for
/// that pair for the session (design section 3, tier 3).
pub const OCC2_STRIKE_LIMIT: usize = 3;

/// Quiet period after which a pair's strike counts decay to zero: transient
/// corruption (a mismatched build, one truncated write) must not disable a
/// pair forever. 24 hours, conservative.
pub const OCC2_STRIKE_DECAY_MS: u64 = 24 * 60 * 60 * 1000;

/// The strike pair key: speaker -> listener, scoped to one dialect
/// version, so a strike against one dialect never disables another.
pub fn occ2_pair_key(speaker: &str, listener: &str, dialect: &str, version: u32) -> String {
    format!("{speaker}->{listener}|{dialect}@{version}")
}

/// Pure strike ledger for OCC-2 decode failures, with the per-pair last
/// incident time the persisted form needs for decay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Occ2StrikeLedger {
    strikes: BTreeMap<String, usize>,
    disabled: BTreeSet<String>,
    last_incident_ms: BTreeMap<String, u64>,
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

impl Occ2StrikeLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Incidents recorded for one pair.
    pub fn incidents(&self, pair: &str) -> usize {
        self.strikes.get(pair).copied().unwrap_or(0)
    }

    /// True when the strike limit disabled OCC-2 for this pair.
    pub fn is_disabled(&self, pair: &str) -> bool {
        self.disabled.contains(pair)
    }

    /// When the pair's most recent incident was recorded.
    pub fn last_incident_ms(&self, pair: &str) -> Option<u64> {
        self.last_incident_ms.get(pair).copied()
    }

    /// Records one incident at the wall clock. Returns true when this
    /// strike disabled the pair.
    pub fn record_incident(&mut self, pair: &str) -> bool {
        self.record_incident_at(pair, unix_now_ms())
    }

    /// Records one incident at an injected time, so decay tests stay
    /// deterministic. Returns true when this strike disabled the pair.
    pub fn record_incident_at(&mut self, pair: &str, now_ms: u64) -> bool {
        let strikes = self.strikes.entry(pair.to_string()).or_insert(0);
        *strikes += 1;
        self.last_incident_ms.insert(pair.to_string(), now_ms);
        if *strikes >= OCC2_STRIKE_LIMIT {
            self.disabled.insert(pair.to_string());
            true
        } else {
            false
        }
    }

    /// Drops every pair whose last incident is older than
    /// [`OCC2_STRIKE_DECAY_MS`], re-enabling decayed pairs. Returns true
    /// when anything was dropped. Entries without a recorded time decay
    /// immediately: an unknown age is not evidence of a persistent fault.
    pub fn decay(&mut self, now_ms: u64) -> bool {
        let expired: Vec<String> = self
            .last_incident_ms
            .iter()
            .filter(|(_, last)| now_ms.saturating_sub(**last) > OCC2_STRIKE_DECAY_MS)
            .map(|(pair, _)| pair.clone())
            .collect();
        let timed_out: Vec<String> = self
            .last_incident_ms
            .keys()
            .filter(|pair| !self.strikes.contains_key(*pair))
            .cloned()
            .collect();
        let mut changed = false;
        for pair in expired.into_iter().chain(timed_out) {
            self.strikes.remove(&pair);
            self.disabled.remove(&pair);
            self.last_incident_ms.remove(&pair);
            changed = true;
        }
        changed
    }

    /// Every recorded pair, ordered by key: (pair id, strikes, last
    /// incident ms).
    pub fn pairs(&self) -> Vec<(String, usize, u64)> {
        self.strikes
            .iter()
            .map(|(pair, strikes)| {
                (
                    pair.clone(),
                    *strikes,
                    self.last_incident_ms.get(pair).copied().unwrap_or(0),
                )
            })
            .collect()
    }

    /// Rebuilds a ledger from persisted pairs; a pair with at least
    /// [`OCC2_STRIKE_LIMIT`] strikes is disabled, exactly as
    /// [`Occ2StrikeLedger::record_incident_at`] would have left it.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, usize, u64)>) -> Self {
        let mut ledger = Self::new();
        for (pair, strikes, last_ms) in pairs {
            if strikes == 0 {
                continue;
            }
            ledger.strikes.insert(pair.clone(), strikes);
            ledger.last_incident_ms.insert(pair.clone(), last_ms);
            if strikes >= OCC2_STRIKE_LIMIT {
                ledger.disabled.insert(pair);
            }
        }
        ledger
    }

    /// Pairs disabled by the strike limit, ordered by id.
    pub fn disabled_pairs(&self) -> Vec<&str> {
        self.disabled.iter().map(String::as_str).collect()
    }
}

// ---------------------------------------------------------------------------
// Strike ledger persistence: one file per listener session, in the same
// artifact directory as the OCC preference (session_settings precedent:
// atomic tempfile write, bounded read, fail-open on error).
// ---------------------------------------------------------------------------

const OCC2_STRIKES_SCHEMA: &str = "optimus.occ2-strikes.v1";
/// Maximum strike-file read, mirroring the bounded reads in
/// session_settings.rs.
const OCC2_STRIKES_MAX_READ: u64 = 64 * 1024;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Occ2StrikeLedgerFile {
    schema: String,
    /// Pair key -> strike record.
    entries: BTreeMap<String, Occ2StrikeEntryFile>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Occ2StrikeEntryFile {
    strikes: usize,
    last_strike_ms: u64,
}

/// The strike ledger file for one listener session, in its artifact
/// directory (sibling of the OCC preference's `occ.json`).
pub fn occ2_strike_ledger_path(session_file: &str) -> PathBuf {
    PathBuf::from(crate::core::session_manager::get_session_artifact_path_for_file(
        session_file, None,
    ))
    .join("occ2-strikes.json")
}

/// The OCC preference file for one session; must match
/// `session_settings::preference_path`, whose write time drives the
/// toggle-clears-strikes rule.
fn occ2_preference_marker_path(session_file: &str) -> PathBuf {
    PathBuf::from(crate::core::session_manager::get_session_artifact_path_for_file(
        session_file, None,
    ))
    .join("occ.json")
}

/// Atomically persists the ledger for one listener session.
pub fn save_occ2_strike_ledger(
    session_file: &str,
    ledger: &Occ2StrikeLedger,
) -> Result<(), String> {
    if session_file.is_empty() {
        return Err("This chat has no saved session identity".into());
    }
    let path = occ2_strike_ledger_path(session_file);
    let parent = path
        .parent()
        .ok_or("Invalid OCC-2 strike ledger path")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let file = Occ2StrikeLedgerFile {
        schema: OCC2_STRIKES_SCHEMA.into(),
        entries: ledger
            .pairs()
            .into_iter()
            .map(|(pair, strikes, last_strike_ms)| {
                (
                    pair,
                    Occ2StrikeEntryFile {
                        strikes,
                        last_strike_ms,
                    },
                )
            })
            .collect(),
    };
    let mut pending = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    serde_json::to_writer(pending.as_file_mut(), &file).map_err(|e| e.to_string())?;
    pending.write_all(b"\n").map_err(|e| e.to_string())?;
    pending.as_file().sync_all().map_err(|e| e.to_string())?;
    pending.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}

/// Loads the strike ledger for one listener session. Missing file is an
/// empty ledger; so is any read or parse error (fail open: strikes degrade
/// a convenience, they never gate delivery). Decay is applied at load, and
/// a preference write newer than the strike file clears the ledger: the
/// `/occ` toggle re-arming OCC-2 starts the chat's pairs from zero.
pub fn load_occ2_strike_ledger(session_file: &str, now_ms: u64) -> Result<Occ2StrikeLedger, String> {
    let path = occ2_strike_ledger_path(session_file);
    let bytes = match std::fs::File::open(&path) {
        Ok(mut file) => {
            let mut bytes = Vec::new();
            file.take(OCC2_STRIKES_MAX_READ + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() as u64 > OCC2_STRIKES_MAX_READ {
                return Err("OCC-2 strike ledger is too large".into());
            }
            bytes
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Occ2StrikeLedger::new()),
        Err(error) => return Err(error.to_string()),
    };
    // A preference write after the last strike clears the ledger (the
    // user re-armed OCC-2 for this chat). Deleting the file makes the
    // clear sticky without rewriting it.
    if let (Ok(strike_meta), Ok(preference_meta)) = (
        std::fs::metadata(&path),
        std::fs::metadata(occ2_preference_marker_path(session_file)),
    ) {
        let newer_preference = preference_meta
            .modified()
            .ok()
            .zip(strike_meta.modified().ok())
            .is_some_and(|(preference, strike)| preference > strike);
        if newer_preference {
            let _ = std::fs::remove_file(&path);
            return Ok(Occ2StrikeLedger::new());
        }
    }
    let file: Occ2StrikeLedgerFile = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if file.schema != OCC2_STRIKES_SCHEMA {
        return Err("Unsupported OCC-2 strike ledger schema".into());
    }
    let mut ledger = Occ2StrikeLedger::from_pairs(
        file.entries
            .into_iter()
            .map(|(pair, entry)| (pair, entry.strikes, entry.last_strike_ms)),
    );
    ledger.decay(now_ms);
    Ok(ledger)
}

/// Explicitly clears the strike ledger for one session (the function a
/// `/occ` toggle can call directly; the load path already clears when the
/// preference file is newer than the strike file).
pub fn occ2_strike_ledger_clear(session_file: &str) -> Result<(), String> {
    let path = occ2_strike_ledger_path(session_file);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

/// True when one (speaker, listener, dialect) pair is strike-disabled for
/// the listener session: the check the send-side selection path applies
/// before negotiating OCC-2 for that target.
pub fn occ2_strike_pair_disabled(
    session_file: &str,
    speaker: &str,
    listener: &str,
    dialect: (&str, u32),
    now_ms: u64,
) -> bool {
    load_occ2_strike_ledger(session_file, now_ms)
        .map(|ledger| {
            ledger.is_disabled(&occ2_pair_key(speaker, listener, dialect.0, dialect.1))
        })
        .unwrap_or(false)
}

/// How one delivered body was consumed (design section 3, tier 3).
#[derive(Debug, Clone, PartialEq)]
pub enum Occ2BodyDecode {
    /// Strict `occ2::decode` succeeded on the symbolic section.
    Symbolic(AgentCommMessage),
    /// Consumed through the OCC-1 fallback section, or plain OCC-1.
    Occ1(AgentCommMessage),
}

impl Occ2BodyDecode {
    /// The decoded typed payload, regardless of the consumed section.
    pub fn message(&self) -> &AgentCommMessage {
        match self {
            Occ2BodyDecode::Symbolic(message) | Occ2BodyDecode::Occ1(message) => message,
        }
    }

    /// True when the symbolic path decoded the message.
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Occ2BodyDecode::Symbolic(_))
    }
}

/// Decodes one delivered body: strict OCC-2 first, OCC-1 fallback second.
/// An OCC-2 body whose strict decode fails counts one incident for the
/// pair and is consumed through its embedded OCC-1 section; after
/// [`OCC2_STRIKE_LIMIT`] incidents the pair decodes straight through the
/// fallback. Nothing is invented: a body with no decodable OCC-1 form is
/// an error.
pub fn decode_agent_session_message_body(
    body: &str,
    registry: &DialectRegistry,
    pair: &str,
    ledger: &mut Occ2StrikeLedger,
) -> Result<Occ2BodyDecode, ContractError> {
    if body.starts_with(OCC2_HEADER) && !ledger.is_disabled(pair) {
        match occ2::decode(body, registry) {
            Ok(message) => return Ok(Occ2BodyDecode::Symbolic(message)),
            Err(_) => {
                ledger.record_incident(pair);
            }
        }
    }
    let message = occ2::occ1_fallback(body)?;
    Ok(Occ2BodyDecode::Occ1(message))
}

// ---------------------------------------------------------------------------
// Delivery decode + render: the one production path that renders an
// agent-session message body for model consumption. Strict OCC-2 decode
// first; a failure consults and increments the strike ledger for the
// (speaker, listener, dialect) pair and the body is consumed through its
// embedded OCC-1 section - the payload is never dropped or nulled. The
// classification is computed here, once, and recorded in the durable
// details so telemetry reads it instead of re-decoding.
// ---------------------------------------------------------------------------

/// How one delivered body reached the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occ2DeliveryKind {
    /// Strict OCC-2 decode succeeded; the symbolic section (with the
    /// amortized codebook legend or reference) was rendered.
    Symbolic,
    /// Strict OCC-2 decode failed; the strike ledger recorded an
    /// incident and the embedded OCC-1 section was consumed.
    FallbackConsumed,
    /// The body carried no OCC/2 header: plain OCC-1, rendered as-is.
    Occ1Native,
    /// The pair was already strike-disabled; the strict decode was
    /// skipped and the embedded OCC-1 section was consumed.
    DisabledByStrikes,
    /// Plain text or an undecodable body, passed through raw (no data
    /// loss).
    Text,
}

impl Occ2DeliveryKind {
    /// Stable serde/telemetry label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Occ2DeliveryKind::Symbolic => "symbolic",
            Occ2DeliveryKind::FallbackConsumed => "fallback_consumed",
            Occ2DeliveryKind::Occ1Native => "occ1_native",
            Occ2DeliveryKind::DisabledByStrikes => "occ2_disabled_by_strikes",
            Occ2DeliveryKind::Text => "text",
        }
    }
}

/// The delivery classification recorded once in the durable details.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Occ2DeliveryDetails {
    /// [`Occ2DeliveryKind::as_str`] label.
    pub kind: String,
    /// The strike pair key, when the body carried an OCC/2 header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair: Option<String>,
    /// True when this delivery's strict-decode failure recorded a strike.
    #[serde(default)]
    pub strike_recorded: bool,
    /// True when this delivery's strike disabled the pair.
    #[serde(default)]
    pub pair_disabled_now: bool,
}

/// The result of one delivery render.
#[derive(Debug, Clone, PartialEq)]
pub struct Occ2DeliveryOutcome {
    /// The rendered prompt body (what the model consumes).
    pub body: String,
    /// How the body was consumed.
    pub kind: Occ2DeliveryKind,
    /// The strike pair key, when the body carried an OCC/2 header.
    pub pair: Option<String>,
    /// True when this render recorded a strike.
    pub strike_recorded: bool,
    /// True when this render's strike disabled the pair.
    pub pair_disabled_now: bool,
}

/// The dialect id and version a body's OCC/2 header advertises, whether
/// or not the registry knows it: strikes are scoped to the advertised
/// dialect, so an unknown dialect still counts against its own pair.
fn occ2_body_dialect_id(message: &str) -> Option<(String, u32)> {
    let header = message.lines().next()?;
    let rest = header.strip_prefix(OCC2_HEADER)?;
    let mut tokens = rest.split(' ').filter(|token| !token.is_empty());
    let dialect = tokens.next()?.strip_prefix("D=")?;
    let (dialect, version) = dialect.split_once('@')?;
    let version: u32 = version.parse().ok()?;
    Some((dialect.to_string(), version))
}

/// The delivery render: ledger-gated strict decode with the OCC-1
/// fallback and the amortized codebook render in one pass.
pub fn render_agent_session_message_delivery(
    message: &str,
    registry: &DialectRegistry,
    codebooks: &Occ2CodebooksPresent,
    speaker: &str,
    listener: &str,
    ledger: &mut Occ2StrikeLedger,
    now_ms: u64,
) -> Occ2DeliveryOutcome {
    if !message.starts_with(OCC2_HEADER) {
        let kind = if message.starts_with(OCC1_HEADER) {
            Occ2DeliveryKind::Occ1Native
        } else {
            Occ2DeliveryKind::Text
        };
        return Occ2DeliveryOutcome {
            body: message.to_string(),
            kind,
            pair: None,
            strike_recorded: false,
            pair_disabled_now: false,
        };
    }
    let split = split_occ2_dual_body(message);
    let advertised = occ2_body_dialect_id(message);
    let pair = advertised
        .as_ref()
        .map(|(dialect, version)| occ2_pair_key(speaker, listener, dialect, *version));
    // A disabled pair skips the strict decode entirely: straight through
    // the embedded OCC-1 section, no further strikes (design section 3).
    if let Some(pair) = pair.as_deref().filter(|pair| ledger.is_disabled(pair)) {
        return Occ2DeliveryOutcome {
            body: render_fallback_section(split.as_ref()).unwrap_or_else(|| message.to_string()),
            kind: Occ2DeliveryKind::DisabledByStrikes,
            pair: Some(pair.to_string()),
            strike_recorded: false,
            pair_disabled_now: false,
        };
    }
    if occ2::decode(message, registry).is_ok() {
        let symbolic = split
            .map(|(body, _)| body)
            .unwrap_or(message)
            .trim_end_matches('\n');
        let rendered = match occ2_body_dialect_card(message, registry) {
            Some(card) => {
                let identity = Occ2CodebookId::of_card(card);
                let lead = if codebooks.contains(&identity) {
                    occ2_codebook_reference(&identity)
                } else {
                    occ2_codebook_legend(card)
                };
                format!("{lead}\n\n{symbolic}")
            }
            None => symbolic.to_string(),
        };
        return Occ2DeliveryOutcome {
            body: rendered,
            kind: Occ2DeliveryKind::Symbolic,
            pair,
            strike_recorded: false,
            pair_disabled_now: false,
        };
    }
    // Strict decode failed: one strike for the advertised dialect's pair,
    // then the embedded OCC-1 section (or the raw body, never dropped).
    let disabled_now = pair
        .as_deref()
        .map(|pair| ledger.record_incident_at(pair, now_ms))
        .unwrap_or(false);
    Occ2DeliveryOutcome {
        body: render_fallback_section(split.as_ref()).unwrap_or_else(|| message.to_string()),
        kind: Occ2DeliveryKind::FallbackConsumed,
        pair: pair.clone(),
        strike_recorded: pair.is_some(),
        pair_disabled_now: disabled_now,
    }
}

/// The embedded OCC-1 section of a dual body, when it parses as a valid
/// contract message; `None` leaves the raw body untouched.
fn render_fallback_section(split: Option<&(&str, &str)>) -> Option<String> {
    let (_, fallback) = split?;
    let occ1 = Occ1Message::decode(fallback).ok()?;
    AgentCommMessage::from_occ1(&occ1).ok()?;
    Some(fallback.to_string())
}

pub fn create_agent_session_message_prompt(payload: &AgentSessionMessagePayload) -> String {
    create_agent_session_message_prompt_with_codebooks(payload, &Occ2CodebooksPresent::new())
}

/// [`create_agent_session_message_prompt`] with the codebook identities
/// already present in the listener's context: the first OCC-2 body of an
/// identity carries the full codebook legend, later ones a one-line
/// reference. The prompt text and the durable record must be built from the
/// same set, so both render identically.
pub fn create_agent_session_message_prompt_with_codebooks(
    payload: &AgentSessionMessagePayload,
    codebooks: &Occ2CodebooksPresent,
) -> String {
    // OCC-2 seam: a dual payload renders its symbolic section only; the
    // durable record keeps the full dual body (design section 3). The
    // codebook legend is amortized by the passed-in presence set.
    let body = agent_session_message_prompt_body_with_codebooks(&payload.message, codebooks);
    create_agent_session_message_prompt_with_body(payload, &body)
}

pub fn create_agent_session_message(
    payload: &AgentSessionMessagePayload,
    timestamp: i64,
) -> CustomAgentMessage {
    create_agent_session_message_with_codebooks(
        payload,
        timestamp,
        &Occ2CodebooksPresent::new(),
    )
}

/// [`create_agent_session_message`] with the codebook identities already
/// present in the listener's context; pass the same set that built the
/// prompt text so the durable record's content matches the delivered text.
pub fn create_agent_session_message_with_codebooks(
    payload: &AgentSessionMessagePayload,
    timestamp: i64,
    codebooks: &Occ2CodebooksPresent,
) -> CustomAgentMessage {
    let details = AgentSessionMessageDetails {
        id: payload.id.clone(),
        message: payload.message.clone(),
        from: payload.from.clone(),
        from_relationship: payload.from_relationship.clone(),
        target: Some(payload.target.clone()),
        occ2_delivery: None,
    };
    CustomAgentMessage::Custom {
        custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(create_agent_session_message_prompt_with_codebooks(
            payload, codebooks,
        )),
        display: true,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    }
}

/// One delivery render result: the visible prompt text, the durable
/// record, and the decode classification, all from a single pass.
#[derive(Debug, Clone)]
pub struct Occ2AgentSessionMessageDelivery {
    /// The prompt text the model consumes (header lines + rendered body).
    pub content: String,
    /// The durable transcript record (dual body verbatim in details,
    /// rendered content, delivery classification).
    pub record: CustomAgentMessage,
    /// How the body was consumed, plus the strike effects.
    pub outcome: Occ2DeliveryOutcome,
}

/// Builds one delivered agent-session message end to end: one
/// strict-decode attempt (ledger-gated), the amortized codebook render,
/// the OCC-1 fallback on failure, and the delivery classification
/// recorded in the durable details. This is the production seam
/// (`daemon_mode.rs` `accept_agent_session_message`); the prompt text
/// and the record are built from the same render so they agree.
pub fn create_agent_session_message_delivery(
    payload: &AgentSessionMessagePayload,
    timestamp: i64,
    codebooks: &Occ2CodebooksPresent,
    speaker: &str,
    listener: &str,
    ledger: &mut Occ2StrikeLedger,
    now_ms: u64,
) -> Occ2AgentSessionMessageDelivery {
    let outcome = render_agent_session_message_delivery(
        &payload.message,
        &DialectRegistry::with_sym_v1(),
        codebooks,
        speaker,
        listener,
        ledger,
        now_ms,
    );
    let content = create_agent_session_message_prompt_with_body(payload, &outcome.body);
    let details = AgentSessionMessageDetails {
        id: payload.id.clone(),
        message: payload.message.clone(),
        from: payload.from.clone(),
        from_relationship: payload.from_relationship.clone(),
        target: Some(payload.target.clone()),
        occ2_delivery: Some(Occ2DeliveryDetails {
            kind: outcome.kind.as_str().to_string(),
            pair: outcome.pair.clone(),
            strike_recorded: outcome.strike_recorded,
            pair_disabled_now: outcome.pair_disabled_now,
        }),
    };
    let record = CustomAgentMessage::Custom {
        custom_type: AGENT_MESSAGE_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(content.clone()),
        display: true,
        details: serde_json::to_value(&details).ok(),
        timestamp,
    };
    Occ2AgentSessionMessageDelivery {
        content,
        record,
        outcome,
    }
}

/// The listener scaffolding (label, Source/From/To/Message id lines)
/// around an already-rendered body: shared by the pure prompt builder
/// and the delivery builder so both produce identical text.
fn create_agent_session_message_prompt_with_body(
    payload: &AgentSessionMessagePayload,
    body: &str,
) -> String {
    let relationship_label = payload.from_relationship.as_ref().map(|relationship| {
        if relationship == FAMILY_RELATIONSHIP_PARENT {
            format!("[from {relationship}]")
        } else {
            let sender = payload.from.as_ref();
            let name = sender
                .and_then(|sender| sender.session_name.clone())
                .or_else(|| sender.and_then(|sender| sender.session_id.clone()))
                .or_else(|| sender.and_then(|sender| sender.active_session_id.clone()))
                .unwrap_or_else(|| "unknown".to_string());
            format!(
                "[from {relationship}:{}]",
                format_agent_session_message_metadata(&name)
            )
        }
    });
    let mut lines: Vec<String> = Vec::new();
    if let Some(label) = relationship_label {
        lines.push(label);
    }
    lines.push("Agent-to-agent message received.".to_string());
    lines.push(format!("Source: {}", payload.source));
    if let Some(from) = &payload.from {
        lines.push(format!("From: {}", format_agent_session_message_sender(from)));
    }
    lines.push(format!(
        "To: {}",
        format_agent_session_message_endpoint(&payload.target)
    ));
    lines.push(format!("Message id: {}", payload.id));
    lines.push(String::new());
    lines.push(body.to_string());
    lines.join("\n")
}

pub fn is_agent_session_message(message: &AgentMessage) -> bool {
    let AgentMessage::Custom(CustomAgentMessage::Custom {
        custom_type,
        details,
        ..
    }) = message
    else {
        return false;
    };
    if custom_type != AGENT_MESSAGE_CUSTOM_TYPE {
        return false;
    }
    let Some(details) = details.as_ref().and_then(Value::as_object) else {
        return false;
    };
    details.get("id").map(Value::is_string).unwrap_or(false)
        && details
            .get("message")
            .map(Value::is_string)
            .unwrap_or(false)
}

// A message that starts a new agent run (prompt-turn boundary).
pub fn starts_agent_run(message: &AgentMessage) -> bool {
    if matches!(
        message,
        AgentMessage::Message(pi_ai::types::Message::User(_))
    ) {
        return true;
    }
    if is_agent_session_message(message) {
        return true;
    }
    matches!(
        message,
        AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, .. })
            if custom_type == HEARTBEAT_PROMPT_CUSTOM_TYPE
                || custom_type == ASYNC_BASH_COMPLETION_CUSTOM_TYPE
    )
}

pub fn create_agent_session_message_receipt(
    payload: &AgentSessionMessagePayload,
    status: &AgentSessionMessageDeliveryStatus,
    at: &str,
) -> AgentSessionMessageReceipt {
    AgentSessionMessageReceipt {
        id: payload.id.clone(),
        source: payload.source.clone(),
        target: payload.target.clone(),
        from: payload.from.clone(),
        message: payload.message.clone(),
        delivery_status: status.clone(),
        delivered_at: if status == DELIVERY_STATUS_DELIVERED {
            Some(at.to_string())
        } else {
            None
        },
        queued_at: if status == DELIVERY_STATUS_DELIVERED {
            None
        } else {
            Some(at.to_string())
        },
        delivery_mode: Some(DELIVERY_MODE_STEER.to_string()),
    }
}

fn format_agent_session_message_metadata(value: &str) -> String {
    let mut out = String::new();
    let mut last_was_space = false;
    for ch in value.chars() {
        let is_separator = ch.is_whitespace() || ch == ',' || ch == '[' || ch == ']';
        if is_separator {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

fn format_agent_session_message_sender(sender: &AgentSessionMessageSender) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(session_name) = &sender.session_name {
        let formatted = format_agent_session_message_metadata(session_name);
        if !formatted.is_empty() {
            parts.push(formatted);
        }
    }
    if let Some(active_session_id) = &sender.active_session_id {
        parts.push(format!(
            "active {}",
            format_agent_session_message_metadata(active_session_id)
        ));
    }
    if let Some(session_id) = &sender.session_id {
        parts.push(format!(
            "session {}",
            format_agent_session_message_metadata(session_id)
        ));
    }
    if let Some(client_id) = &sender.client_id {
        parts.push(format!(
            "client {}",
            format_agent_session_message_metadata(client_id)
        ));
    }
    if parts.is_empty() {
        "unknown sender".to_string()
    } else {
        parts.join(", ")
    }
}

fn format_agent_session_message_endpoint(endpoint: &AgentSessionMessageEndpoint) -> String {
    let name = endpoint
        .session_name
        .as_ref()
        .map(|name| format!("{}, ", format_agent_session_message_metadata(name)))
        .unwrap_or_default();
    format!(
        "{name}active {}, session {}",
        format_agent_session_message_metadata(&endpoint.active_session_id),
        format_agent_session_message_metadata(&endpoint.session_id)
    )
}

/// Rate limiter state for one key (`{ tokens, updatedAt }`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct RateLimitBucket {
    tokens: f64,
    updated_at: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitResult {
    Ok,
    Exceeds { retry_after_ms: f64 },
}

/// Token-bucket limiter; `now` is injected so tests stay deterministic.
pub struct AgentSessionMessageRateLimiter {
    capacity: f64,
    refill_ms: f64,
    buckets: HashMap<String, RateLimitBucket>,
    now: Box<dyn Fn() -> f64 + Send + Sync>,
}

impl Default for AgentSessionMessageRateLimiter {
    fn default() -> Self {
        Self::new(
            DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY,
            DEFAULT_AGENT_MESSAGE_RATE_LIMIT_REFILL_MS,
            None,
        )
    }
}

impl AgentSessionMessageRateLimiter {
    pub fn new(
        capacity: f64,
        refill_ms: f64,
        now: Option<Box<dyn Fn() -> f64 + Send + Sync>>,
    ) -> Self {
        Self {
            capacity,
            refill_ms,
            buckets: HashMap::new(),
            now: now.unwrap_or_else(|| {
                Box::new(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|duration| duration.as_millis() as f64)
                        .unwrap_or(0.0)
                })
            }),
        }
    }

    pub fn try_consume(&mut self, key: &str) -> RateLimitResult {
        let now = (self.now)();
        let mut bucket = self.buckets.get(key).copied().unwrap_or(RateLimitBucket {
            tokens: self.capacity,
            updated_at: now,
        });
        let elapsed = f64::max(0.0, now - bucket.updated_at);
        let refilled_tokens = (elapsed / self.refill_ms).floor();
        if refilled_tokens > 0.0 {
            bucket.tokens = f64::min(self.capacity, bucket.tokens + refilled_tokens);
            bucket.updated_at += refilled_tokens * self.refill_ms;
        }
        if bucket.tokens <= 0.0 {
            self.buckets.insert(key.to_string(), bucket);
            return RateLimitResult::Exceeds {
                retry_after_ms: f64::max(1.0, bucket.updated_at + self.refill_ms - now),
            };
        }
        bucket.tokens -= 1.0;
        self.buckets.insert(key.to_string(), bucket);
        RateLimitResult::Ok
    }

    pub fn refund(&mut self, key: &str) {
        let Some(mut bucket) = self.buckets.get(key).copied() else {
            return;
        };
        bucket.tokens = f64::min(self.capacity, bucket.tokens + 1.0);
        self.buckets.insert(key.to_string(), bucket);
    }

    pub fn clear(&mut self, key: Option<&str>) {
        match key {
            Some(key) => {
                self.buckets.remove(key);
            }
            None => self.buckets.clear(),
        }
    }

    pub fn clear_matching(&mut self, predicate: impl Fn(&str) -> bool) {
        let keys: Vec<String> = self
            .buckets
            .keys()
            .filter(|key| predicate(key))
            .cloned()
            .collect();
        for key in keys {
            self.buckets.remove(&key);
        }
    }
}


/// Internal, process-local authorization. No serde or public daemon wire input
/// can construct a prepared source handoff. Both the real daemon and isolated
/// native harness call this exact family-policy constructor.
#[derive(Clone)]
pub struct AuthorizedNativeHandoff {
    message_id: String,
    from: AgentSessionMessageEndpoint,
    target: AgentSessionMessageEndpoint,
    relationship: String,
}
impl AuthorizedNativeHandoff {
    pub fn from_catalog(
        current: &AgentFamilyCatalogEntry, target: &AgentFamilyCatalogEntry,
        from: AgentSessionMessageEndpoint, to: AgentSessionMessageEndpoint,
        requested_relationship: &str,
    ) -> Result<Self, String> {
        let relationship = assert_agent_family_reach(current, target)?;
        // Native handoff is stricter than plain agent messaging: plain
        // messaging treats any two parentless roots as siblings, but an
        // unrelated top-level session must stay outside the native family
        // reach. Siblings here share an actual parent edge.
        if current.depth == 0.0
            && target.depth == 0.0
            && current.parent_session_path.is_none()
            && target.parent_session_path.is_none()
            && current.parent_session_id.is_none()
            && target.parent_session_id.is_none()
        {
            return Err(AGENT_FAMILY_REACH_ERROR.to_string());
        }
        if current.id != from.session_id || target.id != to.session_id
            || relationship != requested_relationship || from.session_id == to.session_id
        { return Err("Native handoff family binding rejected".into()); }
        Ok(Self {message_id:create_agent_session_message_id(),from,target:to,relationship})
    }
    pub fn message_id(&self) -> &str { &self.message_id }
    pub fn source_session_id(&self) -> &str { &self.from.session_id }
    pub fn target_session_id(&self) -> &str { &self.target.session_id }
    pub fn target(&self) -> &AgentSessionMessageEndpoint { &self.target }
}

#[derive(Clone)]
pub struct PreparedNativeHandoff {
    authorization: AuthorizedNativeHandoff,
    publication: crate::core::occ::occ3::connector::NativePublication,
}
impl PreparedNativeHandoff {
    pub fn new(authorization: AuthorizedNativeHandoff, publication: crate::core::occ::occ3::connector::NativePublication) -> Result<Self, String> {
        let e = &publication.envelope;
        if e.message_id != authorization.message_id || e.sender_session_id != authorization.from.session_id
            || e.receiver_session_id != authorization.target.session_id
            || e.control_summary != "Continue the authorized task using the attached state."
        { return Err("Native handoff publication binding rejected".into()); }
        Ok(Self {authorization,publication})
    }
    pub fn authorization(&self) -> &AuthorizedNativeHandoff { &self.authorization }
    pub fn envelope(&self) -> &crate::core::occ::occ3::types::Occ3Envelope { &self.publication.envelope }
    /// Durable/model-visible message contains control only. Never persist raw
    /// state handles, source receipts or source text in custom-message details.
    pub fn control_payload(&self) -> AgentSessionMessagePayload {
        AgentSessionMessagePayload {
            id:self.authorization.message_id.clone(),source:AGENT_MESSAGE_SOURCE.into(),
            message:self.publication.envelope.control_summary.clone(),
            from:Some(AgentSessionMessageSender {active_session_id:Some(self.authorization.from.active_session_id.clone()),
                session_id:Some(self.authorization.from.session_id.clone()),session_name:self.authorization.from.session_name.clone(),
                runtime_kind:self.authorization.from.runtime_kind.clone(),client_id:None}),
            from_relationship:Some(self.authorization.relationship.clone()),target:self.authorization.target.clone(),
        }
    }
}

/// Controller surface used by the host request handlers.
///
/// `AgentSessionMessageController` is a TypeScript interface with optional
/// members; the Rust trait keeps the same three members and lets each
/// implementation decide whether the optional ones exist.
pub trait AgentSessionMessageController: Send + Sync {
    /// Internal experimental native route; ordinary kernel payloads cannot use
    /// this without actual parsed source tool provenance and an installed permit.
    fn authorize_native_handoff(&self, _relationship: String, _selector: String)
        -> pi_ai::types::BoxFuture<Result<AuthorizedNativeHandoff, String>> {
        Box::pin(async { Err("Native source handoff is not installed".into()) })
    }
    fn send_prepared_native_handoff(&self, _prepared: PreparedNativeHandoff)
        -> pi_ai::types::BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        Box::pin(async { Err("Native source handoff is not installed".into()) })
    }

    /// Internal catalog, including passive children; older hosts may omit it.
    fn list_agents(&self) -> pi_ai::types::BoxFuture<Result<Option<AgentSessionMessageListResult>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn roster(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentFamilyRosterResult, String>> + Send>,
    >;
    fn await_pending_child_publication(
        &self,
        selector: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send>>;
    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentSessionMessageReceipt, String>> + Send>,
    >;
    /// OCC-2 send policy for one target id (OCC-2 integration design,
    /// section 4). The default is the conservative disabled policy, so
    /// existing implementors keep OCC-1 behavior unchanged.
    fn occ2_send_policy(
        &self,
        target: &str,
        message: &str,
    ) -> pi_ai::types::BoxFuture<Occ2SendPolicy> {
        let _ = (target, message);
        Box::pin(async { Occ2SendPolicy::disabled() })
    }

    /// OCC-3 send policy for one target: the async arm that owns
    /// connector I/O. Implementors capture the latent state BEFORE the
    /// send (this is the only async step) and return the handle for
    /// the pure envelope seam. The default is the disabled policy:
    /// OCC-3 requires an explicit opt-in plus a live connector.
    fn occ3_send_policy(
        &self,
        target: &str,
        message: &str,
    ) -> pi_ai::types::BoxFuture<Occ3SendPolicy> {
        let _ = (target, message);
        Box::pin(async { Occ3SendPolicy::disabled() })
    }
}

/// `createAgentMessageHostHandlers(controller)`.
/// Typed provenance route for `agent_message.send`. Implements the
/// fail-closed publication gate: authority classification uses ONLY
/// host-owned frame state (`ctx`), never payload fields, and never the
/// current cell. The ordinary route (no token present, nothing enrolled)
/// delegates byte-identically to [`handle_agent_message_send`].
pub fn create_agent_message_provenance_host_handlers(
    controller: std::sync::Arc<dyn AgentSessionMessageController>,
) -> crate::core::kernel::shared::ProvenanceHostRequestHandlers {
    use crate::core::kernel::shared::{HostRequestContext, ProvenanceHostRequestHandler};
    use std::collections::HashMap;

    let mut handlers: crate::core::kernel::shared::ProvenanceHostRequestHandlers = HashMap::new();

    let send_controller = controller.clone();
    let send: ProvenanceHostRequestHandler = std::sync::Arc::new(
        move |ctx: HostRequestContext, payload: Value| {
            let controller = send_controller.clone();
            Box::pin(async move {
                handle_agent_message_send_provenance(controller, ctx, payload).await
            })
        },
    );
    handlers.insert("agent_message.send".to_string(), send);

    handlers
}

/// The fail-closed publication gate for the provenance route.
///
/// Rules (root correction, mandatory):
/// - ANY present token (well-formed or not, matched or not) that does not
///   claim the live enrolled scope REFUSES before publication, even when no
///   scope is enrolled: a known prior token can never downgrade to plaintext.
/// - An absent token while an execution is enrolled (live or settling)
///   refuses and tombstones the enrolled scope.
/// - An absent token with nothing enrolled is the ordinary route, unchanged.
/// - A matched scope that claims successfully is the Phase C consumer; this
///   build refuses BEFORE publication rather than sending plaintext.
async fn handle_agent_message_send_provenance(
    controller: std::sync::Arc<dyn AgentSessionMessageController>,
    ctx: crate::core::kernel::shared::HostRequestContext,
    payload: Value,
) -> Result<Value, crate::core::kernel::shared::KernelError> {
    use crate::core::kernel::shared::KernelError;

    if ctx.token_present {
        if let Some(scope) = ctx.live_enrolled.clone() {
            let matched = ctx
                .matched_scope
                .as_ref()
                .map(|matched| std::sync::Arc::ptr_eq(matched, &scope))
                .unwrap_or(false);
            if matched && scope.claim().is_ok() {
                // Phase C enrollment is not implemented in this build.
                // Refuse BEFORE publication; never send plaintext. The
                // claim is consumed, so a retry with the same token
                // tombstones below.
                return Err(KernelError::new(
                    "agent_message.send refused: provenance route is not enrolled in this build",
                ));
            }
            scope.tombstone();
            return Err(KernelError::new(
                "agent_message.send refused: execution token did not match the enrolled execution",
            ));
        }
        // No enrolled execution: a present-but-unknown token still refuses.
        return Err(KernelError::new(
            "agent_message.send refused: unknown execution token",
        ));
    }
    if let Some(scope) = ctx.live_enrolled.clone() {
        scope.tombstone();
        return Err(KernelError::new(
            "agent_message.send refused: execution provenance required",
        ));
    }
    handle_agent_message_send(controller, payload).await
}

pub fn create_agent_message_host_handlers(
    controller: std::sync::Arc<dyn AgentSessionMessageController>,
) -> crate::core::kernel::shared::HostRequestHandlers {
    use crate::core::kernel::shared::{HostRequestHandler, HostRequestHandlers, KernelError};
    use std::collections::HashMap;

    let mut handlers: crate::core::kernel::shared::HostRequestHandlers = HashMap::new();

    let roster_controller = controller.clone();
    let list_agents: HostRequestHandler = Arc::new(move |_payload: Value| {
        let controller = roster_controller.clone();
        Box::pin(async move {
            match controller.roster().await {
                Ok(roster) => Ok(serde_json::to_value(roster).unwrap_or(Value::Null)),
                Err(error) => Err(KernelError::new(error)),
            }
        })
    });
    handlers.insert("agent_message.list_agents".to_string(), list_agents);

    let send_controller = controller.clone();
    let send: HostRequestHandler = Arc::new(move |payload: Value| {
        let controller = send_controller.clone();
        Box::pin(async move { handle_agent_message_send(controller, payload).await })
    });
    handlers.insert("agent_message.send".to_string(), send);

    handlers
}

async fn handle_agent_message_send(
    controller: Arc<dyn AgentSessionMessageController>,
    payload: Value,
) -> Result<Value, crate::core::kernel::shared::KernelError> {
    use crate::core::kernel::shared::KernelError;

    let object = payload.as_object().cloned().unwrap_or_default();
    let message = match object.get("message") {
        Some(Value::String(message)) => message.clone(),
        _ => {
            return Err(KernelError::new(
                "agent_message.send message must be a string",
            ))
        }
    };
    let receiver_role = object
        .get("receiver_role")
        .and_then(Value::as_str)
        .map(str::to_string);
    let receiver_name = object
        .get("receiver_name")
        .and_then(Value::as_str)
        .map(str::to_string);

    let target: String;
    if let Some(Value::String(positional)) = object.get("target") {
        if positional != "all" {
            return Err(KernelError::new(
                "positional agent_message.send targets are not supported; use receiver_role and receiver_name",
            ));
        }
        if receiver_role.is_some() || receiver_name.is_some() {
            return Err(KernelError::new(
                "agent_message.send broadcast cannot be combined with receiver_role/receiver_name",
            ));
        }
        let roster = controller.roster().await.map_err(KernelError::new)?;
        let mut receipts: Vec<Value> = Vec::with_capacity(roster.entries.len());
        for entry in &roster.entries {
            // OCC-2 S1 seam, broadcast arm: each recipient encodes against
            // its own negotiated dialect (design R6); no receipt change.
            let policy = controller.occ2_send_policy(&entry.id, &message).await;
            let message = wrap_outgoing_body(&message, &policy);
            // OCC-3 seam, broadcast arm: latent capture per recipient.
            let occ3 = controller.occ3_send_policy(&entry.id, &message).await;
            let message = wrap_outgoing_body_occ3(&message, &occ3);
            let input = AgentSessionMessageSendInput {
                target: entry.id.clone(),
                message,
                receiver_role: Some(entry.relationship.clone()),
            };
            match controller.send_agent_message(input).await {
                Ok(receipt) => receipts.push(serde_json::to_value(receipt).unwrap_or(Value::Null)),
                Err(error) => {
                    let mut failed = Map::new();
                    failed.insert("target".to_string(), Value::String(entry.id.clone()));
                    failed.insert("error".to_string(), Value::String(error));
                    receipts.push(Value::Object(failed));
                }
            }
        }
        return Ok(serde_json::json!({ "receipts": receipts }));
    }

    let role =
        match receiver_role.as_deref() {
            Some("parent") => FAMILY_RELATIONSHIP_PARENT.to_string(),
            Some("sibling") => FAMILY_RELATIONSHIP_SIBLING.to_string(),
            Some("child") => FAMILY_RELATIONSHIP_CHILD.to_string(),
            _ => return Err(KernelError::new(
                "agent_message.send receiver_role must be \"parent\", \"sibling\", or \"child\"",
            )),
        };
    if role == FAMILY_RELATIONSHIP_PARENT && receiver_name.is_some() {
        return Err(KernelError::new(
            "agent_message.send receiver_name must be omitted for parent messages",
        ));
    }
    if role != FAMILY_RELATIONSHIP_PARENT
        && receiver_name
            .as_ref()
            .map(|name| name.trim().is_empty())
            .unwrap_or(true)
    {
        return Err(KernelError::new(
            "agent_message.send receiver_name is required for sibling and child messages",
        ));
    }

    let selector = receiver_name.as_ref().map(|name| name.trim().to_string());
    let published_id = if role == FAMILY_RELATIONSHIP_CHILD {
        match &selector {
            Some(selector) => controller
                .await_pending_child_publication(selector.clone())
                .await
                .map_err(KernelError::new)?,
            None => None,
        }
    } else {
        None
    };
    let roster = controller.roster().await.map_err(KernelError::new)?;
    let matches: Vec<&AgentFamilyRosterEntry> = roster
        .entries
        .iter()
        .filter(|entry| {
            entry.relationship == role
                && (role == FAMILY_RELATIONSHIP_PARENT
                    || Some(entry.name.as_str()) == selector.as_deref()
                    || Some(entry.id.as_str()) == selector.as_deref()
                    || published_id.as_deref() == Some(entry.id.as_str()))
        })
        .collect();
    if matches.len() != 1 {
        return Err(KernelError::new(if matches.is_empty() {
            if role == FAMILY_RELATIONSHIP_PARENT {
                "No parent matches the current agent".to_string()
            } else {
                format!(
                    "No {role} matches {}",
                    serde_json::to_string(&receiver_name).unwrap_or_default()
                )
            }
        } else {
            format!(
                "{role} selector {} is ambiguous",
                serde_json::to_string(&receiver_name).unwrap_or_default()
            )
        }));
    }
    target = matches[0].id.clone();

    // OCC-2 S1 seam, direct arm: the body is substituted only when the
    // feature is enabled and a dialect was negotiated for this target;
    // otherwise the send is byte-identical OCC-0/OCC-1.
    let policy = controller.occ2_send_policy(&target, &message).await;
    let message = wrap_outgoing_body(&message, &policy);
    // OCC-3 seam: the async capture already happened inside
    // `occ3_send_policy`; this wrap is the pure envelope encode. Any
    // failure upstream left the policy disabled, so the message still
    // goes out as OCC-1/OCC-2 (rule R1: never lost, never truncated).
    let occ3 = controller.occ3_send_policy(&target, &message).await;
    let message = wrap_outgoing_body_occ3(&message, &occ3);
    let receipt = controller
        .send_agent_message(AgentSessionMessageSendInput {
            target,
            message,
            receiver_role: Some(role),
        })
        .await
        .map_err(KernelError::new)?;
    Ok(serde_json::to_value(receipt).unwrap_or(Value::Null))
}

/// Strike incidents recorded for one (speaker, listener, dialect) pair,
/// decay-applied: the selection input for the send seams. Any load error
/// reads as zero (fail open).
pub fn occ2_strike_pair_incidents(
    session_file: &str,
    speaker: &str,
    listener: &str,
    dialect: (&str, u32),
    now_ms: u64,
) -> usize {
    load_occ2_strike_ledger(session_file, now_ms)
        .map(|ledger| ledger.incidents(&occ2_pair_key(speaker, listener, dialect.0, dialect.1)))
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Dialect evaluation store: the minimal durable form behind
// SelectionContext.evaluations. One file per sender session, in its
// artifact directory (sibling of occ.json and occ2-strikes.json),
// written atomically, read fail-open. Empty by default: both shipped
// dialects are honestly unevaluated, so production selection stays
// OCC-1 until a validated record lands here.
// ---------------------------------------------------------------------------

const OCC2_EVALS_SCHEMA: &str = "optimus.occ2-evals.v1";
/// Maximum evaluation-store read, mirroring the bounded reads of the
/// other OCC artifact files.
const OCC2_EVALS_MAX_READ: u64 = 64 * 1024;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Occ2EvaluationsFile {
    schema: String,
    /// One record per (dialect, version).
    records: Vec<Occ2EvaluationRecordFile>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Occ2EvaluationRecordFile {
    dialect: String,
    version: u32,
    samples: usize,
    listener_fidelity_mean: f64,
    wire_tokens_vs_occ1: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_evaluated: Option<u64>,
}

/// The evaluation-store file for one session, in its artifact directory.
pub fn occ2_evaluations_path(session_file: &str) -> PathBuf {
    PathBuf::from(crate::core::session_manager::get_session_artifact_path_for_file(
        session_file, None,
    ))
    .join("occ2-evals.json")
}

/// Loads the dialect evaluation records for one session, keyed by
/// (dialect id, version). Missing file, parse error or size violation
/// all read as an empty map (fail open: unevaluated means the production
/// policy conservatively picks OCC-1, never an error).
pub fn load_occ2_evaluations(
    session_file: &str,
) -> BTreeMap<(String, u32), DialectEvaluationRecord> {
    let mut records = BTreeMap::new();
    let path = occ2_evaluations_path(session_file);
    let bytes = match std::fs::File::open(&path) {
        Ok(mut file) => {
            let mut bytes = Vec::new();
            if file
                .take(OCC2_EVALS_MAX_READ + 1)
                .read_to_end(&mut bytes)
                .is_err()
            {
                return records;
            }
            if bytes.len() as u64 > OCC2_EVALS_MAX_READ {
                return records;
            }
            bytes
        }
        Err(_) => return records,
    };
    let Ok(file) = serde_json::from_slice::<Occ2EvaluationsFile>(&bytes) else {
        return records;
    };
    if file.schema != OCC2_EVALS_SCHEMA {
        return records;
    }
    for record in file.records {
        records.insert(
            (record.dialect.clone(), record.version),
            DialectEvaluationRecord {
                samples: record.samples,
                listener_fidelity_mean: record.listener_fidelity_mean,
                wire_tokens_vs_occ1: record.wire_tokens_vs_occ1,
                last_evaluated: record.last_evaluated,
            },
        );
    }
    records
}

/// Atomically persists the dialect evaluation records for one session.
/// The offline evaluation harness (or a test injecting a synthetic
/// record) is the intended writer.
pub fn save_occ2_evaluations(
    session_file: &str,
    records: &BTreeMap<(String, u32), DialectEvaluationRecord>,
) -> Result<(), String> {
    if session_file.is_empty() {
        return Err("This chat has no saved session identity".into());
    }
    let path = occ2_evaluations_path(session_file);
    let parent = path.parent().ok_or("Invalid OCC-2 evaluations path")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let file = Occ2EvaluationsFile {
        schema: OCC2_EVALS_SCHEMA.into(),
        records: records
            .iter()
            .map(|((dialect, version), record)| Occ2EvaluationRecordFile {
                dialect: dialect.clone(),
                version: *version,
                samples: record.samples,
                listener_fidelity_mean: record.listener_fidelity_mean,
                wire_tokens_vs_occ1: record.wire_tokens_vs_occ1,
                last_evaluated: record.last_evaluated,
            })
            .collect(),
    };
    let mut pending = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    serde_json::to_writer(pending.as_file_mut(), &file).map_err(|e| e.to_string())?;
    pending.write_all(b"\n").map_err(|e| e.to_string())?;
    pending.as_file().sync_all().map_err(|e| e.to_string())?;
    pending.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Selection assembly: one function turns a filled SelectionContext into
// the send policy the seams apply, so the decision order lives in
// exactly one place.
// ---------------------------------------------------------------------------

/// The selection outcome trace for one send decision: the production
/// decision, and the development fallback when the experimental path
/// exercised it.
#[derive(Debug, Clone, PartialEq)]
pub struct Occ2SelectionOutcome {
    /// The production-policy decision (always evaluated).
    pub production: SelectionDecision,
    /// The development-policy decision, when the experimental path ran
    /// (production refused an unevaluated dialect).
    pub development: Option<SelectionDecision>,
    /// The final send policy.
    pub policy: Occ2SendPolicy,
}

/// Turns a filled [`SelectionContext`] into the send policy.
///
/// Decision order: the production policy always runs. A validated
/// dialect that wins on measured economics selects OCC-2 outright. When
/// production refuses only because the dialect is unevaluated AND the
/// experimental path is enabled (`experimental = true`, today the
/// `/occ 2` opt-in), the development policy runs with the SILO reference
/// profile so the wire can be exercised end to end - still behind the
/// small-message floor, the strike guard and the codebook economics.
/// Every other production refusal (unknown listener, no shared
/// dialect, strikes, no savings) is final.
pub fn occ2_send_policy_from_selection(
    context: &SelectionContext<'_>,
    experimental: bool,
) -> Occ2SelectionOutcome {
    let for_dialect = |dialect: (String, u32), registry: &DialectRegistry| {
        Occ2SendPolicy::for_dialect(
            registry.clone(),
            dialect,
            DEFAULT_AGENT_MESSAGE_MAX_CHARS,
        )
    };
    let production = select_with_policy(context, &SelectionPolicy::default());
    match &production {
        SelectionDecision::Occ2 { dialect, .. } => Occ2SelectionOutcome {
            production: production.clone(),
            development: None,
            policy: for_dialect(dialect.clone(), context.registry),
        },
        SelectionDecision::Occ1(SelectionReason::UnevaluatedDialect) if experimental => {
            let development = select_with_policy(context, &SelectionPolicy::development());
            let policy = match &development {
                SelectionDecision::Occ2 { dialect, .. } => {
                    for_dialect(dialect.clone(), context.registry)
                }
                _ => Occ2SendPolicy::disabled(),
            };
            Occ2SelectionOutcome {
                production: production.clone(),
                development: Some(development),
                policy,
            }
        }
        _ => Occ2SelectionOutcome {
            production: production.clone(),
            development: None,
            policy: Occ2SendPolicy::disabled(),
        },
    }
}

/// The `KIND=` field of an OCC-1 body when present, else `"TASK"`
/// (informational in policy v1).
pub fn occ2_message_kind_hint(message: &str) -> &str {
    message
        .lines()
        .find_map(|line| line.strip_prefix("KIND="))
        .map(|kind| match kind {
            "TASK" | "PROGRESS" | "RESULT" | "EVIDENCE" | "REQUEST_DETAIL" | "ACK"
            | "CORRECTION" | "BLOCKED" | "CANCELLED" => kind,
            _ => "TASK",
        })
        .unwrap_or("TASK")
}

/// The shared send-seam selection from session-level inputs. Both the
/// in-process seam (`AgentSession::occ2_selected_send_policy`) and the
/// daemon-bridged controller resolve their own speaker/listener/model
/// inputs and call this one function, so the decision order and the
/// artifact reads live in exactly one place.
///
/// `speaker_session_file` scopes the strike-ledger and evaluation-store
/// reads (the sender's session artifacts); `None` reads as no strikes
/// and no evaluations. `experimental` is the `/occ 2` opt-in.
pub fn occ2_send_policy_for_send(
    peer: Option<&OccCapability>,
    listener: &str,
    message: &str,
    speaker_session_file: Option<&str>,
    speaker_model: Option<&str>,
    speaker_session_id: &str,
    experimental: bool,
) -> Occ2SendPolicy {
    let local = OccCapability::here();
    // Negotiate once for the strike lookup key; select_with_policy
    // re-negotiates identically inside.
    let Some((dialect_id, version)) = peer.and_then(|peer| local.negotiate(peer)) else {
        return Occ2SendPolicy::disabled();
    };
    let registry = crate::core::occ::dialect_load::promoted_registry()
        .unwrap_or_else(|_| DialectRegistry::with_sym_v1());
    let now_ms = unix_now_ms();
    let strikes = speaker_session_file
        .map(|file| {
            occ2_strike_pair_incidents(
                file,
                speaker_session_id,
                listener,
                (&dialect_id, version),
                now_ms,
            )
        })
        .unwrap_or(0);
    let evaluations = speaker_session_file
        .map(load_occ2_evaluations)
        .unwrap_or_default();
    let context = SelectionContext {
        local: &local,
        listener: peer,
        speaker_model,
        listener_model: None,
        registry: &registry,
        evaluations: &evaluations,
        message_chars: message.chars().count(),
        kind: occ2_message_kind_hint(message),
        // The sender cannot cheaply know the listener's transcript;
        // conservative false lets the amortization gate decide via
        // messages_remaining.
        codebook_in_context: false,
        strikes,
        messages_remaining: crate::core::occ::selection::EXPECTED_REMAINING_MESSAGES,
    };
    occ2_send_policy_from_selection(&context, experimental).policy
}

/// Build the OCC-3 send policy for one message (the capture side of
/// the seam; see `Occ3SendPolicy`).
///
/// Inputs are resolved by the caller (session settings, the sender
/// model's `occ3` configuration, the live connector registry) so this
/// function stays offline-testable. A `None` model block, a disabled
/// or not-yet-live-verified configuration, a missing connector or any
/// capture failure all return the disabled policy: the message goes
/// out as OCC-1/OCC-2 exactly as before (rule R1: never lost).
///
/// `capture` is the connector call: it exists so tests can substitute
/// the mock connector while production passes the registry-backed
/// closure.
pub async fn occ3_send_policy_for_send<F, Fut>(
    speaker_session_file: Option<&str>,
    model_occ3: Option<&pi_ai::types::Occ3Capability>,
    listener: &str,
    message: &str,
    capture: F,
) -> Occ3SendPolicy
where
    F: FnOnce(crate::core::occ::occ3::connector::CaptureRequest) -> Fut,
    Fut: std::future::Future<
        Output = Result<
            crate::core::occ::occ3::types::Occ3HandleRef,
            crate::core::occ::occ3::types::Occ3Error,
        >,
    >,
{
    let _ = (speaker_session_file, model_occ3, listener, message, capture);
    // Explicitly unsupported for the ordinary kernel host route. A plaintext
    // message cannot retroactively capture its original accepted hidden states.
    Occ3SendPolicy::disabled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn occ3_policy_disabled_without_model_block_or_optin() {
        // No model block: disabled.
        let policy = occ3_send_policy_for_send(Some("session-file"), None, "listener", "msg", |_| async { unreachable!() }).await;
        assert!(!policy.enabled);
        // Model block not live-verified: disabled.
        let config = pi_ai::types::Occ3Capability {
            protocol: "occ3-v1".into(),
            connector: "sglang".into(),
            endpoint: "http://localhost:30001".into(),
            api_version: "v1".into(),
            same_model_only: true,
            enabled: true,
            validation: pi_ai::types::Occ3Validation::Unverified,
            fallback: vec![],
            method: None,
            artifact_digest: None,
        };
        let policy = occ3_send_policy_for_send(Some("session-file"), Some(&config), "listener", "msg", |_| async { unreachable!() }).await;
        assert!(!policy.enabled);
    }

    #[tokio::test]
    async fn occ3_kernel_policy_never_recaptures_text_even_if_profile_claims_live() {
        let config = pi_ai::types::Occ3Capability {
            protocol: "occ3-v1".into(),
            connector: "sglang".into(),
            endpoint: "http://localhost:30001".into(),
            api_version: "v1".into(),
            same_model_only: true,
            enabled: true,
            validation: pi_ai::types::Occ3Validation::LiveVerified,
            fallback: vec![],
            method: None,
            artifact_digest: None,
        };
        // With a temp session file that has occ3 enabled.
        let dir = std::env::temp_dir().join(format!("occ3-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let session_file = dir.join("test-session.jsonl");
        std::fs::write(&session_file, b"{}").unwrap();
        crate::core::occ::session_settings::save_occ3(session_file.to_str().unwrap(), true, true).unwrap();

        let handle = crate::core::occ::occ3::types::Occ3HandleRef {
            handle_id: "chan-1".into(),
            connector_id: "sglang".into(),
            model_fingerprint: "fp".into(),
            runtime_generation: 1,
            expires_at_epoch: 0,
            state_size_bytes: 64,
        };
        let expected = handle.clone();
        let policy = occ3_send_policy_for_send(
            session_file.to_str(),
            Some(&config),
            "listener-1",
            "carry me",
            |request| async move {
                let _ = (request, handle);
                panic!("legacy kernel route must not recapture plaintext")
            },
        )
        .await;
        let _ = expected;
        assert!(!policy.enabled);
        assert!(policy.handle.is_none());

        // The wrap produces an OCC/3 envelope wire body.
        let wire = wrap_outgoing_body_occ3("base body", &policy);
        assert_eq!(wire, "base body");
        assert!(wire.chars().count() <= policy.max_message_chars);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn occ3_policy_falls_back_on_capture_failure() {
        let config = pi_ai::types::Occ3Capability {
            protocol: "occ3-v1".into(),
            connector: "sglang".into(),
            endpoint: "http://localhost:30001".into(),
            api_version: "v1".into(),
            same_model_only: true,
            enabled: true,
            validation: pi_ai::types::Occ3Validation::LiveVerified,
            fallback: vec![],
            method: None,
            artifact_digest: None,
        };
        let dir = std::env::temp_dir().join(format!("occ3-test-fail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Unique session identity per test: the preference store keys
        // artifacts off the session file, so same-named files collide.
        let session_file = dir.join("fail-session.jsonl");
        std::fs::write(&session_file, b"{}").unwrap();
        crate::core::occ::session_settings::save_occ3(session_file.to_str().unwrap(), true, true).unwrap();

        let policy = occ3_send_policy_for_send(
            session_file.to_str(),
            Some(&config),
            "listener",
            "msg",
            |_request| async {
                Err(crate::core::occ::occ3::types::Occ3Error::ConnectorUnreachable {
                    connector_id: "sglang".into(),
                })
            },
        )
        .await;
        assert!(!policy.enabled); // capture failure -> OCC-1/2 fallback

        std::fs::remove_dir_all(&dir).ok();
    }


    fn endpoint(name: Option<&str>, active: &str, session: &str) -> AgentSessionMessageEndpoint {
        AgentSessionMessageEndpoint {
            active_session_id: active.to_string(),
            session_id: session.to_string(),
            session_name: name.map(str::to_string),
            runtime_kind: None,
        }
    }

    fn payload() -> AgentSessionMessagePayload {
        AgentSessionMessagePayload {
            id: "agentmsg_1".to_string(),
            source: AGENT_MESSAGE_SOURCE.to_string(),
            message: "hello".to_string(),
            from: Some(AgentSessionMessageSender {
                active_session_id: Some("active-1".to_string()),
                session_id: Some("session-1".to_string()),
                session_name: Some("parent".to_string()),
                ..Default::default()
            }),
            from_relationship: None,
            target: endpoint(Some("child"), "active-2", "session-2"),
        }
    }

    #[test]
    fn participant_format_falls_back_to_unknown() {
        assert_eq!(
            format_agent_message_participant(
                &AGENT_MESSAGE_DIRECTION_RECEIVED.to_string(),
                None,
                None
            ),
            "from unknown"
        );
        let sender = AgentSessionMessageSender {
            session_name: Some("  name  ".to_string()),
            ..Default::default()
        };
        assert_eq!(
            format_agent_message_participant(
                &AGENT_MESSAGE_DIRECTION_SENT.to_string(),
                Some(&FAMILY_RELATIONSHIP_CHILD.to_string()),
                Some(&sender)
            ),
            "to child name"
        );
    }

    #[test]
    fn prompt_round_trips_through_the_id_parser() {
        let prompt = create_agent_session_message_prompt(&payload());
        assert!(is_agent_session_message_prompt(&prompt));
        assert_eq!(
            parse_agent_session_message_prompt_id(&prompt).as_deref(),
            Some("agentmsg_1")
        );
    }

    #[test]
    fn prompt_with_relationship_line_keeps_the_id_offsets() {
        let mut sibling = payload();
        sibling.from_relationship = Some(FAMILY_RELATIONSHIP_SIBLING.to_string());
        let prompt = create_agent_session_message_prompt(&sibling);
        assert!(prompt.starts_with("[from sibling:parent]"));
        assert_eq!(
            parse_agent_session_message_prompt_id(&prompt).as_deref(),
            Some("agentmsg_1")
        );
        let parent = {
            let mut parent_payload = payload();
            parent_payload.from_relationship = Some(FAMILY_RELATIONSHIP_PARENT.to_string());
            create_agent_session_message_prompt(&parent_payload)
        };
        assert!(parent.starts_with("[from parent]"));
        assert_eq!(
            parse_agent_session_message_prompt_id(&parent).as_deref(),
            Some("agentmsg_1")
        );
    }

    #[test]
    fn prompt_parser_rejects_unrelated_text() {
        assert!(!is_agent_session_message_prompt("hello"));
        assert!(parse_agent_session_message_prompt_id(
            "Agent-to-agent message received.\nSource: other\nTo: x\nMessage id: agentmsg_1"
        )
        .is_none());
    }

    #[test]
    fn message_detection_uses_details_shape() {
        let message: AgentMessage = crate::core::messages::custom_message_to_agent_message(
            match create_agent_session_message(&payload(), 5) {
                CustomAgentMessage::Custom {
                    custom_type,
                    content,
                    display,
                    details,
                    timestamp,
                } => crate::core::messages::CustomMessage {
                    role: "custom".to_string(),
                    custom_type,
                    content,
                    display,
                    details,
                    timestamp,
                },
                _ => unreachable!("agent session messages are custom messages"),
            },
        );
        assert!(is_agent_session_message(&message));
        assert!(starts_agent_run(&message));
        let other = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "other".to_string(),
            content: CustomMessageContent::Text("x".to_string()),
            display: true,
            details: None,
            timestamp: 1,
        });
        assert!(!is_agent_session_message(&other));
        assert!(!starts_agent_run(&other));
    }

    #[test]
    fn receipt_marks_delivered_and_queued_timestamps() {
        let delivered = create_agent_session_message_receipt(
            &payload(),
            &DELIVERY_STATUS_DELIVERED.to_string(),
            "2026-01-01T00:00:00.000Z",
        );
        assert_eq!(
            delivered.delivered_at.as_deref(),
            Some("2026-01-01T00:00:00.000Z")
        );
        assert!(delivered.queued_at.is_none());
        assert_eq!(delivered.delivery_mode.as_deref(), Some("steer"));
        let queued = create_agent_session_message_receipt(
            &payload(),
            &DELIVERY_STATUS_QUEUED.to_string(),
            "2026-01-01T00:00:00.000Z",
        );
        assert!(queued.delivered_at.is_none());
        assert!(queued.queued_at.is_some());
    }

    #[test]
    fn message_normalisation_enforces_limits() {
        assert_eq!(normalize_agent_session_message("  hi  ", 10).unwrap(), "hi");
        assert!(normalize_agent_session_message("   ", 10).is_err());
        assert_eq!(
            normalize_agent_session_message("abcd", 3).unwrap_err(),
            "Agent session message is too long: 4 chars exceeds 3"
        );
    }

    #[test]
    fn direct_target_rejects_broadcast() {
        assert_eq!(
            assert_direct_agent_message_target(" sub-1 ").unwrap(),
            "sub-1"
        );
        assert!(assert_direct_agent_message_target("").is_err());
        assert!(assert_direct_agent_message_target("*").is_err());
        assert!(assert_direct_agent_message_target("ALL").is_err());
        assert!(assert_direct_agent_message_target("Broadcast").is_err());
    }

    #[test]
    fn queue_capacity_matches_limit_message() {
        assert!(assert_agent_message_queue_capacity(19.0, 20).is_ok());
        assert_eq!(
            assert_agent_message_queue_capacity(20.0, 20).unwrap_err(),
            "Target session has too many pending messages: 20 unfinished, limit is 20"
        );
    }

    fn entry(id: &str, name: &str, depth: f64, parent_id: Option<&str>) -> AgentFamilyCatalogEntry {
        AgentFamilyCatalogEntry {
            id: id.to_string(),
            name: Some(name.to_string()),
            depth,
            status: FAMILY_STATUS_RUNNING.to_string(),
            replied_since_task: Some(false),
            parent_session_id: parent_id.map(str::to_string),
            parent_session_path: None,
            session_path: Some(format!("/sessions/{id}.jsonl")),
        }
    }

    #[test]
    fn relationship_covers_parent_child_sibling_and_none() {
        let current = entry("self", "self", 1.0, Some("parent"));
        let parent = entry("parent", "parent", 0.0, None);
        let sibling = entry("sib", "sib", 1.0, Some("parent"));
        let child = entry("kid", "kid", 2.0, Some("self"));
        assert_eq!(
            agent_family_relationship(&current, &parent).as_deref(),
            Some("parent")
        );
        assert_eq!(
            agent_family_relationship(&current, &child).as_deref(),
            Some("child")
        );
        assert_eq!(
            agent_family_relationship(&current, &sibling).as_deref(),
            Some("sibling")
        );
        assert!(agent_family_relationship(&current, &current).is_none());
        assert!(
            agent_family_relationship(&current, &entry("far", "far", 3.0, Some("other"))).is_none()
        );
        assert_eq!(
            assert_agent_family_reach(&current, &entry("far", "far", 3.0, Some("other")))
                .unwrap_err(),
            AGENT_FAMILY_REACH_ERROR
        );
    }

    #[test]
    fn roster_orders_parent_siblings_then_children() {
        let current = entry("self", "self", 1.0, Some("parent"));
        let catalog = vec![
            current.clone(),
            entry("parent", "parent", 0.0, None),
            entry("sib-b", "b", 1.0, Some("parent")),
            entry("sib-a", "a", 1.0, Some("parent")),
            entry("kid", "kid", 2.0, Some("self")),
        ];
        let roster = build_agent_family_roster(&current, &catalog);
        assert_eq!(roster.current.name, "self");
        let relationships: Vec<&str> = roster
            .entries
            .iter()
            .map(|entry| entry.relationship.as_str())
            .collect();
        assert_eq!(relationships, vec!["parent", "sibling", "sibling", "child"]);
        assert_eq!(roster.entries[1].name, "a");
        assert_eq!(roster.entries[3].replied_since_task, Some(false));
        assert!(roster.entries[0].replied_since_task.is_none());
    }

    #[test]
    fn name_availability_reports_the_conflict_message() {
        let catalog = vec![entry("self", "taken", 1.0, Some("parent"))];
        let input = AgentSessionNameAvailabilityInput {
            parent_session_id: Some("parent".to_string()),
            parent_session_path: None,
            depth: 1.0,
            name: "taken".to_string(),
            ignore_session_id: None,
        };
        assert_eq!(
            assert_agent_session_name_available(&catalog, &input).unwrap_err(),
            "Agent name \"taken\" is unavailable: an agent of that name already exists at depth 1 under this parent"
        );
        let ignoring = AgentSessionNameAvailabilityInput {
            ignore_session_id: Some("self".to_string()),
            ..input.clone()
        };
        assert!(assert_agent_session_name_available(&catalog, &ignoring).is_ok());
    }

    #[test]
    fn reservation_key_encodes_depth_parent_and_name() {
        let scope = AgentSessionNameScope {
            parent_session_id: Some("parent".to_string()),
            parent_session_path: None,
            depth: 1.0,
        };
        assert_eq!(
            session_name_reservation_key(&scope, "kid"),
            "[1,\"id\",\"parent\",\"kid\"]"
        );
        let root = AgentSessionNameScope {
            parent_session_id: None,
            parent_session_path: None,
            depth: 0.0,
        };
        assert_eq!(
            session_name_reservation_key(&root, "kid"),
            "[0,\"root\",\"\",\"kid\"]"
        );
    }

    #[test]
    fn rate_limiter_refills_and_reports_retry_after() {
        let now = std::sync::Arc::new(std::sync::Mutex::new(0.0_f64));
        let clock = now.clone();
        let mut limiter = AgentSessionMessageRateLimiter::new(
            3.0,
            1000.0,
            Some(Box::new(move || *clock.lock().unwrap())),
        );
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        assert_eq!(
            limiter.try_consume("k"),
            RateLimitResult::Exceeds {
                retry_after_ms: 1000.0
            }
        );
        limiter.refund("k");
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        *now.lock().unwrap() = 1000.0;
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        limiter.clear_matching(|key| key == "k");
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
        limiter.clear(None);
        assert_eq!(limiter.try_consume("k"), RateLimitResult::Ok);
    }

    #[test]
    fn message_id_uses_the_agentmsg_prefix() {
        assert!(create_agent_session_message_id().starts_with("agentmsg_"));
        assert!(serde_json::to_value(AGENT_MESSAGE_CUSTOM_TYPE).is_ok());
        let _ = json!({});
    }

    // -- OCC-2 seam unit tests (u2/u3 of the integration design) --------

    fn occ1_contract_body() -> String {
        let message = crate::core::occ::contract::AgentCommMessage::new("TASK", "t1", "pending");
        message
            .to_occ1()
            .expect("to_occ1")
            .encode()
            .expect("encode")
    }

    fn enabled_policy() -> Occ2SendPolicy {
        Occ2SendPolicy::for_dialect(
            DialectRegistry::with_sym_v1(),
            ("sym-v1".to_string(), 1),
            DEFAULT_AGENT_MESSAGE_MAX_CHARS,
        )
    }

    #[test]
    fn occ2_disabled_policy_is_the_identity() {
        let body = occ1_contract_body();
        assert_eq!(wrap_outgoing_body(&body, &Occ2SendPolicy::disabled()), body);
        assert_eq!(wrap_outgoing_body("hello", &enabled_policy()), "hello");
        // Enabled but no negotiated dialect: identity.
        let mut policy = enabled_policy();
        policy.dialect = None;
        assert_eq!(wrap_outgoing_body(&body, &policy), body);
    }

    #[test]
    fn occ2_wrap_encodes_contract_bodies_as_dual() {
        let body = occ1_contract_body();
        let dual = wrap_outgoing_body(&body, &enabled_policy());
        assert!(dual.starts_with("OCC/2 D=sym-v1@1 C="), "dual header");
        assert!(dual.contains("\n#FALLBACK-OCC1\n"), "sentinel section");
        assert!(dual.ends_with(body.trim_end_matches('\n')) || dual.contains(&body));
        let registry = DialectRegistry::with_sym_v1();
        let decoded = crate::core::occ::occ2::decode(&dual, &registry).expect("strict decode");
        assert_eq!(
            decoded,
            crate::core::occ::contract::AgentCommMessage::from_occ1(
                &Occ1Message::decode(&body).expect("decode")
            )
            .expect("from_occ1")
        );
    }

    #[test]
    fn occ2_wrap_accepts_file_evidence_kind() {
        // Regression: agents author `E1=file:<path>` for on-disk references
        // far more often than `E1=src:<repo path>`. The evidence-kind
        // allowlist rejected `file`, so the strict contract decode failed
        // and the wrap silently passed the whole body through as OCC-1.
        let body = "OCC/1\nTASK=occ2-wrap-verify\nKIND=RESULT\nSTATUS=done\nF1=transport verification probe fired at parent request from a fresh child on the current build\nF2=parent will inspect the delivery record for the occ2Delivery kind field in its own session transcript\nF3=body authored as a complete OCC-1 contract with task kind status findings and evidence\nE1=file:C:\\Users\\openclawuser\\profile\\tmp\\probe.md\nEL1=1-20\nNEXT=parent inspects the received payload header and reports whether native OCC-2 wrapping engaged\n";
        let dual = wrap_outgoing_body(body, &enabled_policy());
        assert!(dual.starts_with("OCC/2 D=sym-v1@1 C="), "must wrap");
        assert!(dual.contains("\n#FALLBACK-OCC1\n"), "sentinel section");
        let registry = DialectRegistry::with_sym_v1();
        let decoded = crate::core::occ::occ2::decode(&dual, &registry).expect("strict decode");
        assert_eq!(decoded.evidence[0].kind, "file");
        // Unknown evidence kinds are still not contracts: no wrap.
        let bogus = body.replace("E1=file:", "E1=screenshot:");
        assert_eq!(wrap_outgoing_body(&bogus, &enabled_policy()), bogus);
    }

    #[test]
    fn occ2_wrap_leaves_plain_english_and_deltas_unchanged() {
        // Plain English never parses as an OCC-1 contract message.
        assert_eq!(wrap_outgoing_body("plain english body", &enabled_policy()), "plain english body");
        // A delta body (no KIND) is excluded from wrapping in v1 (OQ4).
        let delta_body = "OCC/1\nTASK=t\nSEQ=2\nBASE=1\nSTATUS=done";
        assert_eq!(wrap_outgoing_body(delta_body, &enabled_policy()), delta_body);
        // Malformed OCC-1 passes through unchanged.
        let malformed = "OCC/1\nTASK=t\nKIND=RESULT\nSTATUS=weird";
        assert_eq!(wrap_outgoing_body(malformed, &enabled_policy()), malformed);
    }

    #[test]
    fn occ2_size_guard_sends_plain_occ1_instead_of_truncating() {
        // A body near the transport bound: its dual form would exceed the
        // bound, so the original OCC-1 body is sent, never truncated.
        let mut message =
            crate::core::occ::contract::AgentCommMessage::new("RESULT", "t-big", "done");
        for index in 0..6 {
            message.findings.push(crate::core::occ::contract::Finding {
                id: format!("F{}", index + 1),
                statement: "x".repeat(2_500),
                confidence: None,
                evidence: Vec::new(),
            });
        }
        let body = message.to_occ1().expect("to_occ1").encode().expect("encode");
        assert!(body.chars().count() <= DEFAULT_AGENT_MESSAGE_MAX_CHARS);
        let dual = crate::core::occ::occ2::encode_dual(
            &message,
            DialectRegistry::with_sym_v1().get("sym-v1", 1).expect("card"),
        )
        .expect("encode_dual");
        assert!(dual.chars().count() > DEFAULT_AGENT_MESSAGE_MAX_CHARS);
        assert_eq!(wrap_outgoing_body(&body, &enabled_policy()), body);
    }

    #[test]
    fn occ2_prompt_shows_symbolic_section_and_record_keeps_dual() {
        let body = occ1_contract_body();
        let dual = wrap_outgoing_body(&body, &enabled_policy());
        let mut with_dual = payload();
        with_dual.message = dual.clone();
        let prompt = create_agent_session_message_prompt(&with_dual);
        assert!(prompt.starts_with("Agent-to-agent message received.\nSource: agent_message\n"));
        assert!(prompt.contains("OCC/2 D=sym-v1@1 C="));
        assert!(!prompt.contains("#FALLBACK-OCC1"));
        assert!(!prompt.contains("OCC/1\n"));
        assert_eq!(
            parse_agent_session_message_prompt_id(&prompt).as_deref(),
            Some("agentmsg_1")
        );
        // The durable record keeps the dual payload verbatim.
        let record = create_agent_session_message(&with_dual, 5);
        let CustomAgentMessage::Custom { details: Some(details), .. } = &record else {
            panic!("agent session message is a custom message");
        };
        let stored: AgentSessionMessageDetails = serde_json::from_value(details.clone()).unwrap();
        assert_eq!(stored.message, dual);
        // A plain body renders byte-identically to the pre-OCC-2 format.
        let mut plain = payload();
        plain.message = body.clone();
        let expected = format!(
            "Agent-to-agent message received.\nSource: {}\nFrom: parent, active active-1, session session-1\nTo: child, active active-2, session session-2\nMessage id: agentmsg_1\n\n{}",
            AGENT_MESSAGE_SOURCE,
            body
        );
        assert_eq!(create_agent_session_message_prompt(&plain), expected);
    }

    #[test]
    fn occ2_prompt_uses_valid_fallback_on_bad_dialect_and_teaches_valid_symbols() {
        let body = occ1_contract_body();
        let dual = wrap_outgoing_body(&body, &enabled_policy());
        let rendered = agent_session_message_prompt_body(&dual);
        assert!(rendered.contains("T:task_id"));
        assert!(rendered.contains("OCC/2 D=sym-v1@1"));
        let unknown = dual.replace("D=sym-v1@1", "D=unknown@1");
        assert_eq!(agent_session_message_prompt_body(&unknown), body);
        let broken = "OCC/2 invalid\n#FALLBACK-OCC1\ninvalid";
        assert_eq!(agent_session_message_prompt_body(broken), broken);
    }

    #[test]
    fn occ2_decode_strict_then_fallback_with_three_strikes() {
        let body = occ1_contract_body();
        let dual = wrap_outgoing_body(&body, &enabled_policy());
        let empty_registry = DialectRegistry::new();
        let mut ledger = Occ2StrikeLedger::new();
        let pair = "sender->listener";
        for round in 1..=OCC2_STRIKE_LIMIT {
            let decoded =
                decode_agent_session_message_body(&dual, &empty_registry, pair, &mut ledger)
                    .expect("fallback decodes the embedded OCC-1");
            assert!(!decoded.is_symbolic(), "unknown dialect consumes via fallback");
            assert_eq!(ledger.incidents(pair), round);
        }
        assert!(ledger.is_disabled(pair));
        assert_eq!(ledger.disabled_pairs(), vec![pair]);
        // Disabled pairs decode straight through the fallback, no more strikes.
        let decoded =
            decode_agent_session_message_body(&dual, &empty_registry, pair, &mut ledger).unwrap();
        assert!(!decoded.is_symbolic());
        assert_eq!(ledger.incidents(pair), OCC2_STRIKE_LIMIT);
        // With the dialect registered the strict path wins and costs nothing.
        let registry = DialectRegistry::with_sym_v1();
        let mut fresh = Occ2StrikeLedger::new();
        let decoded = decode_agent_session_message_body(&dual, &registry, pair, &mut fresh).unwrap();
        assert!(decoded.is_symbolic());
        assert_eq!(fresh.incidents(pair), 0);
    }

    /// A minimal `AgentSessionMessageController` that records sends, for
    /// host-handler seam tests.
    struct RecordingController {
        sent: std::sync::Mutex<Vec<(String, String)>>,
        occ2: bool,
    }

    impl AgentSessionMessageController for RecordingController {
        fn roster(
            &self,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<AgentFamilyRosterResult, String>> + Send>,
        > {
            let entries = vec![
                AgentFamilyRosterEntry {
                    relationship: FAMILY_RELATIONSHIP_PARENT.to_string(),
                    name: "parent".to_string(),
                    id: "parent-id".to_string(),
                    depth: 0.0,
                    status: FAMILY_STATUS_RUNNING.to_string(),
                    replied_since_task: None,
                },
                AgentFamilyRosterEntry {
                    relationship: FAMILY_RELATIONSHIP_SIBLING.to_string(),
                    name: "sibling".to_string(),
                    id: "sibling-id".to_string(),
                    depth: 1.0,
                    status: FAMILY_STATUS_RUNNING.to_string(),
                    replied_since_task: None,
                },
            ];
            Box::pin(async move { Ok(AgentFamilyRosterResult { current: AgentFamilyRosterCurrent { name: "self".into(), id: "self-id".into(), depth: 1.0 }, entries }) })
        }

        fn await_pending_child_publication(
            &self,
            _selector: String,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send>,
        > {
            Box::pin(async move { Ok(None) })
        }

        fn send_agent_message(
            &self,
            input: AgentSessionMessageSendInput,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<AgentSessionMessageReceipt, String>> + Send>,
        > {
            self.sent
                .lock()
                .unwrap()
                .push((input.target.clone(), input.message.clone()));
            Box::pin(async move {
                Ok(AgentSessionMessageReceipt {
                    id: create_agent_session_message_id(),
                    source: AGENT_MESSAGE_SOURCE.to_string(),
                    target: AgentSessionMessageEndpoint {
                        active_session_id: input.target.clone(),
                        session_id: input.target,
                        ..Default::default()
                    },
                    from: None,
                    message: input.message,
                    delivery_status: DELIVERY_STATUS_DELIVERED.to_string(),
                    delivered_at: None,
                    queued_at: None,
                    delivery_mode: None,
                })
            })
        }

        fn occ2_send_policy(&self, _target: &str, _message: &str) -> pi_ai::types::BoxFuture<Occ2SendPolicy> {
            let enabled = self.occ2;
            Box::pin(async move {
                if enabled {
                    Occ2SendPolicy::for_dialect(
                        DialectRegistry::with_sym_v1(),
                        ("sym-v1".to_string(), 1),
                        DEFAULT_AGENT_MESSAGE_MAX_CHARS,
                    )
                } else {
                    Occ2SendPolicy::disabled()
                }
            })
        }
    }

    async fn run_send_handler(
        controller: Arc<RecordingController>,
        payload: Value,
    ) -> Result<Value, crate::core::kernel::shared::KernelError> {
        let handlers = create_agent_message_host_handlers(controller);
        let send = handlers.get("agent_message.send").expect("send handler");
        send(payload).await
    }

    #[tokio::test]
    async fn occ2_host_send_seam_wraps_only_when_enabled() {
        let body = occ1_contract_body();

        // Disabled: the direct arm sends the body byte-identically.
        let disabled = Arc::new(RecordingController {
            sent: std::sync::Mutex::new(Vec::new()),
            occ2: false,
        });
        run_send_handler(
            disabled.clone(),
            serde_json::json!({ "message": body, "receiver_role": "parent" }),
        )
        .await
        .expect("disabled send succeeds");
        assert_eq!(disabled.sent.lock().unwrap()[0].1, body, "no wrapping when disabled");

        // Enabled: the direct arm sends the dual payload.
        let enabled = Arc::new(RecordingController {
            sent: std::sync::Mutex::new(Vec::new()),
            occ2: true,
        });
        run_send_handler(
            enabled.clone(),
            serde_json::json!({ "message": body, "receiver_role": "parent" }),
        )
        .await
        .expect("enabled send succeeds");
        let sent = enabled.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].1.starts_with("OCC/2 D=sym-v1@1 C="), "dual payload sent");
        assert_eq!(
            sent[0].1.matches("#FALLBACK-OCC1").count(),
            1,
            "exactly one sentinel: the wrap never compounds"
        );

        // Broadcast: every roster entry receives its own single wrap of the
        // original body (R6), never a re-wrap of a wrapped body.
        let broadcast = Arc::new(RecordingController {
            sent: std::sync::Mutex::new(Vec::new()),
            occ2: true,
        });
        let result = run_send_handler(
            broadcast.clone(),
            serde_json::json!({ "target": "all", "message": body }),
        )
        .await
        .expect("broadcast succeeds");
        let receipts = result
            .get("receipts")
            .and_then(Value::as_array)
            .expect("broadcast receipts");
        assert_eq!(receipts.len(), 2);
        let sent = broadcast.sent.lock().unwrap();
        assert_eq!(sent.len(), 2, "one send per roster entry");
        for (target, message) in sent.iter() {
            assert!(message.starts_with("OCC/2 D=sym-v1@1 C="), "{target}: dual payload");
            assert_eq!(
                message.matches("#FALLBACK-OCC1").count(),
                1,
                "{target}: no compound wrapping across broadcast entries"
            );
            assert!(message.contains(body.trim_end_matches('\n')), "{target}: OCC-1 section intact");
        }
    }

    #[test]
    fn occ2_negotiation_absence_keeps_occ1() {
        let local = OccCapability::here();
        let registry = DialectRegistry::with_sym_v1();
        assert_eq!(
            select_occ2_dialect(&local, Some(&local), &registry, true),
            Some(("sym-v1".to_string(), 1))
        );
        // No reported capability: OCC-1.
        assert_eq!(select_occ2_dialect(&local, None, &registry, true), None);
        // A peer that speaks OCC-1 only: OCC-1.
        let occ1_only = OccCapability { occ1: true, dialects: BTreeMap::new(), occ3: None};
        assert_eq!(select_occ2_dialect(&local, Some(&occ1_only), &registry, true), None);
        // Unevaluated cards are refused without the dev override.
        assert_eq!(select_occ2_dialect(&local, Some(&local), &registry, false), None);
        // The summary field is absent for sessions that report nothing.
        let absent = AgentSessionMessageAgentSummary::default();
        assert!(absent.occ_capability.is_none());
        let serialized = serde_json::to_value(&absent).unwrap();
        assert!(serialized.get("occCapability").is_none());
        let present = AgentSessionMessageAgentSummary {
            occ_capability: Some(OccCapability::here()),
            ..absent
        };
        let serialized = serde_json::to_value(&present).unwrap();
        assert_eq!(serialized["occCapability"]["occ1"], json!(true));
        let round: AgentSessionMessageAgentSummary =
            serde_json::from_value(serialized).unwrap();
        assert_eq!(round.occ_capability, Some(OccCapability::here()));
    }
}

#[cfg(test)]
mod agent_message_provenance_gate_tests {
    use super::*;
    use crate::core::kernel::shared::{HostRequestContext, KernelExecutionScope, KernelScopeState};
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;

    /// A minimal `AgentSessionMessageController` that records sends, for the
    /// provenance gate seam tests (mirrors the RecordingController shape).
    struct GateRecordingController {
        sent: std::sync::Mutex<Vec<(String, String)>>,
    }

    impl GateRecordingController {
        fn shared() -> Arc<Self> {
            Arc::new(GateRecordingController {
                sent: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl AgentSessionMessageController for GateRecordingController {
        fn roster(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<AgentFamilyRosterResult, String>> + Send>> {
            let entries = vec![AgentFamilyRosterEntry {
                relationship: FAMILY_RELATIONSHIP_PARENT.to_string(),
                name: "parent".to_string(),
                id: "parent-id".to_string(),
                depth: 0.0,
                status: FAMILY_STATUS_RUNNING.to_string(),
                replied_since_task: None,
            }];
            Box::pin(async move {
                Ok(AgentFamilyRosterResult {
                    current: AgentFamilyRosterCurrent {
                        name: "self".into(),
                        id: "self-id".into(),
                        depth: 1.0,
                    },
                    entries,
                })
            })
        }

        fn await_pending_child_publication(
            &self,
            _selector: String,
        ) -> Pin<Box<dyn Future<Output = Result<Option<String>, String>> + Send>> {
            Box::pin(async move { Ok(None) })
        }

        fn send_agent_message(
            &self,
            input: AgentSessionMessageSendInput,
        ) -> Pin<Box<dyn Future<Output = Result<AgentSessionMessageReceipt, String>> + Send>> {
            self.sent
                .lock()
                .unwrap()
                .push((input.target.clone(), input.message.clone()));
            Box::pin(async move {
                Ok(AgentSessionMessageReceipt {
                    id: create_agent_session_message_id(),
                    source: AGENT_MESSAGE_SOURCE.to_string(),
                    target: AgentSessionMessageEndpoint {
                        active_session_id: input.target.clone(),
                        session_id: input.target,
                        ..Default::default()
                    },
                    from: None,
                    message: input.message,
                    delivery_status: DELIVERY_STATUS_DELIVERED.to_string(),
                    delivered_at: None,
                    queued_at: None,
                    delivery_mode: None,
                })
            })
        }
    }

    fn ctx(
        token_present: bool,
        matched_scope: Option<Arc<KernelExecutionScope>>,
        live_enrolled: Option<Arc<KernelExecutionScope>>,
    ) -> HostRequestContext {
        HostRequestContext {
            request_id: "request-1".to_string(),
            token_present,
            matched_scope,
            live_enrolled,
            kernel_generation: 1,
            cell_source_code: None,
        }
    }

    fn armed_scope() -> Arc<KernelExecutionScope> {
        let scope = KernelExecutionScope::new_admitted("call-1", "ipython");
        scope.arm(1).expect("arm");
        scope
    }

    fn payload() -> Value {
        json!({
            "message": "hello",
            "receiver_role": "parent",
        })
    }

    #[tokio::test]
    async fn absent_token_with_nothing_enrolled_is_ordinary() {
        let controller = GateRecordingController::shared();
        let result = handle_agent_message_send_provenance(
            controller.clone(),
            ctx(false, None, None),
            payload(),
        )
        .await;
        assert!(result.is_ok(), "ordinary route must not change: {result:?}");
        assert_eq!(controller.sent.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn present_unknown_token_refuses_even_with_no_enrolled_scope() {
        let controller = GateRecordingController::shared();
        let result =
            handle_agent_message_send_provenance(controller.clone(), ctx(true, None, None), payload())
                .await;
        let error = result.err().expect("a present unknown token must refuse");
        assert!(error.to_string().contains("unknown execution token"));
        assert!(controller.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn absent_token_while_enrolled_refuses_and_tombstones() {
        let controller = GateRecordingController::shared();
        let scope = armed_scope();
        let result = handle_agent_message_send_provenance(
            controller.clone(),
            ctx(false, None, Some(scope.clone())),
            payload(),
        )
        .await;
        let error = result.err().expect("enrolled executions require the token");
        assert!(error.to_string().contains("provenance required"));
        assert_eq!(scope.state(), KernelScopeState::Tombstoned);
        assert!(controller.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn matched_token_claims_then_refuses_before_publication() {
        let controller = GateRecordingController::shared();
        let scope = armed_scope();
        let result = handle_agent_message_send_provenance(
            controller.clone(),
            ctx(true, Some(scope.clone()), Some(scope.clone())),
            payload(),
        )
        .await;
        let error = result.err().expect("Phase C is not implemented in this build");
        assert!(error.to_string().contains("not enrolled"));
        assert_eq!(scope.state(), KernelScopeState::Claimed);
        assert!(controller.sent.lock().unwrap().is_empty());
        // A retry with the same token can never claim again: one-shot.
        let retry = handle_agent_message_send_provenance(
            controller.clone(),
            ctx(true, Some(scope.clone()), Some(scope.clone())),
            payload(),
        )
        .await;
        let error = retry.err().expect("the claim is consumed");
        assert!(error.to_string().contains("did not match"));
        assert_eq!(scope.state(), KernelScopeState::Tombstoned);
        assert!(controller.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_token_matching_a_different_scope_tombstones_the_enrolled_one() {
        let controller = GateRecordingController::shared();
        let enrolled = armed_scope();
        let other = armed_scope();
        let result = handle_agent_message_send_provenance(
            controller.clone(),
            ctx(true, Some(other.clone()), Some(enrolled.clone())),
            payload(),
        )
        .await;
        let error = result.err().expect("only the enrolled scope's token matches");
        assert!(error.to_string().contains("did not match"));
        assert_eq!(enrolled.state(), KernelScopeState::Tombstoned);
        assert!(controller.sent.lock().unwrap().is_empty());
    }
}


