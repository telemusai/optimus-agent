//! Port of packages/coding-agent/src/modes/daemon/daemon-protocol.ts
//!
//! Local daemon JSONL protocol.
//!
//! This is the transport used by DaemonAgentConnection today, not the final
//! remote gateway protocol. The protocol primitives below are intentionally
//! JSON-serializable so a future gateway can wrap or proxy this local transport
//! without leaking transport details back into InteractiveMode.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{ImageContent, TextContent};

use super::daemon_errors::DaemonErrorInfo;
use crate::core::agent_messages::{
    AgentSessionMessageDeliveryMode, AgentSessionMessageReceipt, AgentSessionMessageSafetyStatus,
};
use crate::core::agent_session::{AgentAutonomousStatus, AgentCronJob, SessionActionRecoverySnapshot};
use crate::core::bash_executor::BashResult;
use crate::core::mcp::acp_mcp_types::AcpMcpServerConfig;
use crate::core::messages::CustomMessage;
use crate::core::session_action_store::{InputSource, QueuedMessageLane, QueuedMessageMutation};
use crate::core::session_file_actions::DeleteSessionFileResult;
use crate::core::usage::SessionUsageSummary;
use crate::modes::agent_connection::types::{
    AgentConnectionAgentStatus, AgentConnectionHeartbeat,
    AgentConnectionResourceSnapshot, AgentConnectionRlmChildAgentSnapshot,
    AgentConnectionSavedSessionState, AgentConnectionScopedModel, AgentConnectionSessionContext,
    AgentConnectionSessionEvent, AgentConnectionSessionHeader, AgentConnectionSessionTree,
    AgentConnectionSessionTreeNode, AgentConnectionSideQuestionEvent, AgentConnectionSideQuestionTurn,
    AgentConnectionState,
};
use crate::modes::daemon::agent_roster::AgentRosterEntry;
use crate::modes::daemon::daemon_session_list::SessionSummary;

pub const DAEMON_PROTOCOL_NAME: &str = "prime-agent.daemon";
pub const DAEMON_PROTOCOL_VERSION: u32 = 7;
pub const DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION: u32 = 7;
// Revision 9 publishes persisted RLM spawn depth on passive session rows.
// Revision 10 publishes persisted RLM spawn depth on all session catalog rows.
// Revision 11 adds immediate get/set commands for active-session RLM max depth.
// Revision 12 publishes idle-residency metadata on session summary rows.
// Revision 13 narrows agent-origin reach and roster wire shapes to the nuclear family.
// Revision 14 carries the client's monotonic telemetry opt-out on attach and reattach.
// Revision 15 adds the mutate_queued_message command and queue_message_mutation capability.
// Revision 16 adds the "stopping" workerState and stops reporting disconnected workers as "ready".
// Revision 17 gates authoritative child rosters and transient owned-session recovery context.
// Revision 18 adds the opt-in RLM quiescence barrier to headless completion.
// Revision 19 adds daemon-held session input pauses.
// Revision 20 lets cancellation target a prompt the session owns but has not started.
// Revision 21 adds capability-gated, session-scoped ACP MCP server replacement.
// Revision 23 lets workers query the supervisor agent roster on demand.
// Revision 24 adds the capability-gated agent-roster subscription and push.
// Revision 25 adds capability-gated direct worker peer transport discovery.
// Revision 26 publishes own-session usage totals on session summary and saved-session rows.
// Revision 27 adds structured session_recovering failure info for known-but-unaddressable sessions.
// Revision 28 adds optional providerContext to compaction summary messages and maxInputTokens to models.
// Both are backward-compatible response metadata; older clients render the existing summary text.
// Revision 29 adds capability-gated, pinned recent-first history windows and older range reads.
// Revision 30 gates combined Jev mode with jev_features. Feature and compaction
// settings are optional response metadata; legacy Jev commands and events remain compatible.
// Revision 31 adds optional session-local Jev usage to jev_get_status responses.
// The existing jev_control capability gates reads; missing usage degrades locally.
// Revision 32 adds optional typed tool isError / executionReports metadata.
// Native lifecycle host requests are separately capability-gated and never required at startup.
// Revision 33 advertises optional jev_dynamic tool support; existing commands/events are unchanged.
// Revision 34 gates /mode session commands; existing response/event shapes are unchanged.
pub const DAEMON_SCHEMA_REVISION: u32 = 34;
pub const DAEMON_SCHEMA_ID: &str = "protocol-7-schema-34-c16da0e12d5a";

pub type DaemonProtocolName = String;
pub type DaemonProtocolVersion = u32;
pub type DaemonCommandId = String;
pub type DaemonEventId = String;
pub type DaemonEventSequence = u64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonEventCursor {
    pub generation: String,
    pub sequence: DaemonEventSequence,
}

pub type DaemonClientId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonClientCapability {
    AttachSnapshot,
    EventSequence,
    ExtensionUi,
    SlimAttach,
    ChunkedSnapshot,
    HistoryRanges,
    ClientOwnedSessions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonPromptAdmissionCancellationStatus {
    Cancelled,
    Owned,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonPromptAdmissionCancellationResult {
    pub status: DaemonPromptAdmissionCancellationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonServerCapability {
    AttachSnapshot,
    EventSequence,
    ExtensionUi,
    SlimAttach,
    ChunkedSnapshot,
    HistoryRanges,
    ClientOwnedSessions,
    DeleteRlmSubagent,
    HeartbeatCatalog,
    HeartbeatManagement,
    ModelCatalog,
    // The daemon honors previousTurns on start_side_question (multi-turn side
    // conversations). Clients must check before sending follow-up transcripts.
    SideQuestionTranscript,
    // The daemon honors transient and runId on execute_bash (side-conversation
    // bash: never recorded into the session, and its bash_start/bash_end events
    // carry the transient marker and echoed runId so clients correlate runs by
    // identity). Clients must check before sending.
    TransientBash,
    AgentRoster,
    SessionInputAdmission,
    PromptAdmissionCancellation,
    QueueMessageMutation,
    AuthoritativeChildRoster,
    OwnedSessionRecoveryContext,
    RlmQuiescenceBarrier,
    SessionInputPause,
    OwnedPromptCancellation,
    AcpMcpServers,
    DirectPeerTransport,
    // SHARED FILE EDIT (modes/daemon/daemon_protocol.rs, capability-gated addition
    // by jev-ui lane; REPAIR-OVERLAP file - keep the coordinator's version at
    // integration): Jev comparison mode control. The three `jev_*` commands are
    // OPTIONAL. A daemon that does not advertise this capability never receives
    // them, and the client degrades to local (settings-file) mode control.
    JevControl,
    // Combined Compare + Active mode and optional feature/compaction settings metadata.
    // Clients must negotiate this before sending the combined mode or relying on its metadata.
    JevFeatures,
    /// Explicit agent-authored questions via the optional jev_decide tool.
    JevDynamic,
    ExecutionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonReplayStatus {
    Complete,
    Partial,
    Unavailable,
}

impl DaemonReplayStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonReplayStatus::Complete => "complete",
            DaemonReplayStatus::Partial => "partial",
            DaemonReplayStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonProtocolInfo {
    pub name: DaemonProtocolName,
    pub version: DaemonProtocolVersion,
}

/// `DAEMON_PROTOCOL_INFO`.
pub fn daemon_protocol_info() -> DaemonProtocolInfo {
    DaemonProtocolInfo {
        name: DAEMON_PROTOCOL_NAME.to_string(),
        version: DAEMON_PROTOCOL_VERSION,
    }
}

pub const DAEMON_DEFAULT_CLIENT_CAPABILITIES: [DaemonClientCapability; 2] =
    [DaemonClientCapability::AttachSnapshot, DaemonClientCapability::EventSequence];

pub const DAEMON_SUPPORTED_CLIENT_CAPABILITIES: [DaemonClientCapability; 7] = [
    DaemonClientCapability::AttachSnapshot,
    DaemonClientCapability::EventSequence,
    DaemonClientCapability::ExtensionUi,
    DaemonClientCapability::SlimAttach,
    DaemonClientCapability::ChunkedSnapshot,
    DaemonClientCapability::HistoryRanges,
    DaemonClientCapability::ClientOwnedSessions,
];

/// `DAEMON_DEFAULT_SERVER_CAPABILITIES`: the supported client list plus the
/// server-only surfaces. `direct_peer_transport` and `agent_roster` are
/// deliberately absent, exactly as in the TypeScript.
pub const DAEMON_DEFAULT_SERVER_CAPABILITIES: [DaemonServerCapability; 26] = [
    DaemonServerCapability::AttachSnapshot,
    DaemonServerCapability::EventSequence,
    DaemonServerCapability::ExtensionUi,
    DaemonServerCapability::SlimAttach,
    DaemonServerCapability::ChunkedSnapshot,
    DaemonServerCapability::HistoryRanges,
    DaemonServerCapability::ClientOwnedSessions,
    DaemonServerCapability::DeleteRlmSubagent,
    DaemonServerCapability::HeartbeatCatalog,
    DaemonServerCapability::HeartbeatManagement,
    DaemonServerCapability::ModelCatalog,
    DaemonServerCapability::SideQuestionTranscript,
    DaemonServerCapability::TransientBash,
    DaemonServerCapability::SessionInputAdmission,
    DaemonServerCapability::PromptAdmissionCancellation,
    DaemonServerCapability::OwnedPromptCancellation,
    DaemonServerCapability::QueueMessageMutation,
    DaemonServerCapability::AuthoritativeChildRoster,
    DaemonServerCapability::OwnedSessionRecoveryContext,
    DaemonServerCapability::RlmQuiescenceBarrier,
    DaemonServerCapability::SessionInputPause,
    DaemonServerCapability::AcpMcpServers,
    DaemonServerCapability::JevControl,
    DaemonServerCapability::JevFeatures,
    DaemonServerCapability::JevDynamic,
    DaemonServerCapability::ExecutionMode,
];

/// `{ dev: number; ino: number }` on the peer transport ticket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSocketIdentity {
    pub dev: f64,
    pub ino: f64,
}

/// Single-use short-lived credential for one direct TUI connection to one worker process incarnation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonPeerTransportTicket {
    pub purpose: String,
    #[serde(rename = "socketPath")]
    pub socket_path: String,
    /// Filesystem identity of the worker socket at issue time; re-checked by the client before connecting.
    #[serde(rename = "socketIdentity")]
    pub socket_identity: DaemonSocketIdentity,
    #[serde(rename = "workerInstanceId")]
    pub worker_instance_id: String,
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    #[serde(rename = "grantId")]
    pub grant_id: String,
    pub token: String,
    #[serde(rename = "expiresAt")]
    pub expires_at: String,
}

pub const DAEMON_PEER_TRANSPORT_TICKET_PURPOSE: &str = "session_client";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonRuntimeIdentity {
    #[serde(rename = "buildId")]
    pub build_id: String,
    #[serde(rename = "executablePath")]
    pub executable_path: String,
    #[serde(rename = "entrypointPath", skip_serializing_if = "Option::is_none", default)]
    pub entrypoint_path: Option<String>,
    #[serde(rename = "launcherPath", skip_serializing_if = "Option::is_none", default)]
    pub launcher_path: Option<String>,
}

/// `AgentSessionRuntimeConfig` (core/agent-session-config.ts) as the wire
/// carries it. That module belongs to another slice, so the daemon keeps the
/// structural view it serializes, with the same field names.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentSessionRuntimeConfig {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<String>,
    #[serde(rename = "agentDir", skip_serializing_if = "Option::is_none", default)]
    pub agent_dir: Option<String>,
    #[serde(rename = "sessionDir", skip_serializing_if = "Option::is_none", default)]
    pub session_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<String>,
    #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none", default)]
    pub api_key: Option<String>,
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none", default)]
    pub system_prompt: Option<String>,
    #[serde(rename = "appendSystemPrompt", skip_serializing_if = "Option::is_none", default)]
    pub append_system_prompt: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub thinking: Option<ThinkingLevel>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tools: Option<Vec<String>>,
    #[serde(rename = "noTools", skip_serializing_if = "Option::is_none", default)]
    pub no_tools: Option<bool>,
    #[serde(rename = "noBuiltinTools", skip_serializing_if = "Option::is_none", default)]
    pub no_builtin_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extensions: Option<Vec<String>>,
    #[serde(rename = "noExtensions", skip_serializing_if = "Option::is_none", default)]
    pub no_extensions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub skills: Option<Vec<String>>,
    #[serde(rename = "noSkills", skip_serializing_if = "Option::is_none", default)]
    pub no_skills: Option<bool>,
    #[serde(rename = "promptTemplates", skip_serializing_if = "Option::is_none", default)]
    pub prompt_templates: Option<Vec<String>>,
    #[serde(rename = "noPromptTemplates", skip_serializing_if = "Option::is_none", default)]
    pub no_prompt_templates: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub themes: Option<Vec<String>>,
    #[serde(rename = "noThemes", skip_serializing_if = "Option::is_none", default)]
    pub no_themes: Option<bool>,
    #[serde(rename = "noContextFiles", skip_serializing_if = "Option::is_none", default)]
    pub no_context_files: Option<bool>,
    /// `AgentAutonomousConfig` (core/autonomous.ts) stays an opaque object here.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub autonomous: Option<Value>,
    #[serde(rename = "extensionFlagValues", skip_serializing_if = "Option::is_none", default)]
    pub extension_flag_values: Option<IndexMap<String, Value>>,
    #[serde(rename = "serializedRefine", skip_serializing_if = "Option::is_none", default)]
    pub serialized_refine: Option<bool>,
    /// `AgentExecutionMode`.
    #[serde(rename = "executionMode", skip_serializing_if = "Option::is_none", default)]
    pub execution_mode: Option<String>,
    /// `telemetryDisabled?: true` - only `true` is a legal value.
    #[serde(rename = "telemetryDisabled", skip_serializing_if = "Option::is_none", default)]
    pub telemetry_disabled: Option<bool>,
    #[serde(rename = "initialGoal", skip_serializing_if = "Option::is_none", default)]
    pub initial_goal: Option<Value>,
}

/// `AgentSessionRuntimeMetadata` (core/agent-session-runtime.ts); the runtime
/// module belongs to another slice, so the wire view lives here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentSessionRuntimeMetadata {
    /// `"top-level" | "subagent"`.
    pub kind: String,
    #[serde(rename = "createdAt")]
    pub created_at: f64,
    #[serde(rename = "parentActiveSessionId", skip_serializing_if = "Option::is_none", default)]
    pub parent_active_session_id: Option<String>,
    #[serde(rename = "parentSessionId", skip_serializing_if = "Option::is_none", default)]
    pub parent_session_id: Option<String>,
    #[serde(rename = "parentSessionFile", skip_serializing_if = "Option::is_none", default)]
    pub parent_session_file: Option<String>,
    #[serde(rename = "rlmChildId", skip_serializing_if = "Option::is_none", default)]
    pub rlm_child_id: Option<String>,
    #[serde(rename = "rlmParentNodeId", skip_serializing_if = "Option::is_none", default)]
    pub rlm_parent_node_id: Option<String>,
    #[serde(rename = "rehydratedCompleted", skip_serializing_if = "Option::is_none", default)]
    pub rehydrated_completed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prompt: Option<String>,
    #[serde(rename = "spawnCode", skip_serializing_if = "Option::is_none", default)]
    pub spawn_code: Option<String>,
    #[serde(rename = "sessionDir", skip_serializing_if = "Option::is_none", default)]
    pub session_dir: Option<String>,
}

/// `({ activeSessionId?: string } & DaemonEventCursor) | { activeSessionId?: string; eventSequence }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonResumeCursor {
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
    pub active_session_id: Option<String>,
    /// Present together with `sequence` in the full-cursor arm.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sequence: Option<DaemonEventSequence>,
    #[serde(rename = "eventSequence", skip_serializing_if = "Option::is_none", default)]
    pub event_sequence: Option<DaemonEventSequence>,
}

impl DaemonResumeCursor {
    /// `"sequence" in resumeCursor ? resumeCursor.sequence : resumeCursor.eventSequence`.
    pub fn resume_sequence(&self) -> DaemonEventSequence {
        self.sequence.or(self.event_sequence).unwrap_or(0)
    }

    /// `"generation" in resumeCursor` is exactly "the cursor carries a generation".
    pub fn from_cursor(&self) -> Option<DaemonEventCursor> {
        self.generation.as_ref().map(|generation| DaemonEventCursor {
            generation: generation.clone(),
            sequence: self.resume_sequence(),
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonAttachClientMetadata {
    #[serde(rename = "clientId", skip_serializing_if = "Option::is_none", default)]
    pub client_id: Option<DaemonClientId>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub capabilities: Option<Vec<DaemonClientCapability>>,
    #[serde(rename = "resumeCursor", skip_serializing_if = "Option::is_none", default)]
    pub resume_cursor: Option<DaemonResumeCursor>,
    /// Opt-out-only policy. A telemetry-enabled worker must reject this attach.
    #[serde(rename = "telemetryDisabled", skip_serializing_if = "Option::is_none", default)]
    pub telemetry_disabled: Option<bool>,
    /// Fresh owner-supplied runtime context for recovering a client-owned worker. Never persisted.
    #[serde(rename = "recoveryConfig", skip_serializing_if = "Option::is_none", default)]
    pub recovery_config: Option<AgentSessionRuntimeConfig>,
}

/// Carried on create only: attach must not rebind a session's identity, since
/// watchers (agents view, subagent viewers) also attach.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonClientEnv {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub env: Option<IndexMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonSessionLifecycle {
    Resident,
    ClientOwned,
}

impl DaemonSessionLifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonSessionLifecycle::Resident => "resident",
            DaemonSessionLifecycle::ClientOwned => "client_owned",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonLaunchEnv {
    #[serde(rename = "launchEnv", skip_serializing_if = "Option::is_none", default)]
    pub launch_env: Option<IndexMap<String, String>>,
}

/// The allowlist of env vars a client may forward. One shared list because it
/// is the wire contract: clients filter before sending and the daemon
/// re-filters on receipt (the socket peer is untrusted).
pub const DAEMON_CLIENT_ENV_KEYS: [&str; 5] = [
    "HERDR_ENV",
    "HERDR_PANE_ID",
    "HERDR_SOCKET_PATH",
    "HERDR_TAB_ID",
    "HERDR_WORKSPACE_ID",
];

/// Collect the allowlisted env vars from the client process for the create command.
pub fn collect_daemon_client_env(source: &dyn Fn(&str) -> Option<String>) -> Option<IndexMap<String, String>> {
    let mut env: IndexMap<String, String> = IndexMap::new();
    for key in DAEMON_CLIENT_ENV_KEYS {
        if let Some(value) = source(key) {
            env.insert(key.to_string(), value);
        }
    }
    if env.is_empty() {
        None
    } else {
        Some(env)
    }
}

/// `collectDaemonClientEnv()` with the default `process.env` source.
pub fn collect_daemon_client_env_from_process() -> Option<IndexMap<String, String>> {
    collect_daemon_client_env(&|key| std::env::var(key).ok())
}

pub fn collect_daemon_launch_env(
    entries: impl IntoIterator<Item = (String, Option<String>)>,
) -> IndexMap<String, String> {
    let mut env: IndexMap<String, String> = IndexMap::new();
    for (key, value) in entries {
        if let Some(value) = value {
            if !key.starts_with("PRIME_AGENT_INTERNAL_") {
                env.insert(key, value);
            }
        }
    }
    env
}

/// `collectDaemonLaunchEnv()` with the default `process.env` source.
pub fn collect_daemon_launch_env_from_process() -> IndexMap<String, String> {
    collect_daemon_launch_env(std::env::vars().map(|(key, value)| (key, Some(value))))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonReplayInfo {
    pub status: DaemonReplayStatus,
    #[serde(rename = "fromSequence", skip_serializing_if = "Option::is_none", default)]
    pub from_sequence: Option<DaemonEventSequence>,
    #[serde(rename = "toSequence")]
    pub to_sequence: DaemonEventSequence,
    #[serde(rename = "fromCursor", skip_serializing_if = "Option::is_none", default)]
    pub from_cursor: Option<DaemonEventCursor>,
    #[serde(rename = "toCursor", skip_serializing_if = "Option::is_none", default)]
    pub to_cursor: Option<DaemonEventCursor>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonEventMeta {
    pub id: DaemonEventId,
    pub protocol: DaemonProtocolInfo,
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
    pub active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sequence: Option<DaemonEventSequence>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cursor: Option<DaemonEventCursor>,
    #[serde(rename = "emittedAt")]
    pub emitted_at: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub replayed: Option<bool>,
}

/// Like `DaemonCommand`, this envelope embeds `SessionActionRecoverySnapshot` (via
/// `DaemonCommand`) which carries no `PartialEq`; nothing compares these wire values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonCommandEnvelope {
    #[serde(rename = "type")]
    pub type_: String,
    pub id: DaemonCommandId,
    pub protocol: DaemonProtocolInfo,
    #[serde(rename = "clientId", skip_serializing_if = "Option::is_none", default)]
    pub client_id: Option<DaemonClientId>,
    pub command: DaemonCommand,
}

/// `DaemonCommandWire = DaemonCommand | DaemonCommandEnvelope`.
///
/// Like `DaemonCommand`, this union embeds `SessionActionRecoverySnapshot` (via
/// `DaemonCommand`) which carries no `PartialEq`; nothing compares these wire values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DaemonCommandWire {
    Envelope(DaemonCommandEnvelope),
    Command(DaemonCommand),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonEventEnvelope {
    #[serde(rename = "type")]
    pub type_: String,
    pub id: DaemonEventId,
    pub protocol: DaemonProtocolInfo,
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
    pub active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sequence: Option<DaemonEventSequence>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cursor: Option<DaemonEventCursor>,
    #[serde(rename = "emittedAt")]
    pub emitted_at: String,
    pub event: DaemonOutbound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonArtifactReference {
    pub id: String,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(rename = "logicalPath")]
    pub logical_path: String,
    #[serde(rename = "relativePath", skip_serializing_if = "Option::is_none", default)]
    pub relative_path: Option<String>,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none", default)]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub metadata: Option<IndexMap<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonHistoryWindow {
    pub version: f64,
    pub generation: String,
    /// Opaque identity of the target-model representation used to build this window.
    pub representation: String,
    #[serde(rename = "tipEntryId")]
    pub tip_entry_id: Option<String>,
    #[serde(rename = "totalMessageCount")]
    pub total_message_count: f64,
    #[serde(rename = "startIndex")]
    pub start_index: f64,
    #[serde(rename = "entryIds")]
    pub entry_ids: Vec<String>,
    #[serde(rename = "hasOlder")]
    pub has_older: bool,
    /// `"chronological"`.
    pub order: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonHistoryRange {
    pub version: f64,
    pub generation: String,
    pub representation: String,
    #[serde(rename = "tipEntryId")]
    pub tip_entry_id: Option<String>,
    #[serde(rename = "totalMessageCount")]
    pub total_message_count: f64,
    #[serde(rename = "startIndex")]
    pub start_index: f64,
    pub messages: Vec<AgentMessage>,
    #[serde(rename = "entryIds")]
    pub entry_ids: Vec<String>,
    #[serde(rename = "hasOlder")]
    pub has_older: bool,
    pub order: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSessionSnapshot {
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    pub summary: SessionSummary,
    pub state: AgentConnectionState,
    pub messages: Vec<AgentMessage>,
    /// Present only when the attaching client negotiated history_ranges.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub history: Option<DaemonHistoryWindow>,
    #[serde(rename = "sessionContext", skip_serializing_if = "Option::is_none", default)]
    pub session_context: Option<AgentConnectionSessionContext>,
    /// `{ tree: AgentConnectionSessionTreeNode[]; leafId: string | null }`.
    #[serde(rename = "sessionTree", skip_serializing_if = "Option::is_none", default)]
    pub session_tree: Option<DaemonSessionTree>,
    #[serde(rename = "lastEventSequence")]
    pub last_event_sequence: DaemonEventSequence,
    #[serde(rename = "lastEventCursor", skip_serializing_if = "Option::is_none", default)]
    pub last_event_cursor: Option<DaemonEventCursor>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent: Option<DaemonSessionSnapshotParent>,
    /// Live RLM child sessions (including grandchildren) hosted by the daemon under this session.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub children: Option<Vec<AgentConnectionRlmChildAgentSnapshot>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSessionTree {
    pub tree: Vec<AgentConnectionSessionTreeNode>,
    #[serde(rename = "leafId")]
    pub leaf_id: Option<String>,
}

/// The snapshot's inline parent metadata. The TypeScript inlines the object; the
/// field set matches `AgentConnectionParentMetadata`.
pub type DaemonSessionSnapshotParent = crate::modes::agent_connection::types::AgentConnectionParentMetadata;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonAttachResult {
    pub protocol: DaemonProtocolInfo,
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    /// Omitted for clients with the "slim_attach" capability; use snapshot.summary.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub state: Option<SessionSummary>,
    /// Omitted for clients with the "slim_attach" capability; use snapshot.messages.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub messages: Option<Vec<AgentMessage>>,
    pub snapshot: DaemonSessionSnapshot,
    pub replay: DaemonReplayInfo,
    #[serde(rename = "lastEventSequence")]
    pub last_event_sequence: DaemonEventSequence,
    #[serde(rename = "lastEventCursor", skip_serializing_if = "Option::is_none", default)]
    pub last_event_cursor: Option<DaemonEventCursor>,
    #[serde(rename = "snapshotStream", skip_serializing_if = "Option::is_none", default)]
    pub snapshot_stream: Option<DaemonSnapshotStream>,
    pub client: DaemonAttachClient,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSnapshotStream {
    pub id: String,
    #[serde(rename = "messageCount")]
    pub message_count: f64,
    #[serde(rename = "targetChunkBytes")]
    pub target_chunk_bytes: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonAttachClient {
    pub id: DaemonClientId,
    pub capabilities: Vec<DaemonClientCapability>,
}

pub const DAEMON_UPDATE_RESTART_FORMAT_VERSION: f64 = 1.0;

// `SessionActionRecoverySnapshot` carries no `PartialEq`, so neither does the
// queue that embeds it; nothing compares these wire values in the port.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonUpdateRestartQueue {
    pub actions: SessionActionRecoverySnapshot,
    #[serde(rename = "nextTurn")]
    pub next_turn: Vec<CustomMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonUpdateRestartSession {
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "sessionFile")]
    pub session_file: String,
    pub cwd: String,
    pub config: AgentSessionRuntimeConfig,
    #[serde(rename = "runtimeMetadata", skip_serializing_if = "Option::is_none", default)]
    pub runtime_metadata: Option<AgentSessionRuntimeMetadata>,
    #[serde(rename = "clientEnv", skip_serializing_if = "Option::is_none", default)]
    pub client_env: Option<IndexMap<String, String>>,
    pub queue: DaemonUpdateRestartQueue,
    #[serde(rename = "shouldResume")]
    pub should_resume: bool,
    #[serde(rename = "wasStreaming")]
    pub was_streaming: bool,
    #[serde(rename = "wasCompacting")]
    pub was_compacting: bool,
    #[serde(rename = "wasBashRunning")]
    pub was_bash_running: bool,
    #[serde(rename = "hadRunningRlmChildren")]
    pub had_running_rlm_children: bool,
    #[serde(rename = "wasRetrying")]
    pub was_retrying: bool,
    #[serde(rename = "hadAcceptedPromptInFlight")]
    pub had_accepted_prompt_in_flight: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonUpdateRestartManifest {
    #[serde(rename = "formatVersion")]
    pub format_version: f64,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    pub sessions: Vec<DaemonUpdateRestartSession>,
    #[serde(rename = "discardedActiveSessionIds", skip_serializing_if = "Option::is_none", default)]
    pub discarded_active_session_ids: Option<Vec<String>>,
}

/// `{ id?: string; type: "list_saved_sessions"; activeSessionId: string; scope }`
/// `| { id?: string; type: "list_saved_sessions"; cwd: string; sessionDir?: string; scope }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSavedSessionListCommand {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<String>,
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
    pub active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<String>,
    #[serde(rename = "sessionDir", skip_serializing_if = "Option::is_none", default)]
    pub session_dir: Option<String>,
    pub scope: String,
}

/// `DaemonAttachClientMetadata & DaemonClientEnv & DaemonLaunchEnv` for attach/reattach.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DaemonAttachEnvelope {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub env: Option<IndexMap<String, String>>,
    #[serde(rename = "launchEnv", skip_serializing_if = "Option::is_none", default)]
    pub launch_env: Option<IndexMap<String, String>>,
}

/// `(TextContent | ImageContent)[]` on prompt/steer/follow_up commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DaemonPromptContent {
    Text(TextContent),
    Image(ImageContent),
}

/// `Pick<CustomMessage, "customType" | "content" | "display" | "details">`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonCustomMessageBody {
    #[serde(rename = "customType")]
    pub custom_type: String,
    pub content: pi_agent_core::types::CustomMessageContent,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub details: Option<Value>,
}

/// `"steer" | "followUp"` on prompt commands.
pub const DAEMON_PROMPT_STREAMING_BEHAVIOR: [&str; 2] = ["steer", "followUp"];

/// Every `DaemonCommand` variant, discriminated by `type`.
///
/// The TypeScript union is spread across many modules; the field sets here are
/// the union's, unchanged, with camelCase wire names and absent-vs-null kept.
// `SessionActionRecoverySnapshot` carries no `PartialEq`, so neither does this
// command union; nothing compares these wire values in the port (same decision as
// `DaemonUpdateRestartQueue` above).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonCommand {
    #[serde(rename = "ack_result")]
    AckResult {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "commandId")]
        command_id: String,
    },
    #[serde(rename = "list")]
    List {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        all: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        cwd: Option<String>,
        #[serde(rename = "sessionDir", skip_serializing_if = "Option::is_none", default)]
        session_dir: Option<String>,
        #[serde(rename = "includeClientOwned", skip_serializing_if = "Option::is_none", default)]
        include_client_owned: Option<bool>,
    },
    #[serde(rename = "list_saved_sessions")]
    ListSavedSessions {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(flatten)]
        body: DaemonSavedSessionListCommand,
    },
    #[serde(rename = "list_agent_peers")]
    ListAgentPeers {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "workerToken")]
        worker_token: String,
    },
    #[serde(rename = "get_direct_worker_transport")]
    GetDirectWorkerTransport {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "roster_subscribe")]
    RosterSubscribe {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
    },
    #[serde(rename = "roster_unsubscribe")]
    RosterUnsubscribe {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
    },
    #[serde(rename = "create")]
    Create {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "sessionPath", skip_serializing_if = "Option::is_none", default)]
        session_path: Option<String>,
        #[serde(rename = "continueRecent", skip_serializing_if = "Option::is_none", default)]
        continue_recent: Option<bool>,
        #[serde(rename = "noSession", skip_serializing_if = "Option::is_none", default)]
        no_session: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        config: Option<AgentSessionRuntimeConfig>,
        #[serde(rename = "runtimeMetadata", skip_serializing_if = "Option::is_none", default)]
        runtime_metadata: Option<AgentSessionRuntimeMetadata>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lifecycle: Option<DaemonSessionLifecycle>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        env: Option<IndexMap<String, String>>,
        #[serde(rename = "launchEnv", skip_serializing_if = "Option::is_none", default)]
        launch_env: Option<IndexMap<String, String>>,
    },
    #[serde(rename = "attach")]
    Attach {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "supportsExtensionUi", skip_serializing_if = "Option::is_none", default)]
        supports_extension_ui: Option<bool>,
        #[serde(flatten)]
        client: DaemonAttachClientMetadata,
        #[serde(flatten)]
        envs: DaemonAttachEnvelope,
    },
    #[serde(rename = "reattach")]
    Reattach {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "targetActiveSessionId")]
        target_active_session_id: String,
        #[serde(rename = "supportsExtensionUi", skip_serializing_if = "Option::is_none", default)]
        supports_extension_ui: Option<bool>,
        #[serde(flatten)]
        client: DaemonAttachClientMetadata,
        #[serde(flatten)]
        envs: DaemonAttachEnvelope,
    },
    #[serde(rename = "detach")]
    Detach {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
    },
    #[serde(rename = "complete_owned_session")]
    CompleteOwnedSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "promote_owned_session")]
    PromoteOwnedSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "kill")]
    Kill {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "rename")]
    Rename {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        name: String,
    },
    #[serde(rename = "prompt")]
    Prompt {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        content: Option<Vec<DaemonPromptContent>>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        images: Option<Vec<ImageContent>>,
        #[serde(rename = "streamingBehavior", skip_serializing_if = "Option::is_none", default)]
        streaming_behavior: Option<String>,
        #[serde(rename = "queueIfBusy", skip_serializing_if = "Option::is_none", default)]
        queue_if_busy: Option<bool>,
        #[serde(rename = "expandPromptTemplates", skip_serializing_if = "Option::is_none", default)]
        expand_prompt_templates: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        source: Option<InputSource>,
        #[serde(rename = "agentMessageId", skip_serializing_if = "Option::is_none", default)]
        agent_message_id: Option<String>,
        #[serde(rename = "customMessage", skip_serializing_if = "Option::is_none", default)]
        custom_message: Option<CustomMessage>,
        /// Unique only when the caller needs cancellable pre-ownership admission.
        #[serde(rename = "admissionId", skip_serializing_if = "Option::is_none", default)]
        admission_id: Option<String>,
    },
    #[serde(rename = "cancel_prompt_admission")]
    CancelPromptAdmission {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "admissionId")]
        admission_id: String,
        /// Cancel session-owned work too when it has not started delivery.
        #[serde(rename = "cancelOwned", skip_serializing_if = "Option::is_none", default)]
        cancel_owned: Option<bool>,
    },
    #[serde(rename = "prompt_and_wait")]
    PromptAndWait {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        content: Option<Vec<DaemonPromptContent>>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        images: Option<Vec<ImageContent>>,
        #[serde(rename = "streamingBehavior", skip_serializing_if = "Option::is_none", default)]
        streaming_behavior: Option<String>,
        #[serde(rename = "queueIfBusy", skip_serializing_if = "Option::is_none", default)]
        queue_if_busy: Option<bool>,
        #[serde(rename = "expandPromptTemplates", skip_serializing_if = "Option::is_none", default)]
        expand_prompt_templates: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        source: Option<InputSource>,
        /// Unique only when the caller needs cancellable pre-ownership admission.
        #[serde(rename = "admissionId", skip_serializing_if = "Option::is_none", default)]
        admission_id: Option<String>,
    },
    #[serde(rename = "steer")]
    Steer {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        content: Option<Vec<DaemonPromptContent>>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        images: Option<Vec<ImageContent>>,
        #[serde(rename = "queueKey", skip_serializing_if = "Option::is_none", default)]
        queue_key: Option<String>,
        #[serde(rename = "expandPromptTemplates", skip_serializing_if = "Option::is_none", default)]
        expand_prompt_templates: Option<bool>,
        #[serde(rename = "agentMessageId", skip_serializing_if = "Option::is_none", default)]
        agent_message_id: Option<String>,
        #[serde(rename = "customMessage", skip_serializing_if = "Option::is_none", default)]
        custom_message: Option<CustomMessage>,
        #[serde(rename = "prefixMessages", skip_serializing_if = "Option::is_none", default)]
        prefix_messages: Option<Vec<CustomMessage>>,
    },
    #[serde(rename = "follow_up")]
    FollowUp {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        content: Option<Vec<DaemonPromptContent>>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        images: Option<Vec<ImageContent>>,
        #[serde(rename = "queueKey", skip_serializing_if = "Option::is_none", default)]
        queue_key: Option<String>,
        #[serde(rename = "expandPromptTemplates", skip_serializing_if = "Option::is_none", default)]
        expand_prompt_templates: Option<bool>,
        #[serde(rename = "agentMessageId", skip_serializing_if = "Option::is_none", default)]
        agent_message_id: Option<String>,
        #[serde(rename = "customMessage", skip_serializing_if = "Option::is_none", default)]
        custom_message: Option<CustomMessage>,
        #[serde(rename = "prefixMessages", skip_serializing_if = "Option::is_none", default)]
        prefix_messages: Option<Vec<CustomMessage>>,
    },
    #[serde(rename = "restore_next_turn")]
    RestoreNextTurn {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        messages: Vec<CustomMessage>,
    },
    #[serde(rename = "restore_actions")]
    RestoreActions {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        snapshot: SessionActionRecoverySnapshot,
    },
    #[serde(rename = "append_custom_message")]
    AppendCustomMessage {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        message: DaemonCustomMessageBody,
    },
    #[serde(rename = "resume_queue")]
    ResumeQueue {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "send_message")]
    SendMessage {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "targetActiveSessionId")]
        target_active_session_id: String,
        message: String,
        #[serde(rename = "fromActiveSessionId", skip_serializing_if = "Option::is_none", default)]
        from_active_session_id: Option<String>,
        /// Internal worker-origin marker; public clients remain unrestricted.
        #[serde(rename = "agentOrigin", skip_serializing_if = "Option::is_none", default)]
        agent_origin: Option<bool>,
        #[serde(rename = "deliveryMode", skip_serializing_if = "Option::is_none", default)]
        delivery_mode: Option<AgentSessionMessageDeliveryMode>,
    },
    #[serde(rename = "agent_messages_status")]
    AgentMessagesStatus {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
    },
    #[serde(rename = "agent_messages_pause")]
    AgentMessagesPause {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
    },
    #[serde(rename = "agent_messages_resume")]
    AgentMessagesResume {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
    },
    #[serde(rename = "agent_messages_clear")]
    AgentMessagesClear {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "abort")]
    Abort {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "start_side_question")]
    StartSideQuestion {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "sideQuestionId")]
        side_question_id: String,
        question: String,
        #[serde(rename = "previousTurns", skip_serializing_if = "Option::is_none", default)]
        previous_turns: Option<Vec<AgentConnectionSideQuestionTurn>>,
    },
    #[serde(rename = "abort_side_question")]
    AbortSideQuestion {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "sideQuestionId")]
        side_question_id: String,
    },
    #[serde(rename = "execute_bash")]
    ExecuteBash {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        command: String,
        #[serde(rename = "excludeFromContext", skip_serializing_if = "Option::is_none", default)]
        exclude_from_context: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        transient: Option<bool>,
        #[serde(rename = "runId", skip_serializing_if = "Option::is_none", default)]
        run_id: Option<String>,
    },
    #[serde(rename = "abort_bash")]
    AbortBash {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "cancel_rlm_child")]
    CancelRlmChild {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "childId")]
        child_id: String,
    },
    #[serde(rename = "delete_rlm_subagent")]
    DeleteRlmSubagent {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "childId")]
        child_id: String,
    },
    #[serde(rename = "wait_for_idle")]
    WaitForIdle {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "wait_for_headless_completion")]
    WaitForHeadlessCompletion {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "waitForRlmQuiescence", skip_serializing_if = "Option::is_none", default)]
        wait_for_rlm_quiescence: Option<bool>,
    },
    #[serde(rename = "get_session_header")]
    GetSessionHeader {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_state")]
    GetState {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_connection_state")]
    GetConnectionState {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_messages")]
    GetMessages {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_history_range")]
    GetHistoryRange {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        generation: String,
        representation: String,
        #[serde(rename = "tipEntryId")]
        tip_entry_id: Option<String>,
        #[serde(rename = "beforeEntryId", skip_serializing_if = "Option::is_none", default)]
        before_entry_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        limit: Option<f64>,
    },
    #[serde(rename = "get_rlm_children")]
    GetRlmChildren {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_session_stats")]
    GetSessionStats {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_context_tree")]
    GetContextTree {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_commands")]
    GetCommands {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_resource_snapshot")]
    GetResourceSnapshot {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "replace_acp_mcp_servers")]
    ReplaceAcpMcpServers {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "ownerId")]
        owner_id: String,
        servers: Vec<AcpMcpServerConfig>,
    },
    #[serde(rename = "get_model_catalog")]
    GetModelCatalog {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_available_models")]
    GetAvailableModels {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_queue")]
    GetQueue {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "mutate_queued_message")]
    MutateQueuedMessage {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        lane: QueuedMessageLane,
        index: f64,
        #[serde(rename = "expectedText")]
        expected_text: String,
        mutation: QueuedMessageMutation,
    },
    #[serde(rename = "clear_queue")]
    ClearQueue {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "abort_and_clear_queue")]
    AbortAndClearQueue {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "acquire_session_input_pause")]
    AcquireSessionInputPause {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "leaseKey")]
        lease_key: String,
    },
    #[serde(rename = "release_session_input_pause")]
    ReleaseSessionInputPause {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "pauseId")]
        pause_id: String,
    },
    #[serde(rename = "cron_list")]
    CronList {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        #[serde(rename = "includeInactive", skip_serializing_if = "Option::is_none", default)]
        include_inactive: Option<bool>,
    },
    #[serde(rename = "heartbeats_list")]
    HeartbeatsList {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
    },
    #[serde(rename = "heartbeat_manage")]
    HeartbeatManage {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "jobId")]
        job_id: String,
        action: AgentHeartbeatManagementAction,
    },
    #[serde(rename = "cron_add")]
    CronAdd {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        schedule: String,
        prompt: String,
        #[serde(rename = "promoteOwnedSession", skip_serializing_if = "Option::is_none", default)]
        promote_owned_session: Option<bool>,
    },
    #[serde(rename = "cron_cancel")]
    CronCancel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        #[serde(rename = "jobId")]
        job_id: String,
    },
    #[serde(rename = "heartbeat_get")]
    HeartbeatGet {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "heartbeat_set")]
    HeartbeatSet {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        schedule: String,
        prompt: String,
        #[serde(rename = "deliveryMode", skip_serializing_if = "Option::is_none", default)]
        delivery_mode: Option<AgentHeartbeatDeliveryMode>,
        #[serde(rename = "promoteOwnedSession", skip_serializing_if = "Option::is_none", default)]
        promote_owned_session: Option<bool>,
    },
    #[serde(rename = "heartbeat_update")]
    HeartbeatUpdate {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        action: AgentHeartbeatUpdateAction,
    },
    #[serde(rename = "set_model")]
    SetModel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
    #[serde(rename = "cycle_model")]
    CycleModel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        direction: Option<String>,
    },
    #[serde(rename = "set_scoped_models")]
    SetScopedModels {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "scopedModels")]
        scoped_models: Vec<AgentConnectionScopedModel>,
    },
    #[serde(rename = "set_thinking_level")]
    SetThinkingLevel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        level: ThinkingLevel,
    },
    #[serde(rename = "set_service_tier")]
    SetServiceTier {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "serviceTier")]
        service_tier: pi_ai::types::ServiceTier,
    },
    #[serde(rename = "cycle_thinking_level")]
    CycleThinkingLevel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "set_transport")]
    SetTransport {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        transport: pi_ai::types::Transport,
    },
    #[serde(rename = "set_steering_mode")]
    SetSteeringMode {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        /// `AgentConnectionQueueMode` is a string union on the wire.
        mode: String,
    },
    #[serde(rename = "set_follow_up_mode")]
    SetFollowUpMode {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        /// `AgentConnectionQueueMode` is a string union on the wire.
        mode: String,
    },
    #[serde(rename = "set_auto_compaction")]
    SetAutoCompaction {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        enabled: bool,
    },
    #[serde(rename = "set_auto_retry")]
    SetAutoRetry {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        enabled: bool,
    },
    #[serde(rename = "compact")]
    Compact {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "customInstructions", skip_serializing_if = "Option::is_none", default)]
        custom_instructions: Option<String>,
    },
    #[serde(rename = "refine")]
    Refine {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        instructions: Option<String>,
        #[serde(rename = "rollbackId", skip_serializing_if = "Option::is_none", default)]
        rollback_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        global: Option<bool>,
    },
    #[serde(rename = "abort_compaction")]
    AbortCompaction {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "abort_branch_summary")]
    AbortBranchSummary {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "abort_retry")]
    AbortRetry {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "execute_bash_and_wait")]
    ExecuteBashAndWait {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        command: String,
    },
    #[serde(rename = "reload")]
    Reload {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "new_session")]
    NewSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none", default)]
        parent_session: Option<String>,
    },
    #[serde(rename = "switch_session")]
    SwitchSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "sessionPath")]
        session_path: String,
        #[serde(rename = "cwdOverride", skip_serializing_if = "Option::is_none", default)]
        cwd_override: Option<String>,
    },
    #[serde(rename = "fork")]
    Fork {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "entryId")]
        entry_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        position: Option<String>,
    },
    #[serde(rename = "navigate_tree")]
    NavigateTree {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "targetId")]
        target_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        summarize: Option<bool>,
        #[serde(rename = "customInstructions", skip_serializing_if = "Option::is_none", default)]
        custom_instructions: Option<String>,
        #[serde(rename = "replaceInstructions", skip_serializing_if = "Option::is_none", default)]
        replace_instructions: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        label: Option<String>,
    },
    #[serde(rename = "import_jsonl")]
    ImportJsonl {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "inputPath")]
        input_path: String,
        #[serde(rename = "cwdOverride", skip_serializing_if = "Option::is_none", default)]
        cwd_override: Option<String>,
    },
    #[serde(rename = "export_html")]
    ExportHtml {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "outputPath", skip_serializing_if = "Option::is_none", default)]
        output_path: Option<String>,
    },
    #[serde(rename = "export_jsonl")]
    ExportJsonl {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "outputPath", skip_serializing_if = "Option::is_none", default)]
        output_path: Option<String>,
    },
    #[serde(rename = "set_session_name")]
    SetSessionName {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        name: String,
        #[serde(rename = "workerToken", skip_serializing_if = "Option::is_none", default)]
        worker_token: Option<String>,
    },
    #[serde(rename = "get_rlm_max_depth_status")]
    GetRlmMaxDepthStatus {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "set_rlm_max_depth")]
    SetRlmMaxDepth {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "maxDepth")]
        max_depth: f64,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        global: Option<bool>,
    },
    #[serde(rename = "rename_saved_session")]
    RenameSavedSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        #[serde(rename = "sessionPath")]
        session_path: String,
        name: String,
    },
    #[serde(rename = "delete_saved_session")]
    DeleteSavedSession {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        #[serde(rename = "sessionPath")]
        session_path: String,
    },
    #[serde(rename = "get_session_context")]
    GetSessionContext {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_session_tree")]
    GetSessionTree {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_user_messages_for_forking")]
    GetUserMessagesForForking {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_last_assistant_text")]
    GetLastAssistantText {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_system_prompt")]
    GetSystemPrompt {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "get_tool_definition")]
    GetToolDefinition {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        name: String,
    },
    #[serde(rename = "set_session_entry_label")]
    SetSessionEntryLabel {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "entryId")]
        entry_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        label: Option<String>,
    },
    #[serde(rename = "extension_ui_response")]
    ExtensionUiResponse {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "requestId")]
        request_id: String,
        response: DaemonExtensionUIResponse,
    },
    #[serde(rename = "prepare_update_restart")]
    PrepareUpdateRestart {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
    },
    #[serde(rename = "retry_worker")]
    RetryWorker {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "restart")]
    Restart {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
    },
    #[serde(rename = "shutdown")]
    Shutdown {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        force: Option<bool>,
    },
}

/// `"steer" | "follow_up"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHeartbeatDeliveryMode {
    Steer,
    FollowUp,
}

/// `"pause" | "resume" | "stop"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHeartbeatManagementAction {
    Pause,
    Resume,
    Stop,
}

/// `"pause" | "resume" | "clear"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHeartbeatUpdateAction {
    Pause,
    Resume,
    Clear,
}

/// `{ value: string } | { confirmed: boolean } | { cancelled: true }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DaemonExtensionUIResponse {
    Value { value: String },
    Confirmed { confirmed: bool },
    Cancelled { cancelled: bool },
}

impl DaemonCommand {
    /// `command.type`.
    pub fn command_type(&self) -> &'static str {
        match self {
            DaemonCommand::AckResult { .. } => "ack_result",
            DaemonCommand::List { .. } => "list",
            DaemonCommand::ListSavedSessions { .. } => "list_saved_sessions",
            DaemonCommand::ListAgentPeers { .. } => "list_agent_peers",
            DaemonCommand::GetDirectWorkerTransport { .. } => "get_direct_worker_transport",
            DaemonCommand::RosterSubscribe { .. } => "roster_subscribe",
            DaemonCommand::RosterUnsubscribe { .. } => "roster_unsubscribe",
            DaemonCommand::Create { .. } => "create",
            DaemonCommand::Attach { .. } => "attach",
            DaemonCommand::Reattach { .. } => "reattach",
            DaemonCommand::Detach { .. } => "detach",
            DaemonCommand::CompleteOwnedSession { .. } => "complete_owned_session",
            DaemonCommand::PromoteOwnedSession { .. } => "promote_owned_session",
            DaemonCommand::Kill { .. } => "kill",
            DaemonCommand::Rename { .. } => "rename",
            DaemonCommand::Prompt { .. } => "prompt",
            DaemonCommand::CancelPromptAdmission { .. } => "cancel_prompt_admission",
            DaemonCommand::PromptAndWait { .. } => "prompt_and_wait",
            DaemonCommand::Steer { .. } => "steer",
            DaemonCommand::FollowUp { .. } => "follow_up",
            DaemonCommand::RestoreNextTurn { .. } => "restore_next_turn",
            DaemonCommand::RestoreActions { .. } => "restore_actions",
            DaemonCommand::AppendCustomMessage { .. } => "append_custom_message",
            DaemonCommand::ResumeQueue { .. } => "resume_queue",
            DaemonCommand::SendMessage { .. } => "send_message",
            DaemonCommand::AgentMessagesStatus { .. } => "agent_messages_status",
            DaemonCommand::AgentMessagesPause { .. } => "agent_messages_pause",
            DaemonCommand::AgentMessagesResume { .. } => "agent_messages_resume",
            DaemonCommand::AgentMessagesClear { .. } => "agent_messages_clear",
            DaemonCommand::Abort { .. } => "abort",
            DaemonCommand::StartSideQuestion { .. } => "start_side_question",
            DaemonCommand::AbortSideQuestion { .. } => "abort_side_question",
            DaemonCommand::ExecuteBash { .. } => "execute_bash",
            DaemonCommand::AbortBash { .. } => "abort_bash",
            DaemonCommand::CancelRlmChild { .. } => "cancel_rlm_child",
            DaemonCommand::DeleteRlmSubagent { .. } => "delete_rlm_subagent",
            DaemonCommand::WaitForIdle { .. } => "wait_for_idle",
            DaemonCommand::WaitForHeadlessCompletion { .. } => "wait_for_headless_completion",
            DaemonCommand::GetSessionHeader { .. } => "get_session_header",
            DaemonCommand::GetState { .. } => "get_state",
            DaemonCommand::GetConnectionState { .. } => "get_connection_state",
            DaemonCommand::GetMessages { .. } => "get_messages",
            DaemonCommand::GetHistoryRange { .. } => "get_history_range",
            DaemonCommand::GetRlmChildren { .. } => "get_rlm_children",
            DaemonCommand::GetSessionStats { .. } => "get_session_stats",
            DaemonCommand::GetContextTree { .. } => "get_context_tree",
            DaemonCommand::GetCommands { .. } => "get_commands",
            DaemonCommand::GetResourceSnapshot { .. } => "get_resource_snapshot",
            DaemonCommand::ReplaceAcpMcpServers { .. } => "replace_acp_mcp_servers",
            DaemonCommand::GetModelCatalog { .. } => "get_model_catalog",
            DaemonCommand::GetAvailableModels { .. } => "get_available_models",
            DaemonCommand::GetQueue { .. } => "get_queue",
            DaemonCommand::MutateQueuedMessage { .. } => "mutate_queued_message",
            DaemonCommand::ClearQueue { .. } => "clear_queue",
            DaemonCommand::AbortAndClearQueue { .. } => "abort_and_clear_queue",
            DaemonCommand::AcquireSessionInputPause { .. } => "acquire_session_input_pause",
            DaemonCommand::ReleaseSessionInputPause { .. } => "release_session_input_pause",
            DaemonCommand::CronList { .. } => "cron_list",
            DaemonCommand::HeartbeatsList { .. } => "heartbeats_list",
            DaemonCommand::HeartbeatManage { .. } => "heartbeat_manage",
            DaemonCommand::CronAdd { .. } => "cron_add",
            DaemonCommand::CronCancel { .. } => "cron_cancel",
            DaemonCommand::HeartbeatGet { .. } => "heartbeat_get",
            DaemonCommand::HeartbeatSet { .. } => "heartbeat_set",
            DaemonCommand::HeartbeatUpdate { .. } => "heartbeat_update",
            DaemonCommand::SetModel { .. } => "set_model",
            DaemonCommand::CycleModel { .. } => "cycle_model",
            DaemonCommand::SetScopedModels { .. } => "set_scoped_models",
            DaemonCommand::SetThinkingLevel { .. } => "set_thinking_level",
            DaemonCommand::SetServiceTier { .. } => "set_service_tier",
            DaemonCommand::CycleThinkingLevel { .. } => "cycle_thinking_level",
            DaemonCommand::SetTransport { .. } => "set_transport",
            DaemonCommand::SetSteeringMode { .. } => "set_steering_mode",
            DaemonCommand::SetFollowUpMode { .. } => "set_follow_up_mode",
            DaemonCommand::SetAutoCompaction { .. } => "set_auto_compaction",
            DaemonCommand::SetAutoRetry { .. } => "set_auto_retry",
            DaemonCommand::Compact { .. } => "compact",
            DaemonCommand::Refine { .. } => "refine",
            DaemonCommand::AbortCompaction { .. } => "abort_compaction",
            DaemonCommand::AbortBranchSummary { .. } => "abort_branch_summary",
            DaemonCommand::AbortRetry { .. } => "abort_retry",
            DaemonCommand::ExecuteBashAndWait { .. } => "execute_bash_and_wait",
            DaemonCommand::Reload { .. } => "reload",
            DaemonCommand::NewSession { .. } => "new_session",
            DaemonCommand::SwitchSession { .. } => "switch_session",
            DaemonCommand::Fork { .. } => "fork",
            DaemonCommand::NavigateTree { .. } => "navigate_tree",
            DaemonCommand::ImportJsonl { .. } => "import_jsonl",
            DaemonCommand::ExportHtml { .. } => "export_html",
            DaemonCommand::ExportJsonl { .. } => "export_jsonl",
            DaemonCommand::SetSessionName { .. } => "set_session_name",
            DaemonCommand::GetRlmMaxDepthStatus { .. } => "get_rlm_max_depth_status",
            DaemonCommand::SetRlmMaxDepth { .. } => "set_rlm_max_depth",
            DaemonCommand::RenameSavedSession { .. } => "rename_saved_session",
            DaemonCommand::DeleteSavedSession { .. } => "delete_saved_session",
            DaemonCommand::GetSessionContext { .. } => "get_session_context",
            DaemonCommand::GetSessionTree { .. } => "get_session_tree",
            DaemonCommand::GetUserMessagesForForking { .. } => "get_user_messages_for_forking",
            DaemonCommand::GetLastAssistantText { .. } => "get_last_assistant_text",
            DaemonCommand::GetSystemPrompt { .. } => "get_system_prompt",
            DaemonCommand::GetToolDefinition { .. } => "get_tool_definition",
            DaemonCommand::SetSessionEntryLabel { .. } => "set_session_entry_label",
            DaemonCommand::ExtensionUiResponse { .. } => "extension_ui_response",
            DaemonCommand::PrepareUpdateRestart { .. } => "prepare_update_restart",
            DaemonCommand::RetryWorker { .. } => "retry_worker",
            DaemonCommand::Restart { .. } => "restart",
            DaemonCommand::Shutdown { .. } => "shutdown",
        }
    }

    /// `id?: string` on every arm.
    pub fn id(&self) -> Option<&str> {
        match self {
            DaemonCommand::AckResult { id, .. }
            | DaemonCommand::List { id, .. }
            | DaemonCommand::ListAgentPeers { id, .. }
            | DaemonCommand::GetDirectWorkerTransport { id, .. }
            | DaemonCommand::RosterSubscribe { id }
            | DaemonCommand::RosterUnsubscribe { id }
            | DaemonCommand::Create { id, .. }
            | DaemonCommand::Attach { id, .. }
            | DaemonCommand::Reattach { id, .. }
            | DaemonCommand::Detach { id, .. }
            | DaemonCommand::CompleteOwnedSession { id, .. }
            | DaemonCommand::PromoteOwnedSession { id, .. }
            | DaemonCommand::Kill { id, .. }
            | DaemonCommand::Rename { id, .. }
            | DaemonCommand::Prompt { id, .. }
            | DaemonCommand::CancelPromptAdmission { id, .. }
            | DaemonCommand::PromptAndWait { id, .. }
            | DaemonCommand::Steer { id, .. }
            | DaemonCommand::FollowUp { id, .. }
            | DaemonCommand::RestoreNextTurn { id, .. }
            | DaemonCommand::RestoreActions { id, .. }
            | DaemonCommand::AppendCustomMessage { id, .. }
            | DaemonCommand::ResumeQueue { id, .. }
            | DaemonCommand::SendMessage { id, .. }
            | DaemonCommand::AgentMessagesStatus { id, .. }
            | DaemonCommand::AgentMessagesPause { id, .. }
            | DaemonCommand::AgentMessagesResume { id, .. }
            | DaemonCommand::AgentMessagesClear { id, .. }
            | DaemonCommand::Abort { id, .. }
            | DaemonCommand::StartSideQuestion { id, .. }
            | DaemonCommand::AbortSideQuestion { id, .. }
            | DaemonCommand::ExecuteBash { id, .. }
            | DaemonCommand::AbortBash { id, .. }
            | DaemonCommand::CancelRlmChild { id, .. }
            | DaemonCommand::DeleteRlmSubagent { id, .. }
            | DaemonCommand::WaitForIdle { id, .. }
            | DaemonCommand::WaitForHeadlessCompletion { id, .. }
            | DaemonCommand::GetSessionHeader { id, .. }
            | DaemonCommand::GetState { id, .. }
            | DaemonCommand::GetConnectionState { id, .. }
            | DaemonCommand::GetMessages { id, .. }
            | DaemonCommand::GetHistoryRange { id, .. }
            | DaemonCommand::GetRlmChildren { id, .. }
            | DaemonCommand::GetSessionStats { id, .. }
            | DaemonCommand::GetContextTree { id, .. }
            | DaemonCommand::GetCommands { id, .. }
            | DaemonCommand::GetResourceSnapshot { id, .. }
            | DaemonCommand::ReplaceAcpMcpServers { id, .. }
            | DaemonCommand::GetModelCatalog { id, .. }
            | DaemonCommand::GetAvailableModels { id, .. }
            | DaemonCommand::GetQueue { id, .. }
            | DaemonCommand::MutateQueuedMessage { id, .. }
            | DaemonCommand::ClearQueue { id, .. }
            | DaemonCommand::AbortAndClearQueue { id, .. }
            | DaemonCommand::AcquireSessionInputPause { id, .. }
            | DaemonCommand::ReleaseSessionInputPause { id, .. }
            | DaemonCommand::CronList { id, .. }
            | DaemonCommand::HeartbeatsList { id, .. }
            | DaemonCommand::HeartbeatManage { id, .. }
            | DaemonCommand::CronAdd { id, .. }
            | DaemonCommand::CronCancel { id, .. }
            | DaemonCommand::HeartbeatGet { id, .. }
            | DaemonCommand::HeartbeatSet { id, .. }
            | DaemonCommand::HeartbeatUpdate { id, .. }
            | DaemonCommand::SetModel { id, .. }
            | DaemonCommand::CycleModel { id, .. }
            | DaemonCommand::SetScopedModels { id, .. }
            | DaemonCommand::SetThinkingLevel { id, .. }
            | DaemonCommand::SetServiceTier { id, .. }
            | DaemonCommand::CycleThinkingLevel { id, .. }
            | DaemonCommand::SetTransport { id, .. }
            | DaemonCommand::SetSteeringMode { id, .. }
            | DaemonCommand::SetFollowUpMode { id, .. }
            | DaemonCommand::SetAutoCompaction { id, .. }
            | DaemonCommand::SetAutoRetry { id, .. }
            | DaemonCommand::Compact { id, .. }
            | DaemonCommand::Refine { id, .. }
            | DaemonCommand::AbortCompaction { id, .. }
            | DaemonCommand::AbortBranchSummary { id, .. }
            | DaemonCommand::AbortRetry { id, .. }
            | DaemonCommand::ExecuteBashAndWait { id, .. }
            | DaemonCommand::Reload { id, .. }
            | DaemonCommand::NewSession { id, .. }
            | DaemonCommand::SwitchSession { id, .. }
            | DaemonCommand::Fork { id, .. }
            | DaemonCommand::NavigateTree { id, .. }
            | DaemonCommand::ImportJsonl { id, .. }
            | DaemonCommand::ExportHtml { id, .. }
            | DaemonCommand::ExportJsonl { id, .. }
            | DaemonCommand::SetSessionName { id, .. }
            | DaemonCommand::GetRlmMaxDepthStatus { id, .. }
            | DaemonCommand::SetRlmMaxDepth { id, .. }
            | DaemonCommand::RenameSavedSession { id, .. }
            | DaemonCommand::DeleteSavedSession { id, .. }
            | DaemonCommand::GetSessionContext { id, .. }
            | DaemonCommand::GetSessionTree { id, .. }
            | DaemonCommand::GetUserMessagesForForking { id, .. }
            | DaemonCommand::GetLastAssistantText { id, .. }
            | DaemonCommand::GetSystemPrompt { id, .. }
            | DaemonCommand::GetToolDefinition { id, .. }
            | DaemonCommand::SetSessionEntryLabel { id, .. }
            | DaemonCommand::ExtensionUiResponse { id, .. }
            | DaemonCommand::PrepareUpdateRestart { id }
            | DaemonCommand::RetryWorker { id, .. }
            | DaemonCommand::Restart { id }
            | DaemonCommand::ListSavedSessions { id, .. }
            | DaemonCommand::Shutdown { id, .. } => id.as_deref(),
        }
    }

    /// The command's serialized object form (with `id`, when present).
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// `DaemonCommandBody` - the command without its `id`.
    pub fn body_without_id(&self) -> Value {
        let mut value = self.to_value();
        if let Some(object) = value.as_object_mut() {
            object.shift_remove("id");
        }
        value
    }

    pub fn from_value(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        self.to_value().as_object().and_then(|object| object.get(key).cloned())
    }

    /// Mirrors `command.field !== undefined` for the optional fields the
    /// compatibility checks look at.
    pub fn has_field(&self, key: &str) -> bool {
        self.to_value()
            .as_object()
            .is_some_and(|object| object.get(key).is_some_and(|value| !value.is_null()))
    }
}

/// `DaemonCommand["type"]` - the union's discriminant literals.
pub type DaemonCommandName = &'static str;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonCommandCompatibility {
    #[serde(rename = "minProtocol")]
    pub min_protocol: u32,
    #[serde(rename = "minSchemaRevision", skip_serializing_if = "Option::is_none", default)]
    pub min_schema_revision: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub capability: Option<DaemonServerCapability>,
}

impl DaemonCommandCompatibility {
    pub const fn legacy() -> Self {
        Self { min_protocol: 7, min_schema_revision: None, capability: None }
    }

    pub const fn capability(capability: DaemonServerCapability) -> Self {
        Self { min_protocol: 7, min_schema_revision: None, capability: Some(capability) }
    }

    pub const fn revision(min_schema_revision: u32) -> Self {
        Self { min_protocol: 7, min_schema_revision: Some(min_schema_revision), capability: None }
    }

    pub const fn gated(min_schema_revision: u32, capability: DaemonServerCapability) -> Self {
        Self { min_protocol: 7, min_schema_revision: Some(min_schema_revision), capability: Some(capability) }
    }
}

/// `DAEMON_COMMAND_COMPATIBILITY`, verbatim from the TypeScript table
/// (`satisfies Record<DaemonCommandName, DaemonCommandCompatibility>`).
pub fn daemon_command_compatibility(command: &str) -> DaemonCommandCompatibility {
    use DaemonServerCapability as Capability;
    match command {
        "list_agent_peers" => DaemonCommandCompatibility::revision(23),
        "get_direct_worker_transport" => {
            DaemonCommandCompatibility::gated(25, Capability::DirectPeerTransport)
        }
        "complete_owned_session" | "promote_owned_session" => {
            DaemonCommandCompatibility::capability(Capability::ClientOwnedSessions)
        }
        "prompt" | "prompt_and_wait" | "steer" | "follow_up" | "resume_queue" => {
            DaemonCommandCompatibility::capability(Capability::SessionInputAdmission)
        }
        "cancel_prompt_admission" => {
            DaemonCommandCompatibility::gated(8, Capability::PromptAdmissionCancellation)
        }
        "delete_rlm_subagent" => DaemonCommandCompatibility::capability(Capability::DeleteRlmSubagent),
        "get_history_range" => DaemonCommandCompatibility::gated(29, Capability::HistoryRanges),
        "get_rlm_children" => DaemonCommandCompatibility::gated(17, Capability::AuthoritativeChildRoster),
        "replace_acp_mcp_servers" => DaemonCommandCompatibility::gated(22, Capability::AcpMcpServers),
        "get_model_catalog" => DaemonCommandCompatibility::capability(Capability::ModelCatalog),
        "mutate_queued_message" => {
            DaemonCommandCompatibility::gated(15, Capability::QueueMessageMutation)
        }
        "acquire_session_input_pause" | "release_session_input_pause" => {
            DaemonCommandCompatibility::gated(19, Capability::SessionInputPause)
        }
        "heartbeats_list" => DaemonCommandCompatibility::capability(Capability::HeartbeatCatalog),
        "roster_subscribe" | "roster_unsubscribe" => {
            DaemonCommandCompatibility::capability(Capability::AgentRoster)
        }
        "heartbeat_manage" => DaemonCommandCompatibility::capability(Capability::HeartbeatManagement),
        "get_rlm_max_depth_status" | "set_rlm_max_depth" => DaemonCommandCompatibility::revision(11),
        "get_session_tree" => DaemonCommandCompatibility::legacy(),
        // Legacy Jev control remains available without the expansion capability.
        // The free-JSON mode field has an additional gate in daemon_jev_mode_compatibility.
        "jev_get_settings" | "jev_set_session_mode" => {
            DaemonCommandCompatibility::capability(Capability::JevControl)
        }
        // Revision 31 usage is optional; older workers still provide valid status.
        "jev_get_status" => {
            DaemonCommandCompatibility::capability(Capability::JevControl)
        }
        _ => DaemonCommandCompatibility::legacy(),
    }
}

/// Additional requirement for the free-JSON Jev mode setter. Parsing matches
/// the worker, so aliases cannot bypass the capability check on send or replay.
pub fn daemon_jev_mode_compatibility(mode: &str) -> Option<DaemonCommandCompatibility> {
    (pi_jev::types::JevMode::parse(mode) == Some(pi_jev::types::JevMode::CompareAndActive))
        .then_some(DaemonCommandCompatibility::gated(30, DaemonServerCapability::JevFeatures))
}

// The command-level gates the union's members carry on top of the table.
const OWNED_SESSION_RECOVERY_CONTEXT: DaemonCommandCompatibility =
    DaemonCommandCompatibility::gated(17, DaemonServerCapability::OwnedSessionRecoveryContext);
const TELEMETRY_POLICY_COMMAND: DaemonCommandCompatibility = DaemonCommandCompatibility::revision(14);
const PROMPT_ADMISSION_CANCELLATION_COMMAND: DaemonCommandCompatibility =
    DaemonCommandCompatibility::gated(8, DaemonServerCapability::PromptAdmissionCancellation);
const RLM_QUIESCENCE_BARRIER_COMMAND: DaemonCommandCompatibility =
    DaemonCommandCompatibility::gated(18, DaemonServerCapability::RlmQuiescenceBarrier);
const OWNED_PROMPT_CANCELLATION_COMMAND: DaemonCommandCompatibility =
    DaemonCommandCompatibility::gated(20, DaemonServerCapability::OwnedPromptCancellation);

/// `DAEMON_COMMAND_PLANE`: which endpoint serves each command when a client
/// holds both a supervisor and a direct worker connection.
pub fn daemon_command_plane(command: &str) -> Option<&'static str> {
    match command {
        "ack_result" | "list" | "list_saved_sessions" | "list_agent_peers" | "get_direct_worker_transport"
        | "create" | "reattach" | "complete_owned_session" | "promote_owned_session" | "kill" | "rename"
        | "send_message" | "agent_messages_status" | "agent_messages_pause" | "agent_messages_resume"
        | "agent_messages_clear" | "cron_list" | "heartbeats_list" | "roster_subscribe"
        | "roster_unsubscribe" | "heartbeat_manage" | "cron_add" | "cron_cancel" | "heartbeat_get"
        | "heartbeat_set" | "heartbeat_update" | "set_session_name" | "rename_saved_session"
        | "delete_saved_session" | "prepare_update_restart" | "retry_worker" | "restart" | "shutdown" => {
            Some("control")
        }
        "attach" | "detach" | "prompt" | "cancel_prompt_admission" | "prompt_and_wait" | "steer"
        | "follow_up" | "restore_next_turn" | "restore_actions" | "append_custom_message" | "resume_queue"
        | "abort" | "start_side_question" | "abort_side_question" | "execute_bash" | "abort_bash"
        | "cancel_rlm_child" | "delete_rlm_subagent" | "wait_for_idle" | "wait_for_headless_completion"
        | "get_session_header" | "get_state" | "get_connection_state" | "get_messages" | "get_history_range"
        | "get_rlm_children" | "get_session_stats" | "get_context_tree" | "get_commands"
        | "get_resource_snapshot" | "replace_acp_mcp_servers" | "get_model_catalog" | "get_available_models"
        | "get_queue" | "mutate_queued_message" | "clear_queue" | "abort_and_clear_queue"
        | "acquire_session_input_pause" | "release_session_input_pause" | "set_model" | "cycle_model"
        | "set_scoped_models" | "set_thinking_level" | "set_service_tier" | "cycle_thinking_level"
        | "set_transport" | "set_steering_mode" | "set_follow_up_mode" | "set_auto_compaction"
        | "set_auto_retry" | "compact" | "refine" | "abort_compaction" | "abort_branch_summary"
        | "abort_retry" | "execute_bash_and_wait" | "reload" | "new_session" | "switch_session" | "fork"
        | "navigate_tree" | "import_jsonl" | "export_html" | "export_jsonl" | "get_rlm_max_depth_status"
        | "set_rlm_max_depth" | "get_session_context" | "get_session_tree" | "get_user_messages_for_forking"
        | "get_last_assistant_text" | "get_system_prompt" | "get_tool_definition" | "set_session_entry_label"
        | "extension_ui_response" | "jev_get_settings" | "jev_set_session_mode" | "jev_get_status" => {
            Some("session")
        }
        _ => None,
    }
}

pub fn is_session_plane_daemon_command(command_type: &str) -> bool {
    daemon_command_plane(command_type) == Some("session")
}

pub fn get_daemon_command_compatibilities(command: &DaemonCommand) -> Vec<DaemonCommandCompatibility> {
    let mut requirements: Vec<DaemonCommandCompatibility> = Vec::new();
    let command_type = command.command_type();
    if matches!(command_type, "prompt" | "prompt_and_wait" | "steer" | "follow_up")
        && command.get("message").and_then(|message| message.as_str().map(str::to_owned))
            .and_then(|message| crate::core::slash_commands::parse_session_slash_command(&message))
            .is_some_and(|command| command.name == "mode")
    {
        requirements.push(DaemonCommandCompatibility::gated(34, DaemonServerCapability::ExecutionMode));
    }
    if (command_type == "attach" || command_type == "reattach") && command.has_field("recoveryConfig") {
        requirements.push(OWNED_SESSION_RECOVERY_CONTEXT);
    }
    let carries_telemetry_policy = ((command_type == "attach" || command_type == "reattach")
        && command.has_field("telemetryDisabled"))
        || (command_type == "create"
            && command
                .get("config")
                .and_then(|config| config.as_object().map(|config| config.contains_key("telemetryDisabled")))
                .unwrap_or(false));
    if carries_telemetry_policy {
        requirements.push(TELEMETRY_POLICY_COMMAND);
    }
    if (command_type == "prompt" || command_type == "prompt_and_wait") && command.has_field("admissionId") {
        requirements.push(PROMPT_ADMISSION_CANCELLATION_COMMAND);
    }
    if command_type == "wait_for_headless_completion" && command.get("waitForRlmQuiescence") == Some(Value::Bool(true)) {
        requirements.push(RLM_QUIESCENCE_BARRIER_COMMAND);
    }
    if command_type == "cancel_prompt_admission" && command.get("cancelOwned") == Some(Value::Bool(true)) {
        requirements.push(OWNED_PROMPT_CANCELLATION_COMMAND);
    }
    requirements.push(daemon_command_compatibility(command_type));
    requirements
}

/// The `hello` shape compatibility checks read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonCompatibilityHello {
    pub protocol: DaemonProtocolInfo,
    #[serde(rename = "schemaRevision", skip_serializing_if = "Option::is_none", default)]
    pub schema_revision: Option<u32>,
    #[serde(rename = "serverCapabilities", skip_serializing_if = "Option::is_none", default)]
    pub server_capabilities: Option<Vec<DaemonServerCapability>>,
}

pub fn meets_daemon_command_compatibility(
    hello: &DaemonCompatibilityHello,
    compatibility: &DaemonCommandCompatibility,
) -> bool {
    hello.protocol.version >= compatibility.min_protocol
        && compatibility
            .min_schema_revision
            .is_none_or(|min_schema_revision| hello.schema_revision.unwrap_or(0) >= min_schema_revision)
        && compatibility.capability.is_none_or(|capability| {
            hello
                .server_capabilities
                .as_ref()
                .is_some_and(|capabilities| capabilities.contains(&capability))
        })
}

/// `{ id?: string; type: "response"; command: string; success: true; data?: unknown }`
/// `| { id?: string; type: "response"; command: string; success: false; error: string; errorInfo?: DaemonErrorInfo }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonResponse {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub type_: String,
    pub command: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,
    #[serde(rename = "errorInfo", skip_serializing_if = "Option::is_none", default)]
    pub error_info: Option<DaemonErrorInfo>,
}

impl DaemonResponse {
    /// `success(id, command, data?)` (daemon-protocol.ts).
    pub fn success(id: Option<&str>, command: &str, data: Option<Value>) -> Self {
        Self {
            id: id.map(str::to_string),
            type_: "response".to_string(),
            command: command.to_string(),
            success: true,
            data,
            error: None,
            error_info: None,
        }
    }

    /// `failure(id, command, error, errorInfo?)` (daemon-protocol.ts).
    pub fn failure(
        id: Option<&str>,
        command: &str,
        error: &str,
        error_info: Option<DaemonErrorInfo>,
    ) -> Self {
        Self {
            id: id.map(str::to_string),
            type_: "response".to_string(),
            command: command.to_string(),
            success: false,
            data: None,
            error: Some(error.to_string()),
            error_info,
        }
    }

    /// `isDaemonResponse(value)` (daemon-client.ts) plus the cast that follows it.
    pub fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.get("type")?.as_str()? != "response" {
            return None;
        }
        if !object.get("success")?.is_boolean() {
            return None;
        }
        let command = object.get("command")?.as_str()?.to_string();
        Some(Self {
            id: object.get("id").and_then(Value::as_str).map(str::to_string),
            type_: "response".to_string(),
            command,
            success: object.get("success").and_then(Value::as_bool).unwrap_or(false),
            data: object.get("data").cloned(),
            error: object.get("error").and_then(Value::as_str).map(str::to_string),
            error_info: object
                .get("errorInfo")
                .and_then(|value| serde_json::from_value(value.clone()).ok()),
        })
    }
}

/// `DaemonSessionClosedReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonSessionClosedReason {
    Killed,
    Shutdown,
    Completed,
    Replaced,
    Update,
}

impl DaemonSessionClosedReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonSessionClosedReason::Killed => "killed",
            DaemonSessionClosedReason::Shutdown => "shutdown",
            DaemonSessionClosedReason::Completed => "completed",
            DaemonSessionClosedReason::Replaced => "replaced",
            DaemonSessionClosedReason::Update => "update",
        }
    }
}

/// `DaemonClosingReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonClosingReason {
    Shutdown,
    Update,
}

impl DaemonClosingReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonClosingReason::Shutdown => "shutdown",
            DaemonClosingReason::Update => "update",
        }
    }
}

/// `{ type: "select" | "confirm" | "input" | "editor" }` - the dialog methods.
pub const DAEMON_DIALOG_EXTENSION_UI_METHODS: [&str; 4] = ["select", "confirm", "input", "editor"];

pub fn is_daemon_dialog_extension_ui_request(method: &str) -> bool {
    DAEMON_DIALOG_EXTENSION_UI_METHODS.contains(&method)
}

/// True when a daemon rejected a command it does not know, i.e. the daemon
/// process was started from a build that predates the command.
pub fn is_unknown_daemon_command_error(message: &str, command: DaemonCommandName) -> bool {
    message.contains(&format!("Unknown daemon command: {command}"))
}

/// `DaemonRequestProgress` - the `list_saved_sessions` progress frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonRequestProgress {
    #[serde(rename = "session_list_progress")]
    SessionListProgress {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        command: String,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        loaded: f64,
        total: f64,
    },
    #[serde(rename = "session_list_item")]
    SessionListItem {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        id: Option<String>,
        command: String,
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        session: DaemonSavedSessionInfo,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSavedSessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub state: Option<AgentConnectionSavedSessionState>,
    #[serde(rename = "parentSessionPath", skip_serializing_if = "Option::is_none", default)]
    pub parent_session_path: Option<String>,
    #[serde(rename = "rlmDepth", skip_serializing_if = "Option::is_none", default)]
    pub rlm_depth: Option<f64>,
    pub created: String,
    pub modified: String,
    #[serde(rename = "messageCount")]
    pub message_count: f64,
    #[serde(rename = "firstMessage")]
    pub first_message: String,
    #[serde(rename = "allMessagesText")]
    pub all_messages_text: String,
    #[serde(rename = "agentStatus", skip_serializing_if = "Option::is_none", default)]
    pub agent_status: Option<AgentConnectionAgentStatus>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<SessionUsageSummary>,
}

/// Type aliases the protocol publishes for its consumers.
pub type DaemonDeleteSavedSessionResult = DeleteSessionFileResult;
pub type DaemonAutonomousStatus = AgentAutonomousStatus;
pub type DaemonBashResult = BashResult;
pub type DaemonSessionHeader = AgentConnectionSessionHeader;
pub type DaemonResourceSnapshot = AgentConnectionResourceSnapshot;
pub type DaemonCronJob = AgentCronJob;
pub type DaemonHeartbeat = AgentConnectionHeartbeat;
pub type DaemonAgentSessionMessageReceipt = AgentSessionMessageReceipt;
pub type DaemonAgentSessionMessageSafetyStatus = AgentSessionMessageSafetyStatus;

/// `DaemonOutbound` - every frame the daemon can write, discriminated by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonOutbound {
    Response(DaemonResponse),
    #[serde(rename = "session_list_progress")]
    SessionListProgress(DaemonRequestProgress),
    #[serde(rename = "daemon_hello")]
    DaemonHello {
        #[serde(rename = "socketPath")]
        socket_path: String,
        protocol: DaemonProtocolInfo,
        #[serde(rename = "schemaId", skip_serializing_if = "Option::is_none", default)]
        schema_id: Option<String>,
        /// Monotonic wire-schema revision for field-sensitive compatibility checks.
        #[serde(rename = "schemaRevision", skip_serializing_if = "Option::is_none", default)]
        schema_revision: Option<u32>,
        /// App version of the daemon process, used to detect stale daemons after self-update.
        #[serde(rename = "appVersion", skip_serializing_if = "Option::is_none", default)]
        app_version: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        runtime: Option<DaemonRuntimeIdentity>,
        /// Changes whenever the public supervisor process is replaced.
        #[serde(rename = "supervisorGeneration", skip_serializing_if = "Option::is_none", default)]
        supervisor_generation: Option<String>,
        /// Diagnostic process identity for attributing supervisor replacement.
        #[serde(rename = "supervisorPid", skip_serializing_if = "Option::is_none", default)]
        supervisor_pid: Option<f64>,
        /// Durable owner marker for validating update handoff fences.
        #[serde(rename = "supervisorOwnerToken", skip_serializing_if = "Option::is_none", default)]
        supervisor_owner_token: Option<String>,
        /// Process start identity captured when the durable owner was published.
        #[serde(rename = "supervisorProcessStartId", skip_serializing_if = "Option::is_none", default)]
        supervisor_process_start_id: Option<String>,
        /// Normalized socket identity stored in the durable owner record.
        #[serde(rename = "supervisorSocketPath", skip_serializing_if = "Option::is_none", default)]
        supervisor_socket_path: Option<String>,
        #[serde(rename = "clientId")]
        client_id: DaemonClientId,
        #[serde(rename = "serverCapabilities")]
        server_capabilities: Vec<DaemonServerCapability>,
    },
    #[serde(rename = "daemon_closing")]
    DaemonClosing { reason: DaemonClosingReason },
    #[serde(rename = "heartbeats_changed")]
    HeartbeatsChanged {
        #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
        active_session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "roster_update")]
    RosterUpdate {
        changed: Vec<AgentRosterEntry>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        removed: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        resync: Option<bool>,
    },
    #[serde(rename = "session_event")]
    SessionEvent {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(deserialize_with = "crate::modes::agent_connection::types::deserialize_agent_connection_session_event")]
        event: AgentConnectionSessionEvent,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "side_question_event")]
    SideQuestionEvent {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        event: AgentConnectionSideQuestionEvent,
    },
    #[serde(rename = "session_status")]
    SessionStatus {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        recap: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "session_replaced")]
    SessionReplaced {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        state: AgentConnectionState,
        messages: Vec<AgentMessage>,
        #[serde(rename = "snapshotFollows", skip_serializing_if = "Option::is_none", default)]
        snapshot_follows: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "session_resynced")]
    SessionResynced {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        snapshot: DaemonSessionSnapshot,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "session_attached")]
    SessionAttached {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        state: SessionSummary,
        messages: Vec<AgentMessage>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        snapshot: Option<DaemonSessionSnapshot>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        replay: Option<DaemonReplayInfo>,
        #[serde(rename = "lastEventSequence", skip_serializing_if = "Option::is_none", default)]
        last_event_sequence: Option<DaemonEventSequence>,
    },
    #[serde(rename = "session_snapshot_begin")]
    SessionSnapshotBegin {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "snapshotId")]
        snapshot_id: String,
        /// `Omit<DaemonSessionSnapshot, "messages">`.
        snapshot: DaemonSessionSnapshotHead,
        #[serde(rename = "messageCount")]
        message_count: f64,
        #[serde(rename = "targetChunkBytes")]
        target_chunk_bytes: f64,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        purpose: Option<String>,
    },
    #[serde(rename = "session_snapshot_chunk")]
    SessionSnapshotChunk {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "snapshotId")]
        snapshot_id: String,
        index: f64,
        messages: Vec<AgentMessage>,
    },
    #[serde(rename = "session_snapshot_end")]
    SessionSnapshotEnd {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "snapshotId")]
        snapshot_id: String,
        #[serde(rename = "chunkCount")]
        chunk_count: f64,
        #[serde(rename = "lastEventSequence")]
        last_event_sequence: DaemonEventSequence,
        #[serde(rename = "lastEventCursor", skip_serializing_if = "Option::is_none", default)]
        last_event_cursor: Option<DaemonEventCursor>,
    },
    #[serde(rename = "session_snapshot_failed")]
    SessionSnapshotFailed {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "snapshotId")]
        snapshot_id: String,
        error: String,
    },
    #[serde(rename = "session_detached")]
    SessionDetached {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
    },
    #[serde(rename = "session_closed")]
    SessionClosed {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        reason: DaemonSessionClosedReason,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "extension_ui_request")]
    ExtensionUiRequest {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        id: String,
        method: String,
        payload: Map<String, Value>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
    #[serde(rename = "extension_error")]
    ExtensionError {
        #[serde(rename = "activeSessionId")]
        active_session_id: String,
        #[serde(rename = "extensionPath")]
        extension_path: String,
        event: String,
        error: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        meta: Option<DaemonEventMeta>,
    },
}

/// `Omit<DaemonSessionSnapshot, "messages">` on the chunked-snapshot handshake.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonSessionSnapshotHead {
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    pub summary: SessionSummary,
    pub state: AgentConnectionState,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub history: Option<DaemonHistoryWindow>,
    #[serde(rename = "sessionContext", skip_serializing_if = "Option::is_none", default)]
    pub session_context: Option<AgentConnectionSessionContext>,
    #[serde(rename = "sessionTree", skip_serializing_if = "Option::is_none", default)]
    pub session_tree: Option<DaemonSessionTree>,
    #[serde(rename = "lastEventSequence")]
    pub last_event_sequence: DaemonEventSequence,
    #[serde(rename = "lastEventCursor", skip_serializing_if = "Option::is_none", default)]
    pub last_event_cursor: Option<DaemonEventCursor>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent: Option<DaemonSessionSnapshotParent>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub children: Option<Vec<AgentConnectionRlmChildAgentSnapshot>>,
}

impl DaemonOutbound {
    pub fn outbound_type(&self) -> &'static str {
        match self {
            DaemonOutbound::Response(_) => "response",
            DaemonOutbound::SessionListProgress(_) => "session_list_progress",
            DaemonOutbound::DaemonHello { .. } => "daemon_hello",
            DaemonOutbound::DaemonClosing { .. } => "daemon_closing",
            DaemonOutbound::HeartbeatsChanged { .. } => "heartbeats_changed",
            DaemonOutbound::RosterUpdate { .. } => "roster_update",
            DaemonOutbound::SessionEvent { .. } => "session_event",
            DaemonOutbound::SideQuestionEvent { .. } => "side_question_event",
            DaemonOutbound::SessionStatus { .. } => "session_status",
            DaemonOutbound::SessionReplaced { .. } => "session_replaced",
            DaemonOutbound::SessionResynced { .. } => "session_resynced",
            DaemonOutbound::SessionAttached { .. } => "session_attached",
            DaemonOutbound::SessionSnapshotBegin { .. } => "session_snapshot_begin",
            DaemonOutbound::SessionSnapshotChunk { .. } => "session_snapshot_chunk",
            DaemonOutbound::SessionSnapshotEnd { .. } => "session_snapshot_end",
            DaemonOutbound::SessionSnapshotFailed { .. } => "session_snapshot_failed",
            DaemonOutbound::SessionDetached { .. } => "session_detached",
            DaemonOutbound::SessionClosed { .. } => "session_closed",
            DaemonOutbound::ExtensionUiRequest { .. } => "extension_ui_request",
            DaemonOutbound::ExtensionError { .. } => "extension_error",
        }
    }

    /// `isDaemonResponse(value)` - the `response` arm.
    pub fn as_response(&self) -> Option<&DaemonResponse> {
        match self {
            DaemonOutbound::Response(response) => Some(response),
            _ => None,
        }
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    pub fn from_value(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

/// `DAEMON_OUTBOUND_COMPATIBILITY`. Revision 28's opaque context is owned by the
/// worker; message/snapshot consumers may ignore it, so response and session
/// channels retain their protocol-7 floor. Revisions 30 and 31 add no event types;
/// optional Jev response metadata is ignored by legacy readers.
pub fn daemon_outbound_compatibility(outbound: &'static str) -> DaemonCommandCompatibility {
    use DaemonServerCapability as Capability;
    match outbound {
        "heartbeats_changed" => DaemonCommandCompatibility::capability(Capability::HeartbeatCatalog),
        "roster_update" => DaemonCommandCompatibility::capability(Capability::AgentRoster),
        // Jev feature/usage metadata is additive; no new startup or event requirement.
        "response" | "extension_ui_request" => DaemonCommandCompatibility::legacy(),
        _ => DaemonCommandCompatibility::legacy(),
    }
}

pub fn create_daemon_command_envelope(
    command: DaemonCommand,
    id: DaemonCommandId,
    client_id: Option<DaemonClientId>,
    protocol_version: DaemonProtocolVersion,
) -> DaemonCommandEnvelope {
    DaemonCommandEnvelope {
        type_: "command".to_string(),
        id,
        protocol: DaemonProtocolInfo { name: DAEMON_PROTOCOL_NAME.to_string(), version: protocol_version },
        client_id,
        command,
    }
}

pub fn is_daemon_command_envelope(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    let Some(protocol) = candidate.get("protocol").and_then(Value::as_object) else {
        return false;
    };
    let Some(version) = protocol.get("version").and_then(Value::as_u64) else {
        return false;
    };
    let client_id_ok = match candidate.get("clientId") {
        None | Some(Value::Null) => true,
        Some(value) => value.is_string(),
    };
    candidate.get("type").and_then(Value::as_str) == Some("command")
        && candidate.get("id").and_then(Value::as_str).is_some()
        && protocol.get("name").and_then(Value::as_str) == Some(DAEMON_PROTOCOL_NAME)
        && u32::try_from(version)
            .is_ok_and(|version| version >= DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION && version <= DAEMON_PROTOCOL_VERSION)
        && client_id_ok
        && candidate.get("command").is_some_and(|command| command.is_object())
}

/// Best-effort id salvage for rejected command lines, so parse failures reach
/// the sender as correlatable responses instead of client-side timeouts.
/// Deliberately ignores everything but the id itself: whatever made the line
/// unparseable (rejected protocol version, missing or invalid type), the
/// sender still correlates the failure by id.
pub fn salvage_daemon_command_id(line: &str) -> Option<String> {
    let candidate: Value = serde_json::from_str(line).ok()?;
    let candidate = candidate.as_object()?;
    candidate.get("id").and_then(Value::as_str).map(str::to_string)
}

/// `READ_ONLY_DAEMON_COMMANDS`.
pub const READ_ONLY_DAEMON_COMMANDS: [DaemonCommandName; 36] = [
    "ack_result",
    "list",
    "list_saved_sessions",
    "list_agent_peers",
    "get_direct_worker_transport",
    "attach",
    "reattach",
    "roster_subscribe",
    "roster_unsubscribe",
    "agent_messages_status",
    "wait_for_idle",
    "get_session_header",
    "get_state",
    "get_connection_state",
    "get_messages",
    "get_history_range",
    "get_rlm_children",
    "get_session_stats",
    "get_context_tree",
    "get_commands",
    "get_resource_snapshot",
    "get_model_catalog",
    "get_available_models",
    "get_queue",
    "cron_list",
    "heartbeats_list",
    "heartbeat_get",
    "get_session_context",
    "get_session_tree",
    "get_user_messages_for_forking",
    "get_last_assistant_text",
    "get_system_prompt",
    "get_rlm_max_depth_status",
    "get_tool_definition",
    // SHARED FILE EDIT (daemon_protocol.rs, jev-ui lane): the two Jev GETTERS are
    // read-only. The mode SETTER is deliberately absent: it writes the local mode
    // setting, so a read-only client must not send it.
    "jev_get_settings",
    "jev_get_status",
];

pub fn is_daemon_mutating_command(command_type: &str) -> bool {
    !READ_ONLY_DAEMON_COMMANDS.contains(&command_type)
}

/// `UPDATE_RESTART_DRAIN_COMMANDS`.
pub const UPDATE_RESTART_DRAIN_COMMANDS: [DaemonCommandName; 6] = [
    "extension_ui_response",
    "abort",
    "abort_bash",
    "abort_branch_summary",
    "abort_compaction",
    "abort_retry",
];

pub fn create_daemon_event_envelope(event: DaemonOutbound, meta: &DaemonEventMeta) -> DaemonEventEnvelope {
    DaemonEventEnvelope {
        type_: "event".to_string(),
        id: meta.id.clone(),
        protocol: meta.protocol.clone(),
        active_session_id: meta.active_session_id.clone(),
        sequence: meta.sequence,
        cursor: meta.cursor.clone(),
        emitted_at: meta.emitted_at.clone(),
        event,
    }
}

pub fn create_daemon_event_meta(
    active_session_id: &str,
    sequence: DaemonEventSequence,
    emitted_at: String,
    generation: &str,
) -> DaemonEventMeta {
    DaemonEventMeta {
        id: format!("{active_session_id}:{sequence}"),
        protocol: daemon_protocol_info(),
        active_session_id: Some(active_session_id.to_string()),
        sequence: Some(sequence),
        cursor: Some(DaemonEventCursor { generation: generation.to_string(), sequence }),
        emitted_at,
        replayed: None,
    }
}

/// `createDaemonEventMeta(activeSessionId, sequence, emittedAt = new Date().toISOString(), generation = activeSessionId)`.
pub fn create_daemon_event_meta_now(active_session_id: &str, sequence: DaemonEventSequence) -> DaemonEventMeta {
    create_daemon_event_meta(
        active_session_id,
        sequence,
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        active_session_id,
    )
}

pub fn create_daemon_replay_info(
    resume_cursor: Option<&DaemonResumeCursor>,
    last_event_sequence: DaemonEventSequence,
    generation: &str,
) -> DaemonReplayInfo {
    let to_cursor = DaemonEventCursor { generation: generation.to_string(), sequence: last_event_sequence };
    let Some(resume_cursor) = resume_cursor else {
        return DaemonReplayInfo {
            status: DaemonReplayStatus::Complete,
            from_sequence: None,
            to_sequence: last_event_sequence,
            from_cursor: None,
            to_cursor: Some(to_cursor),
            reason: None,
        };
    };
    let resume_sequence = resume_cursor.resume_sequence();
    let from_cursor = resume_cursor.from_cursor();
    if let Some(from_cursor) = &from_cursor {
        if from_cursor.generation != generation {
            return DaemonReplayInfo {
                status: DaemonReplayStatus::Unavailable,
                from_sequence: Some(resume_sequence),
                to_sequence: last_event_sequence,
                from_cursor: Some(from_cursor.clone()),
                to_cursor: Some(to_cursor),
                reason: Some("event_generation_changed".to_string()),
            };
        }
    }
    if resume_sequence > last_event_sequence {
        return DaemonReplayInfo {
            status: DaemonReplayStatus::Unavailable,
            from_sequence: Some(resume_sequence),
            to_sequence: last_event_sequence,
            from_cursor,
            to_cursor: Some(to_cursor),
            reason: Some("resume_cursor_ahead_of_session".to_string()),
        };
    }
    if resume_sequence == last_event_sequence {
        return DaemonReplayInfo {
            status: DaemonReplayStatus::Complete,
            from_sequence: Some(resume_sequence),
            to_sequence: last_event_sequence,
            from_cursor,
            to_cursor: Some(to_cursor),
            reason: None,
        };
    }
    DaemonReplayInfo {
        status: DaemonReplayStatus::Unavailable,
        from_sequence: Some(resume_sequence),
        to_sequence: last_event_sequence,
        from_cursor,
        to_cursor: Some(to_cursor),
        reason: Some("event_replay_not_available".to_string()),
    }
}

pub fn success(
    id: Option<&str>,
    command: DaemonCommandName,
    data: Option<DaemonOutboundData>,
) -> DaemonResponse {
    let data = match data {
        Some(DaemonOutboundData::Value(value)) => Some(value),
        // `data === undefined` drops the key; an explicit `null` keeps it.
        Some(DaemonOutboundData::Undefined) | None => None,
    };
    DaemonResponse {
        id: id.map(str::to_string),
        type_: "response".to_string(),
        command: command.to_string(),
        success: true,
        data,
        error: None,
        error_info: None,
    }
}

/// Distinguishes an omitted `data` argument from an explicit JSON `null`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DaemonOutboundData {
    Value(Value),
    Undefined,
}

pub fn failure(
    id: Option<&str>,
    command: &str,
    error: &str,
    error_info: Option<DaemonErrorInfo>,
) -> DaemonResponse {
    DaemonResponse {
        id: id.map(str::to_string),
        type_: "response".to_string(),
        command: command.to_string(),
        success: false,
        data: None,
        error: Some(error.to_string()),
        error_info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_mode_commands_are_gated_without_changing_legacy_events() {
        for command_type in ["prompt", "prompt_and_wait", "steer", "follow_up"] {
            let command: DaemonCommand = serde_json::from_value(serde_json::json!({
                "type": command_type, "activeSessionId": "test", "message": "/mode direct"
            })).unwrap();
            let requirements = get_daemon_command_compatibilities(&command);
            let mut old = DaemonCompatibilityHello {
                protocol: daemon_protocol_info(), schema_revision: Some(33),
                server_capabilities: Some(vec![DaemonServerCapability::SessionInputAdmission]),
            };
            assert!(!requirements.iter().all(|r| meets_daemon_command_compatibility(&old, r)));
            old.schema_revision = Some(34);
            assert!(!requirements.iter().all(|r| meets_daemon_command_compatibility(&old, r)));
            old.server_capabilities.as_mut().unwrap().push(DaemonServerCapability::ExecutionMode);
            assert!(requirements.iter().all(|r| meets_daemon_command_compatibility(&old, r)));
        }
        // Old clients continue to consume the existing snapshots and queue events.
        let old = DaemonCompatibilityHello { protocol: daemon_protocol_info(), schema_revision: Some(33), server_capabilities: None };
        for event in ["session_event", "session_attached", "session_resynced", "response"] {
            assert!(meets_daemon_command_compatibility(&old, &daemon_outbound_compatibility(event)));
        }
    }

    #[test]
    fn keeps_provider_compaction_metadata_optional_for_older_clients_and_workers() {
        let older_peer = DaemonCompatibilityHello {
            protocol: daemon_protocol_info(),
            schema_revision: Some(27),
            server_capabilities: None,
        };
        for command in ["compact", "get_messages", "get_session_context"] {
            assert!(meets_daemon_command_compatibility(
                &older_peer,
                &daemon_command_compatibility(command)
            ));
        }
        for event in ["response", "session_event", "session_attached", "session_resynced"] {
            assert!(meets_daemon_command_compatibility(
                &older_peer,
                &daemon_outbound_compatibility(event)
            ));
        }
    }

    #[test]
    fn keeps_the_advertised_schema_identity() {
        assert_eq!(DAEMON_SCHEMA_ID, format!("protocol-{DAEMON_PROTOCOL_VERSION}-schema-{DAEMON_SCHEMA_REVISION}-c16da0e12d5a"));
    }

    #[test]
    fn requires_compatibility_metadata_for_the_heartbeat_protocol_surface() {
        assert_eq!(DAEMON_PROTOCOL_VERSION, 7);
        assert!(DAEMON_SCHEMA_ID.contains(&format!("protocol-{DAEMON_PROTOCOL_VERSION}")));
        assert_eq!(
            daemon_command_compatibility("heartbeats_list"),
            DaemonCommandCompatibility::capability(DaemonServerCapability::HeartbeatCatalog)
        );
        assert_eq!(
            daemon_command_compatibility("heartbeat_manage"),
            DaemonCommandCompatibility::capability(DaemonServerCapability::HeartbeatManagement)
        );
        assert_eq!(
            daemon_command_compatibility("complete_owned_session"),
            DaemonCommandCompatibility::capability(DaemonServerCapability::ClientOwnedSessions)
        );
        assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::HeartbeatCatalog));
        assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::HeartbeatManagement));
    }

    #[test]
    fn capability_and_schema_gates_pinned_history_ranges() {
        assert!(DAEMON_SCHEMA_REVISION >= 29);
        assert_eq!(
            daemon_command_compatibility("get_history_range"),
            DaemonCommandCompatibility::gated(29, DaemonServerCapability::HistoryRanges)
        );
        assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::HistoryRanges));
        assert_eq!(daemon_command_plane("get_history_range"), Some("session"));
        let stale_revision = DaemonCompatibilityHello {
            protocol: daemon_protocol_info(),
            schema_revision: Some(28),
            server_capabilities: Some(vec![DaemonServerCapability::HistoryRanges]),
        };
        assert!(!meets_daemon_command_compatibility(
            &stale_revision,
            &daemon_command_compatibility("get_history_range")
        ));
        let missing_capability = DaemonCompatibilityHello {
            protocol: daemon_protocol_info(),
            schema_revision: Some(29),
            server_capabilities: Some(vec![]),
        };
        assert!(!meets_daemon_command_compatibility(
            &missing_capability,
            &daemon_command_compatibility("get_history_range")
        ));
        assert!(!is_daemon_mutating_command("get_history_range"));
    }

    #[test]
    fn gates_explicit_subagent_deletion_by_capability() {
        assert_eq!(
            daemon_command_compatibility("delete_rlm_subagent"),
            DaemonCommandCompatibility::capability(DaemonServerCapability::DeleteRlmSubagent)
        );
    }

    #[test]
    fn capability_and_schema_gates_every_gated_command() {
        assert_eq!(
            daemon_command_compatibility("replace_acp_mcp_servers"),
            DaemonCommandCompatibility::gated(22, DaemonServerCapability::AcpMcpServers)
        );
        assert_eq!(
            daemon_command_compatibility("get_model_catalog"),
            DaemonCommandCompatibility::capability(DaemonServerCapability::ModelCatalog)
        );
        assert_eq!(
            daemon_command_compatibility("mutate_queued_message"),
            DaemonCommandCompatibility::gated(15, DaemonServerCapability::QueueMessageMutation)
        );
        assert_eq!(daemon_command_compatibility("get_rlm_max_depth_status"), DaemonCommandCompatibility::revision(11));
        assert_eq!(daemon_command_compatibility("set_rlm_max_depth"), DaemonCommandCompatibility::revision(11));
        for command in ["acquire_session_input_pause", "release_session_input_pause"] {
            let requirement = daemon_command_compatibility(command);
            assert_eq!(requirement, DaemonCommandCompatibility::gated(19, DaemonServerCapability::SessionInputPause));
            let mut peer = DaemonCompatibilityHello {
                protocol: daemon_protocol_info(),
                schema_revision: Some(18),
                server_capabilities: Some(vec![DaemonServerCapability::SessionInputPause]),
            };
            assert!(!meets_daemon_command_compatibility(&peer, &requirement));
            peer.schema_revision = Some(19);
            assert!(meets_daemon_command_compatibility(&peer, &requirement));
            peer.server_capabilities = None;
            assert!(!meets_daemon_command_compatibility(&peer, &requirement));
        }
        assert_eq!(
            daemon_command_compatibility("cancel_prompt_admission"),
            DaemonCommandCompatibility::gated(8, DaemonServerCapability::PromptAdmissionCancellation)
        );
        assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::OwnedPromptCancellation));
    }

    #[test]
    fn schema_gates_session_commands_that_carry_the_telemetry_policy() {
        let plain_create = DaemonCommand::from_value(&serde_json::json!({
            "type": "create",
            "config": { "cwd": "/tmp" }
        }))
        .expect("create");
        assert_eq!(get_daemon_command_compatibilities(&plain_create), vec![DaemonCommandCompatibility::legacy()]);

        let telemetry_create = DaemonCommand::from_value(&serde_json::json!({
            "type": "create",
            "config": { "cwd": "/tmp", "telemetryDisabled": true }
        }))
        .expect("create");
        assert_eq!(
            get_daemon_command_compatibilities(&telemetry_create),
            vec![DaemonCommandCompatibility::revision(14), DaemonCommandCompatibility::legacy()]
        );

        let telemetry_attach = DaemonCommand::from_value(&serde_json::json!({
            "type": "attach",
            "activeSessionId": "active-1",
            "telemetryDisabled": true
        }))
        .expect("attach");
        assert_eq!(
            get_daemon_command_compatibilities(&telemetry_attach),
            vec![DaemonCommandCompatibility::revision(14), DaemonCommandCompatibility::legacy()]
        );
    }

    #[test]
    fn capability_gates_authoritative_rosters_and_recovery_context() {
        assert_eq!(
            daemon_command_compatibility("get_rlm_children"),
            DaemonCommandCompatibility::gated(17, DaemonServerCapability::AuthoritativeChildRoster)
        );
        let attach = DaemonCommand::from_value(&serde_json::json!({
            "type": "attach",
            "activeSessionId": "active-1",
            "recoveryConfig": { "cwd": "/tmp/fresh-owner" }
        }))
        .expect("attach");
        assert_eq!(
            get_daemon_command_compatibilities(&attach),
            vec![
                DaemonCommandCompatibility::gated(17, DaemonServerCapability::OwnedSessionRecoveryContext),
                DaemonCommandCompatibility::legacy()
            ]
        );
    }

    #[test]
    fn gates_the_opt_in_rlm_quiescence_wire_field() {
        let flagged = DaemonCommand::from_value(&serde_json::json!({
            "type": "wait_for_headless_completion",
            "activeSessionId": "active-1",
            "waitForRlmQuiescence": true
        }))
        .expect("command");
        assert_eq!(
            get_daemon_command_compatibilities(&flagged),
            vec![
                DaemonCommandCompatibility::gated(18, DaemonServerCapability::RlmQuiescenceBarrier),
                DaemonCommandCompatibility::legacy()
            ]
        );
        let plain = DaemonCommand::from_value(&serde_json::json!({
            "type": "wait_for_headless_completion",
            "activeSessionId": "active-1"
        }))
        .expect("command");
        assert_eq!(get_daemon_command_compatibilities(&plain), vec![DaemonCommandCompatibility::legacy()]);
    }

    #[test]
    fn capability_gates_cancellation_after_prompt_ownership() {
        let legacy = DaemonCommand::from_value(&serde_json::json!({
            "type": "cancel_prompt_admission",
            "activeSessionId": "active-1",
            "admissionId": "a-1"
        }))
        .expect("command");
        assert_eq!(
            get_daemon_command_compatibilities(&legacy),
            vec![daemon_command_compatibility("cancel_prompt_admission")]
        );
        let owned = DaemonCommand::from_value(&serde_json::json!({
            "type": "cancel_prompt_admission",
            "activeSessionId": "active-1",
            "admissionId": "a-1",
            "cancelOwned": true
        }))
        .expect("command");
        assert_eq!(
            get_daemon_command_compatibilities(&owned),
            vec![
                DaemonCommandCompatibility::gated(20, DaemonServerCapability::OwnedPromptCancellation),
                daemon_command_compatibility("cancel_prompt_admission")
            ]
        );
    }

    #[test]
    fn creates_versioned_command_and_event_envelopes() {
        let command = DaemonCommand::from_value(&serde_json::json!({
            "id": "cmd-1",
            "type": "attach",
            "activeSessionId": "active-1"
        }))
        .expect("command");
        let envelope = create_daemon_command_envelope(
            command.clone(),
            "cmd-1".to_string(),
            Some("client-1".to_string()),
            DAEMON_PROTOCOL_VERSION,
        );
        assert_eq!(
            serde_json::to_value(&envelope).unwrap(),
            serde_json::json!({
                "type": "command",
                "id": "cmd-1",
                "protocol": { "name": DAEMON_PROTOCOL_NAME, "version": DAEMON_PROTOCOL_VERSION },
                "clientId": "client-1",
                "command": command.to_value(),
            })
        );

        let meta = create_daemon_event_meta("active-1", 3, "2026-01-01T00:00:00.000Z".to_string(), "active-1");
        let event = DaemonOutbound::SessionStatus {
            active_session_id: "active-1".to_string(),
            recap: None,
            meta: Some(meta.clone()),
        };
        assert_eq!(
            serde_json::to_value(create_daemon_event_envelope(event.clone(), &meta)).unwrap(),
            serde_json::json!({
                "type": "event",
                "id": "active-1:3",
                "protocol": { "name": DAEMON_PROTOCOL_NAME, "version": DAEMON_PROTOCOL_VERSION },
                "activeSessionId": "active-1",
                "sequence": 3,
                "cursor": { "generation": "active-1", "sequence": 3 },
                "emittedAt": "2026-01-01T00:00:00.000Z",
                "event": event.to_value(),
            })
        );
    }

    #[test]
    fn rejects_command_envelopes_from_pre_session_action_protocols() {
        let command = DaemonCommand::from_value(&serde_json::json!({
            "id": "cmd-1",
            "type": "attach",
            "activeSessionId": "active-1"
        }))
        .expect("command");
        let current = serde_json::to_value(create_daemon_command_envelope(
            command.clone(),
            "cmd-1".to_string(),
            Some("client-1".to_string()),
            7,
        ))
        .unwrap();
        assert!(is_daemon_command_envelope(&current));
        let legacy = serde_json::to_value(create_daemon_command_envelope(
            command,
            "cmd-1".to_string(),
            Some("client-1".to_string()),
            6,
        ))
        .unwrap();
        assert!(!is_daemon_command_envelope(&legacy));
    }

    #[test]
    fn keeps_attachment_routing_out_of_the_durable_mutation_journal() {
        assert!(!is_daemon_mutating_command("attach"));
        assert!(!is_daemon_mutating_command("reattach"));
        assert!(!is_daemon_mutating_command("roster_subscribe"));
        assert!(!is_daemon_mutating_command("roster_unsubscribe"));
        assert!(is_daemon_mutating_command("wait_for_headless_completion"));
        assert!(is_daemon_mutating_command("switch_session"));
    }

    #[test]
    fn capability_gates_direct_worker_transport_discovery() {
        assert_eq!(
            daemon_command_compatibility("get_direct_worker_transport"),
            DaemonCommandCompatibility::gated(25, DaemonServerCapability::DirectPeerTransport)
        );
        assert!(!DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::DirectPeerTransport));
        assert!(!is_daemon_mutating_command("get_direct_worker_transport"));
    }

    #[test]
    fn classifies_every_command_plane_and_never_defaults_unknown_commands_to_session() {
        assert_eq!(daemon_command_plane("list"), Some("control"));
        assert_eq!(daemon_command_plane("prompt"), Some("session"));
        assert!(!is_session_plane_daemon_command("no_such_command"));
    }

    #[test]
    fn reports_replay_availability_from_resume_cursors() {
        assert_eq!(
            serde_json::to_value(create_daemon_replay_info(None, 5, "generation-1")).unwrap(),
            serde_json::json!({
                "status": "complete",
                "toSequence": 5,
                "toCursor": { "generation": "generation-1", "sequence": 5 }
            })
        );
        let same = DaemonResumeCursor {
            active_session_id: Some("active-1".to_string()),
            generation: Some("generation-1".to_string()),
            sequence: Some(5),
            event_sequence: None,
        };
        assert_eq!(
            serde_json::to_value(create_daemon_replay_info(Some(&same), 5, "generation-1")).unwrap(),
            serde_json::json!({
                "status": "complete",
                "fromSequence": 5,
                "toSequence": 5,
                "fromCursor": { "generation": "generation-1", "sequence": 5 },
                "toCursor": { "generation": "generation-1", "sequence": 5 }
            })
        );
        let behind = DaemonResumeCursor {
            active_session_id: None,
            generation: Some("generation-1".to_string()),
            sequence: Some(2),
            event_sequence: None,
        };
        assert_eq!(
            serde_json::to_value(create_daemon_replay_info(Some(&behind), 5, "generation-1")).unwrap(),
            serde_json::json!({
                "status": "unavailable",
                "fromSequence": 2,
                "toSequence": 5,
                "fromCursor": { "generation": "generation-1", "sequence": 2 },
                "toCursor": { "generation": "generation-1", "sequence": 5 },
                "reason": "event_replay_not_available"
            })
        );
        let stale_generation = DaemonResumeCursor {
            active_session_id: None,
            generation: Some("old".to_string()),
            sequence: Some(5),
            event_sequence: None,
        };
        let replay = create_daemon_replay_info(Some(&stale_generation), 0, "new");
        assert_eq!(replay.status, DaemonReplayStatus::Unavailable);
        assert_eq!(replay.reason.as_deref(), Some("event_generation_changed"));
    }

    #[test]
    fn salvages_command_ids_from_rejected_lines_regardless_of_shape_validity() {
        let old_envelope = serde_json::to_string(&create_daemon_command_envelope(
            DaemonCommand::from_value(&serde_json::json!({ "type": "list" })).expect("command"),
            "list-1".to_string(),
            Some("old-client".to_string()),
            6,
        ))
        .unwrap();
        assert_eq!(salvage_daemon_command_id(&old_envelope).as_deref(), Some("list-1"));
        assert_eq!(
            salvage_daemon_command_id(&serde_json::json!({ "type": "list", "id": "bare-1" }).to_string()).as_deref(),
            Some("bare-1")
        );
        assert_eq!(
            salvage_daemon_command_id(&serde_json::json!({ "type": null, "id": "typeless-1" }).to_string()).as_deref(),
            Some("typeless-1")
        );
        assert_eq!(
            salvage_daemon_command_id(&serde_json::json!({ "id": "no-type" }).to_string()).as_deref(),
            Some("no-type")
        );
        assert!(salvage_daemon_command_id(&serde_json::json!({ "type": "command", "id": 7 }).to_string()).is_none());
        assert!(salvage_daemon_command_id(&serde_json::json!("command").to_string()).is_none());
        assert!(salvage_daemon_command_id("{ not json").is_none());
    }

    #[test]
    fn success_omits_undefined_data_and_keeps_null() {
        assert_eq!(
            serde_json::to_value(success(Some("cmd-1"), "list", None)).unwrap(),
            serde_json::json!({ "id": "cmd-1", "type": "response", "command": "list", "success": true })
        );
        assert_eq!(
            serde_json::to_value(success(Some("cmd-1"), "list", Some(DaemonOutboundData::Value(Value::Null)))).unwrap(),
            serde_json::json!({
                "id": "cmd-1",
                "type": "response",
                "command": "list",
                "success": true,
                "data": null
            })
        );
    }

    #[test]
    fn failure_carries_error_info_only_when_present() {
        assert_eq!(
            serde_json::to_value(failure(None, "attach", "Unknown active session", None)).unwrap(),
            serde_json::json!({
                "type": "response",
                "command": "attach",
                "success": false,
                "error": "Unknown active session"
            })
        );
        let info = DaemonErrorInfo::SessionRecovering { active_session_id: "active-gap".to_string() };
        let value = serde_json::to_value(failure(None, "attach", "boom", Some(info))).unwrap();
        assert_eq!(value["errorInfo"]["code"], "session_recovering");
        assert_eq!(value["errorInfo"]["activeSessionId"], "active-gap");
    }

    #[test]
    fn round_trips_commands_with_flattened_saved_session_rows() {
        let raw = serde_json::json!({
            "id": "cmd-9",
            "type": "list_saved_sessions",
            "activeSessionId": "active-1",
            "scope": "active_session"
        });
        let command = DaemonCommand::from_value(&raw).expect("command");
        assert_eq!(command.command_type(), "list_saved_sessions");
        assert_eq!(command.id(), Some("cmd-9"));
        assert_eq!(command.to_value(), raw);
    }

    #[test]
    fn keeps_the_allowlisted_client_env_contract() {
        let env = collect_daemon_client_env(&|key| {
            (key == "HERDR_PANE_ID").then(|| "pane-1".to_string())
        })
        .expect("env");
        assert_eq!(env.len(), 1);
        assert_eq!(env.get("HERDR_PANE_ID").map(String::as_str), Some("pane-1"));
        assert!(collect_daemon_client_env(&|_| None).is_none());

        let launch = collect_daemon_launch_env(vec![
            ("PRIME_AGENT_INTERNAL_SECRET".to_string(), Some("hidden".to_string())),
            ("HERDR_ENV".to_string(), Some("1".to_string())),
            ("MISSING".to_string(), None),
        ]);
        assert_eq!(launch.len(), 1);
        assert!(launch.contains_key("HERDR_ENV"));
    }
}
