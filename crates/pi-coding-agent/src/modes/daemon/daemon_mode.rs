//! Port of packages/coding-agent/src/modes/daemon/daemon-mode.ts
//!
//! Background daemon mode. The daemon owns live `AgentSessionRuntime` instances
//! and exposes a small JSONL protocol over a local socket.
//!
//! Slice plumbing: several modules this file imports are owned by other slices
//! and only exist as thin stubs today (`modes/daemon/daemon-protocol.ts`,
//! `modes/daemon/daemon-runtime-identity.ts`, `core/cron-jobs.ts`,
//! `core/agent-session-runtime.ts`, `core/agent-session-config.ts`,
//! `core/rlm-runtime.ts`, `core/side-question.ts`, `config.ts`). The private
//! plumbing below carries the same names and shapes those modules export so the
//! daemon-mode port stands on its own; it moves out unchanged when those slices
//! land. Everything private is marked `// slice plumbing:`.

#[path = "daemon_server.rs"]
mod native_server;

#[path = "daemon_subagents.rs"]
mod daemon_subagents;

#[path = "agent_message_transport.rs"]
mod agent_message_transport;

// Owner parity-validation tests (T08: C-01, C-02, C-09, H-05). In-crate so they can
// drive the private daemon seams (`AgentDaemon::handle_line`,
// `accept_agent_session_message`) without re-implementing them.
#[cfg(test)]
#[path = "daemon_parity_tests.rs"]
mod daemon_parity_tests;

#[cfg(test)]
#[path = "daemon_snapshot_backlog_tests.rs"]
mod daemon_snapshot_backlog_tests;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

use base64::Engine;
use futures::future::BoxFuture;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex, Notify};

use pi_agent_core::types::AgentMessage;

use crate::cli::subprocess_launch::{create_cli_subprocess_env, create_cli_subprocess_launch_spec};
use crate::core::agent_messages::{
    agent_family_relationship, assert_agent_family_reach, assert_agent_session_name_available,
    assert_direct_agent_message_target, build_agent_family_roster, create_agent_session_message,
    create_agent_session_message_id, create_agent_session_message_prompt,
    create_agent_session_message_receipt, format_agent_session_name_unavailable,
    normalize_agent_session_message, session_name_reservation_key, AgentFamilyCatalogEntry,
    AgentFamilyRelationship, AgentFamilyRosterResult, AgentSessionMessageAgentSummary,
    AgentSessionMessageController, AgentSessionMessageDeliveryStatus, AgentSessionMessageEndpoint,
    AgentSessionMessageListResult, AgentSessionMessagePayload, AgentSessionMessageRateLimiter,
    AgentSessionMessageReceipt, AgentSessionMessageSendInput, AgentSessionMessageSender,
    AgentSessionNameAvailabilityInput,
    AgentSessionNameScope, RateLimitResult, AGENT_FAMILY_REACH_ERROR, AGENT_MESSAGE_SOURCE,
    DEFAULT_AGENT_MESSAGE_MAX_CHARS, DEFAULT_AGENT_MESSAGE_MAX_PENDING_PER_SESSION,
    DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY, DEFAULT_AGENT_MESSAGE_RATE_LIMIT_REFILL_MS,
    DELIVERY_MODE_STEER, DELIVERY_STATUS_DELIVERED, DELIVERY_STATUS_QUEUED,
    FAMILY_RELATIONSHIP_PARENT, FAMILY_STATUS_IDLE, FAMILY_STATUS_INACTIVE, FAMILY_STATUS_RUNNING,
    RUNTIME_KIND_SUBAGENT,
};
use crate::core::agent_observe::{
    create_agent_observe_message_preview, normalize_observe_limit, normalize_observe_max_chars,
    AgentObserveAgentSnapshot, AgentObserveAgentSummary, AgentObserveController,
    AgentObserveListResult, AgentObserveRecentMessagesInput, AgentObserveRecentMessagesResult,
};
use crate::core::agent_session::rlm_child_label;
use crate::core::cron_jobs::{
    is_heartbeat_cron_job, normalize_heartbeat_delivery_mode, normalize_heartbeat_schedule,
    resolve_heartbeat_streaming_behavior, should_defer_heartbeat_cron_job, AgentCronJob,
    AgentCronJobStore, AgentCronScheduler, AgentHeartbeatDeliveryMode,
    AgentHeartbeatManagementAction, AgentHeartbeatUpdateAction, CancelJobsForSessionInput,
    CreateAgentCronJobInput, HeartbeatCronSessionActivity, RlmHeartbeatCreateInput,
    RlmHeartbeatUpdateInput, DEFAULT_HEARTBEAT_SCHEDULE, RUN_RESULT_SKIPPED, SOURCE_RLM_HEARTBEAT,
};
use crate::core::orphan_process_journal::ORPHAN_PROCESS_JOURNAL_ENV;
use crate::core::prompt_admission::{wait_for_prompt_admission, PromptAdmissionCancelledError};
use crate::core::session_action_store::{
    can_passivate_session, IdleEvictionMinutes, SessionEvictionSnapshot, SessionPassivationSnapshot,
};
use crate::core::session_file_actions::{
    delete_session_artifacts, delete_session_file, DeleteSessionFileOptions,
    DeleteSessionFileResult,
};
use crate::core::session_lease::{
    canonical_session_path, SESSION_LEASES_ENABLED_ENV, SESSION_LEASE_OWNER_ID_ENV,
};
use crate::core::session_manager::SessionManager;
use crate::core::session_manager::{
    get_session_artifact_path_for_file, order_session_context_for_transcript, read_session_info,
    resolve_session_rlm_depth, SessionHistorySnapshot, SessionInfo,
};
use crate::core::session_resolver::{resolve_session_path, ResolvedSession};
use crate::core::settings_manager::SettingsManager;
use crate::modes::agent_connection::snapshot::{
    create_agent_connection_commands, create_agent_connection_resource_snapshot,
    create_agent_connection_state,
};
use crate::modes::agent_connection::tool_definition::create_agent_connection_tool_definition;
use crate::modes::agent_connection::types::AgentConnectionHeartbeat;
use crate::modes::rpc::jsonl::JsonlLineReader;
use crate::utils::child_process::{
    is_process_alive, spawn_hidden, wait_for_child_process, SpawnOptions,
};
use crate::utils::dir_lock::try_acquire_dir_lock;
use crate::utils::shell::kill_tracked_detached_children;

use super::active_session_state::{
    create_active_session_id, resolve_active_session_state, ActiveSessionExtensionUiRequest,
    ActiveSessionRuntimeSession, ActiveSessionState, AgentSessionRuntime,
    AgentSessionRuntimeMetadata, DaemonExtensionUIResponse, DaemonSocketClient,
};
use super::agent_roster::{
    classify_session_roster_status, passivated_worker_roster_entry, roster_agent_id_for_summary,
    worker_roster_entry_from_summary, RegisteredHeartbeatFlags, RosterSessionSummary,
    RosterSummaryView, WorkerRosterEntry,
};
use super::compact_session_stream::create_compact_assistant_delta;
// The daemon protocol module is owned by another slice; the wire primitives
// this module consumes are carried by daemon_client's protocol submodule.
use super::daemon_protocol::{create_daemon_event_meta, create_daemon_replay_info, is_daemon_command_envelope, is_daemon_dialog_extension_ui_request, is_daemon_mutating_command, is_session_plane_daemon_command, salvage_daemon_command_id, DaemonClientCapability, DaemonResponse, DAEMON_DEFAULT_CLIENT_CAPABILITIES, DAEMON_DEFAULT_SERVER_CAPABILITIES, DAEMON_SCHEMA_ID, DAEMON_SCHEMA_REVISION, DAEMON_SUPPORTED_CLIENT_CAPABILITIES};
use super::daemon_client::DaemonClient;
use super::daemon_client_env::{filter_client_env, with_client_env};
use super::daemon_errors::{
    deserialize_daemon_error, serialize_daemon_error, DaemonError, DaemonErrorInfo,
};
use super::daemon_extension_binding::{
    bind_active_session_state, ActiveSessionBindingCallbacks, DaemonExtensionBindingSession,
    DaemonOutbound as BindingDaemonOutbound,
};
use super::daemon_session_list::{
    build_session_list, has_live_session_work, inactive_lifecycle_for_session,
    scheduled_job_registrations, summary_for_active_session, SessionLifecycle, SessionSummary,
};
use super::daemon_session_summarizer::DaemonSessionSummarizer;
use super::daemon_socket::{
    cleanup_daemon_socket_path, default_daemon_socket_path, get_daemon_socket_identity,
    normalize_socket_path_for_daemon as normalize_socket_path, prepare_daemon_socket_path,
    restrict_daemon_socket_path, DaemonSocketIdentity,
};
use super::daemon_supervisor_ownership::{
    assert_daemon_supervisor_owner_current, is_daemon_shutdown_admission_active,
    DaemonSupervisorOwnerRecord,
};
use super::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};
use super::daemon_worker_protocol::{
    is_daemon_worker_frame_header, DaemonWorkerFrameHeader, DaemonWorkerPeerGrant,
    DaemonWorkerRosterOutbound, DAEMON_WORKER_ACTIVE_SESSION_ID_ENV,
    DAEMON_WORKER_PEER_TRANSPORT_CAPABILITY, DAEMON_WORKER_RECOVERY_JOURNAL_ENV,
    DAEMON_WORKER_ROLE_ENV, DAEMON_WORKER_ROSTER_CAPABILITY, DAEMON_WORKER_SUPERVISOR_SOCKET_ENV,
    DAEMON_WORKER_TOKEN_ENV, ROSTER_HEARTBEAT_INTERVAL_MS,
};
use super::mutation_drain_latch::MutationDrainLatch;
use super::rlm_ledger::{
    create_rlm_ledger_registry_seed_source, read_legacy_rlm_subagent_registry,
    tombstone_saved_session_delete, with_passive_rlm_descendant_infos,
    LegacyRlmSubagentRegistryEntry, RlmLedgerDeleteReason, RlmLedgerEdge, RlmSpawnInput,
    RlmSpawnLedger,
};
use super::rlm_subagent_display::{
    read_rlm_subagent_display_entry, rlm_subagent_display_path, write_rlm_subagent_display_entry,
    RlmSubagentDisplayEntry, RlmSubagentModel,
};
use super::saved_session_info::serialize_saved_session_info;
use super::snapshot_transcript_cache::{
    create_snapshot_transcript_chunks, CreateSnapshotTranscriptChunksOptions,
    SNAPSHOT_TARGET_CHUNK_BYTES,
};
use super::worker_recovery_journal::{WorkerRecoveryJournal, WorkerRecoveryRecordInput};

// The server validates raw command fields in handle_line/handle_command. Keep that
// view separate from the public, typed protocol enum, including private worker commands.
#[derive(Clone, Serialize, Deserialize)]
struct ParsedDaemonCommand {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    type_: String,
    #[serde(flatten)]
    body: Value,
}

impl ParsedDaemonCommand {
    fn new(command_type: &str) -> Self {
        Self { id: None, type_: command_type.to_string(), body: Value::Object(Map::new()) }
    }

    fn from_value(value: &Value) -> Option<Self> {
        let mut body = value.as_object()?.clone();
        let type_ = body.remove("type")?.as_str()?.to_string();
        let id = body.remove("id").and_then(|value| value.as_str().map(str::to_string));
        Some(Self { id, type_, body: Value::Object(body) })
    }
}

trait DaemonResponseConstruction {
    fn success(id: Option<&str>, command: &str, data: Option<Value>) -> Self;
    fn failure(id: Option<&str>, command: &str, error: &str, error_info: Option<DaemonErrorInfo>) -> Self;
}

impl DaemonResponseConstruction for DaemonResponse {
    fn success(id: Option<&str>, command: &str, data: Option<Value>) -> Self {
        Self { id: id.map(str::to_string), type_: "response".to_string(), command: command.to_string(), success: true, data, error: None, error_info: None }
    }

    fn failure(id: Option<&str>, command: &str, error: &str, error_info: Option<DaemonErrorInfo>) -> Self {
        Self { id: id.map(str::to_string), type_: "response".to_string(), command: command.to_string(), success: false, data: None, error: Some(error.to_string()), error_info }
    }
}

// slice plumbing: `getLogger("coding-agent.daemon")`.
static STRUCTURED_LOG: std::sync::OnceLock<pi_ai::log::Logger> = std::sync::OnceLock::new();

fn structured_log() -> &'static pi_ai::log::Logger {
    STRUCTURED_LOG.get_or_init(|| pi_ai::log::get_logger("coding-agent.daemon"))
}

const WORKER_SNAPSHOT_TERMINAL_DRAIN_TIMEOUT_MS: u64 = 1_000;
const UPDATE_RESTART_PREPARE_TIMEOUT_MS: u64 = 90_000;
const MAX_SESSION_SNAPSHOT_STABILIZATION_RETRIES: u32 = 3;
pub const INITIAL_HISTORY_WINDOW_MESSAGES: usize = 400;
pub const MAX_HISTORY_RANGE_MESSAGES: usize = 400;

// slice plumbing: `DAEMON_UPDATE_RESTART_FORMAT_VERSION` from daemon-protocol.ts.
const DAEMON_UPDATE_RESTART_FORMAT_VERSION: u32 = 1;

// slice plumbing: `VERSION` from config.ts.
const VERSION: &str = env!("CARGO_PKG_VERSION");

// slice plumbing: `getDaemonUpdateRestartManifestPath` from config.ts.
fn get_daemon_update_restart_manifest_path(socket_path: &str) -> String {
    format!("{socket_path}.update-restart.json")
}

// slice plumbing: `getDaemonLogPath` from config.ts.
fn get_daemon_log_path(socket_path: &str) -> String {
    let base = socket_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("daemon")
        .replace(['.'], "-");
    format!("{base}.log")
}

// slice plumbing: `appendRotatingLog` from config.ts.
fn append_rotating_log(path: &str, message: &str) {
    use std::io::Write;
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{message}");
    }
}

// slice plumbing: `getCronJobsPath` from config.ts.
fn get_cron_jobs_path(agent_dir: &str) -> String {
    Path::new(agent_dir)
        .join("cron-jobs.json")
        .to_string_lossy()
        .to_string()
}

// slice plumbing: `getSessionsDir` from config.ts.
fn get_sessions_dir(agent_dir: &str) -> String {
    Path::new(agent_dir)
        .join("sessions")
        .to_string_lossy()
        .to_string()
}

// slice plumbing: `initTheme` from modes/interactive/theme/theme.ts.
fn init_theme_headless(theme_name: Option<&str>) {
    crate::modes::interactive::theme::theme::init_theme(theme_name, false);
}

// slice plumbing: `getDaemonRuntimeIdentity` from daemon-runtime-identity.ts.
fn get_daemon_runtime_identity() -> Value {
    serde_json::json!({
        "version": VERSION,
        "buildId": std::env::var("PRIME_AGENT_BUILD_ID").unwrap_or_else(|_| "unknown".to_string()),
        "pid": std::process::id(),
    })
}

// slice plumbing: `waitForHeadlessCompletion` from modes/headless-completion.ts.
async fn wait_for_headless_completion(_state: &ActiveSessionState) {
    // The headless-completion slice owns the real wait; the daemon command only
    // needs the completion signal it already awaits on the session.
}

// slice plumbing: `startSideQuestion` from core/side-question.ts.
async fn start_side_question(_state: &ActiveSessionState, _input: &Value) -> Result<Value, String> {
    Err("side questions are not available in this build".to_string())
}

// slice plumbing: `providerRetryPolicy` from core/provider-retry.ts, re-exported
// through the daemon session summarizer slice.
use super::daemon_session_summarizer::provider_retry_policy;

// slice plumbing: `ORPHAN_PROCESS_JOURNAL_ENV` is imported from the core slice.

/// `DaemonModeOptions.defaultSessionConfig` (`AgentSessionRuntimeConfig`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentSessionRuntimeConfig {
    pub cwd: Option<String>,
    #[serde(rename = "agentDir")]
    pub agent_dir: Option<String>,
    #[serde(
        rename = "sessionDir",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub session_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<String>,
    #[serde(
        rename = "thinking",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub thinking_level: Option<String>,
    #[serde(
        rename = "telemetryDisabled",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub telemetry_disabled: Option<bool>,
    #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none", default)]
    pub api_key: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The ensure-supervisor launch environment (daemon-mode.ts:925-934): a
/// fresh `createCliSubprocessEnv()` copy plus the agent dir, with every
/// inherited worker role/token/lease variable deleted so the spawned
/// supervisor cannot launch in worker mode or journal to a literal "1" file.
pub fn daemon_supervisor_launch_env(
    source: &crate::cli::subprocess_launch::ProcessEnv,
    agent_dir: Option<&String>,
) -> crate::cli::subprocess_launch::ProcessEnv {
    let mut env = create_cli_subprocess_env(source, None, &[]);
    if let Some(agent_dir) = agent_dir {
        env.insert("PRIME_AGENT_AGENT_DIR".to_string(), agent_dir.clone());
    }
    for key in [
        DAEMON_WORKER_ROLE_ENV,
        DAEMON_WORKER_TOKEN_ENV,
        DAEMON_WORKER_ACTIVE_SESSION_ID_ENV,
        DAEMON_WORKER_RECOVERY_JOURNAL_ENV,
        DAEMON_WORKER_SUPERVISOR_SOCKET_ENV,
        ORPHAN_PROCESS_JOURNAL_ENV,
        SESSION_LEASES_ENABLED_ENV,
        SESSION_LEASE_OWNER_ID_ENV,
    ] {
        env.shift_remove(key);
    }
    env
}

/// `mergeAgentSessionRuntimeConfig` from core/agent-session-config.ts.
pub fn merge_agent_session_runtime_config(
    base: &AgentSessionRuntimeConfig,
    override_config: Option<&AgentSessionRuntimeConfig>,
) -> AgentSessionRuntimeConfig {
    let Some(override_config) = override_config else {
        return base.clone();
    };
    let mut merged = base.clone();
    if override_config.cwd.is_some() {
        merged.cwd = override_config.cwd.clone();
    }
    if override_config.agent_dir.is_some() {
        merged.agent_dir = override_config.agent_dir.clone();
    }
    if override_config.session_dir.is_some() {
        merged.session_dir = override_config.session_dir.clone();
    }
    if override_config.provider.is_some() {
        merged.provider = override_config.provider.clone();
    }
    if override_config.model.is_some() {
        merged.model = override_config.model.clone();
    }
    if override_config.thinking_level.is_some() {
        merged.thinking_level = override_config.thinking_level.clone();
    }
    if override_config.telemetry_disabled.is_some() {
        merged.telemetry_disabled = override_config.telemetry_disabled;
    }
    if override_config.api_key.is_some() {
        merged.api_key = override_config.api_key.clone();
    }
    for (key, value) in &override_config.extra {
        merged.extra.insert(key.clone(), value.clone());
    }
    merged
}

/// `DaemonModeOptions.worker`.
#[derive(Debug, Clone, Default)]
pub struct DaemonWorkerOptions {
    pub authentication_token: String,
    pub worker_instance_id: Option<String>,
    pub restore_active_session_id: Option<String>,
}

/// `DaemonModeOptions`.
#[derive(Clone)]
pub struct DaemonModeOptions {
    pub socket_path: Option<String>,
    pub default_session_config: AgentSessionRuntimeConfig,
    pub create_runtime: CreateAgentSessionRuntimeFactory,
    pub worker: Option<DaemonWorkerOptions>,
}

/// `CreateAgentSessionRuntimeFactory` from core/agent-session-runtime.ts.
pub type CreateAgentSessionRuntimeFactory = Arc<
    dyn Fn(
            CreateAgentSessionRuntimeInput,
        ) -> BoxFuture<'static, Result<AgentSessionRuntimeHandle, String>>
        + Send
        + Sync,
>;

/// The `createAgentSessionRuntime(factory, input)` argument.
#[derive(Clone)]
pub struct CreateAgentSessionRuntimeInput {
    pub factory: Value,
    pub cwd: String,
    pub agent_dir: Option<String>,
    pub session_manager: Arc<std::sync::Mutex<SessionManager>>,
    pub session_options: SessionRuntimeOptions,
    pub session_config: Option<crate::core::agent_session_config::AgentSessionRuntimeConfig>,
    pub runtime_metadata: Option<Value>,
}

/// `sessionOptions` handed to the runtime factory.
#[derive(Clone, Default)]
pub struct SessionRuntimeOptions {
    pub subagent_options: Option<Arc<crate::core::rlm_runtime::CreateRlmSubagentRuntimeOptions>>,
    pub model: Option<Value>,
    pub rlm_heartbeat_controller: Option<Arc<dyn crate::core::cron_jobs::AgentRlmHeartbeatController>>,
    pub agent_message_controller: Option<Arc<dyn AgentSessionMessageController>>,
    pub agent_observe_controller: Option<Arc<dyn AgentObserveController>>,
}

/// `AgentSessionRuntime` as the daemon holds it.
#[derive(Clone)]
pub struct AgentSessionRuntimeHandle {
    pub session: Arc<dyn DaemonSession>,
    pub metadata: AgentSessionRuntimeMetadata,
    pub model_fallback_message: Option<String>,
    /// `runtime.metadata` (agent-session-runtime.ts).
    pub new_session: Option<
        Arc<
            dyn Fn(Option<NewSessionRuntimeOptions>) -> BoxFuture<'static, Result<Value, String>>
                + Send
                + Sync,
        >,
    >,
    pub switch_session: Option<
        Arc<
            dyn Fn(String, SessionPathOptions) -> BoxFuture<'static, Result<Value, String>>
                + Send
                + Sync,
        >,
    >,
    pub fork: Option<
        Arc<dyn Fn(String, ForkOptions) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>,
    >,
    pub import_from_jsonl: Option<
        Arc<
            dyn Fn(String, Option<String>) -> BoxFuture<'static, Result<Value, String>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for AgentSessionRuntimeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSessionRuntimeHandle")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// `RuntimeOpenGuard = () => boolean | Promise<boolean>`.
pub type RuntimeOpenGuard = Arc<dyn Fn() -> BoxFuture<'static, bool> + Send + Sync>;

/// `SupervisorGenerationClaim` (the `worker_auth` body without id/type/token).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupervisorGenerationClaim {
    #[serde(rename = "supervisorGeneration")]
    pub supervisor_generation: String,
    #[serde(rename = "supervisorPid")]
    pub supervisor_pid: i64,
    #[serde(
        rename = "supervisorProcessStartId",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub supervisor_process_start_id: Option<String>,
    #[serde(rename = "supervisorSocketPath")]
    pub supervisor_socket_path: String,
}

/// `BoundSupervisorGenerationClaim`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundSupervisorGenerationClaim {
    pub claim: SupervisorGenerationClaim,
    pub owner_fingerprint: String,
}

const PEER_GRANT_TTL_LIMIT_MS: u64 = 30_000;
// Only the supervisor registers grants, so the cap is a tripwire, never an eviction policy.
const PEER_GRANT_LIMIT: usize = 1024;

const RLM_SUBAGENT_REGISTRY_FILE: &str = "rlm-subagents.jsonl";

/// One passive child as the daemon presents it: topology (sessionFile, parent,
/// depth, name) from the spawn ledger; hydration metadata (prompt, spawnCode,
/// model, rlmMaxDepth, status, createdAt) from the per-child display file, or
/// the legacy registry for pre-ledger children without one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PassiveRlmSubagentEntry {
    pub child_id: String,
    pub session_name: String,
    pub session_dir: String,
    pub session_file: String,
    pub parent_session_id: String,
    pub parent_session_file: Option<String>,
    pub rlm_depth: Option<i64>,
    pub rlm_max_depth: Option<i64>,
    pub rlm_parent_node_id: Option<String>,
    pub prompt: Option<String>,
    pub spawn_code: Option<String>,
    pub model: Option<RlmSubagentModel>,
    pub status: String,
    pub created_at: f64,
}

/// `AgentSessionRuntime`'s session-verb surface (`runtime.newSession`,
/// `switchSession`, `fork`, `importFromJsonl`). The session slice owns the
/// concrete implementation; the daemon only names the trait so its handle type
/// stays independent of it.
pub trait DaemonRuntimeApi: Send + Sync {
    fn new_session(
        &self,
        options: Option<NewSessionRuntimeOptions>,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn switch_session(
        &self,
        session_path: &str,
        options: SessionPathOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn fork(
        &self,
        entry_id: &str,
        options: ForkOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn import_from_jsonl(
        &self,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>>;
}

/// The `DaemonUpdateRestartSession` fields `appendUpdateRestartMarker` reads.
/// The protocol slice owns the wire type; the daemon carries the projection it
/// consumes (the same shape, serialized by the manifest writer).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateRestartSessionSnapshot {
    pub active_session_id: String,
    pub should_resume: bool,
    pub was_streaming: bool,
    pub was_compacting: bool,
    pub was_bash_running: bool,
    pub had_running_rlm_children: bool,
    pub was_retrying: bool,
    pub had_accepted_prompt_in_flight: bool,
}

/// `recordRlmSubagentState`'s `input` object.
#[derive(Debug, Clone, Default)]
pub struct RlmSubagentStateInput {
    pub child_id: String,
    pub session_name: String,
    pub session_dir: String,
    pub session_file: String,
    pub rlm_depth: i64,
    pub rlm_max_depth: i64,
    pub rlm_parent_node_id: Option<String>,
    pub prompt: Option<String>,
    pub spawn_code: Option<String>,
    pub model: Option<RlmSubagentModel>,
    /// `"running" | "completed"`.
    pub status: String,
    pub created_at: Option<f64>,
}

/// `PassiveRlmRoot`: either a resident root parent state or a saved root info.
#[derive(Clone)]
pub enum PassiveRlmRoot {
    Resident(Arc<StdMutex<ActiveSessionState>>),
    Saved(SessionInfo),
}

/// `PassiveRlmSubagent`.
#[derive(Clone)]
pub struct PassiveRlmSubagent {
    pub root: PassiveRlmRoot,
    pub entry: PassiveRlmSubagentEntry,
    pub info: SessionInfo,
    pub chain: Vec<PassiveRlmSubagentEntry>,
}

/// `rlmSubagentMetadataFields(display)` source shape: a display file read gives the
/// same five optional members a legacy registry entry does, so the display entry is
/// projected onto the shared source struct before the spread runs.
fn passive_entry_from_display(display: &RlmSubagentDisplayEntry) -> PassiveRlmSubagentEntry {
    PassiveRlmSubagentEntry {
        child_id: display.child_id.clone(),
        session_name: display.session_name.clone(),
        session_dir: display.session_dir.clone(),
        session_file: display.session_file.clone(),
        rlm_max_depth: display.rlm_max_depth,
        rlm_parent_node_id: display.rlm_parent_node_id.clone(),
        prompt: display.prompt.clone(),
        spawn_code: display.spawn_code.clone(),
        model: display.model.clone(),
        status: display.status.clone(),
        created_at: display.created_at,
        ..PassiveRlmSubagentEntry::default()
    }
}

/// Spread-ready optional metadata fields shared by display files and legacy
/// registry entries (`rlmSubagentMetadataFields`).
fn rlm_subagent_metadata_fields(source: &PassiveRlmSubagentEntry) -> PassiveRlmSubagentEntry {
    PassiveRlmSubagentEntry {
        rlm_max_depth: source.rlm_max_depth,
        rlm_parent_node_id: source
            .rlm_parent_node_id
            .clone()
            .filter(|value| !value.is_empty()),
        prompt: source.prompt.clone().filter(|value| !value.is_empty()),
        spawn_code: source.spawn_code.clone().filter(|value| !value.is_empty()),
        model: source.model.clone(),
        ..PassiveRlmSubagentEntry::default()
    }
}

/// `DaemonHistoryWindow` / `DaemonHistoryRange` from daemon-protocol.ts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonHistoryRange {
    pub version: u32,
    pub generation: String,
    pub representation: String,
    #[serde(rename = "tipEntryId")]
    pub tip_entry_id: Option<String>,
    #[serde(rename = "totalMessageCount")]
    pub total_message_count: usize,
    #[serde(rename = "startIndex")]
    pub start_index: usize,
    pub messages: Vec<AgentMessage>,
    #[serde(rename = "entryIds")]
    pub entry_ids: Vec<String>,
    #[serde(rename = "hasOlder")]
    pub has_older: bool,
    pub order: String,
}

/// `DAEMON_COMMAND_TYPES` (ported verbatim from daemon-mode.ts).
pub const DAEMON_COMMAND_TYPES: [&str; 103] = [
    "ack_result",
    "list",
    "list_saved_sessions",
    "create",
    "attach",
    "detach",
    "kill",
    "rename",
    "prompt",
    "cancel_prompt_admission",
    "prompt_and_wait",
    "steer",
    "follow_up",
    "restore_next_turn",
    "restore_actions",
    "append_custom_message",
    "resume_queue",
    "send_message",
    "agent_messages_status",
    "agent_messages_pause",
    "agent_messages_resume",
    "agent_messages_clear",
    "abort",
    "start_side_question",
    "abort_side_question",
    "execute_bash",
    "execute_bash_and_wait",
    "abort_bash",
    "cancel_rlm_child",
    "delete_rlm_subagent",
    "wait_for_idle",
    "wait_for_headless_completion",
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
    "replace_acp_mcp_servers",
    "get_model_catalog",
    "get_available_models",
    "get_queue",
    "mutate_queued_message",
    "clear_queue",
    "abort_and_clear_queue",
    "acquire_session_input_pause",
    "release_session_input_pause",
    "cron_list",
    "heartbeats_list",
    "heartbeat_manage",
    "cron_add",
    "cron_cancel",
    "heartbeat_get",
    "heartbeat_set",
    "heartbeat_update",
    "set_model",
    "cycle_model",
    "set_scoped_models",
    "set_thinking_level",
    "set_service_tier",
    "cycle_thinking_level",
    "set_transport",
    "set_steering_mode",
    "set_follow_up_mode",
    "set_auto_compaction",
    "set_auto_retry",
    "compact",
    "refine",
    "abort_compaction",
    "abort_branch_summary",
    "abort_retry",
    "reload",
    "new_session",
    "switch_session",
    "fork",
    "navigate_tree",
    "import_jsonl",
    "export_html",
    "export_jsonl",
    "set_session_name",
    "get_rlm_max_depth_status",
    "set_rlm_max_depth",
    "rename_saved_session",
    "delete_saved_session",
    "get_session_context",
    "get_session_tree",
    "get_user_messages_for_forking",
    "get_last_assistant_text",
    "get_system_prompt",
    "get_tool_definition",
    "set_session_entry_label",
    // SHARED FILE EDIT (modes/daemon/daemon_mode.rs, capability-gated addition by
    // jev-ui lane; REPAIR-OVERLAP file - keep the coordinator's version at
    // integration): the optional Jev mode surface. Optional means a client only
    // sends them after the daemon advertised `jev_control`, and an old daemon
    // answers "Unknown daemon command" to a client that sends them anyway.
    "jev_get_settings",
    "jev_set_session_mode",
    "jev_get_status",
    "extension_ui_response",
    "prepare_update_restart",
    "retry_worker",
    "restart",
    "shutdown",
];

// SHARED FILE EDIT (modes/daemon/daemon_mode.rs, jev-ui lane; REPAIR-OVERLAP
// file - keep the coordinator's version at integration): the two small helpers the
// optional `jev_*` command arms use. They keep the surface minimal: one store path,
// one presence probe and one view builder, no new state on `AgentDaemon`.
impl AgentDaemon {
    /// The same store the interactive `/jev` command uses
    /// (`<agent_dir>/jev/jev-settings.json`, lane A's `JevSettingsStore`), so a
    /// client and the daemon never disagree about a session's mode.
    fn jev_settings_store(&self) -> pi_jev::config::JevSettingsStore {
        pi_jev::config::JevSettingsStore::new(std::path::Path::new(&self.agent_dir))
    }

    /// Saved-credential PRESENCE through lane A's store. It never reads the secret
    /// and never guesses a file layout; an unavailable store reports `false` and
    /// `credentialPresenceKnown: false` in the view.
    fn jev_saved_credential(&self) -> (bool, bool) {
        let store = pi_jev::credential::default_credential_store(
            std::path::Path::new(&self.agent_dir),
        );
        if !store.is_available() {
            return (false, false);
        }
        match store.exists(pi_jev::config::DEFAULT_KEY_ID) {
            Ok(present) => (present, true),
            Err(_) => (false, false),
        }
    }

    /// Settings and worker telemetry use the durable transcript UUID, never the
    /// ephemeral daemon selector (which changes when a worker is reopened).
    fn jev_session_id(&self, selector: &str) -> Result<String, String> {
        if selector.is_empty() {
            return Err("activeSessionId is required".to_string());
        }
        let state = self.get_bound_session_state(selector)?;
        Ok(self.session_of(&state).session_id())
    }

    fn publish_jev_attach_footer(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) {
        let active_session_id = state.lock().expect("active session poisoned").active_session_id.clone();
        if !client.capabilities_for_session(&active_session_id).contains("extension_ui") {
            return;
        }
        let session_id = self.session_of(state).session_id();
        // TWO independent segments: the decision mode and the independent
        // compaction state. `statusCompactText` is the optional SHORT LABELLED
        // narrow form (`Jev C On` / `Jev Cmp on`), never a bare dot; a client
        // that does not know the field ignores it. ALL FOUR texts come from ONE
        // settings snapshot, so a concurrent setting change can never make the
        // two segments on the same row disagree.
        let forms = crate::core::jev_bridge::footer_status_forms(&session_id);
        let messages = [
            ("jev", forms.decision_text, forms.decision_compact_text),
            (
                "jev-compact",
                forms.compaction_text,
                forms.compaction_compact_text,
            ),
        ];
        for (index, (status_key, status_text, status_compact)) in messages.into_iter().enumerate() {
            let message = DaemonOutbound::ExtensionUiRequest {
                active_session_id: active_session_id.clone(),
                id: format!("jev-attach-{}-{index}", self.next_id()),
                method: "setStatus".to_string(),
                payload: serde_json::json!({
                    "statusKey": status_key,
                    "statusText": status_text,
                    "statusCompactText": status_compact,
                }),
            };
            if !self.defer_snapshot_frame(client, &active_session_id, &message) {
                self.write(client, &message);
            }
        }
    }

    /// The mode view both `jev_get_settings` and `jev_get_status` return. It reads
    /// only local settings and credential PRESENCE: no network call and no secret
    /// value. Every mode is reported as the real mode it is: `mode` and `activeMode`
    /// carry the effective mode and whether it allows Active. This view applies nothing
    /// itself, so `applied` stays false; the live Active counters ride in the
    /// `pipeline` block of `jev_get_status`.
    fn jev_settings_view(&self, session_id: &str) -> Result<Value, String> {
        let store = self.jev_settings_store();
        let settings = store.load();
        let env = pi_jev::config::EnvKeyPresence::from_env();
        let (saved_present, saved_presence_known) = self.jev_saved_credential();
        let credential = pi_jev::config::resolve_credential_source_from_presence(saved_present, env);
        let resolution = settings.effective_mode_with_scope(session_id);
        Ok(serde_json::json!({
            "mode": resolution.mode.as_str(),
            "explicitSessionMode": settings.session_mode(session_id).map(|mode| mode.as_str()),
            "globalDefault": settings.global_default.map(|mode| mode.as_str()),
            "scope": resolution.scope.as_str(),
            "credentialSource": credential.as_str(),
            "credentialPresent": credential.is_configured(),
            // Store access failures mean UNKNOWN, not absent.
            "credentialPresenceKnown": saved_presence_known,
            "envConflict": env.has_conflict(),
            // A real, operative mode: this reports whether THIS session's effective
            // mode allows Active. It is not a reservation or readiness marker.
            "activeMode": resolution.mode.allows_active(),
            // Optional metadata; clients without jev_features keep their local settings view.
            "compareMode": resolution.mode.allows_compare(),
            "features": settings.effective_features(session_id),
            "compactionEnabled": settings.effective_compaction_enabled(session_id),
            // Full-jev overlay truth: presence, the overlay's fixed profile
            // constants, its persisted revision, and how many saved session
            // values it currently masks. `mode`/`activeMode`/`features`/
            // `compactionEnabled` above are already overlay-resolved by
            // config.rs, so nothing here recomputes them.
            "fullJev": serde_json::json!({
                "enabled": settings.full_jev_active(),
                "mode": pi_jev::config::FULL_JEV_MODE.as_str(),
                "features": pi_jev::config::FULL_JEV_FEATURES,
                "compactionEnabled": pi_jev::config::FULL_JEV_COMPACTION_ENABLED,
                "revision": settings.full_jev.as_ref().map_or(0, |profile| profile.revision),
                "maskedSessionCount": settings.full_jev_masked_sessions().len(),
            }),
            // This settings view applies nothing itself, in any mode.
            "applied": false,
            // No per-view counters: the live figures are in `pipeline.active`.
            "appliedDecisions": 0,
        }))
    }

    /// Applies one mode change through lane A's store. Every mode is written,
    /// including `Active`, so a request is never silently rewritten into another
    /// mode; `applied` is true for every accepted write.
    fn jev_apply_session_mode(
        &self,
        session_id: &str,
        requested: pi_jev::types::JevMode,
    ) -> Result<(bool, pi_jev::types::JevMode, bool), String> {
        let store = self.jev_settings_store();
        let mut settings = store.load();
        // The optional daemon Jev surface follows the same overlay rules as
        // the interactive bridge (ROOT-CONTRACT v1): while the global full-jev
        // overlay is active a non-Off write is REJECTED (the overlay would mask
        // it, so reporting success would be a lie), and Off is the atomic
        // emergency exit: overlay removed, this chat Off, compaction off, in
        // ONE save. Off is never capability-gated.
        if settings.full_jev_active() {
            if requested != pi_jev::types::JevMode::Off {
                return Err(
                    crate::modes::interactive::native_host::JEV_FULL_JEV_REJECTION.to_string(),
                );
            }
            settings.full_jev_remove();
            settings.set_session_mode(session_id, requested);
            settings.set_session_compaction_enabled(session_id, false);
            store.save(&settings).map_err(|error| error.log_line())?;
            // The interactive host caches settings for 250ms; a daemon-side write
            // must invalidate it so both surfaces agree immediately.
            crate::core::jev_bridge::invalidate_settings_cache();
            return Ok((true, requested, true));
        }
        settings.set_session_mode(session_id, requested);
        store.save(&settings).map_err(|error| error.log_line())?;
        // The interactive host caches settings for 250ms; a daemon-side write
        // must invalidate it so both surfaces agree immediately.
        crate::core::jev_bridge::invalidate_settings_cache();
        Ok((true, requested, false))
    }
}

/// The one message form both the interactive command and the daemon return, so a
/// user sees identical wording through either path. The stable mode name is used
/// so a client can key on `Jev mode: active` / `Jev mode: compare` without
/// matching display text.
fn jev_mode_change_message(applied: bool, mode: pi_jev::types::JevMode) -> String {
    if applied {
        return format!("Jev mode: {} (scope: this chat)", mode.as_str());
    }
    format!(
        "Jev mode not applied; mode stays {} ({}). Nothing changed.",
        mode.as_str(),
        mode.label()
    )
}

#[cfg(test)]
#[path = "daemon_jev_settings_tests.rs"]
mod jev_compatibility_tests;

const CLIENT_CATCHUP_RETRY_MS: u64 = 250;
const UPDATE_RESTART_ABORT_BASH_TIMEOUT_MS: u64 = 5000;
/// Bound for the shutdown/replaced close settle wait (audit A6): the abort is
/// delivered first, so a hung agent cannot park a daemon shutdown forever. The
/// cap matches the worker stop's non-forced graceful deadline.
const CLOSE_SETTLE_WAIT_TIMEOUT: Duration = Duration::from_millis(10_000);
const SUPERVISOR_FENCE_POLL_MS: u64 = 250;
const UPDATE_RESTART_MARKER: &str = "<prime_agent_update_interrupted>\nPrime Agent was updated and intentionally interrupted this session. Continue from the saved transcript and restored tool/kernel state. Any running model, tool, bash, or child-agent work may have been stopped.\n</prime_agent_update_interrupted>";

const RECOVERY_CHECKPOINT_EVENTS: [&str; 16] = [
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "compaction_start",
    "compaction_end",
    "auto_retry_start",
    "auto_retry_end",
    "bash_start",
    "bash_end",
    "session_action_update",
    "rlm_child_update",
];

// slice plumbing: `UPDATE_RESTART_DRAIN_COMMANDS` from daemon-protocol.ts.
const UPDATE_RESTART_DRAIN_COMMANDS: [&str; 8] = [
    "abort",
    "abort_bash",
    "abort_compaction",
    "abort_branch_summary",
    "abort_retry",
    "cancel_prompt_admission",
    "cancel_rlm_child",
    "shutdown",
];

/// `delay(ms)`.
async fn delay(ms: u64) {
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

fn contains(list: &[&str], value: &str) -> bool {
    list.iter().any(|entry| *entry == value)
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn basename(path: &str, suffix: Option<&str>) -> String {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
    match suffix {
        Some(suffix) => name
            .strip_suffix(suffix)
            .map(str::to_string)
            .unwrap_or(name),
        None => name,
    }
}

fn dirname(path: &str) -> String {
    match path.rfind(['/', '\\']) {
        Some(index) if index > 0 => path[..index].to_string(),
        Some(_) => "/".to_string(),
        None => ".".to_string(),
    }
}

fn join_path(base: &str, name: &str) -> String {
    Path::new(base).join(name).to_string_lossy().to_string()
}

fn resolve_path(path: &str) -> String {
    crate::utils::daemon_socket_path::normalize_socket_path(path, None)
}

/// `sessionHistoryRepresentation(model)`.
pub fn session_history_representation(model: Option<&pi_ai::types::Model>) -> String {
    let identity = match model {
        Some(model) => serde_json::json!([
            model.provider,
            model.id,
            model.api,
            model.base_url.trim_end_matches('/')
        ]),
        None => serde_json::json!([Value::Null, Value::Null, Value::Null, Value::Null]),
    };
    let digest = Sha256::digest(identity.to_string().as_bytes());
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    encoded.chars().take(22).collect()
}

/// `Model<Api>` as the daemon seam names it. Canonical owner is
/// `pi_ai::types::Model` (the TypeScript `Model<Api>`); this is a re-export so
/// the daemon adapter keeps naming the seam's model type.
pub type ModelIdentity = pi_ai::types::Model;

/// `slicePinnedSessionHistory(history, options)`.
pub fn slice_pinned_session_history(
    history: &SessionHistorySnapshot,
    options: &SlicePinnedSessionHistoryOptions,
) -> Result<DaemonHistoryRange, String> {
    if history.messages.len() != history.entry_ids.len() {
        return Err("Session history messages and entry ids are not aligned".to_string());
    }
    let mut end_index = history.messages.len();
    if let Some(before_entry_id) = &options.before_entry_id {
        match history
            .entry_ids
            .iter()
            .position(|entry| entry == before_entry_id)
        {
            Some(index) => end_index = index,
            None => {
                return Err(format!(
                    "Session history boundary no longer exists: {before_entry_id}"
                ));
            }
        }
    }
    let requested_limit = options.limit.unwrap_or(INITIAL_HISTORY_WINDOW_MESSAGES);
    if requested_limit == 0 || requested_limit > i32::MAX as usize {
        return Err("Session history range limit must be a positive integer".to_string());
    }
    let limit = requested_limit.min(MAX_HISTORY_RANGE_MESSAGES);
    let start_index = end_index.saturating_sub(limit);
    Ok(DaemonHistoryRange {
        version: 1,
        generation: options.generation.clone(),
        representation: options.representation.clone(),
        tip_entry_id: history.tip_entry_id.clone(),
        total_message_count: history.messages.len(),
        start_index,
        messages: history.messages[start_index..end_index].to_vec(),
        entry_ids: history.entry_ids[start_index..end_index].to_vec(),
        has_older: start_index > 0,
        order: "chronological".to_string(),
    })
}

/// `slicePinnedSessionHistory` options.
#[derive(Debug, Clone, Default)]
pub struct SlicePinnedSessionHistoryOptions {
    pub generation: String,
    pub representation: String,
    pub before_entry_id: Option<String>,
    pub limit: Option<usize>,
}

/// One outbound frame. The full `DaemonOutbound` union lives in
/// daemon-protocol.ts (another slice); the daemon carries the members its own
/// logic inspects plus `Raw` for pass-through frames whose payload another
/// module builds. Wire shape is unchanged: serde writes `type` plus the
/// camelCase fields.
#[derive(Debug, Clone, PartialEq)]
pub enum DaemonOutbound {
    SessionEvent {
        active_session_id: String,
        event: Value,
    },
    SessionStatus {
        active_session_id: String,
        recap: Option<String>,
    },
    SessionReplaced {
        active_session_id: String,
        state: Value,
        messages: Vec<Value>,
    },
    SessionResynced {
        active_session_id: String,
        state: Value,
        messages: Vec<Value>,
        reason: Option<String>,
    },
    SessionClosed {
        active_session_id: String,
        reason: String,
    },
    ExtensionUiRequest {
        active_session_id: String,
        id: String,
        method: String,
        payload: Value,
    },
    ExtensionError {
        active_session_id: String,
        extension_path: Option<String>,
        event: Option<String>,
        error: String,
    },
    SessionStart {
        active_session_id: String,
    },
    SessionAttached {
        active_session_id: String,
        client_id: String,
    },
    SessionDetached {
        active_session_id: String,
        client_id: String,
    },
    SessionSnapshotBegin {
        active_session_id: String,
        snapshot_id: String,
        message_count: usize,
        target_chunk_bytes: usize,
        transfer_id: Option<String>,
    },
    SessionSnapshotChunk {
        active_session_id: String,
        snapshot_id: String,
        index: usize,
        messages: Vec<Value>,
    },
    SessionSnapshotEnd {
        active_session_id: String,
        snapshot_id: String,
        chunk_count: usize,
        bytes: usize,
        last_event_sequence: Option<u64>,
        last_event_cursor: Option<Value>,
        transfer_id: Option<String>,
    },
    SessionSnapshotFailed {
        active_session_id: String,
        snapshot_id: String,
        error: String,
    },
    HeartbeatsChanged,
    RosterDelta {
        payload: Value,
    },
    RosterHeartbeat,
    DaemonClosing {
        reason: String,
    },
    /// A frame built elsewhere (roster frames, list progress, worker snapshots).
    Raw(Value),
}

impl DaemonOutbound {
    /// The wire `type` discriminator.
    pub fn type_name(&self) -> &str {
        match self {
            DaemonOutbound::SessionEvent { .. } => "session_event",
            DaemonOutbound::SessionStatus { .. } => "session_status",
            DaemonOutbound::SessionReplaced { .. } => "session_replaced",
            DaemonOutbound::SessionResynced { .. } => "session_resynced",
            DaemonOutbound::SessionClosed { .. } => "session_closed",
            DaemonOutbound::ExtensionUiRequest { .. } => "extension_ui_request",
            DaemonOutbound::ExtensionError { .. } => "extension_error",
            DaemonOutbound::SessionStart { .. } => "session_start",
            DaemonOutbound::SessionAttached { .. } => "session_attached",
            DaemonOutbound::SessionDetached { .. } => "session_detached",
            DaemonOutbound::SessionSnapshotBegin { .. } => "session_snapshot_begin",
            DaemonOutbound::SessionSnapshotChunk { .. } => "session_snapshot_chunk",
            DaemonOutbound::SessionSnapshotEnd { .. } => "session_snapshot_end",
            DaemonOutbound::SessionSnapshotFailed { .. } => "session_snapshot_failed",
            DaemonOutbound::HeartbeatsChanged => "heartbeats_changed",
            DaemonOutbound::RosterDelta { .. } => "roster_delta",
            DaemonOutbound::RosterHeartbeat => "roster_heartbeat",
            DaemonOutbound::DaemonClosing { .. } => "daemon_closing",
            DaemonOutbound::Raw(value) => value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        }
    }

    fn with_type(type_: &str) -> Map<String, Value> {
        let mut object = Map::new();
        object.insert("type".to_string(), Value::String(type_.to_string()));
        object
    }

    /// Serialize to the wire object.
    pub fn to_value(&self) -> Value {
        match self {
            DaemonOutbound::SessionEvent {
                active_session_id,
                event,
            } => {
                let mut object = Self::with_type("session_event");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("event".to_string(), event.clone());
                Value::Object(object)
            }
            DaemonOutbound::SessionStatus {
                active_session_id,
                recap,
            } => {
                let mut object = Self::with_type("session_status");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                match recap {
                    Some(recap) => object.insert("recap".to_string(), Value::String(recap.clone())),
                    None => object.insert("recap".to_string(), Value::Null),
                };
                Value::Object(object)
            }
            DaemonOutbound::SessionReplaced {
                active_session_id,
                state,
                messages,
            } => {
                let mut object = Self::with_type("session_replaced");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("state".to_string(), state.clone());
                object.insert("messages".to_string(), Value::Array(messages.clone()));
                Value::Object(object)
            }
            DaemonOutbound::SessionResynced {
                active_session_id,
                state,
                messages,
                reason,
            } => {
                let mut object = Self::with_type("session_resynced");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("state".to_string(), state.clone());
                object.insert("messages".to_string(), Value::Array(messages.clone()));
                if let Some(reason) = reason {
                    object.insert("reason".to_string(), Value::String(reason.clone()));
                }
                Value::Object(object)
            }
            DaemonOutbound::SessionClosed {
                active_session_id,
                reason,
            } => {
                let mut object = Self::with_type("session_closed");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("reason".to_string(), Value::String(reason.clone()));
                Value::Object(object)
            }
            DaemonOutbound::ExtensionUiRequest {
                active_session_id,
                id,
                method,
                payload,
            } => {
                let mut object = Self::with_type("extension_ui_request");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("id".to_string(), Value::String(id.clone()));
                object.insert("method".to_string(), Value::String(method.clone()));
                object.insert("payload".to_string(), payload.clone());
                Value::Object(object)
            }
            DaemonOutbound::ExtensionError {
                active_session_id,
                extension_path,
                event,
                error,
            } => {
                let mut object = Self::with_type("extension_error");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                if let Some(extension_path) = extension_path {
                    object.insert(
                        "extensionPath".to_string(),
                        Value::String(extension_path.clone()),
                    );
                }
                if let Some(event) = event {
                    object.insert("event".to_string(), Value::String(event.clone()));
                }
                object.insert("error".to_string(), Value::String(error.clone()));
                Value::Object(object)
            }
            DaemonOutbound::SessionStart { active_session_id } => {
                let mut object = Self::with_type("session_start");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                Value::Object(object)
            }
            DaemonOutbound::SessionAttached {
                active_session_id,
                client_id,
            } => {
                let mut object = Self::with_type("session_attached");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("clientId".to_string(), Value::String(client_id.clone()));
                Value::Object(object)
            }
            DaemonOutbound::SessionDetached {
                active_session_id,
                client_id,
            } => {
                let mut object = Self::with_type("session_detached");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("clientId".to_string(), Value::String(client_id.clone()));
                Value::Object(object)
            }
            DaemonOutbound::SessionSnapshotBegin {
                active_session_id,
                snapshot_id,
                message_count,
                target_chunk_bytes,
                transfer_id,
            } => {
                let mut object = Self::with_type("session_snapshot_begin");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("snapshotId".to_string(), Value::String(snapshot_id.clone()));
                object.insert(
                    "messageCount".to_string(),
                    Value::from(*message_count as f64),
                );
                object.insert(
                    "targetChunkBytes".to_string(),
                    Value::from(*target_chunk_bytes as f64),
                );
                if let Some(transfer_id) = transfer_id {
                    object.insert("transferId".to_string(), Value::String(transfer_id.clone()));
                }
                Value::Object(object)
            }
            DaemonOutbound::SessionSnapshotChunk {
                active_session_id,
                snapshot_id,
                index,
                messages,
            } => {
                let mut object = Self::with_type("session_snapshot_chunk");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("snapshotId".to_string(), Value::String(snapshot_id.clone()));
                object.insert("index".to_string(), Value::from(*index as f64));
                object.insert("messages".to_string(), Value::Array(messages.clone()));
                Value::Object(object)
            }
            DaemonOutbound::SessionSnapshotEnd {
                active_session_id,
                snapshot_id,
                chunk_count,
                bytes,
                last_event_sequence,
                last_event_cursor,
                transfer_id,
            } => {
                let mut object = Self::with_type("session_snapshot_end");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("snapshotId".to_string(), Value::String(snapshot_id.clone()));
                object.insert("chunkCount".to_string(), Value::from(*chunk_count as f64));
                object.insert("bytes".to_string(), Value::from(*bytes as f64));
                if let Some(sequence) = last_event_sequence {
                    object.insert(
                        "lastEventSequence".to_string(),
                        Value::from(*sequence as f64),
                    );
                }
                if let Some(cursor) = last_event_cursor {
                    object.insert("lastEventCursor".to_string(), cursor.clone());
                }
                if let Some(transfer_id) = transfer_id {
                    object.insert("transferId".to_string(), Value::String(transfer_id.clone()));
                }
                Value::Object(object)
            }
            DaemonOutbound::SessionSnapshotFailed {
                active_session_id,
                snapshot_id,
                error,
            } => {
                let mut object = Self::with_type("session_snapshot_failed");
                object.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.clone()),
                );
                object.insert("snapshotId".to_string(), Value::String(snapshot_id.clone()));
                object.insert("error".to_string(), Value::String(error.clone()));
                Value::Object(object)
            }
            DaemonOutbound::HeartbeatsChanged => {
                Value::Object(Self::with_type("heartbeats_changed"))
            }
            DaemonOutbound::RosterDelta { payload } => payload.clone(),
            DaemonOutbound::RosterHeartbeat => Value::Object(Self::with_type("roster_heartbeat")),
            DaemonOutbound::DaemonClosing { reason } => {
                let mut object = Self::with_type("daemon_closing");
                object.insert("reason".to_string(), Value::String(reason.clone()));
                Value::Object(object)
            }
            DaemonOutbound::Raw(value) => value.clone(),
        }
    }

    /// `hasDaemonOutboundActiveSessionId`.
    pub fn active_session_id(&self) -> Option<&str> {
        match self {
            DaemonOutbound::SessionEvent {
                active_session_id, ..
            }
            | DaemonOutbound::SessionStatus {
                active_session_id, ..
            }
            | DaemonOutbound::SessionReplaced {
                active_session_id, ..
            }
            | DaemonOutbound::SessionResynced {
                active_session_id, ..
            }
            | DaemonOutbound::SessionClosed {
                active_session_id, ..
            }
            | DaemonOutbound::ExtensionUiRequest {
                active_session_id, ..
            }
            | DaemonOutbound::ExtensionError {
                active_session_id, ..
            }
            | DaemonOutbound::SessionStart { active_session_id }
            | DaemonOutbound::SessionAttached {
                active_session_id, ..
            }
            | DaemonOutbound::SessionDetached {
                active_session_id, ..
            }
            | DaemonOutbound::SessionSnapshotBegin {
                active_session_id, ..
            }
            | DaemonOutbound::SessionSnapshotChunk {
                active_session_id, ..
            }
            | DaemonOutbound::SessionSnapshotEnd {
                active_session_id, ..
            }
            | DaemonOutbound::SessionSnapshotFailed {
                active_session_id, ..
            } => Some(active_session_id),
            DaemonOutbound::HeartbeatsChanged
            | DaemonOutbound::RosterDelta { .. }
            | DaemonOutbound::RosterHeartbeat
            | DaemonOutbound::DaemonClosing { .. } => None,
            DaemonOutbound::Raw(value) => value.get("activeSessionId").and_then(Value::as_str),
        }
    }
}

/// The `DaemonSessionSnapshot` fields `snapshotTransferId` reads.
pub fn snapshot_transfer_id(snapshot: &Value) -> String {
    let cursor = snapshot
        .get("lastEventCursor")
        .cloned()
        .unwrap_or(Value::Null);
    let history_flavor = match snapshot.get("history") {
        Some(history) if !history.is_null() => format!(
            "history-{}-{}-{}",
            history
                .get("representation")
                .and_then(Value::as_str)
                .unwrap_or(""),
            history
                .get("tipEntryId")
                .and_then(Value::as_str)
                .unwrap_or("empty"),
            history
                .get("startIndex")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        ),
        _ => "full".to_string(),
    };
    format!(
        "{}-{}-{}-{}",
        snapshot
            .get("activeSessionId")
            .and_then(Value::as_str)
            .unwrap_or(""),
        cursor
            .get("generation")
            .and_then(Value::as_str)
            .unwrap_or(""),
        cursor
            .get("sequence")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        history_flavor
    )
}

/// `isSequencedSessionOutbound`.
pub fn is_sequenced_session_outbound(type_: &str) -> bool {
    matches!(
        type_,
        "session_event"
            | "session_status"
            | "session_replaced"
            | "session_resynced"
            | "session_closed"
            | "extension_ui_request"
            | "extension_error"
    )
}

/// `RuntimeOpenCancelledError` (a private daemon control-flow signal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOpenCancelledError;

impl std::fmt::Display for RuntimeOpenCancelledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Runtime open cancelled")
    }
}

impl std::error::Error for RuntimeOpenCancelledError {}

/// `BoundSessionUnavailableError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundSessionUnavailableError {
    pub message: String,
}

impl BoundSessionUnavailableError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BoundSessionUnavailableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BoundSessionUnavailableError {}

/// `DaemonSessionClosedReason` from daemon-protocol.ts.
pub type DaemonSessionClosedReason = String;
/// `DaemonClosingReason` from daemon-protocol.ts.
pub type DaemonClosingReason = String;

const CLOSING_REASON_KILLED: &str = "killed";
const CLOSING_REASON_SHUTDOWN: &str = "shutdown";
const CLOSING_REASON_COMPLETED: &str = "completed";
const CLOSING_REASON_REPLACED: &str = "replaced";
const CLOSING_REASON_UPDATE: &str = "update";

/// Writes serialized lines to one client socket. `end()` mirrors `socket.end()`.
pub struct DaemonClientWriter {
    sender: mpsc::UnboundedSender<Vec<u8>>,
    closed: AtomicBool,
    ended: tokio_util::sync::CancellationToken,
    drained: tokio_util::sync::CancellationToken,
}

impl DaemonClientWriter {
    pub fn new(sender: mpsc::UnboundedSender<Vec<u8>>) -> Self {
        Self {
            sender,
            closed: AtomicBool::new(false),
            ended: tokio_util::sync::CancellationToken::new(),
            drained: tokio_util::sync::CancellationToken::new(),
        }
    }

    pub fn write(&self, line: String) -> bool {
        self.write_bytes(line.into_bytes())
    }

    fn write_bytes(&self, bytes: Vec<u8>) -> bool {
        if self.closed.load(Ordering::SeqCst) {
            return false;
        }
        self.sender.send(bytes).is_ok()
    }

    /// `socket.end()`.
    pub fn end(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.ended.cancel();
    }

    /// `socket.destroyed`.
    pub fn destroyed(&self) -> bool {
        self.closed.load(Ordering::SeqCst) || self.sender.is_closed()
    }
}

/// One attached socket client: the ported `DaemonSocketClient` state
/// (active-session-state.ts) plus the daemon-mode.ts fields that struct does not
/// carry yet (the socket, the in-flight catch-up, the snapshot transfer
/// controllers). Private plumbing: `active_session_state.rs` belongs to another
/// slice.
pub struct DaemonClientHandle {
    pub state: Arc<StdMutex<DaemonSocketClient>>,
    pub writer: Arc<DaemonClientWriter>,
    /// `client.catchupPromise` is running.
    pub catchup_running: AtomicBool,
    /// `client.catchupRetryTimer` is scheduled.
    pub catchup_retry_timer: AtomicBool,
    /// `client.snapshotTransferAbortControllers`.
    pub snapshot_transfer_abort_controllers:
        StdMutex<HashMap<String, tokio_util::sync::CancellationToken>>,
    /// `client.snapshotTransferTails`.
    pub snapshot_transfer_tails: StdMutex<HashMap<String, u64>>,
    deferred_snapshot_frames: StdMutex<HashMap<String, (usize, Vec<DaemonOutbound>)>>,
    /// `client.detachInput()`.
    pub detach_input: Arc<dyn Fn() + Send + Sync>,
    /// The private-frame decoder state for `transport === "private-framed"`.
    pub frame_decoder: Arc<StdMutex<PrivateFrameDecoder>>,
}

impl DaemonClientHandle {
    pub fn new(
        id: impl Into<String>,
        writer: Arc<DaemonClientWriter>,
        detach_input: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let state = Arc::new(StdMutex::new(DaemonSocketClient::new(id, false)));
        Self {
            state,
            writer,
            catchup_running: AtomicBool::new(false),
            catchup_retry_timer: AtomicBool::new(false),
            snapshot_transfer_abort_controllers: StdMutex::new(HashMap::new()),
            snapshot_transfer_tails: StdMutex::new(HashMap::new()),
            deferred_snapshot_frames: StdMutex::new(HashMap::new()),
            detach_input,
            frame_decoder: Arc::new(StdMutex::new(PrivateFrameDecoder::new())),
        }
    }

    pub fn id(&self) -> String {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .id
            .clone()
    }

    pub fn set_id(&self, id: &str) {
        self.state.lock().expect("daemon client poisoned").id = id.to_string();
    }

    pub fn authenticated(&self) -> bool {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .authenticated
            == Some(true)
    }

    pub fn set_authenticated(&self, role: &str) {
        let mut state = self.state.lock().expect("daemon client poisoned");
        state.authenticated = Some(true);
        state.authentication_role = Some(role.to_string());
    }

    /// The session-local capabilities of one attach (`daemonClientCapabilitiesForSession`).
    pub fn capabilities_for_session(&self, active_session_id: &str) -> HashSet<String> {
        let state = self.state.lock().expect("daemon client poisoned");
        state
            .capabilities_by_active_session_id
            .as_ref()
            .and_then(|map| map.get(active_session_id).cloned())
            .unwrap_or_else(|| state.capabilities.clone())
    }

    pub fn supports_extension_ui_for_session(&self, active_session_id: &str) -> bool {
        let state = self.state.lock().expect("daemon client poisoned");
        state
            .capabilities_by_active_session_id
            .as_ref()
            .and_then(|map| {
                map.get(active_session_id)
                    .map(|value| value.contains("extension_ui"))
            })
            .unwrap_or(state.supports_extension_ui)
    }

    pub fn catchup_retry_timer(&self) -> bool {
        self.catchup_retry_timer.load(Ordering::SeqCst)
    }

    pub fn set_catchup_retry_timer(&self, value: bool) {
        self.catchup_retry_timer.store(value, Ordering::SeqCst);
    }

    pub fn snapshot_streaming(&self) -> bool {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .snapshot_streaming
            == Some(true)
    }

    pub fn set_snapshot_streaming(&self, value: bool) {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .snapshot_streaming = Some(value);
    }

    pub fn is_backpressured(&self) -> bool {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .backpressured
            == Some(true)
    }

    pub fn set_backpressured(&self, value: bool) {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .backpressured = Some(value);
    }

    pub fn attached_active_session_ids(&self) -> Vec<String> {
        self.state
            .lock()
            .expect("daemon client poisoned")
            .attached_active_session_ids
            .iter()
            .cloned()
            .collect()
    }
}

impl std::fmt::Debug for DaemonClientHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonClientHandle")
            .field("id", &self.id())
            .finish()
    }
}

/// The wire name of a client capability, as `daemon-protocol.ts` writes it.
fn daemon_client_capability_name(capability: &DaemonClientCapability) -> String {
    match serde_json::to_value(capability) {
        Ok(Value::String(name)) => name,
        _ => String::new(),
    }
}

/// `DAEMON_CLIENT_CAPABILITY_SET.has(capability)`.
fn is_supported_client_capability(capability: &str) -> bool {
    DAEMON_SUPPORTED_CLIENT_CAPABILITIES
        .iter()
        .any(|supported| daemon_client_capability_name(supported) == capability)
}

/// `normalizeClientCapabilities`.
fn normalize_client_capabilities(
    capabilities: Option<&HashSet<String>>,
    supports_extension_ui: Option<bool>,
) -> HashSet<String> {
    let mut normalized: HashSet<String> = HashSet::new();
    let default_capabilities: HashSet<String> = DAEMON_DEFAULT_CLIENT_CAPABILITIES
        .iter()
        .map(daemon_client_capability_name)
        .collect();
    for capability in capabilities.unwrap_or(&default_capabilities) {
        if is_supported_client_capability(capability) {
            normalized.insert(capability.clone());
        }
    }
    if supports_extension_ui == Some(true) {
        normalized.insert("extension_ui".to_string());
    }
    normalized
}

/// `setDaemonClientSessionCapabilities`.
pub fn set_daemon_client_session_capabilities(
    client: &DaemonClientHandle,
    active_session_id: &str,
    capabilities: HashSet<String>,
) {
    let mut state = client.state.lock().expect("daemon client poisoned");
    let map = state
        .capabilities_by_active_session_id
        .get_or_insert_with(HashMap::new);
    map.insert(active_session_id.to_string(), capabilities);
    state.supports_extension_ui = map.values().any(|value| value.contains("extension_ui"));
}

/// `removeDaemonClientSessionCapabilities`.
fn remove_daemon_client_session_capabilities(client: &DaemonClientHandle, active_session_id: &str) {
    let mut state = client.state.lock().expect("daemon client poisoned");
    if let Some(map) = state.capabilities_by_active_session_id.as_mut() {
        map.remove(active_session_id);
    }
    let supports = state
        .capabilities_by_active_session_id
        .as_ref()
        .map(|map| map.values().any(|value| value.contains("extension_ui")))
        .unwrap_or(false);
    state.supports_extension_ui = supports;
}

/// `cancelPendingExtensionUiRequests`.
pub fn cancel_pending_extension_ui_requests(state: &mut ActiveSessionState) {
    let pending: Vec<ActiveSessionExtensionUiRequest> = state
        .extension_ui_requests
        .drain()
        .map(|(_, value)| value)
        .collect();
    for request in pending {
        (request.resolve)(DaemonExtensionUIResponse::Cancelled);
    }
}

/// `detachClientFromActiveSession`.
pub fn detach_client_from_active_session(
    client: &DaemonClientHandle,
    state: &mut ActiveSessionState,
) {
    state
        .clients
        .retain(|candidate| !Arc::ptr_eq(candidate, &client.state));
    client
        .state
        .lock()
        .expect("daemon client poisoned")
        .attached_active_session_ids
        .remove(&state.active_session_id);
    remove_daemon_client_session_capabilities(client, &state.active_session_id);
    if state.clients.is_empty() {
        cancel_pending_extension_ui_requests(state);
    }
}

/// `markClientSnapshotStreaming`; returns the abort token for the transfer.
pub fn mark_client_snapshot_streaming(
    client: &DaemonClientHandle,
    active_session_id: &str,
) -> tokio_util::sync::CancellationToken {
    client.set_snapshot_streaming(true);
    {
        let mut state = client.state.lock().expect("daemon client poisoned");
        state
            .snapshot_active_session_ids
            .get_or_insert_with(HashSet::new)
            .insert(active_session_id.to_string());
        let counts = state
            .snapshot_active_session_counts
            .get_or_insert_with(HashMap::new);
        let entry = counts.entry(active_session_id.to_string()).or_insert(0);
        *entry += 1;
    }
    let existing = client
        .snapshot_transfer_abort_controllers
        .lock()
        .expect("snapshot abort controllers poisoned")
        .get(active_session_id)
        .cloned();
    if let Some(existing) = existing {
        if !existing.is_cancelled() {
            return existing;
        }
    }
    let controller = tokio_util::sync::CancellationToken::new();
    client
        .snapshot_transfer_abort_controllers
        .lock()
        .expect("snapshot abort controllers poisoned")
        .insert(active_session_id.to_string(), controller.clone());
    controller
}

/// `abortClientSnapshotStreaming`.
pub fn abort_client_snapshot_streaming(
    client: &DaemonClientHandle,
    active_session_id: Option<&str>,
) {
    match active_session_id {
        Some(active_session_id) => {
            client
                .snapshot_transfer_abort_controllers
                .lock()
                .expect("snapshot abort controllers poisoned")
                .get(active_session_id)
                .cloned()
                .iter()
                .for_each(tokio_util::sync::CancellationToken::cancel);
        }
        None => {
            client
                .snapshot_transfer_abort_controllers
                .lock()
                .expect("snapshot abort controllers poisoned")
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .iter()
                .for_each(tokio_util::sync::CancellationToken::cancel);
        }
    }
}

/// `finishClientSnapshotStreaming`.
pub fn finish_client_snapshot_streaming(client: &DaemonClientHandle, active_session_id: &str) {
    let count = {
        let state = client.state.lock().expect("daemon client poisoned");
        state
            .snapshot_active_session_counts
            .as_ref()
            .and_then(|map| map.get(active_session_id).copied())
            .unwrap_or(1)
    };
    let mut state = client.state.lock().expect("daemon client poisoned");
    if count > 1 {
        if let Some(map) = state.snapshot_active_session_counts.as_mut() {
            map.insert(active_session_id.to_string(), count - 1);
        }
    } else {
        if let Some(map) = state.snapshot_active_session_counts.as_mut() {
            map.remove(active_session_id);
        }
        if let Some(ids) = state.snapshot_active_session_ids.as_mut() {
            ids.remove(active_session_id);
        }
        drop(state);
        client
            .snapshot_transfer_abort_controllers
            .lock()
            .expect("snapshot abort controllers poisoned")
            .remove(active_session_id);
        client
            .snapshot_transfer_tails
            .lock()
            .expect("snapshot transfer tails poisoned")
            .remove(active_session_id);
        state = client.state.lock().expect("daemon client poisoned");
    }
    let streaming = state
        .snapshot_active_session_ids
        .as_ref()
        .map(|ids| !ids.is_empty())
        .unwrap_or(false);
    state.snapshot_streaming = Some(streaming);
    if !streaming {
        state.backpressured = Some(false);
    }
}

/// `shouldSendDaemonOutboundToClient`.
pub fn should_send_daemon_outbound_to_client(
    client: &DaemonClientHandle,
    message: &DaemonOutbound,
) -> bool {
    match message {
        DaemonOutbound::ExtensionUiRequest {
            active_session_id,
            method,
            ..
        } => {
            !is_daemon_dialog_extension_ui_request(method)
                || client.supports_extension_ui_for_session(active_session_id)
        }
        DaemonOutbound::Raw(value) if value.get("type").and_then(Value::as_str) == Some("extension_ui_request") => {
            let method = value.get("method").and_then(Value::as_str).unwrap_or("");
            !is_daemon_dialog_extension_ui_request(method) || client.supports_extension_ui_for_session(value.get("activeSessionId").and_then(Value::as_str).unwrap_or(""))
        }
        _ => true,
    }
}

/// `getChildActiveSessionStates`.
pub fn get_child_active_session_states(
    sessions: &HashMap<String, Arc<StdMutex<ActiveSessionState>>>,
    parent_state: &Arc<StdMutex<ActiveSessionState>>,
) -> Vec<Arc<StdMutex<ActiveSessionState>>> {
    let parent_active_session_id = parent_state
        .lock()
        .expect("active session poisoned")
        .active_session_id
        .clone();
    sessions
        .values()
        .filter(|state| {
            let state = state.lock().expect("active session poisoned");
            if state.active_session_id == parent_active_session_id {
                return false;
            }
            state
                .runtime
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.parent_active_session_id.as_deref())
                == Some(parent_active_session_id.as_str())
        })
        .cloned()
        .collect()
}

/// `resolveDaemonSessionPath`.
pub async fn resolve_daemon_session_path(
    selector: &str,
    cwd: &str,
    session_dir: Option<&str>,
) -> Result<String, String> {
    let resolved = resolve_session_path(selector, cwd, session_dir)
        .await
        .map_err(|error| error.to_string())?;
    Ok(resolved_path(&resolved))
}

/// The `path` field shared by every `ResolvedSession` variant.
fn resolved_path(resolved: &ResolvedSession) -> String {
    match resolved {
        ResolvedSession::Path { path }
        | ResolvedSession::Local { path }
        | ResolvedSession::Global { path, .. } => path.clone(),
    }
}

/// `WorkerRosterReporterState`.
#[derive(Debug, Default)]
pub struct WorkerRosterReporterState {
    pub last_composed: HashMap<String, WorkerRosterEntry>,
    pub last_composed_json: HashMap<String, String>,
    pub queued_children: HashMap<String, WorkerRosterEntry>,
    /// Pending removals: agentId -> removed sessionId; a new incarnation of the id cancels it.
    pub removed_agent_ids: HashMap<String, Option<String>>,
    pub snapshot_pending: bool,
}

const ROSTER_SESSION_EVENT_TRIGGERS: [&str; 16] = [
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "bash_start",
    "bash_end",
    "compaction_start",
    "compaction_end",
    "auto_retry_start",
    "auto_retry_end",
    "tool_execution_start",
    "tool_execution_end",
    "message_end",
    "session_action_update",
    "session_info_changed",
    "thinking_level_changed",
];

/// `runDaemonMode(options)`.
pub async fn run_daemon_mode(options: DaemonModeOptions) -> Result<(), String> {
    let socket_path = normalize_socket_path(
        &options
            .socket_path
            .clone()
            .unwrap_or_else(default_daemon_socket_path),
        None,
    );
    let daemon = AgentDaemon::new(socket_path, options);
    daemon.start().await?;
    daemon.shutdown_complete.cancelled().await;
    Ok(())
}

/// `PromptOptions` as the daemon command fills it in.
#[derive(Clone, Default)]
pub struct PromptInvocation {
    pub message: String,
    pub content: Option<Value>,
    pub images: Option<Value>,
    pub streaming_behavior: Option<String>,
    pub queue_if_busy: Option<bool>,
    pub resume_if_idle: Option<bool>,
    pub expand_prompt_templates: Option<bool>,
    pub skip_input_handlers: Option<bool>,
    pub source: Option<String>,
    pub agent_message_id: Option<String>,
    pub custom_message: Option<Value>,
    pub queue_key: Option<String>,
    pub prefix_messages: Option<Value>,
    pub signal: Option<tokio_util::sync::CancellationToken>,
    pub admission_committed: Option<Arc<dyn Fn() + Send + Sync>>,
    pub preflight_result: Option<Arc<dyn Fn(bool, bool) + Send + Sync>>,
}

/// `session.acquireSessionInputPause()`: the returned handle releases the pause.
pub type SessionInputPause = Arc<dyn Fn() + Send + Sync>;

/// `SessionPassivationSnapshot` fields the daemon composes (another slice owns
/// the rest of the struct).
#[derive(Debug, Clone, Default)]
pub struct DaemonSessionPassivationFields {
    pub has_registered_cron_job: bool,
    pub last_activity_at: f64,
    pub has_non_passive_descendants: bool,
    pub has_in_flight_work: bool,
}

/// One `updateRestart` transaction.
pub struct UpdateRestartTransaction {
    pub id: u64,
    pub owner: Option<u64>,
    pub abort: tokio_util::sync::CancellationToken,
    pub deadline_expired: Arc<AtomicBool>,
    pub phase: String,
    pub manifest: Option<Value>,
    pub deferred_client_env: Vec<DeferredClientEnv>,
}

/// `deferredClientEnv[i]`.
pub struct DeferredClientEnv {
    pub client: Arc<DaemonClientHandle>,
    pub state: Arc<StdMutex<ActiveSessionState>>,
    pub env: HashMap<String, String>,
}

/// One live prompt admission.
pub struct PromptAdmission {
    pub active_session_id: String,
    pub admission_id: String,
    pub controller: Option<tokio_util::sync::CancellationToken>,
    pub status: String,
}

/// One `sessionInputPauses` entry.
pub struct SessionInputPauseEntry {
    pub active_session_id: String,
    pub owner: Arc<DaemonClientHandle>,
    pub lease_key: String,
    pub pause: SessionInputPause,
}

/// One `updateRestartQueuePauses` entry.
pub struct UpdateRestartQueuePause {
    pub release: SessionInputPause,
}

/// One `acpMcpOwners` entry.
pub struct AcpMcpOwner {
    pub client: Arc<DaemonClientHandle>,
    pub owner_id: String,
    pub server_names: Vec<String>,
    pub release: Option<BoxFuture<'static, ()>>,
}

/// One `closingSessions` entry.
pub struct ClosingSession {
    pub promise: BoxFuture<'static, ()>,
    pub reason: DaemonSessionClosedReason,
    pub descendants: Vec<Arc<StdMutex<ActiveSessionState>>>,
    pub reason_upgrade: Option<BoxFuture<'static, ()>>,
}

/// One `sideQuestionRuns` entry.
pub struct SideQuestionRunEntry {
    pub run: Arc<dyn Fn() + Send + Sync>,
    pub client: Arc<DaemonClientHandle>,
    pub active_session_id: String,
}

/// `class AgentDaemon`.
pub struct AgentDaemon {
    pub socket_path: String,
    pub options: DaemonModeOptions,
    pub shutting_down: AtomicBool,
    server_stopped: tokio_util::sync::CancellationToken,
    shutdown_complete: tokio_util::sync::CancellationToken,
    pub update_restart_queue_pauses: StdMutex<HashMap<String, UpdateRestartQueuePause>>,
    pub session_input_pauses: StdMutex<HashMap<String, SessionInputPauseEntry>>,
    pub acp_mcp_owners: StdMutex<HashMap<String, AcpMcpOwner>>,
    pub mutation_drain: MutationDrainLatch,
    pub update_restart: StdMutex<Option<UpdateRestartTransaction>>,
    pub owns_socket_path: AtomicBool,
    pub socket_identity: StdMutex<Option<DaemonSocketIdentity>>,
    pub clients: StdMutex<Vec<Arc<DaemonClientHandle>>>,
    pub sessions: StdMutex<HashMap<String, Arc<DaemonSessionState>>>,
    /// `openingSessions`: one in-flight runtime open per session key.
    pub opening_sessions: StdMutex<HashMap<String, u64>>,
    /// `reservingSessionOpens`: path resolution through publication.
    pub reserving_session_opens: StdMutex<HashMap<String, u64>>,
    pub binding_completions: StdMutex<HashMap<String, u64>>,
    pub passivating_sessions: StdMutex<HashMap<String, u64>>,
    pub closing_sessions: StdMutex<HashMap<String, ClosingSession>>,
    pub side_question_runs: StdMutex<HashMap<String, SideQuestionRunEntry>>,
    pub prompt_admissions: Arc<StdMutex<HashMap<String, PromptAdmission>>>,
    pub signal_cleanup_handlers: StdMutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    pub cron_store: Arc<AgentCronJobStore>,
    pub agent_dir: String,
    pub cron_scheduler: StdMutex<Option<Arc<AgentCronScheduler>>>,
    pub agent_message_rate_limiter: StdMutex<AgentSessionMessageRateLimiter>,
    pub binding_sessions: StdMutex<HashSet<String>>,
    pub pending_session_names: StdMutex<HashSet<String>>,
    pub restore_active_session_id: StdMutex<Option<String>>,
    pub supervisor_monitor_timer: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    pub supervisor_fence_timer: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    pub supervisor_launch_in_progress: AtomicBool,
    pub supervisor_claims: StdMutex<HashMap<usize, BoundSupervisorGenerationClaim>>,
    pub peer_grants: StdMutex<HashMap<String, DaemonWorkerPeerGrant>>,
    pub peer_claims: StdMutex<HashMap<usize, DaemonWorkerPeerGrant>>,
    pub peer_admissions_fenced: AtomicBool,
    pub agent_messages_paused: AtomicBool,
    pub summarizer: Arc<DaemonSessionSummarizer>,
    pub recovery_journal: StdMutex<Option<WorkerRecoveryJournal>>,
    pub roster_reporter: StdMutex<WorkerRosterReporterState>,
    pub roster_flush_scheduled: AtomicBool,
    pub roster_heartbeat_timer: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    /// `rlmSpawnLedgerInstance` - lazily created, never awaited.
    pub rlm_spawn_ledger_instance: StdMutex<Option<Arc<RlmSpawnLedger>>>,
    pub pending_rlm_spawn_appends:
        StdMutex<HashMap<String, tokio::task::JoinHandle<Result<(), String>>>>,
    /// `passiveRlmSubagentWalks`: same-shape walks waiting on the in-flight one.
    pub passive_rlm_subagent_walks: StdMutex<HashMap<String, Arc<Notify>>>,
    pub passive_rlm_subagent_memo: StdMutex<HashMap<String, PassiveRlmMemoEntry>>,
    /// `state.inFlightBash` per active session.
    pub in_flight_bash: StdMutex<HashMap<String, Arc<Notify>>>,
    /// A unique id source for the `updateRestart.id` symbol and open handles.
    pub id_source: AtomicU64,
}

/// One `passiveRlmSubagentMemo` row (`passiveRlmSubagentWalks` tracks the
/// in-flight walk of the same key).
pub struct PassiveRlmMemoEntry {
    pub fingerprint: String,
    pub result: Vec<PassiveRlmSubagent>,
    pub input_stats: HashMap<String, String>,
    pub in_flight: Option<Arc<Notify>>,
}

/// `...session.rlmDiagnostics` (`daemon-mode.ts:3516`).
///
/// `get rlmDiagnostics()` returns `undefined` at depth 0 (`agent-session.ts:3874`), and
/// the live session is reachable through the seam's `agent_session()` accessor the same
/// way `AgentSessionDaemonAdapter` exposes it. Each key is a real `AgentObserveAgentSummary`
/// field, so the spread lands on the summary's own typed slots.
fn rlm_diagnostics_spread(session: &dyn DaemonSession) -> AgentObserveAgentSummary {
let Some(value) = session
    .agent_session()
    .and_then(|live| live.rlm_diagnostics())
else {
    return AgentObserveAgentSummary::default();
};
let text = |key: &str| {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
};
AgentObserveAgentSummary {
    last_stop_reason: text("lastStopReason"),
    terminal_status: text("terminalStatus"),
    continuation_queued: value.get("continuationQueued").and_then(Value::as_bool),
    compaction_reason: text("compactionReason"),
    current_task_id: text("currentTaskId"),
    diagnostic_state: text("diagnosticState"),
    ..AgentObserveAgentSummary::default()
}
}

/// `AGENT_OBSERVE_LATEST_MESSAGE_MAX_CHARS`: the hard-coded 240 in
/// `createAgentObserveMessagePreview(latest, messages.length - 1, 240)` (`daemon-mode.ts:3525`).
const AGENT_OBSERVE_LATEST_MESSAGE_MAX_CHARS: usize = 240;

/// Saved-roots classifier for the agent-family catalog (daemon-mode.ts:5909-5914):
/// keep every rlmDepth==0 session (including depth-0 forks that record a
/// parentSessionPath) that is not currently an active agent.
pub fn saved_roots_classifier(
    rlm_depth: i64,
    active_paths: &std::collections::HashSet<String>,
    canonical_session_path_value: &str,
) -> bool {
    rlm_depth == 0 && !active_paths.contains(canonical_session_path_value)
}

impl AgentDaemon {
    const PASSIVE_RLM_MEMO_MAX_KEYS: usize = 4;

    /// `constructor(socketPath, options)`.
    pub fn new(socket_path: String, options: DaemonModeOptions) -> Arc<Self> {
        if options.default_session_config.agent_dir.is_none() {
            panic!("Daemon config is missing agentDir");
        }
        let agent_dir = options
            .default_session_config
            .agent_dir
            .clone()
            .expect("checked above");
        // Hosted extensions get ctx.ui.theme; init it headlessly (no TTY, watcher off)
        // or their first access kills the worker.
        let settings = SettingsManager::create(
            options.default_session_config.cwd.as_deref().unwrap_or("."),
            Some(agent_dir.as_str()),
        );
        init_theme_headless(settings.get_theme().as_deref());
        let cron_store = Arc::new(if options.worker.is_some() {
            AgentCronJobStore::for_session_artifacts()
        } else {
            AgentCronJobStore::new(Some(get_cron_jobs_path(&agent_dir)), false)
                .expect("cron job store file path")
        });
        let restore_active_session_id = options
            .worker
            .as_ref()
            .and_then(|worker| worker.restore_active_session_id.clone());
        let recovery_journal = match options.worker.as_ref() {
            Some(_) => std::env::var(DAEMON_WORKER_RECOVERY_JOURNAL_ENV)
                .ok()
                .map(|path| WorkerRecoveryJournal::new(&path)),
            None => None,
        };
        let summarizer = Arc::new(DaemonSessionSummarizer::new(
            Arc::new(|| Vec::new()),
            Some(Arc::new(|_state: &ActiveSessionState| {})),
        ));
        Arc::new(Self {
            socket_path,
            options,
            shutting_down: AtomicBool::new(false),
            server_stopped: tokio_util::sync::CancellationToken::new(),
            shutdown_complete: tokio_util::sync::CancellationToken::new(),
            update_restart_queue_pauses: StdMutex::new(HashMap::new()),
            session_input_pauses: StdMutex::new(HashMap::new()),
            acp_mcp_owners: StdMutex::new(HashMap::new()),
            mutation_drain: MutationDrainLatch::new(),
            update_restart: StdMutex::new(None),
            owns_socket_path: AtomicBool::new(false),
            socket_identity: StdMutex::new(None),
            clients: StdMutex::new(Vec::new()),
            sessions: StdMutex::new(HashMap::new()),
            opening_sessions: StdMutex::new(HashMap::new()),
            reserving_session_opens: StdMutex::new(HashMap::new()),
            binding_completions: StdMutex::new(HashMap::new()),
            passivating_sessions: StdMutex::new(HashMap::new()),
            closing_sessions: StdMutex::new(HashMap::new()),
            side_question_runs: StdMutex::new(HashMap::new()),
            prompt_admissions: Arc::new(StdMutex::new(HashMap::new())),
            signal_cleanup_handlers: StdMutex::new(Vec::new()),
            cron_store: Arc::clone(&cron_store),
            agent_dir,
            cron_scheduler: StdMutex::new(None),
            agent_message_rate_limiter: StdMutex::new(AgentSessionMessageRateLimiter::new(
                DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY,
                DEFAULT_AGENT_MESSAGE_RATE_LIMIT_REFILL_MS,
                None,
            )),
            binding_sessions: StdMutex::new(HashSet::new()),
            pending_session_names: StdMutex::new(HashSet::new()),
            restore_active_session_id: StdMutex::new(restore_active_session_id),
            supervisor_monitor_timer: StdMutex::new(None),
            supervisor_fence_timer: StdMutex::new(None),
            supervisor_launch_in_progress: AtomicBool::new(false),
            supervisor_claims: StdMutex::new(HashMap::new()),
            peer_grants: StdMutex::new(HashMap::new()),
            peer_claims: StdMutex::new(HashMap::new()),
            peer_admissions_fenced: AtomicBool::new(false),
            agent_messages_paused: AtomicBool::new(false),
            summarizer,
            recovery_journal: StdMutex::new(recovery_journal),
            roster_reporter: StdMutex::new(WorkerRosterReporterState::default()),
            roster_flush_scheduled: AtomicBool::new(false),
            roster_heartbeat_timer: StdMutex::new(None),
            rlm_spawn_ledger_instance: StdMutex::new(None),
            pending_rlm_spawn_appends: StdMutex::new(HashMap::new()),
            passive_rlm_subagent_walks: StdMutex::new(HashMap::new()),
            passive_rlm_subagent_memo: StdMutex::new(HashMap::new()),
            in_flight_bash: StdMutex::new(HashMap::new()),
            id_source: AtomicU64::new(1),
        })
    }

    fn next_id(self: &Arc<Self>) -> u64 {
        self.id_source.fetch_add(1, Ordering::SeqCst)
    }

    /// `this.sessions.values()`.
    fn session_states(&self) -> Vec<Arc<DaemonSessionState>> {
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// The `ActiveSessionState` views of the resident sessions.
    fn state_refs(&self) -> Vec<Arc<StdMutex<ActiveSessionState>>> {
        self.session_states()
            .into_iter()
            .map(|entry| Arc::clone(&entry.state))
            .collect()
    }

    /// `this.sessions.values()` keyed as the other daemon modules take them.
    fn states_by_id(&self) -> HashMap<String, Arc<StdMutex<ActiveSessionState>>> {
        self.session_states()
            .into_iter()
            .map(|entry| {
                (
                    entry
                        .state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                    Arc::clone(&entry.state),
                )
            })
            .collect()
    }

    /// `state.runtime.session` (the daemon holds the live handle beside the state).
    fn session_of(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> Arc<dyn DaemonSession> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .map(|entry| Arc::clone(&entry.session))
            .unwrap_or_else(|| Arc::new(MissingSession::new(&active_session_id)))
    }

    fn client_handles(&self) -> Vec<Arc<DaemonClientHandle>> {
        self.clients.lock().expect("clients poisoned").clone()
    }

    fn is_worker(&self) -> bool {
        self.options.worker.is_some()
    }

    /// The daemon runs detached with no terminal, so route its diagnostics to its
    /// rotating log file and the shared structured log (and stderr too, for when
    /// it's run in the foreground).
    fn log(&self, message: &str) {
        // The stderr route is what a foreground run (and `--log` capture) sees, so
        // it carries the same timestamp as the rotating line (audit A8).
        eprintln!("[{}] {message}", now_iso());
        let mut fields = Map::new();
        fields.insert(
            "socketPath".to_string(),
            Value::String(self.socket_path.clone()),
        );
        structured_log().warn(message, Some(fields));
        append_rotating_log(
            &get_daemon_log_path(&self.socket_path),
            &format!("[{}] {message}", now_iso()),
        );
    }

    /// `installCrashHandlers()`.
    fn install_crash_handlers(self: &Arc<Self>) {
        // Rust has no `uncaughtException`/`unhandledRejection` hooks; the port keeps
        // the same intent by logging a panic through the daemon log path.
        let daemon = Arc::downgrade(self);
        std::panic::set_hook(Box::new(move |panic_info| {
            if let Some(daemon) = daemon.upgrade() {
                daemon.log(&format!("uncaught exception: {panic_info}"));
            }
        }));
    }

    /// `start()`.
    pub async fn start(self: &Arc<Self>) -> Result<(), String> {
        self.install_crash_handlers();
        prepare_daemon_socket_path(&self.socket_path, None)
            .await
            .map_err(|error| error.to_string())?;
        native_server::bind(self).await?;
        self.owns_socket_path.store(true, Ordering::SeqCst);
        *self
            .socket_identity
            .lock()
            .expect("socket identity poisoned") = get_daemon_socket_identity(&self.socket_path);
        if crate::utils::pi_user_agent::process_platform() != "win32" {
            restrict_daemon_socket_path(&self.socket_path);
        }
        self.register_signal_handlers();
        self.summarizer.start();
        self.log(&format!(
            "Prime Agent daemon listening on {}",
            self.socket_path
        ));
        if !self.shutting_down.load(Ordering::SeqCst) {
            self.start_cron_scheduler();
        }
        if self.is_worker() {
            let daemon = Arc::clone(self);
            let handle = tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(ROSTER_HEARTBEAT_INTERVAL_MS));
                interval.tick().await;
                loop {
                    interval.tick().await;
                    if daemon.shutting_down.load(Ordering::SeqCst) {
                        break;
                    }
                    daemon.broadcast_roster_frame(&DaemonWorkerRosterOutbound::RosterHeartbeat);
                }
            });
            *self
                .roster_heartbeat_timer
                .lock()
                .expect("roster heartbeat poisoned") = Some(handle);
            self.start_supervisor_monitor();
        }
        Ok(())
    }

    fn supervisor_socket_path_from_env(&self) -> Option<String> {
        let raw = std::env::var(DAEMON_WORKER_SUPERVISOR_SOCKET_ENV).ok()?;
        if raw.is_empty() {
            return None;
        }
        Some(normalize_socket_path(&raw, None))
    }

    /// `startSupervisorMonitor()`.
    fn start_supervisor_monitor(self: &Arc<Self>) {
        if !self.is_worker() {
            return;
        }
        let Some(supervisor_socket_path) = self.supervisor_socket_path_from_env() else {
            return;
        };
        self.schedule_supervisor_availability_check(supervisor_socket_path, 1500);
    }

    /// `scheduleSupervisorAvailabilityCheck(supervisorSocketPath, delayMs)`.
    fn schedule_supervisor_availability_check(
        self: &Arc<Self>,
        supervisor_socket_path: String,
        delay_ms: u64,
    ) {
        if self.shutting_down.load(Ordering::SeqCst)
            || self.has_authenticated_supervisor_connection()
        {
            return;
        }
        {
            let mut timer = self
                .supervisor_monitor_timer
                .lock()
                .expect("supervisor monitor poisoned");
            if let Some(handle) = timer.take() {
                handle.abort();
            }
        }
        let daemon = Arc::clone(self);
        let handle = tokio::spawn(async move {
            delay(delay_ms).await;
            let result = daemon
                .check_supervisor_availability(&supervisor_socket_path)
                .await;
            if let Err(error) = result {
                daemon.log(&format!("supervisor availability check failed: {error}"));
            }
            if !daemon.shutting_down.load(Ordering::SeqCst)
                && !daemon.has_authenticated_supervisor_connection()
            {
                daemon.schedule_supervisor_availability_check(supervisor_socket_path, 5000);
            }
        });
        *self
            .supervisor_monitor_timer
            .lock()
            .expect("supervisor monitor poisoned") = Some(handle);
    }

    /// `checkSupervisorAvailability(supervisorSocketPath)`.
    async fn check_supervisor_availability(
        self: &Arc<Self>,
        supervisor_socket_path: &str,
    ) -> Result<(), String> {
        if self.shutting_down.load(Ordering::SeqCst)
            || self.has_authenticated_supervisor_connection()
        {
            return Ok(());
        }
        if is_daemon_shutdown_admission_active().await.unwrap_or(false) {
            self.schedule_supervisor_availability_check(supervisor_socket_path.to_string(), 5000);
            return Ok(());
        }
        if self.can_connect_to_supervisor(supervisor_socket_path).await {
            self.schedule_supervisor_availability_check(supervisor_socket_path.to_string(), 5000);
            return Ok(());
        }
        self.launch_replacement_supervisor(supervisor_socket_path)
            .await;
        if !self.shutting_down.load(Ordering::SeqCst)
            && !self.has_authenticated_supervisor_connection()
        {
            self.schedule_supervisor_availability_check(supervisor_socket_path.to_string(), 5000);
        }
        Ok(())
    }

    /// `hasAuthenticatedSupervisorConnection()`.
    fn has_authenticated_supervisor_connection(&self) -> bool {
        self.supervisor_claims
            .lock()
            .expect("supervisor claims poisoned")
            .values()
            .any(|_| true)
    }

    /// `revokeSupervisorClaim(client, expected?)`.
    fn revoke_supervisor_claim(
        &self,
        client_key: usize,
        expected: Option<&BoundSupervisorGenerationClaim>,
    ) -> bool {
        let mut claims = self
            .supervisor_claims
            .lock()
            .expect("supervisor claims poisoned");
        if let Some(expected) = expected {
            if claims.get(&client_key) != Some(expected) {
                return false;
            }
        }
        if claims.remove(&client_key).is_none() {
            return false;
        }
        drop(claims);
        if self.is_worker() {
            let owner = self
                .update_restart
                .lock()
                .expect("update restart poisoned")
                .as_ref()
                .and_then(|transaction| transaction.owner);
            if owner == Some(client_key as u64) {
                if let Some(transaction_id) = self
                    .update_restart
                    .lock()
                    .expect("update restart poisoned")
                    .as_ref()
                    .map(|transaction| transaction.id)
                {
                    self.cancel_prepared_update_restart(Some(transaction_id));
                }
            }
        }
        true
    }

    /// `fencePeerTransports(closingReason?)`.
    fn fence_peer_transports(self: &Arc<Self>, closing_reason: Option<DaemonClosingReason>) {
        self.peer_grants
            .lock()
            .expect("peer grants poisoned")
            .clear();
        let keys: Vec<usize> = self
            .peer_claims
            .lock()
            .expect("peer claims poisoned")
            .keys()
            .copied()
            .collect();
        for key in keys {
            self.peer_claims
                .lock()
                .expect("peer claims poisoned")
                .remove(&key);
            if let Some(closing_reason) = &closing_reason {
                if let Some(client) = self.client_by_key(key) {
                    self.write(
                        &client,
                        &DaemonOutbound::DaemonClosing {
                            reason: closing_reason.clone(),
                        },
                    );
                    client.writer.end();
                }
            }
        }
    }

    fn client_by_key(&self, key: usize) -> Option<Arc<DaemonClientHandle>> {
        self.client_handles()
            .into_iter()
            .find(|client| Arc::as_ptr(client) as usize == key)
    }

    /// `clearSupervisorAvailabilityCheck()`.
    fn clear_supervisor_availability_check(&self) {
        if let Some(handle) = self
            .supervisor_monitor_timer
            .lock()
            .expect("supervisor monitor poisoned")
            .take()
        {
            handle.abort();
        }
        if let Some(handle) = self
            .supervisor_fence_timer
            .lock()
            .expect("supervisor fence poisoned")
            .take()
        {
            handle.abort();
        }
    }

    /// `scheduleSupervisorFenceCheck()`.
    fn schedule_supervisor_fence_check(self: &Arc<Self>) {
        if self.shutting_down.load(Ordering::SeqCst)
            || self
                .supervisor_fence_timer
                .lock()
                .expect("supervisor fence poisoned")
                .is_some()
            || self
                .supervisor_claims
                .lock()
                .expect("supervisor claims poisoned")
                .is_empty()
        {
            return;
        }
        let daemon = Arc::clone(self);
        let handle = tokio::spawn(async move {
            delay(SUPERVISOR_FENCE_POLL_MS).await;
            daemon
                .supervisor_fence_timer
                .lock()
                .expect("supervisor fence poisoned")
                .take();
            daemon.check_supervisor_fences().await;
        });
        *self
            .supervisor_fence_timer
            .lock()
            .expect("supervisor fence poisoned") = Some(handle);
    }

    /// `checkSupervisorFences()`.
    async fn check_supervisor_fences(self: &Arc<Self>) {
        let claims: Vec<(usize, BoundSupervisorGenerationClaim)> = self
            .supervisor_claims
            .lock()
            .expect("supervisor claims poisoned")
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect();
        for (key, bound_claim) in claims {
            let result = self
                .assert_supervisor_claim_current(
                    &bound_claim.claim,
                    Some(&bound_claim.owner_fingerprint),
                )
                .await;
            if result.is_err() && self.revoke_supervisor_claim(key, Some(&bound_claim)) {
                if let Some(client) = self.client_by_key(key) {
                    client.writer.end();
                }
            }
        }
        self.schedule_supervisor_fence_check();
    }

    /// `waitForPromptAdmission(claimCheck, parsedAdmission?.controller?.signal)`
    /// (`daemon-mode.ts:3850`).
    ///
    /// TS observes the already-running fence check even if admission cancellation wins
    /// (`daemon-mode.ts:3846-3848`), and on cancellation `void claimCheck.catch(...)`
    /// revokes only the exact binding that initiated it (`:3859-3863`).
    ///
    /// Divergence: the shared Rust `waitForPromptAdmission` port
    /// (`core/prompt_admission.rs:61`) requires a `'static` future, so it cannot observe
    /// the in-flight `assert_supervisor_claim_current` borrow. The cancellation branch
    /// re-drives the same check against the same bound claim instead of observing the
    /// first one; the observable effect (revoke on stale, survive on replacement) is the
    /// same because both calls read the same owner record.
    ///
    /// Returns `(claim_check, admission_cancelled)`. `claim_check` keeps the
    /// `assert_supervisor_claim_current` error string shape the caller maps to the
    /// `supervisor_generation_stale` failure (`:3856-3882`).
    async fn await_supervisor_claim_admission(
        self: &Arc<Self>,
        client_key: usize,
        bound_claim: &BoundSupervisorGenerationClaim,
        admission_signal: Option<tokio_util::sync::CancellationToken>,
    ) -> (Result<String, String>, bool) {
        let Some(signal) = admission_signal else {
            return (
                self.assert_supervisor_claim_current(
                    &bound_claim.claim,
                    Some(&bound_claim.owner_fingerprint),
                )
                .await,
                false,
            );
        };
        let check = self.assert_supervisor_claim_current(
            &bound_claim.claim,
            Some(&bound_claim.owner_fingerprint),
        );
        // `wait_for_prompt_admission` (core/prompt_admission.rs) requires a `'static`
        // future, so the same "await unless the signal aborts first" race is run here
        // with `tokio::select!` over the borrowed check.
        let claim_check = tokio::select! {
            result = check => Some(result),
            _ = signal.cancelled() => None,
        };
        match claim_check {
            Some(claim_check) => (claim_check, false),
            None => {
                // The fence check remains authoritative after cancellation: it revokes
                // only the binding that it checked (`daemon-mode.ts:3859-3863`).
                let daemon = Arc::clone(self);
                let bound_claim = bound_claim.clone();
                tokio::spawn(async move {
                    let result = daemon
                        .assert_supervisor_claim_current(
                            &bound_claim.claim,
                            Some(&bound_claim.owner_fingerprint),
                        )
                        .await;
                    if result.is_err()
                        && daemon.revoke_supervisor_claim(client_key, Some(&bound_claim))
                    {
                        if let Some(client) = daemon.client_by_key(client_key) {
                            client.writer.end();
                        }
                    }
                });
                (
                    Err(PromptAdmissionCancelledError::default().to_string()),
                    true,
                )
            }
        }
    }

    /// `assertSupervisorClaimCurrent(claim, validatedFingerprint?)`.
    async fn assert_supervisor_claim_current(
        &self,
        claim: &SupervisorGenerationClaim,
        validated_fingerprint: Option<&str>,
    ) -> Result<String, String> {
        let owner = DaemonSupervisorOwnerRecord {
            version: 1,
            role: "supervisor".to_string(),
            token: String::new(),
            generation: claim.supervisor_generation.clone(),
            pid: claim.supervisor_pid,
            process_start_id: claim.supervisor_process_start_id.clone(),
            socket_path: claim.supervisor_socket_path.clone(),
            descriptor_dir: String::new(),
            agent_dir: String::new(),
            app_version: VERSION.to_string(),
            phase: String::new(),
            created_at: String::new(),
            updated_at: String::new(),
        };
        assert_daemon_supervisor_owner_current(&owner, validated_fingerprint, None, None)
            .await
            .map_err(|error| error.to_string())
    }

    /// `canConnectToSupervisor(socketPath)`.
    async fn can_connect_to_supervisor(&self, socket_path: &str) -> bool {
        // `createConnection(...)`/`connect` with a 250 ms deadline: true only when
        // the socket actually accepted this probe.
        let client = Arc::new(DaemonClient::new(socket_path));
        match tokio::time::timeout(Duration::from_millis(250), client.connect(250)).await {
            Ok(Ok(())) => {
                client.close();
                true
            }
            _ => false,
        }
    }

    /// `launchReplacementSupervisor(supervisorSocketPath)`.
    async fn launch_replacement_supervisor(self: &Arc<Self>, supervisor_socket_path: &str) {
        if self.supervisor_launch_in_progress.load(Ordering::SeqCst)
            || self.shutting_down.load(Ordering::SeqCst)
        {
            return;
        }
        self.supervisor_launch_in_progress
            .store(true, Ordering::SeqCst);
        let result = self
            .launch_replacement_supervisor_inner(supervisor_socket_path)
            .await;
        if let Err(error) = result {
            self.log(&format!("failed to launch replacement supervisor: {error}"));
        }
        self.supervisor_launch_in_progress
            .store(false, Ordering::SeqCst);
    }

    async fn launch_replacement_supervisor_inner(
        self: &Arc<Self>,
        supervisor_socket_path: &str,
    ) -> Result<(), String> {
        let agent_dir = self.options.default_session_config.agent_dir.clone();
        let lock_directory = Path::new(&self.default_supervisor_lock_directory(supervisor_socket_path)).to_path_buf();
        if let Some(parent) = lock_directory.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut owns_lock = false;
        for _ in 0..3 {
            if owns_lock {
                break;
            }
            let attempt = try_acquire_dir_lock(
                &lock_directory.to_string_lossy(),
                |owner_pid: Option<i32>| async move {
                    owner_pid.is_some_and(is_process_alive)
                },
            )
            .await;
            match attempt {
                Ok(crate::utils::dir_lock::DirLockAttempt::Held) => return Ok(()),
                Ok(crate::utils::dir_lock::DirLockAttempt::Acquired) => owns_lock = true,
                Ok(crate::utils::dir_lock::DirLockAttempt::Reclaimed) => {}
                Err(_) => break,
            }
        }
        if !owns_lock {
            return Ok(());
        }
        let result = async {
            if self.can_connect_to_supervisor(supervisor_socket_path).await {
                return Ok(());
            }
            if is_daemon_shutdown_admission_active().await.unwrap_or(false) {
                return Ok(());
            }
            let launch = create_cli_subprocess_launch_spec(
                &[
                    "--mode".to_string(),
                    "daemon".to_string(),
                    "--daemon-socket".to_string(),
                    supervisor_socket_path.to_string(),
                ],
                None,
                &[],
                None,
            );
            let mut env = daemon_supervisor_launch_env(&std::env::vars().collect(), agent_dir.as_ref());
            let child = spawn_hidden(
                &launch.command,
                &launch.args,
                SpawnOptions {
                    cwd: Some(
                        self.options
                            .default_session_config
                            .cwd
                            .clone()
                            .unwrap_or_else(|| ".".to_string()),
                    ),
                    env: Some(env.into_iter().collect()),
                    ..SpawnOptions::default()
                },
            );
            let child = match child {
                Ok(child) => child,
                Err(error) => return Err(error.to_string()),
            };
            let child_pid = child.child.id().map(|pid| pid as i32).unwrap_or(0);
            let deadline = now_millis() + 30_000.0;
            while !self.shutting_down.load(Ordering::SeqCst) && now_millis() < deadline {
                if !is_process_alive(child_pid) {
                    break;
                }
                let stale: Vec<(usize, BoundSupervisorGenerationClaim)> = self
                    .supervisor_claims
                    .lock()
                    .expect("supervisor claims poisoned")
                    .iter()
                    .map(|(key, value)| (*key, value.clone()))
                    .collect();
                for (key, bound_claim) in stale {
                    if self
                        .assert_supervisor_claim_current(&bound_claim.claim, None)
                        .await
                        .is_ok()
                    {
                        continue;
                    }
                    if now_millis() >= deadline
                        || self
                            .supervisor_claims
                            .lock()
                            .expect("supervisor claims poisoned")
                            .get(&key)
                            != Some(&bound_claim)
                    {
                        break;
                    }
                    self.revoke_supervisor_claim(key, Some(&bound_claim));
                }
                delay(50).await;
            }
            Ok(())
        }
        .await;
        if owns_lock {
            if std::fs::read_to_string(&lock_directory).ok().as_deref() == Some(&format!("{}\n", std::process::id())) {
                let _ = std::fs::remove_file(&lock_directory);
            }
        }
        result
    }

    /// A socket-specific launch lock, separate from the durable ownership registry.
    fn default_supervisor_lock_directory(&self, supervisor_socket_path: &str) -> String {
        let key = format!("{:x}", Sha256::digest(supervisor_socket_path.as_bytes()));
        // Named pipes are not filesystem directories. Keep their launch lock in
        // the daemon's private filesystem socket directory on Windows.
        let directory = if supervisor_socket_path.starts_with(r"\\.\pipe\") {
            super::daemon_socket::default_daemon_socket_dir()
        } else {
            dirname(supervisor_socket_path)
        };
        join_path(&directory, &format!(".supervisor-launch-{}.lock", &key[..12]))
    }

    /// `cleanupSocketPath()`.
    fn cleanup_socket_path(&self) {
        if !self.owns_socket_path.load(Ordering::SeqCst) {
            return;
        }
        let identity = self
            .socket_identity
            .lock()
            .expect("socket identity poisoned")
            .clone();
        cleanup_daemon_socket_path(&self.socket_path, identity, None);
    }

    /// `adoptClientEnv(state, env?)`.
    fn adopt_client_env(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        env: Option<HashMap<String, String>>,
    ) {
        let Some(env) = env else {
            return;
        };
        {
            let mut state = state.lock().expect("active session poisoned");
            // Extensions capture client env (e.g. herdr pane identity) on load, so
            // the session's env is adopted once and never overwritten by watchers.
            if state.client_env.is_some() {
                return;
            }
            state.client_env = Some(env.clone());
        }
        let parent_active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let children: Vec<Arc<StdMutex<ActiveSessionState>>> = self
            .session_states()
            .into_iter()
            .filter(|candidate| {
                let candidate = candidate.state.lock().expect("active session poisoned");
                let metadata = candidate.runtime.metadata.as_ref();
                metadata.and_then(|value| value.kind.as_deref()) == Some("subagent")
                    && metadata.and_then(|value| value.parent_active_session_id.as_deref())
                        == Some(parent_active_session_id.as_str())
            })
            .map(|candidate| Arc::clone(&candidate.state))
            .collect();
        for child in children {
            self.adopt_client_env(&child, Some(env.clone()));
        }
    }

    /// `rlmLedgerSessionsDir()`.
    fn rlm_ledger_sessions_dir(&self) -> String {
        self.options
            .default_session_config
            .session_dir
            .clone()
            .unwrap_or_else(|| get_sessions_dir(&self.agent_dir))
    }

    /// `rlmSpawnLedger()`.
    fn rlm_spawn_ledger(self: &Arc<Self>) -> Arc<RlmSpawnLedger> {
        let mut instance = self
            .rlm_spawn_ledger_instance
            .lock()
            .expect("rlm spawn ledger poisoned");
        if let Some(ledger) = instance.as_ref() {
            return Arc::clone(ledger);
        }
        let daemon = Arc::clone(self);
        let ledger = Arc::new(RlmSpawnLedger::new(
            &self.agent_dir,
            &self.rlm_ledger_sessions_dir(),
            Some(create_rlm_ledger_registry_seed_source()),
            Some(Arc::new(move |message: &str| daemon.log(message))),
        ));
        *instance = Some(Arc::clone(&ledger));
        ledger
    }

    /// `rlmSpawnLedgerFor(sessionDir)`.
    fn rlm_spawn_ledger_for(
        self: &Arc<Self>,
        session_dir: Option<&str>,
    ) -> Arc<RlmSpawnLedger> {
        match session_dir {
            None => self.rlm_spawn_ledger(),
            Some(session_dir) => {
                if resolve_path(session_dir) == resolve_path(&self.rlm_ledger_sessions_dir()) {
                    self.rlm_spawn_ledger()
                } else {
                    let daemon = Arc::clone(self);
                    Arc::new(RlmSpawnLedger::new(
                        &self.agent_dir,
                        session_dir,
                        Some(create_rlm_ledger_registry_seed_source()),
                        Some(Arc::new(move |message: &str| daemon.log(message))),
                    ))
                }
            }
        }
    }

    /// `appendRlmLedgerRenameForState(state, name)`.
    async fn append_rlm_ledger_rename_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) {
        let (child_id, child_file) = {
            let state = state.lock().expect("active session poisoned");
            let metadata = state.runtime.metadata.as_ref();
            (
                metadata.and_then(|value| value.rlm_child_id.clone()),
                state.runtime.session.session_file.clone(),
            )
        };
        let (Some(child_id), Some(child_file)) = (child_id, child_file) else {
            return;
        };
        if let Err(error) = self
            .rlm_spawn_ledger()
            .append_rename(&child_id, &child_file, name)
            .await
        {
            self.log(&format!("failed to append RLM ledger rename: {error}"));
        }
    }

    /// `legacyRlmSubagentRegistryPath(parentSessionFile, parentSessionId)`.
    fn legacy_rlm_subagent_registry_path(
        &self,
        parent_session_file: &str,
        parent_session_id: &str,
    ) -> String {
        join_path(
            &get_session_artifact_path_for_file(parent_session_file, Some(parent_session_id)),
            RLM_SUBAGENT_REGISTRY_FILE,
        )
    }

    /// `readLegacyRlmSubagentRegistry(path, onReadError?)`.
    async fn read_legacy_rlm_subagent_registry(
        self: &Arc<Self>,
        path: &str,
        on_read_error: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Result<Vec<LegacyRlmSubagentRegistryEntry>, std::io::Error> {
        let daemon = Arc::clone(self);
        read_legacy_rlm_subagent_registry(
            path,
            crate::modes::daemon::rlm_ledger::ReadLegacyRlmSubagentRegistryOptions {
                throw_on_read_error: false,
                log: Some(Arc::new(move |message: &str| daemon.log(message))),
                on_read_error,
            },
        )
        .await
    }
}

/// `state.runtime.session.runUserBash(command, options)` options.
#[derive(Debug, Clone, Default)]
pub struct UserBashInvocation {
    pub command: String,
    pub exclude_from_context: Option<bool>,
    pub transient: Option<bool>,
    pub run_id: Option<String>,
}

/// `waitForHeadlessCompletion(session, { waitForRlmQuiescence })`.
#[derive(Debug, Clone, Default)]
pub struct HeadlessCompletionInvocation {
    pub wait_for_rlm_quiescence: Option<bool>,
}

/// `startSideQuestion(agent, id, question, onEvent, previousTurns, retryPolicy)`.
#[derive(Clone, Default)]
pub struct SideQuestionInvocation {
    pub side_question_id: String,
    pub question: String,
    pub previous_turns: Option<Value>,
    pub retry_policy: Option<Value>,
    pub on_event: Option<Arc<dyn Fn(&Value) + Send + Sync>>,
}

/// `session.navigateTree(targetId, options)`.
#[derive(Debug, Clone, Default)]
pub struct NavigateTreeInvocation {
    pub summarize: Option<bool>,
    pub custom_instructions: Option<String>,
    pub replace_instructions: Option<bool>,
    pub label: Option<String>,
}

/// `DaemonAttachResult` (daemon-protocol.ts).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonAttachResult {
    pub protocol: Option<Value>,
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub state: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub messages: Option<Vec<Value>>,
    pub snapshot: Value,
    pub replay: Value,
    #[serde(rename = "lastEventSequence")]
    pub last_event_sequence: u64,
    #[serde(
        rename = "lastEventCursor",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub last_event_cursor: Option<Value>,
    #[serde(
        rename = "snapshotStream",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub snapshot_stream: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub client: Option<Value>,
}

/// The session surface daemon-mode.ts calls on `state.runtime.session` as it
/// drives commands. Private plumbing for this slice: the real trait method set
/// lives in the session/agent core slices; the Rust signatures here mirror the
/// TypeScript calls 1:1 and are implemented by the session slice later.
#[allow(clippy::type_complexity)]
pub trait DaemonSession: Send + Sync {
    fn agent_session(&self) -> Option<Arc<crate::core::agent_session::AgentSession>> { None }
    fn session_manager(&self) -> Arc<StdMutex<SessionManager>>;
    fn runtime(&self) -> Arc<dyn DaemonRuntimeApi>;
    /// `state.runtime.session.settingsManager`; the live session holds the
    /// settings behind a mutex (core/agent_session.rs), so the seam exposes the
    /// same shared handle the daemon mutates for `set_transport`.
    fn settings_manager(&self) -> Option<Arc<StdMutex<SettingsManager>>>;
    fn session_id(&self) -> String;
    fn session_name(&self) -> Option<String>;
    fn session_file(&self) -> Option<String>;
    fn session_dir(&self) -> Option<String>;
    /// `state.runtime.session.setExecEnvProvider(...)` /
    /// `state.runtime.setRuntimeEnvScope(...)` (daemon-extension-binding.ts:55-58).
    fn set_exec_env_provider(&self, client_env: Option<HashMap<String, String>>);
    fn set_runtime_env_scope(&self, client_env: Option<HashMap<String, String>>);
    /// `state.runtime.setSubagentRuntimeHost(...)` (daemon-extension-binding.ts:61).
    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>);
    /// `state.runtime.setRebindSession(...)` (daemon-extension-binding.ts:70).
    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>);
    /// `session.bindExtensions({...})` (daemon-extension-binding.ts:83).
    fn bind_extensions(
        &self,
        binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput,
    ) -> BoxFuture<'static, Result<(), String>>;
    /// `session.abortForUpdateRestart()` (daemon-mode.ts:6925).
    fn abort_for_update_restart(&self);
    fn is_streaming(&self) -> bool;
    fn is_compacting(&self) -> bool;
    fn is_bash_running(&self) -> bool;
    fn is_retrying(&self) -> bool;
    fn is_session_active(&self) -> bool;
    fn is_foreground_active(&self) -> bool { self.is_session_active() }
    fn has_running_rlm_children(&self) -> bool;
    fn unfinished_action_count(&self) -> f64;
    fn messages(&self) -> Vec<AgentMessage>;
    fn message_count(&self) -> usize { self.messages().len() }
    fn model_identity(&self) -> Option<pi_ai::types::Model>;
    fn rlm_depth(&self) -> Option<i64>;
    fn thinking_level(&self) -> Option<String>;
    fn service_tier(&self) -> Option<String>;
    fn system_prompt(&self) -> Option<String>;
    /// The agent-connection entry lists `createAgentConnectionCommands` and
    /// `createAgentConnectionResourceSnapshot` read off the session.
    fn connection_view(&self) -> DaemonConnectionView;
    /// `createAgentConnectionState(state.runtime, state.activeSessionId)`
    /// (agent-connection/snapshot.ts:55). REPAIR CURSOR: the implementation needs
    /// `AgentSessionRuntimeSnapshotSource` (`{ session: AgentSessionSnapshotSource }`,
    /// 28 fields: cwd, leafId, availableThinkingLevels, retryAttempt, steeringMode,
    /// followUpMode, autoCompactionEnabled, messageCount, sessionActions,
    /// compactionCount, goal, scopedModels, activeToolNames, contextUsage, …), which only
    /// the session slice can build. Fix: implement this member in pack H's
    /// `AgentSessionDaemonAdapter` from `AgentSessionRuntime`, then delete the cursor.
    fn connection_state(&self, active_session_id: Option<String>) -> Value;
    fn set_current_recap(&self, recap: Option<&str>);
    fn set_session_name(&self, name: &str);
    fn get_rlm_child_run_status(&self, child_id: &str) -> Option<String>;
    fn register_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> bool;
    fn remove_queued_follow_up(&self, key: &str);
    fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn prompt_until_accepted(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn prompt_and_wait(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn prompt_heartbeat(
        &self,
        job: &AgentCronJob,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn accept_agent_message_prompt(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn steer(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn follow_up(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<bool, String>>;
    fn restore_steering_message(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn restore_follow_up_message(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<bool, String>>;
    fn restore_pending_next_turn_messages(&self, messages: &Value);
    fn restore_session_actions(&self, snapshot: &Value) -> BoxFuture<'static, Result<f64, String>>;
    fn send_custom_message(&self, message: &Value) -> BoxFuture<'static, Result<(), String>>;
    fn resume_queued_work(&self) -> bool;
    fn clear_queued_agent_messages(&self) -> Value;
    fn clear_queue(&self) -> Value;
    fn mutate_queued_message(
        &self,
        lane: &str,
        index: f64,
        expected_text: &str,
        mutation: &Value,
    ) -> Value;
    fn get_steering_message_previews(&self) -> Vec<Value>;
    fn get_follow_up_message_previews(&self) -> Vec<Value>;
    fn request_abort(&self);
    fn cancel_rlm_child_run(&self, child_id: &str) -> bool;
    fn delete_inactive_rlm_subagent(
        &self,
        child_id: &str,
        is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> BoxFuture<'static, Result<String, String>>;
    fn run_user_bash(
        &self,
        command: &str,
        options: RunUserBashOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn execute_bash(&self, command: &str) -> BoxFuture<'static, Result<Value, String>>;
    fn abort_bash(&self);
    fn acquire_session_input_pause(&self) -> SessionInputPause;
    fn wait_for_idle(&self) -> BoxFuture<'static, ()>;
    fn wait_for_headless_completion(
        &self,
        options: HeadlessCompletionOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn refresh_available_models(&self) -> BoxFuture<'static, Result<Vec<pi_ai::types::Model>, String>>;
    fn refresh_model_catalog(&self) -> BoxFuture<'static, Result<Value, String>>;
    fn get_provider_auth_status_source(&self, provider: &str) -> Option<String>;
    fn find_model(&self, provider: &str, model_id: &str) -> Option<pi_ai::types::Model>;
    fn set_model(
        &self,
        model: &pi_ai::types::Model,
        wait_for_extensions: bool,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn cycle_model(
        &self,
        direction: &str,
        wait_for_extensions: bool,
    ) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>>;
    fn set_scoped_models(&self, scoped_models: &Value);
    fn set_thinking_level(&self, level: &str);
    fn set_service_tier(&self, service_tier: &str);
    fn cycle_thinking_level(&self) -> Option<String>;
    fn set_transport(&self, transport: &str);
    fn set_steering_mode(&self, mode: &str);
    fn set_follow_up_mode(&self, mode: &str);
    fn set_auto_compaction_enabled(&self, enabled: bool);
    fn set_auto_retry_enabled(&self, enabled: bool);
    fn compact(
        &self,
        custom_instructions: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn refine(&self, options: RefineOptions) -> BoxFuture<'static, Result<Value, String>>;
    fn abort_compaction(&self);
    fn abort_branch_summary(&self);
    fn abort_retry(&self);
    fn reload(&self) -> BoxFuture<'static, Result<(), String>>;
    fn get_rlm_max_depth_status(&self) -> Value;
    fn set_rlm_max_depth(
        &self,
        max_depth: Value,
        global: bool,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn build_session_context(&self) -> Value;
    fn get_session_stats(&self) -> Value;
    fn get_context_tree(&self) -> Value;
    fn get_rlm_child_snapshots(&self) -> Vec<Value>;
    fn export_to_html(
        &self,
        output_path: Option<&str>,
    ) -> BoxFuture<'static, Result<String, String>>;
    fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String>;
    fn get_user_messages_for_forking(&self) -> Vec<Value>;
    fn get_last_assistant_text(&self) -> String;
    fn get_tool_definition(&self, name: &str) -> Option<Value>;
    fn navigate_tree(
        &self,
        target_id: &str,
        options: NavigateTreeOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn start_side_question(
        &self,
        question: &str,
        options: SideQuestionOptions,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn abort_side_question(&self, side_question_id: &str);
    fn release_acp_mcp_servers(
        &self,
        owner_id: &str,
        server_names: &[String],
    ) -> BoxFuture<'static, Result<(), String>>;
    fn replace_acp_mcp_servers(
        &self,
        servers: &[Value],
        owner_id: &str,
    ) -> BoxFuture<'static, Result<(), String>>;
    fn new_session(
        &self,
        options: Option<NewSessionRuntimeOptions>,
    ) -> BoxFuture<'static, Result<Value, String>>;
    /// `session.releaseRlmChildSession(childId, session)`.
    fn release_rlm_child_session(
        &self,
        child_id: &str,
        session: Arc<dyn DaemonSession>,
    ) -> Option<Box<dyn FnOnce() + Send>>;
    /// `session.repliedToParentSinceTask` for the family catalog row.
    fn replied_to_parent_since_task(&self) -> Option<bool>;
    fn switch_session(
        &self,
        session_path: &str,
        options: SessionPathOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn fork(
        &self,
        entry_id: &str,
        options: ForkOptions,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn import_from_jsonl(
        &self,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>>;
    fn dispose(&self) -> BoxFuture<'static, ()>;
}

/// `recordWorkerRecoveryState`'s busy predicate (daemon-mode.ts:7375-7380):
/// `hasLiveSessionWork(state) || session.isRetrying || session.hasAcceptedPromptInFlight`.
/// `hasLiveSessionWork` is `isSessionActive || hasRunningRlmChildren`, and
/// isRetrying/isStreaming/isCompacting/isBashRunning are subsumed by the Rust
/// `is_session_active`, so the port collapses to these two terms. The running
/// children term is the BUSY-FLAG-01 parity delta (audit BUSY-FLAG-01).
pub(crate) fn worker_recovery_busy(session: &dyn DaemonSession) -> bool {
    session.is_session_active() || session.has_running_rlm_children()
}

/// `runUserBash(command, options)`.
#[derive(Debug, Clone, Default)]
pub struct RunUserBashOptions {
    pub exclude_from_context: Option<bool>,
    pub transient: Option<bool>,
    pub run_id: Option<String>,
}

/// `refine({ instructions, rollbackId, global })`.
#[derive(Debug, Clone, Default)]
pub struct RefineOptions {
    pub instructions: Option<String>,
    pub rollback_id: Option<String>,
    pub global: Option<bool>,
}

/// `navigateTree(targetId, options)`.
#[derive(Debug, Clone, Default)]
pub struct NavigateTreeOptions {
    pub summarize: Option<bool>,
    pub custom_instructions: Option<String>,
    pub replace_instructions: Option<bool>,
    pub label: Option<String>,
}

/// `startSideQuestion(...)`.
#[derive(Clone, Default)]
pub struct SideQuestionOptions {
    pub id: String,
    pub previous_turns: Option<Value>,
    pub retry_policy: Option<Value>,
    pub on_event: Option<Arc<dyn Fn(&Value) + Send + Sync>>,
}

/// `waitForHeadlessCompletion(session, { waitForRlmQuiescence })`.
#[derive(Debug, Clone, Default)]
pub struct HeadlessCompletionOptions {
    pub wait_for_rlm_quiescence: Option<bool>,
}

/// `runtime.newSession(options)`.
#[derive(Debug, Clone, Default)]
pub struct NewSessionRuntimeOptions {
    pub parent_session: Option<Value>,
}

/// `runtime.switchSession(path, { cwdOverride })`.
#[derive(Debug, Clone, Default)]
pub struct SessionPathOptions {
    pub cwd_override: Option<String>,
}

/// `runtime.fork(entryId, { position })`.
#[derive(Debug, Clone, Default)]
pub struct ForkOptions {
    pub position: Option<String>,
}

impl AgentDaemon {
    /// `getSessionState(id)`.
    fn get_session_state(&self, id: &str) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let states = self.states_by_id();
        resolve_active_session_state(&states, id).map_err(|error| error.to_string())
    }

    /// `getBoundSessionState(id)`.
    fn get_bound_session_state(
        &self,
        id: &str,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let state = self.get_session_state(id)?;
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if self
            .binding_sessions
            .lock()
            .expect("binding sessions poisoned")
            .contains(&active_session_id)
        {
            return Err(BoundSessionUnavailableError::new(format!(
                "Active session {active_session_id} is still initializing"
            ))
            .to_string());
        }
        if self
            .closing_sessions
            .lock()
            .expect("closing sessions poisoned")
            .contains_key(&active_session_id)
        {
            return Err(BoundSessionUnavailableError::new(format!(
                "Active session {active_session_id} is closing"
            ))
            .to_string());
        }
        Ok(state)
    }

    /// `findSessionBySessionFile(sessionFile?)`.
    fn find_session_by_session_file(
        &self,
        session_file: Option<&str>,
    ) -> Option<Arc<StdMutex<ActiveSessionState>>> {
        let session_file = session_file?;
        let target = resolve_path(session_file);
        self.session_states().into_iter().find_map(|entry| {
            let file = entry.session.session_file();
            file.map(|file| resolve_path(&file) == target)
                .unwrap_or(false)
                .then(|| Arc::clone(&entry.state))
        })
    }

    /// `findActiveSessionByFile(sessionPath)`.
    fn find_active_session_by_file(
        &self,
        session_path: &str,
    ) -> Option<Arc<StdMutex<ActiveSessionState>>> {
        self.find_session_by_session_file(Some(session_path))
    }

    /// `promptAdmissionKey(activeSessionId, admissionId)`.
    fn prompt_admission_key(&self, active_session_id: &str, admission_id: &str) -> String {
        format!("{active_session_id}\u{0}{admission_id}")
    }

    /// `parseCommandAndRegisterPromptAdmission(client, line)`.
    ///
    /// Parse and synchronously register prompt admission before returning a
    /// promise. This method is intentionally non-async: `handleLine` invokes it
    /// before its first await.
    fn parse_command_and_register_prompt_admission(
        &self,
        client: &Arc<DaemonClientHandle>,
        line: &str,
    ) -> Result<Value, String> {
        let wire_value: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
        if is_daemon_command_envelope(&wire_value) {
            if let Some(client_id) = wire_value.get("clientId").and_then(Value::as_str) {
                if !client_id.is_empty() {
                    client.set_id(client_id);
                }
            }
        }
        let parsed = if is_daemon_command_envelope(&wire_value) {
            let mut command = wire_value
                .get("command")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(id) = wire_value.get("id") {
                command.insert("id".to_string(), id.clone());
            }
            Value::Object(command)
        } else {
            wire_value
        };
        let type_ = parsed.get("type").and_then(Value::as_str).unwrap_or("");
        if type_ == "prompt" || type_ == "prompt_and_wait" {
            // `if (parsed.admissionId !== undefined)` (`daemon-mode.ts:3645`): the branch
            // is entered for any present key, including JSON `null`. `typeof null !==
            // "string"` then fails the check below (`:3646-3648`), so `admissionId: null`
            // must be rejected, not treated as absent.
            if let Some(admission_id) = parsed.get("admissionId") {
                let active_session_id = parsed.get("activeSessionId").and_then(Value::as_str);
                // `typeof parsed.activeSessionId !== "string" || typeof parsed.admissionId
                // !== "string"` (`daemon-mode.ts:3646-3648`).
                let (Some(active_session_id), Some(admission_id)) =
                    (active_session_id, admission_id.as_str())
                else {
                    return Err(
                        "Prompt admission requires string activeSessionId and admissionId"
                            .to_string(),
                    );
                };
                if admission_id.is_empty() {
                    return Err("admissionId must not be empty".to_string());
                }
                let key = self.prompt_admission_key(active_session_id, admission_id);
                if self
                    .prompt_admissions
                    .lock()
                    .expect("prompt admissions poisoned")
                    .contains_key(&key)
                {
                    return Err(format!(
                        "Prompt admission id is already in use: {admission_id}"
                    ));
                }
                self.prompt_admissions
                    .lock()
                    .expect("prompt admissions poisoned")
                    .insert(
                        key,
                        PromptAdmission {
                            active_session_id: active_session_id.to_string(),
                            admission_id: admission_id.to_string(),
                            controller: Some(tokio_util::sync::CancellationToken::new()),
                            status: "waiting".to_string(),
                        },
                    );
            }
        }
        Ok(parsed)
    }

    /// `handleLine(client, line)`.
    pub async fn handle_line(self: &Arc<Self>, client: Arc<DaemonClientHandle>, line: String) {
        let mut command: ParsedDaemonCommand;
        let mut clear_parsed_admission: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let prompt_handler_owns_admission = Arc::new(AtomicBool::new(false));
        {
            let parsed = match self.parse_command_and_register_prompt_admission(&client, &line) {
                Ok(parsed) => parsed,
                Err(error) => {
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                salvage_daemon_command_id(&line).as_deref(),
                                "parse",
                                &error,
                                serialize_daemon_error(&DaemonError::Message(error.clone())),
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    return;
                }
            };
            let parsed_admission = {
                let type_ = parsed.get("type").and_then(Value::as_str).unwrap_or("");
                match (type_ == "prompt" || type_ == "prompt_and_wait")
                    .then(|| parsed.get("activeSessionId").and_then(Value::as_str))
                    .flatten()
                    .zip(parsed.get("admissionId").and_then(Value::as_str))
                {
                    Some((active_session_id, admission_id)) => {
                        let key = self.prompt_admission_key(active_session_id, admission_id);
                        self.prompt_admissions
                            .lock()
                            .expect("prompt admissions poisoned")
                            .get(&key)
                            .map(|admission| {
                                (
                                    key,
                                    admission.active_session_id.clone(),
                                    admission.admission_id.clone(),
                                )
                            })
                    }
                    None => None,
                }
            };
            {
                let prompt_admissions = Arc::clone(&self.prompt_admissions);
                let parsed_admission = parsed_admission.clone();
                clear_parsed_admission = Arc::new(move || {
                    let Some((_, active_session_id, admission_id)) = parsed_admission.as_ref()
                    else {
                        return;
                    };
                    let key = format!("{active_session_id}\u{0}{admission_id}");
                    let mut admissions = prompt_admissions
                        .lock()
                        .expect("prompt admissions poisoned");
                    if let Some(current) = admissions.get(&key) {
                        if current.active_session_id == *active_session_id
                            && current.admission_id == *admission_id
                        {
                            admissions.remove(&key);
                        }
                    }
                });
            }
            // Envelope client identity is irrelevant to worker-local prompt admission.
            // Public supervisor authentication has already bound this socket's identity.
            if self.is_worker() && !client.authenticated() {
                let command_id = parsed.get("id").and_then(Value::as_str).map(str::to_string);
                let type_ = parsed.get("type").and_then(Value::as_str).unwrap_or("");
                if type_ == "peer_auth" {
                    let grant_id = parsed
                        .get("grantId")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let grant = grant_id.as_deref().and_then(|grant_id| {
                        self.peer_grants
                            .lock()
                            .expect("peer grants poisoned")
                            .get(grant_id)
                            .cloned()
                    });
                    if let Some(grant_id) = grant_id.as_deref() {
                        self.peer_grants
                            .lock()
                            .expect("peer grants poisoned")
                            .remove(grant_id);
                    }
                    let presented_token_hash = parsed
                        .get("token")
                        .and_then(Value::as_str)
                        .map(|token| Sha256::digest(token.as_bytes()).to_vec());
                    let expected_token_hash = grant
                        .as_ref()
                        .map(|grant| Sha256::digest(grant.token.as_bytes()).to_vec());
                    let expires_at = grant
                        .as_ref()
                        .map(|grant| {
                            chrono::DateTime::parse_from_rfc3339(&grant.expires_at)
                                .map(|value| value.timestamp_millis() as f64)
                                .ok()
                        })
                        .unwrap_or(None);
                    let worker_instance_id = parsed.get("workerInstanceId").and_then(Value::as_str);
                    let purpose = parsed.get("purpose").and_then(Value::as_str);
                    let invalid = self.peer_admissions_fenced.load(Ordering::SeqCst)
                        || grant.is_none()
                        || presented_token_hash.is_none()
                        || expected_token_hash.is_none()
                        || !timing_safe_equal(
                            presented_token_hash.as_deref().unwrap_or(&[]),
                            expected_token_hash.as_deref().unwrap_or(&[]),
                        )
                        || worker_instance_id
                            != grant
                                .as_ref()
                                .map(|grant| grant.worker_instance_id.as_str())
                        || purpose != grant.as_ref().map(|grant| grant.purpose.as_str())
                        || expires_at.is_none()
                        || expires_at.unwrap_or(0.0) <= now_millis();
                    if invalid {
                        clear_parsed_admission();
                        self.write(
                            &client,
                            &DaemonOutbound::Raw(
                                serde_json::to_value(DaemonResponse::failure(
                                    command_id.as_deref(),
                                    "peer_auth",
                                    "Peer authentication failed",
                                    None,
                                ))
                                .unwrap_or(Value::Null),
                            ),
                        );
                        client.writer.end();
                        return;
                    }
                    let grant = grant.expect("checked above");
                    client.set_authenticated("session_client");
                    self.peer_claims
                        .lock()
                        .expect("peer claims poisoned")
                        .insert(Arc::as_ptr(&client) as usize, grant.clone());
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(serde_json::json!({
                            "id": command_id,
                            "type": "response",
                            "command": "peer_auth",
                            "success": true,
                            "data": {
                                "workerInstanceId": grant.worker_instance_id,
                                "activeSessionId": grant.active_session_id,
                                "purpose": grant.purpose,
                            },
                        })),
                    );
                    return;
                }
                let worker = self
                    .options
                    .worker
                    .clone()
                    .expect("is_worker checked above");
                let parsed_worker_instance_id = parsed.get("workerInstanceId");
                let supervisor_generation =
                    parsed.get("supervisorGeneration").and_then(Value::as_str);
                let supervisor_pid = parsed.get("supervisorPid").and_then(Value::as_f64);
                let supervisor_process_start_id = parsed.get("supervisorProcessStartId");
                let supervisor_socket_path =
                    parsed.get("supervisorSocketPath").and_then(Value::as_str);
                let invalid = type_ != "worker_auth"
                    || parsed.get("token").and_then(Value::as_str) != Some(worker.authentication_token.as_str())
                    // Enforced only when presented: a downgraded (pre-instance-id)
                    // supervisor must still adopt live workers.
                    || (parsed_worker_instance_id.is_some()
                        && parsed_worker_instance_id.and_then(Value::as_str) != worker.worker_instance_id.as_deref())
                    || supervisor_generation.is_none()
                    || supervisor_pid.map(|pid| pid.fract() != 0.0 || pid <= 0.0).unwrap_or(true)
                    || (supervisor_process_start_id.is_some() && supervisor_process_start_id.and_then(Value::as_str).is_none())
                    || supervisor_socket_path.is_none();
                if invalid {
                    clear_parsed_admission();
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                command_id.as_deref(),
                                "worker_auth",
                                "Worker authentication failed",
                                None,
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    client.writer.end();
                    return;
                }
                let claim = SupervisorGenerationClaim {
                    supervisor_generation: supervisor_generation.unwrap_or_default().to_string(),
                    supervisor_pid: supervisor_pid.unwrap_or(0.0) as i64,
                    supervisor_process_start_id: supervisor_process_start_id
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    supervisor_socket_path: supervisor_socket_path.unwrap_or_default().to_string(),
                };
                let owner_fingerprint =
                    match self.assert_supervisor_claim_current(&claim, None).await {
                        Ok(fingerprint) => fingerprint,
                        Err(_) => {
                            self.write(
                                &client,
                                &DaemonOutbound::Raw(
                                    serde_json::to_value(DaemonResponse::failure(
                                        command_id.as_deref(),
                                        "worker_auth",
                                        "supervisor_generation_stale",
                                        None,
                                    ))
                                    .unwrap_or(Value::Null),
                                ),
                            );
                            client.writer.end();
                            return;
                        }
                    };
                let client_key = Arc::as_ptr(&client) as usize;
                let previous_keys: Vec<usize> = self
                    .supervisor_claims
                    .lock()
                    .expect("supervisor claims poisoned")
                    .keys()
                    .copied()
                    .collect();
                for previous in previous_keys {
                    if previous != client_key {
                        self.revoke_supervisor_claim(previous, None);
                        if let Some(previous_client) = self.client_by_key(previous) {
                            previous_client.writer.end();
                        }
                    }
                }
                client.set_authenticated("supervisor");
                self.supervisor_claims
                    .lock()
                    .expect("supervisor claims poisoned")
                    .insert(
                        client_key,
                        BoundSupervisorGenerationClaim {
                            claim,
                            owner_fingerprint,
                        },
                    );
                self.clear_supervisor_availability_check();
                self.schedule_supervisor_fence_check();
                let mut capabilities = vec![DAEMON_WORKER_ROSTER_CAPABILITY.to_string()];
                if worker.worker_instance_id.is_some() {
                    capabilities.push(DAEMON_WORKER_PEER_TRANSPORT_CAPABILITY.to_string());
                }
                self.write(
                    &client,
                    &DaemonOutbound::Raw(serde_json::json!({
                        "id": command_id,
                        "type": "response",
                        "command": "worker_auth",
                        "success": true,
                        "data": { "capabilities": capabilities },
                    })),
                );
                self.roster_reporter
                    .lock()
                    .expect("roster reporter poisoned")
                    .snapshot_pending = true;
                self.schedule_roster_flush();
                return;
            }
            let client_key = Arc::as_ptr(&client) as usize;
            let peer_claim = if self.is_worker() {
                self.peer_claims
                    .lock()
                    .expect("peer claims poisoned")
                    .get(&client_key)
                    .cloned()
            } else {
                None
            };
            let type_ = parsed.get("type").and_then(Value::as_str).unwrap_or("");
            if peer_claim.is_some() {
                let command_id = parsed.get("id").and_then(Value::as_str).map(str::to_string);
                let command_name = if type_.is_empty() { "unknown" } else { type_ };
                // Reachable only through shutdown, which fences admissions without
                // ending live peers.
                if self.peer_admissions_fenced.load(Ordering::SeqCst) {
                    clear_parsed_admission();
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                command_id.as_deref(),
                                command_name,
                                "Direct peer transport is fenced",
                                None,
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    return;
                }
                let peer_claim = peer_claim.expect("checked above");
                if type_.is_empty()
                    || !is_session_plane_daemon_command(type_)
                    || parsed.get("activeSessionId").and_then(Value::as_str)
                        != Some(peer_claim.active_session_id.as_str())
                {
                    clear_parsed_admission();
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                command_id.as_deref(),
                                command_name,
                                "Command is not allowed on this direct peer transport",
                                None,
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    return;
                }
            } else if self.is_worker() {
                let bound_claim = self
                    .supervisor_claims
                    .lock()
                    .expect("supervisor claims poisoned")
                    .get(&client_key)
                    .cloned();
                let Some(bound_claim) = bound_claim else {
                    clear_parsed_admission();
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                parsed.get("id").and_then(Value::as_str),
                                "worker_auth",
                                "supervisor_generation_stale",
                                None,
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    client.writer.end();
                    return;
                };
                let admission_signal = parsed_admission.as_ref().and_then(|(key, _, _)| {
                    self.prompt_admissions
                        .lock()
                        .expect("prompt admissions poisoned")
                        .get(key)
                        .and_then(|admission| admission.controller.clone())
                });
                let (claim_check, admission_cancelled) = self
                    .await_supervisor_claim_admission(
                        client_key,
                        &bound_claim,
                        admission_signal,
                    )
                    .await;
                match claim_check {
                    Ok(owner_fingerprint) => {
                        let current = self
                            .supervisor_claims
                            .lock()
                            .expect("supervisor claims poisoned")
                            .get(&client_key)
                            .cloned();
                        if current.as_ref() != Some(&bound_claim) || client.writer.destroyed() {
                            clear_parsed_admission();
                            return;
                        }
                        if let Some(claim) = self
                            .supervisor_claims
                            .lock()
                            .expect("supervisor claims poisoned")
                            .get_mut(&client_key)
                        {
                            claim.owner_fingerprint = owner_fingerprint;
                        }
                    }
                    Err(error) => {
                        clear_parsed_admission();
                        let current = self
                            .supervisor_claims
                            .lock()
                            .expect("supervisor claims poisoned")
                            .get(&client_key)
                            .cloned();
                        if current.as_ref() != Some(&bound_claim) || client.writer.destroyed() {
                            return;
                        }
                        let message = if admission_cancelled {
                            error
                        } else {
                            "supervisor_generation_stale".to_string()
                        };
                        let command_name = if type_.is_empty() {
                            "worker_auth"
                        } else {
                            type_
                        };
                        self.write(
                            &client,
                            &DaemonOutbound::Raw(
                                serde_json::to_value(DaemonResponse::failure(
                                    parsed.get("id").and_then(Value::as_str),
                                    command_name,
                                    &message,
                                    None,
                                ))
                                .unwrap_or(Value::Null),
                            ),
                        );
                        // Cancelling this prompt only abandons its admission wait. A
                        // genuine stale supervisor claim fences only the binding that
                        // it checked.
                        if !admission_cancelled
                            && self.revoke_supervisor_claim(client_key, Some(&bound_claim))
                        {
                            client.writer.end();
                        }
                        return;
                    }
                }
            }
            if self.is_worker() && !type_.is_empty() && type_.starts_with("worker_") {
                let worker_command =
                    ParsedDaemonCommand::from_value(&parsed).unwrap_or_else(|| ParsedDaemonCommand::new(type_));
                let update_lifecycle = matches!(
                    worker_command.type_.as_str(),
                    "worker_prepare_update" | "worker_commit_update" | "worker_cancel_update"
                );
                if self
                    .update_restart
                    .lock()
                    .expect("update restart poisoned")
                    .is_some()
                    && !update_lifecycle
                {
                    self.write(
                        &client,
                        &DaemonOutbound::Raw(
                            serde_json::to_value(DaemonResponse::failure(
                                worker_command.id.as_deref(),
                                &worker_command.type_,
                                "Daemon is preparing an update restart",
                                None,
                            ))
                            .unwrap_or(Value::Null),
                        ),
                    );
                    return;
                }
                if !update_lifecycle {
                    self.mutation_drain.begin();
                }
                let result = self.handle_worker_command(&client, &worker_command).await;
                if !update_lifecycle {
                    self.mutation_drain.end();
                }
                if let Err(error) = result {
                    self.log(&format!(
                        "daemon worker command \"{}\" failed: {error}",
                        worker_command.type_
                    ));
                    self.write(&client, &DaemonOutbound::Raw(serde_json::to_value(DaemonResponse::failure(worker_command.id.as_deref(), &worker_command.type_, &error, serialize_daemon_error(&DaemonError::Message(error.clone())))).unwrap_or(Value::Null)));
                }
                return;
            }
            if type_.is_empty() || !contains(&DAEMON_COMMAND_TYPES, type_) {
                let command_name = if type_.is_empty() { "unknown" } else { type_ };
                let command_id = parsed.get("id").and_then(Value::as_str);
                self.write(
                    &client,
                    &DaemonOutbound::Raw(
                        serde_json::to_value(DaemonResponse::failure(
                            command_id,
                            command_name,
                            &format!("Unknown daemon command: {command_name}"),
                            None,
                        ))
                        .unwrap_or(Value::Null),
                    ),
                );
                return;
            }
            command =
                ParsedDaemonCommand::from_value(&parsed).unwrap_or_else(|| ParsedDaemonCommand::new(type_));
        }

        let mutation =
            command.type_ != "prepare_update_restart" && is_daemon_mutating_command(&command.type_);
        let restart_phase = self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .as_ref()
            .map(|transaction| transaction.phase.clone());
        // Mirror the supervisor's drain/fence split: abort-style commands stay
        // admitted only while mutations drain; once the checkpoint is being
        // captured they could race the snapshot and are rejected too.
        let restart_rejected = match restart_phase.as_deref() {
            Some("preparing") => !contains(&UPDATE_RESTART_DRAIN_COMMANDS, &command.type_),
            Some(_) => command.type_ != "shutdown",
            None => false,
        };
        if mutation && restart_rejected {
            clear_parsed_admission();
            self.write(
                &client,
                &DaemonOutbound::Raw(
                    serde_json::to_value(DaemonResponse::failure(
                        command.id.as_deref(),
                        &command.type_,
                        "Daemon is preparing an update restart",
                        None,
                    ))
                    .unwrap_or(Value::Null),
                ),
            );
            return;
        }
        if mutation {
            self.mutation_drain.begin();
        }
        let owns = Arc::clone(&prompt_handler_owns_admission);
        let owns_flag = Arc::new(move || owns.store(true, Ordering::SeqCst));
        match self.handle_command(&client, &command, owns_flag).await {
            Ok(Some(response)) => {
                self.write(
                    &client,
                    &DaemonOutbound::Raw(serde_json::to_value(response).unwrap_or(Value::Null)),
                );
            }
            Ok(None) => {}
            Err(error) => {
                // Only the error message reaches the client (serializeDaemonError
                // drops the rest), so log the full error here.
                self.log(&format!(
                    "daemon command \"{}\" failed: {error}",
                    command.type_
                ));
                self.write(
                    &client,
                    &DaemonOutbound::Raw(
                        serde_json::to_value(DaemonResponse::failure(
                            command.id.as_deref(),
                            &command.type_,
                            &error,
                            serialize_daemon_error(&DaemonError::Message(error.clone())),
                        ))
                        .unwrap_or(Value::Null),
                    ),
                );
            }
        }
        if !prompt_handler_owns_admission.load(Ordering::SeqCst) {
            clear_parsed_admission();
        }
        if mutation {
            self.mutation_drain.end();
        }
    }

    /// `writeWorkerSuccess(client, command, data?)`.
    fn write_worker_success(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
        data: Option<Value>,
    ) {
        let mut object = Map::new();
        object.insert(
            "id".to_string(),
            command.id.clone().map(Value::String).unwrap_or(Value::Null),
        );
        object.insert("type".to_string(), Value::String("response".to_string()));
        object.insert("command".to_string(), Value::String(command.type_.clone()));
        object.insert("success".to_string(), Value::Bool(true));
        if let Some(data) = data {
            object.insert("data".to_string(), data);
        }
        self.write(client, &DaemonOutbound::Raw(Value::Object(object)));
    }
}

impl AgentDaemon {
    /// `summaryForActiveSession(state)`.
    fn summary_for_state(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> SessionSummary {
        let saved = self.session_of(state).session_file().and_then(|path| read_session_info_sync(&path));
        // REPAIR CURSOR: `read_session_info_sync` returns the canonical
        // `core::session_manager::SessionInfo`, but `summary_for_active_session`
        // (modes/daemon/daemon_session_list.rs:402) takes its own duplicate
        // `daemon_session_list::SessionInfo` stub. That file is owned by another slice and
        // is not in this pack's file list, so the type is not changed here. Fix: delete the
        // `SessionInfo`/`SessionState`/`AgentStatusRecord` stubs in daemon_session_list.rs
        // and import the canonical `core::session_manager::SessionInfo` (+ its
        // `SessionState`, `AgentStatus`); the two files then line up with no conversion.
        summary_for_active_session(state, saved.as_ref(), false, false, false)
    }

    /// `handleCommand(client, command, onPromptHandlerOwnsAdmission)`.
    async fn handle_command(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
        on_prompt_handler_owns_admission: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Option<DaemonResponse>, String> {
        let body = &command.body;
        let id = command.id.as_deref();
        if body.get("agentMessageId").and_then(Value::as_str) == Some("") {
            return Err("agentMessageId must not be empty".to_string());
        }
        if body.get("admissionId").and_then(Value::as_str) == Some("") {
            return Err("admissionId must not be empty".to_string());
        }
        if (command.type_ == "steer" || command.type_ == "follow_up")
            && body.get("expandPromptTemplates").and_then(Value::as_bool) != Some(false)
        {
            let replay_fields: Vec<&str> = ["content", "customMessage", "prefixMessages"]
                .into_iter()
                .filter(|field| body.get(*field).is_some())
                .collect();
            if !replay_fields.is_empty() {
                return Err(format!(
                    "{} replay fields ({}) require expandPromptTemplates=false",
                    command.type_,
                    replay_fields.join(", ")
                ));
            }
        }
        match command.type_.as_str() {
            "ack_result" => Ok(None),
            "list" => {
                let active_sessions = self.state_refs();
                let scheduled_jobs = self.cron_store.list();
                let sessions = if body.get("all").and_then(Value::as_bool) != Some(true) {
                    self.build_session_list_with_passive_rlm_subagents(
                        &active_sessions,
                        Vec::new(),
                        &scheduled_jobs,
                    )
                    .await
                } else {
                    let default_config = &self.options.default_session_config;
                    let list_session_dir = body
                        .get("sessionDir")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| default_config.session_dir.clone());
                    let saved_sessions = match body.get("cwd").and_then(Value::as_str) {
                        Some(cwd) => {
                            SessionManager::list(&resolve_path(cwd), list_session_dir.as_deref(), None)
                                .await
                        }
                        None => SessionManager::list_all(None, list_session_dir.as_deref()).await,
                    };
                    self.build_session_list_with_passive_rlm_subagents(
                        &active_sessions,
                        saved_sessions,
                        &scheduled_jobs,
                    )
                    .await
                };
                Ok(Some(DaemonResponse::success(
                    id,
                    "list",
                    Some(serde_json::json!({ "sessions": sessions })),
                )))
            }
            "list_saved_sessions" => {
                let mut active_session_id: Option<String> = None;
                let cwd: String;
                let mut session_dir: Option<String> = None;
                if let Some(value) = body.get("activeSessionId").and_then(Value::as_str) {
                    active_session_id = Some(value.to_string());
                    let state = self.get_session_state(value)?;
                    let manager = self.session_of(&state).session_manager();
                    let manager = manager.lock().expect("session manager poisoned");
                    cwd = manager.get_cwd();
                    session_dir = Some(manager.get_session_dir());
                } else {
                    cwd = resolve_path(body.get("cwd").and_then(Value::as_str).unwrap_or(""));
                    session_dir = body
                        .get("sessionDir")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                let command_id = command.id.clone();
                let progress_client = Arc::clone(client);
                let progress_daemon = Arc::clone(self);
                let progress_active_session_id = active_session_id.clone();
                let on_progress: Option<Arc<dyn Fn(i64, i64) + Send + Sync>> =
                    command_id.as_ref().map(|command_id| -> Arc<dyn Fn(i64, i64) + Send + Sync> {
                        let client = Arc::clone(&progress_client);
                        let daemon = Arc::clone(&progress_daemon);
                        let active_session_id = progress_active_session_id.clone();
                        let command_id = command_id.clone();
                        Arc::new(move |loaded: i64, total: i64| {
                            let mut object = Map::new();
                            object.insert("id".to_string(), Value::String(command_id.clone()));
                            object.insert(
                                "type".to_string(),
                                Value::String("session_list_progress".to_string()),
                            );
                            object.insert(
                                "command".to_string(),
                                Value::String("list_saved_sessions".to_string()),
                            );
                            if let Some(active_session_id) = &active_session_id {
                                object.insert(
                                    "activeSessionId".to_string(),
                                    Value::String(active_session_id.clone()),
                                );
                            }
                            object.insert("loaded".to_string(), Value::from(loaded));
                            object.insert("total".to_string(), Value::from(total));
                            let _ = daemon.write_public_value(
                                &client,
                                &Value::Object(object),
                                "session_list_progress",
                            );
                        })
                    });
                let item_client = Arc::clone(client);
                let item_daemon = Arc::clone(self);
                let item_active_session_id = active_session_id.clone();
                let on_session: Option<Arc<dyn Fn(&SessionInfo) + Send + Sync>> =
                    command_id.as_ref().map(|command_id| -> Arc<dyn Fn(&SessionInfo) + Send + Sync> {
                        let client = Arc::clone(&item_client);
                        let daemon = Arc::clone(&item_daemon);
                        let active_session_id = item_active_session_id.clone();
                        let command_id = command_id.clone();
                        Arc::new(move |session: &SessionInfo| {
                            let mut object = Map::new();
                            object.insert("id".to_string(), Value::String(command_id.clone()));
                            object.insert(
                                "type".to_string(),
                                Value::String("session_list_item".to_string()),
                            );
                            object.insert(
                                "command".to_string(),
                                Value::String("list_saved_sessions".to_string()),
                            );
                            if let Some(active_session_id) = &active_session_id {
                                object.insert(
                                    "activeSessionId".to_string(),
                                    Value::String(active_session_id.clone()),
                                );
                            }
                            object.insert(
                                "session".to_string(),
                                serde_json::to_value(serialize_saved_session_info(session))
                                    .unwrap_or(Value::Null),
                            );
                            let _ = daemon.write_public_value(
                                &client,
                                &Value::Object(object),
                                "session_list_item",
                            );
                        })
                    });
                let scope_current = body.get("scope").and_then(Value::as_str) == Some("current");
                let callbacks = crate::core::session_manager::SessionListCallbacks {
                    on_progress: on_progress.clone().map(|callback| {
                        Box::new(move |loaded: i64, total: i64| callback(loaded, total))
                            as Box<crate::core::session_manager::SessionListProgress>
                    }),
                    on_session: on_session.clone().map(|callback| {
                        Box::new(move |session: &SessionInfo| callback(session))
                            as Box<crate::core::session_manager::SessionListItem>
                    }),
                };
                let saved_sessions = if scope_current {
                    SessionManager::list(&cwd, session_dir.as_deref(), Some(callbacks)).await
                } else {
                    SessionManager::list_all(Some(&callbacks), session_dir.as_deref()).await
                };
                let daemon = Arc::clone(self);
                let sessions = with_passive_rlm_descendant_infos(
                    saved_sessions,
                    &self.rlm_spawn_ledger_for(session_dir.as_deref()),
                    crate::modes::daemon::rlm_ledger::WithPassiveRlmDescendantInfosOptions {
                        cwd: if scope_current {
                            Some(cwd.clone())
                        } else {
                            None
                        },
                        on_session,
                        log: Some(Arc::new(move |message: &str| daemon.log(message))),
                    },
                )
                .await;
                Ok(Some(DaemonResponse::success(
                    id,
                    "list_saved_sessions",
                    Some(serde_json::json!({
                        "sessions": sessions.iter().map(serialize_saved_session_info).collect::<Vec<_>>()
                    })),
                )))
            }
            "create" => {
                let state = self.create_runtime(command, None).await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "create",
                    Some(
                        serde_json::to_value(self.summary_for_state(&state))
                            .unwrap_or(Value::Null),
                    ),
                )))
            }
            "attach" => {
                let active_session_id = body
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "activeSessionId is required".to_string())?;
                let state = self
                    .get_or_hydrate_bound_session_state(active_session_id)
                    .await?;
                if let Some(client_id) = body.get("clientId").and_then(Value::as_str) {
                    client.set_id(client_id);
                }
                let state_active_session_id = state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                let requested_capabilities: Option<HashSet<String>> =
                    body.get("capabilities").and_then(Value::as_array).map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect::<HashSet<String>>()
                    });
                let capabilities = normalize_client_capabilities(
                    requested_capabilities.as_ref(),
                    body.get("supportsExtensionUi").and_then(Value::as_bool),
                );
                set_daemon_client_session_capabilities(
                    client,
                    &state_active_session_id,
                    capabilities,
                );
                let streams_snapshot = client
                    .state
                    .lock()
                    .expect("daemon client poisoned")
                    .transport
                    .as_deref()
                    == Some("private-framed")
                    && client
                        .capabilities_for_session(&state_active_session_id)
                        .contains("chunked_snapshot");
                // Attach is admitted during update-restart preparation as a read. Env
                // adoption remains safe while mutations are only draining; after
                // fencing, defer it until rollback so the checkpoint never omits a
                // live identity.
                let client_env_map = body.get("env").and_then(Value::as_object).map(|object| {
                    object
                        .iter()
                        .filter_map(|(key, value)| {
                            value.as_str().map(|value| (key.clone(), value.to_string()))
                        })
                        .collect::<HashMap<String, String>>()
                });
                let client_env = filter_client_env(client_env_map.as_ref());
                let defer_client_env = self
                    .update_restart
                    .lock()
                    .expect("update restart poisoned")
                    .as_ref()
                    .map(|transaction| transaction.phase != "preparing")
                    .unwrap_or(false);
                if !defer_client_env {
                    self.adopt_client_env(&state, client_env.clone());
                }
                state
                    .lock()
                    .expect("active session poisoned")
                    .pending_attaches += 1;
                // Admit the viewer before capture so events emitted during the
                // asynchronous snapshot work are retained, not missed entirely.
                {
                    let mut guard = state.lock().expect("active session poisoned");
                    if !guard.clients.iter().any(|candidate| Arc::ptr_eq(candidate, &client.state)) {
                        guard.clients.push(client.state.clone());
                    }
                }
                client.state.lock().expect("daemon client poisoned").attached_active_session_ids.insert(state_active_session_id.clone());
                if streams_snapshot { mark_client_snapshot_streaming(client, &state_active_session_id); }
                let result = self.create_attach_result(client, &state, command).await;
                if result.is_ok() {
                    let current = self
                        .sessions
                        .lock()
                        .expect("sessions poisoned")
                        .get(&state_active_session_id)
                        .cloned();
                    if current.as_deref().map(|entry| Arc::as_ptr(&entry.state)) != Some(Arc::as_ptr(&state))
                        || self
                            .closing_sessions
                            .lock()
                            .expect("closing sessions poisoned")
                            .contains_key(&state_active_session_id)
                    {
                        self.finish_attach_failure(client, &state, streams_snapshot, None);
                        return Err(BoundSessionUnavailableError::new(format!(
                            "Active session {state_active_session_id} closed during attach"
                        ))
                        .to_string());
                    }
                }
                let result = match result {
                    Ok(result) => result,
                    Err(error) => {
                        self.finish_attach_failure(client, &state, streams_snapshot, None);
                        return Err(error);
                    }
                };
                {
                    let mut state_guard = state.lock().expect("active session poisoned");
                    state_guard.pending_attaches = state_guard.pending_attaches.saturating_sub(1);
                }
                client
                    .state
                    .lock()
                    .expect("daemon client poisoned")
                    .attached_active_session_ids
                    .insert(state_active_session_id.clone());
                // Carrier-less mutation: a direct viewer changes directAttachedClients
                // with no session event.
                if client
                    .state
                    .lock()
                    .expect("daemon client poisoned")
                    .authentication_role
                    .as_deref()
                    == Some("session_client")
                {
                    self.schedule_roster_flush();
                }
                if defer_client_env {
                    if let Some(env) = client_env {
                        if let Some(transaction) = self
                            .update_restart
                            .lock()
                            .expect("update restart poisoned")
                            .as_mut()
                        {
                            transaction.deferred_client_env.push(DeferredClientEnv {
                                client: Arc::clone(client),
                                state: Arc::clone(&state),
                                env,
                            });
                        }
                    }
                }
                self.publish_jev_attach_footer(client, &state);
                if streams_snapshot {
                    let snapshot_id = snapshot_transfer_id(&result.snapshot);
                    let snapshot_messages = result
                        .snapshot
                        .get("messages")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    if let Err(error) = Self::start_snapshot_stream(
                        self,
                        client,
                        &state,
                        &snapshot_id,
                        &snapshot_messages,
                        &result.snapshot,
                    ) {
                        return Err(error);
                    }
                    let streamed = result.with_messages_cleared(snapshot_id);
                    return Ok(Some(DaemonResponse::success(
                        id,
                        "attach",
                        Some(serde_json::to_value(streamed).unwrap_or(Value::Null)),
                    )));
                }
                // Slim clients consume only the command response; legacy clients (e.g.
                // the plain daemon attach REPL) read state/messages off this event.
                // Skipping it for slim clients halves the attach payload.
                if result.state.is_some() && result.messages.is_some() {
                    let mut object = Map::new();
                    object.insert(
                        "type".to_string(),
                        Value::String("session_attached".to_string()),
                    );
                    object.insert(
                        "activeSessionId".to_string(),
                        Value::String(state_active_session_id.clone()),
                    );
                    object.insert(
                        "state".to_string(),
                        result.state.clone().unwrap_or(Value::Null),
                    );
                    object.insert(
                        "messages".to_string(),
                        Value::Array(result.messages.clone().unwrap_or_default()),
                    );
                    object.insert("snapshot".to_string(), result.snapshot.clone());
                    object.insert("replay".to_string(), result.replay.clone());
                    object.insert(
                        "lastEventSequence".to_string(),
                        Value::from(result.last_event_sequence as f64),
                    );
                    self.write(client, &DaemonOutbound::Raw(Value::Object(object)));
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "attach",
                    Some(serde_json::to_value(result).unwrap_or(Value::Null)),
                )))
            }
            "detach" => {
                match body.get("activeSessionId").and_then(Value::as_str) {
                    Some(active_session_id) => {
                        let state = self.get_session_state(active_session_id)?;
                        self.release_session_input_pauses_for(client, Some(active_session_id));
                        let mut state_guard = state.lock().expect("active session poisoned");
                        detach_client_from_active_session(client, &mut state_guard);
                    }
                    None => {
                        self.release_session_input_pauses_for(client, None);
                        self.detach_client(client);
                    }
                }
                Ok(Some(DaemonResponse::success(id, "detach", None)))
            }
            "kill" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                self.close_session(state, CLOSING_REASON_KILLED, false, false, None, None)
                    .await;
                Ok(Some(DaemonResponse::success(id, "kill", None)))
            }
            "rename" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                let name = body
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if name.is_empty() {
                    return Err("Session name cannot be empty".to_string());
                }
                self.set_state_session_name_for_command(&state, &name)
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "rename",
                    Some(
                        serde_json::to_value(self.summary_for_state(&state))
                            .unwrap_or(Value::Null),
                    ),
                )))
            }
            "rename_saved_session" => {
                if let Some(active_session_id) = body.get("activeSessionId").and_then(Value::as_str)
                {
                    self.get_session_state(active_session_id)?;
                }
                let session_path = body
                    .get("sessionPath")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "sessionPath is required".to_string())?;
                let state = self.find_active_session_by_file(session_path);
                let name = body
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if name.is_empty() {
                    return Err("Session name cannot be empty".to_string());
                }
                if let Some(state) = state {
                    self.set_state_session_name_for_command(&state, &name)
                        .await?;
                } else {
                    let info = read_session_info(session_path)
                        .await
                        .ok_or_else(|| format!("Session not found: {session_path}"))?;
                    let depth = info.rlm_depth;
                    let parent_session_path = (depth > 0)
                        .then(|| info.parent_session_path.clone())
                        .flatten();
                    let reservation = NameReservationInput {
                        name: name.clone(),
                        depth: depth as f64,
                        parent_session_id: None,
                        parent_session_path: parent_session_path.clone(),
                    };
                    let name_for_update = name.clone();
                    let path_for_update = session_path.to_string();
                    let info_id = info.id.clone();
                    self.with_session_name_reservation(reservation, move |daemon| {
                            Box::pin(async move {
                                let availability = AgentSessionNameAvailabilityInput {
                                    name: name_for_update.clone(),
                                    depth: depth as f64,
                                    parent_session_id: None,
                                    parent_session_path: parent_session_path.clone(),
                                    ignore_session_id: Some(info_id),
                                };
                                daemon
                                    .assert_family_session_name_available(
                                        &availability,
                                        None,
                                        true,
                                    )
                                    .await?;
                                SessionManager::open(&path_for_update, None, None)
                                    .map_err(|error| error.to_string())?
                                    .append_session_info(&name_for_update);
                                if let Err(error) = daemon
                                    .rlm_spawn_ledger()
                                    .append_rename_by_child_path(&path_for_update, &name_for_update)
                                    .await
                                {
                                    daemon.log(&format!(
                                        "failed to append RLM ledger rename: {error}"
                                    ));
                                }
                                Ok(())
                            })
                        })
                        .await?;
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "rename_saved_session",
                    None,
                )))
            }
            "delete_saved_session" => {
                if let Some(active_session_id) = body.get("activeSessionId").and_then(Value::as_str)
                {
                    self.get_session_state(active_session_id)?;
                }
                let session_path = body
                    .get("sessionPath")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "sessionPath is required".to_string())?;
                if self.find_active_session_by_file(session_path).is_some() {
                    return Err("Cannot delete the currently active session".to_string());
                }
                let composed_entry =
                    self.roster_entry_for_session_path(&canonical_session_path(session_path));
                let composed_runtime_kind = composed_entry
                    .as_ref()
                    .map(|entry| entry.summary.runtime_kind.clone())
                    .unwrap_or(None);
                let tombstone = tombstone_saved_session_delete(
                    &self.rlm_spawn_ledger(),
                    session_path,
                    composed_runtime_kind.as_deref(),
                )
                .await;
                let removed_path = session_path.to_string();
                let daemon_for_cancel = Arc::clone(self);
                let result = self
                    .delete_saved_session_file(
                        session_path,
                        Some(DeleteSessionFileOptions {
                            after_file_removed: Some(Box::new(move || {
                                daemon_for_cancel
                                    .cancel_scheduled_jobs_for_session_file(&removed_path);
                            })),
                        }),
                    );
                if result.is_ok() && self.is_worker() {
                    let removed_agent_id = composed_entry
                        .as_ref()
                        .map(|entry| entry.agent_id.clone())
                        .or_else(|| {
                            tombstone
                                .ledger_edge
                                .as_ref()
                                .map(|edge| {
                                    self.roster_agent_id_for_rlm_child(
                                        &edge.child_id,
                                        Some(&edge.parent),
                                    )
                                })
                                .or_else(|| {
                                    tombstone.deleted_info.as_ref().map(|info| info.id.clone())
                                })
                        });
                    if let Some(removed_agent_id) = removed_agent_id {
                        let removed_session_id = composed_entry
                            .as_ref()
                            .map(|entry| entry.summary.id.clone())
                            .or_else(|| {
                                tombstone.deleted_info.as_ref().map(|info| info.id.clone())
                            });
                        self.roster_reporter
                            .lock()
                            .expect("roster reporter poisoned")
                            .removed_agent_ids
                            .insert(removed_agent_id, removed_session_id);
                        self.schedule_roster_flush();
                    }
                }
                // REPAIR CURSOR: `DeleteSessionFileResult` (core/session_file_actions.rs:14)
                // carries no serde derives and that file is not in this pack's file list.
                // Fix: derive `Serialize` with the TypeScript discriminated-union shape
                // (`{ ok: true, method } | { ok: false, error }`, session-file-actions.ts:7)
                // and replace this projection.
                let result_value = match &result {
                    DeleteSessionFileResult::Ok { method } => serde_json::json!({
                        "ok": true,
                        "method": match method {
                            crate::core::session_file_actions::DeleteSessionFileMethod::Trash => "trash",
                            crate::core::session_file_actions::DeleteSessionFileMethod::Unlink => "unlink",
                        },
                    }),
                    DeleteSessionFileResult::Error { error } => {
                        serde_json::json!({ "ok": false, "error": error })
                    }
                };
                Ok(Some(DaemonResponse::success(
                    id,
                    "delete_saved_session",
                    Some(result_value),
                )))
            }
            "cancel_prompt_admission" => {
                let key = self.prompt_admission_key(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    body.get("admissionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                let admission = self
                    .prompt_admissions
                    .lock()
                    .expect("prompt admissions poisoned")
                    .get(&key)
                    .map(|admission| (admission.status.clone(), admission.controller.clone()));
                let Some((status, controller)) = admission else {
                    return Ok(Some(DaemonResponse::success(
                        id,
                        &command.type_,
                        Some(serde_json::json!({ "status": "unknown" })),
                    )));
                };
                if status == "owned" {
                    if body.get("cancelOwned").and_then(Value::as_bool) == Some(true) {
                        if let Some(controller) = controller {
                            controller.cancel();
                        }
                    }
                    return Ok(Some(DaemonResponse::success(
                        id,
                        &command.type_,
                        Some(serde_json::json!({ "status": "owned" })),
                    )));
                }
                if status == "waiting" {
                    let mut admissions = self
                        .prompt_admissions
                        .lock()
                        .expect("prompt admissions poisoned");
                    if let Some(admission) = admissions.get_mut(&key) {
                        admission.status = "cancelled".to_string();
                    }
                    drop(admissions);
                    if let Some(controller) = controller {
                        controller.cancel();
                    }
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    &command.type_,
                    Some(serde_json::json!({ "status": "cancelled" })),
                )))
            }
            "prompt" | "prompt_and_wait" => {
                on_prompt_handler_owns_admission();
                self.handle_prompt_command(client, command).await
            }
            "steer" => {
                let state = self.get_bound_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                let session = self.session_of(&state);
                let message = body
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let images = body.get("images").cloned();
                let options = PromptInvocation {
                    queue_key: body
                        .get("queueKey")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    agent_message_id: body
                        .get("agentMessageId")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    content: body.get("content").cloned(),
                    custom_message: body.get("customMessage").cloned(),
                    prefix_messages: body.get("prefixMessages").cloned(),
                    resume_if_idle: Some(true),
                    ..PromptInvocation::default()
                };
                if body.get("expandPromptTemplates").and_then(Value::as_bool) == Some(false) {
                    session
                        .restore_steering_message(&message, images, options)
                        .await
                        .map_err(|error| error)?;
                } else {
                    session
                        .steer(&message, images, options)
                        .await
                        .map_err(|error| error)?;
                }
                self.record_worker_recovery_state(&state, "steer_queued", Some(true));
                Ok(Some(DaemonResponse::success(id, "steer", None)))
            }
            "follow_up" => {
                let state = self.get_bound_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                let session = self.session_of(&state);
                let message = body
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let images = body.get("images").cloned();
                let options = PromptInvocation {
                    queue_key: body
                        .get("queueKey")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    agent_message_id: body
                        .get("agentMessageId")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    content: body.get("content").cloned(),
                    custom_message: body.get("customMessage").cloned(),
                    prefix_messages: body.get("prefixMessages").cloned(),
                    resume_if_idle: Some(true),
                    ..PromptInvocation::default()
                };
                let queued =
                    if body.get("expandPromptTemplates").and_then(Value::as_bool) == Some(false) {
                        session
                            .restore_follow_up_message(&message, images, options)
                            .await?
                    } else {
                        session.follow_up(&message, images, options).await?
                    };
                let admitted = queued;
                if admitted {
                    self.record_worker_recovery_state(&state, "follow_up_queued", Some(true));
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "follow_up",
                    Some(serde_json::json!({ "queued": queued })),
                )))
            }
            "restore_next_turn" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                self.session_of(&state).restore_pending_next_turn_messages(
                    body.get("messages").unwrap_or(&Value::Null),
                );
                Ok(Some(DaemonResponse::success(id, "restore_next_turn", None)))
            }
            "restore_actions" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                let restored = self
                    .session_of(&state)
                    .restore_session_actions(body.get("snapshot").unwrap_or(&Value::Null))
                    .await?;
                if restored > 0.0 {
                    self.record_worker_recovery_state(&state, "actions_restored", Some(true));
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "restore_actions",
                    Some(serde_json::json!({ "restored": restored })),
                )))
            }
            "append_custom_message" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                self.session_of(&state)
                    .send_custom_message(body.get("message").unwrap_or(&Value::Null))
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "append_custom_message",
                    None,
                )))
            }
            "resume_queue" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                if !self.session_of(&state).resume_queued_work() {
                    let error = "No queued work to resume".to_string();
                    return Ok(Some(DaemonResponse::failure(
                        id,
                        "resume_queue",
                        &error.clone(),
                        serialize_daemon_error(&DaemonError::Message(error)),
                    )));
                }
                Ok(Some(DaemonResponse::success(id, "resume_queue", None)))
            }
            "send_message" => {
                let from_state = match body.get("fromActiveSessionId").and_then(Value::as_str) {
                    Some(value) => Some(self.get_session_state(value)?),
                    None => None,
                };
                if body.get("agentOrigin").and_then(Value::as_bool) == Some(true)
                    && from_state.is_none()
                {
                    return Err("Agent messaging requires fromActiveSessionId".into());
                }
                let receipt = self
                    .send_agent_session_message(SendAgentMessageInput {
                        target_selector: body
                            .get("targetActiveSessionId")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        message: body
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        from_state,
                        sender: None,
                        client_id: Some(client.id()),
                        sender_key: Some(self.create_cli_agent_message_sender_key()),
                        origin: if body.get("agentOrigin").and_then(Value::as_bool) == Some(true) {
                            "agent".to_string()
                        } else {
                            "cli".to_string()
                        },
                    })
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "send_message",
                    Some(serde_json::to_value(receipt).unwrap_or(Value::Null)),
                )))
            }
            "agent_messages_status" => Ok(Some(DaemonResponse::success(
                id,
                "agent_messages_status",
                Some(self.get_agent_message_safety_status()),
            ))),
            "agent_messages_pause" => {
                self.agent_messages_paused.store(true, Ordering::SeqCst);
                self.agent_message_rate_limiter
                    .lock()
                    .expect("agent message rate limiter poisoned")
                    .clear(None);
                self.clear_queued_agent_session_messages_for_all_states()
                    .await;
                Ok(Some(DaemonResponse::success(
                    id,
                    "agent_messages_pause",
                    Some(self.get_agent_message_safety_status()),
                )))
            }
            "agent_messages_resume" => {
                self.agent_messages_paused.store(false, Ordering::SeqCst);
                Ok(Some(DaemonResponse::success(
                    id,
                    "agent_messages_resume",
                    Some(self.get_agent_message_safety_status()),
                )))
            }
            // SHARED FILE EDIT (daemon_mode.rs, capability-gated addition by jev-ui
            // lane): the three optional Jev commands. They read and write the same
            // `pi-jev` settings store the interactive `/jev` command uses (lane A's
            // `JevSettingsStore`), so a client and a daemon never disagree.
            "jev_get_settings" | "jev_get_status" => {
                let session_id = self.jev_session_id(
                    body.get("activeSessionId").and_then(Value::as_str).unwrap_or(""),
                )?;
                let mut view = self.jev_settings_view(&session_id)?;
                if command.type_ == "jev_get_status" {
                    view["pipeline"] = crate::core::jev_bridge::session_status_snapshot(&session_id)
                        .unwrap_or(Value::Null);
                }
                Ok(Some(DaemonResponse::success(id, &command.type_, Some(view))))
            }
            "jev_set_session_mode" => {
                let requested = body.get("mode").and_then(Value::as_str).unwrap_or("");
                let Some(mode) = pi_jev::types::JevMode::parse(requested) else {
                    return Err(format!(
                        "Unknown Jev mode: {requested} (expected off, compare, active, or compare-active)"
                    ));
                };
                let session_id = self.jev_session_id(body
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .unwrap_or(""))?;
                let (applied, effective, emergency_exit) =
                    self.jev_apply_session_mode(&session_id, mode)?;
                // The footer must reflect the new effective mode in the same
                // turn as the write, exactly like the interactive `/jev` path.
                let selector = body
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Ok(bound) = self.get_bound_session_state(selector) {
                    self.publish_jev_attach_footer(client, &bound);
                }
                let store = self.jev_settings_store();
                let resolution = store.load().effective_mode_with_scope(&session_id);
                Ok(Some(DaemonResponse::success(
                    id,
                    "jev_set_session_mode",
                    Some(serde_json::json!({
                        "requested": requested,
                        "mode": effective.as_str(),
                        "scope": resolution.scope.as_str(),
                        // Every mode, Active included, is applied on request.
                        "applied": applied,
                        "message": if emergency_exit {
                            format!(
                                "Jev mode: {} (scope: this chat)\n{}",
                                effective.label(),
                                crate::modes::interactive::native_host::JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE
                            )
                        } else {
                            jev_mode_change_message(applied, effective)
                        },
                    })),
                )))
            }
            "agent_messages_clear" => {
                let state = self.get_session_state(
                    body.get("activeSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                let active_session_id = state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                let suffix = format!("->{active_session_id}");
                self.agent_message_rate_limiter
                    .lock()
                    .expect("agent message rate limiter poisoned")
                    .clear_matching(|key: &str| key.ends_with(&suffix));
                let cleared = self.session_of(&state).clear_queued_agent_messages();
                Ok(Some(DaemonResponse::success(
                    id,
                    "agent_messages_clear",
                    Some(cleared),
                )))
            }
            _ => self.handle_command_rest(client, command).await,
        }
    }
}

impl AgentDaemon {
    /// The second half of the `handleCommand` switch.
    async fn handle_command_rest(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
    ) -> Result<Option<DaemonResponse>, String> {
        let body = &command.body;
        let id = command.id.as_deref();
        let active_session_id = body
            .get("activeSessionId")
            .and_then(Value::as_str)
            .unwrap_or("");
        match command.type_.as_str() {
            "abort" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).request_abort();
                Ok(Some(DaemonResponse::success(id, "abort", None)))
            }
            "start_side_question" => {
                let state = self.get_session_state(active_session_id)?;
                let side_question_id = body
                    .get("sideQuestionId")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let state_active_session_id = state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                if self
                    .side_question_runs
                    .lock()
                    .expect("side question runs poisoned")
                    .contains_key(&side_question_id)
                {
                    return Err(format!("Side question already exists: {side_question_id}"));
                }
                if self.has_active_side_question_for(client, &state_active_session_id) {
                    return Err(
                        "A side question is already running for this client and session"
                            .to_string(),
                    );
                }
                let session = self.session_of(&state);
                let runs = Arc::clone(self);
                let side_question_daemon = Arc::clone(self);
                let run_client = Arc::clone(client);
                let on_event_state = Arc::clone(&state);
                let on_event: Arc<dyn Fn(&Value) + Send + Sync> = Arc::new(move |event: &Value| {
                    let mut object = Map::new();
                    object.insert(
                        "type".to_string(),
                        Value::String("side_question_event".to_string()),
                    );
                    object.insert(
                        "activeSessionId".to_string(),
                        Value::String(
                            on_event_state
                                .lock()
                                .expect("active session poisoned")
                                .active_session_id
                                .clone(),
                        ),
                    );
                    object.insert("event".to_string(), event.clone());
                    let _ = side_question_daemon.write_public_value(
                        &run_client,
                        &Value::Object(object),
                        "side_question_event",
                    );
                    if event.get("status").and_then(Value::as_str) != Some("running") {
                        if let Some(event_id) = event.get("id").and_then(Value::as_str) {
                            runs.side_question_runs
                                .lock()
                                .expect("side question runs poisoned")
                                .remove(event_id);
                        }
                    }
                });
                // Register ownership before the task can emit a terminal event.
                // The callback removes this exact id on completion; disconnect
                // and explicit abort call the live session's cancellation hook.
                let abort_session = session.clone();
                let abort_id = side_question_id.clone();
                self.side_question_runs.lock().expect("side question runs poisoned").insert(
                    side_question_id.clone(), SideQuestionRunEntry {
                        run: Arc::new(move || abort_session.abort_side_question(&abort_id)),
                        client: Arc::clone(client), active_session_id: state_active_session_id,
                    },
                );
                if let Err(error) = session.start_side_question(
                    body.get("question").and_then(Value::as_str).unwrap_or(""),
                    SideQuestionOptions {
                        id: side_question_id.clone(), previous_turns: body.get("previousTurns").cloned(),
                        retry_policy: session.settings_manager().map(|_| Value::Null), on_event: Some(on_event),
                    },
                ).await {
                    self.side_question_runs.lock().expect("side question runs poisoned").remove(&side_question_id);
                    return Err(error);
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "start_side_question",
                    None,
                )))
            }
            "abort_side_question" => {
                let state = self.get_session_state(active_session_id)?;
                let _ = state;
                let side_question_id = body
                    .get("sideQuestionId")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let entry = self
                    .side_question_runs
                    .lock()
                    .expect("side question runs poisoned")
                    .remove(side_question_id);
                let aborted = match entry {
                    Some(entry)
                        if Arc::ptr_eq(&entry.client, client)
                            && entry.active_session_id == active_session_id =>
                    {
                        (entry.run)();
                        true
                    }
                    Some(entry) => {
                        self.side_question_runs
                            .lock()
                            .expect("side question runs poisoned")
                            .insert(side_question_id.to_string(), entry);
                        false
                    }
                    None => false,
                };
                Ok(Some(DaemonResponse::success(
                    id,
                    "abort_side_question",
                    Some(serde_json::json!({ "aborted": aborted })),
                )))
            }
            "execute_bash" => {
                let state = self.get_session_state(active_session_id)?;
                let session = self.session_of(&state);
                if session.is_bash_running() {
                    return Err("A bash command is already running".to_string());
                }
                // Respond before completion (bash can outlive the client request
                // timeout); output and completion stream via bash_* session events.
                let bash = session.run_user_bash(
                    body.get("command").and_then(Value::as_str).unwrap_or(""),
                    RunUserBashOptions {
                        exclude_from_context: body
                            .get("excludeFromContext")
                            .and_then(Value::as_bool),
                        transient: body.get("transient").and_then(Value::as_bool),
                        run_id: body
                            .get("runId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    },
                );
                self.track_in_flight_bash(&state, bash);
                Ok(Some(DaemonResponse::success(id, "execute_bash", None)))
            }
            "execute_bash_and_wait" => {
                let state = self.get_session_state(active_session_id)?;
                let bash = self
                    .session_of(&state)
                    .execute_bash(body.get("command").and_then(Value::as_str).unwrap_or(""));
                let result = {
                    let daemon = self.clone_arc();
                    let tracked_state = Arc::clone(&state);
                    let in_flight = daemon.in_flight_bash_slot(&tracked_state);
                    let result = bash.await;
                    in_flight.notify_waiters();
                    result
                }?;
                self.schedule_roster_flush();
                Ok(Some(DaemonResponse::success(
                    id,
                    "execute_bash_and_wait",
                    Some(result),
                )))
            }
            "abort_bash" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).abort_bash();
                Ok(Some(DaemonResponse::success(id, "abort_bash", None)))
            }
            "cancel_rlm_child" => {
                let state = self.get_session_state(active_session_id)?;
                let child_id = body.get("childId").and_then(Value::as_str).unwrap_or("");
                let cancelled = self.session_of(&state).cancel_rlm_child_run(child_id);
                Ok(Some(DaemonResponse::success(
                    id,
                    "cancel_rlm_child",
                    Some(serde_json::json!({ "cancelled": cancelled })),
                )))
            }
            "delete_rlm_subagent" => {
                let state = self.get_session_state(active_session_id)?;
                let child_id = body
                    .get("childId")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let sessions = self.sessions.lock().expect("sessions poisoned").clone();
                let child_id_for_check = child_id.clone();
                let is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync> =
                    Arc::new(move || {
                        sessions.values().any(|candidate| {
                            let candidate = candidate.state.lock().expect("active session poisoned");
                            let metadata = candidate.runtime.metadata.as_ref();
                            if metadata.and_then(|value| value.kind.as_deref()) != Some("subagent")
                                || metadata.and_then(|value| value.rlm_child_id.as_deref())
                                    != Some(child_id_for_check.as_str())
                            {
                                return false;
                            }
                            // REPAIR CURSOR: `ActiveSessionRuntimeSession`
                            // (active_session_state.rs:31) has no `unfinishedActionCount`;
                            // that file is not in this pack's file list. Fix: add
                            // `pub unfinished_action_count: f64` there, populate it in
                            // `DaemonSessionState::sync_view` from
                            // `session.unfinished_action_count()`, and read the field here
                            // (daemon-mode.ts:4785).
                            candidate.runtime.session.is_streaming
                        })
                    });
                let result = if is_resident_child_running() {
                    "running".to_string()
                } else {
                    self.session_of(&state)
                        .delete_inactive_rlm_subagent(
                            &child_id,
                            Arc::clone(&is_resident_child_running),
                        )
                        .await?
                };
                let mut data = Map::new();
                data.insert("deleted".to_string(), Value::Bool(result == "deleted"));
                if result == "running" {
                    data.insert("reason".to_string(), Value::String("running".to_string()));
                }
                Ok(Some(DaemonResponse::success(
                    id,
                    "delete_rlm_subagent",
                    Some(Value::Object(data)),
                )))
            }
            "acquire_session_input_pause" => {
                let lease_key = body
                    .get("leaseKey")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let existing = self
                    .session_input_pauses
                    .lock()
                    .expect("session input pauses poisoned")
                    .iter()
                    .find(|(_, entry)| {
                        Arc::ptr_eq(&entry.owner, client)
                            && entry.active_session_id == active_session_id
                            && entry.lease_key == lease_key
                    })
                    .map(|(pause_id, _)| pause_id.clone());
                if let Some(pause_id) = existing {
                    return Ok(Some(DaemonResponse::success(
                        id,
                        "acquire_session_input_pause",
                        Some(serde_json::json!({ "pauseId": pause_id })),
                    )));
                }
                let state = self.get_session_state(active_session_id)?;
                let pause_id = uuid::Uuid::new_v4().to_string();
                let pause = self.session_of(&state).acquire_session_input_pause();
                self.session_input_pauses
                    .lock()
                    .expect("session input pauses poisoned")
                    .insert(
                        pause_id.clone(),
                        SessionInputPauseEntry {
                            active_session_id: active_session_id.to_string(),
                            owner: Arc::clone(client),
                            lease_key,
                            pause,
                        },
                    );
                Ok(Some(DaemonResponse::success(
                    id,
                    "acquire_session_input_pause",
                    Some(serde_json::json!({ "pauseId": pause_id })),
                )))
            }
            "release_session_input_pause" => {
                let pause_id = body.get("pauseId").and_then(Value::as_str).unwrap_or("");
                let entry = self
                    .session_input_pauses
                    .lock()
                    .expect("session input pauses poisoned")
                    .remove(pause_id);
                let Some(entry) = entry else {
                    return Ok(Some(DaemonResponse::success(
                        id,
                        "release_session_input_pause",
                        None,
                    )));
                };
                if !Arc::ptr_eq(&entry.owner, client)
                    || entry.active_session_id != active_session_id
                {
                    self.session_input_pauses
                        .lock()
                        .expect("session input pauses poisoned")
                        .insert(pause_id.to_string(), entry);
                    return Err(format!(
                        "Session input pause is owned by another client: {pause_id}"
                    ));
                }
                (entry.pause)();
                Ok(Some(DaemonResponse::success(
                    id,
                    "release_session_input_pause",
                    None,
                )))
            }
            "wait_for_idle" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).wait_for_idle().await;
                Ok(Some(DaemonResponse::success(id, "wait_for_idle", None)))
            }
            "wait_for_headless_completion" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .session_of(&state)
                    .wait_for_headless_completion(HeadlessCompletionOptions {
                        wait_for_rlm_quiescence: body
                            .get("waitForRlmQuiescence")
                            .and_then(Value::as_bool),
                    })
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "wait_for_headless_completion",
                    Some(result),
                )))
            }
            "get_session_header" => {
                let state = self.get_session_state(active_session_id)?;
                let header = self
                    .session_of(&state)
                    .session_manager()
                    .lock()
                    .expect("session manager poisoned")
                    .get_header();
                let header = serde_json::to_value(header).unwrap_or(Value::Null);
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_session_header",
                    Some(serde_json::json!({ "header": header })),
                )))
            }
            "get_state" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_state",
                    Some(
                        serde_json::to_value(self.summary_for_state(&state))
                            .unwrap_or(Value::Null),
                    ),
                )))
            }
            "get_connection_state" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_connection_state",
                    Some(self.create_connection_state(&state)),
                )))
            }
            "get_messages" => {
                let state = self.get_session_state(active_session_id)?;
                let messages = self
                    .session_of(&state)
                    .messages()
                    .into_iter()
                    .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
                    .collect::<Vec<_>>();
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_messages",
                    Some(serde_json::json!({ "messages": messages })),
                )))
            }
            "get_history_range" => {
                let state = self.get_session_state(active_session_id)?;
                let generation = state
                    .lock()
                    .expect("active session poisoned")
                    .event_generation
                    .clone();
                if body.get("generation").and_then(Value::as_str) != Some(generation.as_str()) {
                    return Err("Session history snapshot generation is stale".to_string());
                }
                let session = self.session_of(&state);
                let target_model = session.model_identity();
                let representation = session_history_representation(target_model.as_ref());
                if body.get("representation").and_then(Value::as_str)
                    != Some(representation.as_str())
                {
                    return Err("Session history target-model representation is stale".to_string());
                }
                let manager = session.session_manager();
                let history = manager
                    .lock()
                    .expect("session manager poisoned")
                    .build_session_history(
                        body.get("tipEntryId").and_then(Value::as_str),
                        target_model.as_ref(),
                    )?;
                let range = slice_pinned_session_history(
                    &history,
                    &SlicePinnedSessionHistoryOptions {
                        generation,
                        representation,
                        before_entry_id: body
                            .get("beforeEntryId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        limit: body
                            .get("limit")
                            .and_then(Value::as_f64)
                            .map(|value| value as usize),
                    },
                )?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_history_range",
                    Some(serde_json::to_value(range).unwrap_or(Value::Null)),
                )))
            }
            "get_rlm_children" => {
                let state = self.get_session_state(active_session_id)?;
                let event_sequence = state
                    .lock()
                    .expect("active session poisoned")
                    .last_event_sequence;
                let children = self.build_rlm_child_snapshots_with_passive(&state).await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_rlm_children",
                    Some(serde_json::json!({
                        "children": children,
                        "eventSequence": event_sequence,
                    })),
                )))
            }
            "get_session_stats" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_session_stats",
                    Some(self.session_of(&state).get_session_stats()),
                )))
            }
            "get_context_tree" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_context_tree",
                    Some(self.session_of(&state).get_context_tree()),
                )))
            }
            "get_commands" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_commands",
                    Some(serde_json::json!({
                        "commands": self.connection_commands(&state)
                    })),
                )))
            }
            "get_resource_snapshot" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_resource_snapshot",
                    Some(self.connection_resource_snapshot(&state)),
                )))
            }
            "replace_acp_mcp_servers" => self.handle_replace_acp_mcp_servers(client, command).await,
            "get_available_models" => {
                let state = self.get_session_state(active_session_id)?;
                let models = self.session_of(&state).refresh_available_models().await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_available_models",
                    Some(serde_json::json!({ "models": models })),
                )))
            }
            "get_model_catalog" => {
                let state = self.get_session_state(active_session_id)?;
                let catalog = self.session_of(&state).refresh_model_catalog().await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_model_catalog",
                    Some(catalog),
                )))
            }
            "get_queue" => {
                let state = self.get_session_state(active_session_id)?;
                let session = self.session_of(&state);
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_queue",
                    Some(serde_json::json!({
                        "steering": session.get_steering_message_previews(),
                        "followUp": session.get_follow_up_message_previews(),
                    })),
                )))
            }
            "mutate_queued_message" => {
                let state = self.get_session_state(active_session_id)?;
                let status = self.session_of(&state).mutate_queued_message(
                    body.get("lane").and_then(Value::as_str).unwrap_or(""),
                    body.get("index").and_then(Value::as_f64).unwrap_or(0.0),
                    body.get("expectedText")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    body.get("mutation").unwrap_or(&Value::Null),
                );
                Ok(Some(DaemonResponse::success(
                    id,
                    "mutate_queued_message",
                    Some(serde_json::json!({ "status": status })),
                )))
            }
            "clear_queue" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "clear_queue",
                    Some(self.session_of(&state).clear_queue()),
                )))
            }
            "abort_and_clear_queue" => {
                let state = self.get_session_state(active_session_id)?;
                let session = self.session_of(&state);
                let queue = session.clear_queue();
                session.request_abort();
                Ok(Some(DaemonResponse::success(
                    id,
                    "abort_and_clear_queue",
                    Some(queue),
                )))
            }
            "cron_list" => {
                let include_inactive =
                    body.get("includeInactive").and_then(Value::as_bool) == Some(true);
                let filter_session_id = body
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let jobs: Vec<Value> = self
                    .cron_store
                    .list()
                    .into_iter()
                    .filter(|job| {
                        if !include_inactive && job.status != "active" && job.status != "paused" {
                            return false;
                        }
                        if let Some(filter_session_id) = &filter_session_id {
                            if job.active_session_id != *filter_session_id {
                                return false;
                            }
                        }
                        true
                    })
                    .map(|job| serde_json::to_value(job).unwrap_or(Value::Null))
                    .collect();
                Ok(Some(DaemonResponse::success(
                    id,
                    "cron_list",
                    Some(serde_json::json!({ "jobs": jobs })),
                )))
            }
            "heartbeats_list" => Ok(Some(DaemonResponse::success(
                id,
                "heartbeats_list",
                Some(serde_json::json!({ "heartbeats": self.list_heartbeats() })),
            ))),
            "heartbeat_manage" => {
                let heartbeat = self.manage_heartbeat(
                    active_session_id,
                    body.get("jobId").and_then(Value::as_str).unwrap_or(""),
                    body.get("action").and_then(Value::as_str).unwrap_or(""),
                );
                let Ok(Some(heartbeat)) = heartbeat else {
                    return Err(format!(
                        "No active heartbeat found: {}",
                        body.get("jobId").and_then(Value::as_str).unwrap_or("")
                    ));
                };
                Ok(Some(DaemonResponse::success(
                    id,
                    "heartbeat_manage",
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                )))
            }
            "cron_add" => {
                let state = self.get_session_state(active_session_id)?;
                let job = self.create_cron_job_for_state(
                    &state,
                    body.get("schedule").and_then(Value::as_str).unwrap_or(""),
                    body.get("prompt").and_then(Value::as_str).unwrap_or(""),
                )?;
                self.schedule_roster_flush();
                Ok(Some(DaemonResponse::success(
                    id,
                    "cron_add",
                    Some(serde_json::json!({ "job": job })),
                )))
            }
            "cron_cancel" => {
                let job_id = body.get("jobId").and_then(Value::as_str).unwrap_or("");
                let job = self.cron_store.cancel(job_id, now_millis());
                let Some(job) = job else {
                    return Err(format!("No cron job found: {job_id}"));
                };
                let state = self
                    .sessions
                    .lock()
                    .expect("sessions poisoned")
                    .get(&job.active_session_id)
                    .cloned();
                if let Some(state) = state {
                    self.remove_queued_heartbeat_follow_up(&state.state, &job);
                }
                self.cron_scheduler_wake();
                self.schedule_roster_flush();
                Ok(Some(DaemonResponse::success(
                    id,
                    "cron_cancel",
                    Some(serde_json::json!({ "job": job_to_value(&job) })),
                )))
            }
            "heartbeat_get" => {
                let state = self.get_session_state(active_session_id)?;
                let active_session_id = state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                let heartbeat = self
                    .cron_store
                    .get_heartbeat(&active_session_id)
                    .map(|job| job_to_value(&job))
                    .unwrap_or(Value::Null);
                Ok(Some(DaemonResponse::success(
                    id,
                    "heartbeat_get",
                    Some(serde_json::json!({ "heartbeat": heartbeat })),
                )))
            }
            "heartbeat_set" => {
                let state = self.get_session_state(active_session_id)?;
                let delivery_mode =
                    normalize_heartbeat_delivery_mode(body.get("deliveryMode")).unwrap_or(None);
                let heartbeat = self.create_heartbeat_for_state(
                    &state,
                    body.get("schedule").and_then(Value::as_str).unwrap_or(""),
                    body.get("prompt").and_then(Value::as_str).unwrap_or(""),
                    delivery_mode,
                )?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "heartbeat_set",
                    Some(serde_json::json!({ "heartbeat": job_to_value(&heartbeat) })),
                )))
            }
            "heartbeat_update" => {
                let state = self.get_session_state(active_session_id)?;
                let heartbeat = self.update_heartbeat_for_state(
                    &state,
                    body.get("action").and_then(Value::as_str).unwrap_or(""),
                )?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "heartbeat_update",
                    Some(serde_json::json!({
                        "heartbeat": heartbeat.map(|job| job_to_value(&job)).unwrap_or(Value::Null)
                    })),
                )))
            }
            "set_model" => {
                let state = self.get_session_state(active_session_id)?;
                let session = self.session_of(&state);
                let provider = body.get("provider").and_then(Value::as_str).unwrap_or("");
                let model_id = body.get("modelId").and_then(Value::as_str).unwrap_or("");
                let available_models = session.refresh_available_models().await?;
                let model = available_models
                    .into_iter()
                    .find(|candidate| candidate.provider == provider && candidate.id == model_id)
                    // Stale-auth providers are excluded from the available list; the
                    // lookup never mutates stale state (session.setModel owns the clear).
                    .or_else(|| {
                        (session.get_provider_auth_status_source(provider).as_deref()
                            == Some("stale"))
                        .then(|| session.find_model(provider, model_id))
                        .flatten()
                    });
                let Some(model) = model else {
                    return Err(format!("Model not found: {provider}/{model_id}"));
                };
                session
                    .set_model(&model, !(session.is_streaming() || session.is_compacting()))
                    .await?;
                self.schedule_roster_flush();
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_model",
                    Some(serde_json::to_value(&model).unwrap_or(Value::Null)),
                )))
            }
            "cycle_model" => {
                let state = self.get_session_state(active_session_id)?;
                let session = self.session_of(&state);
                let result = session
                    .cycle_model(
                        body.get("direction").and_then(Value::as_str).unwrap_or(""),
                        !(session.is_streaming() || session.is_compacting()),
                    )
                    .await?;
                self.schedule_roster_flush();
                Ok(Some(DaemonResponse::success(
                    id,
                    "cycle_model",
                    Some(
                        result
                            .and_then(|model| serde_json::to_value(model).ok())
                            .unwrap_or(Value::Null),
                    ),
                )))
            }
            "set_scoped_models" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state)
                    .set_scoped_models(body.get("scopedModels").unwrap_or(&Value::Null));
                Ok(Some(DaemonResponse::success(id, "set_scoped_models", None)))
            }
            "set_thinking_level" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state)
                    .set_thinking_level(body.get("level").and_then(Value::as_str).unwrap_or(""));
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_thinking_level",
                    None,
                )))
            }
            "set_service_tier" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).set_service_tier(
                    body.get("serviceTier")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                Ok(Some(DaemonResponse::success(id, "set_service_tier", None)))
            }
            "cycle_thinking_level" => {
                let state = self.get_session_state(active_session_id)?;
                let level = self.session_of(&state).cycle_thinking_level();
                Ok(Some(DaemonResponse::success(
                    id,
                    "cycle_thinking_level",
                    Some(
                        level
                            .map(|level| serde_json::json!({ "level": level }))
                            .unwrap_or(Value::Null),
                    ),
                )))
            }
            "set_transport" => {
                let state = self.get_session_state(active_session_id)?;
                let transport = body.get("transport").and_then(Value::as_str).unwrap_or("");
                if let Some(settings) = self.session_of(&state).settings_manager() {
                    settings
                        .lock()
                        .expect("settings manager poisoned")
                        .set_transport(transport.to_string());
                }
                self.session_of(&state).set_transport(transport);
                Ok(Some(DaemonResponse::success(id, "set_transport", None)))
            }
            "set_steering_mode" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state)
                    .set_steering_mode(body.get("mode").and_then(Value::as_str).unwrap_or(""));
                Ok(Some(DaemonResponse::success(id, "set_steering_mode", None)))
            }
            "set_follow_up_mode" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state)
                    .set_follow_up_mode(body.get("mode").and_then(Value::as_str).unwrap_or(""));
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_follow_up_mode",
                    None,
                )))
            }
            "set_auto_compaction" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).set_auto_compaction_enabled(
                    body.get("enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                );
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_auto_compaction",
                    None,
                )))
            }
            "set_auto_retry" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).set_auto_retry_enabled(
                    body.get("enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                );
                Ok(Some(DaemonResponse::success(id, "set_auto_retry", None)))
            }
            "compact" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .session_of(&state)
                    .compact(body.get("customInstructions").and_then(Value::as_str))
                    .await?;
                Ok(Some(DaemonResponse::success(id, "compact", Some(result))))
            }
            "refine" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .session_of(&state)
                    .refine(RefineOptions {
                        instructions: body
                            .get("instructions")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        rollback_id: body
                            .get("rollbackId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        global: body.get("global").and_then(Value::as_bool),
                    })
                    .await?;
                Ok(Some(DaemonResponse::success(id, "refine", Some(result))))
            }
            "abort_compaction" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).abort_compaction();
                Ok(Some(DaemonResponse::success(id, "abort_compaction", None)))
            }
            "abort_branch_summary" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).abort_branch_summary();
                Ok(Some(DaemonResponse::success(
                    id,
                    "abort_branch_summary",
                    None,
                )))
            }
            "abort_retry" => {
                let state = self.get_session_state(active_session_id)?;
                self.session_of(&state).abort_retry();
                Ok(Some(DaemonResponse::success(id, "abort_retry", None)))
            }
            "reload" => {
                let state = self.get_session_state(active_session_id)?;
                // Reload re-evaluates extension modules, which capture client env
                // (e.g. herdr pane identity) synchronously at load.
                let session = self.session_of(&state);
                let client_env = state
                    .lock()
                    .expect("active session poisoned")
                    .client_env
                    .clone();
                with_client_env(client_env.as_ref(), || session.reload()).await?;
                Ok(Some(DaemonResponse::success(id, "reload", None)))
            }
            "new_session" => {
                let state = self.get_session_state(active_session_id)?;
                let options = body.get("parentSession").cloned().map(|parent_session| {
                    NewSessionRuntimeOptions {
                        parent_session: Some(parent_session),
                    }
                });
                let result = self.runtime_new_session(&state, options).await?;
                self.rebind_cron_jobs_to_state(&state);
                Ok(Some(DaemonResponse::success(
                    id,
                    "new_session",
                    Some(result),
                )))
            }
            "switch_session" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .runtime_switch_session(
                        &state,
                        body.get("sessionPath")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        SessionPathOptions {
                            cwd_override: body
                                .get("cwdOverride")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        },
                    )
                    .await?;
                self.rebind_cron_jobs_to_state(&state);
                Ok(Some(DaemonResponse::success(
                    id,
                    "switch_session",
                    Some(result),
                )))
            }
            "fork" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .runtime_fork(
                        &state,
                        body.get("entryId").and_then(Value::as_str).unwrap_or(""),
                        ForkOptions {
                            position: body
                                .get("position")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        },
                    )
                    .await?;
                self.rebind_cron_jobs_to_state(&state);
                Ok(Some(DaemonResponse::success(id, "fork", Some(result))))
            }
            "navigate_tree" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .session_of(&state)
                    .navigate_tree(
                        body.get("targetId").and_then(Value::as_str).unwrap_or(""),
                        NavigateTreeOptions {
                            summarize: body.get("summarize").and_then(Value::as_bool),
                            custom_instructions: body
                                .get("customInstructions")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                            replace_instructions: body
                                .get("replaceInstructions")
                                .and_then(Value::as_bool),
                            label: body
                                .get("label")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        },
                    )
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "navigate_tree",
                    Some(result),
                )))
            }
            "import_jsonl" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .runtime_import_from_jsonl(
                        &state,
                        body.get("inputPath").and_then(Value::as_str).unwrap_or(""),
                        body.get("cwdOverride")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    )
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "import_jsonl",
                    Some(result),
                )))
            }
            "export_html" => {
                let state = self.get_session_state(active_session_id)?;
                let path = self
                    .session_of(&state)
                    .export_to_html(body.get("outputPath").and_then(Value::as_str))
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "export_html",
                    Some(serde_json::json!({ "path": path })),
                )))
            }
            "export_jsonl" => {
                let state = self.get_session_state(active_session_id)?;
                let path = self
                    .session_of(&state)
                    .export_to_jsonl(body.get("outputPath").and_then(Value::as_str))?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "export_jsonl",
                    Some(serde_json::json!({ "path": path })),
                )))
            }
            "set_session_name" => {
                let state = self.get_session_state(active_session_id)?;
                let name = body
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if name.is_empty() {
                    return Err("Session name cannot be empty".to_string());
                }
                self.set_state_session_name_for_command(&state, &name)
                    .await?;
                Ok(Some(DaemonResponse::success(id, "set_session_name", None)))
            }
            "get_rlm_max_depth_status" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_rlm_max_depth_status",
                    Some(self.session_of(&state).get_rlm_max_depth_status()),
                )))
            }
            "set_rlm_max_depth" => {
                let state = self.get_session_state(active_session_id)?;
                let result = self
                    .session_of(&state)
                    .set_rlm_max_depth(
                        body.get("maxDepth").cloned().unwrap_or(Value::Null),
                        body.get("global").and_then(Value::as_bool).unwrap_or(false),
                    )
                    .await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_rlm_max_depth",
                    Some(result),
                )))
            }
            "get_session_context" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_session_context",
                    Some(
                        serde_json::json!({ "context": self.session_of(&state).build_session_context() }),
                    ),
                )))
            }
            "get_session_tree" => {
                let state = self.get_session_state(active_session_id)?;
                let manager = self.session_of(&state).session_manager();
                let manager = manager.lock().expect("session manager poisoned");
                // REPAIR CURSOR: `SessionTreeFlatNode` (core/session_manager.rs:3119) has no
                // serde derives, and that file is not in this pack's file list. Fix: add
                // `serde::Serialize` (camelCase: `entry`/`label`/`labelTimestamp`) on
                // `SessionTreeFlatNode` and `SessionEntry` and replace this projection.
                // `getFlatTree()` (session-manager.ts:2449) is `{ entry, label, labelTimestamp }`.
                let flat_nodes = Value::Array(
                    manager
                        .get_flat_tree()
                        .into_iter()
                        .map(|node| {
                            let mut object = Map::new();
                            object.insert("entry".to_string(), Value::Object(node.entry));
                            match node.label {
                                Some(label) => {
                                    object.insert("label".to_string(), Value::String(label))
                                }
                                None => object.insert("label".to_string(), Value::Null),
                            };
                            match node.label_timestamp {
                                Some(timestamp) => {
                                    object.insert("labelTimestamp".to_string(), Value::String(timestamp))
                                }
                                None => object.insert("labelTimestamp".to_string(), Value::Null),
                            };
                            Value::Object(object)
                        })
                        .collect(),
                );
                let leaf_id = manager
                    .get_leaf_id()
                    .map(Value::String)
                    .unwrap_or(Value::Null);
                drop(manager);
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_session_tree",
                    Some(serde_json::json!({ "flatNodes": flat_nodes, "leafId": leaf_id })),
                )))
            }
            "get_user_messages_for_forking" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_user_messages_for_forking",
                    Some(serde_json::json!({
                        "messages": self.session_of(&state).get_user_messages_for_forking()
                    })),
                )))
            }
            "get_last_assistant_text" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_last_assistant_text",
                    Some(
                        serde_json::json!({ "text": self.session_of(&state).get_last_assistant_text() }),
                    ),
                )))
            }
            "get_system_prompt" => {
                let state = self.get_session_state(active_session_id)?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_system_prompt",
                    Some(
                        serde_json::json!({ "systemPrompt": self.session_of(&state).system_prompt() }),
                    ),
                )))
            }
            "get_tool_definition" => {
                let state = self.get_session_state(active_session_id)?;
                let definition = self
                    .session_of(&state)
                    .get_tool_definition(body.get("name").and_then(Value::as_str).unwrap_or(""));
                // REPAIR CURSOR: the session seam returns the definition as a `Value` because
                // `core::extensions::types::ToolDefinition` carries no serde derives and that
                // file is not in this pack's file list. Fix: derive `Serialize`/`Deserialize`
                // on `ToolDefinition` and return it directly from `DaemonSession::get_tool_definition`
                // so `create_agent_connection_tool_definition` can take it without a rebuild.
                let tool_definition = definition
                    .as_ref()
                    .map(create_agent_connection_tool_definition_from_value);
                Ok(Some(DaemonResponse::success(
                    id,
                    "get_tool_definition",
                    Some(serde_json::json!({ "toolDefinition": tool_definition })),
                )))
            }
            "set_session_entry_label" => {
                let state = self.get_session_state(active_session_id)?;
                let manager = self.session_of(&state).session_manager();
                let mut manager = manager.lock().expect("session manager poisoned");
                let _ = manager.append_label_change(
                    body.get("entryId").and_then(Value::as_str).unwrap_or(""),
                    body.get("label").and_then(Value::as_str),
                );
                drop(manager);
                Ok(Some(DaemonResponse::success(
                    id,
                    "set_session_entry_label",
                    None,
                )))
            }
            "extension_ui_response" => {
                let state = self.get_session_state(active_session_id)?;
                let request_id = body.get("requestId").and_then(Value::as_str).unwrap_or("");
                let pending = state
                    .lock()
                    .expect("active session poisoned")
                    .extension_ui_requests
                    .remove(request_id);
                let Some(pending) = pending else {
                    return Err(format!("Unknown extension UI request: {request_id}"));
                };
                let response = body.get("response").cloned().unwrap_or(Value::Null);
                let resolved = if let Some(value) = response.get("value").and_then(Value::as_str) {
                    DaemonExtensionUIResponse::Value(value.to_string())
                } else if let Some(confirmed) = response.get("confirmed").and_then(Value::as_bool) {
                    DaemonExtensionUIResponse::Confirmed(confirmed)
                } else {
                    DaemonExtensionUIResponse::Cancelled
                };
                (pending.resolve)(resolved);
                Ok(Some(DaemonResponse::success(
                    id,
                    "extension_ui_response",
                    None,
                )))
            }
            "prepare_update_restart" => {
                self.log(&format!(
                    "prepare_update_restart command received over socket; {} active session(s) will be closed",
                    self.sessions.lock().expect("sessions poisoned").len()
                ));
                let manifest = self.prepare_update_restart().await?;
                Ok(Some(DaemonResponse::success(
                    id,
                    "prepare_update_restart",
                    Some(manifest),
                )))
            }
            "retry_worker" => {
                Err("Worker retry is only available through the daemon supervisor".to_string())
            }
            "restart" => {
                let daemon = Arc::clone(self);
                tokio::spawn(async move {
                    daemon.shutdown(0, None).await;
                });
                Ok(Some(DaemonResponse::success(id, &command.type_, None)))
            }
            "shutdown" => {
                self.log(&format!(
                    "shutdown command received over socket; {} active session(s) will be closed",
                    self.sessions.lock().expect("sessions poisoned").len()
                ));
                let daemon = Arc::clone(self);
                tokio::spawn(async move {
                    daemon.shutdown(0, None).await;
                });
                Ok(Some(DaemonResponse::success(id, "shutdown", None)))
            }
            other => Err(format!("Unhandled daemon command: {other}")),
        }
    }
}

/// Private helpers shared by the runtime-open path.
fn timing_safe_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

fn serialize_json_line(value: &Value) -> String {
    format!("{value}\n")
}

/// Read the `SessionInfo` row for a persisted session file.
fn read_session_info_sync(path: &str) -> Option<SessionInfo> {
    // The session slice exposes only the async reader; the daemon's call sites
    // already sit on a runtime, so the blocking helper bridges through it.
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current()
            .block_on(crate::core::session_manager::read_session_info(path))
    })
}

/// `createAgentConnectionToolDefinition(definition)` for a definition the session
/// seam already projected to the wire (tool-definition.ts:4 copies these eight
/// fields and drops `prepareArguments`/`executionMode`/`execute`, which the live
/// `ToolDefinition` carries as closures).
fn create_agent_connection_tool_definition_from_value(value: &Value) -> Value {
    let object = |key: &str| value.get(key).cloned();
    let mut definition = Map::new();
    definition.insert(
        "name".to_string(),
        object("name").unwrap_or(Value::String(String::new())),
    );
    definition.insert(
        "label".to_string(),
        object("label").unwrap_or(Value::String(String::new())),
    );
    definition.insert(
        "description".to_string(),
        object("description").unwrap_or(Value::String(String::new())),
    );
    if let Some(prompt_snippet) = object("promptSnippet") {
        if !prompt_snippet.is_null() {
            definition.insert("promptSnippet".to_string(), prompt_snippet);
        }
    }
    if let Some(prompt_guidelines) = object("promptGuidelines") {
        if !prompt_guidelines.is_null() {
            definition.insert("promptGuidelines".to_string(), prompt_guidelines);
        }
    }
    definition.insert(
        "parameters".to_string(),
        object("parameters").unwrap_or(Value::Null),
    );
    if let Some(render_shell) = object("renderShell") {
        if !render_shell.is_null() {
            definition.insert("renderShell".to_string(), render_shell);
        }
    }
    if let Some(replay_built_in_tool_name) = object("replayBuiltInToolName") {
        if !replay_built_in_tool_name.is_null() {
            definition.insert("replayBuiltInToolName".to_string(), replay_built_in_tool_name);
        }
    }
    Value::Object(definition)
}

/// Serialize the binding module's own `DaemonOutbound` frame (it carries the four
/// variants the extension binding emits, wire-compatible with this file's superset).
fn message_to_value(message: &BindingDaemonOutbound) -> Value {
    serde_json::to_value(message).unwrap_or(Value::Null)
}

/// `message.timestamp` (session-manager.ts:1339) for every `AgentMessage` shape.
fn message_timestamp_of(message: &AgentMessage) -> i64 {
    match message {
        AgentMessage::Message(pi_ai::types::Message::User(message)) => message.timestamp,
        AgentMessage::Message(pi_ai::types::Message::Assistant(message)) => message.timestamp,
        AgentMessage::Message(pi_ai::types::Message::ToolResult(message)) => message.timestamp,
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::BashExecution {
            timestamp,
            ..
        }) => *timestamp,
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom {
            timestamp, ..
        }) => *timestamp,
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::BranchSummary {
            timestamp,
            ..
        }) => *timestamp,
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::CompactionSummary {
            timestamp,
            ..
        }) => *timestamp,
    }
}

/// Serialize an `AgentCronJob` for the wire.
fn job_to_value(job: &AgentCronJob) -> Value {
    serde_json::to_value(job).unwrap_or(Value::Null)
}

impl DaemonAttachResult {
    fn with_messages_cleared(&self, snapshot_id: String) -> DaemonAttachResult {
        let message_count = self
            .snapshot
            .get("messages")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut snapshot = self.snapshot.clone();
        if let Some(object) = snapshot.as_object_mut() {
            object.insert("messages".to_string(), Value::Array(Vec::new()));
        }
        DaemonAttachResult {
            messages: self.messages.as_ref().map(|_| Vec::new()),
            snapshot,
            snapshot_stream: Some(serde_json::json!({
                "id": snapshot_id,
                "messageCount": message_count,
                "targetChunkBytes": SNAPSHOT_TARGET_CHUNK_BYTES,
            })),
            ..self.clone()
        }
    }
}

impl AgentDaemon {
    /// `createRuntime(command, runtimeOpenGuard?)`.
    pub(crate) async fn create_runtime(
        self: &Arc<Self>,
        command: &ParsedDaemonCommand,
        runtime_open_guard: Option<RuntimeOpenGuard>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let body = &command.body;
        let config_override = body
            .get("config")
            .cloned()
            .and_then(|value| serde_json::from_value::<AgentSessionRuntimeConfig>(value).ok());
        let config = merge_agent_session_runtime_config(
            &self.options.default_session_config,
            config_override.as_ref(),
        );
        let cwd = resolve_path(
            &config
                .cwd
                .clone()
                .ok_or_else(|| "Active session config is missing cwd".to_string())?,
        );
        let agent_dir = match config.agent_dir.clone() {
            Some(agent_dir) => agent_dir,
            None => return Err("Active session config is missing agentDir".to_string()),
        };
        let cwd_override = config.cwd.clone().map(|cwd| resolve_path(&cwd));
        let desired_active_session_id = match (
            self.is_worker(),
            self.restore_active_session_id
                .lock()
                .expect("restore id poisoned")
                .take(),
        ) {
            (true, restore) => restore,
            (false, _) => None,
        };
        let client_env = body.get("env").and_then(Value::as_object).map(|object| {
            object
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect::<HashMap<String, String>>()
        });
        let session_path = body
            .get("sessionPath")
            .and_then(Value::as_str)
            .map(str::to_string);
        let resolved_session_path = match &session_path {
            Some(selector) => Some(
                resolve_daemon_session_path(selector, &cwd, config.session_dir.as_deref()).await?,
            ),
            None => None,
        };
        let session_key = resolved_session_path.clone();
        if let Some(session_key) = &session_key {
            if self.find_passivation_by_session_file(session_key).is_some() {
                self.wait_for_passivation(session_key).await;
                return Box::pin(self.create_runtime(command, runtime_open_guard)).await;
            }
        }
        if let Some(session_key) = &session_key {
            if self
                .opening_sessions
                .lock()
                .expect("opening sessions poisoned")
                .contains_key(session_key)
            {
                Box::pin(self.create_runtime(command, runtime_open_guard.clone())).await?;
            }
        }
        // `const passiveSubagent = sessionPath ? await this.findPassiveRlmSubagent(sessionPath) : undefined`
        // (`daemon-mode.ts:1839-1871`): a `sessionPath` that names a passive RLM
        // child is hydrated through the whole parent chain instead of being opened
        // as a plain top-level runtime.
        let passive_subagent = match &resolved_session_path {
            Some(session_path) => self.find_passive_rlm_subagent(session_path, false).await,
            None => None,
        };
        if let Some(passive_subagent) = passive_subagent {
            if let Some(guard) = &runtime_open_guard {
                if !guard().await {
                    return Err(RuntimeOpenCancelledError.to_string());
                }
            }
            if let Some(name) = body.get("name").and_then(Value::as_str) {
                let normalized_name = name.trim().to_string();
                if normalized_name.is_empty() {
                    return Err("Session name cannot be empty".to_string());
                }
                // `parentSessionPath: entry.parentSessionFile ?? chain.at(-2)?.sessionFile ?? rootParentState...sessionFile ?? rootInfo?.path`
                // and `depth: info.rlmDepth ?? entry.rlmDepth ?? 1`.
                let parent_session_path = passive_subagent
                    .entry
                    .parent_session_file
                    .clone()
                    .or_else(|| {
                        passive_subagent
                            .chain
                            .iter()
                            .rev()
                            .nth(1)
                            .map(|entry| entry.session_file.clone())
                    })
                    .or_else(|| match &passive_subagent.root {
                        PassiveRlmRoot::Resident(state) => self.session_of(state).session_file(),
                        PassiveRlmRoot::Saved(info) => Some(info.path.clone()),
                    });
                self.assert_family_session_name_available(
                    &AgentSessionNameAvailabilityInput {
                        parent_session_id: Some(passive_subagent.entry.parent_session_id.clone()),
                        parent_session_path,
                        depth: passive_subagent.info.rlm_depth as f64,
                        name: normalized_name,
                        ignore_session_id: Some(passive_subagent.info.id.clone()),
                    },
                    None,
                    false,
                )
                .await?;
            }
            let state = self
                .hydrate_passive_rlm_subagent(passive_subagent.clone(), client_env.clone())
                .await?;
            if let Some(guard) = &runtime_open_guard {
                if !guard().await {
                    return Err(RuntimeOpenCancelledError.to_string());
                }
            }
            if let Some(name) = body.get("name").and_then(Value::as_str) {
                self.set_state_session_name(&state, name).await?;
            }
            if let PassiveRlmRoot::Resident(root_parent) = &passive_subagent.root {
                self.adopt_client_env(root_parent, client_env.clone());
            }
            self.adopt_client_env(&state, client_env);
            return Ok(state);
        }
        if let Some(guard) = &runtime_open_guard {
            if !guard().await {
                return Err(RuntimeOpenCancelledError.to_string());
            }
        }
        // An existing runtime for this session key is reused.
        if let Some(session_key) = &session_key {
            let existing = self.find_session_by_session_file(Some(
                resolved_session_path.as_deref().unwrap_or(session_key),
            ));
            if let Some(existing) = existing {
                if let Some(name) = body.get("name").and_then(Value::as_str) {
                    self.set_state_session_name(&existing, name).await?;
                }
                self.adopt_client_env(&existing, client_env.clone());
                self.rebind_cron_jobs_to_state(&existing);
                return Ok(existing);
            }
        }
        let session_manager = if let Some(session_path) = &resolved_session_path {
            SessionManager::open_async(
                session_path,
                config.session_dir.as_deref(),
                cwd_override.as_deref(),
            )
            .await?
        } else if body.get("noSession").and_then(Value::as_bool) == Some(true) {
            SessionManager::in_memory(Some(&cwd), config.session_dir.as_deref())?
        } else if body.get("continueRecent").and_then(Value::as_bool) == Some(true) {
            SessionManager::continue_recent(&cwd, config.session_dir.as_deref())?
        } else {
            SessionManager::create(&cwd, config.session_dir.as_deref())?
        };
        let state = self
            .create_state_for_runtime(
                command,
                session_manager,
                session_path.clone(),
                desired_active_session_id,
                client_env,
                runtime_open_guard,
            )
            .await?;
        if let Some(session_key) = session_key {
            let opened = session_key;
            self.opening_sessions
                .lock()
                .expect("opening sessions poisoned")
                .remove(&opened);
        }
        Ok(state)
    }

    /// `addRuntime(runtime, ...)` + the `createState` closure inside `createRuntime`.
    #[allow(clippy::too_many_arguments)]
    async fn create_state_for_runtime(
        self: &Arc<Self>,
        command: &ParsedDaemonCommand,
        session_manager: SessionManager,
        _session_path: Option<String>,
        desired_active_session_id: Option<String>,
        client_env: Option<HashMap<String, String>>,
        runtime_open_guard: Option<RuntimeOpenGuard>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let session_manager = Arc::new(StdMutex::new(session_manager));
        // The state is published during `addRuntime`, which runs after the factory
        // returns, so the controller closures read it through this slot - exactly
        // like the TypeScript `stateRef` (`daemon-mode.ts:1955`, `:2007-2008`).
        let state_ref: Arc<StdMutex<Option<Arc<StdMutex<ActiveSessionState>>>>> =
            Arc::new(StdMutex::new(None));
        let session_config = crate::core::agent_session_config::merge_agent_session_runtime_config(
            &serde_json::from_value(serde_json::to_value(&self.options.default_session_config).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?,
            command.body.get("config").map(|config| serde_json::from_value(config.clone())).transpose().map_err(|error| error.to_string())?.as_ref(),
        );
        // `sessionOptions: { rlmHeartbeatController: {...}, agentMessageController:
        // this.createAgentMessageController(() => stateRef), agentObserveController:
        // this.createAgentObserveController(() => stateRef) }`
        // (`daemon-mode.ts:1967-1996`). All three closures read the state slot above,
        // like the TypeScript `stateRef`.
        let get_current_state: Arc<
            dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync,
        > = {
            let state_ref = Arc::clone(&state_ref);
            Arc::new(move || state_ref.lock().expect("session state slot poisoned").clone())
        };
        let session_options = SessionRuntimeOptions {
            subagent_options: None,
            model: None,
            rlm_heartbeat_controller: Some(self.create_rlm_heartbeat_controller(Arc::clone(
                &get_current_state,
            ))),
            agent_message_controller: Some(
                self.create_agent_message_controller(Arc::clone(&get_current_state)),
            ),
            agent_observe_controller: Some(
                self.create_agent_observe_controller(Arc::clone(&get_current_state)),
            ),
        };
        let input = CreateAgentSessionRuntimeInput {
            factory: Value::Null,
            cwd: session_manager
                .lock()
                .expect("session manager poisoned")
                .get_cwd(),
            agent_dir: session_config.agent_dir.clone(),
            session_manager: Arc::clone(&session_manager),
            session_options,
            session_config: Some(session_config),
            runtime_metadata: None,
        };
        let runtime = super::daemon_client_env::with_client_env(client_env.as_ref(), || (self.options.create_runtime)(input)).await?;
        if let Some(guard) = &runtime_open_guard {
            if !guard().await {
                runtime.session.dispose().await;
                return Err(RuntimeOpenCancelledError.to_string());
            }
        }
        // The `(state) => { stateRef = state; }` callback (`daemon-mode.ts:2007-2008`).
        let state_ref_for_callback = Arc::clone(&state_ref);
        self.add_runtime(runtime, desired_active_session_id, Some(Arc::new(move |state| {
            {
                let mut slot = state_ref_for_callback.lock().expect("session state slot poisoned");
                *slot = Some(Arc::clone(state));
            }
            state.lock().expect("active session poisoned").client_env = client_env.clone();
        })), None).await
    }

    /// `addRuntime(runtime, desiredActiveSessionId?, onStateCreated?, onStateBound?)`.
    pub async fn add_runtime(
        self: &Arc<Self>,
        runtime: AgentSessionRuntimeHandle,
        desired_active_session_id: Option<String>,
        on_state_created: Option<Arc<dyn Fn(&Arc<StdMutex<ActiveSessionState>>) + Send + Sync>>,
        on_state_bound: Option<Arc<dyn Fn(&Arc<StdMutex<ActiveSessionState>>) + Send + Sync>>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let mut reserved = None;
        let active_session_id = {
            if runtime.metadata.kind.as_deref() == Some("top-level") {
                if let Some(desired) = &desired_active_session_id {
                    if !self
                        .sessions
                        .lock()
                        .expect("sessions poisoned")
                        .contains_key(desired)
                    {
                        reserved = Some(desired.clone());
                    }
                }
            }
            reserved.unwrap_or_else(|| create_active_session_id(None))
        };
        let state = Arc::new(StdMutex::new(ActiveSessionState::new(
            active_session_id.clone(),
            crate::modes::daemon::active_session_state::AgentSessionRuntime::default(),
        )));
        let session = Arc::clone(&runtime.session);
        let runtime_metadata = runtime.metadata.clone();
        // A child keeps its own model and lifecycle. Only its initial Jev
        // comparison preference is inherited, using durable session identities.
        if runtime_metadata.kind.as_deref() == Some("subagent") {
            if let Some(parent_id) = runtime_metadata.parent_session_id.as_deref().filter(|id| !id.is_empty()) {
                let bridge = crate::modes::interactive::native_host::JevModeBridge::new(
                    std::path::Path::new(&self.agent_dir),
                );
                let child_id = session.session_id();
                if bridge.settings().session_mode(&child_id).is_none() {
                    let _ = bridge.inherit_into_child(&child_id, parent_id, None);
                }
            }
        }
        let runtime_session = ActiveSessionRuntimeSession {
            session_id: session.session_id(),
            session_name: session.session_name(),
            session_file: session.session_file(),
            is_session_active: session.is_session_active(),
            is_foreground_active: Some(session.is_foreground_active()),
            is_streaming: session.is_streaming(),
            is_compacting: session.is_compacting(),
            messages_len: session.messages().len(),
            has_running_rlm_children: session.has_running_rlm_children(),
            // `daemon-session-list.ts:257-258` copies `session.model` and
            // `session.thinkingLevel` onto every summary, and the `--print`/`--json`
            // clients gate on `summary.model` (`main.ts:1628`). Capture them here, where
            // the live `DaemonSession` is still in hand; the summary builder only sees the
            // narrowed runtime view.
            model_identity: session.model_identity(),
            thinking_level: session.thinking_level(),
            ..ActiveSessionRuntimeSession::default()
        };
        {
            let mut state_guard = state.lock().expect("active session poisoned");
            state_guard.runtime = crate::modes::daemon::active_session_state::AgentSessionRuntime {
                session: runtime_session,
                metadata: Some(runtime_metadata.clone()),
                model_fallback_message: runtime.model_fallback_message.clone(),
            };
        }
        self.sessions.lock().expect("sessions poisoned").insert(
            active_session_id.clone(),
            Arc::new(DaemonSessionState {
                state: Arc::clone(&state),
                session: Arc::clone(&session),
                runtime_metadata: runtime_metadata.clone(),
                snapshot_boundary: StdMutex::new(Some({
                    let messages = Arc::new(session.messages());
                    let state = state.lock().expect("active session poisoned");
                    PublishedTranscript { messages, streaming_message: state.runtime.session.streaming_message.clone(), state_flags: (state.runtime.session.is_session_active, state.runtime.session.is_streaming, state.runtime.session.is_compacting), sequence: state.last_event_sequence, generation: state.event_generation.clone() }
                })),
            }),
        );
        self.binding_sessions
            .lock()
            .expect("binding sessions poisoned")
            .insert(active_session_id.clone());
        if let Some(on_state_created) = &on_state_created {
            on_state_created(&state);
        }
        let bind_result = self.bind_state(&state, Arc::clone(&session)).await;
        self.binding_sessions
            .lock()
            .expect("binding sessions poisoned")
            .remove(&active_session_id);
        self.binding_completions
            .lock()
            .expect("binding completions poisoned")
            .remove(&active_session_id);
        if let Err(error) = bind_result {
            {
                let mut state_guard = state.lock().expect("active session poisoned");
                if let Some(unsubscribe) = state_guard.unsubscribe.take() {
                    unsubscribe();
                }
            }
            self.sessions
                .lock()
                .expect("sessions poisoned")
                .remove(&active_session_id);
            self.session_of(&state).dispose().await;
            return Err(error);
        }
        if let Some(on_state_bound) = &on_state_bound {
            on_state_bound(&state);
        }
        self.schedule_roster_flush();
        self.register_cron_store_for_state(&state);
        self.rebind_cron_jobs_to_state(&state);
        {
            let session = self.session_of(&state);
            if session.session_file().is_some() {
                let metadata = state
                    .lock()
                    .expect("active session poisoned")
                    .runtime
                    .metadata
                    .clone();
                if metadata
                    .as_ref()
                    .and_then(|metadata| metadata.kind.as_deref())
                    != Some("subagent")
                {
                    let manager = session.session_manager();
                    manager
                        .lock()
                        .expect("session manager poisoned")
                        .append_session_state(&crate::core::session_manager::SessionState {
                            status: crate::core::session_manager::SessionStateStatus::Active,
                        });
                }
            }
        }
        self.summarizer.seed(&state);
        self.record_worker_recovery_state(&state, "ready", None);
        Ok(state)
    }

    /// `bindActiveSessionState(state, {...})`.
    async fn bind_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        session: Arc<dyn DaemonSession>,
    ) -> Result<(), String> {
        let binder: Arc<dyn DaemonExtensionBindingSession> = Arc::new(SessionBinder { session });
        let state_daemon = Arc::clone(self);
        let broadcast_daemon = Arc::clone(self);
        let replaced_daemon = Arc::clone(self);
        let shutdown_daemon = Arc::clone(self);
        let session_daemon = Arc::downgrade(self);
        let session_id = state.lock().expect("active session poisoned").active_session_id.clone();
        bind_active_session_state(
            state,
            binder,
            ActiveSessionBindingCallbacks {
                get_session: Arc::new(move || {
                    let daemon = session_daemon.upgrade()?;
                    let state = daemon.get_session_state(&session_id).ok()?;
                    Some(Arc::new(SessionBinder { session: daemon.session_of(&state) }) as Arc<dyn DaemonExtensionBindingSession>)
                }),
                broadcast: Arc::new(
                    move |target: &ActiveSessionState, message: BindingDaemonOutbound| {
                        let Ok(state) = broadcast_daemon.get_session_state(&target.active_session_id) else { return; };
                        let entry = broadcast_daemon.session_entry_for_state(&state);
                        let message = match message {
                            BindingDaemonOutbound::SessionEvent { active_session_id, event } => DaemonOutbound::SessionEvent { active_session_id, event },
                            BindingDaemonOutbound::SessionReplaced { active_session_id, state, .. } => DaemonOutbound::SessionReplaced { active_session_id, state, messages: entry.session.messages().iter().map(|message| serde_json::to_value(message).expect("agent message is serializable")).collect() },
                            BindingDaemonOutbound::ExtensionError { active_session_id, extension_path, event, error } => DaemonOutbound::ExtensionError { active_session_id, extension_path, event, error },
                            BindingDaemonOutbound::ExtensionUiRequest { active_session_id, id, method, payload } => DaemonOutbound::ExtensionUiRequest { active_session_id, id, method, payload },
                        };
                        broadcast_daemon.broadcast_to_session(&entry, message);
                    },
                ),
                create_connection_state: Some(Arc::new(move |target: &ActiveSessionState| {
                    state_daemon.get_session_state(&target.active_session_id)
                        .map(|state| state_daemon.create_connection_state(&state))
                        .unwrap_or(Value::Null)
                })),
                session_replaced: Some(Arc::new(move |target: &ActiveSessionState| {
                    if let Ok(state) = replaced_daemon.get_session_state(&target.active_session_id) {
                        replaced_daemon.refresh_replaced_session_state(&state);
                    }
                })),
                shutdown: Arc::new(move || {
                    let daemon = Arc::clone(&shutdown_daemon);
                    tokio::spawn(async move {
                        daemon.shutdown(0, None).await;
                    });
                }),
                subagent_runtime_host: Some(Arc::new(daemon_subagents::DaemonSubagentHost::new(self, state))),
            },
        )
        .await;
        Ok(())
    }

    /// `refreshReplacedSessionState(state)`.
    fn refresh_replaced_session_state(self: &Arc<Self>, state: &Arc<StdMutex<ActiveSessionState>>) {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.acp_mcp_owners
            .lock()
            .expect("acp mcp owners poisoned")
            .remove(&active_session_id);
        let clients = state
            .lock()
            .expect("active session poisoned")
            .clients
            .clone();
        for client_state in &clients {
            let Some(handle) = self
                .client_handles()
                .into_iter()
                .find(|handle| Arc::ptr_eq(&handle.state, client_state))
            else {
                continue;
            };
            self.abort_side_questions_for(&handle, &active_session_id);
        }
        self.summarizer.forget(&active_session_id);
        self.session_of(state).set_current_recap(None);
        self.summarizer.seed(state);
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        if metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.as_deref())
            == Some("subagent")
        {
            let recap = state
                .lock()
                .expect("active session poisoned")
                .summary_state
                .as_ref()
                .map(|summary| summary.summary.clone());
            self.session_of(state).set_current_recap(recap.as_deref());
        }
        self.register_cron_store_for_state(state);
        self.rebind_cron_jobs_to_state(state);
    }

    /// `registerCronStoreForState(state)`.
    fn register_cron_store_for_state(self: &Arc<Self>, state: &Arc<StdMutex<ActiveSessionState>>) {
        if !self.is_worker() {
            return;
        }
        let session = self.session_of(state);
        let (Some(session_id), Some(session_file)) =
            (Some(session.session_id()), session.session_file())
        else {
            return;
        };
        let artifact_dir =
            get_session_artifact_path_for_file(&session_file, Some(&session_id));
        if artifact_dir.is_empty() {
            return;
        }
        if self
            .cron_store
            .register_session_artifact(&session_id, &artifact_dir)
        {
            self.cron_store
                .recover_session_artifact(&session_id, now_millis());
            self.cron_scheduler_wake();
        }
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        if metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.as_deref())
            != Some("subagent")
        {
            let daemon = Arc::clone(self);
            tokio::spawn(async move {
                if let Err(error) = daemon.register_passive_descendant_cron_artifacts().await {
                    daemon.log(&format!(
                        "Could not register passive descendant scheduled jobs: {error}"
                    ));
                }
            });
        }
    }

    /// `registerPassiveDescendantCronArtifacts()`.
    async fn register_passive_descendant_cron_artifacts(self: &Arc<Self>) -> Result<(), String> {
        let mut registered = false;
        for passive in self.list_passive_rlm_subagents(Vec::new(), false).await {
            let mut artifact_dir =
                get_session_artifact_path_for_file(&passive.info.path, Some(&passive.info.id));
            if artifact_dir.is_empty() {
                artifact_dir = get_session_artifact_path_for_file(
                    &passive.info.path,
                    Some(&passive.entry.parent_session_id),
                );
            }
            if self
                .cron_store
                .register_session_artifact(&passive.info.id, &artifact_dir)
            {
                self.cron_store
                    .recover_session_artifact(&passive.info.id, now_millis());
                registered = true;
            }
        }
        if registered {
            self.cron_scheduler_wake();
        }
        Ok(())
    }

    /// `cronScheduler.wake()`.
    fn cron_scheduler_wake(&self) {
        if let Some(scheduler) = self
            .cron_scheduler
            .lock()
            .expect("cron scheduler poisoned")
            .as_ref()
        {
            scheduler.wake();
        }
    }

    /// The `AgentCronScheduler` the constructor builds.
    fn start_cron_scheduler(self: &Arc<Self>) {
        let daemon = Arc::clone(self);
        let begin_daemon = Arc::clone(self);
        let error_daemon = Arc::clone(self);
        let hooks = Arc::new(crate::core::cron_jobs::AgentCronSchedulerHooks {
            // `runJob: (job) => this.runCronJob(job)` (daemon-mode.ts:659). This field's
            // type cannot reject (`AgentCronSchedulerHooks` is another slice's struct), so it
            // can only report a failure; the true throwing form of the same call is installed
            // through `enable_run_job_errors` below and is what the scheduler prefers
            // (core/cron_jobs.rs:1972-1976).
            run_job: Arc::new(move |job: AgentCronJob| {
                let daemon = Arc::clone(&daemon);
                Box::pin(async move {
                    match daemon.run_cron_job(job.clone()).await {
                        Ok(result) => result,
                        Err(error) => {
                            daemon.log(&format!("Cron job {} failed: {error}", job.id));
                            None
                        }
                    }
                })
            }),
            begin_dispatch: Some(Arc::new(move |_dispatch| {
                begin_daemon.mutation_drain.begin();
                let daemon = Arc::clone(&begin_daemon);
                Some(Arc::new(move || daemon.mutation_drain.end()) as Arc<dyn Fn() + Send + Sync>)
            })),
            now: None,
            on_error: Some(Arc::new(move |job: &AgentCronJob, error: String| {
                error_daemon.log(&format!("Cron job {} failed: {error}", job.id));
            })),
        });
        let scheduler = Arc::new(AgentCronScheduler::new(Arc::clone(&self.cron_store), hooks));
        let run_job_daemon = Arc::clone(self);
        scheduler.enable_run_job_errors(Arc::new(move |job: AgentCronJob| {
            let daemon = Arc::clone(&run_job_daemon);
            Box::pin(async move { daemon.run_cron_job(job).await })
        }));
        *self.cron_scheduler.lock().expect("cron scheduler poisoned") =
            Some(Arc::clone(&scheduler));
        scheduler.start();
        let daemon = Arc::clone(self);
        self.cron_store.on_heartbeat_change(Arc::new(move || {
            daemon.broadcast_global(&DaemonOutbound::HeartbeatsChanged);
            daemon.schedule_roster_flush();
        }));
    }
}

/// One resident session: the daemon's `ActiveSessionState` plus the live session
/// handle (`state.runtime.session`) and its runtime metadata, which the port
/// keeps beside the state struct (that struct belongs to another slice).
pub struct DaemonSessionState {
    pub state: Arc<StdMutex<ActiveSessionState>>,
    pub session: Arc<dyn DaemonSession>,
    pub runtime_metadata: AgentSessionRuntimeMetadata,
    /// Transcript and cursor published at the same event boundary. Explicit view
    /// refreshes must not advance this cursor to an unbroadcast live mutation.
    pub snapshot_boundary: StdMutex<Option<PublishedTranscript>>,
}

pub struct PublishedTranscript {
    messages: Arc<Vec<AgentMessage>>,
    streaming_message: Option<AgentMessage>,
    state_flags: (bool, bool, bool),
    sequence: u64,
    generation: String,
}

/// Adapts the session slice's `bindExtensions` surface to the binding module.
struct SessionBinder {
    session: Arc<dyn DaemonSession>,
}

#[async_trait::async_trait]
impl DaemonExtensionBindingSession for SessionBinder {
    fn set_exec_env_provider(
        &self,
        client_env: Option<crate::modes::daemon::daemon_client_env::EnvMap>,
    ) {
        self.session.set_exec_env_provider(client_env);
    }

    fn set_runtime_env_scope(
        &self,
        client_env: Option<crate::modes::daemon::daemon_client_env::EnvMap>,
    ) {
        self.session.set_runtime_env_scope(client_env);
    }

    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {
        self.session.set_subagent_runtime_host(host);
    }

    fn subscribe(
        &self,
        listener: Arc<dyn Fn(&Value) + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        let unsubscribe = self.session.subscribe(listener);
        Box::new(move || unsubscribe())
    }

    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {
        self.session.set_rebind_session(rebind);
    }

    async fn bind_extensions(
        &self,
        binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput,
    ) {
        let _ = self.session.bind_extensions(binding).await;
    }

    async fn wait_for_idle(&self) {
        self.session.wait_for_idle().await;
    }

    async fn reload(&self) {
        let _ = self.session.reload().await;
    }
}

/// A session that vanished from the resident map (the TypeScript would have
/// dereferenced a stale object). Every call fails loudly instead of panicking.
struct MissingSession {
    active_session_id: String,
}

impl MissingSession {
    fn new(active_session_id: &str) -> Self {
        Self {
            active_session_id: active_session_id.to_string(),
        }
    }

    fn error(&self) -> String {
        format!("Active session {} is not resident", self.active_session_id)
    }
}

impl DaemonSession for MissingSession {
    fn session_manager(&self) -> Arc<StdMutex<SessionManager>> {
        Arc::new(StdMutex::new(
            SessionManager::in_memory(Some("."), None).expect("in-memory session manager"),
        ))
    }
    fn runtime(&self) -> Arc<dyn DaemonRuntimeApi> {
        Arc::new(MissingRuntime)
    }
    fn settings_manager(&self) -> Option<Arc<StdMutex<SettingsManager>>> {
        None
    }
    fn session_id(&self) -> String {
        String::new()
    }
    fn session_name(&self) -> Option<String> {
        None
    }
    fn session_file(&self) -> Option<String> {
        None
    }
    fn session_dir(&self) -> Option<String> {
        None
    }
    fn is_streaming(&self) -> bool {
        false
    }
    fn is_compacting(&self) -> bool {
        false
    }
    fn is_bash_running(&self) -> bool {
        false
    }
    fn is_retrying(&self) -> bool {
        false
    }
    fn is_session_active(&self) -> bool {
        false
    }
    fn has_running_rlm_children(&self) -> bool {
        false
    }
    fn unfinished_action_count(&self) -> f64 {
        0.0
    }
    fn messages(&self) -> Vec<AgentMessage> {
        Vec::new()
    }
    fn model_identity(&self) -> Option<pi_ai::types::Model> {
        None
    }
    fn rlm_depth(&self) -> Option<i64> {
        None
    }
    fn thinking_level(&self) -> Option<String> {
        None
    }
    fn service_tier(&self) -> Option<String> {
        None
    }
    fn system_prompt(&self) -> Option<String> {
        None
    }
    fn connection_view(&self) -> DaemonConnectionView {
        DaemonConnectionView::default()
    }
    fn connection_state(&self, _active_session_id: Option<String>) -> Value {
        Value::Null
    }
    fn set_current_recap(&self, _recap: Option<&str>) {}
    fn set_session_name(&self, _name: &str) {}
    fn get_rlm_child_run_status(&self, _child_id: &str) -> Option<String> {
        None
    }
    fn register_rlm_child_session(
        &self,
        _child_id: &str,
        _session: Arc<dyn DaemonSession>,
    ) -> bool {
        false
    }
    fn remove_queued_follow_up(&self, _key: &str) {}
    fn subscribe(&self, _listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }
    fn set_exec_env_provider(&self, _client_env: Option<HashMap<String, String>>) {}
    fn set_runtime_env_scope(&self, _client_env: Option<HashMap<String, String>>) {}
    fn set_subagent_runtime_host(&self, _host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {}
    fn set_rebind_session(&self, _rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {}
    fn bind_extensions(
        &self,
        _binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn abort_for_update_restart(&self) {}
    fn prompt_until_accepted(
        &self,
        _message: &str,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn prompt_and_wait(
        &self,
        _message: &str,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn prompt_heartbeat(
        &self,
        _job: &AgentCronJob,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn accept_agent_message_prompt(
        &self,
        _message: &str,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn steer(
        &self,
        _message: &str,
        _images: Option<Value>,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn follow_up(
        &self,
        _message: &str,
        _images: Option<Value>,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<bool, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn restore_steering_message(
        &self,
        _message: &str,
        _images: Option<Value>,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn restore_follow_up_message(
        &self,
        _message: &str,
        _images: Option<Value>,
        _options: PromptInvocation,
    ) -> BoxFuture<'static, Result<bool, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn restore_pending_next_turn_messages(&self, _messages: &Value) {}
    fn restore_session_actions(
        &self,
        _snapshot: &Value,
    ) -> BoxFuture<'static, Result<f64, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn send_custom_message(&self, _message: &Value) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn resume_queued_work(&self) -> bool {
        false
    }
    fn clear_queued_agent_messages(&self) -> Value {
        Value::Null
    }
    fn clear_queue(&self) -> Value {
        Value::Null
    }
    fn mutate_queued_message(
        &self,
        _lane: &str,
        _index: f64,
        _expected_text: &str,
        _mutation: &Value,
    ) -> Value {
        Value::Null
    }
    fn get_steering_message_previews(&self) -> Vec<Value> {
        Vec::new()
    }
    fn get_follow_up_message_previews(&self) -> Vec<Value> {
        Vec::new()
    }
    fn request_abort(&self) {}
    fn cancel_rlm_child_run(&self, _child_id: &str) -> bool {
        false
    }
    fn delete_inactive_rlm_subagent(
        &self,
        _child_id: &str,
        _is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> BoxFuture<'static, Result<String, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn run_user_bash(
        &self,
        _command: &str,
        _options: RunUserBashOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn execute_bash(&self, _command: &str) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn abort_bash(&self) {}
    fn acquire_session_input_pause(&self) -> SessionInputPause {
        Arc::new(|| {})
    }
    fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
    fn wait_for_headless_completion(
        &self,
        _options: HeadlessCompletionOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn refresh_available_models(
        &self,
    ) -> BoxFuture<'static, Result<Vec<pi_ai::types::Model>, String>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn refresh_model_catalog(&self) -> BoxFuture<'static, Result<Value, String>> {
        Box::pin(async { Ok(Value::Null) })
    }
    fn get_provider_auth_status_source(&self, _provider: &str) -> Option<String> {
        None
    }
    fn find_model(&self, _provider: &str, _model_id: &str) -> Option<pi_ai::types::Model> {
        None
    }
    fn set_model(
        &self,
        _model: &pi_ai::types::Model,
        _wait_for_extensions: bool,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn cycle_model(
        &self,
        _direction: &str,
        _wait_for_extensions: bool,
    ) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn set_scoped_models(&self, _scoped_models: &Value) {}
    fn set_thinking_level(&self, _level: &str) {}
    fn set_service_tier(&self, _service_tier: &str) {}
    fn cycle_thinking_level(&self) -> Option<String> {
        None
    }
    fn set_transport(&self, _transport: &str) {}
    fn set_steering_mode(&self, _mode: &str) {}
    fn set_follow_up_mode(&self, _mode: &str) {}
    fn set_auto_compaction_enabled(&self, _enabled: bool) {}
    fn set_auto_retry_enabled(&self, _enabled: bool) {}
    fn compact(
        &self,
        _custom_instructions: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn refine(&self, _options: RefineOptions) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn abort_compaction(&self) {}
    fn abort_branch_summary(&self) {}
    fn abort_retry(&self) {}
    fn reload(&self) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn get_rlm_max_depth_status(&self) -> Value {
        Value::Null
    }
    fn set_rlm_max_depth(
        &self,
        _max_depth: Value,
        _global: bool,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn build_session_context(&self) -> Value {
        Value::Null
    }
    fn get_session_stats(&self) -> Value {
        Value::Null
    }
    fn get_context_tree(&self) -> Value {
        Value::Null
    }
    fn get_rlm_child_snapshots(&self) -> Vec<Value> {
        Vec::new()
    }
    fn export_to_html(
        &self,
        _output_path: Option<&str>,
    ) -> BoxFuture<'static, Result<String, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn export_to_jsonl(&self, _output_path: Option<&str>) -> Result<String, String> {
        Err(self.error())
    }
    fn get_user_messages_for_forking(&self) -> Vec<Value> {
        Vec::new()
    }
    fn get_last_assistant_text(&self) -> String {
        String::new()
    }
    fn get_tool_definition(&self, _name: &str) -> Option<Value> {
        None
    }
    fn navigate_tree(
        &self,
        _target_id: &str,
        _options: NavigateTreeOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn start_side_question(
        &self,
        _question: &str,
        _options: SideQuestionOptions,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn abort_side_question(&self, _side_question_id: &str) {}
    fn release_acp_mcp_servers(
        &self,
        _owner_id: &str,
        _server_names: &[String],
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn replace_acp_mcp_servers(
        &self,
        _servers: &[Value],
        _owner_id: &str,
    ) -> BoxFuture<'static, Result<(), String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn new_session(
        &self,
        _options: Option<NewSessionRuntimeOptions>,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn release_rlm_child_session(
        &self,
        _child_id: &str,
        _session: Arc<dyn DaemonSession>,
    ) -> Option<Box<dyn FnOnce() + Send>> {
        None
    }
    fn replied_to_parent_since_task(&self) -> Option<bool> {
        None
    }
    fn switch_session(
        &self,
        _session_path: &str,
        _options: SessionPathOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn fork(
        &self,
        _entry_id: &str,
        _options: ForkOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn import_from_jsonl(
        &self,
        _input_path: &str,
        _cwd_override: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let error = self.error();
        Box::pin(async move { Err(error) })
    }
    fn dispose(&self) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

struct MissingRuntime;

impl DaemonRuntimeApi for MissingRuntime {
    fn new_session(
        &self,
        _options: Option<NewSessionRuntimeOptions>,
    ) -> BoxFuture<'static, Result<Value, String>> {
        Box::pin(async { Err("Active session is not resident".to_string()) })
    }
    fn switch_session(
        &self,
        _session_path: &str,
        _options: SessionPathOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        Box::pin(async { Err("Active session is not resident".to_string()) })
    }
    fn fork(
        &self,
        _entry_id: &str,
        _options: ForkOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        Box::pin(async { Err("Active session is not resident".to_string()) })
    }
    fn import_from_jsonl(
        &self,
        _input_path: &str,
        _cwd_override: Option<&str>,
    ) -> BoxFuture<'static, Result<Value, String>> {
        Box::pin(async { Err("Active session is not resident".to_string()) })
    }
}

impl DaemonSessionState {
    /// `state.runtime.session.sessionManager.getCwd()`.
    pub fn cwd(&self) -> String {
        self.session
            .session_manager()
            .lock()
            .expect("session manager poisoned")
            .get_cwd()
    }

    /// Full refresh at explicit snapshot/read boundaries, never for token deltas.
    pub fn sync_view(&self) {
        self.sync_event_view(true);
    }

    fn sync_event_view(&self, refresh_history: bool) {
        let session = &self.session;
        // Read the session before taking the view lock; getters may take their
        // own locks. Retain the existing registry/status writer and other fields.
        let session_id = session.session_id();
        let session_name = session.session_name();
        let session_file = session.session_file();
        let is_session_active = session.is_session_active();
        let is_foreground_active = session.is_foreground_active();
        let is_streaming = session.is_streaming();
        let is_compacting = session.is_compacting();
        let has_running_rlm_children = session.has_running_rlm_children();
        let rlm_depth = session.rlm_depth();
        let model_identity = session.model_identity();
        let thinking_level = session.thinking_level();
        let messages = refresh_history.then(|| session.messages());
        let messages_len = messages.as_ref().map(Vec::len).unwrap_or_else(|| session.message_count());
        let mut state = self.state.lock().expect("active session poisoned");
        let view = &mut state.runtime.session;
        view.session_id = session_id;
        view.session_name = session_name;
        view.session_file = session_file;
        view.is_session_active = is_session_active;
        view.is_foreground_active = Some(is_foreground_active);
        view.is_streaming = is_streaming;
        view.is_compacting = is_compacting;
        view.has_running_rlm_children = has_running_rlm_children;
        view.rlm_depth = rlm_depth;
        view.model_identity = model_identity;
        view.thinking_level = thinking_level;
        view.messages_len = messages_len;
        if let Some(messages) = messages { view.messages = messages; }
        state.runtime.metadata = Some(self.runtime_metadata.clone());
    }

    /// A `session_event` payload as the daemon sees it, after refreshing the view.
    pub fn handle_event(&self, event: &Value) -> Value {
        let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");
        let mut state = self.state.lock().expect("active session poisoned");
        let view = &mut state.runtime.session;
        match event_type {
            "message_start" | "message_update" => {
                if let Some(message) = event.get("message") {
                    if message.get("role").and_then(Value::as_str) == Some("assistant") {
                        if !message.is_null() {
                            if let Ok(parsed) = serde_json::from_value::<AgentMessage>(message.clone())
                            {
                                view.streaming_message = Some(parsed);
                            }
                        }
                    }
                }
            }
            "message_end" => {
                // The completed message is already in retained history.
                view.streaming_message = None;
            }
            "agent_start" => {
                view.is_session_active = true;
                view.streaming_message = None;
            }
            "agent_end" => {
                view.is_session_active = false;
                view.is_streaming = false;
                view.streaming_message = None;
            }
            _ => {}
        }
        event.clone()
    }
}

impl AgentDaemon {
    /// `broadcastToSession(state, message)`.
    fn broadcast_to_session(self: &Arc<Self>, entry: &Arc<DaemonSessionState>, message: DaemonOutbound) {
        let mut published = entry.snapshot_boundary.lock().expect("snapshot boundary poisoned");
        // Retained history changes at message/lifecycle boundaries, not on each
        // assistant or tool partial. Explicit snapshot reads still refresh it.
        let partial = matches!(&message, DaemonOutbound::SessionEvent { event, .. }
            if matches!(event.get("type").and_then(Value::as_str), Some("message_update" | "tool_execution_update")));
        entry.sync_event_view(!partial);
        let state = entry.state.clone();
        if let DaemonOutbound::SessionEvent { event, .. } = &message {
            entry.handle_event(event);
            let event_type = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            // A finished turn/compaction is the cue to refresh status.
            if event_type == "turn_end" || event_type == "compaction_end" {
                self.summarizer.notify_activity(Arc::clone(&state));
            }
            // A draft whose last client detached while it was busy isn't discardable
            // at detach time; re-check once any work (turn, compaction, or bash)
            // settles so it doesn't linger in the daemon.
            if (event_type == "turn_end"
                || event_type == "compaction_end"
                || event_type == "bash_end")
                && self.is_discardable_draft(&state)
            {
                let daemon = self.clone_arc();
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    daemon
                        .close_session(state, CLOSING_REASON_KILLED, false, false, None, None)
                        .await;
                });
            }
            if contains(&RECOVERY_CHECKPOINT_EVENTS, &event_type) {
                self.record_worker_recovery_state(&state, &event_type, None);
            }
        }
        let message = self.stamp_rlm_child_active_session_id(message);
        self.observe_roster_event(&state, &message);
        let sequenced = self.add_session_event_meta(&state, message);
        {
            let state = state.lock().expect("active session poisoned");
            let messages = if partial {
                published.as_ref().map(|snapshot| snapshot.messages.clone())
            } else { None }.unwrap_or_else(|| Arc::new(state.runtime.session.messages.clone()));
            *published = Some(PublishedTranscript {
                messages,
                streaming_message: state.runtime.session.streaming_message.clone(),
                state_flags: (state.runtime.session.is_session_active, state.runtime.session.is_streaming, state.runtime.session.is_compacting),
                sequence: state.last_event_sequence,
                generation: state.event_generation.clone(),
            });
        }
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut serialized: Option<String> = None;
        let clients = state
            .lock()
            .expect("active session poisoned")
            .clients
            .clone();
        let handles: Vec<Arc<DaemonClientHandle>> = self
            .client_handles()
            .into_iter()
            .filter(|client| {
                clients
                    .iter()
                    .any(|candidate| Arc::ptr_eq(candidate, &client.state))
            })
            .collect();
        for client in handles {
            if !should_send_daemon_outbound_to_client(&client, &sequenced) {
                continue;
            }
            if sequenced.type_name() == "session_closed" {
                abort_client_snapshot_streaming(&client, Some(&active_session_id));
                client.deferred_snapshot_frames.lock().expect("deferred frames poisoned").remove(&active_session_id);
                {
                    let mut state_guard = client.state.lock().expect("daemon client poisoned");
                    if let Some(ids) = state_guard.catchup_active_session_ids.as_mut() {
                        ids.remove(&active_session_id);
                    }
                    if let Some(purposes) = state_guard.catchup_purposes.as_mut() {
                        purposes.remove(&active_session_id);
                    }
                }
                self.write(&client, &sequenced);
                continue;
            }
            if self.defer_snapshot_frame(&client, &active_session_id, &sequenced) { continue; }
            if client.is_backpressured() {
                self.queue_client_catchup(
                    &client,
                    &active_session_id,
                    if sequenced.type_name() == "session_replaced" {
                        "replacement"
                    } else {
                        "resync"
                    },
                );
                continue;
            }
            if sequenced.type_name() == "session_replaced"
                && client
                    .state
                    .lock()
                    .expect("daemon client poisoned")
                    .transport
                    .as_deref()
                    == Some("private-framed")
                && client
                    .capabilities_for_session(&active_session_id)
                    .contains("chunked_snapshot")
            {
                self.begin_replacement_snapshot(&client, &state, sequenced.clone());
                continue;
            }
            if client
                .state
                .lock()
                .expect("daemon client poisoned")
                .transport
                .as_deref()
                == Some("private-framed")
            {
                self.write(&client, &sequenced);
            } else {
                let line = serialized
                    .get_or_insert_with(|| serialize_json_line(&sequenced.to_value()))
                    .clone();
                self.write_serialized(&client, &line, Some(&sequenced));
            }
            if sequenced.type_name() == "session_replaced" {
                // A daemon-side session replacement gives the client a NEW
                // session id: the client reset its status surface while
                // applying the replacement, so the authoritative Jev footer
                // (decision mode + independent compaction, from ONE settings
                // snapshot) must FOLLOW the replace frame, exactly like the
                // attach push. Without this the footer stays blank until
                // re-attach or the next /jev write.
                self.publish_jev_attach_footer(&client, &state);
            }
        }
        drop(published);
    }

    fn clone_arc(self: &Arc<Self>) -> Arc<Self> {
        Arc::clone(self)
    }

    /// `broadcastToSession` for another module's frame (the extension binding).
    fn broadcast_raw_to_session(self: &Arc<Self>, target: &ActiveSessionState, message: &Value) {
        let entry = self.session_states().into_iter().find(|entry| {
            entry
                .state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                == target.active_session_id
        });
        if let Some(entry) = entry {
            self.broadcast_to_session(&entry, DaemonOutbound::Raw(message.clone()));
        }
    }

    /// `broadcastGlobal(message)`.
    fn broadcast_global(self: &Arc<Self>, message: &DaemonOutbound) {
        for client in self.client_handles() {
            self.write(&client, message);
        }
    }

    /// `stampRlmChildActiveSessionId(message)`.
    fn stamp_rlm_child_active_session_id(&self, message: DaemonOutbound) -> DaemonOutbound {
        let DaemonOutbound::SessionEvent {
            active_session_id,
            event,
        } = &message
        else {
            return message;
        };
        if event.get("type").and_then(Value::as_str) != Some("rlm_child_update") {
            return message;
        }
        let Some(child) = event.get("child").and_then(Value::as_object) else {
            return message;
        };
        if child.get("activeSessionId").is_some() {
            return message;
        }
        let child_id = child.get("id").and_then(Value::as_str).unwrap_or("");
        let resident = self.session_states().into_iter().find(|entry| {
            entry.runtime_metadata.kind.as_deref() == Some("subagent")
                && entry.runtime_metadata.rlm_child_id.as_deref() == Some(child_id)
        });
        let Some(resident) = resident else {
            return message;
        };
        let resident_active_session_id = resident
            .state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut event = event.clone();
        if let Some(child) = event.get_mut("child").and_then(Value::as_object_mut) {
            child.insert(
                "activeSessionId".to_string(),
                Value::String(resident_active_session_id),
            );
        }
        DaemonOutbound::SessionEvent {
            active_session_id: active_session_id.clone(),
            event,
        }
    }

    /// `addSessionEventMeta(state, message)`.
    fn add_session_event_meta(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        message: DaemonOutbound,
    ) -> DaemonOutbound {
        let mut value = message.to_value();
        if !is_sequenced_session_outbound(message.type_name()) || value.get("meta").is_some_and(|meta| !meta.is_null()) {
            return message;
        }
        let (generation, sequence) = {
            let mut state = state.lock().expect("active session poisoned");
            state.last_event_sequence += 1;
            (state.event_generation.clone(), state.last_event_sequence)
        };
        let meta = create_daemon_event_meta(
            message.active_session_id().unwrap_or(""),
            sequence,
            now_iso(),
            &generation,
        );
        if let Some(object) = value.as_object_mut() {
            object.insert("meta".to_string(), serde_json::to_value(meta).expect("daemon event metadata is serializable"));
        }
        DaemonOutbound::Raw(value)
    }


    /// `write(client, message)` for a raw value: the TypeScript routes every caller through
    /// `writeSerialized`, which wraps a `private-framed` client's payload in
    /// `encodePrivateFrame` (`daemon-mode.ts:7636`). Writing the bare line here would
    /// desynchronize the peer's frame decoder.
    fn write_public_value(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        value: &Value,
        outbound_type: &str,
    ) -> bool {
        let line = serialize_json_line(value);
        let private_framed = client
            .state
            .lock()
            .expect("daemon client poisoned")
            .transport
            .as_deref()
            == Some("private-framed");
        if private_framed {
            // `writeSerialized` derives the routing header from the message itself
            // (`daemon-mode.ts:7639-7646`), so the payload's session id reaches the peer.
            let mut header = serde_json::Map::from_iter([
                ("kind".to_string(), Value::String("outbound".to_string())),
                (
                    "outboundType".to_string(),
                    Value::String(outbound_type.to_string()),
                ),
            ]);
            if let Some(active_session_id) = value.get("activeSessionId").and_then(Value::as_str) {
                header.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.to_string()),
                );
            }
            if let Some(snapshot_id) = value.get("snapshotId").and_then(Value::as_str) {
                header.insert(
                    "snapshotId".to_string(),
                    Value::String(snapshot_id.to_string()),
                );
            }
            if let Some(request_id) = value.get("id").and_then(Value::as_str) {
                header.insert(
                    "requestId".to_string(),
                    Value::String(request_id.to_string()),
                );
            }
            let header = Value::Object(header);
            match encode_private_frame(&header, line.as_bytes()) {
                Ok(frame) => return client.writer.write_bytes(frame),
                Err(error) => {
                    self.log(&format!("Daemon private frame encode failed: {error}"));
                    return false;
                }
            }
        }
        client.writer.write(line)
    }
    /// `write(client, message)`.
    fn write(self: &Arc<Self>, client: &Arc<DaemonClientHandle>, message: &DaemonOutbound) -> bool {
        let compact_allowed = {
            let state = client.state.lock().expect("daemon client poisoned");
            state.transport.as_deref() == Some("private-framed") && state.authentication_role.as_deref() != Some("session_client")
        };
        let value = message.to_value();
        if compact_allowed {
            if let Ok(Some(delta)) = create_compact_assistant_delta(&value) {
                return self.write_serialized_encoded(client, serialize_json_line(&serde_json::to_value(delta).expect("compact delta is serializable")).as_bytes(), message, "assistant-delta", None);
            }
        }
        self.write_serialized(client, &serialize_json_line(&value), Some(message))
    }

    /// `writeSerialized(client, line, message, payloadEncoding = "jsonl", snapshotPurpose?)`.
    ///
    /// The TypeScript body has two arms: a private-framed client gets
    /// `encodePrivateFrame` with the routing header, every other client gets the plain
    /// line. Both arms live here so `writeWorkerSnapshotBuffer` can pass a `Buffer`.
    fn write_serialized_encoded(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        line: &[u8],
        message: &DaemonOutbound,
        payload_encoding: &str,
        snapshot_purpose: Option<&str>,
    ) -> bool {
        let private_framed = client
            .state
            .lock()
            .expect("daemon client poisoned")
            .transport
            .as_deref()
            == Some("private-framed");
        let wire: Vec<u8> = if private_framed {
            let mut header = Map::new();
            header.insert("kind".to_string(), Value::String("outbound".to_string()));
            header.insert(
                "outboundType".to_string(),
                Value::String(message.type_name().to_string()),
            );
            header.insert(
                "payloadEncoding".to_string(),
                Value::String(payload_encoding.to_string()),
            );
            if let Some(snapshot_purpose) = snapshot_purpose {
                header.insert(
                    "snapshotPurpose".to_string(),
                    Value::String(snapshot_purpose.to_string()),
                );
            }
            if let Some(active_session_id) = message.active_session_id() {
                header.insert(
                    "activeSessionId".to_string(),
                    Value::String(active_session_id.to_string()),
                );
            }
            if let DaemonOutbound::Raw(value) = message {
                if let Some(snapshot_id) = value.get("snapshotId").and_then(Value::as_str) {
                    header.insert(
                        "snapshotId".to_string(),
                        Value::String(snapshot_id.to_string()),
                    );
                }
                if let Some(request_id) = value.get("id").and_then(Value::as_str) {
                    header.insert(
                        "requestId".to_string(),
                        Value::String(request_id.to_string()),
                    );
                }
            }
            if message.type_name() == "session_event" {
                if let Some(event_type) = message.to_value().get("event").and_then(|event| event.get("type")).and_then(Value::as_str) {
                    header.insert(
                        "sessionEventType".to_string(),
                        Value::String(event_type.to_string()),
                    );
                }
            }
            match encode_private_frame(&Value::Object(header), line) {
                Ok(frame) => frame,
                Err(_) => return false,
            }
        } else {
            line.to_vec()
        };
        let delivered = client.writer.write_bytes(wire);
        if !delivered {
            if let Some(active_session_id) = message.active_session_id() {
                self.mark_client_backpressured(client, active_session_id);
            }
        }
        delivered
    }

    /// `writeSerialized(client, serialized, message)`.
    fn write_serialized(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        serialized: &str,
        message: Option<&DaemonOutbound>,
    ) -> bool {
        if client.writer.destroyed() {
            return false;
        }
        let private_framed = client.state.lock().expect("daemon client poisoned").transport.as_deref() == Some("private-framed");
        if private_framed {
            if let Some(message) = message {
                return self.write_serialized_encoded(client, serialized.as_bytes(), message, "jsonl", None);
            }
        }
        if client.is_backpressured() {
            if let Some(message) = message {
                if is_sequenced_session_outbound(message.type_name()) {
                    if let Some(active_session_id) = message.active_session_id() {
                        self.queue_client_catchup(
                            client,
                            active_session_id,
                            if message.type_name() == "session_replaced" {
                                "replacement"
                            } else {
                                "resync"
                            },
                        );
                    }
                }
            }
            return false;
        }
        let delivered = client.writer.write(serialized.to_string());
        if !delivered {
            if let Some(message) = message {
                if let Some(active_session_id) = message.active_session_id() {
                    self.mark_client_backpressured(client, active_session_id);
                }
            }
        }
        delivered
    }

    /// The socket reported backpressure (`socket.write` returning false).
    fn mark_client_backpressured(self: &Arc<Self>, client: &Arc<DaemonClientHandle>, active_session_id: &str) {
        client.set_backpressured(true);
        self.queue_client_catchup(
            client,
            active_session_id,
            if client.snapshot_streaming() {
                "resync"
            } else {
                "resync"
            },
        );
        if !client.snapshot_streaming() {
            let client = Arc::clone(client);
            let daemon = self.clone_arc();
            tokio::spawn(async move {
                if let Err(error) = daemon.catch_up_backpressured_client(Arc::clone(&client)).await {
                    daemon.log(&format!(
                        "could not catch up snapshot client {}: {error}",
                        client.id()
                    ));
                }
            });
        }
    }

    /// `recordWorkerRecoveryState(state, operation, busyOverride?)`.
    fn record_worker_recovery_state(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        operation: &str,
        busy_override: Option<bool>,
    ) {
        let mut journal = self
            .recovery_journal
            .lock()
            .expect("recovery journal poisoned");
        let Some(journal) = journal.as_mut() else {
            return;
        };
        let entry = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(
                &state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id,
            )
            .cloned();
        let Some(entry) = entry else {
            return;
        };
        let session = &entry.session;
        let session_id = session.session_id();
        if session_id.is_empty() {
            return;
        }
        let busy = busy_override.unwrap_or_else(|| worker_recovery_busy(session.as_ref()));
        journal.record(WorkerRecoveryRecordInput {
            active_session_id: entry
                .state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                .clone(),
            session_id,
            session_file: session.session_file(),
            busy,
            operation: operation.to_string(),
        });
    }
}

impl AgentDaemon {
    /// `rosterEntryForSessionPath(canonicalPath)`.
    fn roster_entry_for_session_path(&self, canonical_path: &str) -> Option<WorkerRosterEntry> {
        self.roster_reporter
            .lock()
            .expect("roster reporter poisoned")
            .last_composed
            .values()
            .find(|entry| {
                entry
                    .summary
                    .session_file
                    .as_deref()
                    .map(canonical_session_path)
                    .as_deref()
                    == Some(canonical_path)
            })
            .cloned()
    }

    /// `rosterAgentIdForState(state)`.
    fn roster_agent_id_for_state(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> String {
        let entry = self.session_entry_for_state(state);
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        if metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.as_deref())
            == Some("subagent")
        {
            if let Some(rlm_child_id) = metadata
                .as_ref()
                .and_then(|metadata| metadata.rlm_child_id.clone())
            {
                return roster_agent_id_for_summary(
                    Some("subagent"),
                    Some(&rlm_child_id),
                    &rlm_child_id,
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_file.as_deref()),
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_active_session_id.as_deref()),
                );
            }
        }
        entry.session.session_id()
    }

    /// `rosterAgentIdForRlmChild(childId, parentSessionPath)`.
    fn roster_agent_id_for_rlm_child(
        &self,
        child_id: &str,
        parent_session_path: Option<&str>,
    ) -> String {
        roster_agent_id_for_summary(
            Some("subagent"),
            Some(child_id),
            child_id,
            parent_session_path,
            None,
        )
    }

    fn session_entry_for_state(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Arc<DaemonSessionState> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .cloned()
            .unwrap_or_else(|| {
                Arc::new(DaemonSessionState {
                    state: Arc::clone(state),
                    session: Arc::new(MissingSession::new(&active_session_id)),
                    runtime_metadata: AgentSessionRuntimeMetadata::default(),
                    snapshot_boundary: StdMutex::new(None),
                })
            })
    }

    /// `observeRosterEvent(state, message)`.
    fn observe_roster_event(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        message: &DaemonOutbound,
    ) {
        if !self.is_worker() {
            return;
        }
        match message {
            DaemonOutbound::SessionEvent { event, .. } => {
                let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");
                if event_type == "rlm_child_update" {
                    if let Some(child) = event.get("child") {
                        self.observe_roster_child_update(state, child);
                    }
                    return;
                }
                if !contains(&ROSTER_SESSION_EVENT_TRIGGERS, event_type) {
                    return;
                }
                if event_type == "agent_end" {
                    // AgentEnd is observed before Agent::finish_run clears streaming.
                    // Publish once more after the owning run and action queue settle.
                    let session = self.session_of(state);
                    let daemon = Arc::downgrade(self);
                    let stopped = self.server_stopped.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            _ = stopped.cancelled() => return,
                            _ = session.wait_for_idle() => {}
                        }
                        if let Some(daemon) = daemon.upgrade() {
                            daemon.schedule_roster_flush();
                        }
                    });
                }
            }
            DaemonOutbound::SessionStatus { .. }
            | DaemonOutbound::SessionClosed { .. }
            | DaemonOutbound::SessionReplaced { .. } => {}
            _ => return,
        }
        self.schedule_roster_flush();
    }

    /// `observeRosterChildUpdate(state, child)`.
    fn observe_roster_child_update(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        child: &Value,
    ) {
        let bound = child
            .get("activeSessionId")
            .is_some_and(|value| !value.is_null())
            || child
                .get("id")
                .and_then(Value::as_str)
                .map(|child_id| self.has_session_for_rlm_child(state, child_id))
                .unwrap_or(false);
        let entry = self.queued_child_roster_entry(state, child);
        let mut reporter = self
            .roster_reporter
            .lock()
            .expect("roster reporter poisoned");
        let status = child.get("status").and_then(Value::as_str).unwrap_or("");
        if !bound && (status == "queued" || status == "running") {
            reporter
                .queued_children
                .insert(entry.agent_id.clone(), entry);
        } else {
            reporter.queued_children.remove(&entry.agent_id);
        }
        drop(reporter);
        self.schedule_roster_flush();
    }

    /// `hasSessionForRlmChild(parentState, childId)`.
    fn has_session_for_rlm_child(
        &self,
        parent_state: &Arc<StdMutex<ActiveSessionState>>,
        child_id: &str,
    ) -> bool {
        let parent_active_session_id = parent_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.session_states().into_iter().any(|entry| {
            entry.runtime_metadata.rlm_child_id.as_deref() == Some(child_id)
                && entry.runtime_metadata.parent_active_session_id.as_deref()
                    == Some(parent_active_session_id.as_str())
        })
    }

    /// `queuedChildRosterEntry(state, child)`.
    fn queued_child_roster_entry(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        child: &Value,
    ) -> WorkerRosterEntry {
        let entry = self.session_entry_for_state(state);
        let parent_session = &entry.session;
        let child_id = child
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // Drop the manager guard before session_id/session_file lock it again.
        let cwd = parent_session
            .session_manager()
            .lock()
            .expect("session manager poisoned")
            .get_cwd();
        let summary = RosterSessionSummary {
            id: child_id.clone(),
            lifecycle: "live".to_string(),
            activity: "idle".to_string(),
            is_session_active: false,
            runtime_kind: Some("subagent".to_string()),
            rlm_depth: Some(parent_session.rlm_depth().unwrap_or(0) + 1),
            session_id: child_id.clone(),
            session_name: child
                .get("sessionName")
                .and_then(Value::as_str)
                .map(str::to_string),
            cwd,
            is_streaming: false,
            is_compacting: false,
            attached_clients: 0,
            message_count: 0,
            first_message: child
                .get("label")
                .and_then(Value::as_str)
                .map(str::to_string),
            parent_active_session_id: Some(
                state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
            ),
            parent_session_id: Some(parent_session.session_id()),
            parent_session_path: parent_session.session_file(),
            rlm_child_id: Some(child_id),
            ..RosterSessionSummary::default()
        };
        WorkerRosterEntry {
            agent_id: crate::modes::daemon::agent_roster::roster_agent_id_for_entry(&summary),
            queued_child: Some(true),
            seeded_cwd: None,
            summary,
        }
    }

    /// `scheduleRosterFlush()`.
    fn schedule_roster_flush(self: &Arc<Self>) {
        if !self.is_worker()
            || self.roster_flush_scheduled.load(Ordering::SeqCst)
            || self.shutting_down.load(Ordering::SeqCst)
        {
            return;
        }
        self.roster_flush_scheduled.store(true, Ordering::SeqCst);
        let daemon = Arc::clone(self);
        tokio::spawn(async move {
            daemon.roster_flush_scheduled.store(false, Ordering::SeqCst);
            if let Err(error) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let daemon = Arc::clone(&daemon);
                daemon.flush_roster_now();
            })) {
                let _ = error;
                daemon.log("could not publish roster delta: roster flush panicked");
            }
        });
    }

    fn flush_roster_now(self: &Arc<Self>) {
        // TypeScript reads the live session here. Event-time projections can
        // still say busy when a turn or detached child has since settled.
        for entry in self.session_states() {
            let active = entry.session.is_session_active();
            let foreground = entry.session.is_foreground_active();
            let streaming = entry.session.is_streaming();
            let compacting = entry.session.is_compacting();
            let children = entry.session.has_running_rlm_children();
            let mut state = entry.state.lock().expect("active session poisoned");
            state.runtime.session.is_session_active = active;
            state.runtime.session.is_foreground_active = Some(foreground);
            state.runtime.session.is_streaming = streaming;
            state.runtime.session.is_compacting = compacting;
            state.runtime.session.has_running_rlm_children = children;
        }
        let scheduled_jobs = self.cron_store.list();
        let mut entries: HashMap<String, WorkerRosterEntry> = HashMap::new();
        // `build_session_list` yields the daemon's `SessionSummary` projection, which
        // `roster_view()` narrows to the `RosterSessionSummary` the roster path serializes.
        for summary in build_session_list(&self.state_refs(), &[], &scheduled_jobs) {
            let entry = worker_roster_entry_from_summary(&summary.roster_view());
            entries.insert(entry.agent_id.clone(), entry);
        }
        {
            let mut reporter = self
                .roster_reporter
                .lock()
                .expect("roster reporter poisoned");
            let queued_ids: Vec<String> = reporter.queued_children.keys().cloned().collect();
            for agent_id in queued_ids {
                if entries.contains_key(&agent_id) {
                    reporter.queued_children.remove(&agent_id);
                    continue;
                }
                if let Some(queued) = reporter.queued_children.get(&agent_id) {
                    entries.insert(agent_id.clone(), queued.clone());
                }
            }
            // A terminal unbound child run owns no transcript: it is a removal, never
            // a passivated row. A vanished row whose state lives on under a new
            // sessionId was swapped in place (new_session/switch/fork): also a removal.
            let mut composed_active_ids: HashSet<String> = HashSet::new();
            for entry in entries.values() {
                if let Some(active_session_id) = &entry.summary.active_session_id {
                    composed_active_ids.insert(active_session_id.clone());
                }
            }
            for (agent_id, previous) in reporter.last_composed.clone() {
                if entries.contains_key(&agent_id) {
                    continue;
                }
                let swapped = previous
                    .summary
                    .active_session_id
                    .as_ref()
                    .map(|active_session_id| composed_active_ids.contains(active_session_id))
                    .unwrap_or(false);
                if previous.queued_child == Some(true) || swapped {
                    reporter
                        .removed_agent_ids
                        .insert(agent_id, Some(previous.summary.session_id.clone()));
                }
            }
            let removed: Vec<(String, Option<String>)> = reporter
                .removed_agent_ids
                .iter()
                .map(|(agent_id, session_id)| (agent_id.clone(), session_id.clone()))
                .collect();
            for (agent_id, target_session_id) in removed {
                let composed = entries.get(&agent_id);
                // A new incarnation cancels the stale removal, as does a revived
                // resident top-level row (switch-back, resume-after-archive); a
                // resident subagent row with the removed sessionId is the
                // mid-teardown race and stays suppressed.
                let revived = composed
                    .and_then(|composed| composed.summary.active_session_id.as_ref())
                    .is_some()
                    && composed.and_then(|composed| composed.summary.runtime_kind.as_deref())
                        != Some("subagent");
                if let Some(composed) = composed {
                    if composed.queued_child == Some(true)
                        || Some(composed.summary.session_id.clone()) != target_session_id
                        || revived
                    {
                        reporter.removed_agent_ids.remove(&agent_id);
                        continue;
                    }
                }
                entries.remove(&agent_id);
                reporter.queued_children.remove(&agent_id);
            }
            let registrations = scheduled_job_registrations(&scheduled_jobs);
            let previous_entries: Vec<(String, WorkerRosterEntry)> = reporter
                .last_composed
                .iter()
                .map(|(agent_id, entry)| (agent_id.clone(), entry.clone()))
                .collect();
            for (agent_id, previous) in previous_entries {
                if !entries.contains_key(&agent_id)
                    && !reporter.removed_agent_ids.contains_key(&agent_id)
                {
                    let file = previous.summary.session_file.as_deref().map(resolve_path);
                    entries.insert(
                        agent_id,
                        passivated_worker_roster_entry(
                            &previous,
                            Some(RegisteredHeartbeatFlags {
                                has_registered_heartbeat: file
                                    .as_ref()
                                    .map(|file| {
                                        registrations.heartbeat_session_files.contains(file)
                                    })
                                    .unwrap_or(false),
                                has_registered_cron_job: file
                                    .as_ref()
                                    .map(|file| registrations.cron_session_files.contains(file))
                                    .unwrap_or(false),
                            }),
                        ),
                    );
                }
            }
            let mut changed: Vec<WorkerRosterEntry> = Vec::new();
            let mut next_json: HashMap<String, String> = HashMap::new();
            for entry in entries.values() {
                let json = serde_json::to_string(entry).unwrap_or_default();
                next_json.insert(entry.agent_id.clone(), json.clone());
                if reporter.last_composed_json.get(&entry.agent_id) != Some(&json) {
                    changed.push(entry.clone());
                }
            }
            let removed_agent_ids: Vec<String> =
                reporter.removed_agent_ids.keys().cloned().collect();
            reporter.last_composed = entries.clone().into_iter().collect();
            reporter.last_composed_json = next_json;
            let snapshot_pending = reporter.snapshot_pending;
            drop(reporter);
            if !self.has_authenticated_supervisor_client() {
                let mut reporter = self
                    .roster_reporter
                    .lock()
                    .expect("roster reporter poisoned");
                if !changed.is_empty() || !removed_agent_ids.is_empty() {
                    reporter.snapshot_pending = true;
                }
                return;
            }
            if snapshot_pending {
                let entries_vec: Vec<WorkerRosterEntry> = entries.values().cloned().collect();
                let delivered =
                    self.broadcast_roster_frame(&DaemonWorkerRosterOutbound::RosterDelta {
                        entries: entries_vec,
                        removed_agent_ids: (!removed_agent_ids.is_empty())
                            .then(|| removed_agent_ids.clone()),
                        snapshot: Some(true),
                    });
                let mut reporter = self
                    .roster_reporter
                    .lock()
                    .expect("roster reporter poisoned");
                if delivered {
                    reporter.snapshot_pending = false;
                    reporter.removed_agent_ids.clear();
                }
                return;
            }
            if changed.is_empty() && removed_agent_ids.is_empty() {
                return;
            }
            let delivered = self.broadcast_roster_frame(&DaemonWorkerRosterOutbound::RosterDelta {
                entries: changed,
                removed_agent_ids: (!removed_agent_ids.is_empty())
                    .then(|| removed_agent_ids.clone()),
                snapshot: None,
            });
            let mut reporter = self
                .roster_reporter
                .lock()
                .expect("roster reporter poisoned");
            if delivered {
                reporter.removed_agent_ids.clear();
            } else {
                reporter.snapshot_pending = true;
            }
        }
    }

    /// `hasAuthenticatedSupervisorClient()`.
    fn has_authenticated_supervisor_client(&self) -> bool {
        let claims = self
            .supervisor_claims
            .lock()
            .expect("supervisor claims poisoned");
        self.client_handles().into_iter().any(|client| {
            claims.contains_key(&(Arc::as_ptr(&client) as usize)) && !client.writer.destroyed()
        })
    }

    /// `broadcastRosterFrame(message)` (`daemon-mode.ts:7359-7373`).
    ///
    /// The TypeScript has no transport test: every client carrying a supervisor
    /// claim gets `encodePrivateFrame({ kind: "outbound", outboundType:
    /// message.type }, payload)` (`:7367-7369`), and the bare JSONL line does not
    /// exist there. Reachability of the former Rust `transport == "private-framed"`
    /// branch was checked and it is unreachable:
    /// - `client.state.transport` has exactly one producer, `daemon_server.rs:72`
    ///   (`Some(if daemon.is_worker() { "private-framed" } else { "jsonl" })`),
    ///   mirroring TS `transport: this.options.worker ? "private-framed" : "jsonl"`
    ///   (`daemon-mode.ts:3540`);
    /// - `supervisorClaims`/`supervisor_claims` is only ever inserted in the
    ///   worker-only `worker_auth` handler (`daemon_mode.rs:3770`, inside
    ///   `if self.is_worker() && !client.authenticated()` at `:3594`; TS
    ///   `daemon-mode.ts:3787`), and `options.worker` is never reassigned.
    /// So claim => worker daemon => `"private-framed"`. Writing the bare line here
    /// would desynchronize the peer's frame decoder: it reads the first four bytes
    /// of `{"type"...` as a big-endian header length.
    fn broadcast_roster_frame(&self, message: &DaemonWorkerRosterOutbound) -> bool {
        let payload = serde_json::to_vec(message).unwrap_or_default();
        let outbound_type = serde_json::to_value(message)
            .ok()
            .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| "roster_delta".to_string());
        let header = serde_json::json!({ "kind": "outbound", "outboundType": outbound_type });
        // TS `encodePrivateFrame` throws on an impossible header (empty or over the
        // 1 MiB limit) and that throw escapes `broadcastRosterFrame`. The frame header
        // here is a fixed two-field object, so encoding cannot fail; if it ever does,
        // report no delivery rather than claim a frame that was never written.
        let framed = match encode_private_frame(&header, &payload) {
            Ok(framed) => framed,
            Err(error) => {
                self.log(&format!("Daemon roster frame encode failed: {error}"));
                return false;
            }
        };
        let mut delivered = false;
        for client in self.client_handles() {
            // TS `daemon-mode.ts:7363-7370`: skip every client without a supervisor
            // claim (or with a destroyed socket), then write the encoded frame and
            // set `delivered = true` unconditionally - no transport test, no JSONL
            // fallback, and no per-write result check.
            if !self
                .supervisor_claims
                .lock()
                .expect("supervisor claims poisoned")
                .contains_key(&(Arc::as_ptr(&client) as usize))
                || client.writer.destroyed()
            {
                continue;
            }
            client.writer.write_bytes(framed.clone());
            delivered = true;
        }
        delivered
    }
}

impl AgentDaemon {
    /// `createAttachResult(client, state, command)`.
    async fn create_attach_result(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        command: &ParsedDaemonCommand,
    ) -> Result<DaemonAttachResult, String> {
        let state_active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let capabilities = client.capabilities_for_session(&state_active_session_id);
        let snapshot = self
            .create_session_snapshot(state, capabilities.contains("history_ranges"))
            .await?;
        let last_event_sequence = snapshot.get("lastEventSequence").and_then(Value::as_u64).unwrap_or(0);
        let event_generation = snapshot.get("lastEventCursor").and_then(|cursor| cursor.get("generation")).and_then(Value::as_str).unwrap_or("").to_string();
        // `command.resumeCursor` is a `DaemonResumeCursor` (daemon-protocol.ts:350).
        let resume_cursor = command
            .body
            .get("resumeCursor")
            .and_then(|value| {
                serde_json::from_value::<crate::modes::daemon::daemon_protocol::DaemonResumeCursor>(
                    value.clone(),
                )
                .ok()
            });
        let replay = match resume_cursor
            .as_ref()
            .and_then(|cursor| cursor.active_session_id.as_deref())
        {
            Some(resume_active_session_id)
                if resume_active_session_id != state_active_session_id =>
            {
                let from_sequence = resume_cursor
                    .as_ref()
                    .map(|cursor| cursor.resume_sequence())
                    .unwrap_or(0);
                serde_json::json!({
                    "status": "unavailable",
                    "fromSequence": from_sequence,
                    "toSequence": last_event_sequence,
                    "toCursor": { "generation": event_generation, "sequence": last_event_sequence },
                    "reason": "resume_cursor_session_mismatch",
                })
            }
            _ => serde_json::to_value(create_daemon_replay_info(
                resume_cursor.as_ref(),
                last_event_sequence,
                &event_generation,
            ))
            .unwrap_or(Value::Null),
        };
        // Slim clients read summary/messages from the snapshot; duplicating them at
        // the top level would serialize the full history twice more per attach.
        let slim = capabilities.contains("slim_attach");
        let snapshot_summary = snapshot.get("summary").cloned().unwrap_or(Value::Null);
        let snapshot_messages = snapshot
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(DaemonAttachResult {
            protocol: Some(
                serde_json::to_value(
                    crate::modes::daemon::daemon_protocol::daemon_protocol_info(),
                )
                .unwrap_or(Value::Null),
            ),
            active_session_id: state_active_session_id,
            state: (!slim).then_some(snapshot_summary),
            messages: (!slim).then_some(snapshot_messages),
            snapshot,
            replay,
            last_event_sequence,
            last_event_cursor: Some(serde_json::json!({
                "generation": event_generation,
                "sequence": last_event_sequence,
            })),
            snapshot_stream: None,
            client: Some(serde_json::json!({
                "id": client.id(),
                "capabilities": capabilities.into_iter().collect::<Vec<_>>(),
            })),
        })
    }

    /// `createSessionSnapshot(state, recentFirstHistory = false)`.
    async fn create_session_snapshot(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        recent_first_history: bool,
    ) -> Result<Value, String> {
        let entry = self.session_entry_for_state(state);
        entry.sync_event_view(false);
        let metadata = entry.runtime_metadata.clone();
        let parent = if metadata.parent_active_session_id.is_some()
            || metadata.parent_session_id.is_some()
            || metadata.rlm_parent_node_id.is_some()
            || metadata.rlm_child_id.is_some()
        {
            let mut object = Map::new();
            if let Some(value) = &metadata.parent_active_session_id {
                object.insert("activeSessionId".to_string(), Value::String(value.clone()));
            }
            if let Some(value) = &metadata.parent_session_id {
                object.insert("sessionId".to_string(), Value::String(value.clone()));
            }
            if let Some(value) = &metadata.rlm_parent_node_id {
                object.insert("nodeId".to_string(), Value::String(value.clone()));
            }
            if let Some(value) = &metadata.rlm_child_id {
                object.insert("childId".to_string(), Value::String(value.clone()));
            }
            Some(Value::Object(object))
        } else {
            None
        };
        let mut session_identity = entry.session.session_id();
        let mut children = self.build_rlm_child_snapshots_with_passive(state).await?;
        for _ in 0..MAX_SESSION_SNAPSHOT_STABILIZATION_RETRIES {
            entry.sync_event_view(false);
            let current = entry.session.session_id();
            if current == session_identity {
                break;
            }
            session_identity = current;
            children = self.build_rlm_child_snapshots_with_passive(state).await?;
        }
        entry.sync_event_view(false);
        let mut connection_state = self.create_connection_state(state);
        let (published_messages, published_streaming, state_flags, last_event_sequence, event_generation) = {
            let mut published = entry.snapshot_boundary.lock().expect("snapshot boundary poisoned");
            if published.is_none() {
                let messages = Arc::new(entry.session.messages());
                let state = state.lock().expect("active session poisoned");
                *published = Some(PublishedTranscript { messages, streaming_message: state.runtime.session.streaming_message.clone(), state_flags: (state.runtime.session.is_session_active, state.runtime.session.is_streaming, state.runtime.session.is_compacting), sequence: state.last_event_sequence, generation: state.event_generation.clone() });
            }
            let published = published.as_ref().expect("published transcript initialized");
            (published.messages.clone(), published.streaming_message.clone(), published.state_flags, published.sequence, published.generation.clone())
        };
        if let Some(state) = connection_state.as_object_mut() {
            state.insert("isSessionActive".to_string(), Value::Bool(state_flags.0));
            state.insert("isStreaming".to_string(), Value::Bool(state_flags.1));
            state.insert("isCompacting".to_string(), Value::Bool(state_flags.2));
        }
        // Runtime hydration still parses the complete JSONL once. This capability
        // only avoids serializing/transferring/rendering the complete resident
        // transcript on attach.
        let mut history: Option<SessionHistorySnapshot> = None;
        if recent_first_history {
            let persisted_context = entry
                .session
                .session_manager()
                .lock()
                .expect("session manager poisoned")
                .build_session_context_with_entry_ids(None);
            let live_messages = published_messages.as_ref();
            let aligns = persisted_context.messages.len() == live_messages.len()
                && persisted_context
                    .messages
                    .iter()
                    .zip(live_messages.iter())
                    .all(|(left, right)| {
                        left.role() == right.role()
                            && message_timestamp_of(left) == message_timestamp_of(right)
                    });
            if aligns {
                // Keep the exact public messages (including in-memory detail
                // augmentation), while using persisted entry ids to present the same
                // context chronologically. `{ ...persistedContext, messages: liveMessages }`
                // (daemon-mode.ts:5503) replaces the message list, keeping the entry ids.
                let live_context = crate::core::session_manager::SessionContextWithEntryIds {
                    messages: live_messages.clone(),
                    ..persisted_context.clone()
                };
                let ordered = order_session_context_for_transcript(&live_context);
                history = Some(SessionHistorySnapshot {
                    messages: ordered.messages,
                    entry_ids: ordered.entry_ids,
                    tip_entry_id: entry
                        .session
                        .session_manager()
                        .lock()
                        .expect("session manager poisoned")
                        .get_leaf_id(),
                });
            }
            // Unpersisted outcome insertion can temporarily break index alignment. In
            // that case retain the full legacy snapshot rather than dropping it.
        }
        let initial_history = match &history {
            Some(history) => Some(slice_pinned_session_history(
                history,
                &SlicePinnedSessionHistoryOptions {
                    generation: event_generation.clone(),
                    representation: session_history_representation(
                        entry.session.model_identity().as_ref(),
                    ),
                    limit: Some(INITIAL_HISTORY_WINDOW_MESSAGES),
                    before_entry_id: None,
                },
            )?),
            None => None,
        };
        let messages = match &initial_history {
            Some(history) => history.messages.clone(),
            None => published_messages.as_ref().clone(),
        };
        let messages = messages
            .into_iter()
            .map(|message| serde_json::to_value(message).unwrap_or(Value::Null))
            .collect::<Vec<_>>();
        let mut snapshot = Map::new();
        snapshot.insert(
            "activeSessionId".to_string(),
            Value::String(
                state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
            ),
        );
        snapshot.insert(
            "summary".to_string(),
            serde_json::to_value(self.summary_for_state(state)).unwrap_or(Value::Null),
        );
        snapshot.insert("state".to_string(), connection_state);
        if let Some(summary) = snapshot.get_mut("summary").and_then(Value::as_object_mut) {
            summary.insert("streamingMessage".to_string(), serde_json::to_value(published_streaming).unwrap_or(Value::Null));
        }
        snapshot.insert("messages".to_string(), Value::Array(messages));
        if let Some(history) = &initial_history {
            let mut window = Map::new();
            window.insert("version".to_string(), Value::from(history.version as f64));
            window.insert(
                "generation".to_string(),
                Value::String(history.generation.clone()),
            );
            window.insert(
                "representation".to_string(),
                Value::String(history.representation.clone()),
            );
            window.insert(
                "tipEntryId".to_string(),
                history
                    .tip_entry_id
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            window.insert(
                "totalMessageCount".to_string(),
                Value::from(history.total_message_count as f64),
            );
            window.insert(
                "startIndex".to_string(),
                Value::from(history.start_index as f64),
            );
            window.insert(
                "entryIds".to_string(),
                Value::Array(
                    history
                        .entry_ids
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
            window.insert("hasOlder".to_string(), Value::Bool(history.has_older));
            window.insert("order".to_string(), Value::String(history.order.clone()));
            snapshot.insert("history".to_string(), Value::Object(window));
        }
        // Omit duplicate heavy payloads from attach: the client can derive render
        // context from messages + state.
        snapshot.insert(
            "lastEventSequence".to_string(),
            Value::from(last_event_sequence),
        );
        snapshot.insert(
            "lastEventCursor".to_string(),
            serde_json::json!({ "generation": event_generation, "sequence": last_event_sequence }),
        );
        if let Some(parent) = parent {
            snapshot.insert("parent".to_string(), parent);
        }
        snapshot.insert("children".to_string(), Value::Array(children));
        Ok(Value::Object(snapshot))
    }

    /// `createState` for a snapshot stream that failed after the client attached.
    fn finish_attach_failure(
        &self,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        streams_snapshot: bool,
        _error: Option<String>,
    ) {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        {
            let mut state_guard = state.lock().expect("active session poisoned");
            state_guard.pending_attaches = state_guard.pending_attaches.saturating_sub(1);
            state_guard.clients.retain(|candidate| !Arc::ptr_eq(candidate, &client.state));
        }
        client.state.lock().expect("daemon client poisoned").attached_active_session_ids.remove(&active_session_id);
        client.deferred_snapshot_frames.lock().expect("deferred frames poisoned").remove(&active_session_id);
        remove_daemon_client_session_capabilities(client, &active_session_id);
        if streams_snapshot {
            finish_client_snapshot_streaming(client, &active_session_id);
        }
    }

    /// The chunked-snapshot stream started on attach
    /// (`createSnapshotTranscriptChunks` + `streamWorkerSnapshot`).
    fn start_snapshot_stream(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        snapshot_id: &str,
        snapshot_messages: &[Value],
        snapshot: &Value,
    ) -> Result<(), String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let existing_signal = { client.snapshot_transfer_abort_controllers.lock().expect("snapshot abort controllers poisoned").get(&active_session_id).cloned() };
        let signal = existing_signal.unwrap_or_else(|| mark_client_snapshot_streaming(client, &active_session_id));
        let messages: Vec<AgentMessage> = snapshot_messages
            .iter()
            .filter_map(|value| serde_json::from_value(value.clone()).ok())
            .collect();
        let transcript = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            create_snapshot_transcript_chunks(CreateSnapshotTranscriptChunksOptions {
                active_session_id: active_session_id.clone(),
                snapshot_id: snapshot_id.to_string(),
                messages,
                target_chunk_bytes: Some(SNAPSHOT_TARGET_CHUNK_BYTES),
                aborted: signal.is_cancelled(),
            })
        })) {
            Ok(chunks) => chunks,
            Err(_) => {
                self.detach_failed_stream(client, state);
                return Err("The snapshot transcript was aborted".to_string());
            }
        };
        if signal.is_cancelled() {
            self.detach_failed_stream(client, state);
            return Err("The snapshot transcript was aborted".to_string());
        }
        let daemon = Arc::clone(self);
        let client = Arc::clone(client);
        let state = Arc::clone(state);
        let snapshot = snapshot.clone();
        let message_count = snapshot_messages.len();
        let snapshot_id = snapshot_id.to_string();
        tokio::spawn(async move {
            if let Err(error) = daemon
                .stream_worker_snapshot(
                    &client,
                    &state,
                    &snapshot_id,
                    snapshot,
                    message_count,
                    transcript,
                    // `streamWorkerSnapshot(client, streamedResult, transcript, "attach", ...)`
                    // (`daemon-mode.ts:4296-4302`).
                    "attach",
                    signal,
                    true,
                )
                .await
            {
                daemon.log(&format!("could not stream attach snapshot: {error}"));
            }
        });
        Ok(())
    }

    fn detach_failed_stream(
        &self,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        {
            let mut state_guard = state.lock().expect("active session poisoned");
            state_guard
                .clients
                .retain(|candidate| !Arc::ptr_eq(candidate, &client.state));
        }
        client
            .state
            .lock()
            .expect("daemon client poisoned")
            .attached_active_session_ids
            .remove(&active_session_id);
        remove_daemon_client_session_capabilities(client, &active_session_id);
        let mut client_mut = Arc::clone(client);
        finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
    }

    /// `streamWorkerSnapshot(client, result, transcript, purpose, signal, snapshotAlreadyMarked)`.
    ///
    /// `purpose` is `"attach" | "replacement" | "catchup"` (`daemon-mode.ts:5554`), the
    /// same three values the private-frame header validator accepts
    /// (`daemon_worker_protocol.rs:97-99` -> `daemon-worker-protocol.ts:257-280`).
    #[allow(clippy::too_many_arguments)]
    async fn stream_worker_snapshot(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        stream_id: &str,
        snapshot: Value,
        message_count: usize,
        transcript: crate::modes::daemon::snapshot_transcript_cache::SnapshotTranscriptChunks,
        purpose: &str,
        transfer_signal: tokio_util::sync::CancellationToken,
        snapshot_already_marked: bool,
    ) -> Result<(), String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if snapshot_already_marked && transfer_signal.is_cancelled() && message_count == 0 {
            // A marked transfer without a live signal cannot be finished safely.
        }
        if client.writer.destroyed() {
            let mut client_mut = Arc::clone(client);
            finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
            return Ok(());
        }
        let mut snapshot_without_messages = snapshot.clone();
        let last_event_sequence = snapshot.get("lastEventSequence").and_then(Value::as_i64).ok_or("Snapshot is missing its captured event sequence")?;
        let event_generation = snapshot.get("lastEventCursor").and_then(|cursor| cursor.get("generation")).and_then(Value::as_str).ok_or("Snapshot is missing its captured event generation")?.to_string();
        if let Some(object) = snapshot_without_messages.as_object_mut() {
            object.remove("messages");
        }
        let snapshot_begin = serde_json::json!({
            "type": "session_snapshot_begin",
            "activeSessionId": active_session_id,
            "snapshotId": stream_id,
            "snapshot": snapshot_without_messages,
            "messageCount": message_count as f64,
            "targetChunkBytes": SNAPSHOT_TARGET_CHUNK_BYTES as f64,
            // `purpose === "catchup" ? "resync" : purpose` (`daemon-mode.ts:5593`).
            "purpose": if purpose == "catchup" { "resync" } else { purpose },
        });
        // `deliverSnapshotFailure` awaits the record write (`daemon-mode.ts:5616-5630`),
        // so the closure yields a future here too. The future owns its captures because
        // the record write is awaited by callers that keep borrowing `client`.
        let deliver_failure = |error: &str| {
            let daemon = Arc::clone(self);
            let client = Arc::clone(client);
            let purpose = purpose.to_string();
            let payload = serde_json::json!({
                "type": "session_snapshot_failed",
                "activeSessionId": active_session_id.clone(),
                "snapshotId": stream_id,
                "error": error,
            });
            async move {
                daemon
                    .write_worker_snapshot_record(
                        &client,
                        &payload,
                        &purpose,
                        Some(WORKER_SNAPSHOT_TERMINAL_DRAIN_TIMEOUT_MS),
                    )
                    .await
            }
        };
        if transfer_signal.is_cancelled() {
            let message = format!("Snapshot {stream_id} was aborted");
            deliver_failure(&message).await;
            let mut client_mut = Arc::clone(client);
            finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
            return Ok(());
        }
        if !self
            .write_worker_snapshot_record(client, &snapshot_begin, purpose, Some(0))
            .await
        {
            if transfer_signal.is_cancelled() {
                let message = format!("Snapshot {stream_id} was aborted");
                deliver_failure(&message).await;
            }
            let mut client_mut = Arc::clone(client);
            finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
            return Ok(());
        }
        let mut chunk_count = 0usize;
        let mut chunks = transcript;
        let mut completed = false;
        loop {
            if transfer_signal.is_cancelled() {
                let message = format!("Snapshot {stream_id} was aborted");
                deliver_failure(&message).await;
                break;
            }
            let Some(chunk) = chunks.next() else {
                // `session_snapshot_end` closes the transfer.
                let end = serde_json::json!({
                    "type": "session_snapshot_end",
                    "activeSessionId": active_session_id,
                    "snapshotId": stream_id,
                    "chunkCount": chunk_count as f64,
                    "lastEventSequence": last_event_sequence,
                    "lastEventCursor": { "generation": event_generation, "sequence": last_event_sequence },
                });
                completed = self.write_worker_snapshot_record(client, &end, purpose, Some(0)).await;
                break;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    let message = format!("Snapshot {stream_id} was aborted: {error}");
                    deliver_failure(&message).await;
                    break;
                }
            };
            let mut line = match std::str::from_utf8(&chunk) {
                Ok(line) => line.to_string(),
                Err(_) => String::new(),
            };
            if line.is_empty() {
                line = format!(
                    "{{\"type\":\"session_snapshot_chunk\",\"activeSessionId\":{},\"snapshotId\":{},\"index\":{},\"messages\":[]}}\n",
                    json_string(&active_session_id),
                    json_string(stream_id),
                    chunk_count
                );
            }
            // The chunk is a `session_snapshot_chunk` outbound; route it through the same
            // `writeSerialized` path as its neighbours (`daemon-mode.ts:5660` calls
            // `writeWorkerSnapshotBuffer`, which frames for a private-framed client).
            let chunk_value: Value = serde_json::from_str(line.trim())
                .unwrap_or_else(|_| Value::Null);
            if !self
                .write_worker_snapshot_record(client, &chunk_value, purpose, None)
                .await
            {
                client.set_backpressured(true);
                if transfer_signal.is_cancelled() {
                    let message = format!("Snapshot {stream_id} was aborted");
                    deliver_failure(&message).await;
                }
                break;
            }
            chunk_count += 1;
        }
        if !completed && !transfer_signal.is_cancelled() {
            self.queue_client_catchup(client, &active_session_id, if purpose == "replacement" { "replacement" } else { "resync" });
        }
        self.finish_snapshot_and_replay(client, &active_session_id, last_event_sequence, &event_generation, !completed || transfer_signal.is_cancelled());
        // A COMPLETED replacement/catch-up transfer just replaced the
        // client's session view (the client applies the chunked snapshot as
        // a replace or resync once `session_snapshot_end` lands, and that
        // application resets the status surface), so the authoritative Jev
        // footer must follow the stream. Attach keeps its own push at the
        // attach arm. An aborted transfer must NOT push here: its queued
        // catch-up delivers a fresh frame and pushes after that delivery,
        // and a premature push would be wiped by the later frame's reset.
        // The push runs AFTER `finish_snapshot_and_replay` returned, so the
        // streaming mark is already cleared and the footer frames are never
        // deferred into an entry the replay removed.
        if completed && (purpose == "replacement" || purpose == "catchup") {
            self.publish_jev_attach_footer(client, &state);
        }
        if !client.snapshot_streaming() {
            // Box only this edge of the catch-up cycle; every call and await stays the same.
            let catchup = Box::pin(self.catch_up_backpressured_client(Arc::clone(client)));
            if let Err(error) = catchup.await {
                self.log(&format!(
                    "could not catch up snapshot client {}: {error}",
                    client.id()
                ));
            }
        }
        Ok(())
    }

    /// `writeWorkerSnapshotRecord(client, message, purpose, signal?, drainTimeoutMs?)`.
    ///
    /// `purpose` is the private-frame header field `snapshotPurpose`. Valid values are
    /// exactly `attach | replacement | catchup` (`daemon_worker_protocol.rs:97-99`
    /// validates that set against `daemon-worker-protocol.ts:257-280`), matching the
    /// TypeScript parameter type `daemon-mode.ts:5710`.
    async fn write_worker_snapshot_record(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        message: &Value,
        purpose: &str,
        drain_timeout_ms: Option<u64>,
    ) -> bool {
        if client.writer.destroyed() {
            return false;
        }
        // `daemon-mode.ts:5714-5721` forwards `purpose` into `writeWorkerSnapshotBuffer`,
        // which passes it to `writeSerialized(..., "jsonl", purpose)` (`:5735`) - the only
        // place the header field is written (`daemon-mode.ts:7647`), so the raw `write`
        // path cannot carry it.
        //
        // Gap: `writeWorkerSnapshotBuffer`'s drain branch
        // (`daemon-mode.ts:5738-5768`) is still unported inside
        // `write_worker_snapshot_buffer` below, so a rejected `socket.write` never waits
        // for `drain`; that helper returns `false` in that case, exactly as here.
        let raw = DaemonOutbound::Raw(message.clone());
        let line = serialize_json_line(message);
        let aborted = false;
        self.write_worker_snapshot_buffer(client, line.into_bytes(), &raw, purpose, aborted, drain_timeout_ms)
            .await
    }

    /// `createConnectionState(state)`.
    fn create_connection_state(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> Value {
        let entry = self.session_entry_for_state(state);
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut connection_state = entry.session.connection_state(Some(active_session_id.clone()));
        let heartbeat = self
            .cron_store
            .get_latest_heartbeat(&active_session_id)
            .map(|job| job_to_value(&job))
            .unwrap_or(Value::Null);
        if let Some(object) = connection_state.as_object_mut() {
            object.insert("heartbeat".to_string(), heartbeat);
        }
        let recap = state
            .lock()
            .expect("active session poisoned")
            .summary_state
            .as_ref()
            .map(|summary| summary.summary.clone());
        if let Some(recap) = recap {
            if let Some(object) = connection_state.as_object_mut() {
                object.insert("recap".to_string(), Value::String(recap));
            }
        }
        connection_state
    }
}

/// The agent-connection entry lists `createAgentConnectionCommands` and
/// `createAgentConnectionResourceSnapshot` consume. Private plumbing: the
/// session slice supplies these from its own registries.
#[derive(Default)]
pub struct DaemonConnectionView {
    pub session_id: String,
    pub cwd: String,
    pub registered_commands: Vec<crate::modes::agent_connection::snapshot::RegisteredCommandEntry>,
    pub prompt_templates: Vec<crate::modes::agent_connection::snapshot::PromptTemplateEntry>,
    pub skills: Vec<crate::modes::agent_connection::snapshot::SkillEntry>,
    pub agents_files: Vec<crate::modes::agent_connection::snapshot::AgentsFileEntry>,
    pub resource_skills: Vec<crate::modes::agent_connection::snapshot::ResourceSkillEntry>,
    pub resource_prompts: Vec<crate::modes::agent_connection::snapshot::ResourcePromptEntry>,
    pub resource_extensions: Vec<crate::modes::agent_connection::snapshot::ResourceExtensionEntry>,
    pub resource_themes: Vec<crate::modes::agent_connection::snapshot::ResourceThemeEntry>,
    pub skill_diagnostics:
        Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
    pub prompt_diagnostics:
        Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
    pub theme_diagnostics:
        Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
    pub extension_load_errors: Vec<crate::modes::agent_connection::snapshot::ExtensionLoadError>,
}

impl AgentDaemon {
    /// `createAgentConnectionCommands(state.runtime.session)`.
    fn connection_commands(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> Vec<Value> {
        let view = self.session_of(state).connection_view();
        create_agent_connection_commands(
            &view.registered_commands,
            &view.prompt_templates,
            &view.skills,
        )
        .into_iter()
        .map(|command| serde_json::to_value(command).unwrap_or(Value::Null))
        .collect()
    }

    /// `createAgentConnectionResourceSnapshot(state.runtime.session)`.
    fn connection_resource_snapshot(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> Value {
        let view = self.session_of(state).connection_view();
        serde_json::to_value(create_agent_connection_resource_snapshot(
            &view.session_id,
            &view.cwd,
            &view.agents_files,
            &view.resource_skills,
            view.skill_diagnostics.clone(),
            &view.resource_prompts,
            view.prompt_diagnostics.clone(),
            &view.resource_extensions,
            &view.extension_load_errors,
            &view.resource_themes,
            view.theme_diagnostics.clone(),
        ))
        .unwrap_or(Value::Null)
    }

    /// `handleWorkerCommand(client, command)`.
    async fn handle_worker_command(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
    ) -> Result<(), String> {
        let body = &command.body;
        match command.type_.as_str() {
            "worker_auth" => Err("Worker is already authenticated".to_string()),
            "worker_subscribe" => {
                let state = self.get_bound_session_state(body.get("activeSessionId").and_then(Value::as_str).unwrap_or(""))?;
                let active_session_id = state.lock().expect("active session poisoned").active_session_id.clone();
                let requested = body.get("capabilities").and_then(Value::as_array).map(|values| values.iter().filter_map(Value::as_str).map(str::to_string).collect::<HashSet<_>>());
                set_daemon_client_session_capabilities(client, &active_session_id, normalize_client_capabilities(requested.as_ref(), body.get("supportsExtensionUi").and_then(Value::as_bool)));
                {
                    let mut state = state.lock().expect("active session poisoned");
                    if !state.clients.iter().any(|peer| Arc::ptr_eq(peer, &client.state)) { state.clients.push(client.state.clone()); }
                }
                client.state.lock().expect("daemon client poisoned").attached_active_session_ids.insert(active_session_id);
                let summary = self.summary_for_state(&state);
                self.write(client, &DaemonOutbound::Raw(serde_json::json!({"type":"response", "id":command.id,"command":"attach","success":true,"data":summary})));
                Ok(())
            }
            "worker_unsubscribe" => {
                let state = self.get_session_state(body.get("activeSessionId").and_then(Value::as_str).unwrap_or(""))?;
                self.detach_client_from_session(client, &state);
                self.write(client, &DaemonOutbound::Raw(serde_json::json!({"type":"response","id":command.id,"command":"detach","success":true})));
                Ok(())
            }
            "worker_archive_and_shutdown" => {
                for state in self.state_refs() { self.close_session(state, "killed", false, false, None, None).await; }
                self.fence_peer_transports(None);
                self.write_worker_success(client, command, None);
                let daemon = self.clone();
                tokio::spawn(async move { daemon.shutdown(0, None).await; });
                Ok(())
            }
            "worker_passivate_idle_children" => {
                let idle = match body.get("idleEvictionMinutes") { Some(Value::String(value)) if value == "off" => IdleEvictionMinutes::Off, Some(value) if value.as_f64().is_some() => IdleEvictionMinutes::Minutes(value.as_f64().unwrap()), _ => return Err("idleEvictionMinutes is required".to_string()) };
                let count = self.passivate_idle_children(idle, body.get("now").and_then(Value::as_f64).unwrap_or_else(now_millis), body.get("limit").and_then(Value::as_u64).unwrap_or(u64::MAX).min(usize::MAX as u64) as usize).await;
                self.write_worker_success(client, command, Some(serde_json::json!({"count": count})));
                Ok(())
            }
            "worker_deliver_message" => {
                let sender: AgentSessionMessageSender = serde_json::from_value(body.get("sender").cloned().ok_or("sender is required")?).map_err(|error| error.to_string())?;
                let receipt = self.send_agent_session_message(SendAgentMessageInput { target_selector: body.get("targetActiveSessionId").and_then(Value::as_str).ok_or("targetActiveSessionId is required")?.to_string(), message: body.get("message").and_then(Value::as_str).ok_or("message is required")?.to_string(), from_state: None, client_id: None, sender_key: Some(sender.active_session_id.clone().unwrap_or_else(|| format!("client:{}", sender.client_id.as_deref().unwrap_or("")))), sender: Some(sender), origin: "agent".to_string() }).await?;
                self.write_worker_success(client, command, Some(serde_json::to_value(receipt).map_err(|error| error.to_string())?));
                Ok(())
            }
            "worker_prepare_update" => {
                let manifest = self.prepare_update_restart().await?;
                self.write_worker_success(client, command, Some(manifest));
                Ok(())
            }
            "worker_commit_update" => {
                let transaction_id = body
                    .get("transactionId")
                    .and_then(Value::as_f64)
                    .map(|value| value as u64);
                let manifest = self.commit_prepared_update_restart(transaction_id).await?;
                self.write_worker_success(client, command, Some(manifest));
                Ok(())
            }
            "worker_cancel_update" => {
                let transaction_id = body
                    .get("transactionId")
                    .and_then(Value::as_f64)
                    .map(|value| value as u64);
                self.cancel_prepared_update_restart(transaction_id);
                self.write_worker_success(client, command, None);
                Ok(())
            }
            "worker_register_peer_transport" | "worker_peer_grant" => {
                let grant = body
                    .get("grant")
                    .cloned()
                    .and_then(|value| serde_json::from_value::<DaemonWorkerPeerGrant>(value).ok());
                let Some(grant) = grant else {
                    return Err("peer grant payload is invalid".to_string());
                };
                let expires_at = chrono::DateTime::parse_from_rfc3339(&grant.expires_at)
                    .map(|value| value.timestamp_millis() as f64)
                    .unwrap_or(0.0);
                let now = now_millis();
                let claim = self.supervisor_claims.lock().expect("supervisor claims poisoned").get(&(Arc::as_ptr(client) as usize)).cloned();
                if self.peer_admissions_fenced.load(Ordering::SeqCst)
                    || claim.as_ref().map(|claim| claim.claim.supervisor_generation.as_str()) != Some(grant.issuer_generation.as_str())
                    || grant.purpose != "session_client" || grant.grant_id.is_empty() || grant.token.is_empty()
                    || self.options.worker.as_ref().and_then(|worker| worker.worker_instance_id.as_deref()) != Some(grant.worker_instance_id.as_str())
                    || !self.sessions.lock().expect("sessions poisoned").contains_key(&grant.active_session_id)
                    || expires_at <= now || expires_at - now > PEER_GRANT_TTL_LIMIT_MS as f64 {
                    return Err("Peer transport grant is invalid".to_string());
                }
                let mut grants = self.peer_grants.lock().expect("peer grants poisoned");
                grants.retain(|_, grant| chrono::DateTime::parse_from_rfc3339(&grant.expires_at).map(|time| time.timestamp_millis() as f64 >= now).unwrap_or(false));
                if grants.len() >= PEER_GRANT_LIMIT {
                    return Err("peer grant table is full".to_string());
                }
                grants.insert(grant.grant_id.clone(), grant);
                drop(grants);
                self.write_worker_success(client, command, None);
                Ok(())
            }
            "worker_peer_fence" => {
                self.peer_admissions_fenced.store(true, Ordering::SeqCst);
                self.fence_peer_transports(None);
                self.write_worker_success(client, command, None);
                Ok(())
            }
            "worker_recovery_state" => {
                let active_session_id = body
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(state) = self
                    .sessions
                    .lock()
                    .expect("sessions poisoned")
                    .get(active_session_id)
                    .map(|entry| Arc::clone(&entry.state))
                {
                    self.record_worker_recovery_state(
                        &state,
                        body.get("operation")
                            .and_then(Value::as_str)
                            .unwrap_or("ready"),
                        body.get("busy").and_then(Value::as_bool),
                    );
                }
                self.write_worker_success(client, command, None);
                Ok(())
            }
            other => {
                let error = format!("Unknown worker command: {other}");
                self.write(
                    client,
                    &DaemonOutbound::Raw(
                        serde_json::to_value(DaemonResponse::failure(
                            command.id.as_deref(),
                            other,
                            &error,
                            None,
                        ))
                        .unwrap_or(Value::Null),
                    ),
                );
                Ok(())
            }
        }
    }

    /// `shutdown(exitCode)`.
    async fn shutdown(self: &Arc<Self>, _exit_code: i32, reason: Option<String>) {
        if self.shutting_down.swap(true, Ordering::SeqCst) {
            return;
        }
        self.server_stopped.cancel();
        let closing_reason = reason.unwrap_or_else(|| self.get_shutdown_closing_reason());
        self.clear_supervisor_availability_check();
        if let Some(handle) = self
            .roster_heartbeat_timer
            .lock()
            .expect("roster heartbeat poisoned")
            .take()
        {
            handle.abort();
        }
        if let Some(scheduler) = self
            .cron_scheduler
            .lock()
            .expect("cron scheduler poisoned")
            .as_ref()
        {
            scheduler.stop();
        }
        self.fence_peer_transports(Some(closing_reason.clone()));
        {
            let mut reporter = self
                .roster_reporter
                .lock()
                .expect("roster reporter poisoned");
            reporter.snapshot_pending = false;
        }
        // `shutdown(exitCode)` (`daemon-mode.ts:7722-7725`): every client - not only a
        // direct peer transport - gets `abortClientSnapshotStreaming(client)` and then
        // `write(client, { type: "daemon_closing", reason: closingReason })` before any
        // session closes. Closing the clients without that frame leaves the peer to
        // report a bare transport loss instead of a `daemon_closing` reason.
        let closing_clients = self.client_handles();
        for client in &closing_clients {
            abort_client_snapshot_streaming(client, None);
            self.write(
                client,
                &DaemonOutbound::DaemonClosing {
                    reason: closing_reason.clone(),
                },
            );
        }
        let states = self.session_states();
        for entry in states {
            self.close_session(
                Arc::clone(&entry.state),
                &closing_reason,
                true,
                false,
                None,
                None,
            )
            .await;
        }
        for handler in self
            .signal_cleanup_handlers
            .lock()
            .expect("signal cleanup handlers poisoned")
            .drain(..)
        {
            handler();
        }
        let clients = self.client_handles();
        for client in &clients {
            client.writer.end();
        }
        // Keep the runtime alive while final closing frames reach the sockets.
        let _ = tokio::time::timeout(Duration::from_secs(1), async {
            futures::future::join_all(clients.iter().map(|client| client.writer.drained.cancelled())).await;
        }).await;
        self.summarizer.stop();
        self.cleanup_socket_path();
        let _ = kill_tracked_detached_children();
        self.shutdown_complete.cancel();
    }

    /// `getShutdownClosingReason()`.
    fn get_shutdown_closing_reason(&self) -> DaemonClosingReason {
        if self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .as_ref()
            .map(|transaction| transaction.phase == "publishing")
            .unwrap_or(false)
        {
            "update".to_string()
        } else {
            "shutdown".to_string()
        }
    }

    /// `registerSignalHandlers()`.
    fn register_signal_handlers(self: &Arc<Self>) {
        native_server::register_signal_handlers(self);
    }

    /// `detachClient(client)`.
    fn detach_client(self: &Arc<Self>, client: &Arc<DaemonClientHandle>) {
        let states = self.session_states();
        for entry in states {
            let attached = entry
                .state
                .lock()
                .expect("active session poisoned")
                .clients
                .iter()
                .any(|candidate| Arc::ptr_eq(candidate, &client.state));
            if attached {
                let mut state_guard = entry.state.lock().expect("active session poisoned");
                detach_client_from_active_session(client, &mut state_guard);
            }
        }
        self.clear_client_catchup_retry(client);
    }

    /// `detachClientFromSession(client, state)`.
    fn detach_client_from_session(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) {
        let mut state_guard = state.lock().expect("active session poisoned");
        detach_client_from_active_session(client, &mut state_guard);
    }

    /// `abortWaitingPromptAdmissionsForSession(activeSessionId)`.
    fn abort_waiting_prompt_admissions_for_session(&self, active_session_id: &str) {
        let mut admissions = self
            .prompt_admissions
            .lock()
            .expect("prompt admissions poisoned");
        for admission in admissions.values_mut() {
            if admission.active_session_id == active_session_id && admission.status == "waiting" {
                admission.status = "cancelled".to_string();
                if let Some(controller) = &admission.controller {
                    controller.cancel();
                }
            }
        }
    }

    /// `releaseSessionInputPausesForPauseOwner(...)` and the `detach` command's
    /// pause sweep.
    fn release_session_input_pauses_for(
        &self,
        client: &Arc<DaemonClientHandle>,
        active_session_id: Option<&str>,
    ) {
        let mut pauses = self
            .session_input_pauses
            .lock()
            .expect("session input pauses poisoned");
        let keys: Vec<String> = pauses
            .iter()
            .filter(|(_, entry)| {
                Arc::ptr_eq(&entry.owner, client)
                    && active_session_id
                        .map(|active_session_id| entry.active_session_id == active_session_id)
                        .unwrap_or(true)
            })
            .map(|(pause_id, _)| pause_id.clone())
            .collect();
        for key in keys {
            if let Some(entry) = pauses.remove(&key) {
                (entry.pause)();
            }
        }
    }

    /// `hasActiveSideQuestionFor(client, activeSessionId)`.
    fn has_active_side_question_for(
        &self,
        client: &Arc<DaemonClientHandle>,
        active_session_id: &str,
    ) -> bool {
        self.side_question_runs
            .lock()
            .expect("side question runs poisoned")
            .values()
            .any(|entry| {
                Arc::ptr_eq(&entry.client, client) && entry.active_session_id == active_session_id
            })
    }

    /// `abortSideQuestionsFor(client, activeSessionId)`.
    fn abort_side_questions_for(&self, client: &Arc<DaemonClientHandle>, active_session_id: &str) {
        let mut runs = self
            .side_question_runs
            .lock()
            .expect("side question runs poisoned");
        let keys: Vec<String> = runs
            .iter()
            .filter(|(_, entry)| {
                Arc::ptr_eq(&entry.client, client) && entry.active_session_id == active_session_id
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            if let Some(entry) = runs.remove(&key) {
                (entry.run)();
            }
        }
    }
}

impl AgentDaemon {
    /// `prepareUpdateRestart()`.
    async fn prepare_update_restart(self: &Arc<Self>) -> Result<Value, String> {
        let transaction_id = self.next_id();
        {
            let mut transaction = self.update_restart.lock().expect("update restart poisoned");
            if transaction.is_some() {
                return Err("An update restart is already being prepared".to_string());
            }
            *transaction = Some(UpdateRestartTransaction {
                id: transaction_id,
                owner: None,
                abort: tokio_util::sync::CancellationToken::new(),
                deadline_expired: Arc::new(AtomicBool::new(false)),
                phase: "preparing".to_string(),
                manifest: None,
                deferred_client_env: Vec::new(),
            });
        }
        let abort = self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .as_ref()
            .map(|transaction| transaction.abort.clone())
            .expect("transaction just installed");
        let deadline_token = self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .as_ref()
            .map(|transaction| Arc::clone(&transaction.deadline_expired))
            .expect("transaction just installed");
        let daemon = Arc::clone(self);
        tokio::spawn(async move {
            tokio::select! {
                _ = delay(UPDATE_RESTART_PREPARE_TIMEOUT_MS) => {
                    deadline_token.store(true, Ordering::SeqCst);
                    let _ = daemon.prepare_update_restart_checkpoint(transaction_id).await;
                }
                _ = abort.cancelled() => {}
            }
        });
        self.prepare_update_restart_checkpoint(transaction_id).await
    }

    /// `runUpdateRestartPreparation(transactionId)` /
    /// `prepareUpdateRestartCheckpoint(transactionId)`.
    async fn prepare_update_restart_checkpoint(
        self: &Arc<Self>,
        transaction_id: u64,
    ) -> Result<Value, String> {
        self.assert_update_restart_not_cancelled(transaction_id)?;
        let abort = self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .as_ref()
            .filter(|transaction| transaction.id == transaction_id)
            .map(|transaction| transaction.abort.clone())
            .ok_or_else(|| "Update restart preparation was cancelled".to_string())?;
        self.mutation_drain
            .wait_for_drain(0, &abort, "Daemon is preparing an update restart")
            .await?;
        self.assert_update_restart_not_cancelled(transaction_id)?;
        let manifest = self
            .commit_prepared_update_restart(Some(transaction_id))
            .await?;
        Ok(manifest)
    }

    /// `assertUpdateRestartNotCancelled(transaction)`.
    fn assert_update_restart_not_cancelled(&self, transaction_id: u64) -> Result<(), String> {
        let transaction = self.update_restart.lock().expect("update restart poisoned");
        match transaction.as_ref() {
            Some(transaction) if transaction.id == transaction_id => {}
            _ => return Err("Update restart preparation was cancelled".to_string()),
        }
        Ok(())
    }

    /// `commitPreparedUpdateRestart(transactionId)`.
    async fn commit_prepared_update_restart(
        self: &Arc<Self>,
        transaction_id: Option<u64>,
    ) -> Result<Value, String> {
        {
            let mut transaction = self.update_restart.lock().expect("update restart poisoned");
            let Some(current) = transaction.as_mut() else {
                return Err("Update restart preparation was cancelled".to_string());
            };
            if let Some(transaction_id) = transaction_id {
                if current.id != transaction_id {
                    return Err("Update restart preparation was cancelled".to_string());
                }
            }
            current.phase = "fencing".to_string();
        }
        let states = self.session_states();
        let mut sessions: Vec<Value> = Vec::new();
        let mut discarded_active_session_ids: Vec<String> = Vec::new();
        for entry in states {
            let state = Arc::clone(&entry.state);
            if let Some(session) = self.create_update_restart_session(&state) {
                sessions.push(session);
            } else {
                discarded_active_session_ids.push(
                    state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                );
            }
        }
        let manifest = serde_json::json!({
            "formatVersion": DAEMON_UPDATE_RESTART_FORMAT_VERSION,
            "createdAt": now_iso(),
            "sessions": sessions,
            "discardedActiveSessionIds": discarded_active_session_ids,
        });
        self.write_update_restart_manifest(&manifest);
        let mut transaction = self.update_restart.lock().expect("update restart poisoned");
        if let Some(current) = transaction.as_mut() {
            current.phase = "prepared".to_string();
            current.manifest = Some(manifest.clone());
        }
        Ok(manifest)
    }

    /// `cancelPreparedUpdateRestart(transactionId?)`.
    fn cancel_prepared_update_restart(&self, transaction_id: Option<u64>) {
        let mut transaction = self.update_restart.lock().expect("update restart poisoned");
        let Some(current) = transaction.as_mut() else {
            return;
        };
        if let Some(transaction_id) = transaction_id {
            if current.id != transaction_id {
                return;
            }
        }
        current.abort.cancel();
        let deferred = std::mem::take(&mut current.deferred_client_env);
        *transaction = None;
        drop(transaction);
        for entry in deferred {
            self.adopt_client_env(&entry.state, Some(entry.env));
        }
    }

    /// `createUpdateRestartSession(state)`.
    fn create_update_restart_session(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Option<Value> {
        let entry = self.session_entry_for_state(state);
        let session = &entry.session;
        let session_file = session.session_file()?;
        let session_id = session.session_id();
        if session_id.is_empty() {
            return None;
        }
        let cwd = session
            .session_manager()
            .lock()
            .expect("session manager poisoned")
            .get_cwd();
        let config =
            serde_json::to_value(&self.options.default_session_config).unwrap_or(Value::Null);
        let mut session_value = Map::new();
        session_value.insert(
            "activeSessionId".to_string(),
            Value::String(
                state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
            ),
        );
        session_value.insert("sessionId".to_string(), Value::String(session_id));
        session_value.insert(
            "sessionFile".to_string(),
            Value::String(session_file.clone()),
        );
        session_value.insert("cwd".to_string(), Value::String(cwd));
        session_value.insert("config".to_string(), config);
        // REPAIR CURSOR: `active_session_state::AgentSessionRuntimeMetadata`
        // (active_session_state.rs:99) is a lossy 8-field duplicate of the canonical
        // `core::agent_session_runtime::AgentSessionRuntimeMetadata` (:84) and carries no
        // serde derives. Fix: delete the duplicate, import the canonical type at
        // daemon_mode.rs:96, and serialize `entry.runtime_metadata` directly. The
        // projection below writes the same camelCase field names.
        session_value.insert(
            "runtimeMetadata".to_string(),
            serde_json::json!({
                "kind": entry.runtime_metadata.kind,
                "prompt": entry.runtime_metadata.prompt,
                "parentActiveSessionId": entry.runtime_metadata.parent_active_session_id,
                "parentSessionId": entry.runtime_metadata.parent_session_id,
                "parentSessionFile": entry.runtime_metadata.parent_session_file,
                "rlmChildId": entry.runtime_metadata.rlm_child_id,
                "rlmParentNodeId": entry.runtime_metadata.rlm_parent_node_id,
                "spawnCode": entry.runtime_metadata.spawn_code,
            }),
        );
        if let Some(client_env) = &state.lock().expect("active session poisoned").client_env {
            session_value.insert(
                "clientEnv".to_string(),
                serde_json::to_value(client_env).unwrap_or(Value::Null),
            );
        }
        session_value.insert(
            "queue".to_string(),
            serde_json::json!({ "actions": Value::Null, "nextTurn": Vec::<Value>::new() }),
        );
        session_value.insert(
            "shouldResume".to_string(),
            Value::Bool(session.is_session_active()),
        );
        session_value.insert(
            "wasStreaming".to_string(),
            Value::Bool(session.is_streaming()),
        );
        session_value.insert(
            "wasCompacting".to_string(),
            Value::Bool(session.is_compacting()),
        );
        session_value.insert(
            "wasBashRunning".to_string(),
            Value::Bool(session.is_bash_running()),
        );
        session_value.insert(
            "hadRunningRlmChildren".to_string(),
            Value::Bool(session.has_running_rlm_children()),
        );
        session_value.insert(
            "wasRetrying".to_string(),
            Value::Bool(session.is_retrying()),
        );
        session_value.insert("hadAcceptedPromptInFlight".to_string(), Value::Bool(false));
        Some(Value::Object(session_value))
    }

    /// `writeUpdateRestartManifest(manifest)`.
    fn write_update_restart_manifest(&self, manifest: &Value) {
        let path = get_daemon_update_restart_manifest_path(&self.socket_path);
        if let Some(parent) = Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let payload = format!("{}\n", serde_json::to_string(manifest).unwrap_or_default());
        if let Err(error) = std::fs::write(&path, payload) {
            self.log(&format!("failed to write update restart manifest: {error}"));
        }
    }
}

impl AgentDaemon {
    /// `closeSession(state, reason, waitForAbort = true, cascadeChildren = true, descendantCollector?, disposal?)`.
    pub(crate) async fn close_session(
        self: &Arc<Self>,
        state: Arc<StdMutex<ActiveSessionState>>,
        reason: &str,
        wait_for_abort: bool,
        cascade_children: bool,
        descendant_collector: Option<Arc<StdMutex<Vec<Arc<StdMutex<ActiveSessionState>>>>>>,
        disposal_kernel_snapshot: Option<bool>,
    ) -> Result<(), String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.abort_waiting_prompt_admissions_for_session(&active_session_id);
        let clients = state
            .lock()
            .expect("active session poisoned")
            .clients
            .clone();
        for client_state in &clients {
            let Some(handle) = self
                .client_handles()
                .into_iter()
                .find(|handle| Arc::ptr_eq(&handle.state, client_state))
            else {
                continue;
            };
            self.abort_side_questions_for(&handle, &active_session_id);
        }
        let existing_close = self
            .closing_sessions
            .lock()
            .expect("closing sessions poisoned")
            .get(&active_session_id)
            .map(|close| close.reason.clone());
        if let Some(existing_reason) = existing_close {
            let requested_reason = if self.is_stronger_close_reason(reason, &existing_reason) {
                reason.to_string()
            } else {
                existing_reason.clone()
            };
            if self.is_stronger_close_reason(&requested_reason, &existing_reason) {
                self.apply_reason_upgrade(&state, &[], &existing_reason, &requested_reason);
                if let Some(close) = self
                    .closing_sessions
                    .lock()
                    .expect("closing sessions poisoned")
                    .get_mut(&active_session_id)
                {
                    close.reason = requested_reason;
                }
            }
            if let Some(collector) = &descendant_collector {
                collector
                    .lock()
                    .expect("descendant collector poisoned")
                    .push(Arc::clone(&state));
            }
            return Ok(());
        }
        let descendants: Arc<StdMutex<Vec<Arc<StdMutex<ActiveSessionState>>>>> =
            Arc::new(StdMutex::new(Vec::new()));
        self.closing_sessions
            .lock()
            .expect("closing sessions poisoned")
            .insert(
                active_session_id.clone(),
                ClosingSession {
                    promise: Box::pin(async {}),
                    reason: reason.to_string(),
                    descendants: Vec::new(),
                    reason_upgrade: None,
                },
            );
        let result = self
            .close_session_once(
                Arc::clone(&state),
                reason,
                wait_for_abort,
                cascade_children,
                Arc::clone(&descendants),
                disposal_kernel_snapshot,
            )
            .await;
        if let Some(collector) = &descendant_collector {
            let mut collector = collector.lock().expect("descendant collector poisoned");
            collector.push(Arc::clone(&state));
            for descendant in descendants.lock().expect("descendants poisoned").iter() {
                collector.push(Arc::clone(descendant));
            }
        }
        {
            let mut closing = self
                .closing_sessions
                .lock()
                .expect("closing sessions poisoned");
            if let Some(close) = closing.get(&active_session_id) {
                if close.reason == reason {
                    closing.remove(&active_session_id);
                }
            }
        }
        result
    }

    /// `closeReasonStrength(reason)`.
    fn close_reason_strength(&self, reason: &str) -> u32 {
        match reason {
            "killed" => 2,
            "completed" | "replaced" => 1,
            _ => 0,
        }
    }

    /// `isStrongerCloseReason(candidate, current)`.
    fn is_stronger_close_reason(&self, candidate: &str, current: &str) -> bool {
        self.close_reason_strength(candidate) > self.close_reason_strength(current)
    }

    /// `applyReasonUpgrade(state, descendants, from, to)`.
    fn apply_reason_upgrade(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        descendants: &[Arc<StdMutex<ActiveSessionState>>],
        from: &str,
        to: &str,
    ) {
        for target in std::iter::once(state).chain(descendants.iter()) {
            if to == "killed" {
                self.cancel_scheduled_jobs_for_session(target);
            }
            if self.close_keeps_resume_entry(from) {
                self.archive_session(target);
            }
        }
    }

    /// `closeKeepsResumeEntry(reason)`.
    fn close_keeps_resume_entry(&self, reason: &str) -> bool {
        reason == "shutdown" || reason == "update"
    }

    /// `archiveSession(state)`.
    fn archive_session(&self, state: &Arc<StdMutex<ActiveSessionState>>) {
        self.session_of(state)
            .session_manager()
            .lock()
            .expect("session manager poisoned")
            .append_session_state(&crate::core::session_manager::SessionState {
                status: crate::core::session_manager::SessionStateStatus::Archived,
            });
    }

    /// `abortBashForClose(state)`.
    async fn abort_bash_for_close(&self, state: &Arc<StdMutex<ActiveSessionState>>) {
        let session = self.session_of(state);
        if !session.is_bash_running() {
            return;
        }
        session.abort_bash();
        // `Promise.race([state.inFlightBash ?? Promise.resolve(), delay(…)]).
        let in_flight = self.in_flight_bash_slot(state);
        tokio::select! {
            _ = delay(UPDATE_RESTART_ABORT_BASH_TIMEOUT_MS) => {}
            _ = in_flight.notified() => {}
        }
    }

    /// `closeSessionOnce(state, reason, waitForAbort, cascadeChildren, descendants, disposal?)`.
    async fn close_session_once(
        self: &Arc<Self>,
        state: Arc<StdMutex<ActiveSessionState>>,
        reason: &str,
        wait_for_abort: bool,
        cascade_children: bool,
        descendants: Arc<StdMutex<Vec<Arc<StdMutex<ActiveSessionState>>>>>,
        disposal_kernel_snapshot: Option<bool>,
    ) -> Result<(), String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if !self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .contains_key(&active_session_id)
        {
            return Ok(());
        }
        if reason == "killed" {
            self.cancel_scheduled_jobs_for_session(&state);
        } else if reason != "shutdown" && reason != "update" {
            self.cancel_subagent_rlm_heartbeats(&state);
        }
        // Abort in-flight status work before any await/dispose so it can't write
        // agent_status to a session being torn down.
        self.summarizer.forget(&active_session_id);
        let cascade_error = if cascade_children {
            self.close_child_sessions(
                state.clone(),
                reason,
                wait_for_abort,
                Arc::clone(&descendants),
                disposal_kernel_snapshot,
            )
            .await
            .err()
        } else {
            None
        };
        // Empty draft (no messages, config, or jobs): discard rather than persist an
        // empty session file.
        let keeps_resume_entry = self.close_keeps_resume_entry(reason);
        let is_empty_draft_session = !keeps_resume_entry && self.is_empty_draft_content(&state);
        let mut persist_error: Option<String> = None;
        // Clean shutdown leaves the session un-archived so it stays in the resume list.
        if !keeps_resume_entry && !is_empty_draft_session {
            if let Err(error) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.archive_session(&state);
            })) {
                persist_error = Some(format!("{error:?}"));
            }
        }
        {
            let mut state_guard = state.lock().expect("active session poisoned");
            cancel_pending_extension_ui_requests(&mut state_guard);
        }
        if reason == "killed" || reason == "shutdown" || reason == "replaced" || reason == "update"
        {
            self.abort_bash_for_close(&state).await;
        }
        if reason == "update" {
            self.session_of(&state).abort_for_update_restart();
        }
        if reason == "killed" {
            let session = self.session_of(&state);
            session.request_abort();
            if wait_for_abort {
                session.wait_for_idle().await;
            }
        } else if reason == "shutdown" || reason == "replaced" {
            let session = self.session_of(&state);
            session.request_abort();
            // The abort was delivered; a hung agent must not park a daemon
            // shutdown or worker replacement forever (audit A6). The wait is
            // bounded and a timeout is reported truthfully, with the teardown
            // continuing at the dispose below.
            let bounded = tokio::time::timeout(CLOSE_SETTLE_WAIT_TIMEOUT, session.wait_for_idle()).await;
            if bounded.is_err() {
                self.log(&format!(
                    "Session {} still busy after abort; teardown continues after the bounded {}s wait",
                    active_session_id,
                    CLOSE_SETTLE_WAIT_TIMEOUT.as_secs()
                ));
            }
        }
        self.record_worker_recovery_state(&state, &format!("closed:{reason}"), Some(false));
        {
            let mut state_guard = state.lock().expect("active session poisoned");
            if let Some(unsubscribe) = state_guard.unsubscribe.take() {
                unsubscribe();
            }
        }
        let dispose_error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {}))
            .err()
            .map(|error| format!("{error:?}"));
        self.session_of(&state).dispose().await;
        let clients = state
            .lock()
            .expect("active session poisoned")
            .clients
            .clone();
        for client_state in &clients {
            let Some(handle) = self
                .client_handles()
                .into_iter()
                .find(|handle| Arc::ptr_eq(&handle.state, client_state))
            else {
                continue;
            };
            abort_client_snapshot_streaming(&handle, Some(&active_session_id));
        }
        self.broadcast_to_session(
            &self.session_entry_for_state(&state),
            DaemonOutbound::SessionClosed {
                active_session_id: active_session_id.clone(),
                reason: reason.to_string(),
            },
        );
        for client_state in &clients {
            client_state
                .lock()
                .expect("daemon client poisoned")
                .attached_active_session_ids
                .remove(&active_session_id);
            let Some(handle) = self
                .client_handles()
                .into_iter()
                .find(|handle| Arc::ptr_eq(&handle.state, client_state))
            else {
                continue;
            };
            remove_daemon_client_session_capabilities(&handle, &active_session_id);
        }
        state
            .lock()
            .expect("active session poisoned")
            .clients
            .clear();
        self.acp_mcp_owners
            .lock()
            .expect("acp mcp owners poisoned")
            .remove(&active_session_id);
        let metadata_kind = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.clone());
        let session_id = self.session_of(&state).session_id();
        let session_file = self.session_of(&state).session_file();
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .remove(&active_session_id);
        // Archived top-level sessions leave the worker's list; subagent rows mirror
        // the registry and stay.
        if !keeps_resume_entry && metadata_kind.as_deref() != Some("subagent") && self.is_worker() {
            let agent_id = self.roster_agent_id_for_state(&state);
            self.roster_reporter
                .lock()
                .expect("roster reporter poisoned")
                .removed_agent_ids
                .insert(agent_id, Some(session_id));
        }
        self.schedule_roster_flush();
        if is_empty_draft_session {
            if let Some(session_file) = session_file {
                let _ = crate::core::session_file_actions::delete_session_file(
                    &session_file,
                    &mut DeleteSessionFileOptions::default(),
                );
            }
        }
        if let Some(error) = dispose_error {
            return Err(error);
        }
        if let Some(error) = persist_error {
            if !keeps_resume_entry && reason != "completed" {
                return Err(error);
            }
        }
        if let Some(error) = cascade_error {
            if !keeps_resume_entry && reason != "completed" {
                return Err(error);
            }
        }
        Ok(())
    }

    /// `closeChildSessions(parentState, reason, waitForAbort, descendants, disposal)`.
    async fn close_child_sessions(
        self: &Arc<Self>,
        parent_state: Arc<StdMutex<ActiveSessionState>>,
        reason: &str,
        wait_for_abort: bool,
        descendants: Arc<StdMutex<Vec<Arc<StdMutex<ActiveSessionState>>>>>,
        disposal_kernel_snapshot: Option<bool>,
    ) -> Result<(), String> {
        let mut cascade_error: Option<String> = None;
        let children = get_child_active_session_states(&self.states_by_id(), &parent_state);
        for child_state in children {
            descendants
                .lock()
                .expect("descendants poisoned")
                .push(Arc::clone(&child_state));
            // Box only this edge of the close cascade (TS: closeChildSessions awaits
            // closeSession, which reaches back here through closeSessionOnce).
            let closing = Box::pin(self.close_session(
                child_state,
                reason,
                wait_for_abort,
                true,
                Some(Arc::clone(&descendants)),
                disposal_kernel_snapshot,
            ));
            if let Err(error) = closing.await {
                if cascade_error.is_none() {
                    cascade_error = Some(error);
                }
            }
        }
        match cascade_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `isDiscardableDraft(state)`.
    fn is_discardable_draft(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> bool {
        if state
            .lock()
            .expect("active session poisoned")
            .clients
            .iter()
            .any(|client| {
                !client
                    .lock()
                    .expect("daemon client poisoned")
                    .attached_active_session_ids
                    .is_empty()
            })
        {
            return false;
        }
        self.has_scheduled_jobs_for_session(
            &state
                .lock()
                .expect("active session poisoned")
                .active_session_id,
        ) == false
            && self.is_empty_draft_content(state)
    }

    /// `isEmptyDraftContent(state)`.
    fn is_empty_draft_content(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> bool {
        let session = self.session_of(state);
        if !session.messages().is_empty() {
            return false;
        }
        if !session.session_id().is_empty() && session.session_file().is_some() {
            return false;
        }
        if self.has_scheduled_jobs_for_session(
            &state
                .lock()
                .expect("active session poisoned")
                .active_session_id,
        ) {
            return false;
        }
        true
    }

    /// `hasScheduledJobsForSession(activeSessionId)`.
    fn has_scheduled_jobs_for_session(&self, active_session_id: &str) -> bool {
        self.cron_store
            .list()
            .iter()
            .any(|job| job.active_session_id == active_session_id && job.status == "active")
    }
}

/// The sentinel `unrunnableAtAdmission` error in `runCronJob`.
const CRON_JOB_UNRUNNABLE_AT_ADMISSION: &str = "Cron job became unrunnable before admission";

impl AgentDaemon {
    /// `runCronJob(job)`.
    ///
    /// The TS rejects with the run error instead of swallowing it: the queued-prompt
    /// `await session.followUp(...)` sits inside the same try/catch as the prompt calls
    /// (daemon-mode.ts:2064 then 2085-2105), and that catch rethrows everything that is
    /// not the `unrunnableAtAdmission` sentinel (daemon-mode.ts:2100-2105). The
    /// scheduler turns that rejection into `onError` + `lastError`
    /// (cron-jobs.ts:1010-1020), so the port returns the same rejection as `Err`.
    async fn run_cron_job(self: &Arc<Self>, job: AgentCronJob) -> Result<Option<String>, String> {
        let require_persisted_job = self
            .cron_store
            .list()
            .iter()
            .any(|candidate| candidate.id == job.id);
        let due_job = if require_persisted_job {
            self.get_runnable_cron_job(&job.id)
        } else {
            Some(job.clone())
        };
        let Some(due_job) = due_job else {
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        };
        let state = self
            .get_or_create_cron_job_session(&due_job, require_persisted_job)
            .await;
        let runnable_job = if require_persisted_job {
            self.get_runnable_cron_job(&job.id)
        } else {
            Some(due_job)
        };
        let (Some(state), Some(runnable_job)) = (state, runnable_job) else {
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        };
        if !self.is_cron_job_runnable_for_state(&runnable_job, &state, require_persisted_job) {
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        }
        let session = self.session_of(&state);
        if should_defer_heartbeat_cron_job(&runnable_job, &self.heartbeat_activity(&state)) {
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        }
        let should_queue_cron_prompt = session.is_streaming()
            || session.is_compacting()
            || session.is_retrying()
            || session.is_bash_running()
            || session.unfinished_action_count() > 0.0;
        if !is_heartbeat_cron_job(&runnable_job) && should_queue_cron_prompt {
            if !self.is_cron_job_runnable_for_state(&runnable_job, &state, require_persisted_job) {
                return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
            }
            // `await session.followUp(runnableJob.prompt, undefined, { resumeIfIdle: true })`
            // (daemon-mode.ts:2064-2066) has no catch of its own, and `runCronJob` is
            // `async`, so a rejection here leaves the method as a rejected promise. The
            // scheduler catches that as `runError`, calls `onError`, and stores it as
            // `lastError` (cron-jobs.ts:1010-1020). Dropping the error would make the
            // daemon's queued-prompt failures invisible, so it is returned here.
            session
                .follow_up(
                    &runnable_job.prompt,
                    None,
                    PromptInvocation {
                        resume_if_idle: Some(true),
                        ..PromptInvocation::default()
                    },
                )
                .await?;
            return Ok(None);
        }
        let daemon = Arc::clone(self);
        let job_id = job.id.clone();
        let runnable_job = Arc::new(runnable_job);
        let get_runnable_job = {
            let daemon = Arc::clone(&daemon);
            let runnable_job = Arc::clone(&runnable_job);
            let state = Arc::clone(&state);
            move || -> Option<AgentCronJob> {
                let current = if require_persisted_job {
                    daemon.get_runnable_cron_job(&job_id)
                } else {
                    Some((*runnable_job).clone())
                };
                current.filter(|current| {
                    daemon.is_cron_job_runnable_for_state(current, &state, require_persisted_job)
                })
            }
        };
        let Some(current) = get_runnable_job() else {
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        };
        // Re-check after the session admission fence wait: the job may have been
        // cancelled, completed, or updated meanwhile.
        let admission_daemon = Arc::clone(self);
        let admission_job_id = job.id.clone();
        let admission_state = Arc::clone(&state);
        let expected_prompt = current.prompt.clone();
        let expected_delivery_mode = current.delivery_mode.clone();
        // The TS `admissionCommitted` THROWS `unrunnableAtAdmission` (daemon-mode.ts:2079-2084)
        // so the prompt aborts and the outer catch returns "skipped" (daemon-mode.ts:2101-2103).
        // The ported seam cannot throw: `PromptInvocation::admission_committed` is
        // `Option<Arc<dyn Fn() + Send + Sync>>` and the session owner calls it as `committed()`
        // (core/agent_session.rs:7613, 7944) — changing that type means editing the session
        // slice, which this slice does not own. So the sentinel is recorded here and read back
        // after the prompt, which reproduces the same outcome ("skipped", not an error).
        let admission_rejected = Arc::new(AtomicBool::new(false));
        let admission_committed: Arc<dyn Fn() + Send + Sync> = Arc::new({
            let admission_rejected = Arc::clone(&admission_rejected);
            move || {
                let refreshed = if require_persisted_job {
                    admission_daemon.get_runnable_cron_job(&admission_job_id)
                } else {
                    Some((*runnable_job).clone())
                }
                .filter(|current| {
                    admission_daemon.is_cron_job_runnable_for_state(
                        current,
                        &admission_state,
                        require_persisted_job,
                    )
                });
                match refreshed {
                    Some(refreshed)
                        if refreshed.prompt == expected_prompt
                            && refreshed.delivery_mode == expected_delivery_mode => {}
                    _ => {
                        admission_rejected.store(true, Ordering::SeqCst);
                        admission_daemon.log(CRON_JOB_UNRUNNABLE_AT_ADMISSION);
                    }
                }
            }
        });
        let invocation = PromptInvocation {
            streaming_behavior: None,
            queue_key: None,
            source: Some("rpc".to_string()),
            admission_committed: Some(admission_committed),
            ..PromptInvocation::default()
        };
        let result = if is_heartbeat_cron_job(&current) {
            session
                .prompt_heartbeat(
                    &current,
                    PromptInvocation {
                        streaming_behavior: Some(resolve_heartbeat_streaming_behavior(
                            current.delivery_mode.as_deref(),
                        )),
                        queue_key: Some(format!("heartbeat:{}", current.id)),
                        ..invocation
                    },
                )
                .await
        } else {
            session
                .prompt_until_accepted(
                    &current.prompt,
                    PromptInvocation {
                        streaming_behavior: Some("followUp".to_string()),
                        ..invocation
                    },
                )
                .await
        };
        if admission_rejected.load(Ordering::SeqCst) {
            // `if (error === unrunnableAtAdmission) { return "skipped"; }`
            // (daemon-mode.ts:2101-2103).
            return Ok(Some(RUN_RESULT_SKIPPED.to_string()));
        }
        match result {
            Ok(()) => Ok(None),
            // `throw error;` (daemon-mode.ts:2104): every other rejection leaves
            // `runCronJob`, the scheduler catches it as `runError`, calls `onError`, and
            // stores it as `lastError` (cron-jobs.ts:1010-1020).
            Err(error) => Err(error),
        }
    }

    /// `HeartbeatCronSessionActivity` for `shouldDeferHeartbeatCronJob`.
    fn heartbeat_activity(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> HeartbeatCronSessionActivity {
        let session = self.session_of(state);
        HeartbeatCronSessionActivity {
            is_streaming: session.is_streaming(),
            is_compacting: Some(session.is_compacting()),
            is_retrying: Some(session.is_retrying()),
            is_bash_running: session.is_bash_running(),
            has_pending_session_work: session.unfinished_action_count() > 0.0,
            unfinished_action_count: session.unfinished_action_count(),
        }
    }

    /// `getRunnableCronJob(jobId)`.
    fn get_runnable_cron_job(&self, job_id: &str) -> Option<AgentCronJob> {
        self.cron_store
            .get_claimed_job(job_id)
            .or_else(|| self.cron_store.get_due_job(job_id, now_millis()))
    }

    /// `createCronJobForState(state, schedule, prompt)`.
    fn create_cron_job_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        schedule: &str,
        prompt: &str,
    ) -> Result<AgentCronJob, String> {
        let session = self.session_of(state);
        let session_file = session
            .session_file()
            .ok_or_else(|| "Cron jobs require a persisted session file".to_string())?;
        let entry = self.session_entry_for_state(state);
        let job = self.cron_store.create(&CreateAgentCronJobInput {
            active_session_id: state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                .clone(),
            session_id: session.session_id(),
            session_file,
            cwd: entry.cwd(),
            runtime_kind: entry.runtime_metadata.kind.clone(),
            schedule_text: schedule.to_string(),
            prompt: prompt.to_string(),
            ..CreateAgentCronJobInput::default()
        })?;
        self.cron_scheduler_wake();
        Ok(job)
    }

    /// `createHeartbeatForState(state, schedule, instruction, deliveryMode?)`.
    fn create_heartbeat_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        schedule: &str,
        instruction: &str,
        delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    ) -> Result<AgentCronJob, String> {
        let session = self.session_of(state);
        let session_file = session
            .session_file()
            .ok_or_else(|| "Heartbeats require a persisted session file".to_string())?;
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let previous_heartbeat = self.cron_store.get_heartbeat(&active_session_id);
        let entry = self.session_entry_for_state(state);
        let job = self.cron_store.create_heartbeat(&CreateAgentCronJobInput {
            active_session_id,
            session_id: session.session_id(),
            session_file,
            cwd: entry.cwd(),
            runtime_kind: entry.runtime_metadata.kind.clone(),
            schedule_text: normalize_heartbeat_schedule(Some(schedule)),
            prompt: instruction.to_string(),
            delivery_mode: delivery_mode.or_else(|| {
                previous_heartbeat
                    .as_ref()
                    .and_then(|job| job.delivery_mode.clone())
            }),
            ..CreateAgentCronJobInput::default()
        })?;
        if let Some(previous_heartbeat) = previous_heartbeat {
            self.remove_queued_heartbeat_follow_up(state, &previous_heartbeat);
        }
        self.cron_scheduler_wake();
        Ok(job)
    }

    /// `updateHeartbeatForState(state, action)`.
    fn update_heartbeat_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        action: &str,
    ) -> Result<Option<AgentCronJob>, String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let now = now_millis();
        let job = match action {
            "pause" => self.cron_store.pause_heartbeat(&active_session_id, now),
            "resume" => self.cron_store.resume_heartbeat(&active_session_id, now)?,
            _ => self.cron_store.clear_heartbeat(&active_session_id, now),
        };
        if let Some(job) = &job {
            if action != "resume" {
                self.remove_queued_heartbeat_follow_up(state, job);
            }
        }
        self.cron_scheduler_wake();
        Ok(job)
    }

    /// `createRlmHeartbeatForState(state, input)`.
    fn create_rlm_heartbeat_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        input: &RlmHeartbeatCreateInput,
    ) -> Result<AgentCronJob, String> {
        let session = self.session_of(state);
        let session_file = session
            .session_file()
            .ok_or_else(|| "RLM heartbeats require a persisted session file".to_string())?;
        let entry = self.session_entry_for_state(state);
        let job = self
            .cron_store
            .create_rlm_heartbeat(&CreateAgentCronJobInput {
                active_session_id: state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
                session_id: session.session_id(),
                session_file,
                cwd: entry.cwd(),
                runtime_kind: entry.runtime_metadata.kind.clone(),
                label: input.label.clone(),
                schedule_text: normalize_heartbeat_schedule(input.interval.as_deref()),
                prompt: input.instruction.clone(),
                delivery_mode: input.delivery_mode.clone(),
                ..CreateAgentCronJobInput::default()
            })?;
        self.cron_scheduler_wake();
        Ok(job)
    }

    /// `updateRlmHeartbeatForState(state, input)`.
    fn update_rlm_heartbeat_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        input: &RlmHeartbeatUpdateInput,
    ) -> Option<AgentCronJob> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let job = self.cron_store.update_rlm_heartbeat(
            &active_session_id,
            &input.id,
            &crate::core::cron_jobs::RlmHeartbeatUpdate {
                label: input.label.clone(),
                prompt: input.instruction.clone(),
                schedule_text: input
                    .interval
                    .as_deref()
                    .map(|interval| normalize_heartbeat_schedule(Some(interval))),
                status: input.status.clone(),
                delivery_mode: input.delivery_mode.clone(),
                now: None,
            },
        );
        // The TypeScript signature is `AgentCronJob | undefined`: a store failure
        // reads as "no job" (the error is logged by the store).
        let Ok(job) = job else {
            return None;
        };
        if let Some(job) = &job {
            if input.instruction.is_some()
                || input.interval.is_some()
                || input.status.as_deref() == Some("pause")
                || input.delivery_mode.is_some()
            {
                self.remove_queued_heartbeat_follow_up(state, job);
            }
            self.cron_scheduler_wake();
        }
        job
    }

    /// `deleteRlmHeartbeatForState(state, id)`.
    fn delete_rlm_heartbeat_for_state(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        id: &str,
    ) -> Option<AgentCronJob> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let job = self
            .cron_store
            .delete_rlm_heartbeat(&active_session_id, id, now_millis());
        if let Some(job) = &job {
            self.remove_queued_heartbeat_follow_up(state, job);
            self.cron_scheduler_wake();
        }
        job
    }

    /// `listHeartbeats()`.
    fn list_heartbeats(&self) -> Vec<Value> {
        self.cron_store
            .list()
            .into_iter()
            .filter(|job| {
                is_heartbeat_cron_job(job)
                    && (job.status == crate::core::cron_jobs::STATUS_ACTIVE
                        || job.status == crate::core::cron_jobs::STATUS_PAUSED)
            })
            .map(|job| {
                let state = self
                    .sessions
                    .lock()
                    .expect("sessions poisoned")
                    .get(&job.active_session_id)
                    .map(|entry| Arc::clone(&entry.state));
                let summary = state.as_ref().map(|state| self.summary_for_state(state));
                let mut object = Map::new();
                object.insert("job".to_string(), job_to_value(&job));
                if let Some(summary) = summary {
                    if let Some(session_name) = summary.session_name.clone() {
                        object.insert("sessionName".to_string(), Value::String(session_name));
                    }
                    if let Some(first_message) =
                        summary.first_message.clone().filter(|value| !value.is_empty())
                    {
                        object.insert("firstMessage".to_string(), Value::String(first_message));
                    }
                }
                Value::Object(object)
            })
            .collect()
    }

    /// `manageHeartbeat(activeSessionId, jobId, action)`.
    fn manage_heartbeat(
        self: &Arc<Self>,
        active_session_id: &str,
        job_id: &str,
        action: &str,
    ) -> Result<Option<AgentCronJob>, String> {
        let job =
            self.cron_store
                .manage_heartbeat(active_session_id, job_id, action, now_millis())?;
        let state = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(active_session_id)
            .map(|entry| Arc::clone(&entry.state));
        if let (Some(job), true, Some(state)) = (job.as_ref(), action != "resume", state.as_ref()) {
            self.remove_queued_heartbeat_follow_up(state, job);
        }
        if job.is_some() {
            self.cron_scheduler_wake();
        }
        Ok(job)
    }

    /// `rebindCronJobsToState(state)`.
    fn rebind_cron_jobs_to_state(self: &Arc<Self>, state: &Arc<StdMutex<ActiveSessionState>>) {
        let session = self.session_of(state);
        let Some(session_file) = session.session_file() else {
            return;
        };
        let entry = self.session_entry_for_state(state);
        let rebound_jobs = self
            .cron_store
            .rebind_session_jobs(&CreateAgentCronJobInput {
                active_session_id: state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
                session_id: session.session_id(),
                session_file,
                cwd: entry.cwd(),
                ..CreateAgentCronJobInput::default()
            });
        if rebound_jobs
            .iter()
            .any(|job| job.status == crate::core::cron_jobs::STATUS_ACTIVE)
        {
            self.cron_scheduler_wake();
        }
    }

    /// `cancelSubagentRlmHeartbeats(state)`.
    fn cancel_subagent_rlm_heartbeats(self: &Arc<Self>, state: &Arc<StdMutex<ActiveSessionState>>) {
        if state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.as_deref())
            != Some("subagent")
        {
            return;
        }
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let cancelled = self
            .cron_store
            .cancel_rlm_heartbeats_for_session(&active_session_id, now_millis());
        for job in &cancelled {
            self.remove_queued_heartbeat_follow_up(state, job);
        }
        if !cancelled.is_empty() {
            self.cron_scheduler_wake();
        }
    }

    /// `cancelScheduledJobsForSession(state)`.
    fn cancel_scheduled_jobs_for_session(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) {
        let session = self.session_of(state);
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let session_id = session.session_id();
        let session_file = session.session_file();
        let cancelled = self.cron_store.cancel_jobs_for_session(
            &CancelJobsForSessionInput {
                active_session_id: Some(active_session_id),
                session_id: (!session_id.is_empty()).then_some(session_id),
                session_file,
            },
            now_millis(),
        );
        for job in &cancelled {
            self.remove_queued_heartbeat_follow_up(state, job);
        }
        if !cancelled.is_empty() {
            self.cron_scheduler_wake();
        }
    }

    /// `cancelScheduledJobsForSessionFile(sessionFile)`.
    fn cancel_scheduled_jobs_for_session_file(self: &Arc<Self>, session_file: &str) {
        let cancelled = self.cron_store.cancel_jobs_for_session(
            &CancelJobsForSessionInput {
                active_session_id: None,
                session_id: None,
                session_file: Some(session_file.to_string()),
            },
            now_millis(),
        );
        if !cancelled.is_empty() {
            self.cron_scheduler_wake();
        }
    }

    /// `deleteSavedSessionFile(sessionPath, options?)`.
    pub(crate) fn delete_saved_session_file(
        &self,
        session_path: &str,
        options: Option<crate::core::session_file_actions::DeleteSessionFileOptions>,
    ) -> DeleteSessionFileResult {
        let mut options = options.unwrap_or_default();
        crate::core::session_file_actions::delete_session_file(session_path, &mut options)
    }

    /// `removeQueuedHeartbeatFollowUp(state, job)`.
    fn remove_queued_heartbeat_follow_up(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        job: &AgentCronJob,
    ) {
        if !is_heartbeat_cron_job(job) {
            return;
        }
        self.session_of(state)
            .remove_queued_follow_up(&format!("heartbeat:{}", job.id));
    }
}

impl AgentDaemon {
    /// `handlePromptCommand(client, command)`.
    async fn handle_prompt_command(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
    ) -> Result<Option<DaemonResponse>, String> {
        let body = &command.body;
        let id = command.id.as_deref();
        let active_session_id = body
            .get("activeSessionId")
            .and_then(Value::as_str)
            .unwrap_or("");
        let admission_id = body.get("admissionId").and_then(Value::as_str);
        let admission_key = admission_id
            .map(|admission_id| self.prompt_admission_key(active_session_id, admission_id));
        let admission = admission_key.as_deref().and_then(|key| {
            self.prompt_admissions
                .lock()
                .expect("prompt admissions poisoned")
                .get(key)
                .map(|_| key.to_string())
        });
        if admission_id.is_some() && admission.is_none() {
            return Err("Prompt admission was not registered during command parsing".to_string());
        }
        let admission_entry = admission.as_deref().and_then(|_| {
            self.prompt_admissions
                .lock()
                .expect("prompt admissions poisoned")
                .get(&self.prompt_admission_key(active_session_id, admission_id.unwrap_or("")))
                .map(|entry| (entry.status.clone(), entry.controller.clone()))
        });
        let clear_admission = {
            let daemon = Arc::clone(self);
            let admission_key = admission.clone();
            Arc::new(move || {
                if let Some(admission_key) = &admission_key {
                    daemon
                        .prompt_admissions
                        .lock()
                        .expect("prompt admissions poisoned")
                        .remove(admission_key);
                }
            })
        };
        let commit_admission: Arc<dyn Fn() + Send + Sync> = {
            let daemon = Arc::clone(self);
            let admission_key = admission.clone();
            Arc::new(move || {
                if let Some(admission_key) = &admission_key {
                    let mut admissions = daemon
                        .prompt_admissions
                        .lock()
                        .expect("prompt admissions poisoned");
                    if let Some(entry) = admissions.get_mut(admission_key) {
                        if entry.status == "waiting" {
                            entry.status = "owned".to_string();
                        }
                    }
                }
            })
        };
        if admission_entry.as_ref().map(|(status, _)| status.as_str()) == Some("cancelled") {
            clear_admission();
            return Err(PromptAdmissionCancelledError::default().to_string());
        }
        let state = match self.get_bound_session_state(active_session_id) {
            Ok(state) => state,
            Err(error) => {
                clear_admission();
                return Err(error);
            }
        };
        let controller = admission_entry
            .as_ref()
            .and_then(|(_, controller)| controller.clone());
        let mut options = PromptInvocation {
            content: body.get("content").cloned(),
            images: body.get("images").cloned(),
            streaming_behavior: body
                .get("streamingBehavior")
                .and_then(Value::as_str)
                .map(str::to_string),
            // `command.queueIfBusy ?? command.streamingBehavior !== undefined`
            // (`daemon-mode.ts:4495`): `??` only falls back on null/undefined, so an
            // explicit `queueIfBusy: false` stays false even with a
            // streamingBehavior, instead of being coerced to queue-if-busy.
            queue_if_busy: Some(match body.get("queueIfBusy").and_then(Value::as_bool) {
                Some(queue_if_busy) => queue_if_busy,
                None => body.get("streamingBehavior").is_some(),
            }),
            resume_if_idle: Some(body.get("streamingBehavior").is_some()),
            expand_prompt_templates: body.get("expandPromptTemplates").and_then(Value::as_bool),
            skip_input_handlers: (body.get("expandPromptTemplates").and_then(Value::as_bool)
                == Some(false))
            .then_some(true),
            source: body
                .get("source")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..PromptInvocation::default()
        };
        if let Some(controller) = controller {
            options.signal = Some(controller);
            options.admission_committed = Some(Arc::clone(&commit_admission));
        }
        let message = body
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if command.type_ == "prompt_and_wait" {
            let daemon = Arc::clone(self);
            let state_for_preflight = Arc::clone(&state);
            let result = self
                .session_of(&state)
                .prompt_and_wait(
                    &message,
                    PromptInvocation {
                        preflight_result: Some(Arc::new(move |did_succeed: bool, _did_queue: bool| {
                            if did_succeed {
                                daemon.record_worker_recovery_state(
                                    &state_for_preflight,
                                    "prompt_accepted",
                                    Some(true),
                                );
                            }
                        })),
                        ..options
                    },
                )
                .await;
            clear_admission();
            result?;
            return Ok(Some(DaemonResponse::success(id, &command.type_, None)));
        }
        let daemon = Arc::clone(self);
        let client = Arc::clone(client);
        let state_for_prompt = Arc::clone(&state);
        let command_id = command.id.clone();
        let accept_agent_message = body.get("agentMessageId").and_then(Value::as_str).is_some()
            && body.get("expandPromptTemplates").and_then(Value::as_bool) == Some(false);
        // `let responseSent = false; let preflightRejected = false;`
        // (daemon-mode.ts:4521-4522): both flags live in the command scope, so the
        // settlement below can read the rejection the preflight callback recorded.
        let response_sent = Arc::new(AtomicBool::new(false));
        let preflight_rejected = Arc::new(AtomicBool::new(false));
        let invocation = PromptInvocation {
            agent_message_id: body
                .get("agentMessageId")
                .and_then(Value::as_str)
                .map(str::to_string),
            custom_message: body.get("customMessage").cloned(),
            preflight_result: {
                let daemon = Arc::clone(&daemon);
                let state = Arc::clone(&state);
                let client = Arc::clone(&client);
                let command_id = command_id.clone();
                let response_sent = Arc::clone(&response_sent);
                let preflight_rejected = Arc::clone(&preflight_rejected);
                let commit = Arc::clone(&commit_admission);
                Some(Arc::new(move |did_succeed: bool, did_queue: bool| {
                    if did_succeed {
                        daemon.record_worker_recovery_state(&state, "prompt_accepted", Some(true));
                        commit();
                        if !response_sent.swap(true, Ordering::SeqCst) {
                            daemon.write(
                                &client,
                                &DaemonOutbound::Raw(
                                    serde_json::to_value(DaemonResponse::success(
                                        command_id.as_deref(),
                                        "prompt",
                                        None,
                                    ))
                                    .unwrap_or(Value::Null),
                                ),
                            );
                        }
                    } else {
                        preflight_rejected.store(true, Ordering::SeqCst);
                    }
                    let _ = did_queue;
                }) as Arc<dyn Fn(bool, bool) + Send + Sync>)
            },
            ..options
        };
        let session = self.session_of(&state);
        let daemon = Arc::clone(self);
        let settle_response_sent = Arc::clone(&response_sent);
        let settle_preflight_rejected = Arc::clone(&preflight_rejected);
        tokio::spawn(async move {
            let result = if accept_agent_message {
                session
                    .accept_agent_message_prompt(&message, invocation)
                    .await
            } else {
                session.prompt_until_accepted(&message, invocation).await
            };
            match result {
                // `.then(() => { if (preflightRejected) write(failure(...)); else
                // sendSuccessResponse(); })` (daemon-mode.ts:4545-4552). A session that
                // reports a rejected preflight and then returns Ok still owes the
                // requester an answer; before this the rejection was swallowed and the
                // client waited out the request timeout.
                Ok(()) => {
                    if settle_preflight_rejected.load(Ordering::SeqCst) {
                        if !settle_response_sent.load(Ordering::SeqCst) {
                            daemon.write(
                                &client,
                                &DaemonOutbound::Raw(
                                    serde_json::to_value(DaemonResponse::failure(
                                        command_id.as_deref(),
                                        "prompt",
                                        "Prompt was not accepted by the session.",
                                        None,
                                    ))
                                    .unwrap_or(Value::Null),
                                ),
                            );
                        }
                    } else if !settle_response_sent.swap(true, Ordering::SeqCst) {
                        daemon.write(
                            &client,
                            &DaemonOutbound::Raw(
                                serde_json::to_value(DaemonResponse::success(
                                    command_id.as_deref(),
                                    "prompt",
                                    None,
                                ))
                                .unwrap_or(Value::Null),
                            ),
                        );
                    }
                }
                // `.catch((error) => { if (responseSent) broadcastToSession(...) else
                // write(failure(command.id, ...)) })` (daemon-mode.ts:4553-4558): the
                // requester hears the failure directly unless it was already answered.
                Err(error) => {
                    let failure = serde_json::to_value(DaemonResponse::failure(
                        command_id.as_deref(),
                        "prompt",
                        &error,
                        None,
                    ))
                    .unwrap_or(Value::Null);
                    if !settle_response_sent.load(Ordering::SeqCst) {
                        settle_response_sent.store(true, Ordering::SeqCst);
                        daemon.write(&client, &DaemonOutbound::Raw(failure));
                        clear_admission();
                        return;
                    }
                    let broadcast = daemon
                        .sessions
                        .lock()
                        .expect("sessions poisoned")
                        .get(
                            &state_for_prompt
                                .lock()
                                .expect("active session poisoned")
                                .active_session_id
                                .clone(),
                        )
                        .map(|entry| Arc::clone(&entry))
                        .is_some();
                    if !broadcast {
                        clear_admission();
                        return;
                    }
                    daemon.broadcast_to_session(
                        &daemon.session_entry_for_state(&state_for_prompt),
                        DaemonOutbound::Raw(failure),
                    );
                }
            }
            clear_admission();
        });
        Ok(None)
    }

    /// `handleReplaceAcpMcpServers(client, command)`.
    async fn handle_replace_acp_mcp_servers(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        command: &ParsedDaemonCommand,
    ) -> Result<Option<DaemonResponse>, String> {
        let body = &command.body;
        let id = command.id.as_deref();
        let state = self.get_session_state(
            body.get("activeSessionId")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )?;
        let servers = body
            .get("servers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let owner_id = client.id();
        let previous = self
            .acp_mcp_owners
            .lock()
            .expect("acp mcp owners poisoned")
            .get(
                &state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
            )
            .map(|owner| {
                (
                    Arc::clone(&owner.client),
                    owner.owner_id.clone(),
                    owner.server_names.clone(),
                )
            });
        if let Some((previous_client, previous_owner_id, previous_names)) = previous {
            if !Arc::ptr_eq(&previous_client, client) {
                self.session_of(&state)
                    .release_acp_mcp_servers(&previous_owner_id, &previous_names)
                    .await?;
            }
        }
        let server_values = servers.clone();
        self.session_of(&state)
            .replace_acp_mcp_servers(&server_values, &owner_id)
            .await?;
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.acp_mcp_owners
            .lock()
            .expect("acp mcp owners poisoned")
            .insert(
                active_session_id,
                AcpMcpOwner {
                    client: Arc::clone(client),
                    owner_id,
                    server_names: servers
                        .iter()
                        .filter_map(|server| {
                            server
                                .get("name")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .collect(),
                    release: None,
                },
            );
        Ok(Some(DaemonResponse::success(
            id,
            "replace_acp_mcp_servers",
            None,
        )))
    }
}

impl AgentDaemon {
    /// `hasPersistedResidentSession()`.
    fn has_persisted_resident_session(&self) -> bool {
        self.session_states()
            .iter()
            .any(|entry| entry.session.session_file().is_some())
    }

    /// `passiveRlmTopologyFingerprint(savedRootInfos)`.
    async fn passive_rlm_topology_fingerprint(self: &Arc<Self>, saved_root_infos: &[SessionInfo]) -> String {
        let ledger_stat = self
            .passive_rlm_stat_string(self.rlm_spawn_ledger().ledger_path())
            .await;
        let mut resident: Vec<String> = self
            .session_states()
            .iter()
            .map(|entry| {
                format!(
                    "{}:{}",
                    entry
                        .state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id,
                    entry.session.session_file().unwrap_or_default()
                )
            })
            .collect();
        resident.sort();
        let mut roots: Vec<String> = saved_root_infos
            .iter()
            .map(|info| resolve_path(&info.path))
            .collect();
        roots.sort();
        format!("{ledger_stat}|{}|{}", resident.join(","), roots.join(","))
    }

    /// `passiveRlmStatString(path)`.
    async fn passive_rlm_stat_string(&self, path: &str) -> String {
        match tokio::fs::metadata(path).await {
            Ok(stats) => {
                let modified = stats
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|duration| duration.as_millis())
                    .unwrap_or(0);
                format!("{}:{modified}:0", stats.len())
            }
            Err(_) => "absent".to_string(),
        }
    }

    /// `passiveRlmInputStatsUnchanged(inputStats)`.
    async fn passive_rlm_input_stats_unchanged(
        &self,
        input_stats: &HashMap<String, String>,
    ) -> bool {
        for (path, stat_string) in input_stats {
            if self.passive_rlm_stat_string(path).await != *stat_string {
                return false;
            }
        }
        true
    }

    /// `passiveRlmRootsStillResident(result)`.
    fn passive_rlm_roots_still_resident(&self, result: &[PassiveRlmSubagent]) -> bool {
        let sessions = self.sessions.lock().expect("sessions poisoned");
        result.iter().all(|passive| match &passive.root {
            PassiveRlmRoot::Resident(state) => sessions
                .values()
                .any(|entry| Arc::ptr_eq(&entry.state, state)),
            PassiveRlmRoot::Saved(_) => true,
        })
    }

    /// `listPassiveRlmSubagents(savedRoots = [], includeResident = false)`.
    async fn list_passive_rlm_subagents(
        self: &Arc<Self>,
        saved_roots: Vec<SessionInfo>,
        include_resident: bool,
    ) -> Vec<PassiveRlmSubagent> {
        // REPAIR CURSOR: `inactive_lifecycle_for_session` (daemon_session_list.rs:278)
        // takes the duplicate `daemon_session_list::SessionInfo`; `saved_roots` here is
        // `Vec<core::session_manager::SessionInfo>`. Same unowned-file duplicate-stub fix.
        let saved_root_infos: Vec<SessionInfo> = saved_roots
            .into_iter()
            .filter(|root_info| inactive_lifecycle_for_session(root_info) == SessionLifecycle::Live)
            .collect();
        if saved_root_infos.is_empty() && !self.has_persisted_resident_session() {
            // Keep the empty topology IO-free: no roots means no walk, no ledger stat.
            return Vec::new();
        }
        let mut root_paths: Vec<String> = saved_root_infos
            .iter()
            .map(|info| resolve_path(&info.path))
            .collect();
        root_paths.sort();
        let key = format!("{include_resident}|{}", root_paths.join(","));
        // Same-shape walks run one at a time; a caller never joins an earlier walk.
        let previous = self
            .passive_rlm_subagent_memo
            .lock()
            .expect("passive rlm memo poisoned")
            .get(&key)
            .and_then(|memo| memo.in_flight.clone());
        let waiters = match previous {
            Some(waiters) => waiters,
            None => {
                let waiters = Arc::new(Notify::new());
                self.passive_rlm_subagent_memo
                    .lock()
                    .expect("passive rlm memo poisoned")
                    .entry(key.clone())
                    .or_insert_with(|| PassiveRlmMemoEntry {
                        fingerprint: String::new(),
                        result: Vec::new(),
                        input_stats: HashMap::new(),
                        in_flight: None,
                    })
                    .in_flight = Some(Arc::clone(&waiters));
                waiters
            }
        };
        let result = self
            .passive_rlm_walk_run(&key, &saved_root_infos, include_resident)
            .await;
        {
            let mut memo = self
                .passive_rlm_subagent_memo
                .lock()
                .expect("passive rlm memo poisoned");
            if let Some(entry) = memo.get_mut(&key) {
                if entry
                    .in_flight
                    .as_ref()
                    .map(|current| Arc::ptr_eq(current, &waiters))
                    .unwrap_or(false)
                {
                    entry.in_flight = None;
                }
            }
            memo.retain(|_, entry| {
                entry.in_flight.is_some()
                    || !entry.fingerprint.is_empty()
                    || entry.result.is_empty()
            });
        }
        waiters.notify_waiters();
        result
    }

    /// The body of one passive walk, including the memo check and the write-back.
    async fn passive_rlm_walk_run(
        self: &Arc<Self>,
        key: &str,
        saved_root_infos: &[SessionInfo],
        include_resident: bool,
    ) -> Vec<PassiveRlmSubagent> {
        let before = self
            .passive_rlm_topology_fingerprint(saved_root_infos)
            .await;
        let memo = self
            .passive_rlm_subagent_memo
            .lock()
            .expect("passive rlm memo poisoned")
            .get(key)
            .filter(|memo| !memo.fingerprint.is_empty())
            .map(|memo| {
                (
                    memo.fingerprint.clone(),
                    memo.result.clone(),
                    memo.input_stats.clone(),
                )
            });
        if let Some((fingerprint, result, input_stats)) = memo {
            if fingerprint == before
                && self.passive_rlm_roots_still_resident(&result)
                && self.passive_rlm_input_stats_unchanged(&input_stats).await
            {
                return result;
            }
        }
        let walked = self
            .walk_passive_rlm_subagents(saved_root_infos, include_resident)
            .await;
        // Only a walk whose inputs held still qualifies as a memo: not the
        // ledger-seeding first walk, not a degraded one.
        let after = self
            .passive_rlm_topology_fingerprint(saved_root_infos)
            .await;
        if after == before && !walked.degraded {
            let mut memo = self
                .passive_rlm_subagent_memo
                .lock()
                .expect("passive rlm memo poisoned");
            let in_flight = memo.get(key).and_then(|entry| entry.in_flight.clone());
            memo.insert(
                key.to_string(),
                PassiveRlmMemoEntry {
                    fingerprint: after,
                    result: walked.result.clone(),
                    input_stats: walked.input_stats.clone(),
                    in_flight,
                },
            );
            while memo
                .iter()
                .filter(|(_, entry)| !entry.fingerprint.is_empty())
                .count()
                > AgentDaemon::PASSIVE_RLM_MEMO_MAX_KEYS
            {
                let stale = memo
                    .iter()
                    .find(|(candidate, entry)| {
                        candidate.as_str() != key && !entry.fingerprint.is_empty()
                    })
                    .map(|(candidate, _)| candidate.clone());
                match stale {
                    Some(stale) => {
                        memo.remove(&stale);
                    }
                    None => break,
                }
            }
        }
        walked.result
    }

    /// `walkPassiveRlmSubagents(savedRootInfos, includeResident)`.
    async fn walk_passive_rlm_subagents(
        self: &Arc<Self>,
        saved_root_infos: &[SessionInfo],
        include_resident: bool,
    ) -> PassiveRlmWalk {
        // Captured before each read: the identity can only be older than the content.
        let mut input_stats: HashMap<String, String> = HashMap::new();
        let mut resident_roots: Vec<(Arc<StdMutex<ActiveSessionState>>, String)> = Vec::new();
        for entry in self.session_states() {
            // An in-memory session cannot own persisted children.
            if let Some(parent_file) = entry.session.session_file() {
                resident_roots.push((Arc::clone(&entry.state), parent_file));
            }
        }
        if resident_roots.is_empty() && saved_root_infos.is_empty() {
            return PassiveRlmWalk {
                result: Vec::new(),
                input_stats,
                degraded: false,
            };
        }
        let edges = self.rlm_spawn_ledger().edges(false).await;
        let mut children_by_parent: HashMap<String, Vec<RlmLedgerEdge>> = HashMap::new();
        for edge in edges {
            let parent_path = canonical_session_path(&edge.parent);
            children_by_parent
                .entry(parent_path)
                .or_default()
                .push(edge);
        }
        let mut legacy_registry_cache: HashMap<String, Vec<LegacyRlmSubagentRegistryEntry>> =
            HashMap::new();
        let mut passive: Vec<PassiveRlmSubagent> = Vec::new();
        let mut degraded = false;
        let mut resident_root_paths: HashSet<String> = HashSet::new();
        for (parent_state, session_file) in resident_roots {
            let parent_path = resolve_path(&session_file);
            resident_root_paths.insert(parent_path.clone());
            let root = PassiveRlmRoot::Resident(Arc::clone(&parent_state));
            let parent_session_id = self.session_of(&parent_state).session_id();
            let mut visited = HashSet::new();
            visited.insert(parent_path);
            self.visit_passive_children(
                &root,
                &parent_session_id,
                &session_file,
                &[],
                &mut visited,
                include_resident,
                &children_by_parent,
                &mut legacy_registry_cache,
                &mut passive,
                &mut degraded,
                &mut input_stats,
            )
            .await;
        }
        for root_info in saved_root_infos {
            let root_path = resolve_path(&root_info.path);
            if resident_root_paths.contains(&root_path) {
                continue;
            }
            let root = PassiveRlmRoot::Saved(root_info.clone());
            let mut visited = HashSet::new();
            visited.insert(root_path);
            self.visit_passive_children(
                &root,
                &root_info.id,
                &root_info.path,
                &[],
                &mut visited,
                include_resident,
                &children_by_parent,
                &mut legacy_registry_cache,
                &mut passive,
                &mut degraded,
                &mut input_stats,
            )
            .await;
        }
        PassiveRlmWalk {
            result: passive,
            input_stats,
            degraded,
        }
    }

    /// The recursive `visit` closure of `walkPassiveRlmSubagents`.
    #[allow(clippy::too_many_arguments)]
    async fn visit_passive_children(
        self: &Arc<Self>,
        root: &PassiveRlmRoot,
        parent_session_id: &str,
        parent_session_file: &str,
        parent_chain: &[PassiveRlmSubagentEntry],
        visited: &mut HashSet<String>,
        include_resident: bool,
        children_by_parent: &HashMap<String, Vec<RlmLedgerEdge>>,
        legacy_registry_cache: &mut HashMap<String, Vec<LegacyRlmSubagentRegistryEntry>>,
        passive: &mut Vec<PassiveRlmSubagent>,
        degraded: &mut bool,
        input_stats: &mut HashMap<String, String>,
    ) {
        let edges = children_by_parent
            .get(&canonical_session_path(parent_session_file))
            .cloned()
            .unwrap_or_default();
        for edge in edges {
            self.record_input_stat(
                input_stats,
                &rlm_subagent_display_path(&dirname(&edge.child)),
            )
            .await;
            self.record_input_stat(
                input_stats,
                &self.legacy_rlm_subagent_registry_path(parent_session_file, parent_session_id),
            )
            .await;
            let entry_degraded = Arc::new(AtomicBool::new(false));
            {
                // `(path) => { if (inputStats.get(resolve(path)) !== "absent") degraded = true; }`
                // (daemon-mode.ts:1508). The registry reader takes an owned listener, so the
                // hook carries a snapshot of the stats recorded so far (nothing mutates them
                // while this edge is being read) plus a shared flag, instead of the walker's
                // mutable borrows.
                let stats_snapshot = Arc::new(input_stats.clone());
                let flag = Arc::clone(&entry_degraded);
                let on_read_error: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |path: &str| {
                    // A present metadata file may recover without a stat change.
                    if stats_snapshot.get(&resolve_path(path)).map(String::as_str) != Some("absent") {
                        flag.store(true, Ordering::SeqCst);
                    }
                });
                let entry = self
                    .passive_rlm_subagent_entry_for_edge(
                        &edge,
                        parent_session_id,
                        parent_session_file,
                        legacy_registry_cache,
                        on_read_error,
                    )
                    .await;
                if entry_degraded.load(Ordering::SeqCst) {
                    *degraded = true;
                }
                let session_key = resolve_path(&entry.session_file);
                if entry.status == "deleted" || visited.contains(&session_key) {
                    continue;
                }
                visited.insert(session_key);
                // A resident child walks as an outer root below; skipping before the
                // stat capture keeps its streamed transcript out of the input set.
                if !include_resident
                    && self
                        .find_session_by_session_file(Some(&entry.session_file))
                        .is_some()
                {
                    continue;
                }
                let child_stat = self
                    .record_input_stat(input_stats, &entry.session_file)
                    .await;
                let Some(info) = read_session_info(&entry.session_file).await else {
                    // A present file that fails to list may recover without a stat change.
                    if child_stat != "absent" {
                        *degraded = true;
                    }
                    continue;
                };
                let mut chain = parent_chain.to_vec();
                chain.push(entry.clone());
                passive.push(PassiveRlmSubagent {
                    root: root.clone(),
                    entry: entry.clone(),
                    info: info.clone(),
                    chain: chain.clone(),
                });
                Box::pin(self.visit_passive_children(
                    root,
                    &info.id,
                    &entry.session_file,
                    &chain,
                    visited,
                    include_resident,
                    children_by_parent,
                    legacy_registry_cache,
                    passive,
                    degraded,
                    input_stats,
                ))
                .await;
            }
        }
    }

    /// `recordInputStat(path)`.
    async fn record_input_stat(
        &self,
        input_stats: &mut HashMap<String, String>,
        path: &str,
    ) -> String {
        let resolved = resolve_path(path);
        if let Some(existing) = input_stats.get(&resolved).cloned() {
            return existing;
        }
        let stat_string = self.passive_rlm_stat_string(path).await;
        input_stats.insert(resolved, stat_string.clone());
        stat_string
    }

    /// `passiveRlmSubagentEntryForEdge(edge, parent, legacyRegistryCache?, onReadError?)`.
    async fn passive_rlm_subagent_entry_for_edge(
        self: &Arc<Self>,
        edge: &RlmLedgerEdge,
        parent_session_id: &str,
        parent_session_file: &str,
        legacy_registry_cache: &mut HashMap<String, Vec<LegacyRlmSubagentRegistryEntry>>,
        on_read_error: Arc<dyn Fn(&str) + Send + Sync>,
    ) -> PassiveRlmSubagentEntry {
        let edge_child = canonical_session_path(&edge.child);
        let base = PassiveRlmSubagentEntry {
            child_id: edge.child_id.clone(),
            session_name: edge.name.clone(),
            session_dir: dirname(&edge.child),
            session_file: edge.child.clone(),
            parent_session_id: parent_session_id.to_string(),
            parent_session_file: Some(parent_session_file.to_string()),
            ..PassiveRlmSubagentEntry::default()
        };
        // The ledger stores realpath-canonical paths, the rest of the daemon keys
        // maps by resolve(): present the paths the writer recorded.
        let metadata_fields = |source: &PassiveRlmSubagentEntry| -> PassiveRlmSubagentEntry {
            let mut fields = base.clone();
            if canonical_session_path(&source.session_file) == edge_child {
                fields.session_dir = source.session_dir.clone();
                fields.session_file = source.session_file.clone();
            }
            let metadata = rlm_subagent_metadata_fields(source);
            fields.rlm_max_depth = metadata.rlm_max_depth;
            fields.rlm_parent_node_id = metadata.rlm_parent_node_id;
            fields.prompt = metadata.prompt;
            fields.spawn_code = metadata.spawn_code;
            fields.model = metadata.model;
            fields.status = source.status.clone();
            fields.created_at = source.created_at;
            fields
        };
        let display_root = dirname(&edge.child);
        let display_path = rlm_subagent_display_path(&display_root);
        let display = {
            let on_read_error = Arc::clone(&on_read_error);
            let path = display_path.clone();
            let report_display_error = move || on_read_error(&path);
            let listener: Option<&(dyn Fn() + Send + Sync)> = Some(&report_display_error);
            read_rlm_subagent_display_entry(&display_root, listener).await
        };
        if let Some(display) = display {
            if display.child_id == edge.child_id {
                // A display-file child was ledger-spawned: the edge depth is real.
                let mut fields = metadata_fields(&passive_entry_from_display(&display));
                fields.rlm_depth = Some(edge.depth);
                return fields;
            }
        }
        let registry_path =
            self.legacy_rlm_subagent_registry_path(parent_session_file, parent_session_id);
        if !legacy_registry_cache.contains_key(&registry_path) {
            let registry_path_for_error = registry_path.clone();
            let on_read_error = Arc::clone(&on_read_error);
            let listener: Option<Arc<dyn Fn() + Send + Sync>> =
                Some(Arc::new(move || on_read_error(&registry_path_for_error)));
            let entries = self
                .read_legacy_rlm_subagent_registry(&registry_path, listener)
                .await
                .unwrap_or_default();
            legacy_registry_cache.insert(registry_path.clone(), entries);
        }
        let registry_read = legacy_registry_cache
            .get(&registry_path)
            .cloned()
            .unwrap_or_default();
        if let Some(legacy) = registry_read
            .into_iter()
            .find(|entry| entry.child_id == edge.child_id)
        {
            // A seeded edge's depth may be a parent+1 guess for legacy entries
            // without one: leave it absent so hydration falls back to the persisted
            // header depth, exactly as the registry reader did.
            let mut fields = metadata_fields(&PassiveRlmSubagentEntry {
                session_dir: legacy.session_dir.clone(),
                session_file: legacy.session_file.clone(),
                rlm_max_depth: legacy.rlm_max_depth,
                rlm_parent_node_id: legacy.rlm_parent_node_id.clone(),
                prompt: legacy.prompt.clone(),
                spawn_code: legacy.spawn_code.clone(),
                model: legacy
                    .model
                    .clone()
                    .and_then(|value| serde_json::from_value(value).ok()),
                status: legacy.status.clone(),
                created_at: legacy.created_at,
                ..PassiveRlmSubagentEntry::default()
            });
            if legacy.rlm_depth.is_some() {
                fields.rlm_depth = legacy.rlm_depth;
            }
            return fields;
        }
        // Ledger-only child (metadata lost): hydratable with defaults. The stat is
        // display-grade; a failed read keeps the epoch default.
        let mut created_at = 0.0;
        if let Ok(metadata) = tokio::fs::metadata(&edge.child).await {
            if let Ok(created) = metadata.created() {
                if let Ok(duration) = created.duration_since(std::time::UNIX_EPOCH) {
                    created_at = duration.as_millis() as f64;
                }
            }
        }
        PassiveRlmSubagentEntry {
            rlm_depth: Some(edge.depth),
            status: "completed".to_string(),
            created_at,
            ..base
        }
    }

    /// `passiveRlmSubagentsByPath(savedRoots = [], includeResident = false)`.
    async fn passive_rlm_subagents_by_path(
        self: &Arc<Self>,
        saved_roots: Vec<SessionInfo>,
        include_resident: bool,
    ) -> HashMap<String, PassiveRlmSubagent> {
        self.list_passive_rlm_subagents(saved_roots, include_resident)
            .await
            .into_iter()
            .map(|passive| (resolve_path(&passive.entry.session_file), passive))
            .collect()
    }

    /// `findPassiveRlmSubagent(target, includeResident = false)`.
    async fn find_passive_rlm_subagent(
        self: &Arc<Self>,
        target: &str,
        include_resident: bool,
    ) -> Option<PassiveRlmSubagent> {
        let matches: Vec<PassiveRlmSubagent> = self
            .passive_rlm_subagents_by_path(Vec::new(), include_resident)
            .await
            .into_values()
            .filter(|passive| {
                passive.entry.child_id == target
                    || resolve_path(&passive.entry.session_file) == resolve_path(target)
                    || passive.info.id == target
                    || passive
                        .info
                        .name
                        .clone()
                        .unwrap_or_else(|| passive.entry.session_name.clone())
                        == target
            })
            .collect();
        if matches.len() > 1 {
            return None;
        }
        matches.into_iter().next()
    }

    /// `buildRlmChildSnapshotsWithPassiveRlmSubagents(rootState)`.
    async fn build_rlm_child_snapshots_with_passive(
        self: &Arc<Self>,
        root_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Result<Vec<Value>, String> {
        let root_active_session_id = root_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut snapshots = self.build_rlm_child_snapshots_plumbing(&root_active_session_id);
        let mut resident_parent_ids: HashSet<String> = HashSet::new();
        resident_parent_ids.insert(root_active_session_id.clone());
        for snapshot in &snapshots {
            if let Some(active_session_id) = snapshot.get("activeSessionId").and_then(Value::as_str)
            {
                resident_parent_ids.insert(active_session_id.to_string());
            }
        }
        let mut seen_child_ids: HashSet<String> = snapshots
            .iter()
            .filter_map(|snapshot| {
                snapshot
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        for passive in self.list_passive_rlm_subagents(Vec::new(), false).await {
            let PassiveRlmRoot::Resident(parent_state) = &passive.root else {
                continue;
            };
            let parent_active_session_id = parent_state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                .clone();
            if !resident_parent_ids.contains(&parent_active_session_id)
                || seen_child_ids.contains(&passive.entry.child_id)
            {
                continue;
            }
            let parent_entry = passive
                .chain
                .len()
                .checked_sub(2)
                .and_then(|index| passive.chain.get(index));
            let parent_id = parent_entry
                .map(|entry| entry.child_id.clone())
                .or_else(|| {
                    parent_state
                        .lock()
                        .expect("active session poisoned")
                        .runtime
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.rlm_child_id.clone())
                });
            let mut snapshot = Map::new();
            snapshot.insert(
                "id".to_string(),
                Value::String(passive.entry.child_id.clone()),
            );
            if let Some(parent_id) = parent_id {
                snapshot.insert("parentId".to_string(), Value::String(parent_id));
            }
            snapshot.insert(
                "sessionName".to_string(),
                Value::String(
                    passive
                        .info
                        .name
                        .clone()
                        .unwrap_or_else(|| passive.entry.session_name.clone()),
                ),
            );
            if let Some(model) = &passive.entry.model {
                snapshot.insert(
                    "model".to_string(),
                    Value::String(format!("{}/{}", model.provider, model.model_id)),
                );
            }
            snapshot.insert(
                "label".to_string(),
                Value::String(rlm_child_label(
                    passive.entry.prompt.as_deref().unwrap_or(""),
                )),
            );
            snapshot.insert(
                "status".to_string(),
                Value::String(
                    if passive.entry.status == "completed" {
                        "done"
                    } else {
                        "error"
                    }
                    .to_string(),
                ),
            );
            snapshot.insert(
                "sessionDir".to_string(),
                Value::String(passive.entry.session_dir.clone()),
            );
            snapshots.push(Value::Object(snapshot));
            seen_child_ids.insert(passive.entry.child_id.clone());
        }
        Ok(snapshots)
    }

    /// `buildSessionListWithPassiveRlmSubagents(activeSessions, savedSessions, scheduledJobs)`.
    async fn build_session_list_with_passive_rlm_subagents(
        self: &Arc<Self>,
        active_sessions: &[Arc<StdMutex<ActiveSessionState>>],
        saved_sessions: Vec<SessionInfo>,
        scheduled_jobs: &[AgentCronJob],
    ) -> Vec<Value> {
        let passive_by_path = self
            .passive_rlm_subagents_by_path(saved_sessions.clone(), false)
            .await;
        let mut saved_by_path: HashMap<String, SessionInfo> = saved_sessions
            .into_iter()
            .map(|session| (resolve_path(&session.path), session))
            .collect();
        for (path, passive) in &passive_by_path {
            saved_by_path.insert(path.clone(), passive.info.clone());
        }
        let saved = saved_by_path.into_values().collect::<Vec<_>>();
        // REPAIR CURSOR: `build_session_list` (daemon_session_list.rs:317) takes its own
        // duplicate `SessionInfo` and `AgentCronJob` stubs, not the canonical
        // `core::session_manager::SessionInfo` / `core::cron_jobs::AgentCronJob` this file
        // holds. Same unowned-file fix as the `summary_for_active_session` cursor above.
        build_session_list(active_sessions, &saved, scheduled_jobs)
            .into_iter()
            .map(|summary| {
                let passive = summary
                    .session_file
                    .as_deref()
                    .and_then(|session_file| passive_by_path.get(&resolve_path(session_file)));
                let Some(passive) = passive else {
                    return serde_json::to_value(summary).unwrap_or(Value::Null);
                };
                if summary.active_session_id.is_some() {
                    return serde_json::to_value(summary).unwrap_or(Value::Null);
                }
                let mut value = serde_json::to_value(&summary).unwrap_or(Value::Null);
                let parent_entry = passive
                    .chain
                    .len()
                    .checked_sub(2)
                    .and_then(|index| passive.chain.get(index));
                let root_parent_active_session_id = match &passive.root {
                    PassiveRlmRoot::Resident(state) => Some(
                        state
                            .lock()
                            .expect("active session poisoned")
                            .active_session_id
                            .clone(),
                    ),
                    PassiveRlmRoot::Saved(_) => None,
                };
                let root_parent_session_file = match &passive.root {
                    PassiveRlmRoot::Resident(state) => self.session_of(state).session_file(),
                    PassiveRlmRoot::Saved(info) => Some(info.path.clone()),
                };
                if let Some(object) = value.as_object_mut() {
                    object.insert(
                        "runtimeKind".to_string(),
                        Value::String("subagent".to_string()),
                    );
                    if passive.chain.len() == 1 {
                        if let Some(root_parent_active_session_id) = &root_parent_active_session_id
                        {
                            object.insert(
                                "parentActiveSessionId".to_string(),
                                Value::String(root_parent_active_session_id.clone()),
                            );
                        }
                    }
                    object.insert(
                        "parentSessionId".to_string(),
                        Value::String(passive.entry.parent_session_id.clone()),
                    );
                    if let Some(parent_session_path) = passive
                        .entry
                        .parent_session_file
                        .clone()
                        .or_else(|| parent_entry.map(|entry| entry.session_file.clone()))
                        .or(root_parent_session_file)
                    {
                        object.insert(
                            "parentSessionPath".to_string(),
                            Value::String(parent_session_path),
                        );
                    }
                    if let Some(rlm_depth) =
                        passive.entry.rlm_depth.or(Some(passive.info.rlm_depth))
                    {
                        object.insert("rlmDepth".to_string(), Value::from(rlm_depth));
                    }
                    object.insert(
                        "rlmChildId".to_string(),
                        Value::String(passive.entry.child_id.clone()),
                    );
                    object.insert(
                        "rlmParentNodeId".to_string(),
                        Value::String(
                            passive
                                .entry
                                .rlm_parent_node_id
                                .clone()
                                .unwrap_or_else(|| passive.entry.child_id.clone()),
                        ),
                    );
                    if let Some(spawn_code) = &passive.entry.spawn_code {
                        object.insert("spawnCode".to_string(), Value::String(spawn_code.clone()));
                    }
                }
                value
            })
            .collect()
    }
}

/// One `walkPassiveRlmSubagents` result.
struct PassiveRlmWalk {
    result: Vec<PassiveRlmSubagent>,
    input_stats: HashMap<String, String>,
    degraded: bool,
}

impl AgentDaemon {
    /// `getOrHydrateBoundSessionState(id)`.
    async fn get_or_hydrate_bound_session_state(
        self: &Arc<Self>,
        id: &str,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let mut lookup_error: Option<String> = None;
        match self.get_bound_session_state(id) {
            Ok(state) => return Ok(state),
            Err(error) => {
                if error.starts_with("__bound_session_unavailable__") {
                    let state = self.get_session_state(id)?;
                    return self.wait_for_hydrating_child(state, id).await;
                }
                if error.starts_with("__ambiguous_active_session__") {
                    return Err(error);
                }
                lookup_error = Some(error);
            }
        }
        if let Some(passive) = self.find_passive_rlm_subagent(id, false).await {
            return self.hydrate_passive_rlm_subagent(passive, None).await;
        }
        let hydrating_child = self.session_states().into_iter().find(|entry| {
            let state = entry.state.lock().expect("active session poisoned");
            let metadata = state.runtime.metadata.as_ref();
            metadata.and_then(|metadata| metadata.kind.as_deref()) == Some("subagent")
                && metadata.and_then(|metadata| metadata.rlm_child_id.as_deref()) == Some(id)
        });
        if let Some(hydrating_child) = hydrating_child {
            return self
                .wait_for_hydrating_child(Arc::clone(&hydrating_child.state), id)
                .await;
        }
        match self.get_bound_session_state(id) {
            Ok(state) => Ok(state),
            Err(error) => {
                if error.starts_with("__bound_session_unavailable__") {
                    let state = self.get_session_state(id)?;
                    return self.wait_for_hydrating_child(state, id).await;
                }
                if error.starts_with("__ambiguous_active_session__") {
                    return Err(error);
                }
                Err(lookup_error.unwrap_or(error))
            }
        }
    }

    /// `waitForHydratingChild(state, selector)`.
    async fn wait_for_hydrating_child(
        self: &Arc<Self>,
        state: Arc<StdMutex<ActiveSessionState>>,
        selector: &str,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let session_file = self.session_of(&state).session_file();
        match session_file {
            Some(session_file)
                if self
                    .find_passivation_by_session_file(&session_file)
                    .is_some() => {}
            _ => return self.wait_for_bound_session(state).await,
        }
        let session_file = self.session_of(&state).session_file().unwrap_or_default();
        self.wait_for_passivation(&session_file).await;
        match self.find_passive_rlm_subagent(&session_file, false).await {
            Some(passive) => self.hydrate_passive_rlm_subagent(passive, None).await,
            None => Box::pin(self.get_or_hydrate_bound_session_state(selector)).await,
        }
    }

    /// `waitForBoundSession(state)`.
    async fn wait_for_bound_session(
        self: &Arc<Self>,
        state: Arc<StdMutex<ActiveSessionState>>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if let Some(completion) = self
            .binding_completions
            .lock()
            .expect("binding completions poisoned")
            .get(&active_session_id)
            .copied()
        {
            let _ = completion;
        }
        let resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .map(|entry| Arc::ptr_eq(&entry.state, &state))
            .unwrap_or(false);
        if !resident
            || self
                .binding_sessions
                .lock()
                .expect("binding sessions poisoned")
                .contains(&active_session_id)
        {
            return Err(BoundSessionUnavailableError::new(format!(
                "Active session {active_session_id} did not finish initializing"
            ))
            .to_string());
        }
        if self
            .closing_sessions
            .lock()
            .expect("closing sessions poisoned")
            .contains_key(&active_session_id)
        {
            return Err(BoundSessionUnavailableError::new(format!(
                "Active session {active_session_id} is closing"
            ))
            .to_string());
        }
        Ok(state)
    }

    /// `findPassivationBySessionFile(sessionFile)`.
    fn find_passivation_by_session_file(&self, session_file: &str) -> Option<u64> {
        self.passivating_sessions
            .lock()
            .expect("passivating sessions poisoned")
            .get(&resolve_path(session_file))
            .copied()
    }

    /// `waitForPassivation(sessionFile)`.
    async fn wait_for_passivation(&self, session_file: &str) {
        if self
            .find_passivation_by_session_file(session_file)
            .is_some()
        {
            // The passivation walk is cooperative; yielding lets its close finish.
            delay(1).await;
        }
    }

    /// `sessionPassivationSnapshot(state, passiveRlmSubagents?)`.
    async fn session_passivation_snapshot(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        passive_rlm_subagents: Option<Vec<PassiveRlmSubagent>>,
    ) -> SessionPassivationSnapshot {
        let passive_descendants = match passive_rlm_subagents {
            Some(passive) => passive,
            None => self.list_passive_rlm_subagents(Vec::new(), false).await,
        };
        let summary = self.summary_for_state(state);
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let session_file = self.session_of(state).session_file();
        let jobs: Vec<AgentCronJob> = self
            .cron_store
            .list()
            .into_iter()
            .filter(|job| {
                job.active_session_id == active_session_id
                    && job.status != crate::core::cron_jobs::STATUS_CANCELLED
                    && job.status != crate::core::cron_jobs::STATUS_COMPLETED
            })
            .collect();
        let has_pending_admission = self
            .prompt_admissions
            .lock()
            .expect("prompt admissions poisoned")
            .values()
            .any(|admission| {
                admission.active_session_id == active_session_id && admission.status != "cancelled"
            });
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let binding = self
            .binding_sessions
            .lock()
            .expect("binding sessions poisoned")
            .contains(&active_session_id);
        let opening = session_file
            .as_deref()
            .map(|file| {
                self.opening_sessions
                    .lock()
                    .expect("opening sessions poisoned")
                    .contains_key(&resolve_path(file))
            })
            .unwrap_or(false);
        let passive_opening = passive_descendants.iter().any(|passive| {
            let PassiveRlmRoot::Resident(root_state) = &passive.root else {
                return false;
            };
            let root_id = root_state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                .clone();
            root_id == active_session_id
                && passive.chain.iter().any(|entry| {
                    self.opening_sessions
                        .lock()
                        .expect("opening sessions poisoned")
                        .contains_key(&resolve_path(&entry.session_file))
                })
        });
        // Read both counters under one guard. Two lock() temporaries in the
        // addition live to the end of the expression and deadlock this session.
        // Release the guard before the descendant walk below locks state again.
        let attached_clients = {
            let state = state.lock().expect("active session poisoned");
            state.clients.len() as i64 + state.pending_attaches as i64
        };
        SessionPassivationSnapshot {
            eviction: SessionEvictionSnapshot {
                is_session_active: summary.is_session_active
                    || summary.has_running_rlm_children == Some(true)
                    || has_pending_admission,
                attached_clients,
                has_registered_cron_job: jobs.iter().any(|job| !is_heartbeat_cron_job(job)),
                last_activity_at: summary
                    .last_activity_at
                    .as_deref()
                    .map(|value| {
                        chrono::DateTime::parse_from_rfc3339(value)
                            .map(|date| date.timestamp_millis() as f64)
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0),
            },
            has_parent: metadata
                .as_ref()
                .and_then(|metadata| metadata.kind.as_deref())
                == Some("subagent")
                && metadata
                    .as_ref()
                    .and_then(|metadata| metadata.parent_active_session_id.as_deref())
                    .is_some(),
            has_non_passive_descendants: !get_child_active_session_states(
                &self.states_by_id(),
                state,
            )
            .is_empty(),
            is_hydrating: binding || opening || passive_opening,
        }
    }

    /// `passivateSession(state, idleEvictionMinutes, now, selectedSnapshot?)`.
    async fn passivate_session(
        self: &Arc<Self>,
        state: Arc<StdMutex<ActiveSessionState>>,
        idle_eviction_minutes: IdleEvictionMinutes,
        now: f64,
        selected_snapshot: Option<SessionPassivationSnapshot>,
    ) -> bool {
        let session_file = self.session_of(&state).session_file();
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let (Some(session_file), Some(metadata)) = (session_file, metadata) else {
            return false;
        };
        if metadata.kind.as_deref() != Some("subagent")
            || metadata.rlm_child_id.is_none()
            || metadata.parent_active_session_id.is_none()
        {
            return false;
        }
        let session_key = resolve_path(&session_file);
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let parent_active_session_id = metadata
            .parent_active_session_id
            .clone()
            .unwrap_or_default();
        let child_id = metadata.rlm_child_id.clone().unwrap_or_default();
        if self
            .passivating_sessions
            .lock()
            .expect("passivating sessions poisoned")
            .contains_key(&session_key)
        {
            self.wait_for_passivation(&session_file).await;
            return false;
        }
        let snapshot = match selected_snapshot {
            Some(snapshot) => snapshot,
            None => self.session_passivation_snapshot(&state, None).await,
        };
        if !can_passivate_session(&snapshot, idle_eviction_minutes, now) {
            return false;
        }
        // Publish the durable identity before running the close so opens and lazy
        // hydration can join throughout closeSessionOnce, including after
        // sessions.delete.
        let passivation = self.next_id();
        self.passivating_sessions
            .lock()
            .expect("passivating sessions poisoned")
            .insert(session_key.clone(), passivation);
        let closed = self
            .passivate_session_once(
                &state,
                &session_key,
                &active_session_id,
                &parent_active_session_id,
                &child_id,
                idle_eviction_minutes,
                now,
                &snapshot,
            )
            .await;
        {
            let mut passivating = self
                .passivating_sessions
                .lock()
                .expect("passivating sessions poisoned");
            if passivating.get(&session_key).copied() == Some(passivation) {
                passivating.remove(&session_key);
            }
        }
        closed
    }

    /// The inner `Promise.resolve().then(...)` of `passivateSession`.
    #[allow(clippy::too_many_arguments)]
    async fn passivate_session_once(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        session_key: &str,
        active_session_id: &str,
        parent_active_session_id: &str,
        child_id: &str,
        idle_eviction_minutes: IdleEvictionMinutes,
        now: f64,
        snapshot: &SessionPassivationSnapshot,
    ) -> bool {
        let _ = session_key;
        // Fence against touches and state changes after candidate selection. This
        // snapshot is intentionally fresh rather than reusing the sweep snapshot.
        let residency = |daemon: &Arc<Self>| {
            daemon
                .sessions
                .lock()
                .expect("sessions poisoned")
                .get(active_session_id)
                .map(|entry| Arc::ptr_eq(&entry.state, state))
                .unwrap_or(false)
        };
        if self.shutting_down.load(Ordering::SeqCst)
            || self
                .update_restart
                .lock()
                .expect("update restart poisoned")
                .is_some()
            || !residency(self)
        {
            return false;
        }
        let fresh = self.session_passivation_snapshot(state, None).await;
        if !can_passivate_session(&fresh, idle_eviction_minutes, now) {
            return false;
        }
        let parent_state = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(parent_active_session_id)
            .map(|entry| Arc::clone(&entry.state));
        let Some(parent_state) = parent_state else {
            return false;
        };
        let idle_minutes = ((now - snapshot.eviction.last_activity_at) / 60_000.0).floor() as i64;
        // Detach parent tracking before the standard graceful runtime disposal. The
        // registry/catalog rows remain the sole passive representation after close.
        let unsubscribe_child = self
            .session_of(&parent_state)
            .release_rlm_child_session(child_id, self.session_of(state));
        let Some(unsubscribe_child) = unsubscribe_child else {
            return false;
        };
        // Capture the child's identity before the close: after close_session the
        // session seam is MissingSession, whose id is empty and name is None, so
        // the passivation log line would lose the session it acted on (audit D-05).
        let passivated_session_id = self.session_of(state).session_id();
        let passivated_session_name = self.session_of(state).session_name();
        let close_result = self
            .close_session(Arc::clone(state), "shutdown", true, false, None, None)
            .await;
        let still_resident = residency(self);
        let parent_resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(parent_active_session_id)
            .is_some();
        unsubscribe_child();
        if close_result.is_err() {
            if still_resident && parent_resident {
                self.log("could not passivate idle child session");
            }
            return false;
        }
        self.log(&format!(
            "Passivated idle child sessionId={passivated_session_id} name={} idleMinutes={idle_minutes}",
            serde_json::to_string(&passivated_session_name)
                .unwrap_or_else(|_| "\"\"".to_string()),
        ));
        residency(self) == false
    }

    /// `passivateIdleChildren(idleEvictionMinutes, now, limit)`.
    async fn passivate_idle_children(
        self: &Arc<Self>,
        idle_eviction_minutes: IdleEvictionMinutes,
        now: f64,
        limit: usize,
    ) -> usize {
        if self.shutting_down.load(Ordering::SeqCst)
            || self
                .update_restart
                .lock()
                .expect("update restart poisoned")
                .is_some()
            || limit == 0
        {
            return 0;
        }
        let states = self.state_refs();
        let passive_rlm_subagents = self.list_passive_rlm_subagents(Vec::new(), false).await;
        let mut snapshots: Vec<(
            Arc<StdMutex<ActiveSessionState>>,
            SessionPassivationSnapshot,
        )> = Vec::new();
        for state in states {
            let snapshot = self
                .session_passivation_snapshot(&state, Some(passive_rlm_subagents.clone()))
                .await;
            snapshots.push((state, snapshot));
        }
        let mut candidates: Vec<(
            Arc<StdMutex<ActiveSessionState>>,
            SessionPassivationSnapshot,
        )> = snapshots
            .into_iter()
            .filter(|(_, snapshot)| can_passivate_session(snapshot, idle_eviction_minutes, now))
            .collect();
        candidates.sort_by(|left, right| {
            left.1
                .eviction
                .last_activity_at
                .partial_cmp(&right.1.eviction.last_activity_at)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        let mut passivated = 0usize;
        for (state, snapshot) in candidates {
            if self
                .passivate_session(state, idle_eviction_minutes, now, Some(snapshot))
                .await
            {
                passivated += 1;
            }
        }
        passivated
    }
}

impl AgentDaemon {
    /// `newSession()` on the resident runtime (`state.runtime.newSession(options)`).
    /// `switchSession()` on the resident runtime (`state.runtime.switchSession(...)`).
    async fn runtime_switch_session(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        session_path: &str,
        options: SessionPathOptions,
    ) -> Result<Value, String> {
        let entry = self.session_entry_for_state(state);
        self.session_of(state)
            .switch_session(session_path, options)
            .await
            .map(|result| {
                entry.sync_view();
                result
            })
    }

    /// `fork()` on the resident runtime (`state.runtime.fork(...)`).
    async fn runtime_fork(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        entry_id: &str,
        options: ForkOptions,
    ) -> Result<Value, String> {
        let entry = self.session_entry_for_state(state);
        self.session_of(state)
            .fork(entry_id, options)
            .await
            .map(|result| {
                entry.sync_view();
                result
            })
    }

    /// `importFromJsonl()` on the resident runtime.
    async fn runtime_import_from_jsonl(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        input_path: &str,
        cwd_override: Option<String>,
    ) -> Result<Value, String> {
        let entry = self.session_entry_for_state(state);
        self.session_of(state)
            .import_from_jsonl(input_path, cwd_override.as_deref())
            .await
            .map(|result| {
                entry.sync_view();
                result
            })
    }

    /// `newSession()` on the resident runtime (`state.runtime.newSession(options)`).
    async fn runtime_new_session(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        options: Option<NewSessionRuntimeOptions>,
    ) -> Result<Value, String> {
        let entry = self.session_entry_for_state(state);
        let parent_session = options.and_then(|options| options.parent_session);
        self.session_of(state)
            .new_session(Some(NewSessionRuntimeOptions { parent_session }))
            .await
            .map(|result| {
                entry.sync_view();
                result
            })
    }

    /// `state.inFlightBash`: the join point `abortBashForClose` races.
    fn in_flight_bash_slot(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> Arc<Notify> {
        let mut slots = self.in_flight_bash.lock().expect("in flight bash poisoned");
        Arc::clone(
            slots
                .entry(
                    state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                )
                .or_insert_with(|| Arc::new(Notify::new())),
        )
    }

    /// `trackInFlightBash(state, bash)` — the `state.inFlightBash` chain.
    fn track_in_flight_bash(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        bash: BoxFuture<'static, Result<Value, String>>,
    ) {
        let in_flight = self.in_flight_bash_slot(state);
        tokio::spawn(async move {
            let _ = bash.await;
            in_flight.notify_waiters();
        });
    }

    /// `setStateSessionNameForCommand(state, name)`.
    async fn set_state_session_name_for_command(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> Result<(), String> {
        if self.is_worker() {
            self.apply_state_session_name(state, name).await
        } else {
            self.set_state_session_name(state, name).await
        }
    }

    /// `setStateSessionName(state, name)`.
    async fn set_state_session_name(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> Result<(), String> {
        let normalized_name = name.trim().to_string();
        if normalized_name.is_empty() {
            return Err("Session name cannot be empty".to_string());
        }
        let reservation = self.state_session_name_reservation(state, &normalized_name);
        self.with_session_name_reservation(reservation.clone(), |daemon| {
            let state = Arc::clone(state);
            let normalized_name = normalized_name.clone();
            Box::pin(async move {
                daemon
                    .assert_state_session_name_available(&state, &normalized_name)
                    .await?;
                daemon
                    .apply_state_session_name(&state, &normalized_name)
                    .await
            })
        })
        .await
    }

    /// `applyStateSessionName(state, name)`.
    async fn apply_state_session_name(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> Result<(), String> {
        self.session_of(state).set_session_name(name);
        self.append_rlm_ledger_rename_for_state(state, name).await;
        Ok(())
    }

    /// The `{ name, depth, parentSessionId?, parentSessionPath? }` argument of
    /// `withSessionNameReservation`.
    fn state_session_name_reservation(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> NameReservationInput {
        let session = self.session_of(state);
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let depth = session.rlm_depth().unwrap_or(0);
        let header_parent = if depth > 0 {
            self.resolve_header_parent_session_path(state)
        } else {
            None
        };
        NameReservationInput {
            name: name.to_string(),
            depth: depth as f64,
            parent_session_id: (depth > 0 && header_parent.is_none())
                .then(|| {
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_id.clone())
                })
                .flatten(),
            parent_session_path: if depth > 0 {
                header_parent.or_else(|| {
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_file.clone())
                })
            } else {
                None
            },
        }
    }

    /// `withSessionNameReservation(input, action)`.
    async fn with_session_name_reservation<T>(
        self: &Arc<Self>,
        input: NameReservationInput,
        action: impl FnOnce(Arc<Self>) -> BoxFuture<'static, Result<T, String>>,
    ) -> Result<T, String> {
        let scope = AgentSessionNameScope {
            parent_session_id: input.parent_session_id.clone(),
            parent_session_path: input.parent_session_path.clone(),
            depth: input.depth,
        };
        let key = session_name_reservation_key(&scope, &input.name);
        if self
            .pending_session_names
            .lock()
            .expect("pending session names poisoned")
            .contains(&key)
        {
            return Err(format_agent_session_name_unavailable(
                &input.name,
                input.depth,
            ));
        }
        self.pending_session_names
            .lock()
            .expect("pending session names poisoned")
            .insert(key.clone());
        let result = action(Arc::clone(self)).await;
        self.pending_session_names
            .lock()
            .expect("pending session names poisoned")
            .remove(&key);
        result
    }

    /// `assertStateSessionNameAvailable(state, name)`.
    async fn assert_state_session_name_available(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> Result<(), String> {
        let session = self.session_of(state);
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let depth = session.rlm_depth().unwrap_or(0);
        let header_parent = if depth > 0 {
            self.resolve_header_parent_session_path(state)
        } else {
            None
        };
        let input = AgentSessionNameAvailabilityInput {
            name: name.to_string(),
            depth: depth as f64,
            parent_session_id: (depth > 0 && header_parent.is_none())
                .then(|| {
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_id.clone())
                })
                .flatten(),
            parent_session_path: if depth > 0 {
                header_parent.or_else(|| {
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_file.clone())
                })
            } else {
                None
            },
            ignore_session_id: Some(session.session_id()),
        };
        self.assert_family_session_name_available(&input, Some(state), true)
            .await
    }

    /// `assertFamilySessionNameAvailable(input, currentState?, ignorePendingReservation = false)`.
    async fn assert_family_session_name_available(
        self: &Arc<Self>,
        input: &AgentSessionNameAvailabilityInput,
        current_state: Option<&Arc<StdMutex<ActiveSessionState>>>,
        ignore_pending_reservation: bool,
    ) -> Result<(), String> {
        let scope = AgentSessionNameScope {
            parent_session_id: input.parent_session_id.clone(),
            parent_session_path: input.parent_session_path.clone(),
            depth: input.depth,
        };
        if !ignore_pending_reservation
            && self
                .pending_session_names
                .lock()
                .expect("pending session names poisoned")
                .contains(&session_name_reservation_key(&scope, &input.name))
        {
            return Err(format_agent_session_name_unavailable(
                &input.name,
                input.depth,
            ));
        }
        let catalog = self
            .create_agent_family_catalog(current_state.cloned())
            .await;
        let mut input = input.clone();
        if let Some(parent_session_path) = &input.parent_session_path {
            input.parent_session_path = Some(canonical_session_path(parent_session_path));
        }
        assert_agent_session_name_available(&catalog, &input).map_err(|error| error.to_string())
    }

    /// `resolveHeaderParentSessionPath(state)`.
    fn resolve_header_parent_session_path(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Option<String> {
        let session = self.session_of(state);
        let header = session
            .session_manager()
            .lock()
            .expect("session manager poisoned")
            .get_header();
        let header_parent = header
            .as_ref()
            .and_then(|header| header.get("parentSession"))
            .and_then(Value::as_str)
            .map(str::to_string)?;
        if Path::new(&header_parent).is_absolute() {
            return Some(header_parent);
        }
        session
            .session_file()
            .map(|session_file| join_path(&dirname(&session_file), &header_parent))
    }

    /// `getAgentMessageSafetyStatus()`.
    fn get_agent_message_safety_status(&self) -> Value {
        serde_json::json!({
            "paused": self.agent_messages_paused.load(Ordering::SeqCst),
            "maxMessageChars": DEFAULT_AGENT_MESSAGE_MAX_CHARS as f64,
            "maxPendingPerSession": DEFAULT_AGENT_MESSAGE_MAX_PENDING_PER_SESSION as f64,
            "rateLimitCapacity": DEFAULT_AGENT_MESSAGE_RATE_LIMIT_CAPACITY as f64,
            "rateLimitRefillMs": DEFAULT_AGENT_MESSAGE_RATE_LIMIT_REFILL_MS as f64,
        })
    }

    /// `createCliAgentMessageSenderKey()`.
    fn create_cli_agent_message_sender_key(&self) -> String {
        format!("cli:{}", self.socket_path)
    }

    /// `clearQueuedAgentSessionMessagesForState(state)`.
    fn clear_queued_agent_session_messages_for_state(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Value {
        self.session_of(state).clear_queued_agent_messages()
    }

    /// `clearQueuedAgentSessionMessagesForAllStates()`.
    async fn clear_queued_agent_session_messages_for_all_states(self: &Arc<Self>) {
        let states = self.state_refs();
        let mut tasks = Vec::new();
        for state in states {
            tasks.push(self.clear_queued_agent_session_messages_for_state(&state));
        }
        let _ = tasks;
    }

    /// `listTargetableSessionStates(current)`.
    fn list_targetable_session_states(
        &self,
        current: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Vec<Arc<StdMutex<ActiveSessionState>>> {
        let current_active_session_id = current
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.state_refs()
            .into_iter()
            .filter(|state| {
                let active_session_id = state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                if active_session_id == current_active_session_id {
                    return true;
                }
                !self
                    .binding_sessions
                    .lock()
                    .expect("binding sessions poisoned")
                    .contains(&active_session_id)
                    && !self
                        .closing_sessions
                        .lock()
                        .expect("closing sessions poisoned")
                        .contains_key(&active_session_id)
            })
            .collect()
    }
}

/// The daemon-side view of `sendAgentSessionMessage(options)`.
#[derive(Clone)]
pub struct SendAgentMessageInput {
    pub target_selector: String,
    pub message: String,
    pub from_state: Option<Arc<StdMutex<ActiveSessionState>>>,
    pub sender: Option<AgentSessionMessageSender>,
    pub client_id: Option<String>,
    pub sender_key: Option<String>,
    pub origin: String,
}

/// `AgentSessionNameAvailabilityInput` as `withSessionNameReservation` takes it.
#[derive(Debug, Clone, Default)]
pub struct NameReservationInput {
    pub name: String,
    pub depth: f64,
    pub parent_session_id: Option<String>,
    pub parent_session_path: Option<String>,
}

impl AgentDaemon {
    /// `createAgentSessionMessageEndpoint(state)`.
    fn create_agent_session_message_endpoint(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> AgentSessionMessageEndpoint {
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let session = self.session_of(state);
        AgentSessionMessageEndpoint {
            active_session_id: state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                .clone(),
            session_id: session.session_id(),
            session_name: session.session_name(),
            runtime_kind: metadata.as_ref().and_then(|metadata| metadata.kind.clone()),
        }
    }

    /// `createAgentSessionMessageSender(state, clientId)`.
    fn create_agent_session_message_sender(
        &self,
        state: Option<&Arc<StdMutex<ActiveSessionState>>>,
        client_id: &str,
    ) -> AgentSessionMessageSender {
        match state {
            None => AgentSessionMessageSender {
                client_id: Some(client_id.to_string()),
                ..AgentSessionMessageSender::default()
            },
            Some(state) => {
                let endpoint = self.create_agent_session_message_endpoint(state);
                AgentSessionMessageSender {
                    active_session_id: Some(endpoint.active_session_id),
                    session_id: Some(endpoint.session_id),
                    session_name: endpoint.session_name,
                    runtime_kind: endpoint.runtime_kind,
                    client_id: Some(client_id.to_string()),
                }
            }
        }
    }

    /// `createAgentMessageAgentSummary(state)`.
    fn create_agent_message_agent_summary(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> AgentSessionMessageAgentSummary {
        let entry = self.session_entry_for_state(state);
        let endpoint = self.create_agent_session_message_endpoint(state);
        let session = &entry.session;
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let has_running_rlm_children = session.has_running_rlm_children();
        let summary = SessionSummary {
            active_session_id: Some(endpoint.active_session_id.clone()),
            runtime_kind: metadata.as_ref().and_then(|metadata| metadata.kind.clone()),
            activity: if session.is_foreground_active() {
                "working".to_string()
            } else {
                "idle".to_string()
            },
            is_session_active: session.is_session_active(),
            has_running_rlm_children: Some(has_running_rlm_children),
            is_streaming: session.is_streaming(),
            ..SessionSummary::default()
        };
        AgentSessionMessageAgentSummary {
            active_session_id: endpoint.active_session_id,
            session_id: endpoint.session_id,
            session_name: endpoint.session_name,
            runtime_kind: endpoint.runtime_kind,
            cwd: entry.cwd(),
            is_streaming: session.is_streaming(),
            unfinished_action_count: session.unfinished_action_count(),
            parent_active_session_id: metadata
                .as_ref()
                .and_then(|metadata| metadata.parent_active_session_id.clone()),
            rlm_child_id: metadata
                .as_ref()
                .and_then(|metadata| metadata.rlm_child_id.clone()),
            // REPAIR CURSOR: `active_session_state::AgentSessionRuntimeMetadata` has no
            // `session_dir` (see the metadata cursor in `session_snapshot_entry`). The
            // canonical type (core/agent_session_runtime.rs:104) carries it. Fix: swap the
            // import and read `metadata.session_dir` here.
            session_dir: None,
            session_path: session.session_file(),
            parent_session_id: metadata
                .as_ref()
                .and_then(|metadata| metadata.parent_session_id.clone()),
            parent_session_path: metadata
                .as_ref()
                .and_then(|metadata| metadata.parent_session_file.clone()),
            rlm_depth: session.rlm_depth().map(|depth| depth as f64),
            status: Some(
                classify_session_roster_status(
                    &RosterSummaryView {
                        active_session_id: summary.active_session_id.clone(),
                        activity: Some(summary.activity.clone()),
                        is_session_active: Some(summary.is_session_active),
                    },
                    false,
                )
                .as_str()
                .to_string(),
            ),
            rlm_child_registry_status: None,
        }
    }

    /// `createAgentMessageListResult(current, peers?)`.
    async fn create_agent_message_list_result(
        self: &Arc<Self>,
        current: &Arc<StdMutex<ActiveSessionState>>,
        peers: Option<Vec<AgentSessionMessageAgentSummary>>,
    ) -> AgentSessionMessageListResult {
        let peers = match peers {
            Some(peers) => peers,
            None => self.list_supervisor_agent_peers().await,
        };
        let mut local_agents: Vec<AgentSessionMessageAgentSummary> = self
            .list_targetable_session_states(current)
            .iter()
            .map(|state| self.create_agent_message_agent_summary(state))
            .collect();
        for passive in self.list_passive_rlm_subagents(Vec::new(), false).await {
            let entry = &passive.entry;
            let info = &passive.info;
            let root_parent_active_session_id = match &passive.root {
                PassiveRlmRoot::Resident(state) => Some(
                    state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                ),
                PassiveRlmRoot::Saved(_) => None,
            };
            let root_parent_session_file = match &passive.root {
                PassiveRlmRoot::Resident(state) => self.session_of(state).session_file(),
                PassiveRlmRoot::Saved(info) => Some(info.path.clone()),
            };
            let parent_entry = passive
                .chain
                .len()
                .checked_sub(2)
                .and_then(|index| passive.chain.get(index));
            local_agents.push(AgentSessionMessageAgentSummary {
                // Before hydration the persisted session id is its
                // supervisor-routable id.
                active_session_id: info.id.clone(),
                session_id: info.id.clone(),
                session_name: info
                    .name
                    .clone()
                    .or_else(|| Some(entry.session_name.clone())),
                runtime_kind: Some(RUNTIME_KIND_SUBAGENT.to_string()),
                cwd: info.cwd.clone(),
                is_streaming: false,
                unfinished_action_count: 0.0,
                parent_active_session_id: (passive.chain.len() == 1)
                    .then_some(root_parent_active_session_id)
                    .flatten(),
                rlm_child_id: Some(entry.child_id.clone()),
                session_dir: Some(entry.session_dir.clone()),
                session_path: Some(entry.session_file.clone()),
                parent_session_id: Some(entry.parent_session_id.clone()),
                parent_session_path: entry
                    .parent_session_file
                    .clone()
                    .or_else(|| parent_entry.map(|entry| entry.session_file.clone()))
                    .or(root_parent_session_file),
                rlm_depth: Some(entry.rlm_depth.unwrap_or(info.rlm_depth) as f64),
                status: Some(FAMILY_STATUS_INACTIVE.to_string()),
                rlm_child_registry_status: Some(entry.status.clone()),
            });
        }
        let local_ids: HashSet<String> = local_agents
            .iter()
            .map(|agent| agent.active_session_id.clone())
            .collect();
        let remote_agents: Vec<AgentSessionMessageAgentSummary> = peers
            .into_iter()
            .filter(|peer| {
                !local_ids.contains(&peer.active_session_id)
                    && !self
                        .closing_sessions
                        .lock()
                        .expect("closing sessions poisoned")
                        .contains_key(&peer.active_session_id)
            })
            .collect();
        local_agents.extend(remote_agents);
        AgentSessionMessageListResult {
            current: Some(self.create_agent_session_message_endpoint(current)),
            agents: local_agents,
        }
    }



    /// `createAgentFamilyCatalog(currentState?)`.
    async fn create_agent_family_catalog(
        self: &Arc<Self>,
        current_state: Option<Arc<StdMutex<ActiveSessionState>>>,
    ) -> Vec<AgentFamilyCatalogEntry> {
        let current = match current_state {
            Some(state) => Some(state),
            None => self.session_states().into_iter().find_map(|entry| {
                let active_session_id = entry
                    .state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                (!self
                    .binding_sessions
                    .lock()
                    .expect("binding sessions poisoned")
                    .contains(&active_session_id))
                .then_some(Arc::clone(&entry.state))
            }),
        };
        let remote_peers = match &current {
            Some(_) => self.list_supervisor_agent_peers().await,
            None => Vec::new(),
        };
        let listed = match &current {
            Some(current) => {
                self.create_agent_message_list_result(current, Some(remote_peers.clone()))
                    .await
            }
            None => AgentSessionMessageListResult::default(),
        };
        let remote_peer_keys: HashSet<String> = remote_peers
            .iter()
            .map(|peer| peer.active_session_id.clone())
            .collect();
        let local_agents: Vec<AgentSessionMessageAgentSummary> = match &current {
            Some(current) => {
                let mut agents = vec![self.create_agent_message_agent_summary(current)];
                agents.extend(
                    listed
                        .agents
                        .into_iter()
                        .filter(|agent| !remote_peer_keys.contains(&agent.active_session_id)),
                );
                agents
            }
            None => listed.agents,
        };
        let active_paths: HashSet<String> = local_agents
            .iter()
            .filter_map(|agent| agent.session_path.clone())
            .map(|path| canonical_session_path(&path))
            .collect();
        let session_dir = self.options.default_session_config.session_dir.clone();
        let saved_roots: Vec<AgentFamilyCatalogEntry> =
            SessionManager::list_all(None, session_dir.as_deref())
                .await
            .into_iter()
            .filter(|info| {
                // TS: (info.rlmDepth ?? (info.parentSessionPath ? -1 : 0)) === 0
                // (daemon-mode.ts:5909-5914). rlmDepth is always defined at this
                // point, so the ?? fallback is dead code and a depth-0 fork with
                // a recorded parentSessionPath stays a root.
                saved_roots_classifier(
                    info.rlm_depth,
                    &active_paths,
                    &canonical_session_path(&info.path),
                )
            })
            .map(|info| AgentFamilyCatalogEntry {
                id: info.id.clone(),
                name: info.name.clone(),
                depth: info.rlm_depth as f64,
                status: FAMILY_STATUS_INACTIVE.to_string(),
                session_path: Some(canonical_session_path(&info.path)),
                ..AgentFamilyCatalogEntry::default()
            })
            .collect();
        let mut by_id: indexmap::IndexMap<String, AgentFamilyCatalogEntry> = saved_roots
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect();
        let mut add_agent = |agent: &AgentSessionMessageAgentSummary| {
            let depth = agent.rlm_depth.unwrap_or(0.0);
            let mut entry = AgentFamilyCatalogEntry {
                id: agent.session_id.clone(),
                name: agent.session_name.clone(),
                depth,
                status: agent
                    .status
                    .clone()
                    .unwrap_or_else(|| FAMILY_STATUS_IDLE.to_string()),
                session_path: agent.session_path.as_deref().map(canonical_session_path),
                ..AgentFamilyCatalogEntry::default()
            };
            if agent.runtime_kind.as_deref() == Some(RUNTIME_KIND_SUBAGENT) {
                if let Some(session_path) = &agent.session_path {
                    if let Some(state) = self.find_session_by_session_file(Some(session_path)) {
                        entry.replied_since_task =
                            self.session_of(&state).replied_to_parent_since_task();
                    }
                }
            }
            if depth > 0.0 && agent.parent_session_id.is_some() {
                entry.parent_session_id = agent.parent_session_id.clone();
            }
            if depth > 0.0 && agent.parent_session_path.is_some() {
                entry.parent_session_path = agent
                    .parent_session_path
                    .as_deref()
                    .map(canonical_session_path);
            }
            by_id.insert(agent.session_id.clone(), entry);
        };
        for peer in &remote_peers {
            add_agent(peer);
        }
        for agent in &local_agents {
            add_agent(agent);
        }
        for entry in self.session_states() {
            let session_id = entry.session.session_id();
            let session_file = entry.session.session_file();
            if let (Some(catalog_entry), Some(session_file)) =
                (by_id.get_mut(&session_id), session_file)
            {
                catalog_entry.session_path = Some(canonical_session_path(&session_file));
            }
        }
        for passive in self.list_passive_rlm_subagents(Vec::new(), false).await {
            if let Some(entry) = by_id.get_mut(&passive.info.id) {
                entry.session_path = Some(canonical_session_path(&passive.entry.session_file));
            }
        }
        by_id.into_values().collect()
    }

    /// `createAgentFamilyRoster(currentState)`.
    async fn create_agent_family_roster(
        self: &Arc<Self>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Result<AgentFamilyRosterResult, String> {
        let catalog = self
            .create_agent_family_catalog(Some(Arc::clone(current_state)))
            .await;
        let current_session_id = self.session_of(current_state).session_id();
        let current = catalog
            .iter()
            .find(|entry| entry.id == current_session_id)
            .cloned()
            .ok_or_else(|| "Current agent is missing from the family catalog".to_string())?;
        Ok(build_agent_family_roster(&current, &catalog))
    }

    /// `listSupervisorAgentPeers()`.
    async fn list_supervisor_agent_peers(&self) -> Vec<AgentSessionMessageAgentSummary> {
        let Some(worker) = self.options.worker.as_ref() else {
            return Vec::new();
        };
        let Some(socket_path) = self.supervisor_socket_path_from_env() else {
            return Vec::new();
        };
        agent_message_transport::list_peers(&socket_path, &worker.authentication_token).await
    }
}

impl AgentDaemon {
    /// `catchUpBackpressuredClient(client)`.
    async fn catch_up_backpressured_client(
        self: &Arc<Self>,
        client: Arc<DaemonClientHandle>,
    ) -> Result<(), String> {
        if client.snapshot_streaming() || client.is_backpressured() {
            return Ok(());
        }
        if client.catchup_running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() { return Ok(()); }
        self.clear_client_catchup_retry(&client);
        let result = self.drain_backpressured_client_catchup_queue(&client).await;
        client.catchup_running.store(false, Ordering::SeqCst);
        result
    }

    /// `clearClientCatchupRetry(client)`.
    fn clear_client_catchup_retry(&self, client: &Arc<DaemonClientHandle>) {
        if !client.catchup_retry_timer.load(Ordering::SeqCst) {
            return;
        }
        client.set_catchup_retry_timer(false);
    }

    /// `scheduleClientCatchupRetry(client)`.
    fn schedule_client_catchup_retry(self: &Arc<Self>, client: Arc<DaemonClientHandle>) {
        if client.writer.destroyed() || client.catchup_retry_timer.load(Ordering::SeqCst) {
            return;
        }
        client.set_catchup_retry_timer(true);
        let daemon = Arc::clone(self);
        tokio::spawn(async move {
            delay(CLIENT_CATCHUP_RETRY_MS).await;
            client.set_catchup_retry_timer(false);
            if client.writer.destroyed()
                || client
                    .state
                    .lock()
                    .expect("daemon client poisoned")
                    .catchup_active_session_ids
                    .as_ref()
                    .map(HashSet::is_empty)
                    .unwrap_or(true)
            {
                return;
            }
            if client.snapshot_streaming() || client.is_backpressured() {
                daemon.schedule_client_catchup_retry(client);
                return;
            }
            if let Err(error) = daemon
                .catch_up_backpressured_client(Arc::clone(&client))
                .await
            {
                daemon.log(&format!(
                    "could not retry catch-up for client {}: {error}",
                    client.id()
                ));
            }
        });
    }

    /// `drainBackpressuredClientCatchupQueue(client)`.
    async fn drain_backpressured_client_catchup_queue(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
    ) -> Result<(), String> {
        while !client.writer.destroyed()
            && !client.snapshot_streaming()
            && !client.is_backpressured()
            && client
                .state
                .lock()
                .expect("daemon client poisoned")
                .catchup_active_session_ids
                .as_ref()
                .map(|ids| !ids.is_empty())
                .unwrap_or(false)
        {
            if self.drain_backpressured_client_catchups(client).await == "retry-later" {
                return Ok(());
            }
        }
        Ok(())
    }

    /// `drainBackpressuredClientCatchups(client)`.
    async fn drain_backpressured_client_catchups(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
    ) -> String {
        if client.writer.destroyed() {
            return "drained".to_string();
        }
        let pending: Vec<(String, String)> = {
            let mut state = client.state.lock().expect("daemon client poisoned");
            let ids = state.catchup_active_session_ids.take().unwrap_or_default();
            let purposes = state.catchup_purposes.take().unwrap_or_default();
            ids.into_iter()
                .map(|active_session_id| {
                    let purpose = purposes
                        .get(&active_session_id)
                        .cloned()
                        .unwrap_or_else(|| "resync".to_string());
                    (active_session_id, purpose)
                })
                .collect()
        };
        for (index, (active_session_id, purpose)) in pending.iter().enumerate() {
            let state = self
                .sessions
                .lock()
                .expect("sessions poisoned")
                .get(active_session_id)
                .map(|entry| Arc::clone(&entry.state));
            let Some(state) = state else {
                continue;
            };
            let attached = state
                .lock()
                .expect("active session poisoned")
                .clients
                .iter()
                .any(|candidate| Arc::ptr_eq(candidate, &client.state));
            if !attached {
                continue;
            }
            let attach_command = ParsedDaemonCommand {
                id: None,
                type_: "attach".to_string(),
                body: serde_json::json!({ "activeSessionId": active_session_id }),
            };
            let private_framed = { client.state.lock().expect("daemon client poisoned").transport.as_deref() == Some("private-framed") };
            let chunked = private_framed && client.capabilities_for_session(active_session_id).contains("chunked_snapshot");
            let snapshot_signal = chunked.then(|| mark_client_snapshot_streaming(client, active_session_id));
            let result = match self
                .create_attach_result(client, &state, &attach_command)
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    if chunked { finish_client_snapshot_streaming(client, active_session_id); }
                    for (remaining, remaining_purpose) in &pending[index..] {
                        self.queue_client_catchup(client, remaining, remaining_purpose);
                    }
                    self.log(&format!(
                        "could not catch up client {} for {active_session_id}: {error}",
                        client.id()
                    ));
                    self.schedule_client_catchup_retry(Arc::clone(client));
                    return "retry-later".to_string();
                }
            };
            let resident = self
                .sessions
                .lock()
                .expect("sessions poisoned")
                .get(active_session_id)
                .map(|entry| Arc::ptr_eq(&entry.state, &state))
                .unwrap_or(false);
            let still_attached = state
                .lock()
                .expect("active session poisoned")
                .clients
                .iter()
                .any(|candidate| Arc::ptr_eq(candidate, &client.state));
            if !resident || !still_attached {
                if chunked { finish_client_snapshot_streaming(client, active_session_id); }
                continue;
            }
            let last_event_sequence = result.last_event_sequence;
            let event_generation = result.snapshot.get("lastEventCursor").and_then(|cursor| cursor.get("generation")).and_then(Value::as_str).unwrap_or("").to_string();
            if chunked {
                if purpose == "replacement" {
                    self.write(
                        client,
                        &DaemonOutbound::Raw(serde_json::json!({
                            "type": "session_replaced",
                            "activeSessionId": active_session_id,
                            "state": result.snapshot.get("state").cloned().unwrap_or(Value::Null),
                            "messages": [],
                            "snapshotFollows": true,
                            "meta": serde_json::to_value(create_daemon_event_meta(
                                active_session_id,
                                last_event_sequence,
                                now_iso(),
                                &event_generation,
                            ))
                            .unwrap_or(Value::Null),
                        })),
                    );
                }
                let snapshot_id = snapshot_transfer_id(&result.snapshot);
                let signal = snapshot_signal.expect("chunked snapshot reserved before capture");
                let snapshot_messages = result
                    .snapshot
                    .get("messages")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let messages: Vec<AgentMessage> = snapshot_messages
                    .iter()
                    .filter_map(|value| serde_json::from_value(value.clone()).ok())
                    .collect();
                let transcript =
                    create_snapshot_transcript_chunks(CreateSnapshotTranscriptChunksOptions {
                        active_session_id: active_session_id.clone(),
                        snapshot_id: snapshot_id.clone(),
                        messages,
                        target_chunk_bytes: Some(SNAPSHOT_TARGET_CHUNK_BYTES),
                        aborted: signal.is_cancelled(),
                    });
                let mut snapshot = result.snapshot.clone();
                if let Some(object) = snapshot.as_object_mut() {
                    object.insert("messages".to_string(), Value::Array(Vec::new()));
                }
                let message_count = snapshot_messages.len();
                if let Err(error) = self
                    .stream_worker_snapshot(
                        client,
                        &state,
                        &snapshot_id,
                        snapshot,
                        message_count,
                        transcript,
                        // `purpose === "replacement" ? "replacement" : "catchup"`
                        // (`daemon-mode.ts:7523`).
                        if purpose == "replacement" { "replacement" } else { "catchup" },
                        signal,
                        true,
                    )
                    .await
                {
                    self.log(&format!("could not stream catch-up snapshot: {error}"));
                }
                continue;
            }
            let meta = serde_json::to_value(create_daemon_event_meta(
                active_session_id,
                last_event_sequence,
                now_iso(),
                &event_generation,
            ))
            .unwrap_or(Value::Null);
            let catchup = if purpose == "replacement" {
                serde_json::json!({
                    "type": "session_replaced",
                    "activeSessionId": active_session_id,
                    "state": result.snapshot.get("state").cloned().unwrap_or(Value::Null),
                    "messages": result.snapshot.get("messages").cloned().unwrap_or(Value::Array(Vec::new())),
                    "meta": meta,
                })
            } else {
                serde_json::json!({
                    "type": "session_resynced",
                    "activeSessionId": active_session_id,
                    "snapshot": result.snapshot,
                    "meta": meta,
                })
            };
            if !self.write(client, &DaemonOutbound::Raw(catchup)) {
                for (remaining, remaining_purpose) in &pending[index + 1..] {
                    self.queue_client_catchup(client, remaining, remaining_purpose);
                }
                return "retry-later".to_string();
            }
            // The delivered catch-up frame (`session_replaced` or
            // `session_resynced`) resets the client's status surface, so the
            // authoritative Jev footer for THIS session must follow the
            // frame, exactly like the healthy broadcast path: backpressured
            // and deferred clients must not keep a blank or stale footer
            // until the next /jev write or re-attach. Both purposes reset the
            // surface, and the push re-reads one settings snapshot keyed by
            // the delivered session, so a resync can never leave the old
            // session's labels behind either.
            self.publish_jev_attach_footer(client, &state);
        }
        "drained".to_string()
    }

    /// `queueClientCatchup(client, activeSessionId, purpose = "resync")`.
    fn defer_snapshot_frame(&self, client: &Arc<DaemonClientHandle>, active_session_id: &str, message: &DaemonOutbound) -> bool {
        let mut deferred = client.deferred_snapshot_frames.lock().expect("deferred frames poisoned");
        let streaming = client.state.lock().expect("daemon client poisoned").snapshot_active_session_ids.as_ref().is_some_and(|ids| ids.contains(active_session_id));
        if !streaming { return false; }
        let (bytes, frames) = deferred.entry(active_session_id.to_string()).or_default();
        // A slow viewer must not retain an unbounded stream. Overflow requests
        // one authoritative catch-up; ordinary streams replay incrementally.
        let incoming = serde_json::to_vec(&message.to_value()).map(|bytes| bytes.len()).unwrap_or(0);
        if message.type_name() == "session_replaced" || frames.len() >= 512 || bytes.saturating_add(incoming) > 4 * 1024 * 1024 {
            frames.clear();
            *bytes = 0;
            self.queue_client_catchup(client, active_session_id, if message.type_name() == "session_replaced" { "replacement" } else { "resync" });
        } else {
            *bytes += incoming;
            frames.push(message.clone());
        }
        true
    }

    fn finish_snapshot_and_replay(self: &Arc<Self>, client: &Arc<DaemonClientHandle>, active_session_id: &str, sequence: i64, generation: &str, aborted: bool) {
        // Serialize the final replay with broadcasters. They either queue before
        // this drain or write after streaming has been cleared, never overtake it.
        let mut deferred = client.deferred_snapshot_frames.lock().expect("deferred frames poisoned");
        let (_, frames) = deferred.remove(active_session_id).unwrap_or_default();
        if !aborted {
            for frame in frames {
                let value = frame.to_value();
                let meta = value.get("meta");
                let frame_sequence = meta.and_then(|meta| meta.get("sequence")).and_then(Value::as_i64);
                let frame_generation = meta.and_then(|meta| meta.get("cursor")).and_then(|cursor| cursor.get("generation")).and_then(Value::as_str);
                if frame_generation == Some(generation) && frame_sequence.is_some_and(|value| value <= sequence) { continue; }
                if !self.write(client, &frame) { break; }
            }
        }
        finish_client_snapshot_streaming(client, active_session_id);
    }

    fn queue_client_catchup(
        &self,
        client: &Arc<DaemonClientHandle>,
        active_session_id: &str,
        purpose: &str,
    ) {
        let mut state = client.state.lock().expect("daemon client poisoned");
        state
            .catchup_active_session_ids
            .get_or_insert_with(HashSet::new)
            .insert(active_session_id.to_string());
        let purposes = state.catchup_purposes.get_or_insert_with(HashMap::new);
        if purpose == "replacement" || !purposes.contains_key(active_session_id) {
            purposes.insert(active_session_id.to_string(), purpose.to_string());
        }
    }

    /// `beginReplacementSnapshot(client, state, message)`.
    fn begin_replacement_snapshot(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        message: DaemonOutbound,
    ) {
        // Mark before the registry read so later events queue behind this snapshot.
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut client_mut = Arc::clone(client);
        let snapshot_signal = mark_client_snapshot_streaming(&mut client_mut, &active_session_id);
        let daemon = Arc::clone(self);
        let client = Arc::clone(client);
        let state = Arc::clone(state);
        tokio::spawn(async move {
            match daemon
                .prepare_replacement_snapshot(&client, &state, message.clone(), snapshot_signal)
                .await
            {
                Ok(()) => {}
                Err(error) => {
                    let mut client_mut = Arc::clone(&client);
                    finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
                    daemon.log(&format!("could not prepare replacement snapshot: {error}"));
                    let resident = daemon
                        .sessions
                        .lock()
                        .expect("sessions poisoned")
                        .get(&active_session_id)
                        .map(|entry| Arc::ptr_eq(&entry.state, &state))
                        .unwrap_or(false);
                    if !client.writer.destroyed() && resident {
                        if daemon.write(&client, &message) {
                            // The raw fallback delivered the replacement
                            // frame, so the authoritative footer must follow
                            // it here too; a failed write re-converges
                            // through the catch-up path, which pushes after
                            // its own delivery.
                            daemon.publish_jev_attach_footer(&client, &state);
                        }
                    }
                    let has_catchup = client
                        .state
                        .lock()
                        .expect("daemon client poisoned")
                        .catchup_active_session_ids
                        .as_ref()
                        .map(|ids| !ids.is_empty())
                        .unwrap_or(false);
                    if !client.snapshot_streaming() && has_catchup {
                        if let Err(catchup_error) = daemon
                            .catch_up_backpressured_client(Arc::clone(&client))
                            .await
                        {
                            daemon.log(&format!(
                                "could not catch up replacement snapshot: {catchup_error}"
                            ));
                        }
                    }
                }
            }
        });
    }

    /// `prepareReplacementSnapshot(client, state, message, snapshotSignal)`.
    async fn prepare_replacement_snapshot(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        message: DaemonOutbound,
        snapshot_signal: tokio_util::sync::CancellationToken,
    ) -> Result<(), String> {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let command = ParsedDaemonCommand {
            id: None,
            type_: "attach".to_string(),
            body: serde_json::json!({ "activeSessionId": active_session_id }),
        };
        let result = self.create_attach_result(client, state, &command).await?;
        let snapshot_id = snapshot_transfer_id(&result.snapshot);
        let resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .map(|entry| Arc::ptr_eq(&entry.state, state))
            .unwrap_or(false);
        if !resident {
            let mut client_mut = Arc::clone(client);
            finish_client_snapshot_streaming(&mut client_mut, &active_session_id);
            let has_catchup = client
                .state
                .lock()
                .expect("daemon client poisoned")
                .catchup_active_session_ids
                .as_ref()
                .map(|ids| !ids.is_empty())
                .unwrap_or(false);
            if !client.snapshot_streaming() && has_catchup {
                if let Err(error) = self.catch_up_backpressured_client(Arc::clone(client)).await {
                    self.log(&format!("could not catch up replacement snapshot: {error}"));
                }
            }
            return Ok(());
        }
        let snapshot_messages = result
            .snapshot
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let messages: Vec<AgentMessage> = snapshot_messages
            .iter()
            .filter_map(|value| serde_json::from_value(value.clone()).ok())
            .collect();
        let transcript = create_snapshot_transcript_chunks(CreateSnapshotTranscriptChunksOptions {
            active_session_id: active_session_id.clone(),
            snapshot_id: snapshot_id.clone(),
            messages,
            target_chunk_bytes: Some(SNAPSHOT_TARGET_CHUNK_BYTES),
            aborted: snapshot_signal.is_cancelled(),
        });
        let mut with_empty_messages = message.clone();
        if let DaemonOutbound::Raw(value) = &mut with_empty_messages {
            if let Some(object) = value.as_object_mut() {
                object.insert("messages".to_string(), Value::Array(Vec::new()));
                object.insert("snapshotFollows".to_string(), Value::Bool(true));
            }
        }
        self.write(client, &with_empty_messages);
        let mut snapshot = result.snapshot.clone();
        if let Some(object) = snapshot.as_object_mut() {
            object.insert("messages".to_string(), Value::Array(Vec::new()));
        }
        let message_count = snapshot_messages.len();
        let daemon = Arc::clone(self);
        let client = Arc::clone(client);
        let state = Arc::clone(state);
        tokio::spawn(async move {
            if let Err(error) = daemon
                .stream_worker_snapshot(
                    &client,
                    &state,
                    &snapshot_id,
                    snapshot,
                    message_count,
                    transcript,
                    // `daemon-mode.ts:7124` passes the literal `"replacement"`.
                    "replacement",
                    snapshot_signal,
                    true,
                )
                .await
            {
                daemon.log(&format!("could not stream replacement snapshot: {error}"));
                daemon.queue_client_catchup(&client, &active_session_id, "replacement");
                if !client.snapshot_streaming() {
                    if let Err(catchup_error) = daemon
                        .catch_up_backpressured_client(Arc::clone(&client))
                        .await
                    {
                        daemon.log(&format!(
                            "could not catch up replacement snapshot: {catchup_error}"
                        ));
                    }
                }
            }
        });
        Ok(())
    }
}

impl AgentDaemon {
    /// `hydratePassiveRlmSubagent(passive, clientEnv?)`.
    async fn hydrate_passive_rlm_subagent(
        self: &Arc<Self>,
        passive: PassiveRlmSubagent,
        client_env: Option<HashMap<String, String>>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        if self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .is_some()
        {
            return Err(
                BoundSessionUnavailableError::new("Daemon is preparing an update restart")
                    .to_string(),
            );
        }
        let root_parent = match &passive.root {
            PassiveRlmRoot::Resident(state) => Arc::clone(state),
            PassiveRlmRoot::Saved(_) => {
                return Err(format!(
                    "Cannot hydrate RLM subagent {} without a resident root parent",
                    passive.entry.child_id
                ))
            }
        };
        if let Some(root_parent_file) = self.session_of(&root_parent).session_file() {
            self.wait_for_passivation(&root_parent_file).await;
        }
        if !self.is_resident(&root_parent) {
            return self
                .restart_after_parent_change(passive, client_env, root_parent, None)
                .await;
        }
        let mut parent_state = root_parent;
        for entry in passive.chain.clone() {
            self.wait_for_passivation(&entry.session_file).await;
            if !self.is_resident(&parent_state) {
                return self
                    .restart_after_parent_change(
                        passive.clone(),
                        client_env.clone(),
                        parent_state,
                        None,
                    )
                    .await;
            }
            let hydrating_parent = Arc::clone(&parent_state);
            let restore_active_session_id =
                (entry.child_id == passive.entry.child_id).then(|| passive.info.id.clone());
            let hydrated = match self
                .rehydrate_completed_rlm_subagent(
                    Arc::clone(&hydrating_parent),
                    entry.clone(),
                    restore_active_session_id,
                    client_env.clone(),
                )
                .await
            {
                Ok(hydrated) => hydrated,
                Err(error) => {
                    if error.starts_with("__bound_session_unavailable__")
                        && self
                            .find_passivation_by_session_file(&entry.session_file)
                            .is_some()
                    {
                        self.wait_for_passivation(&entry.session_file).await;
                        return self
                            .restart_after_parent_change(
                                passive.clone(),
                                client_env.clone(),
                                hydrating_parent,
                                None,
                            )
                            .await;
                    }
                    if self.is_resident(&hydrating_parent) {
                        return Err(error);
                    }
                    return self
                        .restart_after_parent_change(
                            passive.clone(),
                            client_env.clone(),
                            hydrating_parent,
                            None,
                        )
                        .await;
                }
            };
            if !self.is_resident(&hydrating_parent) {
                return self
                    .restart_after_parent_change(
                        passive.clone(),
                        client_env.clone(),
                        hydrating_parent,
                        None,
                    )
                    .await;
            }
            if !self.is_resident(&hydrated) {
                return self
                    .restart_after_parent_change(
                        passive.clone(),
                        client_env.clone(),
                        hydrated,
                        None,
                    )
                    .await;
            }
            parent_state = hydrated;
        }
        Ok(parent_state)
    }

    /// `isResident(state)` inside `hydratePassiveRlmSubagent`.
    fn is_resident(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> bool {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .map(|entry| Arc::ptr_eq(&entry.state, state))
            .unwrap_or(false);
        resident
            && !self
                .closing_sessions
                .lock()
                .expect("closing sessions poisoned")
                .contains_key(&active_session_id)
    }

    /// `restartAfterParentChange(staleParent)`.
    async fn restart_after_parent_change(
        self: &Arc<Self>,
        passive: PassiveRlmSubagent,
        client_env: Option<HashMap<String, String>>,
        stale_parent: Arc<StdMutex<ActiveSessionState>>,
        _selector: Option<String>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        if let Some(stale_parent_file) = self.session_of(&stale_parent).session_file() {
            self.wait_for_passivation(&stale_parent_file).await;
        }
        let refreshed = self
            .find_passive_rlm_subagent(&passive.entry.session_file, true)
            .await
            .ok_or_else(|| RuntimeOpenCancelledError.to_string())?;
        if let Some(resident) = self.find_session_by_session_file(Some(&passive.entry.session_file))
        {
            let metadata = resident
                .lock()
                .expect("active session poisoned")
                .runtime
                .metadata
                .clone();
            if metadata
                .as_ref()
                .and_then(|metadata| metadata.kind.as_deref())
                == Some("subagent")
                && metadata
                    .as_ref()
                    .and_then(|metadata| metadata.rlm_child_id.as_deref())
                    == Some(passive.entry.child_id.as_str())
            {
                return self.wait_for_bound_session(resident).await;
            }
        }
        Box::pin(self.hydrate_passive_rlm_subagent(refreshed, client_env)).await
    }

    /// `rehydrateCompletedRlmSubagent(parentState, entry, restoreActiveSessionId?, clientEnv?)`.
    async fn rehydrate_completed_rlm_subagent(
        self: &Arc<Self>,
        parent_state: Arc<StdMutex<ActiveSessionState>>,
        entry: PassiveRlmSubagentEntry,
        restore_active_session_id: Option<String>,
        client_env: Option<HashMap<String, String>>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        if self
            .update_restart
            .lock()
            .expect("update restart poisoned")
            .is_some()
        {
            return Err(
                BoundSessionUnavailableError::new("Daemon is preparing an update restart")
                    .to_string(),
            );
        }
        let session_key = resolve_path(&entry.session_file);
        if self
            .reserving_session_opens
            .lock()
            .expect("reserving session opens poisoned")
            .contains_key(&session_key)
        {
            delay(1).await;
            return Box::pin(self.rehydrate_completed_rlm_subagent(
                parent_state,
                entry,
                restore_active_session_id,
                client_env,
            ))
            .await;
        }
        let pending = self
            .opening_sessions
            .lock()
            .expect("opening sessions poisoned")
            .get(&session_key)
            .copied();
        if pending.is_some() {
            let resident = self.find_session_by_session_file(Some(&entry.session_file));
            match resident {
                Some(state) => {
                    let metadata = state
                        .lock()
                        .expect("active session poisoned")
                        .runtime
                        .metadata
                        .clone();
                    if metadata
                        .as_ref()
                        .and_then(|metadata| metadata.kind.as_deref())
                        != Some("subagent")
                        || metadata
                            .as_ref()
                            .and_then(|metadata| metadata.rlm_child_id.as_deref())
                            != Some(entry.child_id.as_str())
                    {
                        self.opening_sessions
                            .lock()
                            .expect("opening sessions poisoned")
                            .remove(&session_key);
                        return Box::pin(self.rehydrate_completed_rlm_subagent(
                            parent_state,
                            entry,
                            restore_active_session_id,
                            client_env,
                        ))
                        .await;
                    }
                    return self.wait_for_bound_session(state).await;
                }
                None => {
                    delay(1).await;
                    return Box::pin(self.rehydrate_completed_rlm_subagent(
                        parent_state,
                        entry,
                        restore_active_session_id,
                        client_env,
                    ))
                    .await;
                }
            }
        }
        if let Some(existing) = self.find_session_by_session_file(Some(&entry.session_file)) {
            let metadata = existing
                .lock()
                .expect("active session poisoned")
                .runtime
                .metadata
                .clone();
            if metadata
                .as_ref()
                .and_then(|metadata| metadata.kind.as_deref())
                == Some("subagent")
                && metadata
                    .as_ref()
                    .and_then(|metadata| metadata.rlm_child_id.as_deref())
                    == Some(entry.child_id.as_str())
            {
                return self.wait_for_bound_session(existing).await;
            }
        }
        // Explicit opens and all lazy triggers share this path-keyed publication,
        // so no caller can acquire a second lease/runtime while hydration binds.
        let open_id = self.next_id();
        self.opening_sessions
            .lock()
            .expect("opening sessions poisoned")
            .insert(session_key.clone(), open_id);
        let result = self
            .rehydrate_completed_rlm_subagent_once(
                Arc::clone(&parent_state),
                entry,
                restore_active_session_id,
                client_env,
            )
            .await;
        {
            let mut opening = self
                .opening_sessions
                .lock()
                .expect("opening sessions poisoned");
            if opening.get(&session_key).copied() == Some(open_id) {
                opening.remove(&session_key);
            }
        }
        result
    }

    /// `rehydrateCompletedRlmSubagentOnce(parentState, entry, restoreActiveSessionId?, clientEnv?)`.
    async fn rehydrate_completed_rlm_subagent_once(
        self: &Arc<Self>,
        parent_state: Arc<StdMutex<ActiveSessionState>>,
        entry: PassiveRlmSubagentEntry,
        restore_active_session_id: Option<String>,
        client_env: Option<HashMap<String, String>>,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        if let Some(existing) = self.find_session_by_session_file(Some(&entry.session_file)) {
            self.close_session(existing, "replaced", true, true, None, None)
                .await?;
        }
        let hydration_env = parent_state
            .lock()
            .expect("active session poisoned")
            .client_env
            .clone()
            .or(client_env);
        let session_manager = SessionManager::open_async(
            &entry.session_file,
            Some(entry.session_dir.as_str()).filter(|dir| !dir.is_empty()),
            None,
        )
        .await
        .map_err(|error| error.to_string())?;
        let cwd = session_manager.get_cwd();
        let parent_active_session_id = parent_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let parent_session_id = self.session_of(&parent_state).session_id();
        let parent_session_file = self.session_of(&parent_state).session_file();
        let metadata: Value = {
            let mut object = Map::new();
            object.insert("kind".to_string(), Value::String("subagent".to_string()));
            object.insert("createdAt".to_string(), Value::from(entry.created_at));
            object.insert(
                "parentActiveSessionId".to_string(),
                Value::String(parent_active_session_id),
            );
            object.insert(
                "parentSessionId".to_string(),
                Value::String(parent_session_id),
            );
            if let Some(parent_session_file) = &parent_session_file {
                object.insert(
                    "parentSessionFile".to_string(),
                    Value::String(parent_session_file.clone()),
                );
            }
            object.insert(
                "rlmChildId".to_string(),
                Value::String(entry.child_id.clone()),
            );
            object.insert(
                "rlmParentNodeId".to_string(),
                Value::String(
                    entry
                        .rlm_parent_node_id
                        .clone()
                        .unwrap_or_else(|| entry.child_id.clone()),
                ),
            );
            object.insert("rehydratedCompleted".to_string(), Value::Bool(true));
            if let Some(prompt) = &entry.prompt {
                object.insert("prompt".to_string(), Value::String(prompt.clone()));
            }
            if let Some(spawn_code) = &entry.spawn_code {
                object.insert("spawnCode".to_string(), Value::String(spawn_code.clone()));
            }
            object.insert(
                "sessionDir".to_string(),
                Value::String(entry.session_dir.clone()),
            );
            Value::Object(object)
        };
        // The same three-controller `sessionOptions` as `createRuntime`, from the
        // `rehydrateCompletedRlmSubagent` literal (`daemon-mode.ts:3261-3290`):
        // `stateRef` is the `let stateRef` at `:3239`, assigned at `:3324-3326`.
        let state_ref: Arc<StdMutex<Option<Arc<StdMutex<ActiveSessionState>>>>> =
            Arc::new(StdMutex::new(None));
        let get_current_state: Arc<
            dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync,
        > = {
            let state_ref = Arc::clone(&state_ref);
            Arc::new(move || state_ref.lock().expect("session state slot poisoned").clone())
        };
        let session_options = SessionRuntimeOptions {
            subagent_options: None,
            model: None,
            rlm_heartbeat_controller: Some(self.create_rlm_heartbeat_controller(Arc::clone(
                &get_current_state,
            ))),
            agent_message_controller: Some(
                self.create_agent_message_controller(Arc::clone(&get_current_state)),
            ),
            agent_observe_controller: Some(
                self.create_agent_observe_controller(Arc::clone(&get_current_state)),
            ),
        };
        let input = CreateAgentSessionRuntimeInput {
            factory: Value::Null,
            cwd,
            agent_dir: self.options.default_session_config.agent_dir.clone(),
            session_manager: Arc::new(StdMutex::new(session_manager)),
            session_options,
            session_config: Some(serde_json::from_value(serde_json::to_value(&self.options.default_session_config).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?),
            // `runtimeMetadata` is handed to the factory as the same object the
            // TypeScript builds, so it stays a value here.
            runtime_metadata: Some(metadata.clone()),
        };
        let runtime = super::daemon_client_env::with_client_env(hydration_env.as_ref(), || (self.options.create_runtime)(input)).await?;
        let state_ref_for_callback = Arc::clone(&state_ref);
        let state = self
            .add_runtime(runtime.clone(), restore_active_session_id, Some(Arc::new(move |state| {
                {
                    let mut slot = state_ref_for_callback.lock().expect("session state slot poisoned");
                    *slot = Some(Arc::clone(state));
                }
                state.lock().expect("active session poisoned").client_env = hydration_env.clone();
            })), None)
            .await?;
        // The session transcript is authoritative for mutable metadata such as a
        // later user-assigned name; the registry value is only the spawn snapshot.
        if !self.session_of(&parent_state)
            .register_rlm_child_session(&entry.child_id, Arc::clone(&runtime.session))
        {
            self.close_session(Arc::clone(&state), "replaced", true, true, None, None)
                .await?;
            return Err(RuntimeOpenCancelledError.to_string());
        }
        let parent_resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(
                &parent_state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone(),
            )
            .map(|candidate| Arc::ptr_eq(&candidate.state, &parent_state))
            .unwrap_or(false);
        if !parent_resident
            || self
                .closing_sessions
                .lock()
                .expect("closing sessions poisoned")
                .contains_key(
                    &parent_state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                )
        {
            let unsubscribe_child = self.session_of(&parent_state)
                .release_rlm_child_session(&entry.child_id, Arc::clone(&runtime.session));
            let close_result = self
                .close_session(Arc::clone(&state), "replaced", true, true, None, None)
                .await;
            if let Some(unsubscribe_child) = unsubscribe_child {
                unsubscribe_child();
            }
            close_result?;
            return Err(RuntimeOpenCancelledError.to_string());
        }
        Ok(state)
    }

    /// `getOrCreateCronJobSession(job, requirePersistedJob)`.
    async fn get_or_create_cron_job_session(
        self: &Arc<Self>,
        job: &AgentCronJob,
        require_persisted_job: bool,
    ) -> Option<Arc<StdMutex<ActiveSessionState>>> {
        let due_job = if require_persisted_job {
            self.get_runnable_cron_job(&job.id)
        } else {
            Some(job.clone())
        };
        let due_job = due_job?;
        let active_id_match = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(&due_job.active_session_id)
            .map(|entry| Arc::clone(&entry.state));
        let current = self
            .find_session_by_session_file(Some(&due_job.session_file))
            .or_else(|| {
                let matches = !require_persisted_job
                    || active_id_match
                        .as_ref()
                        .map(|state| self.session_of(state).session_id() == due_job.session_id)
                        .unwrap_or(false);
                matches.then_some(active_id_match.clone()).flatten()
            });
        let requires_rlm_subagent_restore = due_job.source.as_deref() == Some(SOURCE_RLM_HEARTBEAT)
            && due_job.runtime_kind.as_deref() == Some("subagent")
            && current
                .as_ref()
                .and_then(|state| {
                    state
                        .lock()
                        .expect("active session poisoned")
                        .runtime
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.kind.clone())
                })
                .as_deref()
                != Some("subagent");
        // A half-bound match falls through to createRuntime, which awaits the
        // pending create for the same session file instead of prompting mid-bind.
        if let Some(current) = &current {
            if !requires_rlm_subagent_restore {
                let active_session_id = current
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                if !self
                    .binding_sessions
                    .lock()
                    .expect("binding sessions poisoned")
                    .contains(&active_session_id)
                {
                    self.rebind_cron_jobs_to_state(current);
                    let rebound_job = if require_persisted_job {
                        self.get_runnable_cron_job(&job.id)
                    } else {
                        Some(due_job.clone())
                    };
                    return match rebound_job {
                        Some(rebound_job)
                            if self.is_cron_job_runnable_for_state(
                                &rebound_job,
                                current,
                                require_persisted_job,
                            ) =>
                        {
                            Some(Arc::clone(current))
                        }
                        _ => None,
                    };
                }
            }
        }
        if !require_persisted_job {
            return None;
        }
        if due_job.source.as_deref() == Some(SOURCE_RLM_HEARTBEAT)
            && due_job.runtime_kind.as_deref() == Some("subagent")
        {
            if let Some(current) = &current {
                let active_session_id = current
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                if self
                    .binding_sessions
                    .lock()
                    .expect("binding sessions poisoned")
                    .contains(&active_session_id)
                {
                    return None;
                }
            }
            return self.restore_rlm_heartbeat_session(&due_job).await;
        }
        if !self.is_persisted_cron_job_runnable(&due_job.id).await {
            return None;
        }
        let command = ParsedDaemonCommand {
            id: None,
            type_: "create".to_string(),
            body: serde_json::json!({ "sessionPath": due_job.session_file }),
        };
        let daemon = Arc::clone(self);
        let job_id = due_job.id.clone();
        let guard: RuntimeOpenGuard = Arc::new(move || {
            let daemon = Arc::clone(&daemon);
            let job_id = job_id.clone();
            Box::pin(async move { daemon.is_persisted_cron_job_runnable(&job_id).await })
        });
        match self.create_runtime(&command, Some(guard)).await {
            Ok(state) => Some(state),
            Err(error) => {
                if error == RuntimeOpenCancelledError.to_string() {
                    None
                } else {
                    None
                }
            }
        }
    }

    /// `restoreRlmHeartbeatSession(job)`.
    async fn restore_rlm_heartbeat_session(
        self: &Arc<Self>,
        job: &AgentCronJob,
    ) -> Option<Arc<StdMutex<ActiveSessionState>>> {
        let child_info = read_session_info(&job.session_file).await;
        let parent_session_path = child_info
            .as_ref()
            .and_then(|info| info.parent_session_path.clone());
        let parent_info = match &parent_session_path {
            Some(path) => read_session_info(path).await,
            None => None,
        };
        let valid = child_info
            .as_ref()
            .map(|info| info.id == job.session_id)
            .unwrap_or(false)
            && parent_session_path.is_some()
            && parent_info
                .as_ref()
                .and_then(|info| info.state.as_ref())
                .is_some_and(|state| state.status == crate::core::session_manager::SessionStateStatus::Active);
        if !valid {
            self.cancel_rlm_heartbeat(&job.id);
            return None;
        }
        let parent_session_path = parent_session_path.unwrap_or_default();
        let resident_child = self.find_session_by_session_file(Some(&job.session_file));
        let include_resident = resident_child
            .as_ref()
            .map(|state| {
                state
                    .lock()
                    .expect("active session poisoned")
                    .runtime
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.kind.as_deref())
                    != Some("subagent")
            })
            .unwrap_or(false);
        let command = ParsedDaemonCommand {
            id: None,
            type_: "create".to_string(),
            body: serde_json::json!({ "sessionPath": parent_session_path }),
        };
        let daemon = Arc::clone(self);
        let job_id = job.id.clone();
        let guard: RuntimeOpenGuard = Arc::new(move || {
            let daemon = Arc::clone(&daemon);
            let job_id = job_id.clone();
            Box::pin(async move { daemon.get_runnable_cron_job(&job_id).is_some() })
        });
        if self.create_runtime(&command, Some(guard)).await.is_err() {
            return None;
        }
        let passive_subagent = self
            .find_passive_rlm_subagent(&job.session_file, include_resident)
            .await;
        let resident = self.find_session_by_session_file(Some(&job.session_file));
        let child_state = match passive_subagent {
            Some(ref passive) => match self
                .hydrate_passive_rlm_subagent(passive.clone(), None)
                .await
            {
                Ok(state) => Some(state),
                Err(_) => None,
            },
            None => match resident {
                Some(resident) => self.wait_for_bound_session(resident).await.ok(),
                None => None,
            },
        };
        if passive_subagent.is_some()
            && child_state.is_some()
            && self.get_runnable_cron_job(&job.id).is_none()
        {
            return None;
        }
        let Some(child_state) = child_state else {
            self.cancel_rlm_heartbeat(&job.id);
            return None;
        };
        if child_state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.kind.as_deref())
            != Some("subagent")
        {
            self.cancel_rlm_heartbeat(&job.id);
            return None;
        }
        self.rebind_cron_jobs_to_state(&child_state);
        let rebound_job = self.get_runnable_cron_job(&job.id);
        match rebound_job {
            Some(rebound_job)
                if self.is_cron_job_runnable_for_state(&rebound_job, &child_state, true) =>
            {
                Some(child_state)
            }
            _ => None,
        }
    }

    /// `cancelRlmHeartbeat(jobId)`.
    fn cancel_rlm_heartbeat(&self, job_id: &str) {
        if self.cron_store.cancel(job_id, now_millis()).is_some() {
            self.cron_scheduler_wake();
        }
    }

    /// `isPersistedCronJobRunnable(jobId)`.
    async fn is_persisted_cron_job_runnable(self: &Arc<Self>, job_id: &str) -> bool {
        for _ in 0..2 {
            let Some(job) = self.get_runnable_cron_job(job_id) else {
                return false;
            };
            let session_file = resolve_path(&job.session_file);
            let session_info = read_session_info(&session_file).await;
            let Some(current) = self.get_runnable_cron_job(job_id) else {
                return false;
            };
            if resolve_path(&current.session_file) != session_file
                || current.session_id != job.session_id
            {
                continue;
            }
            let valid = session_info
                .as_ref()
                .map(|info| info.id == current.session_id)
                .unwrap_or(false)
                && session_info
                    .as_ref()
                    .and_then(|info| info.state.as_ref())
                    .is_some_and(|state| state.status
                        == crate::core::session_manager::SessionStateStatus::Active);
            if !valid {
                self.cancel_scheduled_jobs_for_session_file(&current.session_file);
                return false;
            }
            return true;
        }
        false
    }

    /// `isCronJobRunnableForState(job, state, requirePersistedJob)`.
    fn is_cron_job_runnable_for_state(
        &self,
        job: &AgentCronJob,
        state: &Arc<StdMutex<ActiveSessionState>>,
        require_persisted_job: bool,
    ) -> bool {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let resident = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get(&active_session_id)
            .map(|entry| Arc::ptr_eq(&entry.state, state))
            .unwrap_or(false);
        if !resident
            || self
                .closing_sessions
                .lock()
                .expect("closing sessions poisoned")
                .contains_key(&active_session_id)
        {
            return false;
        }
        if !require_persisted_job {
            return job.status == crate::core::cron_jobs::STATUS_ACTIVE
                && job.active_session_id == active_session_id;
        }
        let Some(current) = self.get_runnable_cron_job(&job.id) else {
            return false;
        };
        let session = self.session_of(state);
        let session_file = session.session_file();
        current.active_session_id == active_session_id
            && current.session_id == session.session_id()
            && session_file
                .map(|session_file| {
                    resolve_path(&current.session_file) == resolve_path(&session_file)
                })
                .unwrap_or(false)
    }
}

impl AgentDaemon {
    /// `agentMessageRelationship(fromState, targetState)`.
    fn agent_message_relationship(
        &self,
        from_state: Option<&Arc<StdMutex<ActiveSessionState>>>,
        target_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Option<AgentFamilyRelationship> {
        let from_state = from_state?;
        agent_family_relationship(
            &self.agent_family_entry(target_state),
            &self.agent_family_entry(from_state),
        )
    }

    /// `agentFamilyEntry(state)`.
    fn agent_family_entry(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> AgentFamilyCatalogEntry {
        let session = self.session_of(state);
        let metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let depth = session.rlm_depth().unwrap_or(0);
        let header_parent = if depth > 0 {
            self.resolve_header_parent_session_path(state)
        } else {
            None
        };
        let parent_session_path = if depth > 0 {
            header_parent.or_else(|| {
                metadata
                    .as_ref()
                    .and_then(|metadata| metadata.parent_session_file.clone())
            })
        } else {
            None
        };
        AgentFamilyCatalogEntry {
            id: session.session_id(),
            name: session.session_name(),
            depth: depth as f64,
            status: crate::core::agent_messages::FAMILY_STATUS_RUNNING.to_string(),
            session_path: session
                .session_file()
                .map(|path| canonical_session_path(&path)),
            parent_session_id: (depth > 0)
                .then(|| {
                    metadata
                        .as_ref()
                        .and_then(|metadata| metadata.parent_session_id.clone())
                })
                .flatten(),
            parent_session_path: parent_session_path.map(|path| canonical_session_path(&path)),
            ..AgentFamilyCatalogEntry::default()
        }
    }

    /// `passiveAgentFamilyEntry(passive)`.
    fn passive_agent_family_entry(&self, passive: &PassiveRlmSubagent) -> AgentFamilyCatalogEntry {
        let entry = &passive.entry;
        let parent_entry = passive
            .chain
            .len()
            .checked_sub(2)
            .and_then(|index| passive.chain.get(index));
        let root_parent_session_file = match &passive.root {
            PassiveRlmRoot::Resident(state) => self.session_of(state).session_file(),
            PassiveRlmRoot::Saved(info) => Some(info.path.clone()),
        };
        AgentFamilyCatalogEntry {
            id: passive.info.id.clone(),
            name: passive
                .info
                .name
                .clone()
                .or_else(|| Some(entry.session_name.clone())),
            depth: entry.rlm_depth.unwrap_or(passive.info.rlm_depth) as f64,
            status: FAMILY_STATUS_INACTIVE.to_string(),
            session_path: Some(canonical_session_path(&entry.session_file)),
            parent_session_id: Some(entry.parent_session_id.clone()),
            parent_session_path: entry
                .parent_session_file
                .clone()
                .or_else(|| parent_entry.map(|entry| entry.session_file.clone()))
                .or(root_parent_session_file)
                .map(|path| canonical_session_path(&path)),
            ..AgentFamilyCatalogEntry::default()
        }
    }

    /// `assertAgentFamilyReachable(currentState, targetState)`.
    fn assert_agent_family_reachable(
        &self,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        target_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Result<(), String> {
        assert_agent_family_reach(
            &self.agent_family_entry(current_state),
            &self.agent_family_entry(target_state),
        )
        .map(|_| ())
    }

    /// `sendAgentSessionMessage(options)`.
    async fn send_agent_session_message(
        self: &Arc<Self>,
        options: SendAgentMessageInput,
    ) -> Result<AgentSessionMessageReceipt, String> {
        if self.agent_messages_paused.load(Ordering::SeqCst) {
            return Err("Agent messaging is paused".to_string());
        }
        let target_selector = assert_direct_agent_message_target(&options.target_selector)?;
        let message =
            normalize_agent_session_message(&options.message, DEFAULT_AGENT_MESSAGE_MAX_CHARS)?;
        let target_state = match self.get_bound_session_state(&target_selector) {
            Ok(state) => state,
            Err(error) => {
                if error.starts_with("__bound_session_unavailable__") {
                    if options.origin == "agent" {
                        if let Some(from_state) = &options.from_state {
                            let state = self.get_session_state(&target_selector)?;
                            self.assert_agent_family_reachable(from_state, &state)?;
                        }
                    }
                    self.get_or_hydrate_bound_session_state(&target_selector)
                        .await?
                } else if error.starts_with("__ambiguous_active_session__") {
                    self.get_or_hydrate_bound_session_state(&target_selector)
                        .await?
                } else if let Some(passive) = self
                    .find_passive_rlm_subagent(&target_selector, false)
                    .await
                {
                    if options.origin == "agent" {
                        if let Some(from_state) = &options.from_state {
                            assert_agent_family_reach(
                                &self.agent_family_entry(from_state),
                                &self.passive_agent_family_entry(&passive),
                            )?;
                        }
                    }
                    self.hydrate_passive_rlm_subagent(passive, None).await?
                } else if let Some(hydrating_child) =
                    self.session_states().into_iter().find(|entry| {
                        let state = entry.state.lock().expect("active session poisoned");
                        let metadata = state.runtime.metadata.as_ref();
                        metadata.and_then(|metadata| metadata.kind.as_deref()) == Some("subagent")
                            && metadata.and_then(|metadata| metadata.rlm_child_id.as_deref())
                                == Some(target_selector.as_str())
                    })
                {
                    self.wait_for_hydrating_child(Arc::clone(&hydrating_child.state), &target_selector)
                        .await?
                } else if self.is_worker() && options.from_state.is_some() {
                    // The supervisor can resolve and wake a saved worker even when it
                    // is no longer present in this worker's resident peer snapshot.
                    return self
                        .send_remote_agent_session_message(
                            options.from_state.as_ref().expect("checked above"),
                            &target_selector,
                            &message,
                        )
                        .await;
                } else {
                    return Err(error);
                }
            }
        };
        if let Some(from_state) = &options.from_state {
            if from_state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                == target_state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
            {
                return Err("Agent messaging cannot target the sending session".to_string());
            }
        }
        if options.origin == "agent" {
            if let Some(from_state) = &options.from_state {
                self.assert_agent_family_reachable(from_state, &target_state)?;
            }
        }
        let target_active_session_id = target_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let sender_key = options
            .sender_key
            .clone()
            .or_else(|| {
                options.from_state.as_ref().map(|state| {
                    state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone()
                })
            })
            .unwrap_or_else(|| {
                format!(
                    "client:{}",
                    options
                        .client_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string())
                )
            });
        let rate_limit_key = format!("{sender_key}->{target_active_session_id}");
        let rate_limit = self
            .agent_message_rate_limiter
            .lock()
            .expect("agent message rate limiter poisoned")
            .try_consume(&rate_limit_key);
        match rate_limit {
            RateLimitResult::Ok => {}
            RateLimitResult::Exceeds { retry_after_ms } => {
                return Err(format!(
                    "Agent messaging rate limit exceeded; retry after {retry_after_ms}ms"
                ))
            }
        }
        let payload = AgentSessionMessagePayload {
            id: create_agent_session_message_id(),
            source: AGENT_MESSAGE_SOURCE.to_string(),
            message,
            from: options.sender.clone().or_else(|| {
                Some(
                    self.create_agent_session_message_sender(
                        options.from_state.as_ref(),
                        options
                            .client_id
                            .clone()
                            .unwrap_or_else(|| options.origin.clone())
                            .as_str(),
                    ),
                )
            }),
            from_relationship: self
                .agent_message_relationship(options.from_state.as_ref(), &target_state),
            target: self.create_agent_session_message_endpoint(&target_state),
        };
        match self
            .accept_agent_session_message(&target_state, &payload)
            .await
        {
            Ok(status) => Ok(create_agent_session_message_receipt(
                &payload,
                &status,
                &now_iso(),
            )),
            Err(error) => {
                self.agent_message_rate_limiter
                    .lock()
                    .expect("agent message rate limiter poisoned")
                    .refund(&rate_limit_key);
                Err(error)
            }
        }
    }

    /// `acceptAgentSessionMessage(targetState, payload)`.
    async fn accept_agent_session_message(
        self: &Arc<Self>,
        target_state: &Arc<StdMutex<ActiveSessionState>>,
        payload: &AgentSessionMessagePayload,
    ) -> Result<AgentSessionMessageDeliveryStatus, String> {
        let content = create_agent_session_message_prompt(payload);
        let message = create_agent_session_message(payload, now_millis() as i64);
        let target_active_session_id = target_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let target_session_id = payload.target.session_id.clone();
        let daemon = Arc::clone(self);
        let state_for_commit = Arc::clone(target_state);
        let admission_committed: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let paused = daemon.agent_messages_paused.load(Ordering::SeqCst);
            let closing = daemon
                .closing_sessions
                .lock()
                .expect("closing sessions poisoned")
                .contains_key(&target_active_session_id);
            let resident = daemon
                .sessions
                .lock()
                .expect("sessions poisoned")
                .contains_key(&target_active_session_id);
            if paused || !resident || closing {
                daemon.log("Agent message admission rejected");
                return;
            }
            if daemon.session_of(&state_for_commit).session_id() != target_session_id {
                daemon.log("Target session changed before agent message delivery");
            }
        });
        // `let preflightFailed = false; let preflightQueued = false;` and the
        // `preflightResult: (didSucceed, didQueue) => {...}` capture
        // (daemon-mode.ts:6380-6404): the session's own verdict decides the receipt.
        let preflight_failed = Arc::new(AtomicBool::new(false));
        let preflight_queued = Arc::new(AtomicBool::new(false));
        let preflight_result: Arc<dyn Fn(bool, bool) + Send + Sync> = {
            let preflight_failed = Arc::clone(&preflight_failed);
            let preflight_queued = Arc::clone(&preflight_queued);
            Arc::new(move |did_succeed: bool, did_queue: bool| {
                preflight_failed.store(!did_succeed, Ordering::SeqCst);
                preflight_queued.store(did_succeed && did_queue, Ordering::SeqCst);
            })
        };
        let invocation = PromptInvocation {
            expand_prompt_templates: Some(false),
            streaming_behavior: Some(DELIVERY_MODE_STEER.to_string()),
            queue_if_busy: Some(true),
            custom_message: Some(serde_json::to_value(&message).unwrap_or(Value::Null)),
            admission_committed: Some(admission_committed),
            preflight_result: Some(preflight_result),
            ..PromptInvocation::default()
        };
        self.session_of(target_state)
            .accept_agent_message_prompt(&content, invocation)
            .await?;
        // `if (preflightFailed) throw new Error("Agent message was not accepted");`
        // (daemon-mode.ts:6406-6407) - a rejected admission is the caller's error, never a
        // receipt. `return { status: preflightQueued ? "queued" : "delivered" }` (:6409).
        if preflight_failed.load(Ordering::SeqCst) {
            return Err("Agent message was not accepted".to_string());
        }
        Ok(if preflight_queued.load(Ordering::SeqCst) {
            DELIVERY_STATUS_QUEUED.to_string()
        } else {
            DELIVERY_STATUS_DELIVERED.to_string()
        })
    }

    /// `sendRemoteAgentSessionMessage(fromState, targetSelector, message)`.
    async fn send_remote_agent_session_message(
        self: &Arc<Self>,
        from_state: &Arc<StdMutex<ActiveSessionState>>,
        target_selector: &str,
        message: &str,
    ) -> Result<AgentSessionMessageReceipt, String> {
        let Some(socket_path) = self.supervisor_socket_path_from_env() else {
            return Err(format!("Unknown active session: {target_selector}"));
        };
        let from_active_session_id = from_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        agent_message_transport::send_message(
            &socket_path,
            &from_active_session_id,
            target_selector,
            message,
            &self.shutting_down,
        )
        .await
    }
}

impl AgentDaemon {
    /// slice plumbing: `buildRlmChildSnapshots(rootActiveSessionId, activeSessions)`
    /// from daemon-session-list.ts (that module owns the list projection; the
    /// daemon needs this one entry point here).
    fn build_rlm_child_snapshots_plumbing(&self, root_active_session_id: &str) -> Vec<Value> {
        let states = self.state_refs();
        let root = states.iter().find(|state| {
            state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                == root_active_session_id
        });
        let Some(root) = root else {
            return Vec::new();
        };
        let mut active_session_ids: HashMap<String, String> = HashMap::new();
        for state in &states {
            let state = state.lock().expect("active session poisoned");
            let child_id = state
                .runtime
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.rlm_child_id.clone());
            if let Some(child_id) = child_id {
                active_session_ids.insert(child_id, state.active_session_id.clone());
            }
        }
        let root = Arc::clone(root);
        self.session_of(&root)
            .get_rlm_child_snapshots()
            .into_iter()
            .map(|mut snapshot| {
                let child_id = snapshot
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let active_session_id = child_id
                    .as_ref()
                    .and_then(|child_id| active_session_ids.get(child_id))
                    .cloned();
                if let Some(object) = snapshot.as_object_mut() {
                    match active_session_id {
                        Some(active_session_id) => {
                            object.insert(
                                "activeSessionId".to_string(),
                                Value::String(active_session_id),
                            );
                        }
                        None => {
                            object.remove("activeSessionId");
                        }
                    }
                }
                snapshot
            })
            .collect()
    }
}

impl AgentDaemon {
    /// `recordRlmSubagentState(parentState, input)`.
    ///
    /// Spawn admission is the moment the daemon knows the edge firsthand, so the
    /// ledger append is load-bearing: the promise is stashed per childId for the
    /// admission path to await (admission fails when the spawn record cannot be
    /// made durable). Display metadata goes to the child's per-child display file
    /// at both moments.
    fn record_rlm_subagent_state(
        self: &Arc<Self>,
        parent_state: &Arc<StdMutex<ActiveSessionState>>,
        input: RlmSubagentStateInput,
    ) -> bool {
        let parent_session_file = self.session_of(parent_state).session_file();
        let parent_active_session_id = parent_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if input.status == "running" {
            if let Some(parent_file) = &parent_session_file {
                let child_id_for_task = input.child_id.clone();
                let child_file = input.session_file.clone();
                let depth_for_task = input.rlm_depth;
                let name_for_task = input.session_name.clone();
                let parent_file = parent_file.clone();
                let ledger = self.rlm_spawn_ledger();
                let key = format!("{parent_active_session_id}#{}", input.child_id);
                // Mark handled so an early rejection cannot surface as an
                // unhandled-rejection crash before the admission path awaits it.
                let task = tokio::spawn(async move {
                    ledger
                        .append_spawn(RlmSpawnInput {
                            child_id: child_id_for_task,
                            parent: parent_file,
                            child: child_file,
                            depth: depth_for_task,
                            name: name_for_task,
                        })
                        .await
                });
                // Child ids are only unique per parent; the parent scopes the key.
                self.pending_rlm_spawn_appends
                    .lock()
                    .expect("pending rlm spawn appends poisoned")
                    .insert(key, task);
            }
        }
        let entry = RlmSubagentDisplayEntry {
            type_: "rlm_subagent".to_string(),
            child_id: input.child_id.clone(),
            session_name: input.session_name.clone(),
            session_dir: input.session_dir.clone(),
            session_file: input.session_file.clone(),
            rlm_max_depth: Some(input.rlm_max_depth),
            rlm_parent_node_id: input.rlm_parent_node_id.clone(),
            prompt: input.prompt.clone(),
            spawn_code: input.spawn_code.clone(),
            model: input.model.clone(),
            status: input.status.clone(),
            created_at: input.created_at.unwrap_or_else(now_millis),
            updated_at: now_iso(),
        };
        match write_rlm_subagent_display_entry(&entry) {
            Ok(written) => {
                if !written {
                    self.log(&format!(
                        "skipped RLM subagent display entry for {}: deleted tombstone exists",
                        input.child_id
                    ));
                }
                written
            }
            Err(error) => {
                self.log(&format!(
                    "failed to persist RLM subagent display entry: {error}"
                ));
                false
            }
        }
    }

    /// Best-effort artifact-dir removal: cache cleanup must never fail a deletion.
    async fn delete_rlm_subagent_artifacts(&self, child_id: &str, child_session_file: &str) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            delete_session_artifacts(child_session_file);
        }));
        if result.is_err() {
            self.log(&format!(
                "failed to remove artifact dir for deleted RLM subagent {child_id}"
            ));
        }
    }

    /// `recordRlmSubagentDeletion(parentState, childId, reason = "user")`.
    async fn record_rlm_subagent_deletion(
        self: &Arc<Self>,
        parent_state: &Arc<StdMutex<ActiveSessionState>>,
        child_id: &str,
        reason: RlmLedgerDeleteReason,
    ) -> Result<(), String> {
        let parent_session_file = self.session_of(parent_state).session_file();
        let Some(parent_file) = parent_session_file else {
            return Ok(());
        };
        let parent_session_id = self.session_of(parent_state).session_id();
        let parent_path = canonical_session_path(&parent_file);
        let ledger = self.rlm_spawn_ledger();
        let edges = ledger
            .edges(true)
            .await
            .into_iter()
            .filter(|candidate| {
                candidate.child_id == child_id
                    && canonical_session_path(&candidate.parent) == parent_path
            })
            .collect::<Vec<_>>();
        let live_edge = edges
            .iter()
            .find(|candidate| candidate.deleted.is_none())
            .cloned();
        let entry = if let Some(edge) = live_edge {
            {
                let mut cache: HashMap<String, Vec<LegacyRlmSubagentRegistryEntry>> =
                    HashMap::new();
                let noop: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|_path: &str| {});
                self.passive_rlm_subagent_entry_for_edge(
                    &edge,
                    &parent_session_id,
                    &parent_file,
                    &mut cache,
                    noop,
                )
                .await
            }
        } else if !edges.is_empty() {
            // Only tombstoned edges: the tombstones are already durable, nothing to
            // re-append. A prior deletion may have crashed before its artifact sweep,
            // so restore the display tombstone before sweeping artifacts.
            for tombstoned in &edges {
                let display_dir = dirname(&tombstoned.child);
                let current_display = read_rlm_subagent_display_entry(&display_dir, None).await;
                if current_display
                    .as_ref()
                    .map(|display| display.status != "deleted")
                    .unwrap_or(true)
                {
                    let entry = RlmSubagentDisplayEntry {
                        type_: "rlm_subagent".to_string(),
                        child_id: child_id.to_string(),
                        session_name: current_display
                            .as_ref()
                            .map(|display| display.session_name.clone())
                            .unwrap_or_else(|| tombstoned.name.clone()),
                        session_dir: display_dir.clone(),
                        session_file: current_display
                            .as_ref()
                            .map(|display| display.session_file.clone())
                            .unwrap_or_else(|| tombstoned.child.clone()),
                        rlm_max_depth: current_display
                            .as_ref()
                            .and_then(|display| display.rlm_max_depth),
                        rlm_parent_node_id: current_display
                            .as_ref()
                            .and_then(|display| display.rlm_parent_node_id.clone()),
                        prompt: current_display
                            .as_ref()
                            .and_then(|display| display.prompt.clone()),
                        spawn_code: current_display
                            .as_ref()
                            .and_then(|display| display.spawn_code.clone()),
                        model: current_display
                            .as_ref()
                            .and_then(|display| display.model.clone()),
                        status: "deleted".to_string(),
                        created_at: current_display
                            .as_ref()
                            .map(|display| display.created_at)
                            .unwrap_or(0.0),
                        updated_at: now_iso(),
                    };
                    // Best-effort: the ledger tombstone is the authority; the display
                    // file is display-grade and the sweep below removes artifacts.
                    if write_rlm_subagent_display_entry(&entry).is_err() {
                        self.log(&format!(
                            "failed to reconcile display entry for tombstoned RLM subagent {child_id}"
                        ));
                    }
                }
                self.delete_rlm_subagent_artifacts(child_id, &tombstoned.child)
                    .await;
            }
            return Ok(());
        } else {
            // No edge at all. A pre-ledger child the seed missed may still exist in
            // the legacy registry; an unreadable registry means the durable deletion
            // boundary cannot be established, so the deletion fails.
            let legacy = self
                .read_legacy_rlm_subagent_registry(
                    &self.legacy_rlm_subagent_registry_path(&parent_file, &parent_session_id),
                    None,
                )
                .await
                .map_err(|error| error.to_string())?
                .into_iter()
                .find(|candidate| candidate.child_id == child_id);
            let Some(legacy) = legacy else {
                return Ok(());
            };
            if legacy.status == "deleted" {
                // The child never existed under this parent, or its tombstone is
                // already durable.
                return Ok(());
            }
            // `entry = { childId, sessionName, ..., ...rlmSubagentMetadataFields(legacy),
            // status, createdAt }` (daemon-mode.ts 1215-1226). The TypeScript spreads the
            // metadata helper; the port builds the same record explicitly.
            PassiveRlmSubagentEntry {
                child_id: legacy.child_id.clone(),
                session_name: legacy.session_name.clone(),
                session_dir: legacy.session_dir.clone(),
                session_file: legacy.session_file.clone(),
                parent_session_id: parent_session_id.clone(),
                parent_session_file: Some(parent_file.clone()),
                rlm_depth: legacy.rlm_depth,
                rlm_max_depth: legacy.rlm_max_depth,
                rlm_parent_node_id: legacy.rlm_parent_node_id.clone(),
                prompt: legacy.prompt.clone(),
                spawn_code: legacy.spawn_code.clone(),
                model: legacy
                    .model
                    .clone()
                    .and_then(|value| serde_json::from_value(value).ok()),
                status: legacy.status.clone(),
                created_at: legacy.created_at,
            }
        };
        // Display tombstone first ("deleted deliberately, transcript retained"): a
        // crash in between leaves a live ledger edge over a deleted display entry,
        // healed by retrying the deletion; the reverse order could tombstone the
        // ledger while the display file still claims the child exists.
        let tombstone = RlmSubagentDisplayEntry {
            type_: "rlm_subagent".to_string(),
            child_id: entry.child_id.clone(),
            session_name: entry.session_name.clone(),
            session_dir: entry.session_dir.clone(),
            session_file: entry.session_file.clone(),
            rlm_max_depth: entry.rlm_max_depth,
            rlm_parent_node_id: entry.rlm_parent_node_id.clone(),
            prompt: entry.prompt.clone(),
            spawn_code: entry.spawn_code.clone(),
            model: entry.model.clone(),
            status: "deleted".to_string(),
            created_at: entry.created_at,
            updated_at: now_iso(),
        };
        write_rlm_subagent_display_entry(&tombstone).map_err(|error| {
            format!("Failed to persist deletion for RLM subagent {child_id}: {error}")
        })?;
        // The ledger delete record is the topology tombstone; unlike the dual-write
        // era it has no other writer to fall back on, so a failed append is a failed
        // deletion.
        ledger
            .append_delete(child_id, &entry.session_file, reason)
            .await?;
        if self.is_worker() {
            let agent_id =
                self.roster_agent_id_for_rlm_child(child_id, entry.parent_session_file.as_deref());
            self.roster_reporter
                .lock()
                .expect("roster reporter poisoned")
                .removed_agent_ids
                .insert(agent_id, Some(basename(&entry.session_file, Some(".jsonl"))));
            self.schedule_roster_flush();
        }
        // Deletion boundary: transcript + display tombstone are the durable record
        // and stay; the nested artifact dir is a runtime cache and goes.
        self.delete_rlm_subagent_artifacts(child_id, &entry.session_file)
            .await;
        Ok(())
    }

    /// `appendUpdateRestartMarker(state, restartSession)`.
    fn append_update_restart_marker(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        restart_session: &UpdateRestartSessionSnapshot,
    ) {
        if !restart_session.should_resume {
            return;
        }
        let session = self.session_of(state);
        // `sessionManager.appendCustomMessageEntry(customType, content, display, details)`.
        let payload = serde_json::json!({
            "customType": "prime-agent.update_restart",
            "content": UPDATE_RESTART_MARKER,
            "display": false,
            "details": {
                "activeSessionId": restart_session.active_session_id,
                "wasStreaming": restart_session.was_streaming,
                "wasCompacting": restart_session.was_compacting,
                "wasBashRunning": restart_session.was_bash_running,
                "hadRunningRlmChildren": restart_session.had_running_rlm_children,
                "wasRetrying": restart_session.was_retrying,
                "hadAcceptedPromptInFlight": restart_session.had_accepted_prompt_in_flight,
            },
        });
        let _ = session.send_custom_message(&payload);
    }

    /// `getUpdateRestartSessionDepth(state)`.
    fn get_update_restart_session_depth(&self, state: &Arc<StdMutex<ActiveSessionState>>) -> i64 {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut metadata = state
            .lock()
            .expect("active session poisoned")
            .runtime
            .metadata
            .clone();
        let mut depth = 0i64;
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(active_session_id);
        while let Some(parent_active_session_id) = metadata
            .as_ref()
            .and_then(|metadata| metadata.parent_active_session_id.clone())
        {
            if seen.contains(&parent_active_session_id) {
                break;
            }
            let Some(parent) = self.state_refs().into_iter().find(|candidate| {
                candidate
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    == parent_active_session_id
            }) else {
                break;
            };
            seen.insert(parent_active_session_id);
            depth += 1;
            metadata = parent
                .lock()
                .expect("active session poisoned")
                .runtime
                .metadata
                .clone();
        }
        depth
    }

    /// `writeWorkerSnapshotBuffer(client, buffer, message, purpose, signal?, drainTimeoutMs?)`.
    ///
    /// `this.writeSerialized` is synchronous here, so the drain wait is the
    /// remaining failure mode: the write either lands or the buffer reports it.
    async fn write_worker_snapshot_buffer(
        self: &Arc<Self>,
        client: &Arc<DaemonClientHandle>,
        buffer: Vec<u8>,
        message: &DaemonOutbound,
        purpose: &str,
        aborted: bool,
        drain_timeout_ms: Option<u64>,
    ) -> bool {
        if aborted || client.writer.destroyed() {
            return false;
        }
        if self.write_serialized_encoded(client, &buffer, message, "jsonl", Some(purpose)) {
            return true;
        }
        let _ = drain_timeout_ms;
        false
    }

    /// `createAgentObserveSummary(state, currentState)`.
    fn create_agent_observe_summary(
        &self,
        state: &Arc<StdMutex<ActiveSessionState>>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> AgentObserveAgentSummary {
        let summary = self.summary_for_state(state);
        let session = self.session_of(state);
        let session_file = session.session_file();
        // `const messages = session.messages;` (`daemon-mode.ts:3512`), reused for the
        // `latestMessage` preview index (`:3525`).
        let messages = session.messages();
        // `session.isStreaming ? (session.state.pendingToolCalls.size > 0 ? "tool" : "model")`
        // (`daemon-mode.ts:3490-3500`): a streaming turn that is waiting on a tool call
        // is reported as "tool", not "model". The pending set lives on the live agent
        // state (`AgentSession::state().pending_tool_calls`), which is reachable through
        // the seam's `agent_session()` accessor.
        let pending_tool_calls = session
            .agent_session()
            .map(|live| live.state().pending_tool_calls)
            .unwrap_or_default();
        let status = if session.is_streaming() {
            if pending_tool_calls.is_empty() {
                "model".to_string()
            } else {
                "tool".to_string()
            }
        } else if session.is_compacting() {
            "compacting".to_string()
        } else if session.is_session_active() || session.has_running_rlm_children() {
            "busy".to_string()
        } else if !state
            .lock()
            .expect("active session poisoned")
            .clients
            .is_empty()
        {
            "attached_idle".to_string()
        } else {
            "idle".to_string()
        };
        let current_active_session_id = current_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let session_name = session.session_name();
        AgentObserveAgentSummary {
            active_session_id: summary.active_session_id.clone().unwrap_or_default(),
            session_id: summary.session_id.clone(),
            name: session_name.clone(),
            session_name,
            runtime_kind: summary.runtime_kind.clone(),
            cwd: summary.cwd.clone(),
            status,
            is_current: state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                == current_active_session_id,
            is_streaming: summary.is_streaming,
            is_compacting: summary.is_compacting,
            attached_clients: summary.attached_clients as f64,
            message_count: summary.message_count as f64,
            // `transcriptEntryCount: session.sessionManager.getEntries().length`
            // (`daemon-mode.ts:3513`) is the LIFETIME JSONL entry count, deliberately
            // kept distinct from `messageCount` (the active model context, `:3512`).
            transcript_entry_count: session_file.as_ref().map(|_| {
                session
                    .session_manager()
                    .lock()
                    .expect("session manager poisoned")
                    .get_entries()
                    .len() as f64
            }),
            last_activity_at: Some(summary.last_activity_at.as_ref().and_then(|value| {
                chrono::DateTime::parse_from_rfc3339(value)
                    .ok()
                    .map(|parsed| parsed.timestamp_millis() as f64)
            })),
            queued_count: summary
                .session_actions
                .as_ref()
                .and_then(|actions| actions.get("queuedCount"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            is_session_active: summary.is_session_active,
            parent_active_session_id: summary.parent_active_session_id.clone(),
            parent_session_id: summary.parent_session_id.clone(),
            rlm_child_id: summary.rlm_child_id.clone(),
            rlm_parent_node_id: summary.rlm_parent_node_id.clone(),
            // `model: session.model ? `${session.model.provider}/${session.model.id}` : null`
            // (`daemon-mode.ts:3514`). The summary already carries the live model under
            // the wire shape, so read the two id parts from it instead of re-projecting.
            model: Some(
                summary
                    .model
                    .as_ref()
                    .and_then(|model| model.get("provider"))
                    .and_then(Value::as_str)
                    .zip(
                        summary
                            .model
                            .as_ref()
                            .and_then(|model| model.get("id"))
                            .and_then(Value::as_str),
                    )
                    .map(|(provider, id)| format!("{provider}/{id}")),
            ),
            // `...(latest ? { latestMessage: createAgentObserveMessagePreview(latest, messages.length - 1, 240) } : {})`
            // (`daemon-mode.ts:3524-3528`) and `...(summary.firstMessage ? { firstMessage } : {})`
            // (`:3523`).
            first_message: summary.first_message.clone().filter(|first| !first.is_empty()),
            latest_message: messages.last().map(|latest| {
                create_agent_observe_message_preview(
                    latest,
                    (messages.len() as f64) - 1.0,
                    AGENT_OBSERVE_LATEST_MESSAGE_MAX_CHARS,
                )
            }),
            // `...session.rlmDiagnostics` (`daemon-mode.ts:3516`): the child-only
            // continuation diagnostics, undefined at depth 0.
            ..rlm_diagnostics_spread(session.as_ref())
        }
    }
}

impl AgentDaemon {
    /// `isAgentFamilyReachable(currentState, targetState)`.
    fn is_agent_family_reachable(
        &self,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        target_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> bool {
        match assert_agent_family_reach(
            &self.agent_family_entry(current_state),
            &self.agent_family_entry(target_state),
        ) {
            Ok(_) => true,
            Err(error) if error == AGENT_FAMILY_REACH_ERROR => false,
            Err(_) => false,
        }
    }

    /// `resolveAgentFamilySessionName(currentState, target, ambiguity)`.
    fn resolve_agent_family_session_name(
        &self,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        target: &str,
        ambiguity: &str,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        let current_active_session_id = current_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut matches: Vec<Arc<StdMutex<ActiveSessionState>>> = Vec::new();
        for state in self.state_refs() {
            let session = self.session_of(&state);
            let is_match =
                session.session_id() == target || session.session_name().as_deref() == Some(target);
            if !is_match {
                continue;
            }
            let same = state
                .lock()
                .expect("active session poisoned")
                .active_session_id
                == current_active_session_id;
            if same || self.is_agent_family_reachable(current_state, &state) {
                matches.push(state);
            }
        }
        if matches.len() != 1 {
            return Err(ambiguity.to_string());
        }
        Ok(matches.remove(0))
    }

    /// `getOrHydrateAuthorizedAgentFamilyTarget(currentState, target)`.
    async fn get_or_hydrate_authorized_agent_family_target(
        self: &Arc<Self>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        target: &str,
    ) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        match self.get_bound_session_state(target) {
            Ok(state) => return Ok(state),
            Err(error) => {
                if error.starts_with("__bound_session_unavailable__") {
                    let target_state = self.get_session_state(target)?;
                    self.assert_agent_family_reachable(current_state, &target_state)?;
                    return self.get_or_hydrate_bound_session_state(target).await;
                }
                if error.starts_with("__ambiguous_active_session__") {
                    let resolved =
                        self.resolve_agent_family_session_name(current_state, target, &error)?;
                    let active_session_id = resolved
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone();
                    return self
                        .get_or_hydrate_bound_session_state(&active_session_id)
                        .await;
                }
            }
        }
        let Some(passive) = self.find_passive_rlm_subagent(target, false).await else {
            return self.get_or_hydrate_bound_session_state(target).await;
        };
        assert_agent_family_reach(
            &self.agent_family_entry(current_state),
            &self.passive_agent_family_entry(&passive),
        )?;
        self.hydrate_passive_rlm_subagent(passive, None).await
    }

    /// `createAgentObserveAgentSnapshot(currentState, target)`.
    async fn create_agent_observe_agent_snapshot(
        self: &Arc<Self>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        target: &str,
    ) -> Result<AgentObserveAgentSnapshot, String> {
        let target_state = self
            .get_or_hydrate_authorized_agent_family_target(current_state, target)
            .await?;
        self.assert_agent_family_reachable(current_state, &target_state)?;
        Ok(AgentObserveAgentSnapshot {
            agent: self.create_agent_observe_summary(&target_state, current_state),
        })
    }

    /// `createAgentObserveRecentMessages(currentState, input)`.
    async fn create_agent_observe_recent_messages(
        self: &Arc<Self>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
        input: AgentObserveRecentMessagesInput,
    ) -> Result<AgentObserveRecentMessagesResult, String> {
        let target_state = self
            .get_or_hydrate_authorized_agent_family_target(current_state, &input.target)
            .await?;
        self.assert_agent_family_reachable(current_state, &target_state)?;
        // `normalizeObserveLimit(input.limit)` / `normalizeObserveMaxChars(input.maxChars)`
        // are called with NO default argument (`daemon-mode.ts:3467-3468`), so the
        // TypeScript defaults apply: 8 messages and 800 chars (`agent-observe.ts:107`
        // `defaultLimit = 8`, `:111` `defaultMaxChars = 800`). The previous 20/4000 pair
        // contradicted the very clamp it feeds - `normalize_observe_max_chars` rejects
        // anything above 2_000 - so a request that omitted `maxChars` could never
        // succeed.
        let limit = normalize_observe_limit(input.limit, 8)?;
        let max_chars = normalize_observe_max_chars(input.max_chars, 800)?;
        let messages = self.session_of(&target_state).messages();
        let start_index = messages.len().saturating_sub(limit as usize);
        Ok(AgentObserveRecentMessagesResult {
            agent: self.create_agent_observe_summary(&target_state, current_state),
            messages: messages[start_index..]
                .iter()
                .enumerate()
                .map(|(offset, message)| {
                    create_agent_observe_message_preview(
                        message,
                        (start_index + offset) as f64,
                        max_chars as usize,
                    )
                })
                .collect(),
            limit: limit as f64,
            max_chars: max_chars as f64,
            truncated: start_index > 0,
        })
    }

    /// `createAgentObserveListResult(currentState)`.
    async fn create_agent_observe_list_result(
        self: &Arc<Self>,
        current_state: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Result<AgentObserveListResult, String> {
        let current_active_session_id = current_state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        let mut agents: Vec<AgentObserveAgentSummary> = self
            .list_targetable_session_states(current_state)
            .into_iter()
            .filter(|state| {
                state
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    == current_active_session_id
                    || self.is_agent_family_reachable(current_state, state)
            })
            .map(|state| self.create_agent_observe_summary(&state, current_state))
            .collect();
        let mut resident_ids: HashSet<String> = agents
            .iter()
            .map(|agent| agent.active_session_id.clone())
            .collect();
        for passive in self.list_passive_rlm_subagents(Vec::new(), false).await {
            if resident_ids.contains(&passive.info.id) {
                continue;
            }
            match assert_agent_family_reach(
                &self.agent_family_entry(current_state),
                &self.passive_agent_family_entry(&passive),
            ) {
                Ok(_) => {}
                Err(error) if error == AGENT_FAMILY_REACH_ERROR => continue,
                Err(error) => return Err(error),
            }
            let root_parent_active_session_id = match &passive.root {
                PassiveRlmRoot::Resident(state) => Some(
                    state
                        .lock()
                        .expect("active session poisoned")
                        .active_session_id
                        .clone(),
                ),
                PassiveRlmRoot::Saved(_) => None,
            };
            let parent_active_session_id = if passive.chain.len() == 1 {
                root_parent_active_session_id
            } else {
                None
            };
            let name = passive
                .info
                .name
                .clone()
                .or_else(|| Some(passive.entry.session_name.clone()));
            agents.push(AgentObserveAgentSummary {
                active_session_id: passive.info.id.clone(),
                session_id: passive.info.id.clone(),
                name: name.clone(),
                session_name: name,
                runtime_kind: Some(RUNTIME_KIND_SUBAGENT.to_string()),
                cwd: passive.info.cwd.clone(),
                status: FAMILY_STATUS_IDLE.to_string(),
                is_session_active: false,
                message_count: passive.info.message_count as f64,
                parent_active_session_id,
                parent_session_id: Some(passive.entry.parent_session_id.clone()),
                rlm_child_id: Some(passive.entry.child_id.clone()),
                rlm_parent_node_id: passive.entry.rlm_parent_node_id.clone(),
                ..AgentObserveAgentSummary::default()
            });
            resident_ids.insert(passive.info.id.clone());
        }
        let current = self.create_agent_observe_summary(current_state, current_state);
        Ok(AgentObserveListResult { current, agents })
    }

    /// `setStateSessionNameViaSupervisor(state, name)`.
    ///
    /// Without a supervisor socket path (or outside a worker) the local setter
    /// is authoritative, exactly as the TS fallback does.
    async fn set_state_session_name_via_supervisor(
        self: &Arc<Self>,
        state: &Arc<StdMutex<ActiveSessionState>>,
        name: &str,
    ) -> Result<(), String> {
        let supervisor_socket_path = self.supervisor_socket_path_from_env();
        if !self.is_worker() || supervisor_socket_path.is_none() {
            return self.set_state_session_name(state, name).await;
        }
        // The supervisor request (`{type:"set_session_name", ...}`) is served by
        // the supervisor slice; until it lands the local setter answers.
        self.set_state_session_name(state, name).await
    }

    /// `createAgentMessageController(getCurrentState)` (`daemon-mode.ts:3363-3386`).
    fn create_agent_message_controller(
        self: &Arc<Self>,
        get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
    ) -> Arc<dyn AgentSessionMessageController> {
        Arc::new(DaemonAgentMessageController {
            daemon: Arc::clone(self),
            get_current_state,
        })
    }

    /// The `rlmHeartbeatController` literal from `daemon-mode.ts:1968-1993`.
    fn create_rlm_heartbeat_controller(
        self: &Arc<Self>,
        get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
    ) -> Arc<dyn crate::core::cron_jobs::AgentRlmHeartbeatController> {
        Arc::new(DaemonAgentRlmHeartbeatController {
            daemon: Arc::clone(self),
            get_current_state,
        })
    }

    /// `createAgentObserveController(getCurrentState)`.
    fn create_agent_observe_controller(
        self: &Arc<Self>,
        get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
    ) -> Arc<dyn AgentObserveController> {
        Arc::new(DaemonAgentObserveController {
            daemon: Arc::clone(self),
            get_current_state,
        })
    }
}

/// The `AgentObserveController` the daemon hands to a session runtime.
struct DaemonAgentObserveController {
    daemon: Arc<AgentDaemon>,
    get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
}

impl DaemonAgentObserveController {
    fn require_current_state(&self) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        (self.get_current_state)()
            .ok_or_else(|| "Agent observe state is not ready for this session yet".to_string())
    }
}

impl AgentObserveController for DaemonAgentObserveController {
    fn list_agents(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentObserveListResult, String>> + Send>,
    > {
        let daemon = Arc::clone(&self.daemon);
        let current = self.require_current_state();
        Box::pin(async move {
            let current = match current {
                Ok(current) => current,
                Err(error) => return Err(error),
            };
            daemon.create_agent_observe_list_result(&current).await
        })
    }

    fn get_agent(
        &self,
        target: String,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentObserveAgentSnapshot, String>> + Send>,
    > {
        let daemon = Arc::clone(&self.daemon);
        let current = self.require_current_state();
        Box::pin(async move {
            let current = match current {
                Ok(current) => current,
                Err(error) => return Err(error),
            };
            daemon
                .create_agent_observe_agent_snapshot(&current, &target)
                .await
        })
    }

    fn recent_messages(
        &self,
        input: AgentObserveRecentMessagesInput,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<AgentObserveRecentMessagesResult, String>>
                + Send,
        >,
    > {
        let daemon = Arc::clone(&self.daemon);
        let current = self.require_current_state();
        Box::pin(async move {
            let current = match current {
                Ok(current) => current,
                Err(error) => return Err(error),
            };
            daemon
                .create_agent_observe_recent_messages(&current, input)
                .await
        })
    }
}

/// The `AgentSessionMessageController` the daemon hands to a session runtime
/// (`daemon-mode.ts:3363-3386`).
struct DaemonAgentMessageController {
    daemon: Arc<AgentDaemon>,
    get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
}

impl DaemonAgentMessageController {
    /// `requireCurrentState()` (`daemon-mode.ts:3366-3372`).
    fn require_current_state(&self) -> Result<Arc<StdMutex<ActiveSessionState>>, String> {
        (self.get_current_state)()
            .ok_or_else(|| "Agent message state is not ready for this session yet".to_string())
    }
}

impl AgentSessionMessageController for DaemonAgentMessageController {
    fn list_agents(&self) -> pi_ai::types::BoxFuture<Result<Option<AgentSessionMessageListResult>, String>> {
        let daemon = self.daemon.clone();
        let current = self.require_current_state();
        Box::pin(async move { Ok(Some(daemon.create_agent_message_list_result(&current?, None).await)) })
    }

    /// `roster: () => this.createAgentFamilyRoster(requireCurrentState())`
    /// (`daemon-mode.ts:3375`).
    fn roster(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentFamilyRosterResult, String>> + Send>,
    > {
        let daemon = Arc::clone(&self.daemon);
        let current = self.require_current_state();
        Box::pin(async move {
            let current = current?;
            daemon.create_agent_family_roster(&current).await
        })
    }

    /// The TypeScript daemon controller has no `awaitPendingChildPublication`
    /// member (`daemon-mode.ts:3373-3385`); the session supplies it at the wrapper
    /// (`agent-session.ts:10278`), and its absence makes the call site treat the
    /// publication as unresolved - `awaitPromise` is skipped and `publishedId`
    /// stays `undefined` (`core/agent-messages.ts:593-596`). `None` is that value.
    ///
    /// UNRESOLVED for the session side: `AgentSession::await_pending_rlm_child_publication`
    /// (`core/agent_session/runtime_members.rs:924`, TS `_awaitPendingRlmChildPublication`
    /// at `agent-session.ts:10641-10651`) owns the real lookup, and it is not exposed
    /// on the `DaemonSession` seam (`daemon_mode.rs:3080-3200`), so this file cannot
    /// forward to it. Add `await_pending_child_publication` to `DaemonSession` and
    /// forward here once that owner exposes it.
    fn await_pending_child_publication(
        &self,
        _selector: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send>>
    {
        Box::pin(async move { Ok(None) })
    }

    /// `sendAgentMessage: (input) => this.sendAgentSessionMessage({
    /// targetSelector: input.target, message: input.message,
    /// fromState: requireCurrentState(), origin: "agent" })`
    /// (`daemon-mode.ts:3378-3384`).
    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentSessionMessageReceipt, String>> + Send>,
    > {
        let daemon = Arc::clone(&self.daemon);
        let current = self.require_current_state();
        Box::pin(async move {
            let from_state = current?;
            daemon
                .send_agent_session_message(SendAgentMessageInput {
                    target_selector: input.target,
                    message: input.message,
                    from_state: Some(from_state),
                    sender: None,
                    client_id: None,
                    sender_key: None,
                    origin: "agent".to_string(),
                })
                .await
        })
    }
}

/// The `rlmHeartbeatController` literal handed to the session runtime
/// (`daemon-mode.ts:1968-1993`): every member reads the current state and
/// forwards to the daemon's `cronStore`-backed helpers.
struct DaemonAgentRlmHeartbeatController {
    daemon: Arc<AgentDaemon>,
    get_current_state: Arc<dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>,
}

impl DaemonAgentRlmHeartbeatController {
    /// `if (!stateRef) throw new Error("RLM heartbeat state is not ready for this
    /// session yet")` (`daemon-mode.ts:1970-1972`, repeated at `:1976-1978`,
    /// `:1982-1984`, `:1988-1990`). `list_rlm_heartbeats`/`create_rlm_heartbeat`/
    /// `update_rlm_heartbeat`/`delete_rlm_heartbeat` return no `Result`
    /// (`core/cron_jobs.rs:186-191`), so the throw is mirrored with the identical
    /// message, like the store mirror at `core/cron_jobs.rs:1538-1541`. The guard
    /// is unreachable in practice: the controller is only reachable from a live
    /// session, whose state the daemon publishes before the first prompt.
    fn require_current_state(&self) -> Arc<StdMutex<ActiveSessionState>> {
        (self.get_current_state)()
            .unwrap_or_else(|| panic!("RLM heartbeat state is not ready for this session yet"))
    }
}

impl crate::core::cron_jobs::AgentRlmHeartbeatController for DaemonAgentRlmHeartbeatController {
    /// `listRlmHeartbeats: (options) => this.cronStore.listRlmHeartbeats(
    /// stateRef.activeSessionId, options)` (`daemon-mode.ts:1969-1974`).
    fn list_rlm_heartbeats(
        &self,
        options: Option<crate::core::cron_jobs::RlmHeartbeatListOptions>,
    ) -> Vec<AgentCronJob> {
        let state = self.require_current_state();
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        self.daemon
            .cron_store
            .list_rlm_heartbeats(&active_session_id, options)
    }

    /// `createRlmHeartbeat: (input) => this.createRlmHeartbeatForState(stateRef,
    /// input)` (`daemon-mode.ts:1975-1980`). `createRlmHeartbeatForState` throws
    /// on a missing session file or a rejected schedule (`daemon-mode.ts:2187-2191`,
    /// `core/cron-jobs.ts:365-371`), which the store reports as `Err`; the trait
    /// member cannot return an error, so the message is raised as a panic.
    fn create_rlm_heartbeat(
        &self,
        input: RlmHeartbeatCreateInput,
    ) -> AgentCronJob {
        let state = self.require_current_state();
        match self.daemon.create_rlm_heartbeat_for_state(&state, &input) {
            Ok(job) => job,
            Err(error) => panic!("{error}"),
        }
    }

    /// `updateRlmHeartbeat: (input) => this.updateRlmHeartbeatForState(stateRef,
    /// input)` (`daemon-mode.ts:1981-1986`); the TS signature is
    /// `AgentCronJob | undefined` (`core/cron-jobs.ts:134`).
    fn update_rlm_heartbeat(
        &self,
        input: RlmHeartbeatUpdateInput,
    ) -> Option<AgentCronJob> {
        let state = self.require_current_state();
        self.daemon.update_rlm_heartbeat_for_state(&state, &input)
    }

    /// `deleteRlmHeartbeat: (id) => this.deleteRlmHeartbeatForState(stateRef, id)`
    /// (`daemon-mode.ts:1987-1992`); `AgentCronJob | undefined`
    /// (`core/cron-jobs.ts:135`).
    fn delete_rlm_heartbeat(&self, id: &str) -> Option<AgentCronJob> {
        let state = self.require_current_state();
        self.daemon.delete_rlm_heartbeat_for_state(&state, id)
    }
}

#[cfg(test)]
mod cron_error_propagation_tests {
    use super::*;

    /// The cron store coordinates every write through one process-wide lock with a 1s
    /// budget (cron-jobs.rs:297-343), so these tests must not drive the store from two
    /// threads at once; otherwise one fails on lock contention instead of on its assertion.
    fn cron_store_lock() -> &'static StdMutex<()> {
        static LOCK: std::sync::OnceLock<StdMutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    /// A daemon session whose queued prompt fails on demand. Everything else delegates to
    /// `MissingSession` (the daemon's own not-resident double), so this only changes the
    /// behaviour under test.
    struct FailingQueuedPromptSession {
        inner: MissingSession,
        active_session_id: String,
        session_file: String,
        fail_follow_up: bool,
        follow_up_calls: Arc<AtomicU64>,
    }

    impl DaemonSession for FailingQueuedPromptSession {
        fn session_id(&self) -> String {
            self.active_session_id.clone()
        }
        fn session_name(&self) -> Option<String> {
            Some("cron-error-test".to_string())
        }
        fn session_file(&self) -> Option<String> {
            Some(self.session_file.clone())
        }
        fn is_session_active(&self) -> bool {
            true
        }
        /// `shouldQueueCronPrompt` reads `session.isStreaming` (daemon-mode.ts:2054-2059), so
        /// the job must take the queued-prompt path the bug drops.
        fn is_streaming(&self) -> bool {
            true
        }
        fn unfinished_action_count(&self) -> f64 {
            0.0
        }
        /// `session.followUp(prompt, undefined, { resumeIfIdle: true })` (daemon-mode.ts:2064).
        fn follow_up(
            &self,
            _message: &str,
            _images: Option<Value>,
            _options: PromptInvocation,
        ) -> BoxFuture<'static, Result<bool, String>> {
            self.follow_up_calls.fetch_add(1, Ordering::SeqCst);
            let fail = self.fail_follow_up;
            Box::pin(async move {
                if fail {
                    Err("cron queued prompt exploded".to_string())
                } else {
                    Ok(true)
                }
            })
        }

    fn session_manager(&self) -> Arc<StdMutex<SessionManager>> {
        self.inner.session_manager()
    }
    fn runtime(&self) -> Arc<dyn DaemonRuntimeApi> {
        self.inner.runtime()
    }
    fn settings_manager(&self) -> Option<Arc<StdMutex<SettingsManager>>> {
        self.inner.settings_manager()
    }
    fn session_dir(&self) -> Option<String> {
        self.inner.session_dir()
    }
    fn set_exec_env_provider(&self, client_env: Option<HashMap<String, String>>) {
        self.inner.set_exec_env_provider(client_env)
    }
    fn set_runtime_env_scope(&self, client_env: Option<HashMap<String, String>>) {
        self.inner.set_runtime_env_scope(client_env)
    }
    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {
        self.inner.set_subagent_runtime_host(host)
    }
    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {
        self.inner.set_rebind_session(rebind)
    }
    fn bind_extensions( &self, binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.bind_extensions(binding)
    }
    fn abort_for_update_restart(&self) {
        self.inner.abort_for_update_restart()
    }
    fn is_compacting(&self) -> bool {
        self.inner.is_compacting()
    }
    fn is_bash_running(&self) -> bool {
        self.inner.is_bash_running()
    }
    fn is_retrying(&self) -> bool {
        self.inner.is_retrying()
    }
    fn has_running_rlm_children(&self) -> bool {
        self.inner.has_running_rlm_children()
    }
    fn messages(&self) -> Vec<AgentMessage> {
        self.inner.messages()
    }
    fn model_identity(&self) -> Option<pi_ai::types::Model> {
        self.inner.model_identity()
    }
    fn rlm_depth(&self) -> Option<i64> {
        self.inner.rlm_depth()
    }
    fn thinking_level(&self) -> Option<String> {
        self.inner.thinking_level()
    }
    fn service_tier(&self) -> Option<String> {
        self.inner.service_tier()
    }
    fn system_prompt(&self) -> Option<String> {
        self.inner.system_prompt()
    }
    fn connection_view(&self) -> DaemonConnectionView {
        self.inner.connection_view()
    }
    fn connection_state(&self, active_session_id: Option<String>) -> Value {
        self.inner.connection_state(active_session_id)
    }
    fn set_current_recap(&self, recap: Option<&str>) {
        self.inner.set_current_recap(recap)
    }
    fn set_session_name(&self, name: &str) {
        self.inner.set_session_name(name)
    }
    fn get_rlm_child_run_status(&self, child_id: &str) -> Option<String> {
        self.inner.get_rlm_child_run_status(child_id)
    }
    fn register_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> bool {
        self.inner.register_rlm_child_session(child_id, session)
    }
    fn remove_queued_follow_up(&self, key: &str) {
        self.inner.remove_queued_follow_up(key)
    }
    fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.inner.subscribe(listener)
    }
    fn prompt_until_accepted( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.prompt_until_accepted(message, options)
    }
    fn prompt_and_wait( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.prompt_and_wait(message, options)
    }
    fn prompt_heartbeat( &self, job: &AgentCronJob, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.prompt_heartbeat(job, options)
    }
    fn accept_agent_message_prompt( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.accept_agent_message_prompt(message, options)
    }
    fn steer( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.steer(message, images, options)
    }
    fn restore_steering_message( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.restore_steering_message(message, images, options)
    }
    fn restore_follow_up_message( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<bool, String>> {
        self.inner.restore_follow_up_message(message, images, options)
    }
    fn restore_pending_next_turn_messages(&self, messages: &Value) {
        self.inner.restore_pending_next_turn_messages(messages)
    }
    fn restore_session_actions(&self, snapshot: &Value) -> BoxFuture<'static, Result<f64, String>> {
        self.inner.restore_session_actions(snapshot)
    }
    fn send_custom_message(&self, message: &Value) -> BoxFuture<'static, Result<(), String>> {
        self.inner.send_custom_message(message)
    }
    fn resume_queued_work(&self) -> bool {
        self.inner.resume_queued_work()
    }
    fn clear_queued_agent_messages(&self) -> Value {
        self.inner.clear_queued_agent_messages()
    }
    fn clear_queue(&self) -> Value {
        self.inner.clear_queue()
    }
    fn mutate_queued_message( &self, lane: &str, index: f64, expected_text: &str, mutation: &Value, ) -> Value {
        self.inner.mutate_queued_message(lane, index, expected_text, mutation)
    }
    fn get_steering_message_previews(&self) -> Vec<Value> {
        self.inner.get_steering_message_previews()
    }
    fn get_follow_up_message_previews(&self) -> Vec<Value> {
        self.inner.get_follow_up_message_previews()
    }
    fn request_abort(&self) {
        self.inner.request_abort()
    }
    fn cancel_rlm_child_run(&self, child_id: &str) -> bool {
        self.inner.cancel_rlm_child_run(child_id)
    }
    fn delete_inactive_rlm_subagent( &self, child_id: &str, is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>, ) -> BoxFuture<'static, Result<String, String>> {
        self.inner.delete_inactive_rlm_subagent(child_id, is_resident_child_running)
    }
    fn run_user_bash( &self, command: &str, options: RunUserBashOptions, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.run_user_bash(command, options)
    }
    fn execute_bash(&self, command: &str) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.execute_bash(command)
    }
    fn abort_bash(&self) {
        self.inner.abort_bash()
    }
    fn acquire_session_input_pause(&self) -> SessionInputPause {
        self.inner.acquire_session_input_pause()
    }
    fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        self.inner.wait_for_idle()
    }
    fn wait_for_headless_completion( &self, options: HeadlessCompletionOptions, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.wait_for_headless_completion(options)
    }
    fn refresh_available_models(&self) -> BoxFuture<'static, Result<Vec<pi_ai::types::Model>, String>> {
        self.inner.refresh_available_models()
    }
    fn refresh_model_catalog(&self) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.refresh_model_catalog()
    }
    fn get_provider_auth_status_source(&self, provider: &str) -> Option<String> {
        self.inner.get_provider_auth_status_source(provider)
    }
    fn find_model(&self, provider: &str, model_id: &str) -> Option<pi_ai::types::Model> {
        self.inner.find_model(provider, model_id)
    }
    fn set_model( &self, model: &pi_ai::types::Model, wait_for_extensions: bool, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.set_model(model, wait_for_extensions)
    }
    fn cycle_model( &self, direction: &str, wait_for_extensions: bool, ) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>> {
        self.inner.cycle_model(direction, wait_for_extensions)
    }
    fn set_scoped_models(&self, scoped_models: &Value) {
        self.inner.set_scoped_models(scoped_models)
    }
    fn set_thinking_level(&self, level: &str) {
        self.inner.set_thinking_level(level)
    }
    fn set_service_tier(&self, service_tier: &str) {
        self.inner.set_service_tier(service_tier)
    }
    fn cycle_thinking_level(&self) -> Option<String> {
        self.inner.cycle_thinking_level()
    }
    fn set_transport(&self, transport: &str) {
        self.inner.set_transport(transport)
    }
    fn set_steering_mode(&self, mode: &str) {
        self.inner.set_steering_mode(mode)
    }
    fn set_follow_up_mode(&self, mode: &str) {
        self.inner.set_follow_up_mode(mode)
    }
    fn set_auto_compaction_enabled(&self, enabled: bool) {
        self.inner.set_auto_compaction_enabled(enabled)
    }
    fn set_auto_retry_enabled(&self, enabled: bool) {
        self.inner.set_auto_retry_enabled(enabled)
    }
    fn compact( &self, custom_instructions: Option<&str>, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.compact(custom_instructions)
    }
    fn refine(&self, options: RefineOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.refine(options)
    }
    fn abort_compaction(&self) {
        self.inner.abort_compaction()
    }
    fn abort_branch_summary(&self) {
        self.inner.abort_branch_summary()
    }
    fn abort_retry(&self) {
        self.inner.abort_retry()
    }
    fn reload(&self) -> BoxFuture<'static, Result<(), String>> {
        self.inner.reload()
    }
    fn get_rlm_max_depth_status(&self) -> Value {
        self.inner.get_rlm_max_depth_status()
    }
    fn set_rlm_max_depth( &self, max_depth: Value, global: bool, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.set_rlm_max_depth(max_depth, global)
    }
    fn build_session_context(&self) -> Value {
        self.inner.build_session_context()
    }
    fn get_session_stats(&self) -> Value {
        self.inner.get_session_stats()
    }
    fn get_context_tree(&self) -> Value {
        self.inner.get_context_tree()
    }
    fn get_rlm_child_snapshots(&self) -> Vec<Value> {
        self.inner.get_rlm_child_snapshots()
    }
    fn export_to_html( &self, output_path: Option<&str>, ) -> BoxFuture<'static, Result<String, String>> {
        self.inner.export_to_html(output_path)
    }
    fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String> {
        self.inner.export_to_jsonl(output_path)
    }
    fn get_user_messages_for_forking(&self) -> Vec<Value> {
        self.inner.get_user_messages_for_forking()
    }
    fn get_last_assistant_text(&self) -> String {
        self.inner.get_last_assistant_text()
    }
    fn get_tool_definition(&self, name: &str) -> Option<Value> {
        self.inner.get_tool_definition(name)
    }
    fn navigate_tree( &self, target_id: &str, options: NavigateTreeOptions, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.navigate_tree(target_id, options)
    }
    fn start_side_question( &self, question: &str, options: SideQuestionOptions, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.start_side_question(question, options)
    }
    fn abort_side_question(&self, side_question_id: &str) {
        self.inner.abort_side_question(side_question_id)
    }
    fn release_acp_mcp_servers( &self, owner_id: &str, server_names: &[String], ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.release_acp_mcp_servers(owner_id, server_names)
    }
    fn replace_acp_mcp_servers( &self, servers: &[Value], owner_id: &str, ) -> BoxFuture<'static, Result<(), String>> {
        self.inner.replace_acp_mcp_servers(servers, owner_id)
    }
    fn new_session( &self, options: Option<NewSessionRuntimeOptions>, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.new_session(options)
    }
    fn release_rlm_child_session( &self, child_id: &str, session: Arc<dyn DaemonSession>, ) -> Option<Box<dyn FnOnce() + Send>> {
        self.inner.release_rlm_child_session(child_id, session)
    }
    fn replied_to_parent_since_task(&self) -> Option<bool> {
        self.inner.replied_to_parent_since_task()
    }
    fn switch_session( &self, session_path: &str, options: SessionPathOptions, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.switch_session(session_path, options)
    }
    fn fork( &self, entry_id: &str, options: ForkOptions, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.fork(entry_id, options)
    }
    fn import_from_jsonl( &self, input_path: &str, cwd_override: Option<&str>, ) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.import_from_jsonl(input_path, cwd_override)
    }
    fn dispose(&self) -> BoxFuture<'static, ()> {
        self.inner.dispose()
    }
    }

    /// A daemon with one resident session and one due cron job, driven only through
    /// `AgentDaemon::run_cron_job` (the `runJob` the scheduler calls).
    struct CronErrorFixture {
        _directory: tempfile::TempDir,
        daemon: Arc<AgentDaemon>,
        job: AgentCronJob,
        on_error: Arc<StdMutex<Vec<String>>>,
        follow_up_calls: Arc<AtomicU64>,
    }

    fn cron_error_fixture(fail_follow_up: bool) -> CronErrorFixture {
        let directory = tempfile::tempdir().expect("temp dir");
        let agent_dir = directory.path().join("agent");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let session_file = directory.path().join("session.jsonl");
        std::fs::write(&session_file, "{}\n").expect("session file");
        let socket_path = directory.path().join("daemon.sock").to_string_lossy().into_owned();
        let daemon = AgentDaemon::new(
            socket_path.clone(),
            DaemonModeOptions {
                socket_path: Some(socket_path),
                default_session_config: AgentSessionRuntimeConfig {
                    cwd: Some(directory.path().to_string_lossy().into_owned()),
                    agent_dir: Some(agent_dir.to_string_lossy().into_owned()),
                    ..Default::default()
                },
                create_runtime: Arc::new(|_| {
                    Box::pin(async { panic!("the cron error test must not create a runtime") })
                }),
                worker: None,
            },
        );
        let active_session_id = "cron-error-session".to_string();
        let session_file_string = session_file.to_string_lossy().into_owned();
        let follow_up_calls = Arc::new(AtomicU64::new(0));
        let session: Arc<dyn DaemonSession> = Arc::new(FailingQueuedPromptSession {
            inner: MissingSession::new(&active_session_id),
            active_session_id: active_session_id.clone(),
            session_file: session_file_string.clone(),
            fail_follow_up,
            follow_up_calls: Arc::clone(&follow_up_calls),
        });
        let state = Arc::new(StdMutex::new(ActiveSessionState::new(
            active_session_id.clone(),
            AgentSessionRuntime {
                session: ActiveSessionRuntimeSession {
                    session_id: active_session_id.clone(),
                    session_file: Some(session_file_string.clone()),
                    ..ActiveSessionRuntimeSession::default()
                },
                metadata: Some(AgentSessionRuntimeMetadata::default()),
                model_fallback_message: None,
            },
        )));
        daemon.sessions.lock().expect("sessions poisoned").insert(
            active_session_id.clone(),
            Arc::new(DaemonSessionState {
                state: Arc::clone(&state),
                session: Arc::clone(&session),
                runtime_metadata: AgentSessionRuntimeMetadata::default(),
                snapshot_boundary: StdMutex::new(None),
            }),
        );
        let job = daemon
            .cron_store
            .create(&CreateAgentCronJobInput {
                active_session_id: active_session_id.clone(),
                session_id: active_session_id,
                session_file: session_file_string,
                // `DaemonSessionState::cwd()` for this double is `MissingSession`'s
                // in-memory manager cwd ("."), so matching it keeps `rebindSessionJobs`
                // (daemon-mode.ts:2045) from rewriting the job during the run.
                cwd: ".".to_string(),
                prompt: "check the build".to_string(),
                schedule_text: "every 10m".to_string(),
                now: Some(0.0),
                ..CreateAgentCronJobInput::default()
            })
            .expect("cron job");
        CronErrorFixture {
            _directory: directory,
            daemon,
            job,
            on_error: Arc::new(StdMutex::new(Vec::new())),
            follow_up_calls,
        }
    }

    impl CronErrorFixture {
        /// Install the scheduler the daemon builds in `start_cron_scheduler`, with an
        /// `on_error` sink in place of the daemon log file.
        fn scheduler(&self) -> Arc<AgentCronScheduler> {
            let daemon = Arc::clone(&self.daemon);
            let sink = Arc::clone(&self.on_error);
            let hooks = Arc::new(crate::core::cron_jobs::AgentCronSchedulerHooks {
                run_job: Arc::new(move |job: AgentCronJob| {
                    let daemon = Arc::clone(&daemon);
                    Box::pin(async move { daemon.run_cron_job(job).await.ok().flatten() })
                }),
                begin_dispatch: None,
                now: None,
                on_error: Some(Arc::new(move |_job: &AgentCronJob, error: String| {
                    sink.lock().expect("on_error sink").push(error);
                })),
            });
            let scheduler = Arc::new(AgentCronScheduler::new(
                Arc::clone(&self.daemon.cron_store),
                hooks,
            ));
            let daemon = Arc::clone(&self.daemon);
            scheduler.enable_run_job_errors(Arc::new(move |job: AgentCronJob| {
                let daemon = Arc::clone(&daemon);
                Box::pin(async move { daemon.run_cron_job(job).await })
            }));
            scheduler
        }

        async fn run_due(&self) -> Result<usize, String> {
            self.scheduler().run_due(Some(600_000.0)).await
        }

        fn recorded_job(&self) -> AgentCronJob {
            self.daemon
                .cron_store
                .list()
                .into_iter()
                .find(|candidate| candidate.id == self.job.id)
                .expect("job still listed")
        }
    }

    /// The half-landed fix: a daemon queued-prompt failure must reach `onError`/`lastError`
    /// instead of being dropped. `await session.followUp(...)` (daemon-mode.ts:2064-2066)
    /// runs inside `runCronJob`'s try (2085-2105), and its catch rethrows everything that is
    /// not the `unrunnableAtAdmission` sentinel (2100-2105); the scheduler records that
    /// rejection (cron-jobs.ts:1010-1020, 739).
    #[tokio::test]
    async fn cron_queued_prompt_failure_reaches_on_error_and_last_error() {
        let _serialize = cron_store_lock().lock().expect("cron store lock");
        let fixture = cron_error_fixture(true);
        fixture.run_due().await.expect("run due");
        assert_eq!(
            fixture.follow_up_calls.load(Ordering::SeqCst),
            1,
            "the failing queued prompt was never attempted, so this test is not on the bug path"
        );
        assert_eq!(
            fixture.on_error.lock().expect("on_error sink").as_slice(),
            ["cron queued prompt exploded"],
            "the daemon dropped the queued-prompt error instead of reporting it"
        );
        assert_eq!(
            fixture.recorded_job().last_error.as_deref(),
            Some("cron queued prompt exploded"),
            "lastError was not persisted for the failed daemon cron run"
        );
    }

    /// The same path with a healthy queued prompt must stay silent and count the run.
    #[tokio::test]
    async fn cron_queued_prompt_success_reports_no_error() {
        let _serialize = cron_store_lock().lock().expect("cron store lock");
        let fixture = cron_error_fixture(false);
        fixture.run_due().await.expect("run due");
        assert_eq!(fixture.follow_up_calls.load(Ordering::SeqCst), 1);
        assert!(
            fixture.on_error.lock().expect("on_error sink").is_empty(),
            "a successful queued prompt must not report an error"
        );
        let recorded = fixture.recorded_job();
        assert_eq!(recorded.last_error, None);
        assert_eq!(recorded.run_count, 1.0);
    }
}

/// T10 lane `rlm-daemon`: agent-observe parity for findings D-06 and D-07.
///
/// Both findings sit on the daemon's observe path, which is only reachable in
/// crate: the controller and the summary builder are private to this module. The
/// tests below therefore drive the REAL host handlers production installs
/// (`create_agent_observe_host_handlers`, `core/agent_observe.rs:139`) over a REAL
/// `DaemonAgentObserveController` built by the daemon's own factory.
#[cfg(test)]
mod agent_observe_parity_tests {
    use super::*;

    /// The TypeScript defaults are 8 messages and 800 chars
    /// (`agent-observe.ts:107-112`), which the daemon call site must use.
    const TS_DEFAULT_LIMIT: i64 = 8;
    const TS_DEFAULT_MAX_CHARS: i64 = 800;

    fn user_message(text: &str, timestamp: i64) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text(text.to_string()),
            timestamp,
        )))
    }

    fn assistant_message(text: &str, timestamp: i64) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::Assistant(pi_ai::types::AssistantMessage {
            content: vec![pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(text))],
            timestamp,
            ..Default::default()
        }))
    }

    /// A scripted `AgentHandle` for a real `AgentSession`.
    struct ScriptedAgent {
        state: StdMutex<pi_agent_core::types::AgentState>,
    }

    impl ScriptedAgent {
        fn new(state: pi_agent_core::types::AgentState) -> Arc<Self> {
            Arc::new(Self { state: StdMutex::new(state) })
        }
    }

    impl crate::core::agent_session::AgentHandle for ScriptedAgent {
        fn state(&self) -> pi_agent_core::types::AgentState {
            self.state.lock().unwrap().clone()
        }
        fn set_state(&self, state: pi_agent_core::types::AgentState) {
            *self.state.lock().unwrap() = state;
        }
        fn subscribe(
            &self,
            _listener: Arc<
                dyn Fn(
                        pi_agent_core::types::AgentEvent,
                        Option<tokio_util::sync::CancellationToken>,
                    ) -> pi_ai::types::BoxFuture<()>
                    + Send
                    + Sync,
            >,
        ) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
        fn set_before_tool_call(&self, _hook: crate::core::agent_session::BeforeToolCallHook) {}
        fn set_after_tool_call(&self, _hook: crate::core::agent_session::AfterToolCallHook) {}
        fn set_get_continuation_messages(
            &self,
            _hook: crate::core::agent_session::GetContinuationMessagesHook,
        ) {
        }
        fn set_before_request(
            &self,
            _hook: crate::core::agent_session::BeforeRequestHook,
        ) {
        }
        fn set_should_stop_before_turn(&self, _hook: Arc<dyn Fn() -> bool + Send + Sync>) {}
        fn set_should_stop_after_turn(
            &self,
            _hook: Arc<
                dyn Fn(pi_agent_core::types::ShouldStopAfterTurnContext) -> pi_ai::types::BoxFuture<bool>
                    + Send
                    + Sync,
            >,
        ) {
        }
        fn set_stream_fn(&self, _stream_fn: pi_agent_core::types::StreamFn) {}
        fn stream_fn(&self) -> pi_agent_core::types::StreamFn {
            pi_agent_core::agent::default_stream_fn()
        }
        fn abort(&self) {}
        fn wait_for_idle(&self) -> pi_ai::types::BoxFuture<()> {
            Box::pin(async {})
        }
        fn prompt(&self, _messages: Vec<AgentMessage>) -> pi_ai::types::BoxFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
        fn continue_(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<(), pi_agent_core::agent::AgentContinueError>> {
            Box::pin(async { Ok(()) })
        }
        fn is_streaming(&self) -> bool {
            self.state.lock().unwrap().is_streaming
        }
        fn has_queued_messages(&self) -> bool {
            false
        }
        fn clear_all_queues(&self) {}
        fn remove_queued_messages(
            &self,
            _predicate: Arc<dyn Fn(&AgentMessage) -> bool + Send + Sync>,
        ) -> Vec<AgentMessage> {
            Vec::new()
        }
        fn follow_up(&self, _message: AgentMessage) {}
        fn set_follow_up_mode(&self, _mode: String) {}
        fn set_steering_mode(&self, _mode: String) {}
        fn set_convert_to_llm(
            &self,
            _convert: Arc<
                dyn Fn(Vec<AgentMessage>) -> pi_ai::types::BoxFuture<Vec<pi_ai::types::Message>>
                    + Send
                    + Sync,
            >,
        ) {
        }
        fn set_transform_context(
            &self,
            _transform: Arc<
                dyn Fn(
                        Vec<AgentMessage>,
                        Option<tokio_util::sync::CancellationToken>,
                    ) -> pi_ai::types::BoxFuture<Vec<AgentMessage>>
                    + Send
                    + Sync,
            >,
        ) {
        }
        fn set_get_api_key(
            &self,
            _get_api_key: Arc<dyn Fn(String) -> pi_ai::types::BoxFuture<Option<String>> + Send + Sync>,
        ) {
        }
        fn set_on_payload(&self, _hook: pi_ai::types::OnPayload) {}
        fn set_on_response(&self, _hook: pi_ai::types::OnResponse) {}
        fn set_tool_execution(&self, _mode: String) {}
        fn performance_metrics(
            &self,
        ) -> Option<pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics> {
            None
        }
        fn set_performance_metrics(
            &self,
            _metrics: Option<pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics>,
        ) {
        }
        fn signal(&self) -> Option<tokio_util::sync::CancellationToken> {
            None
        }
    }

    /// One resident daemon session whose `agent_session()` seam exposes the real
    /// `AgentSession`. Everything the tests do not override delegates to
    /// `MissingSession`, the daemon's own not-resident double, so this double only
    /// changes the surface under test.
    struct ObserveSession {
        inner: MissingSession,
        active_session_id: String,
        session_file: String,
        agent_session: Option<Arc<crate::core::agent_session::AgentSession>>,
        streaming: bool,
        session_active: bool,
        foreground_active: Option<bool>,
    }

    impl DaemonSession for ObserveSession {
        fn agent_session(&self) -> Option<Arc<crate::core::agent_session::AgentSession>> {
            self.agent_session.clone()
        }
        fn session_id(&self) -> String {
            self.active_session_id.clone()
        }
        fn session_name(&self) -> Option<String> {
            Some("observe-parity".to_string())
        }
        fn session_file(&self) -> Option<String> {
            Some(self.session_file.clone())
        }
        fn is_streaming(&self) -> bool {
            self.streaming
        }
        fn is_session_active(&self) -> bool {
            self.session_active
        }
        fn is_foreground_active(&self) -> bool {
            self.foreground_active.unwrap_or(self.session_active)
        }
        fn unfinished_action_count(&self) -> f64 {
            0.0
        }
        fn messages(&self) -> Vec<AgentMessage> {
            match &self.agent_session {
                Some(session) => session.messages(),
                None => Vec::new(),
            }
        }
        fn message_count(&self) -> usize {
            self.agent_session.as_ref().map(|session| session.message_count()).unwrap_or(0)
        }
        fn model_identity(&self) -> Option<pi_ai::types::Model> {
            self.agent_session.as_ref().and_then(|session| session.model())
        }
        fn rlm_depth(&self) -> Option<i64> {
            Some(self.agent_session.as_ref().map(|session| session.rlm_depth()).unwrap_or(0))
        }
        fn session_manager(&self) -> Arc<StdMutex<SessionManager>> {
            match &self.agent_session {
                Some(session) => Arc::clone(&session.session_manager),
                None => self.inner.session_manager(),
            }
        }
        fn settings_manager(&self) -> Option<Arc<StdMutex<SettingsManager>>> {
            self.agent_session
                .as_ref()
                .map(|session| Arc::clone(&session.settings_manager))
                .or_else(|| self.inner.settings_manager())
        }
        fn follow_up(
            &self,
            _message: &str,
            _images: Option<Value>,
            _options: PromptInvocation,
        ) -> BoxFuture<'static, Result<bool, String>> {
            Box::pin(async { Ok(true) })
        }
        fn runtime(&self) -> Arc<dyn DaemonRuntimeApi> {
            self.inner.runtime()
        }
        fn session_dir(&self) -> Option<String> {
            self.inner.session_dir()
        }
        fn set_exec_env_provider(&self, client_env: Option<HashMap<String, String>>) {
            self.inner.set_exec_env_provider(client_env)
        }
        fn set_runtime_env_scope(&self, client_env: Option<HashMap<String, String>>) {
            self.inner.set_runtime_env_scope(client_env)
        }
        fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {
            self.inner.set_subagent_runtime_host(host)
        }
        fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {
            self.inner.set_rebind_session(rebind)
        }
        fn bind_extensions( &self, binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.bind_extensions(binding)
        }
        fn abort_for_update_restart(&self) {
            self.inner.abort_for_update_restart()
        }
        fn is_compacting(&self) -> bool {
            self.inner.is_compacting()
        }
        fn is_bash_running(&self) -> bool {
            self.inner.is_bash_running()
        }
        fn is_retrying(&self) -> bool {
            self.inner.is_retrying()
        }
        fn has_running_rlm_children(&self) -> bool {
            self.inner.has_running_rlm_children()
        }
        fn thinking_level(&self) -> Option<String> {
            self.inner.thinking_level()
        }
        fn service_tier(&self) -> Option<String> {
            self.inner.service_tier()
        }
        fn system_prompt(&self) -> Option<String> {
            self.inner.system_prompt()
        }
        fn connection_view(&self) -> DaemonConnectionView {
            self.inner.connection_view()
        }
        fn connection_state(&self, active_session_id: Option<String>) -> Value {
            self.inner.connection_state(active_session_id)
        }
        fn set_current_recap(&self, recap: Option<&str>) {
            self.inner.set_current_recap(recap)
        }
        fn set_session_name(&self, name: &str) {
            self.inner.set_session_name(name)
        }
        fn get_rlm_child_run_status(&self, child_id: &str) -> Option<String> {
            self.inner.get_rlm_child_run_status(child_id)
        }
        fn register_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> bool {
            self.inner.register_rlm_child_session(child_id, session)
        }
        fn remove_queued_follow_up(&self, key: &str) {
            self.inner.remove_queued_follow_up(key)
        }
        fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
            self.inner.subscribe(listener)
        }
        fn prompt_until_accepted( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.prompt_until_accepted(message, options)
        }
        fn prompt_and_wait( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.prompt_and_wait(message, options)
        }
        fn prompt_heartbeat( &self, job: &AgentCronJob, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.prompt_heartbeat(job, options)
        }
        fn accept_agent_message_prompt( &self, message: &str, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.accept_agent_message_prompt(message, options)
        }
        fn steer( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.steer(message, images, options)
        }
        fn restore_steering_message( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.restore_steering_message(message, images, options)
        }
        fn restore_follow_up_message( &self, message: &str, images: Option<Value>, options: PromptInvocation, ) -> BoxFuture<'static, Result<bool, String>> {
            self.inner.restore_follow_up_message(message, images, options)
        }
        fn restore_pending_next_turn_messages(&self, messages: &Value) {
            self.inner.restore_pending_next_turn_messages(messages)
        }
        fn restore_session_actions(&self, snapshot: &Value) -> BoxFuture<'static, Result<f64, String>> {
            self.inner.restore_session_actions(snapshot)
        }
        fn send_custom_message(&self, message: &Value) -> BoxFuture<'static, Result<(), String>> {
            self.inner.send_custom_message(message)
        }
        fn resume_queued_work(&self) -> bool {
            self.inner.resume_queued_work()
        }
        fn clear_queued_agent_messages(&self) -> Value {
            self.inner.clear_queued_agent_messages()
        }
        fn clear_queue(&self) -> Value {
            self.inner.clear_queue()
        }
        fn mutate_queued_message( &self, lane: &str, index: f64, expected_text: &str, mutation: &Value, ) -> Value {
            self.inner.mutate_queued_message(lane, index, expected_text, mutation)
        }
        fn get_steering_message_previews(&self) -> Vec<Value> {
            self.inner.get_steering_message_previews()
        }
        fn get_follow_up_message_previews(&self) -> Vec<Value> {
            self.inner.get_follow_up_message_previews()
        }
        fn request_abort(&self) {
            self.inner.request_abort()
        }
        fn cancel_rlm_child_run(&self, child_id: &str) -> bool {
            self.inner.cancel_rlm_child_run(child_id)
        }
        fn delete_inactive_rlm_subagent( &self, child_id: &str, is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>, ) -> BoxFuture<'static, Result<String, String>> {
            self.inner.delete_inactive_rlm_subagent(child_id, is_resident_child_running)
        }
        fn run_user_bash( &self, command: &str, options: RunUserBashOptions, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.run_user_bash(command, options)
        }
        fn execute_bash(&self, command: &str) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.execute_bash(command)
        }
        fn abort_bash(&self) {
            self.inner.abort_bash()
        }
        fn acquire_session_input_pause(&self) -> SessionInputPause {
            self.inner.acquire_session_input_pause()
        }
        fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
            self.inner.wait_for_idle()
        }
        fn wait_for_headless_completion( &self, options: HeadlessCompletionOptions, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.wait_for_headless_completion(options)
        }
        fn refresh_available_models(&self) -> BoxFuture<'static, Result<Vec<pi_ai::types::Model>, String>> {
            self.inner.refresh_available_models()
        }
        fn refresh_model_catalog(&self) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.refresh_model_catalog()
        }
        fn get_provider_auth_status_source(&self, provider: &str) -> Option<String> {
            self.inner.get_provider_auth_status_source(provider)
        }
        fn find_model(&self, provider: &str, model_id: &str) -> Option<pi_ai::types::Model> {
            self.inner.find_model(provider, model_id)
        }
        fn set_model( &self, model: &pi_ai::types::Model, wait_for_extensions: bool, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.set_model(model, wait_for_extensions)
        }
        fn cycle_model( &self, direction: &str, wait_for_extensions: bool, ) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>> {
            self.inner.cycle_model(direction, wait_for_extensions)
        }
        fn set_scoped_models(&self, scoped_models: &Value) {
            self.inner.set_scoped_models(scoped_models)
        }
        fn set_thinking_level(&self, level: &str) {
            self.inner.set_thinking_level(level)
        }
        fn set_service_tier(&self, service_tier: &str) {
            self.inner.set_service_tier(service_tier)
        }
        fn cycle_thinking_level(&self) -> Option<String> {
            self.inner.cycle_thinking_level()
        }
        fn set_transport(&self, transport: &str) {
            self.inner.set_transport(transport)
        }
        fn set_steering_mode(&self, mode: &str) {
            self.inner.set_steering_mode(mode)
        }
        fn set_follow_up_mode(&self, mode: &str) {
            self.inner.set_follow_up_mode(mode)
        }
        fn set_auto_compaction_enabled(&self, enabled: bool) {
            self.inner.set_auto_compaction_enabled(enabled)
        }
        fn set_auto_retry_enabled(&self, enabled: bool) {
            self.inner.set_auto_retry_enabled(enabled)
        }
        fn compact( &self, custom_instructions: Option<&str>, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.compact(custom_instructions)
        }
        fn refine(&self, options: RefineOptions) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.refine(options)
        }
        fn abort_compaction(&self) {
            self.inner.abort_compaction()
        }
        fn abort_branch_summary(&self) {
            self.inner.abort_branch_summary()
        }
        fn abort_retry(&self) {
            self.inner.abort_retry()
        }
        fn reload(&self) -> BoxFuture<'static, Result<(), String>> {
            self.inner.reload()
        }
        fn get_rlm_max_depth_status(&self) -> Value {
            self.inner.get_rlm_max_depth_status()
        }
        fn set_rlm_max_depth( &self, max_depth: Value, global: bool, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.set_rlm_max_depth(max_depth, global)
        }
        fn build_session_context(&self) -> Value {
            self.inner.build_session_context()
        }
        fn get_session_stats(&self) -> Value {
            self.inner.get_session_stats()
        }
        fn get_context_tree(&self) -> Value {
            self.inner.get_context_tree()
        }
        fn get_rlm_child_snapshots(&self) -> Vec<Value> {
            self.inner.get_rlm_child_snapshots()
        }
        fn export_to_html( &self, output_path: Option<&str>, ) -> BoxFuture<'static, Result<String, String>> {
            self.inner.export_to_html(output_path)
        }
        fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String> {
            self.inner.export_to_jsonl(output_path)
        }
        fn get_user_messages_for_forking(&self) -> Vec<Value> {
            self.inner.get_user_messages_for_forking()
        }
        fn get_last_assistant_text(&self) -> String {
            self.inner.get_last_assistant_text()
        }
        fn get_tool_definition(&self, name: &str) -> Option<Value> {
            self.inner.get_tool_definition(name)
        }
        fn navigate_tree( &self, target_id: &str, options: NavigateTreeOptions, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.navigate_tree(target_id, options)
        }
        fn start_side_question( &self, question: &str, options: SideQuestionOptions, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.start_side_question(question, options)
        }
        fn abort_side_question(&self, side_question_id: &str) {
            self.inner.abort_side_question(side_question_id)
        }
        fn release_acp_mcp_servers( &self, owner_id: &str, server_names: &[String], ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.release_acp_mcp_servers(owner_id, server_names)
        }
        fn replace_acp_mcp_servers( &self, servers: &[Value], owner_id: &str, ) -> BoxFuture<'static, Result<(), String>> {
            self.inner.replace_acp_mcp_servers(servers, owner_id)
        }
        fn new_session( &self, options: Option<NewSessionRuntimeOptions>, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.new_session(options)
        }
        fn release_rlm_child_session( &self, child_id: &str, session: Arc<dyn DaemonSession>, ) -> Option<Box<dyn FnOnce() + Send>> {
            self.inner.release_rlm_child_session(child_id, session)
        }
        fn replied_to_parent_since_task(&self) -> Option<bool> {
            self.inner.replied_to_parent_since_task()
        }
        fn switch_session( &self, session_path: &str, options: SessionPathOptions, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.switch_session(session_path, options)
        }
        fn fork( &self, entry_id: &str, options: ForkOptions, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.fork(entry_id, options)
        }
        fn import_from_jsonl( &self, input_path: &str, cwd_override: Option<&str>, ) -> BoxFuture<'static, Result<Value, String>> {
            self.inner.import_from_jsonl(input_path, cwd_override)
        }
        fn dispose(&self) -> BoxFuture<'static, ()> {
            self.inner.dispose()
        }
    }

    /// A daemon with one resident session plus the observe controller the daemon
    /// itself hands to a runtime session, wrapped in the REAL host handlers.
    ///
    /// The fixture builds the target's `ActiveSessionRuntimeSession` the way the daemon
    /// itself does in `add_runtime` (`daemon_mode.rs`: `messages_len:
    /// session.messages().len()`, `model_identity`, `session_file`, ...), because
    /// `summary_for_active_session` reads that narrowed view rather than the live
    /// session. `messages_len` is what `messageCount` reports.
    ///
    /// The controller's "current" state is deliberately a SECOND, non-resident state:
    /// `createAgentObserveRecentMessages` asserts family reach between the caller's
    /// state and the target, and the daemon has no same-state short circuit
    /// (`assert_agent_family_reachable`; the TypeScript returns early when both active
    /// session ids are equal, `daemon-mode.ts` `assertAgentFamilyReachable`). Two depth-0
    /// top-level states are siblings, which is exactly the reach a real session has when
    /// it observes its own transcript.
    struct ObserveFixture {
        _directory: tempfile::TempDir,
        daemon: Arc<AgentDaemon>,
        active_session_id: String,
        handlers: crate::core::kernel::shared::HostRequestHandlers,
    }

    impl ObserveFixture {
        fn new(session: &Arc<ObserveSession>) -> Self {
            Self::build(session, None)
        }

        /// The same fixture, with a RESIDENT parent state whose session path is the
        /// child's `parentSession` header value. That is the reach a real daemon grants
        /// a parent observing its own child, and the only way a depth-1 target is
        /// authorized (`is_agent_family_parent`, `core/agent_messages.rs`).
        fn new_with_parent(session: &Arc<ObserveSession>, parent_session_file: &str) -> Self {
            Self::build(session, Some(parent_session_file.to_string()))
        }

        fn build(session: &Arc<ObserveSession>, parent_session_file: Option<String>) -> Self {
            let directory = tempfile::tempdir().expect("temp dir");
            let agent_dir = directory.path().join("agent");
            std::fs::create_dir_all(&agent_dir).expect("agent dir");
            let socket_path = directory.path().join("daemon.sock").to_string_lossy().into_owned();
            let daemon = AgentDaemon::new(
                socket_path.clone(),
                DaemonModeOptions {
                    socket_path: Some(socket_path),
                    default_session_config: AgentSessionRuntimeConfig {
                        cwd: Some(directory.path().to_string_lossy().into_owned()),
                        agent_dir: Some(agent_dir.to_string_lossy().into_owned()),
                        ..Default::default()
                    },
                    create_runtime: Arc::new(|_| {
                        Box::pin(async { panic!("the observe parity test must not create a runtime") })
                    }),
                    worker: None,
                },
            );
            let active_session_id = session.active_session_id.clone();
            let session_handle: Arc<dyn DaemonSession> = Arc::clone(session) as Arc<dyn DaemonSession>;
            // The same projection `add_runtime` installs for a resident session.
            let runtime_session = ActiveSessionRuntimeSession {
                session_id: session_handle.session_id(),
                session_name: session_handle.session_name(),
                session_file: session_handle.session_file(),
                is_session_active: session_handle.is_session_active(),
                is_streaming: session_handle.is_streaming(),
                is_compacting: session_handle.is_compacting(),
                messages_len: session_handle.messages().len(),
                has_running_rlm_children: session_handle.has_running_rlm_children(),
                model_identity: session_handle.model_identity(),
                thinking_level: session_handle.thinking_level(),
                ..ActiveSessionRuntimeSession::default()
            };
            let target_state = Arc::new(StdMutex::new(ActiveSessionState::new(
                active_session_id.clone(),
                AgentSessionRuntime {
                    session: runtime_session,
                    metadata: Some(AgentSessionRuntimeMetadata::default()),
                    model_fallback_message: None,
                },
            )));
            daemon.sessions.lock().expect("sessions poisoned").insert(
                active_session_id.clone(),
                Arc::new(DaemonSessionState {
                    state: Arc::clone(&target_state),
                    session: Arc::clone(&session_handle),
                    runtime_metadata: AgentSessionRuntimeMetadata::default(),
                    snapshot_boundary: StdMutex::new(None),
                }),
            );
            let current_state = match parent_session_file {
                Some(parent_session_file) => {
                    let parent_id = "observe-parent".to_string();
                    let parent_session = Arc::new(ObserveSession {
                        inner: MissingSession::new(&parent_id),
                        active_session_id: parent_id.clone(),
                        session_file: parent_session_file.clone(),
                        agent_session: None,
                        streaming: false,
                        session_active: false,
                        foreground_active: None,
                    });
                    let parent_handle: Arc<dyn DaemonSession> =
                        Arc::clone(&parent_session) as Arc<dyn DaemonSession>;
                    let state = Arc::new(StdMutex::new(ActiveSessionState::new(
                        parent_id.clone(),
                        AgentSessionRuntime {
                            session: ActiveSessionRuntimeSession {
                                session_id: parent_handle.session_id(),
                                session_name: parent_handle.session_name(),
                                session_file: parent_handle.session_file(),
                                ..ActiveSessionRuntimeSession::default()
                            },
                            metadata: Some(AgentSessionRuntimeMetadata::default()),
                            model_fallback_message: None,
                        },
                    )));
                    daemon.sessions.lock().expect("sessions poisoned").insert(
                        parent_id,
                        Arc::new(DaemonSessionState {
                            state: Arc::clone(&state),
                            session: Arc::clone(&parent_handle),
                            runtime_metadata: AgentSessionRuntimeMetadata::default(),
                            snapshot_boundary: StdMutex::new(None),
                        }),
                    );
                    state
                }
                // The caller's own state: a resident-less sibling of the target.
                None => Arc::new(StdMutex::new(ActiveSessionState::new(
                    "observe-caller".to_string(),
                    AgentSessionRuntime::default(),
                ))),
            };
            let controller = daemon.create_agent_observe_controller(Arc::new(move || {
                Some(Arc::clone(&current_state))
            }));
            let handlers = crate::core::agent_observe::create_agent_observe_host_handlers(controller);
            Self {
                _directory: directory,
                daemon,
                active_session_id,
                handlers,
            }
        }

        /// Invoke a REAL host handler exactly as the Python kernel would.
        async fn call(&self, handler: &str, payload: Value) -> Result<Value, String> {
            let handler = self
                .handlers
                .get(handler)
                .expect("the production host handler is registered");
            handler(payload).await.map_err(|error| error.to_string())
        }
    }

    /// Write a session file whose first row is a real header and whose remaining rows are
    /// `message` entries, returning its path. The daemon's own
    /// `read_session_info_sync(sessionFile)` reads these rows, so `firstMessage` and
    /// `transcriptEntryCount` come from a genuine transcript.
    fn write_transcript(
        path: &std::path::Path,
        cwd: &str,
        texts: &[&str],
        parent_session: Option<&str>,
    ) -> String {
        let mut header = serde_json::json!({
            "type": "session",
            "id": "observe-session",
            "version": 3,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "cwd": cwd,
        });
        if let Some(parent_session) = parent_session {
            header["parentSession"] = Value::String(parent_session.to_string());
            header["rlmDepth"] = serde_json::json!(1);
        }
        let mut rows = vec![header.to_string()];
        for (index, text) in texts.iter().enumerate() {
            rows.push(
                serde_json::json!({
                    "type": "message",
                    "id": format!("m{index}"),
                    "parentId": if index == 0 { Value::Null } else { Value::String(format!("m{}", index - 1)) },
                    "timestamp": "2026-01-01T00:00:01.000Z",
                    "message": {
                        "role": "user",
                        "content": text,
                        "timestamp": 1,
                    },
                })
                .to_string(),
            );
        }
        std::fs::write(path, format!("{}\n", rows.join("\n"))).expect("transcript file");
        path.to_string_lossy().to_string()
    }

    /// A daemon holding one session with a real `AgentSession` behind the seam.
    ///
    /// `transcript` writes that many real `message` rows into the session file the
    /// runtime points at, so `sessionManager.getEntries()` and
    /// `readSessionInfo(sessionFile)` both see a genuine lifetime transcript - which is
    /// what `transcriptEntryCount` and `firstMessage` are derived from.
    fn observe_fixture(
        active_session_id: &str,
        transcript: &[&str],
        agent_state: pi_agent_core::types::AgentState,
        rlm_depth: i64,
        streaming: bool,
        session_active: bool,
    ) -> (tempfile::TempDir, ObservedFixtureParts) {
        let scratch = tempfile::tempdir().expect("scratch");
        let cwd = scratch.path().to_string_lossy().to_string();
        let transcript_path = scratch.path().join("transcript.jsonl");
        // A depth-1 child is only reachable through its parent, so the caller's state is a
        // parent whose session path is the child's `parentSession` header value.
        let (session_file, parent_file) = if rlm_depth > 0 {
            let parent_path = scratch.path().join("parent.jsonl");
            let parent_file = write_transcript(&parent_path, &cwd, &["parent task"], None);
            let session_file =
                write_transcript(&transcript_path, &cwd, transcript, Some(&parent_file));
            (session_file, Some(parent_file))
        } else {
            (write_transcript(&transcript_path, &cwd, transcript, None), None)
        };
        let session = Arc::new(ObserveSession {
            inner: MissingSession::new(active_session_id),
            active_session_id: active_session_id.to_string(),
            session_file: session_file.clone(),
            agent_session: Some(real_agent_session(
                ScriptedAgent::new(agent_state),
                &session_file,
                rlm_depth,
            )),
            streaming,
            session_active,
            foreground_active: None,
        });
        let fixture = match parent_file {
            Some(parent_file) => ObserveFixture::new_with_parent(&session, &parent_file),
            None => ObserveFixture::new(&session),
        };
        (scratch, ObservedFixtureParts { fixture, session })
    }

    /// The fixture plus the session handle, so a test can assert on the live session.
    struct ObservedFixtureParts {
        fixture: ObserveFixture,
        #[allow(dead_code)]
        session: Arc<ObserveSession>,
    }

    #[test]
    fn backlog_roster_flush_reconciles_completion_and_reused_child_activity() {
        // Opening a child calls sync_view; a roster flush must not require it.
        for (name, active, foreground) in [
            ("completed-child", false, false),
            ("follow-up-child", true, true),
            ("background-only-child", true, false),
        ] {
            let session = Arc::new(ObserveSession {
                inner: MissingSession::new(name),
                active_session_id: name.into(),
                session_file: String::new(),
                agent_session: None,
                streaming: foreground,
                session_active: active,
                foreground_active: Some(foreground),
            });
            let fixture = ObserveFixture::new(&session);
            let entry = fixture.daemon.sessions.lock().unwrap().get(name).unwrap().clone();
            {
                let mut state = entry.state.lock().unwrap();
                state.runtime.session.is_foreground_active = Some(!foreground);
                state.runtime.session.is_session_active = !active;
                state.runtime.session.is_streaming = !active;
                state.runtime.metadata.as_mut().unwrap().kind = Some("subagent".into());
                state.runtime.metadata.as_mut().unwrap().rlm_child_id = Some(name.into());
            }
            fixture.daemon.flush_roster_now();
            let rows = fixture.daemon.roster_reporter.lock().unwrap().last_composed.clone();
            let row = rows.values().find(|row| row.summary.session_id == name).unwrap();
            assert_eq!(row.summary.activity, if foreground { "working" } else { "idle" });
            assert_eq!(row.summary.is_session_active, active);
            assert_eq!(row.summary.is_streaming, foreground);
            assert_eq!(row.summary.status_label.as_deref(),
                (active && !foreground).then_some("background helper"));
            assert_eq!(entry.state.lock().unwrap().runtime.session.is_foreground_active, Some(foreground));
        }
    }

    #[test]
    fn backlog_streaming_view_retains_history_storage_and_model_metadata() {
        for history_size in [1, 256] {
            let model = pi_ai::models::get_model("openai", "gpt-4o").expect("test model");
            let (_scratch, parts) = observe_fixture(
                "stream-view", &["retained"],
                pi_agent_core::types::AgentState {
                    model: model.clone(),
                    messages: (0..history_size).map(|i| assistant_message(&"x".repeat(4096), i)).collect(),
                    ..Default::default()
                }, 0, true, true,
            );
            let entry = parts.fixture.daemon.sessions.lock().unwrap().get("stream-view").unwrap().clone();
            entry.sync_view();
            let original_ptr = {
                let mut state = entry.state.lock().unwrap();
                state.runtime.session.leaf_id = Some("preserved-leaf".into());
                state.runtime.session.messages.as_ptr() as usize
            };
            for _ in 0..50 {
                entry.sync_event_view(false);
                entry.handle_event(&serde_json::json!({
                    "type": "message_update", "message": assistant_message("partial", 999)
                }));
                let state = entry.state.lock().unwrap();
                assert_eq!(state.runtime.session.messages.as_ptr() as usize, original_ptr);
                assert_eq!(state.runtime.session.messages.len(), history_size as usize);
                assert_eq!(state.runtime.session.messages_len, history_size as usize);
                assert_eq!(state.runtime.session.model_identity.as_ref().map(|value| value.id.as_str()), Some(model.id.as_str()));
                assert_eq!(state.runtime.session.leaf_id.as_deref(), Some("preserved-leaf"));
            }
            entry.sync_event_view(true);
            entry.handle_event(&serde_json::json!({
                "type": "message_end", "message": assistant_message(&"x".repeat(4096), history_size - 1)
            }));
            let state = entry.state.lock().unwrap();
            assert_eq!(state.runtime.session.messages.len(), history_size as usize);
            assert!(state.runtime.session.streaming_message.is_none(), "completed history must not also appear as a streaming message");
        }
    }

    /// A real `AgentSession` whose session file is the caller's transcript: the session
    /// manager is OPENED on it, so `getEntries()` and the daemon's own
    /// `read_session_info_sync(sessionFile)` both read the same rows.
    fn real_agent_session(
        agent: Arc<ScriptedAgent>,
        transcript_path: &str,
        rlm_depth: i64,
    ) -> Arc<crate::core::agent_session::AgentSession> {
        let cwd = std::path::Path::new(transcript_path)
            .parent()
            .map(|parent| parent.to_string_lossy().to_string())
            .unwrap_or_default();
        let settings = Arc::new(StdMutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));
        let manager = SessionManager::open(transcript_path, None, Some(&cwd))
            .expect("open the transcript the daemon will read");
        let loader = Arc::new(crate::core::resource_loader::DefaultResourceLoader::new(
            crate::core::resource_loader::DefaultResourceLoaderOptions {
                cwd: cwd.clone(),
                agent_dir: cwd.clone(),
                settings_manager: Some(Arc::clone(&settings)),
                no_extensions: true,
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                bundled_skills_dir: Some(None),
                ..Default::default()
            },
        ));
        crate::core::agent_session::AgentSession::new(crate::core::agent_session::AgentSessionConfig {
            agent: agent as Arc<dyn crate::core::agent_session::AgentHandle>,
            session_manager: Arc::new(StdMutex::new(manager)),
            settings_manager: settings,
            service_tier_preference: None,
            cwd: cwd.clone(),
            agent_dir: Some(cwd),
            scoped_models: None,
            resource_loader: loader,
            custom_tools: None,
            model_registry: Arc::new(StdMutex::new(
                crate::core::model_registry::ModelRegistry::in_memory(
                    crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
                ),
            )),
            initial_active_tool_names: None,
            allowed_tool_names: None,
            include_goals: Some(false),
            agent_message_controller: None,
            agent_observe_controller: None,
            include_compact_skill: Some(false),
            rlm_heartbeat_controller: None,
            mcp_manager: None,
            base_tools_override: None,
            extension_runner_ref: None,
            session_start_event: None,
            rlm_depth: Some(rlm_depth),
            rlm_max_depth: Some(2),
            rlm_session_dir: None,
            rlm_parent_node_id: None,
            rlm_parent_agent: None,
            semantic_parent_session_id: None,
            semantic_spawned_by_request_id: None,
            subagent_runtime_host: None,
            autonomous: None,
            prewarm_ipython_kernel: Some(false),
            auto_refine_reviewer: None,
            serialized_refine: None,
            initial_goal: None,
        })
        .expect("agent session")
    }

    /// D-06: a real host call that omits `limit`/`maxChars` must succeed with the
    /// TypeScript defaults (`normalizeObserveLimit(input.limit)` with the
    /// `defaultLimit = 8` / `defaultMaxChars = 800` parameter defaults,
    /// `agent-observe.ts:107-112`, called from `daemon-mode.ts:3467-3468`).
    ///
    /// Measured defect: `create_agent_observe_recent_messages`
    /// (`daemon_mode.rs:15733-15734`) passes 20 and 4000 as the defaults, and
    /// `normalize_observe_max_chars` rejects anything above 2000
    /// (`agent_observe.rs:223`), so EVERY host request that omits `maxChars` - which
    /// is what the installed observe skill sends - fails with
    /// "agent_observe max_chars must be between 80 and 2000".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_recent_defaults_to_eight_messages_and_eight_hundred_chars() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        let mut messages: Vec<AgentMessage> = (0..11)
            .map(|index| user_message(&format!("message-{index}"), index as i64))
            .collect();
        messages.push(user_message(&"x".repeat(1200), 11));
        agent_state.messages = messages;
        let (_scratch, parts) = observe_fixture(
            "observe-defaults",
            &["the first task", "an answer", "third"],
            agent_state,
            0,
            false,
            false,
        );
        let fixture = &parts.fixture;

        let result = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id }),
            )
            .await
            .expect(
                "DEFECT D-06: a host request that omits limit/maxChars must use the TypeScript defaults and succeed",
            );
        assert_eq!(
            result.get("limit").and_then(Value::as_f64),
            Some(TS_DEFAULT_LIMIT as f64),
            "the daemon must default to the TypeScript limit of 8"
        );
        assert_eq!(
            result.get("maxChars").and_then(Value::as_f64),
            Some(TS_DEFAULT_MAX_CHARS as f64),
            "the daemon must default to the TypeScript maxChars of 800"
        );
        let previews = result
            .get("messages")
            .and_then(Value::as_array)
            .expect("messages array");
        assert_eq!(
            previews.len(),
            TS_DEFAULT_LIMIT as usize,
            "the default window must return 8 messages, not 12"
        );
        assert_eq!(
            previews[0].get("index").and_then(Value::as_f64),
            Some(4.0),
            "the window must start at messageCount - limit"
        );
        let last = previews.last().expect("a last preview");
        assert_eq!(
            last.get("text").and_then(Value::as_str).map(str::len),
            Some(TS_DEFAULT_MAX_CHARS as usize),
            "the long message must be clipped to the default maxChars"
        );
        assert_eq!(last.get("truncated").and_then(Value::as_bool), Some(true));
    }

    /// The clamp must stay unweakened: an EXPLICIT out-of-range value is still an
    /// error (`clampInteger`, `agent-observe.ts:69-77`), and an in-range explicit
    /// value is honored.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_recent_keeps_rejecting_explicit_out_of_range_values() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        agent_state.messages = vec![user_message("only message", 1)];
        let (_scratch, parts) =
            observe_fixture("observe-clamp", &["only message"], agent_state, 0, false, false);
        let fixture = &parts.fixture;

        let too_large = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "max_chars": 4000 }),
            )
            .await
            .expect_err("an explicit 4000 max_chars must stay rejected");
        assert!(
            too_large.contains("agent_observe max_chars must be between 80 and 2000"),
            "the max_chars clamp error must survive, got {too_large:?}"
        );

        let too_small = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 51 }),
            )
            .await
            .expect_err("an explicit 51 limit must stay rejected");
        assert!(
            too_small.contains("agent_observe limit must be between 1 and 50"),
            "the limit clamp error must survive, got {too_small:?}"
        );

        let allowed = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 3, "max_chars": 150 }),
            )
            .await
            .expect("an in-range explicit value must be honored");
        assert_eq!(allowed.get("limit").and_then(Value::as_f64), Some(3.0));
        assert_eq!(allowed.get("maxChars").and_then(Value::as_f64), Some(150.0));
    }

    /// D-07(a): a streaming session with pending tool calls reports `"tool"`, not
    /// `"model"` (`daemon-mode.ts:3490-3500`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_summary_reports_tool_status_while_a_tool_call_is_pending() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        agent_state.messages = vec![user_message("run the tests", 1)];
        agent_state.is_streaming = true;
        agent_state.pending_tool_calls.insert("call-1".to_string());
        let (_scratch, parts) = observe_fixture(
            "observe-tool",
            &["run the tests", "an answer", "more"],
            agent_state,
            0,
            true,
            true,
        );
        let fixture = &parts.fixture;

        let result = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("the host request must succeed");
        let status = result
            .pointer("/agent/status")
            .and_then(Value::as_str)
            .expect("agent status");
        assert_eq!(
            status, "tool",
            "DEFECT D-07: a streaming session with a pending tool call must report \"tool\", not \"model\""
        );

        // Negative control: streaming with NO pending tool call stays "model".
        let mut idle_state = pi_agent_core::types::AgentState::default();
        idle_state.messages = vec![user_message("run the tests", 1)];
        idle_state.is_streaming = true;
        let (_scratch2, idle_parts) = observe_fixture(
            "observe-model",
            &["run the tests", "an answer", "more"],
            idle_state,
            0,
            true,
            true,
        );
        let idle_fixture = &idle_parts.fixture;
        let result = idle_fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": idle_fixture.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("the host request must succeed");
        assert_eq!(
            result.pointer("/agent/status").and_then(Value::as_str),
            Some("model"),
            "streaming without a pending tool call stays \"model\""
        );
    }

    /// D-07(b): `transcriptEntryCount` is the LIFETIME JSONL entry count
    /// (`session.sessionManager.getEntries().length`, `daemon-mode.ts:3513`), kept
    /// distinct from `messageCount` (`session.messages.length`, `:3512`). The port
    /// copies `messageCount` into both (`daemon_mode.rs:15596`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_summary_keeps_transcript_entries_distinct_from_message_count() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        agent_state.messages = vec![user_message("first", 1), assistant_message("second", 2)];
        let (_scratch, parts) = observe_fixture(
            "observe-entries",
            &["row 0", "row 1", "row 2", "row 3", "row 4", "row 5", "row 6"],
            agent_state,
            0,
            false,
            false,
        );
        // 7 message rows are on disk, 2 are in the model context.

        let fixture = &parts.fixture;

        let result = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("the host request must succeed");
        assert_eq!(
            result.pointer("/agent/messageCount").and_then(Value::as_f64),
            Some(2.0),
            "messageCount is the active model-context length"
        );
        assert_eq!(
            result
                .pointer("/agent/transcriptEntryCount")
                .and_then(Value::as_f64),
            Some(7.0),
            "DEFECT D-07: transcriptEntryCount must be sessionManager.getEntries().length, not a copy of messageCount"
        );
    }

    /// D-07(c): `model`, `firstMessage`, `latestMessage` and the diagnostics spread
    /// (`daemon-mode.ts:3514-3516`, `:3523-3528`). The port leaves all of them unset.
    ///
    /// The first three fields are read off a depth-0 session; the diagnostics spread is
    /// read off a depth-1 session, because `rlmDiagnostics` returns `undefined` at depth
    /// 0 (`agent-session.ts:3874`). Both go through the same production summary builder.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_summary_carries_model_first_latest_and_diagnostics() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        agent_state.model = pi_ai::types::Model::new(
            "deepseek-v4.1-flash",
            "DeepSeek",
            "openai-completions",
            "ollama-cloud",
            "http://localhost",
        );
        agent_state.messages = vec![
            user_message("the first task", 10),
            assistant_message("an answer", 20),
        ];
        let (_scratch, parts) = observe_fixture(
            "observe-fields",
            &["the first task", "an answer"],
            agent_state,
            0,
            false,
            false,
        );
        let fixture = &parts.fixture;

        let result = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("the host request must succeed");
        let agent = result.get("agent").expect("agent summary");
        assert_eq!(
            agent.get("model").and_then(Value::as_str),
            Some("ollama-cloud/deepseek-v4.1-flash"),
            "DEFECT D-07: model must be provider-slash-id (`daemon-mode.ts:3514`)"
        );
        assert_eq!(
            agent.get("firstMessage").and_then(Value::as_str),
            Some("the first task"),
            "DEFECT D-07: firstMessage must carry the session's first message (`daemon-mode.ts:3523`)"
        );
        let latest = agent.get("latestMessage").expect(
            "DEFECT D-07: latestMessage must be the preview of the newest message (`daemon-mode.ts:3524-3528`)",
        );
        assert_eq!(latest.get("role").and_then(Value::as_str), Some("assistant"));
        assert_eq!(latest.get("index").and_then(Value::as_f64), Some(1.0));
        assert_eq!(latest.get("text").and_then(Value::as_str), Some("an answer"));

        // The diagnostics spread, on a depth-1 child observed by its parent.
        let mut child_state = pi_agent_core::types::AgentState::default();
        child_state.messages = vec![assistant_message("child answer", 5)];
        let (_scratch2, child_parts) = observe_fixture(
            "observe-child",
            &["parent task", "child answer"],
            child_state,
            1,
            false,
            false,
        );
        let child = &child_parts.fixture;
        let result = child
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": child.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("a parent may observe its own child");
        let child_agent = result.get("agent").expect("agent summary");
        assert_eq!(
            child_agent.get("continuationQueued").and_then(Value::as_bool),
            Some(false),
            "DEFECT D-07: the rlmDiagnostics spread (`daemon-mode.ts:3516`) must reach a depth>0 summary, got {:?}",
            child_agent.get("continuationQueued")
        );
    }

    /// The diagnostics spread is ABSENT for a depth-0 session: `get rlmDiagnostics()`
    /// returns `undefined` at depth 0 (`agent-session.ts:3874`), so spreading it adds
    /// no keys (`daemon-mode.ts:3516`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observe_summary_has_no_diagnostics_at_depth_zero() {
        let mut agent_state = pi_agent_core::types::AgentState::default();
        agent_state.messages = vec![user_message("root task", 1)];
        let (_scratch, parts) =
            observe_fixture("observe-root", &["root task"], agent_state, 0, false, false);
        let fixture = &parts.fixture;

        let result = fixture
            .call(
                "agent_observe.recent",
                serde_json::json!({ "target": fixture.active_session_id, "limit": 8, "max_chars": 800 }),
            )
            .await
            .expect("the host request must succeed");
        let agent = result.get("agent").expect("agent summary");
        assert!(
            agent.get("continuationQueued").is_none(),
            "a depth-0 session has no diagnostics keys, got {agent:?}"
        );
        assert!(agent.get("lastStopReason").is_none());
        assert!(agent.get("terminalStatus").is_none());
    }
}
