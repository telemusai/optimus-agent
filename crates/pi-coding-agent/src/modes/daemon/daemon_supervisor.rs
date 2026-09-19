//! Port of packages/coding-agent/src/modes/daemon/daemon-supervisor.ts
//!
//! Shared protocol constants are kept here; the private native adapter owns
//! socket serving, durable supervisor ownership and isolated worker processes.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as TokioMutex, Notify};

use crate::core::agent_messages::{
    assert_agent_family_reach, assert_agent_session_name_available, format_agent_session_name_unavailable,
    session_name_reservation_key, AgentFamilyCatalogEntry, AgentSessionMessageAgentSummary,
    AgentSessionNameScope,
};
use crate::core::orphan_process_journal::{
    clear_orphan_process_journal, kill_orphan_process, read_active_orphan_processes,
    should_reap_orphan_process, ORPHAN_PROCESS_JOURNAL_ENV,
};
use crate::core::session_action_store::{
    can_evict_worker, IdleEvictionMinutes, SessionEvictionSnapshot, WorkerEvictionSnapshot,
    WorkerLifecycle,
};
use crate::core::session_lease::{canonical_session_path, get_process_start_id, SessionAlreadyActiveError};
use crate::core::session_manager::{get_session_artifact_path_for_file, read_session_info, SessionInfo};
use crate::core::session_resolver::looks_like_session_path;
use crate::core::settings_manager::SettingsManager;
use crate::modes::agent_connection::types::AgentConnectionHeartbeat;
use crate::modes::rpc::jsonl::serialize_json_line;
use crate::utils::atomic_file::{
    remove_file_durably, write_file_atomic_sync, AtomicFileWriteCoordinator, WriteFileAtomicOptions,
};
use crate::utils::child_process::{
    is_process_alive, process_id_exists, signal_process_group_or_process, spawn_hidden, Signal, SpawnOptions,
};

use super::active_session_state::{create_active_session_id, DaemonSocketClient};
use super::agent_roster::{
    passivated_worker_roster_entry, roster_agent_id_for_summary, session_summary_from_roster_entry,
    worker_roster_entry_from_summary, AgentRoster, AgentRosterEntry, AgentRosterMutation,
    RegisteredHeartbeatFlags, RosterSessionSummary, WorkerRosterEntry,
};
use super::command_recovery_journal::{create_command_idempotency_key, CommandRecoveryJournal};
use super::compact_session_stream::{
    create_compact_assistant_delta, is_compact_assistant_delta, CompactAssistantStreamReconstructor,
};
use super::daemon_catalog_process::{DaemonCatalogClient, DAEMON_CATALOG_ROLE_ENV};
use super::daemon_protocol::{collect_daemon_client_env, create_daemon_event_meta, is_daemon_command_envelope, is_daemon_mutating_command, salvage_daemon_command_id, DaemonAttachResult, DaemonCommand, DaemonResponse, DAEMON_COMMAND_ENVELOPE_MIN_PROTOCOL_VERSION, DAEMON_DEFAULT_CLIENT_CAPABILITIES, DAEMON_DEFAULT_SERVER_CAPABILITIES, DAEMON_SCHEMA_ID, DAEMON_SCHEMA_REVISION, DAEMON_UPDATE_RESTART_FORMAT_VERSION};
use super::daemon_client::{DaemonHello};
use super::daemon_errors::{deserialize_daemon_error, serialize_daemon_error, DaemonSessionRecoveringError};
use super::daemon_session_id::matches_session_id_suffix;
use super::agent_roster::{classify_session_roster_status, is_session_summary_busy};
use super::daemon_session_list::{
    is_evictable_empty_session_summary, summary_for_inactive_session, SessionSummary,
};
use super::daemon_socket::{
    acquire_daemon_socket_path_lease, cleanup_daemon_socket_path, default_daemon_socket_dir,
    default_daemon_socket_path, get_daemon_socket_identity, normalize_socket_path_for_daemon,
    prepare_daemon_socket_path, restrict_daemon_socket_path, DaemonSocketIdentity, DaemonSocketPathLease,
};
use super::daemon_worker_client::{
    DaemonWorkerAuthenticationError, DaemonWorkerClient, DaemonWorkerProbeTimeoutError,
};
use super::daemon_worker_protocol::{
    durable_daemon_create_command, durable_daemon_worker_descriptor, DurableDaemonCreateCommand,
    DaemonWorkerDescriptor, DaemonWorkerLifecycle, DaemonWorkerRosterOutbound,
    DAEMON_WORKER_ACTIVE_SESSION_ID_ENV, DAEMON_WORKER_INSTANCE_ID_ENV,
    DAEMON_WORKER_PEER_TRANSPORT_CAPABILITY, DAEMON_WORKER_RECOVERY_JOURNAL_ENV, DAEMON_WORKER_ROLE_ENV,
    DAEMON_WORKER_ROSTER_CAPABILITY, DAEMON_WORKER_STARTUP_GATE_COMMIT, DAEMON_WORKER_STARTUP_GATE_FD_ENV,
    DAEMON_WORKER_SUPERVISOR_SOCKET_ENV, DAEMON_WORKER_TOKEN_ENV, ROSTER_HEARTBEAT_INTERVAL_MS,
    SESSION_LEASE_OWNER_ID_ENV, SESSION_LEASES_ENABLED_ENV,
};
use super::mutation_drain_latch::MutationDrainLatch;
use super::rlm_ledger::{
    create_rlm_ledger_registry_seed_source, tombstone_saved_session_delete,
    with_passive_rlm_descendant_infos, RlmLedgerEdge, RlmSpawnLedger,
};
use super::saved_session_info::serialize_saved_session_info;
use super::snapshot_transcript_cache::{SnapshotTranscriptCache, SNAPSHOT_TARGET_CHUNK_BYTES};
use super::worker_recovery_journal::WorkerRecoveryJournal;

/// `type DaemonCommandBody = DistributiveOmit<DaemonCommand, "id">`.
pub type DaemonCommandBody = DaemonCommand;
/// `type DaemonCreateCommand = Extract<DaemonCommand, { type: "create" }>`.
pub type DaemonCreateCommand = DaemonCommand;

const WORKER_CONNECT_TIMEOUT_MS: u64 = if cfg!(windows) { 90_000 } else { 30_000 };
const WORKER_CONNECT_PROBE_MS: u64 = if cfg!(windows) { 2_000 } else { 500 };
const WORKER_PROBE_BACKOFF_MIN_MS: u64 = 25;
const WORKER_PROBE_BACKOFF_MAX_MS: u64 = if cfg!(windows) { 2_000 } else { 25 };

/// Per-attempt handshake waits consume the remaining outer connect budget; a
/// smaller fixed clock makes a consistently slow (win32) handshake fail every retry.
pub fn handshake_budget_ms(deadline: f64, now: f64) -> Result<f64, DaemonWorkerProbeTimeoutError> {
    let remaining = deadline - now;
    if remaining <= 0.0 {
        return Err(DaemonWorkerProbeTimeoutError {
            message: "Worker connection deadline elapsed".to_string(),
        });
    }
    Ok(remaining)
}

const ROSTER_WATCHDOG_INTERVAL_MS: u64 = 15_000;
const ROSTER_STALE_AFTER_MS: u64 = 3 * ROSTER_HEARTBEAT_INTERVAL_MS;
const PEER_TRANSPORT_GRANT_TTL_MS: u64 = 10_000;
const WORKER_REQUEST_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const INPUT_PAUSE_CLEANUP_TIMEOUT_MS: u64 = 5_000;
const UPDATE_RESTART_MUTATION_DRAIN_TIMEOUT_MS: u64 = 80_000;
const UPDATE_RESTART_WORKER_REQUEST_TIMEOUT_MS: u64 = 90_000;
// The whole pre-commit prepare (drain + worker fencing) must finish inside the
// caller's 120s prepare_update_restart request timeout, or roll back; otherwise
// an abandoned prepare leaves the daemon permanently fenced with workers stopped.
const UPDATE_RESTART_PREPARE_DEADLINE_MS: u64 = 100_000;
const WORKER_RETRY_DELAYS_MS: [u64; 3] = [250, 1000, 5000];
const DEFERRED_RECOVERY_RECHECK_MS: u64 = 5000;
// ~2.5 minutes of probing: each round is one 5s defer recheck plus a ~11s three-delay probe pass.
const MAX_DEFERRED_RECOVERY_ROUNDS: u32 = 10;
const STOP_FINALIZATION_RECHECK_MS: u64 = 250;
const STOP_FINALIZATION_SIGKILL_GRACE_MS: u64 = 5000;
const STOP_FINALIZATION_RETRY_MS: u64 = 5000;
const STALE_RECLAIM_WAIT_MS: u64 = 10_000;
const DESCRIPTOR_WRITE_DRAIN_TIMEOUT_MS: u64 = 5_000;
const MAX_SUPERVISOR_PERFORMANCE_RECORDERS: usize = 32;
// Polling loops probe existence cheaply via kill(0); the ps-backed zombie and
// identity checks are throttled so a wedged worker cannot saturate the
// supervisor event loop with synchronous subprocess spawns.
// Windows identity lookups launch PowerShell, so recheck less often there.
const LIVENESS_IDENTITY_RECHECK_MS: u64 = if cfg!(windows) { 3_000 } else { 500 };
const OWNED_WORKER_DISCONNECT_GRACE_MS: u64 = 30_000;
const IDLE_EVICTION_MAX_SWEEP_INTERVAL_MS: u64 = 5 * 60_000;
const IDLE_EVICTION_MIN_SWEEP_INTERVAL_MS: u64 = 60_000;
const IDLE_EVICTION_DRAIN_TIMEOUT_MS: u64 = 5_000;
const CHILD_PASSIVATION_PER_WORKER_CAP: u32 = 2;
const SCHEDULED_WAKE_RETRY_MS: f64 = 60_000.0;
const SCHEDULED_WAKE_MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const SCHEDULED_WAKE_CLIENT_ID: &str = "scheduled-wake";
const SUPERVISOR_CONFIG_FILE_NAME: &str = "supervisor-config";
const WORKER_STARTUP_GATE_FD: u32 = 3;

/// The supervisor's advertised capabilities (`SUPERVISOR_SERVER_CAPABILITIES`).
pub fn supervisor_server_capabilities() -> Vec<String> {
    let mut capabilities: Vec<String> = DAEMON_DEFAULT_SERVER_CAPABILITIES
        .iter()
        .map(super::daemon_client::capability_name)
        .collect();
    capabilities.push("agent_roster".to_string());
    capabilities.push("direct_peer_transport".to_string());
    capabilities
}

/// `DAEMON_COMMAND_TYPES`.
pub const DAEMON_COMMAND_TYPES: [&str; 110] = [
    "ack_result",
    "list",
    "list_agent_peers",
    "get_direct_worker_transport",
    "roster_subscribe",
    "roster_unsubscribe",
    "list_saved_sessions",
    "create",
    "attach",
    "reattach",
    "detach",
    "complete_owned_session",
    "promote_owned_session",
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
    "cycle_thinking_level",
    "set_service_tier",
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
    // Optional Jev surface, advertised only when the daemon capability
    // `jev_control` is negotiated (no protocol/schema bump).
    "jev_get_settings",
    "jev_set_session_mode",
    "jev_get_status",
    "extension_ui_response",
    "prepare_update_restart",
    "retry_worker",
    "restart",
    "shutdown",
];

#[path = "native_supervisor.rs"]
mod native_supervisor;
pub(crate) use native_supervisor::run_daemon_supervisor_mode;
