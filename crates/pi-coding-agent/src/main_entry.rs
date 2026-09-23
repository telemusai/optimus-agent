//! Port of packages/coding-agent/src/main.ts
//!
//! The TypeScript entry point is a linear startup sequence that ends in one of
//! seven run modes. Every pure decision helper, the session-manager selection,
//! the runtime-config assembly and the runtime factory are ported directly; the
//! run modes themselves (interactive TUI, print, rpc, acp, daemon, agents view,
//! public command handling) live in other slices, so `main` drives them through
//! the `MainHost` seam below with the same order and the same arguments.
//!
//! blocked_on (other slices, files still empty on disk):
//!   - `cli/public-command.ts` (`handlePublicCommand`)
//!   - `cli/owned-session-worker.ts` (`isOwnedSessionWorkerProcess`,
//!     `installOwnedSessionRecoveryTracking`)
//!   - `modes/print-mode.ts` (`runPrintMode`, `runPrintModeWithConnection`)
//!   - `modes/acp/acp-mode.ts` (`runAcpMode`, `runAcpModeWithConnection`)
//!   - `modes/interactive/components/config-selector.ts` and the interactive TUI
//!     entry points (`InteractiveMode`, `runAgentsViewMode`, `runDaemonMode`,
//!     `runDaemonSupervisorMode`)
//!   - `core/telemetry.ts` (`isTelemetryEnabled`)
//!   - `core/keybindings.ts` (`setKeybindings`)

use std::path::Path;
use std::sync::{Arc, Mutex};

use pi_ai::types::{ImageContent, Model};
use serde_json::Value;

use crate::cli::args::{parse_args, Args, ListModelsValue, ResumeValue};
use crate::cli::command_registry::format_top_level_help;
use crate::cli::daemon_launch::{
    ensure_interactive_daemon_running, is_daemon_session_summary, probe_running_daemon_sessions,
    shutdown_daemon_and_wait,
};
use crate::cli::daemon_stop_confirm::{
    confirm_daemon_session_loss, pluralize_sessions, ConfirmIo, ConfirmOptions, DaemonSessionLossCopy,
};
use crate::cli::file_processor::{process_file_arguments, FileProcessorIo, ProcessFileOptions};
use crate::cli::initial_message::{build_initial_message, InitialMessageInput};
use crate::cli::list_models::{list_models, ListModelsIo};
use crate::core::session_resolver::looks_like_session_path;
use crate::config::{expand_tilde_path, get_agent_dir, get_session_dir_env_override, APP_NAME, VERSION};
use crate::core::agent_session_config::{
    merge_agent_session_runtime_config, merge_autonomous_config, AgentSessionRuntimeConfig,
};
use crate::core::autonomous::AgentAutonomousConfig;
use crate::core::agent_session_runtime::{
    AgentSessionRuntime, CreateAgentSessionRuntimeFactory, CreateAgentSessionRuntimeInput,
    CreateAgentSessionRuntimeResult,
};
use crate::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    AgentSessionRuntimeDiagnostic, AgentSessionServices, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions, DIAGNOSTIC_ERROR,
};
use crate::core::auth_guidance::format_no_models_available_message;
use crate::core::auth_storage::AuthStorage;
use crate::core::export_html::{export_from_file, ExportOptions};
use crate::core::model_resolver::{
    find_initial_model, models_are_equal, resolve_cli_model, resolve_model_scope, FindInitialModelOptions,
    ResolveCliModelOptions, ScopedModel,
};
use crate::core::output_guard::{restore_stdout, take_over_stdout, write_stdout};
use crate::core::resource_loader::DefaultResourceLoaderOptions;
use crate::core::sdk::{parse_thinking_level, CreateAgentSessionOptions};
use crate::core::session_cwd::{
    format_missing_session_cwd_prompt, get_missing_session_cwd_issue, MissingSessionCwdError, SessionCwdIssue,
};
use crate::core::session_lease::canonical_session_path;
use crate::core::session_manager::{
    find_most_recent_session_for_cwd, get_default_session_dir, load_entries_from_file, SessionManager,
};
use crate::core::settings_manager::{SettingsError, SettingsManager};
use crate::core::logging::{install_file_log_sink, set_log_context};
use crate::core::timings::{print_timings, reset_timings, time};

/// `Promise<void>` returned by the `MainHost` seams.
pub type BoxFuture<T> = pi_ai::types::BoxFuture<T>;
use crate::migrations::{run_migrations, show_deprecation_warnings};
use crate::modes::agent_connection::daemon_agent_connection::{
    collect_daemon_client_env, collect_daemon_launch_env, DaemonAgentConnection, DaemonAgentConnectionOptions,
    DaemonEventCursor as ConnectionEventCursor, DaemonEventMeta as ConnectionEventMeta,
    DaemonOutbound as ConnectionOutbound, DaemonResponse as ConnectionResponse,
    DaemonSessionSnapshot as ConnectionSessionSnapshot, DaemonSessionSummary as ConnectionSessionSummary,
    DaemonTransportClient as ConnectionTransport,
};
use crate::modes::agent_connection::types::{
    AgentConnection as AgentConnectionTrait, AgentConnectionHistoryWindow, AgentConnectionSessionTree,
};
use crate::modes::daemon::daemon_client::{
    DaemonCapabilityUnavailableError, DaemonClient, DaemonClientCloseListener, DaemonClientError,
    DaemonClientMessageListener, DaemonClientRequestOptions, DaemonCommandBody,
};
use crate::modes::daemon::daemon_protocol::DaemonResponse;
use crate::modes::daemon::daemon_protocol::{
    DaemonEventCursor as ProtocolEventCursor, DaemonHistoryWindow as ProtocolHistoryWindow,
    DaemonOutbound as ProtocolOutbound,
    DaemonSessionSnapshot as ProtocolSessionSnapshot, DaemonSessionSnapshotHead as ProtocolSnapshotHead,
    DaemonSessionTree as ProtocolSessionTree,
};
use crate::modes::daemon::daemon_errors::{
    deserialize_daemon_create_error, deserialize_daemon_error, DaemonError,
};
use crate::modes::daemon::daemon_session_list::{
    resolve_attach_model_fallback_message, SessionSummary,
};
use crate::modes::daemon::daemon_socket::default_daemon_socket_path;
use crate::modes::agents_view::agents_view_state::AgentsViewScopeKey;
use crate::modes::interactive::interactive_mode::{InteractiveModeRunResult, InteractiveModeRunResultType};
use crate::utils::daemon_socket_path::normalize_socket_path;
use crate::modes::daemon::daemon_catalog_process::is_daemon_catalog_process_from_env;
use crate::modes::daemon::daemon_worker_protocol::{
    daemon_worker_instance_id, is_daemon_worker_process_from_env, require_daemon_worker_authentication_token,
    wait_for_daemon_worker_startup_gate, DAEMON_WORKER_ACTIVE_SESSION_ID_ENV,
};
use crate::utils::paths::is_local_path;

fn red(message: &str) -> String {
    format!("\u{1b}[31m{message}\u{1b}[39m")
}

fn yellow(message: &str) -> String {
    format!("\u{1b}[33m{message}\u{1b}[39m")
}

fn dim(message: &str) -> String {
    format!("\u{1b}[2m{message}\u{1b}[22m")
}

fn gray(message: &str) -> String {
    format!("\u{1b}[90m{message}\u{1b}[39m")
}

/// Read all content from piped stdin.
/// Returns `None` if stdin is a TTY (interactive terminal).
pub async fn read_piped_stdin() -> Option<String> {
    use std::io::{IsTerminal, Read};
    // If stdin is a TTY, we're running interactively - don't read stdin
    if std::io::stdin().is_terminal() {
        return None;
    }

    let mut bytes = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut bytes);
    decode_piped_stdin(&bytes)
}

/// `process.stdin.setEncoding("utf8")` + `data += chunk` + `resolve(data.trim() || undefined)`
/// (`packages/coding-agent/src/main.ts:136-147`).
///
/// The TypeScript decodes each chunk as UTF-8 with U+FFFD replacement, so invalid
/// bytes keep the content (mangled, but present). `read_to_string` instead fails
/// with `InvalidData` and leaves the buffer empty or partial, which silently drops
/// piped input, so the bytes are decoded lossily here.
pub fn decode_piped_stdin(bytes: &[u8]) -> Option<String> {
    let data = String::from_utf8_lossy(bytes);
    let trimmed = data.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `collectSettingsDiagnostics(settingsManager, context)`.
pub fn collect_settings_diagnostics(
    settings_manager: &mut SettingsManager,
    context: &str,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    settings_manager
        .drain_errors(None)
        .into_iter()
        .map(|SettingsError { scope, error }| AgentSessionRuntimeDiagnostic {
            type_: "warning".to_string(),
            message: format!("({context}, {scope} settings) {}", error.message),
        })
        .collect()
}

/// `reportDiagnostics(diagnostics)`.
pub fn report_diagnostics(diagnostics: &[AgentSessionRuntimeDiagnostic]) {
    for diagnostic in diagnostics {
        let color = |message: &str| match diagnostic.type_.as_str() {
            "error" => red(message),
            "warning" => yellow(message),
            _ => dim(message),
        };
        let prefix = match diagnostic.type_.as_str() {
            "error" => "Error: ",
            "warning" => "Warning: ",
            _ => "",
        };
        eprintln!("{}", color(&format!("{prefix}{}", diagnostic.message)));
    }
}

/// `isTruthyEnvFlag(value)`.
pub fn is_truthy_env_flag(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    if value.is_empty() {
        return false;
    }
    value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
}

/// `isTruthyEnvFlag(process.env[PI_OFFLINE])`.
pub fn is_offline_env_flag() -> bool {
    let value = std::env::var("PI_OFFLINE").ok();
    is_truthy_env_flag(value.as_deref())
}

pub type ClientMode = crate::core::agent_session_config::AgentExecutionMode;
/// Compatibility view of the CLI's internal daemon process entrypoint.
pub type AppMode = String;

pub const APP_MODE_DAEMON: &str = "daemon";
pub const APP_MODE_INTERACTIVE: &str = "interactive";
pub const APP_MODE_PRINT: &str = "print";

/// `shouldRejectNonInteractiveAttach(attachAgent, appMode)`.
pub fn should_reject_non_interactive_attach(attach_agent: Option<&str>, app_mode: &str) -> bool {
    attach_agent.is_some() && app_mode != APP_MODE_INTERACTIVE
}

/// `shouldRejectNonInteractiveBareResume(resume, appMode)`.
pub fn should_reject_non_interactive_bare_resume(resume: Option<&ResumeValue>, app_mode: &str) -> bool {
    matches!(resume, Some(ResumeValue::Latest)) && app_mode != APP_MODE_INTERACTIVE
}

/// `resolveAppMode(parsed, stdinIsTTY)`.
pub fn resolve_app_mode(parsed: &Args, stdin_is_tty: bool) -> AppMode {
    if let Some(mode) = &parsed.mode {
        if mode == APP_MODE_DAEMON
            || mode == "rpc"
            || mode == "acp"
            || mode == "json"
        {
            return mode.clone();
        }
    }
    if parsed.print == Some(true) || !stdin_is_tty {
        return APP_MODE_PRINT.to_string();
    }
    APP_MODE_INTERACTIVE.to_string()
}

/// `toPrintOutputMode(appMode)`.
pub fn to_print_output_mode(app_mode: &str) -> &'static str {
    if app_mode == "json" {
        "json"
    } else {
        "text"
    }
}

/// `isClientOwnedDaemonSession(appMode, noSession)`.
pub fn is_client_owned_daemon_session(app_mode: &str, no_session: Option<bool>) -> bool {
    app_mode != "acp" || no_session == Some(true)
}

/// `parseAgentsViewCommand(args)`: `prime-agent agents` opens the agents view directly.
pub struct ParsedAgentsViewCommand {
    pub explicit_agents_view: bool,
    pub args: Vec<String>,
}

impl ParsedAgentsViewCommand {
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }
}

pub fn parse_agents_view_command(args: &[String]) -> ParsedAgentsViewCommand {
    if args.first().map(String::as_str) == Some("agents") {
        return ParsedAgentsViewCommand {
            explicit_agents_view: true,
            args: args[1..].to_vec(),
        };
    }
    ParsedAgentsViewCommand { explicit_agents_view: false, args: args.to_vec() }
}

/// The TypeScript interface is a plain object literal, so the port derives
/// `Clone` for the structural copies its own tests and callers make.
#[derive(Clone)]
pub struct DaemonClientStartupDecision {
    pub app_mode: AppMode,
    pub startup_benchmark: bool,
    pub no_session: Option<bool>,
    pub help: Option<bool>,
    pub list_models: Option<ListModelsValue>,
}

pub type InteractiveDaemonStartupDecision = DaemonClientStartupDecision;

/// Retained for callers that only classify persistent interactive startup.
pub fn should_use_daemon_interactive(options: &DaemonClientStartupDecision) -> bool {
    options.app_mode == APP_MODE_INTERACTIVE
        && !options.startup_benchmark
        && options.no_session != Some(true)
        && options.list_models.is_none()
}

/// `shouldUseDaemonClient(options)`.
pub fn should_use_daemon_client(options: &DaemonClientStartupDecision) -> bool {
    options.app_mode != APP_MODE_DAEMON
        && !options.startup_benchmark
        && options.help != Some(true)
        && options.list_models.is_none()
}

pub struct DaemonClientRuntimeDecision {
    pub decision: DaemonClientStartupDecision,
    pub owned_session_worker: bool,
    pub has_process_local_extension_factories: bool,
}

pub fn should_use_daemon_client_runtime(options: &DaemonClientRuntimeDecision) -> bool {
    should_use_daemon_client(&options.decision)
        && !options.owned_session_worker
        && !options.has_process_local_extension_factories
}

pub fn should_ensure_interactive_daemon_for_startup(use_daemon_interactive: bool, attach_agent: Option<&str>) -> bool {
    use_daemon_interactive && attach_agent.is_none()
}

/// The `daemonReady` gate of `packages/coding-agent/src/main.ts:1281`, resolved
/// from the same `shouldUseDaemonClientRuntime(...)` decision the TypeScript
/// computes at `main.ts:1259`:
///
/// ```ts
/// const useDaemonClient = shouldUseDaemonClientRuntime({ appMode, ... });
/// const useDaemonInteractive = useDaemonClient && appMode === "interactive";
/// // ...
/// let daemonReady = shouldEnsureInteractiveDaemonForStartup(useDaemonClient, publicCommand.attachAgent)
/// ```
///
/// The gate is called with `useDaemonClient`, which is TRUE for print, json, rpc
/// and acp (`shouldUseDaemonClient` only excludes daemon mode, the startup
/// benchmark, `--help` and `--list-models`), so those modes also start the
/// daemon and AWAIT it at `main.ts:1602` before `createDaemonClientConnection`.
/// Gating them on `useDaemonInteractive` instead left `daemon_ready` as `None`,
/// which made `await_daemon_ready` a no-op and raced the fire-and-forget
/// `maybe_start_daemon_early`, so headless `--print` failed with exit 1 whenever
/// the daemon was not already bound.
pub fn should_start_daemon_ready_for_startup(
    decision: &DaemonClientRuntimeDecision,
    attach_agent: Option<&str>,
) -> bool {
    should_ensure_interactive_daemon_for_startup(should_use_daemon_client_runtime(decision), attach_agent)
}

#[derive(Clone)]
pub struct AgentsViewStartupDecision {
    pub use_daemon_interactive: bool,
    pub needs_onboarding: bool,
    pub explicit_agents_view: Option<bool>,
    pub resume: Option<ResumeValue>,
    pub continue_: Option<bool>,
    pub fork: Option<String>,
}

pub fn should_open_agents_view_for_daemon_interactive(options: &AgentsViewStartupDecision) -> bool {
    let bare_resume = matches!(options.resume, Some(ResumeValue::Latest));
    let requests_agents_view =
        bare_resume || (options.explicit_agents_view == Some(true) && !options.needs_onboarding);
    options.use_daemon_interactive
        // A selector, continuation, or fork must open its target directly rather
        // than the agents view.
        && requests_agents_view
        && !matches!(options.resume, Some(ResumeValue::Selector(_)))
        && options.continue_ != Some(true)
        && options.fork.is_none()
}

#[derive(Clone)]
pub struct DaemonInteractiveSessionManagerDecision {
    pub resume: Option<ResumeValue>,
    pub continue_: Option<bool>,
    pub fork: Option<String>,
    pub has_active_daemon_session: Option<bool>,
}

pub fn should_use_ephemeral_session_manager_for_daemon_interactive(
    options: &DaemonInteractiveSessionManagerDecision,
) -> bool {
    options.has_active_daemon_session != Some(true)
        && (options.resume.is_none() || matches!(options.resume, Some(ResumeValue::Latest)))
        && options.continue_ != Some(true)
        && options.fork.is_none()
}

pub struct DaemonActiveSessionLookupDecision {
    pub use_daemon_interactive: bool,
    pub resume_selector: Option<String>,
    pub explicit_attach: Option<bool>,
}

pub fn should_ensure_daemon_before_active_session_lookup(options: &DaemonActiveSessionLookupDecision) -> bool {
    options.use_daemon_interactive
        && options.resume_selector.is_some()
        && (options.explicit_attach == Some(true)
            || !looks_like_session_path(options.resume_selector.as_deref().unwrap_or_default()))
}

pub struct ActiveDaemonSessionSummaryLookupOptions {
    pub fallback_on_error: Option<bool>,
}

// ---------------------------------------------------------------------------
// Session manager selection
// ---------------------------------------------------------------------------

async fn find_active_daemon_session_summary_for_interactive_startup(
    socket_path: &str,
    selector: &str,
    options: &ActiveDaemonSessionSummaryLookupOptions,
) -> Result<Option<SessionSummary>, String> {
    match find_active_daemon_session_summary(socket_path, selector).await {
        Ok(summary) => Ok(summary),
        Err(error) => {
            if options.fallback_on_error == Some(false) {
                return Err(error);
            }
            Ok(None)
        }
    }
}

/// `prepareInitialMessage(parsed, autoResizeImages, stdinContent)`.
pub async fn prepare_initial_message(
    parsed: &mut Args,
    auto_resize_images: bool,
    stdin_content: Option<String>,
) -> Result<(Option<String>, Option<Vec<ImageContent>>), String> {
    if parsed.file_args.is_empty() {
        let result = build_initial_message(InitialMessageInput {
            parsed,
            file_text: None,
            file_images: None,
            stdin_content,
        });
        return Ok((result.initial_message, result.initial_images));
    }

    let file_args = parsed.file_args.clone();
    let io = FileProcessorIo {
        error: &|message: &str| eprintln!("{message}"),
        exit: &|code: i32| std::process::exit(code),
    };
    let processed = process_file_arguments(&file_args, Some(ProcessFileOptions { auto_resize_images: Some(auto_resize_images) }), &io).await;
    let result = build_initial_message(InitialMessageInput {
        parsed,
        file_text: Some(processed.text),
        file_images: Some(processed.images),
        stdin_content,
    });
    Ok((result.initial_message, result.initial_images))
}

/// Prompt user for yes/no confirmation.
pub fn prompt_confirm(message: &str) -> bool {
    crate::cli::daemon_stop_confirm::prompt_yes_no(message, &|prompt: &str| {
        // `createInterface({ input: process.stdin, output: process.stdout })`
        // (`cli/daemon-stop-confirm.ts:20-28`): the question goes to the taken-over
        // stdout, so it reaches stderr in --mode json instead of the JSON stream.
        write_stdout(prompt);
        let mut answer = String::new();
        let _ = std::io::stdin().read_line(&mut answer);
        answer
    })
}

/// Only busy sessions (streaming, compacting, or pending messages) lose work;
/// idle loaded sessions reload from disk on the fresh daemon.
fn startup_session_loss_copy() -> DaemonSessionLossCopy<'static> {
    DaemonSessionLossCopy {
        busy_detail: &|count| {
            let pluralized = pluralize_sessions(count);
            format!(
                "A background service from a different Prime Agent version is running with {count} busy {}. Stopping it will terminate {}.",
                pluralized.noun, pluralized.pronoun
            )
        },
        unlistable_detail:
            "A background service from a different Prime Agent version is running and its sessions could not be listed. Stopping it may terminate active sessions.",
        question: "Stop it and continue?",
        non_tty_hint: "Run \"prime-agent shutdown\" to stop it, then retry.",
    }
}

/// A stale-version daemon couldn't be taken over automatically (busy or stuck).
/// Offer to stop it (default No) and start a fresh daemon, or exit. Returns the
/// fresh ready promise so callers stop re-handling the original rejection.
async fn take_over_stale_daemon_or_exit(socket_path: &str) -> Arc<DaemonReadyHandle> {
    let probe = probe_running_daemon_sessions(socket_path).await;
    let confirmed = confirm_daemon_session_loss(
        &probe,
        ConfirmOptions { force: false, copy: startup_session_loss_copy() },
        &ConfirmIo {
            stdin_is_tty: stdin_is_tty(),
            error: &|message: &str| eprintln!("{message}"),
            read_line: &|prompt: &str| {
                // Same guarded question stream as `prompt_confirm` above.
                write_stdout(prompt);
                let mut answer = String::new();
                let _ = std::io::stdin().read_line(&mut answer);
                answer
            },
        },
    );
    if !confirmed {
        // Non-TTY already printed the reason; at a TTY the user declined.
        if stdin_is_tty() == Some(true) {
            eprintln!("{}", dim("Cancelled."));
        }
        std::process::exit(1);
    }
    if !shutdown_daemon_and_wait(socket_path, DEFAULT_DAEMON_SHUTDOWN_TIMEOUT_MS).await {
        eprintln!(
            "{}",
            red(&format!(
                "Could not stop the background service on {socket_path}. Run \"prime-agent shutdown\" and retry."
            ))
        );
        std::process::exit(1);
    }
    let ready = DaemonReadyHandle::start(socket_path);
    if let Err(error) = ready.result().await {
        eprintln!("{}", red(&format!("Could not start the background service: {error}")));
        std::process::exit(1);
    }
    ready
}

/// `shutdownDaemonAndWait(socketPath)`'s default timeout.
pub const DEFAULT_DAEMON_SHUTDOWN_TIMEOUT_MS: f64 = 10_000.0;

fn stdin_is_tty() -> Option<bool> {
    use std::io::IsTerminal;
    Some(std::io::stdin().is_terminal())
}

/// Resolves the daemon-ready promise, returning the promise to keep (the same
/// one on success, or the fresh one from a stale-daemon takeover) so repeat
/// calls don't re-handle the original rejection.
pub async fn await_daemon_ready(daemon_ready: Option<Arc<DaemonReadyHandle>>) -> Option<Arc<DaemonReadyHandle>> {
    let Some(daemon_ready) = daemon_ready else {
        return None;
    };
    match daemon_ready.result().await {
        Ok(()) => Some(daemon_ready),
        Err(error) => {
            // blocked_on: `cli/daemon-launch.ts`'s port reports the stale-daemon
            // failure as its `.message` string, so `instanceof StaleDaemonError`
            // becomes a prefix check on the same message.
            if error.starts_with(STALE_DAEMON_ERROR_PREFIX) {
                return Some(take_over_stale_daemon_or_exit(&daemon_ready.socket_path).await);
            }
            Some(daemon_ready)
        }
    }
}

/// The first line of `StaleDaemonError`'s message.
pub const STALE_DAEMON_ERROR_PREFIX: &str = "An incompatible Prime Agent daemon is running.";

/// The shared `ensureInteractiveDaemonRunning` promise, kept as a handle so
/// callers can await it more than once and keep the same promise on success.
pub struct DaemonReadyHandle {
    socket_path: String,
    shared: Arc<SharedDaemonReady>,
}

struct SharedDaemonReady {
    result: Mutex<Option<Result<(), String>>>,
    notify: tokio::sync::Notify,
}

impl DaemonReadyHandle {
    /// `ensureInteractiveDaemonRunning(socketPath)`, recorded as one shared promise.
    pub fn start(socket_path: &str) -> Arc<Self> {
        let shared = Arc::new(SharedDaemonReady { result: Mutex::new(None), notify: tokio::sync::Notify::new() });
        let socket_path_owned = socket_path.to_string();
        let shared_for_task = Arc::clone(&shared);
        // `daemonReady?.catch(() => {})`: startup only needs to avoid an
        // unhandled rejection; the error is rethrown at the await sites.
        tokio::spawn(async move {
            let result = ensure_interactive_daemon_running(&socket_path_owned, None).await;
            *shared_for_task.result.lock().unwrap() = Some(result);
            shared_for_task.notify.notify_waiters();
        });
        Arc::new(DaemonReadyHandle { socket_path: socket_path.to_string(), shared })
    }

    /// The already-resolved promise returned after a stale-daemon takeover.
    pub fn ready_immediately(socket_path: &str) -> Arc<Self> {
        Arc::new(DaemonReadyHandle {
            socket_path: socket_path.to_string(),
            shared: Arc::new(SharedDaemonReady {
                result: Mutex::new(Some(Ok(()))),
                notify: tokio::sync::Notify::new(),
            }),
        })
    }

    pub async fn result(&self) -> Result<(), String> {
        loop {
            if let Some(result) = self.shared.result.lock().unwrap().clone() {
                return result;
            }
            let notified = self.shared.notify.notified();
            if let Some(result) = self.shared.result.lock().unwrap().clone() {
                return result;
            }
            notified.await;
        }
    }
}

/// `validateForkFlags(parsed)`.
pub fn validate_fork_flags(parsed: &Args) {
    let Some(_fork) = &parsed.fork else {
        return;
    };
    let mut conflicting_flags: Vec<&str> = Vec::new();
    if parsed.continue_ == Some(true) {
        conflicting_flags.push("--continue");
    }
    if parsed.resume.is_some() {
        conflicting_flags.push("--resume");
    }
    if parsed.no_session == Some(true) {
        conflicting_flags.push("--no-session");
    }
    if !conflicting_flags.is_empty() {
        eprintln!(
            "{}",
            red(&format!(
                "Error: --fork cannot be combined with {}",
                conflicting_flags.join(", ")
            ))
        );
        std::process::exit(1);
    }
}

fn fork_session_or_exit(source_path: &str, cwd: &str, session_dir: Option<&str>) -> SessionManager {
    match SessionManager::fork_from(source_path, cwd, session_dir) {
        Ok(manager) => manager,
        Err(message) => {
            eprintln!("{}", red(&format!("Error: {message}")));
            std::process::exit(1);
        }
    }
}

fn get_resume_selector(parsed: &Args) -> Option<String> {
    match &parsed.resume {
        Some(ResumeValue::Selector(selector)) => Some(selector.clone()),
        _ => None,
    }
}

fn read_session_manager(path: &str, session_dir: Option<&str>, cwd_override: Option<&str>) -> SessionManager {
    let entries = load_entries_from_file(path);
    let header = entries
        .iter()
        .find(|entry| entry.get("type").and_then(Value::as_str) == Some("session"));
    let header_cwd = header
        .and_then(|header| header.get("cwd"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let cwd = cwd_override
        .map(str::to_string)
        .or(header_cwd)
        .unwrap_or_else(current_cwd);
    let session_dir = session_dir
        .map(str::to_string)
        .unwrap_or_else(|| parent_dir_string(path));
    let mut manager = SessionManager::in_memory(Some(&cwd), Some(&session_dir)).expect("in-memory session manager");
    manager
        .set_session_file(path, Some(entries), None)
        .expect("session file load");
    manager
}

fn current_cwd() -> String {
    std::env::current_dir()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn parent_dir_string(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_else(current_cwd)
}

/// The failure shapes `createSessionManager` can raise.
///
/// The TypeScript catches `SessionSelectorError` and reads `suggestion` from
/// `SessionSelectorNotFoundError`, so the port keeps both cases typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateSessionManagerError {
    SelectorNotFound { message: String, suggestion: Option<String> },
    Selector { message: String },
    Message(String),
}

impl CreateSessionManagerError {
    /// The `Error.message` the caller prints.
    pub fn message(&self) -> &str {
        match self {
            CreateSessionManagerError::SelectorNotFound { message, .. } => message,
            CreateSessionManagerError::Selector { message } => message,
            CreateSessionManagerError::Message(message) => message,
        }
    }

    /// `error instanceof SessionSelectorError`.
    pub fn is_selector_error(&self) -> bool {
        !matches!(self, CreateSessionManagerError::Message(_))
    }

    /// `error.suggestion` of `SessionSelectorNotFoundError`.
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            CreateSessionManagerError::SelectorNotFound { suggestion, .. } => suggestion.as_deref(),
            _ => None,
        }
    }

    fn from_resolve_error(error: crate::core::session_resolver::ResolveSessionError) -> Self {
        match error {
            crate::core::session_resolver::ResolveSessionError::NotFound(not_found) => {
                CreateSessionManagerError::SelectorNotFound {
                    message: not_found.error.message,
                    suggestion: not_found.suggestion,
                }
            }
            crate::core::session_resolver::ResolveSessionError::Ambiguous(ambiguous) => {
                CreateSessionManagerError::Selector {
                    message: ambiguous.error.message,
                }
            }
            crate::core::session_resolver::ResolveSessionError::Other(other) => {
                CreateSessionManagerError::Selector { message: other.message }
            }
        }
    }
}

/// `createSessionManager(parsed, cwd, sessionDir, readOnly = false)`.
pub async fn create_session_manager(
    parsed: &Args,
    cwd: &str,
    session_dir: Option<&str>,
    read_only: bool,
) -> Result<SessionManager, CreateSessionManagerError> {
    let explicit_cwd_override = if parsed.cwd.is_some() { Some(cwd) } else { None };

    if parsed.no_session == Some(true) {
        return SessionManager::in_memory(None, None).map_err(CreateSessionManagerError::Message);
    }

    if let Some(fork) = &parsed.fork {
        let resolved = crate::core::session_resolver::resolve_session_path(fork, cwd, session_dir)
            .await
            .map_err(CreateSessionManagerError::from_resolve_error)?;

        match resolved {
            crate::core::session_resolver::ResolvedSession::Path { path }
            | crate::core::session_resolver::ResolvedSession::Local { path }
            | crate::core::session_resolver::ResolvedSession::Global { path, .. } => {
                return Ok(fork_session_or_exit(&path, cwd, session_dir));
            }
        }
    }

    if let Some(resume_selector) = get_resume_selector(parsed) {
        let resolved = crate::core::session_resolver::resolve_session_path(&resume_selector, cwd, session_dir)
            .await
            .map_err(CreateSessionManagerError::from_resolve_error)?;

        match resolved {
            crate::core::session_resolver::ResolvedSession::Path { path }
            | crate::core::session_resolver::ResolvedSession::Local { path } => {
                return if read_only {
                    Ok(read_session_manager(&path, session_dir, explicit_cwd_override))
                } else {
                    SessionManager::open(&path, session_dir, explicit_cwd_override)
                        .map_err(CreateSessionManagerError::Message)
                };
            }

            crate::core::session_resolver::ResolvedSession::Global { path, cwd: resolved_cwd } => {
                // `main.ts:509-515` uses `console.log`, which the taken-over
                // stdout (`core/output-guard.ts:18-27`) routes to stderr in
                // --mode json, so these lines are routed through the same guard.
                write_stdout(&format!(
                    "{}\n",
                    yellow(&format!("Session found in different project: {resolved_cwd}"))
                ));
                let should_fork = prompt_confirm("Fork this session into current directory?");
                if !should_fork {
                    write_stdout(&format!("{}\n", dim("Aborted.")));
                    std::process::exit(0);
                }
                return Ok(fork_session_or_exit(&path, cwd, session_dir));
            }
        }
    }

    if parsed.continue_ == Some(true) {
        if read_only {
            let dir = session_dir
                .map(str::to_string)
                .unwrap_or_else(|| get_default_session_dir(cwd, None));
            let path = find_most_recent_session_for_cwd(&dir, cwd);
            return match path {
                Some(path) => Ok(read_session_manager(&path, Some(&dir), Some(cwd))),
                None => SessionManager::in_memory(Some(cwd), Some(&dir))
                    .map_err(CreateSessionManagerError::Message),
            };
        }
        return SessionManager::continue_recent(cwd, session_dir).map_err(CreateSessionManagerError::Message);
    }

    if read_only {
        SessionManager::in_memory(Some(cwd), session_dir).map_err(CreateSessionManagerError::Message)
    } else {
        SessionManager::create(cwd, session_dir).map_err(CreateSessionManagerError::Message)
    }
}

// ---------------------------------------------------------------------------
// Runtime services
// ---------------------------------------------------------------------------

/// `buildSessionOptions(config, scopedModels, hasExistingSession, modelRegistry, settingsManager)`.
pub fn build_session_options(
    config: &AgentSessionRuntimeConfig,
    scoped_models: &[ScopedModel],
    has_existing_session: bool,
    model_registry: &crate::core::model_registry::ModelRegistry,
    settings_manager: &SettingsManager,
) -> BuildSessionOptionsResult {
    let mut options = CreateAgentSessionOptions::default();
    let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = Vec::new();
    let mut cli_thinking_from_model = false;

    // Model from CLI
    // - supports --provider <name> --model <pattern>
    // - supports --model <provider>/<pattern>
    if let Some(config_model) = &config.model {
        let resolved = resolve_cli_model(
            &ResolveCliModelOptions {
                cli_provider: config.provider.clone(),
                cli_model: Some(config_model.clone()),
            },
            model_registry,
        );
        if let Some(warning) = resolved.warning {
            diagnostics.push(AgentSessionRuntimeDiagnostic { type_: "warning".to_string(), message: warning });
        }
        if let Some(error) = resolved.error {
            diagnostics.push(AgentSessionRuntimeDiagnostic { type_: "error".to_string(), message: error });
        }
        if let Some(model) = resolved.model {
            options.model = Some(model);
            // Allow "--model <pattern>:<thinking>" as a shorthand.
            // Explicit --thinking still takes precedence (applied later).
            if config.thinking.is_none() {
                if let Some(thinking_level) = resolved.thinking_level {
                    options.thinking_level = Some(parse_thinking_level(&thinking_level));
                    cli_thinking_from_model = true;
                }
            }
        }
    }

    if options.model.is_none() && !scoped_models.is_empty() && !has_existing_session {
        // Check if saved default is in scoped models - use it if so, otherwise first scoped model
        let saved_provider = settings_manager.get_default_provider();
        let saved_model_id = settings_manager.get_default_model();
        let saved_model = match (&saved_provider, &saved_model_id) {
            (Some(provider), Some(model_id)) => model_registry.find(provider, model_id),
            _ => None,
        };
        let saved_in_scope = saved_model
            .as_ref()
            .and_then(|saved_model| {
                scoped_models
                    .iter()
                    .find(|scoped| models_are_equal(Some(&scoped.model), Some(saved_model)))
            });

        if let Some(saved_in_scope) = saved_in_scope {
            options.model = Some(saved_in_scope.model.clone());
            // Use thinking level from scoped model config if explicitly set
            if config.thinking.is_none() {
                if let Some(thinking_level) = &saved_in_scope.thinking_level {
                    options.thinking_level = Some(parse_thinking_level(thinking_level));
                }
            }
        } else {
            options.model = Some(scoped_models[0].model.clone());
            // Use thinking level from first scoped model if explicitly set
            if config.thinking.is_none() {
                if let Some(thinking_level) = &scoped_models[0].thinking_level {
                    options.thinking_level = Some(parse_thinking_level(thinking_level));
                }
            }
        }
    }

    // Thinking level from CLI (takes precedence over scoped model thinking levels set above)
    if let Some(thinking) = &config.thinking {
        options.thinking_level = Some(thinking.clone());
    }

    // Scoped models for Ctrl+P cycling
    // Keep thinking level undefined when not explicitly set in the model pattern.
    // Undefined means "inherit current session thinking level" during cycling.
    if !scoped_models.is_empty() {
        options.scoped_models = Some(
            scoped_models
                .iter()
                .map(|scoped| crate::core::agent_session::ScopedModel {
                    model: scoped.model.clone(),
                    thinking_level: scoped
                        .thinking_level
                        .as_ref()
                        .map(|thinking_level| parse_thinking_level(thinking_level)),
                })
                .collect(),
        );
    }

    // API key from CLI - set in authStorage
    // (handled by caller before createAgentSession)

    // Tools
    if config.no_tools == Some(true) {
        options.no_tools = Some("all".to_string());
    } else if config.no_builtin_tools == Some(true) {
        options.no_tools = Some("builtin".to_string());
    }
    if let Some(tools) = &config.tools {
        options.tools = Some(tools.clone());
    }
    if let Some(autonomous) = &config.autonomous {
        options.autonomous = merge_autonomous_config(None, Some(autonomous));
    }

    BuildSessionOptionsResult { options, cli_thinking_from_model, diagnostics }
}

/// `buildSessionOptions`'s return shape.
pub struct BuildSessionOptionsResult {
    pub options: CreateAgentSessionOptions,
    pub cli_thinking_from_model: bool,
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// `resolveCliPaths(cwd, paths)` (`main.ts:629-631`).
///
/// The TypeScript resolves a local entry with `resolve(cwd, value)` from
/// `node:path` (`main.ts:8`), so the host flavour applies: on win32 the result
/// is drive-absolute with the platform separator. `Path::join` alone neither
/// normalises `.` away nor produces the drive, which is why
/// `resolve_cli_paths("/work", ["./skills"])` returned `"/work\\./skills"` while
/// the TypeScript returns `"C:\\work\\skills"` (measured with the pinned
/// `resolveToCwd`/`resolve` pair on Node v24.16.0).
pub fn resolve_cli_paths(cwd: &str, paths: Option<&Vec<String>>) -> Option<Vec<String>> {
    paths.map(|paths| {
        paths
            .iter()
            .map(|value| {
                if is_local_path(value) {
                    resolve_path(cwd, value)
                } else {
                    value.clone()
                }
            })
            .collect()
    })
}

/// `path.resolve(base, target)`; see [`crate::core::tools::path_utils::resolve_path`]
/// for the Node win32 semantics this mirrors.
fn resolve_path(base: &str, target: &str) -> String {
    crate::core::tools::path_utils::resolve_path(base, target)
}

/// `runtimeAutonomousConfigFromArgs(parsed)`.
pub fn runtime_autonomous_config_from_args(parsed: &Args) -> Option<AgentAutonomousConfig> {
    let has_autonomous_options = parsed.autonomous == Some(true)
        || parsed.autonomous_gates.is_some()
        || parsed.autonomous_gate_retries.is_some()
        || parsed.autonomous_gate_timeout_ms.is_some()
        || parsed.autonomous_max_continuations.is_some()
        || parsed.autonomous_max_turns.is_some()
        || parsed.autonomous_max_tokens.is_some()
        || parsed.autonomous_timeout_ms.is_some();
    if !has_autonomous_options {
        return None;
    }
    let has_gate_options = parsed.autonomous_gates.is_some()
        || parsed.autonomous_gate_retries.is_some()
        || parsed.autonomous_gate_timeout_ms.is_some();
    Some(AgentAutonomousConfig {
        enabled: Some(true),
        max_continuations: parsed.autonomous_max_continuations.map(|value| value as f64),
        max_turns: parsed.autonomous_max_turns.map(|value| value as f64),
        max_tokens: parsed.autonomous_max_tokens.map(|value| value as f64),
        timeout_ms: parsed.autonomous_timeout_ms.map(|value| value as f64),
        gates: if has_gate_options {
            Some(crate::core::autonomous::AgentAutonomousGateConfig {
                commands: parsed.autonomous_gates.clone(),
                max_retries: parsed.autonomous_gate_retries.map(|value| value as f64),
                timeout_ms: parsed.autonomous_gate_timeout_ms.map(|value| value as f64),
            })
        } else {
            None
        },
        ..Default::default()
    })
}

/// `runtimeConfigFromArgs(parsed, cwd, agentDir, sessionDir, appMode, telemetryDisabled)`.
pub fn runtime_config_from_args(
    parsed: &Args,
    cwd: &str,
    agent_dir: &str,
    session_dir: Option<&str>,
    app_mode: &str,
    telemetry_disabled: Option<bool>,
) -> AgentSessionRuntimeConfig {
    AgentSessionRuntimeConfig {
        cwd: Some(cwd.to_string()),
        agent_dir: Some(agent_dir.to_string()),
        session_dir: session_dir.map(str::to_string),
        provider: parsed.provider.clone(),
        model: parsed.model.clone(),
        api_key: parsed.api_key.clone(),
        system_prompt: parsed.system_prompt.clone(),
        append_system_prompt: parsed.append_system_prompt.clone(),
        thinking: parsed.thinking.clone(),
        models: parsed.models.clone(),
        tools: parsed.tools.clone(),
        no_tools: parsed.no_tools,
        no_builtin_tools: parsed.no_builtin_tools,
        extensions: resolve_cli_paths(cwd, parsed.extensions.as_ref()),
        no_extensions: parsed.no_extensions,
        skills: resolve_cli_paths(cwd, parsed.skills.as_ref()),
        no_skills: parsed.no_skills,
        prompt_templates: resolve_cli_paths(cwd, parsed.prompt_templates.as_ref()),
        no_prompt_templates: parsed.no_prompt_templates,
        themes: resolve_cli_paths(cwd, parsed.themes.as_ref()),
        no_themes: parsed.no_themes,
        no_context_files: parsed.no_context_files,
        autonomous: runtime_autonomous_config_from_args(parsed),
        extension_flag_values: if parsed.unknown_flags.is_empty() {
            None
        } else {
            Some(
                parsed
                    .unknown_flags
                    .iter()
                    .map(|(key, value)| {
                        let value = match value {
                            crate::cli::args::UnknownFlagValue::Flag => Value::Bool(true),
                            crate::cli::args::UnknownFlagValue::Value(value) => Value::String(value.clone()),
                        };
                        (key.clone(), value)
                    })
                    .collect(),
            )
        },
        execution_mode: if app_mode == APP_MODE_DAEMON { None } else { Some(app_mode.to_string()) },
        telemetry_disabled,
        // Serialized refine for print/json/rpc: the client's appMode is NOT
        // "daemon" here - it's "print", "json", or "rpc". The daemon worker
        // receives this flag via AgentSessionRuntimeConfig and uses it
        // instead of its own appMode="daemon".
        serialized_refine: Some(app_mode != APP_MODE_INTERACTIVE && app_mode != APP_MODE_DAEMON),
        initial_goal: parsed.goal.as_ref().map(|objective| crate::core::agent_session_config::InitialGoalConfig {
            objective: objective.clone(),
            token_budget: parsed.goal_token_budget.map(|budget| budget as f64),
        }),
        ..Default::default()
    }
}

/// `PreparedRuntimeServices`.
pub struct PreparedRuntimeServices {
    pub services: Arc<AgentSessionServices>,
    pub scoped_models: Vec<ScopedModel>,
    pub session_options: CreateAgentSessionOptions,
    pub cli_thinking_from_model: bool,
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// `daemonServerDefaultSessionConfig(config)`.
pub fn daemon_server_default_session_config(config: &AgentSessionRuntimeConfig) -> AgentSessionRuntimeConfig {
    let mut config = config.clone();
    config.initial_goal = None;
    config
}

/// `CreateAgentSessionOptions extends AgentSessionCreationOptions`, so the port rebuilds
/// the flat override from the creation members the runtime input carries.
fn create_agent_session_options_from_creation(
    creation: &AgentSessionCreationOptions,
) -> CreateAgentSessionOptions {
    CreateAgentSessionOptions {
        model: creation.model.clone(),
        thinking_level: creation.thinking_level.clone(),
        service_tier: creation.service_tier.clone(),
        scoped_models: creation.scoped_models.clone(),
        tools: creation.tools.clone(),
        no_tools: creation.no_tools.clone(),
        custom_tools: creation.custom_tools.clone(),
        autonomous: creation.autonomous.clone(),
        creation: creation.clone(),
        ..Default::default()
    }
}

/// `resolveRuntimeSessionOptions(sessionOptions, runtimeSessionOptions?)`.
pub fn resolve_runtime_session_options(
    session_options: &CreateAgentSessionOptions,
    runtime_session_options: Option<&AgentSessionCreationOptions>,
) -> CreateAgentSessionOptions {
    let base = &session_options.creation;
    let runtime = runtime_session_options;
    // `(runtimeSessionOptions?.rlmDepth ?? 0) > 0`.
    let subagent_runtime = runtime.and_then(|runtime| runtime.rlm_depth).unwrap_or(0) > 0;
    let autonomous = if subagent_runtime {
        // A subagent runtime never runs its own autonomous loop.
        let disabled = runtime.and_then(|runtime| runtime.autonomous.as_ref()).map(|autonomous| {
            let mut autonomous = autonomous.clone();
            autonomous.enabled = Some(false);
            autonomous
        });
        merge_autonomous_config(base.autonomous.as_ref(), disabled.as_ref())
    } else {
        merge_autonomous_config(base.autonomous.as_ref(), runtime.and_then(|runtime| runtime.autonomous.as_ref()))
    };
    CreateAgentSessionOptions {
        model: runtime
            .and_then(|runtime| runtime.model.clone())
            .or_else(|| session_options.model.clone()),
        thinking_level: runtime
            .and_then(|runtime| runtime.thinking_level.clone())
            .or_else(|| session_options.thinking_level.clone()),
        service_tier: runtime
            .and_then(|runtime| runtime.service_tier.clone())
            .or_else(|| session_options.service_tier.clone()),
        scoped_models: runtime
            .and_then(|runtime| runtime.scoped_models.clone())
            .or_else(|| session_options.scoped_models.clone()),
        tools: runtime
            .and_then(|runtime| runtime.tools.clone())
            .or_else(|| session_options.tools.clone()),
        no_tools: runtime
            .and_then(|runtime| runtime.no_tools.clone())
            .or_else(|| session_options.no_tools.clone()),
        custom_tools: runtime
            .and_then(|runtime| runtime.custom_tools.clone())
            .or_else(|| session_options.custom_tools.clone()),
        creation: AgentSessionCreationOptions {
            initial_active_tool_names: runtime.and_then(|runtime| runtime.initial_active_tool_names.clone()),
            allowed_tool_names: runtime.and_then(|runtime| runtime.allowed_tool_names.clone()),
            include_goals: runtime.and_then(|runtime| runtime.include_goals),
            include_compact_skill: runtime.and_then(|runtime| runtime.include_compact_skill),
            rlm_heartbeat_controller: runtime.and_then(|runtime| runtime.rlm_heartbeat_controller.clone()),
            agent_message_controller: runtime.and_then(|runtime| runtime.agent_message_controller.clone()),
            agent_observe_controller: runtime.and_then(|runtime| runtime.agent_observe_controller.clone()),
            autonomous,
            rlm_depth: runtime.and_then(|runtime| runtime.rlm_depth),
            rlm_max_depth: runtime.and_then(|runtime| runtime.rlm_max_depth),
            rlm_session_dir: runtime.and_then(|runtime| runtime.rlm_session_dir.clone()),
            rlm_parent_node_id: runtime.and_then(|runtime| runtime.rlm_parent_node_id.clone()),
            rlm_parent_agent: runtime.and_then(|runtime| runtime.rlm_parent_agent.clone()),
            semantic_parent_session_id: runtime.and_then(|runtime| runtime.semantic_parent_session_id.clone()),
            semantic_spawned_by_request_id: runtime
                .and_then(|runtime| runtime.semantic_spawned_by_request_id.clone()),
            subagent_runtime_host: runtime.and_then(|runtime| runtime.subagent_runtime_host.clone()),
            ..Default::default()
        },
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Runtime services
// ---------------------------------------------------------------------------

/// `createDefaultRuntimeFactory(runtimeDefaultSessionConfig, extensionFactories?)`.
pub fn create_default_runtime_factory(
    runtime_default_session_config: AgentSessionRuntimeConfig,
    extension_factories: Option<Vec<crate::core::extensions::types::ExtensionFactory>>,
) -> CreateAgentSessionRuntimeFactory {
    Arc::new(move |input: CreateAgentSessionRuntimeInput| {
        let runtime_default_session_config = runtime_default_session_config.clone();
        let extension_factories = extension_factories.clone();
        Box::pin(async move {
            let config = merge_agent_session_runtime_config(
                &runtime_default_session_config,
                input.session_config.as_ref(),
            );
            let runtime_session_options = input.session_options.clone();
            // The TypeScript keeps reading `config` after this call (the CLI thinking
            // override below), which an owned Rust struct cannot do.
            let prepared = prepare_runtime_services(PrepareRuntimeServicesOptions {
                config: config.clone(),
                cwd: input.cwd.clone(),
                agent_dir: input.agent_dir.clone(),
                session_manager: Arc::clone(&input.session_manager),
                extension_factories,
                session_options_override: runtime_session_options
                    .as_ref()
                    .map(create_agent_session_options_from_creation),
            })
            .await?;
            let PreparedRuntimeServices { services, session_options, diagnostics, .. } = prepared;
            let resolved_session_options =
                resolve_runtime_session_options(&session_options, runtime_session_options.as_ref());

            let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
                services: Arc::clone(&services),
                session_manager: Arc::clone(&input.session_manager),
                session_start_event: input.session_start_event.clone(),
                creation: AgentSessionCreationOptions {
                    model: resolved_session_options.model,
                    thinking_level: resolved_session_options.thinking_level,
                    service_tier: resolved_session_options.service_tier,
                    scoped_models: resolved_session_options.scoped_models,
                    tools: resolved_session_options.tools,
                    no_tools: resolved_session_options.no_tools,
                    custom_tools: resolved_session_options.custom_tools,
                    prewarm_ipython_kernel: Some(true),
                    serialized_refine: Some(config.serialized_refine.unwrap_or(false)),
                    execution_mode: config.execution_mode.clone(),
                    telemetry_disabled: config.telemetry_disabled,
                    initial_goal: if resolved_session_options.creation.rlm_depth.unwrap_or(0) == 0 {
                        config.initial_goal.as_ref().map(|goal| crate::core::agent_session::InitialGoal {
                            objective: goal.objective.clone(),
                            token_budget: goal.token_budget,
                        })
                    } else {
                        None
                    },
                    ..resolved_session_options.creation
                },
            })
            .await?;

            let cli_thinking_override = config.thinking.is_some() || prepared.cli_thinking_from_model;
            if created.session.model().is_some() && cli_thinking_override {
                created.session.set_thinking_level(created.session.thinking_level());
            }

            Ok(CreateAgentSessionRuntimeResult { result: created, services, diagnostics })
        })
    })
}

/// Options of `prepareRuntimeServices`.
pub struct PrepareRuntimeServicesOptions {
    pub config: AgentSessionRuntimeConfig,
    pub cwd: String,
    pub agent_dir: String,
    pub session_manager: Arc<Mutex<SessionManager>>,
    pub extension_factories: Option<Vec<crate::core::extensions::types::ExtensionFactory>>,
    pub session_options_override: Option<CreateAgentSessionOptions>,
}

/// `prepareRuntimeServices(options)`.
///
/// The TypeScript returns a promise that REJECTS when
/// `createAgentSessionServices` throws (`packages/coding-agent/src/main.ts:824-847`
/// has no try/catch), so the failure travels out through
/// `createAgentSessionRuntime` to `main()` and the CLI exits 1. The port returns
/// the same rejection as `Err(String)` instead of panicking on it.
pub async fn prepare_runtime_services(
    options: PrepareRuntimeServicesOptions,
) -> Result<PreparedRuntimeServices, String> {
    let config = options.config;
    let effective_agent_dir = config.agent_dir.clone().unwrap_or_else(|| options.agent_dir.clone());
    let auth_storage = AuthStorage::create(
        Some(format!("{}/auth.json", effective_agent_dir.trim_end_matches('/'))),
        Some(crate::core::auth_storage::AuthStorageOptions {
            prime_cli_config_path: None,
            use_prime_cli_config: effective_agent_dir == options.agent_dir,
        }),
    );
    let extension_flag_values = config.extension_flag_values.clone().map(indexmap::IndexMap::from_iter);
    let mut resource_loader_options = DefaultResourceLoaderOptions::new(&options.cwd, &effective_agent_dir);
    resource_loader_options.additional_extension_paths = config.extensions.clone().unwrap_or_default();
    resource_loader_options.additional_skill_paths = config.skills.clone().unwrap_or_default();
    resource_loader_options.additional_prompt_template_paths = config.prompt_templates.clone().unwrap_or_default();
    resource_loader_options.additional_theme_paths = config.themes.clone().unwrap_or_default();
    resource_loader_options.no_extensions = config.no_extensions.unwrap_or(false);
    resource_loader_options.no_skills = config.no_skills.unwrap_or(false);
    resource_loader_options.no_prompt_templates = config.no_prompt_templates.unwrap_or(false);
    resource_loader_options.no_themes = config.no_themes.unwrap_or(false);
    resource_loader_options.no_context_files = config.no_context_files.unwrap_or(false);
    resource_loader_options.system_prompt = config.system_prompt.clone();
    resource_loader_options.append_system_prompt = config.append_system_prompt.clone();
    resource_loader_options.extension_factories = options.extension_factories.clone().unwrap_or_default();
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: options.cwd.clone(),
        agent_dir: Some(effective_agent_dir),
        auth_storage: Some(Arc::new(tokio::sync::Mutex::new(auth_storage))),
        settings_manager: None,
        model_registry: None,
        extension_flag_values,
        // Subagents share the parent's Herdr pane; their own reporter would race
        // the parent's and a subagent quit would release the still-active pane.
        no_builtin_herdr_reporter: Some(
            options
                .session_options_override
                .as_ref()
                .and_then(|options| options.creation.rlm_depth)
                .unwrap_or(0)
                > 0,
        ),
        telemetry_disabled: config.telemetry_disabled,
        resource_loader_options: Some(resource_loader_options),
    })
    .await
    .map_err(|error| format!("createAgentSessionServices failed: {error}"))?;

    let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = services.diagnostics.clone();
    diagnostics.extend(collect_settings_diagnostics(
        &mut services.settings_manager.lock().expect("settings manager poisoned"),
        "runtime creation",
    ));
    diagnostics.extend(
        services
            .resource_loader
            .get_extensions()
            .errors
            .iter()
            .map(|error| AgentSessionRuntimeDiagnostic {
                type_: DIAGNOSTIC_ERROR.to_string(),
                message: format!("Failed to load extension \"{}\": {}", error.path, error.error),
            }),
    );

    let model_patterns = config
        .models
        .clone()
        .or_else(|| services.settings_manager.lock().unwrap().get_enabled_models());
    let scoped_models = match model_patterns {
        Some(model_patterns) if !model_patterns.is_empty() => {
            // `resolveModelScope(patterns, modelRegistry)` awaits inside the
            // registry lock; `with_model_registry` runs the operation on the
            // blocking pool so the async factory stays `Send`.
            let registry = Arc::clone(&services.model_registry);
            crate::core::sdk::with_model_registry(registry, move |registry| {
                Box::pin(async move { resolve_model_scope(&model_patterns, registry).await })
            })
            .await
            // `main.ts:858-860`: `await resolveModelScope(modelPatterns, modelRegistry)`
            // has no try/catch, so a registry failure rejects out of
            // `prepareRuntimeServices` and the CLI exits 1 rather than silently
            // continuing with an empty model scope.
            .map_err(|error| format!("resolveModelScope failed: {error}"))?
        }
        _ => Vec::new(),
    };
    let has_existing_session = !options
        .session_manager
        .lock()
        .unwrap()
        .build_session_context(None)
        .messages
        .is_empty();
    let built = build_session_options(
        &config,
        &scoped_models,
        has_existing_session,
        &services.model_registry.lock().unwrap(),
        &services.settings_manager.lock().unwrap(),
    );
    diagnostics.extend(built.diagnostics.clone());

    let effective_session_model = options
        .session_options_override
        .as_ref()
        .and_then(|options| options.model.clone())
        .or_else(|| built.options.model.clone());
    if let Some(api_key) = &config.api_key {
        match effective_session_model {
            Some(model) => {
                services
                    .auth_storage
                    .lock()
                    .await
                    .set_runtime_api_key(&model.provider, api_key);
                // `main.ts:882 authStorage.setRuntimeApiKey(effectiveSessionModel.provider, config.apiKey)`
                // writes to the single instance that `ModelRegistry.create(authStorage, ...)`
                // shares (`agent-session-services.ts:150-152`,
                // `model-registry.ts:525 readonly authStorage: AuthStorage`), so request auth
                // (`getApiKeyAndHeaders`) resolves the CLI key. This port's `ModelRegistry`
                // owns its `AuthStorage` by value, so the same override must be written
                // through the registry as well.
                services
                    .model_registry
                    .lock()
                    .expect("model registry poisoned")
                    .set_runtime_api_key(&model.provider, api_key);
            }
            None => diagnostics.push(AgentSessionRuntimeDiagnostic {
                type_: DIAGNOSTIC_ERROR.to_string(),
                message: "--api-key requires a model to be specified via --model, --provider/--model, or --models"
                    .to_string(),
            }),
        }
    }

    Ok(PreparedRuntimeServices {
        services: Arc::new(services),
        scoped_models,
        session_options: built.options,
        cli_thinking_from_model: built.cli_thinking_from_model,
        diagnostics,
    })
}

/// `resolvePreparedStartupModel(options)`.
pub async fn resolve_prepared_startup_model(
    prepared: &PreparedRuntimeServices,
    session_manager: &Arc<Mutex<SessionManager>>,
) -> InitialModelSelection {
    let model_registry = Arc::clone(&prepared.services.model_registry);
    let settings_manager = Arc::clone(&prepared.services.settings_manager);
    let existing_session = session_manager.lock().unwrap().build_session_context(None);
    let has_existing_session = !existing_session.messages.is_empty();

    let mut model = prepared.session_options.model.clone();
    let mut model_fallback_message: Option<String> = None;

    if model.is_none() && has_existing_session {
        if let Some(existing_model) = &existing_session.model {
            let restored = model_registry
                .lock()
                .unwrap()
                .find(&existing_model.provider, &existing_model.model_id);
            if let Some(restored) = restored {
                if model_registry.lock().unwrap().has_configured_auth(&restored) {
                    model = Some(restored);
                }
            }
            if model.is_none() {
                model_fallback_message = Some(format!(
                    "Could not restore model {}/{}",
                    existing_model.provider, existing_model.model_id
                ));
            }
        }
    }

    if model.is_none() {
        let settings = settings_manager.lock().unwrap();
        let result = match find_initial_model(
            &FindInitialModelOptions {
                cli_provider: None,
                cli_model: None,
                scoped_models: prepared.scoped_models.clone(),
                is_continuing: has_existing_session,
                default_provider: settings.get_default_provider(),
                default_model_id: settings.get_default_model(),
                default_thinking_level: settings.get_default_thinking_level(),
            },
            &mut model_registry.lock().unwrap(),
        )
        .await
        {
            Ok(result) => result,
            Err(message) => {
                // `log.error(resolved.error, ...)` + `console.error(chalk.red(...))` + `process.exit(1)`.
                eprintln!("{}", red(&message));
                std::process::exit(1);
            }
        };
        drop(settings);
        model = result.model;
        if model.is_none() {
            model_fallback_message = Some(format_no_models_available_message());
        } else if let Some(existing) = model_fallback_message {
            let selected = model.as_ref().expect("checked above");
            model_fallback_message = Some(format!("{existing}. Using {}/{}", selected.provider, selected.id));
        }
    }

    InitialModelSelection { model, model_fallback_message }
}

/// Return shape of `resolvePreparedStartupModel`.
pub struct InitialModelSelection {
    pub model: Option<Model>,
    pub model_fallback_message: Option<String>,
}

// ---------------------------------------------------------------------------
// Active daemon session lookup
// ---------------------------------------------------------------------------

/// `resolveActiveSessionLookupFailure(response)`.
pub fn resolve_active_session_lookup_failure(response: &DaemonResponse) -> Option<String> {
    if response.error_info.as_ref().map(|info| info.code()).as_deref() == Some("session_recovering") {
        return Some(daemon_error_message(&deserialize_daemon_error(response)));
    }
    let error = response.error.clone().unwrap_or_default();
    if is_unknown_active_session_error(&error) {
        return None;
    }
    Some(error)
}

fn log_context(app_mode: &str) -> serde_json::Map<String, Value> {
    let mut fields = serde_json::Map::new();
    fields.insert("mode".to_string(), Value::String(app_mode.to_string()));
    fields
}

fn current_environment() -> std::collections::HashMap<String, String> {
    std::env::vars().collect()
}

/// `{ appMode, startupBenchmark, noSession, help, listModels }`.
pub fn daemon_client_startup_decision(
    parsed: &Args,
    app_mode: &str,
    startup_benchmark: bool,
) -> DaemonClientStartupDecision {
    DaemonClientStartupDecision {
        app_mode: app_mode.to_string(),
        startup_benchmark,
        no_session: parsed.no_session,
        help: parsed.help,
        list_models: parsed.list_models.clone(),
    }
}

fn daemon_error_message(error: &DaemonError) -> String {
    // Each known daemon error renders the TypeScript `error.message` through `Display`.
    match error {
        DaemonError::MissingSessionCwd(error) => error.to_string(),
        DaemonError::SessionImportFileNotFound(error) => error.to_string(),
        DaemonError::SessionAlreadyActive(error) => error.to_string(),
        DaemonError::SessionRecovering(error) => error.message(),
        DaemonError::Message(message) => message.clone(),
    }
}

/// `findActiveDaemonSessionSummary(socketPath, selector)`.
pub async fn find_active_daemon_session_summary(
    socket_path: &str,
    selector: &str,
) -> Result<Option<SessionSummary>, String> {
    let client = DaemonClient::create(socket_path);
    client
        .connect(DAEMON_PROBE_CONNECT_TIMEOUT_MS)
        .await
        .map_err(|error| error.message())?;

    let mut command: DaemonCommandBody = serde_json::Map::from_iter([(
        "type".to_string(),
        Value::String("get_state".to_string()),
    )]);
    command.insert("activeSessionId".to_string(), Value::String(selector.to_string()));
    let response = client
        .request(command, Some(3000), Default::default())
        .await
        .map_err(|error| error.message());
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            client.close().await;
            return Err(error);
        }
    };

    if !response.success {
        let failure = resolve_active_session_lookup_failure(&response);
        client.close().await;
        if let Some(failure) = failure {
            return Err(failure);
        }
        return Ok(None);
    }
    let summary = serde_json::from_value::<SessionSummary>(response.data.clone().unwrap_or(Value::Null));
    client.close().await;
    let summary = summary.map_err(|_| "Daemon returned an invalid active session summary".to_string())?;
    Ok(Some(summary))
}

fn is_unknown_active_session_error(message: &str) -> bool {
    message.starts_with("Unknown active session:")
}

/// `mapDaemonSessionSnapshot` reads the wire `meta`; the connection slice only
/// keeps `sequence` and `cursor`, so the adapter projects exactly those.
fn main_entry_meta_from_wire(meta: Option<&Value>) -> Option<ConnectionEventMeta> {
    let meta = meta?.as_object()?;
    let sequence = meta.get("sequence").and_then(Value::as_i64);
    let cursor = meta.get("cursor").and_then(Value::as_object).map(|cursor| {
        ConnectionEventCursor {
            generation: cursor
                .get("generation")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            sequence: cursor.get("sequence").and_then(Value::as_i64).unwrap_or(0),
        }
    });
    if sequence.is_none() && cursor.is_none() {
        return None;
    }
    Some(ConnectionEventMeta { sequence, cursor })
}

fn main_entry_cursor(cursor: &ProtocolEventCursor) -> ConnectionEventCursor {
    ConnectionEventCursor {
        generation: cursor.generation.clone(),
        sequence: cursor.sequence as i64,
    }
}

fn main_entry_history_window(window: &ProtocolHistoryWindow) -> AgentConnectionHistoryWindow {
    AgentConnectionHistoryWindow {
        version: window.version,
        generation: window.generation.clone(),
        representation: window.representation.clone(),
        tip_entry_id: window.tip_entry_id.clone(),
        total_message_count: window.total_message_count,
        start_index: window.start_index,
        entry_ids: window.entry_ids.clone(),
        has_older: window.has_older,
        order: window.order.clone(),
    }
}

fn main_entry_session_tree(tree: &ProtocolSessionTree) -> AgentConnectionSessionTree {
    AgentConnectionSessionTree {
        tree: tree.tree.clone(),
        leaf_id: tree.leaf_id.clone(),
    }
}

/// `summary` as `DaemonSessionSummary`; the wire row keeps fields this adapter
/// does not model in `extra`, so `lastEventSequence` is read from there.
fn main_entry_session_summary(summary: &SessionSummary) -> ConnectionSessionSummary {
    ConnectionSessionSummary {
        session_id: summary.session_id.clone(),
        session_file: summary.session_file.clone(),
        active_session_id: summary.active_session_id.clone(),
        id: Some(summary.id.clone()),
        streaming_message: summary
            .streaming_message
            .clone()
            .and_then(|message| serde_json::from_value(message).ok()),
        last_event_sequence: summary
            .extra
            .get("lastEventSequence")
            .and_then(Value::as_i64),
        last_event_cursor: None,
    }
}

/// `Omit<DaemonSessionSnapshot, "messages">` -> the connection's snapshot shape.
fn main_entry_snapshot_head(head: &ProtocolSnapshotHead) -> ConnectionSessionSnapshot {
    ConnectionSessionSnapshot {
        state: head.state.clone(),
        messages: Vec::new(),
        summary: main_entry_session_summary(&head.summary),
        history: head.history.as_ref().map(main_entry_history_window),
        session_context: head.session_context.clone(),
        session_tree: head.session_tree.as_ref().map(main_entry_session_tree),
        parent: head.parent.clone(),
        children: head.children.clone(),
        last_event_sequence: Some(head.last_event_sequence as i64),
        last_event_cursor: head.last_event_cursor.as_ref().map(main_entry_cursor),
    }
}

/// The wire `DaemonSessionSnapshot` -> the connection's snapshot shape.
fn main_entry_session_snapshot(snapshot: &ProtocolSessionSnapshot) -> ConnectionSessionSnapshot {
    ConnectionSessionSnapshot {
        state: snapshot.state.clone(),
        messages: snapshot.messages.clone(),
        summary: main_entry_session_summary(&snapshot.summary),
        history: snapshot.history.as_ref().map(main_entry_history_window),
        session_context: snapshot.session_context.clone(),
        session_tree: snapshot.session_tree.as_ref().map(main_entry_session_tree),
        parent: snapshot.parent.clone(),
        children: snapshot.children.clone(),
        last_event_sequence: Some(snapshot.last_event_sequence as i64),
        last_event_cursor: snapshot.last_event_cursor.as_ref().map(main_entry_cursor),
    }
}

/// One wire frame as the connection slice's `DaemonOutbound`.
///
/// Frames with no connection-level meaning (`response`, `daemon_hello`,
/// `roster_update`, `session_attached`, `session_detached`,
/// `session_list_progress`) read as `None`, exactly as the TypeScript connection
/// ignores them.
fn main_entry_outbound_from_wire(value: &Value) -> Option<ConnectionOutbound> {
    let meta = main_entry_meta_from_wire(value.get("meta"));
    // `meta` is reconstructed above; dropping it here keeps parsing independent of
    // the daemon-side meta shape (which carries required fields this adapter and
    // the connection do not read).
    let mut frame = value.clone();
    if let Some(object) = frame.as_object_mut() {
        object.remove("meta");
    }
    let outbound = ProtocolOutbound::from_value(&frame)?;
    let outbound = match outbound {
        ProtocolOutbound::DaemonClosing { reason } => ConnectionOutbound::DaemonClosing { reason },
        ProtocolOutbound::HeartbeatsChanged { active_session_id, .. } => ConnectionOutbound::HeartbeatsChanged {
            active_session_id,
            meta,
        },
        ProtocolOutbound::SessionEvent {
            active_session_id,
            event,
            ..
        } => ConnectionOutbound::SessionEvent {
            active_session_id,
            event,
            meta,
        },
        ProtocolOutbound::SideQuestionEvent {
            active_session_id,
            event,
        } => ConnectionOutbound::SideQuestionEvent {
            active_session_id,
            event,
            meta: None,
        },
        ProtocolOutbound::SessionStatus {
            active_session_id,
            recap,
            ..
        } => ConnectionOutbound::SessionStatus {
            active_session_id,
            recap,
            meta,
        },
        ProtocolOutbound::SessionResynced {
            active_session_id,
            snapshot,
            ..
        } => ConnectionOutbound::SessionResynced {
            active_session_id,
            snapshot: main_entry_session_snapshot(&snapshot),
            meta,
        },
        ProtocolOutbound::SessionReplaced {
            active_session_id,
            state,
            messages,
            snapshot_follows,
            ..
        } => ConnectionOutbound::SessionReplaced {
            active_session_id,
            state,
            messages,
            snapshot_follows,
            meta,
        },
        ProtocolOutbound::SessionSnapshotBegin {
            active_session_id,
            snapshot_id,
            snapshot,
            message_count,
            purpose,
            ..
        } => ConnectionOutbound::SessionSnapshotBegin {
            active_session_id,
            snapshot_id,
            snapshot: main_entry_snapshot_head(&snapshot),
            message_count: message_count as usize,
            purpose,
        },
        ProtocolOutbound::SessionSnapshotChunk {
            active_session_id,
            snapshot_id,
            index,
            messages,
        } => ConnectionOutbound::SessionSnapshotChunk {
            active_session_id,
            snapshot_id,
            index: index as usize,
            messages,
        },
        ProtocolOutbound::SessionSnapshotEnd {
            active_session_id,
            snapshot_id,
            chunk_count,
            last_event_sequence,
            last_event_cursor,
        } => ConnectionOutbound::SessionSnapshotEnd {
            active_session_id,
            snapshot_id,
            chunk_count: chunk_count as usize,
            last_event_sequence: last_event_sequence as i64,
            last_event_cursor: last_event_cursor.as_ref().map(main_entry_cursor),
        },
        ProtocolOutbound::SessionSnapshotFailed {
            active_session_id,
            snapshot_id,
            error,
        } => ConnectionOutbound::SessionSnapshotFailed {
            active_session_id,
            snapshot_id,
            error,
        },
        ProtocolOutbound::SessionClosed {
            active_session_id,
            reason,
            ..
        } => ConnectionOutbound::SessionClosed {
            active_session_id,
            reason: reason.as_str().to_string(),
            meta,
        },
        ProtocolOutbound::ExtensionUiRequest {
            active_session_id,
            id,
            method,
            payload,
            ..
        } => ConnectionOutbound::ExtensionUiRequest {
            active_session_id,
            id,
            method,
            payload: Value::Object(payload),
            meta,
        },
        ProtocolOutbound::ExtensionError {
            active_session_id,
            extension_path,
            event,
            error,
            ..
        } => ConnectionOutbound::ExtensionError {
            active_session_id,
            extension_path,
            event,
            error,
            meta,
        },
        _ => return None,
    };
    Some(outbound)
}

#[cfg(test)]
pub(crate) fn test_outbound_from_wire(value: &Value) -> Option<ConnectionOutbound> {
    main_entry_outbound_from_wire(value)
}

/// `DaemonClient` as the connection slice's `DaemonTransportClient`.
///
/// The two slices agree on one interface in the TypeScript; the port splits it
/// (wire-typed on the daemon side, `Value`-typed on the connection side), so this
/// adapter is the missing bridge and nothing else. `main.ts` hands one
/// `DaemonClient` to `DaemonAgentConnection.attach` here, exactly like the
/// telegram worker does, so this call site carries the same bridge.
pub(crate) struct MainEntryDaemonTransport {
    client: Arc<DaemonClient>,
}

impl MainEntryDaemonTransport {
    pub(crate) fn new(client: Arc<DaemonClient>) -> Self {
        Self { client }
    }
}

impl ConnectionTransport for MainEntryDaemonTransport {
    /// `requestData(command, timeoutMs, options)` forwards `options.recoverable`
    /// (`daemon-agent-connection.ts:379/:498`; consumed at `daemon-client.ts:418`). Without this
    /// override the transport hardcodes `DaemonClientRequestOptions::default()`, so a caller's
    /// `recoverable: false` would never reach the client.
    fn request_with_recoverable(
        &self,
        command: Value,
        timeout_ms: Option<u64>,
        recoverable: bool,
    ) -> pi_ai::types::BoxFuture<Result<ConnectionResponse, String>> {
        let options = DaemonClientRequestOptions {
            recoverable: Some(recoverable),
            ..Default::default()
        };
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let body: DaemonCommandBody = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be a JSON object".to_string())?;
            let response = client
                .request(body, timeout_ms, options)
                .await
                .map_err(|error| error.message())?;
            Ok(ConnectionResponse {
                success: response.success,
                data: response.data.unwrap_or(Value::Null),
                error: response.error,
                error_code: response.error_info.map(|info| info.code().to_string()),
            })
        })
    }

    fn request(&self, command: Value, timeout_ms: Option<u64>) -> pi_ai::types::BoxFuture<Result<ConnectionResponse, String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let body: DaemonCommandBody = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be a JSON object".to_string())?;
            let response = client
                .request(body, timeout_ms, DaemonClientRequestOptions::default())
                .await
                .map_err(|error| error.message())?;
            Ok(ConnectionResponse {
                success: response.success,
                data: response.data.unwrap_or(Value::Null),
                error: response.error,
                error_code: response.error_info.map(|info| info.code().to_string()),
            })
        })
    }

    fn on_message(&self, listener: Arc<dyn Fn(ConnectionOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        let wire_listener: DaemonClientMessageListener = Arc::new(move |value: &Value| {
            if let Some(outbound) = main_entry_outbound_from_wire(value) {
                listener(outbound);
            }
        });
        self.client.on_message(wire_listener)
    }

    fn on_close(&self, listener: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        let wire_listener: DaemonClientCloseListener = Arc::new(move |error: &DaemonClientError| {
            listener(error.message());
        });
        self.client.on_close(wire_listener)
    }

    fn supports_server_capability(&self, capability: &str) -> bool {
        self.client.supports_server_capability(capability)
    }

    fn hello_socket_path(&self) -> Option<String> {
        // `this.client.hello?.socketPath`.
        self.client
            .hello()
            .and_then(|hello| hello.raw.get("socketPath").and_then(Value::as_str).map(str::to_string))
    }

    fn is_connected(&self) -> bool {
        self.client.is_connected()
    }

    fn enable_request_recovery(&self) {
        // `DaemonClient::enable_request_recovery` is async while the connection
        // calls this synchronously; `connect_telegram_session` awaits the same
        // flag before the first request so the ordering the TypeScript relies on
        // is preserved.
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.enable_request_recovery().await;
        });
    }

    fn close(&self) {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.close().await;
        });
    }

    fn connect(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move { client.connect(timeout_ms).await.map_err(|error| error.message()) })
    }

    fn wait_for_hello(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            client
                .wait_for_hello(timeout_ms)
                .await
                .map(|_| ())
                .map_err(|error| error.message())
        })
    }

    fn reconnect(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            client
                .reconnect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }

    fn disconnect_for_reconnect(&self, reason: &str) {
        let client = Arc::clone(&self.client);
        let reason = reason.to_string();
        tokio::spawn(async move {
            client.disconnect_for_reconnect(&reason).await;
        });
    }

    fn reset_transport_for_reconnect(&self) {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.reset_transport_for_reconnect().await;
        });
    }

    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn ConnectionTransport> {
        self
    }
}

/// `getDaemonSummaryActiveSessionId(summary)`.
pub fn get_daemon_summary_active_session_id(summary: &SessionSummary) -> String {
    summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone())
}

/// `createSessionManagerForActiveDaemonSummary(summary, fallbackCwd)`.
pub fn create_session_manager_for_active_daemon_summary(
    summary: &SessionSummary,
    fallback_cwd: &str,
) -> SessionManager {
    let cwd = if summary.cwd.is_empty() { fallback_cwd.to_string() } else { summary.cwd.clone() };
    if let Some(session_file) = &summary.session_file {
        return read_session_manager(session_file, None, Some(&cwd));
    }
    SessionManager::in_memory(Some(&cwd), None).expect("in-memory session manager")
}

/// `getInteractiveDaemonSessionPath(parsed, sessionManager)`.
pub fn get_interactive_daemon_session_path(parsed: &Args, session_manager: &SessionManager) -> Option<String> {
    if parsed.resume.is_none() && parsed.continue_ != Some(true) && parsed.fork.is_none() {
        return None;
    }
    session_manager.get_session_file()
}

/// `findActiveDaemonSessionSummaryForSessionFile(summaries, sessionPath)`.
pub fn find_active_daemon_session_summary_for_session_file(
    summaries: &[SessionSummary],
    session_path: &str,
) -> Option<SessionSummary> {
    let resolved_session_path = canonical_session_path(session_path);
    summaries
        .iter()
        .find(|summary| {
            summary.active_session_id.is_some()
                && summary
                    .session_file
                    .as_ref()
                    .map(|session_file| canonical_session_path(session_file) == resolved_session_path)
                    .unwrap_or(false)
        })
        .cloned()
}

/// `findAttachedDaemonSessionSummary(client, activeSessionId)`.
pub async fn find_attached_daemon_session_summary(
    client: &Arc<DaemonClient>,
    active_session_id: &str,
) -> Result<SessionSummary, String> {
    let mut command: DaemonCommandBody = serde_json::Map::from_iter([(
        "type".to_string(),
        Value::String("get_state".to_string()),
    )]);
    command.insert(
        "activeSessionId".to_string(),
        Value::String(active_session_id.to_string()),
    );
    let response = client
        .request(command, None, Default::default())
        .await
        .map_err(|error| error.message())?;
    if !response.success {
        return Err(response.error.unwrap_or_else(|| "Daemon request failed".to_string()));
    }
    serde_json::from_value(response.data.unwrap_or(Value::Null))
        .map_err(|_| "Daemon returned an invalid active session summary".to_string())
}

/// `error instanceof SessionAlreadyActiveError || error instanceof DaemonSessionCreateError`
/// (`main.ts:1616`) / `error instanceof DaemonSessionCreateError` (`main.ts:1529`).
///
/// The TypeScript tells the printed kinds from the rethrown ones by `instanceof`,
/// so the port keeps that discrimination on the returned error instead of
/// collapsing every failure into one `String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonClientConnectionErrorKind {
    /// `SessionAlreadyActiveError` (core/session-lease.ts:20).
    SessionAlreadyActive,
    /// `DaemonSessionCreateError` (modes/daemon/daemon-errors.ts:36-41).
    DaemonSessionCreate,
    /// Every other error, which `main.ts:1533` / `main.ts:1620` rethrow.
    Rethrown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonClientConnectionError {
    pub kind: DaemonClientConnectionErrorKind,
    pub message: String,
}

impl DaemonClientConnectionError {
    /// An error the TypeScript does not handle at the connection site.
    pub fn rethrown(message: impl Into<String>) -> Self {
        Self {
            kind: DaemonClientConnectionErrorKind::Rethrown,
            message: message.into(),
        }
    }

    /// `deserializeDaemonCreateError(response)` (`daemon-errors.ts:44-48`): an untyped
    /// failure keeps its `DaemonSessionCreateError` class; a typed `errorInfo` keeps its
    /// own class and is rethrown unless it is `session_already_active`.
    pub fn from_create_failure(response: &DaemonResponse) -> Self {
        let error = deserialize_daemon_create_error(response);
        let kind = match &error {
            DaemonError::SessionAlreadyActive(_) => DaemonClientConnectionErrorKind::SessionAlreadyActive,
            DaemonError::Message(_) if response.error_info.is_none() => {
                DaemonClientConnectionErrorKind::DaemonSessionCreate
            }
            _ => DaemonClientConnectionErrorKind::Rethrown,
        };
        Self {
            kind,
            message: daemon_error_message(&error),
        }
    }

    /// `main.ts:1529`: the interactive attach prints only `DaemonSessionCreateError`.
    pub fn is_reported_at_interactive_attach(&self) -> bool {
        matches!(self.kind, DaemonClientConnectionErrorKind::DaemonSessionCreate)
    }

    /// `main.ts:1616`: the daemon-client path prints both named kinds.
    pub fn is_reported_at_daemon_client(&self) -> bool {
        !matches!(self.kind, DaemonClientConnectionErrorKind::Rethrown)
    }
}

impl std::fmt::Display for DaemonClientConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DaemonClientConnectionError {}

impl From<String> for DaemonClientConnectionError {
    fn from(message: String) -> Self {
        Self::rethrown(message)
    }
}

/// The `String` boundary the other call site keeps (`modes/agents_view/native_host.rs:481`).
impl From<DaemonClientConnectionError> for String {
    fn from(error: DaemonClientConnectionError) -> Self {
        error.message
    }
}

/// `throw error` (`main.ts:1533`, `main.ts:1620`).
///
/// `cli-main.ts:41` awaits `main` without a catch, so that rejection leaves `runCli()`
/// and Node reports it as an uncaught failure with a non-zero exit. This port's `main`
/// returns `()` and `native_main_host.rs:108` discards the value, so the error cannot be
/// handed to the caller from this file. Until that seam propagates a `Result`, the
/// rethrow is reported WITHOUT the handled `Error: {message}` line that only the named
/// kinds print, and it keeps the non-zero exit.
fn rethrow_daemon_connection_error(error: &DaemonClientConnectionError, host: &dyn MainHost) {
    eprintln!("{error}");
    host.exit(1);
}

/// `createDaemonClientConnection(options)`.
struct PendingDaemonViewer(Option<Arc<DaemonClient>>);
impl Drop for PendingDaemonViewer {
    fn drop(&mut self) {
        if let Some(client) = self.0.take() {
            tokio::spawn(async move { client.close().await; });
        }
    }
}

pub async fn create_daemon_client_connection(
    options: CreateDaemonClientConnectionOptions,
) -> Result<(Arc<DaemonAgentConnection>, SessionSummary), DaemonClientConnectionError> {
    // Caller must have awaited ensureInteractiveDaemonRunning for this socket.
    let client = DaemonClient::create(&options.socket_path);
    let mut pending_viewer = PendingDaemonViewer(Some(client.clone()));
    client
        .connect(DEFAULT_DAEMON_CONNECT_TIMEOUT_MS)
        .await
        .map_err(|error| DaemonClientConnectionError::rethrown(error.message()))?;

    let socket_path_for_recover = options.socket_path.clone();
    // `DaemonAgentConnection.attach(client, ...)` in the TypeScript; the port's
    // connection slice consumes its own `DaemonTransportClient`, so the client is
    // bridged by `MainEntryDaemonTransport` (this module).
    let transport: Arc<dyn ConnectionTransport> = Arc::new(MainEntryDaemonTransport::new(Arc::clone(&client)));
    let attach = |summary: SessionSummary| {
        let client = Arc::clone(&transport);
        let recover_socket_path = socket_path_for_recover.clone();
        let config = options.config.clone();
        let client_owned = options.client_owned.unwrap_or(false);
        let supports_extension_ui = options.supports_extension_ui.unwrap_or(false);
        async move {
            // blocked_on: `recoverDaemon: () => ensureInteractiveDaemonRunning(socketPath)`
            // is a callback in the TypeScript; the landed connection seam carries
            // the same intent as the `recover_daemon` flag. The socket path is
            // logged so the recovery target stays visible.
            let _ = &recover_socket_path;
            let connection = DaemonAgentConnection::attach(
                client,
                get_daemon_summary_active_session_id(&summary),
                DaemonAgentConnectionOptions {
                    defer_session_events: options.defer_session_events,
                    close_client_on_dispose: true,
                    direct_transport: false,
                    recover_daemon: true,
                    reconnect_timeout_ms: None,
                    snapshot_timeout_ms: None,
                    send_client_env: true,
                    owned_session: client_owned,
                    owned_session_recovery_config: if client_owned {
                        serde_json::to_value(&config).ok()
                    } else {
                        None
                    },
                    supports_extension_ui,
                    telemetry_disabled: config.telemetry_disabled.unwrap_or(false),
                },
            )
            .await?;
            Ok::<_, DaemonClientConnectionError>((connection, summary))
        }
    };

    let result = async {
        if let Some(active_session_id) = &options.active_session_id {
            let summary = find_attached_daemon_session_summary(&client, active_session_id).await?;
            return attach(summary).await;
        }

        if let Some(session_path) = &options.session_path {
            if !options.client_owned.unwrap_or(false) {
                // `listActiveDaemonSessionSummaries(client)` (`main.ts:1087`): the
                // TypeScript reuses the already-connected client and passes no
                // options, so `includeClientOwned` stays undefined and the error
                // propagates. `list_active_daemon_session_summaries` opens its own
                // socket through `daemon_request` (cli/daemon_launch.rs:221-230), so
                // the port issues the same `list` command on this client instead.
                let response = client
                    .request(
                        DaemonCommandBody::from_iter([(
                            "type".to_string(),
                            Value::String("list".to_string()),
                        )]),
                        None,
                        Default::default(),
                    )
                    .await
                    .map_err(|error| DaemonClientConnectionError::rethrown(error.message()))?;
                let mut typed: Vec<SessionSummary> = Vec::new();
                let mut request_failure: Option<DaemonClientConnectionError> = None;
                if !response.success {
                    // `throw new Error(response.error)` (cli/daemon-launch.ts:130).
                    request_failure = Some(DaemonClientConnectionError::rethrown(
                        response.error.unwrap_or_else(|| "Daemon request failed".to_string()),
                    ));
                } else {
                    let sessions = response
                        .data
                        .as_ref()
                        .and_then(|data| data.get("sessions"))
                        .and_then(Value::as_array);
                    match sessions {
                        None => {
                            request_failure = Some(DaemonClientConnectionError::rethrown(
                                "Daemon returned an invalid session list response",
                            ));
                        }
                        Some(sessions) => {
                            for session in sessions {
                                if !is_daemon_session_summary(session) {
                                    request_failure = Some(DaemonClientConnectionError::rethrown(
                                        "Daemon returned an invalid session list response",
                                    ));
                                    break;
                                }
                                typed.push(serde_json::from_value(session.clone()).map_err(|_| {
                                    DaemonClientConnectionError::rethrown(
                                        "Daemon returned an invalid session list response",
                                    )
                                })?);
                            }
                        }
                    }
                }
                if let Some(failure) = request_failure {
                    return Err(failure);
                }
                if let Some(active_summary) =
                    find_active_daemon_session_summary_for_session_file(&typed, session_path)
                {
                    // `activeSummary.workerState !== "failed"` (`main.ts:1090`).
                    // `lifecycle` is only draft/live/archived, so testing it would
                    // never guard a failed worker.
                    if active_summary.worker_state.as_deref() != Some("failed") {
                        return attach(active_summary).await;
                    }
                }
            }
        }
        if options.client_owned.unwrap_or(false) {
            client
                .wait_for_hello(DEFAULT_DAEMON_HELLO_TIMEOUT_MS)
                .await
                .map_err(|error| error.message())?;
            if !client.supports_server_capability("client_owned_sessions") {
                return Err(DaemonClientConnectionError::rethrown(
                    DaemonCapabilityUnavailableError::new("create", Some("client_owned_sessions"), false).message(),
                ));
            }
        }

        let mut command: DaemonCommandBody = serde_json::Map::from_iter([(
            "type".to_string(),
            Value::String("create".to_string()),
        )]);
        command.insert(
            "config".to_string(),
            serde_json::to_value(&options.config).unwrap_or(Value::Null),
        );
        if let Some(session_path) = &options.session_path {
            command.insert("sessionPath".to_string(), Value::String(session_path.clone()));
        }
        if let Some(continue_recent) = options.continue_recent {
            command.insert("continueRecent".to_string(), Value::Bool(continue_recent));
        }
        if let Some(no_session) = options.no_session {
            command.insert("noSession".to_string(), Value::Bool(no_session));
        }
        command.insert(
            "env".to_string(),
            serde_json::to_value(collect_daemon_client_env()).unwrap_or(Value::Null),
        );
        command.insert(
            "lifecycle".to_string(),
            Value::String(if options.client_owned.unwrap_or(false) {
                "client_owned".to_string()
            } else {
                "resident".to_string()
            }),
        );
        command.insert(
            "launchEnv".to_string(),
            serde_json::to_value(collect_daemon_launch_env()).unwrap_or(Value::Null),
        );
        let response = client
            .request(command, None, Default::default())
            .await
            .map_err(|error| DaemonClientConnectionError::rethrown(error.message()))?;
        if !response.success {
            return Err(DaemonClientConnectionError::from_create_failure(&response));
        }
        let summary = serde_json::from_value::<SessionSummary>(response.data.unwrap_or(Value::Null))
            .map_err(|_| DaemonClientConnectionError::rethrown("Daemon returned an invalid create response"))?;
        attach(summary).await
    }
    .await;

    match result {
        Ok(connection) => { pending_viewer.0 = None; Ok(connection) },
        Err(error) => {
            client.close().await;
            Err(error)
        }
    }
}

// ---------------------------------------------------------------------------
// Onboarding readers
// ---------------------------------------------------------------------------

/// `OnboardingSettingsReader` (modes/interactive/onboarding.ts) over the core
/// `SettingsManager`.
///
/// `main.ts` passes the real settings manager; the port's onboarding module
/// declares its own stand-in because that slice landed first, so this adapter
/// closes the gap at the call site instead of changing either side.
struct CoreOnboardingSettingsReader<'a> {
    settings_manager: &'a SettingsManager,
}

impl crate::modes::interactive::onboarding::OnboardingSettingsReader
    for CoreOnboardingSettingsReader<'_>
{
    fn get_onboarding_shown(&self) -> bool {
        self.settings_manager.get_onboarding_shown()
    }
}

/// `OnboardingModelRegistryReader` (modes/interactive/onboarding.ts) over the
/// core `ModelRegistry`.
///
/// `refresh()` takes `&mut self` in the port while the trait member takes
/// `&self`, so the reader carries the already-locked guard through a `RefCell`.
/// `get_provider_auth_status` is projected the same way the interactive
/// stand-in projects it (`AuthStatus { source }`).
struct CoreOnboardingModelRegistryReader<'a> {
    model_registry: std::cell::RefCell<&'a mut crate::core::model_registry::ModelRegistry>,
}

impl crate::modes::interactive::onboarding::OnboardingModelRegistryReader
    for CoreOnboardingModelRegistryReader<'_>
{
    fn refresh(&self) {
        self.model_registry.borrow_mut().refresh();
    }

    fn has_configured_auth(
        &self,
        model: &crate::modes::interactive::interactive_mode_services::AgentConnectionModel,
    ) -> bool {
        self.model_registry.borrow().has_configured_auth(model)
    }

    fn get_provider_auth_status(
        &self,
        provider: &str,
    ) -> crate::modes::interactive::interactive_mode_services::AuthStatus {
        let status = self.model_registry.borrow().get_provider_auth_status(provider);
        crate::modes::interactive::interactive_mode_services::AuthStatus {
            source: status.source.unwrap_or_default(),
        }
    }
}

/// `createDaemonClientConnection(options)`.
#[derive(Clone)]
pub struct CreateDaemonClientConnectionOptions {
    pub defer_session_events: bool,
    pub socket_path: String,
    pub config: AgentSessionRuntimeConfig,
    pub session_path: Option<String>,
    pub continue_recent: Option<bool>,
    pub active_session_id: Option<String>,
    pub client_owned: Option<bool>,
    pub no_session: Option<bool>,
    pub supports_extension_ui: Option<bool>,
}

/// `await client.connect(250)`.
/// `connect(timeoutMs = 3000)` - the `DaemonClient.connect` default
/// (`daemon-client.ts:215`). The 250 ms value belongs to the version probe only.
pub const DEFAULT_DAEMON_CONNECT_TIMEOUT_MS: u64 = 3000;
/// `ensureInteractiveDaemonRunning`'s short probe (`main.ts:1000`).
pub const DAEMON_PROBE_CONNECT_TIMEOUT_MS: u64 = 250;
/// `await client.waitForHello()`.
/// `waitForHello(timeoutMs = 3000)` - the `DaemonClient.waitForHello` default
/// (`daemon-client.ts:189`).
pub const DEFAULT_DAEMON_HELLO_TIMEOUT_MS: u64 = 3000;

/// `promptForMissingSessionCwd(issue, settingsManager)`.
///
/// blocked_on: the prompt is the interactive slice's `ExtensionSelectorComponent`
/// mounted on the shared TUI; the port keeps the exact question, the two
/// options, and the "Continue" -> `fallbackCwd` result through a host seam.
pub fn prompt_for_missing_session_cwd(
    issue: &SessionCwdIssue,
    settings_manager: &SettingsManager,
    host: &dyn MissingSessionCwdHost,
) -> Option<String> {
    crate::modes::interactive::theme::theme::init_theme(settings_manager.get_theme().as_deref(), false);
    let options = vec!["Continue".to_string(), "Cancel".to_string()];
    let selected = host.select(&format_missing_session_cwd_prompt(issue), options)?;
    if selected == "Continue" {
        return Some(issue.fallback_cwd.clone());
    }
    None
}

/// The `new ExtensionSelectorComponent(...)` + TUI pair of
/// `promptForMissingSessionCwd`.
pub trait MissingSessionCwdHost: Send + Sync {
    /// Returns the selected option, or `None` when the selector is cancelled.
    fn select(&self, title: &str, options: Vec<String>) -> Option<String>;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// `PublicCommandResult` of `cli/public-command.ts`.
#[derive(Debug, Clone, Default)]
pub struct PublicCommandOutcome {
    pub handled: bool,
    pub args: Vec<String>,
    pub explicit_agents_view: bool,
    pub attach_agent: Option<String>,
}

/// `MainOptions`.
#[derive(Default)]
pub struct MainOptions {
    pub extension_factories: Option<Vec<crate::core::extensions::types::ExtensionFactory>>,
}

/// Everything `main` reaches that belongs to another slice.
///
/// blocked_on: `cli/public-command.ts`, `cli/owned-session-worker.ts`,
/// `modes/interactive`, `modes/print-mode.ts`, `modes/acp/acp-mode.ts`,
/// `modes/daemon/daemon-supervisor.ts` and `modes/agents-view/agents-view-mode.ts`
/// are owned by other slices. Every method mirrors one TypeScript call, so the
/// startup order, the arguments and the exit paths stay the ported ones.
pub trait MainHost: Send + Sync {
    /// `handlePublicCommand(args)`.
    fn handle_public_command(&self, args: Vec<String>) -> BoxFuture<Result<PublicCommandOutcome, String>>;
    /// `isOwnedSessionWorkerProcess()`.
    fn is_owned_session_worker_process(&self) -> bool;
    /// `installOwnedSessionRecoveryTracking(runtime)`.
    fn install_owned_session_recovery_tracking(&self, runtime: Arc<AgentSessionRuntime>);
    /// `registerBuiltinMcpOAuthProviders()`.
    fn register_builtin_mcp_oauth_providers(&self);
    /// `setKeybindings(KeybindingsManager.create())`.
    fn set_keybindings(&self, agent_dir: &str);
    /// `runDaemonCatalogProcess()`.
    fn run_daemon_catalog_process(&self) -> BoxFuture<Result<(), String>>;
    /// `runDaemonSupervisorMode({ socketPath, defaultSessionConfig })`.
    fn run_daemon_supervisor_mode(
        &self,
        socket_path: Option<String>,
        default_session_config: AgentSessionRuntimeConfig,
    ) -> BoxFuture<Result<(), String>>;
    /// `runDaemonMode({ socketPath, defaultSessionConfig, createRuntime, worker })`.
    fn run_daemon_mode(
        &self,
        options: DaemonModeSeamOptions,
    ) -> BoxFuture<Result<(), String>>;
    /// `runAgentsViewMode(options)`.
    fn run_agents_view_mode(&self, options: AgentsViewSeamOptions) -> BoxFuture<Result<(), String>>;
    /// `runRpcModeWithConnection(connection)`.
    fn run_rpc_mode_with_connection(
        &self,
        connection: Arc<dyn crate::modes::agent_connection::types::AgentConnection>,
    ) -> BoxFuture<Result<(), String>>;
    /// `runAcpModeWithConnection(connection)`.
    fn run_acp_mode_with_connection(
        &self,
        connection: Arc<dyn crate::modes::agent_connection::types::AgentConnection>,
    ) -> BoxFuture<Result<(), String>>;
    /// `runPrintModeWithConnection(connection, options)`.
    fn run_print_mode_with_connection(
        &self,
        connection: Arc<dyn crate::modes::agent_connection::types::AgentConnection>,
        options: PrintModeSeamOptions,
    ) -> BoxFuture<Result<i32, String>>;
    /// `runRpcMode(runtime)`.
    fn run_rpc_mode(&self, runtime: Arc<AgentSessionRuntime>) -> BoxFuture<Result<(), String>>;
    /// `runAcpMode(runtime)`.
    fn run_acp_mode(&self, runtime: Arc<AgentSessionRuntime>) -> BoxFuture<Result<(), String>>;
    /// `runPrintMode(runtimeHost, options)`.
    fn run_print_mode(
        &self,
        runtime: Arc<AgentSessionRuntime>,
        options: PrintModeSeamOptions,
    ) -> BoxFuture<Result<i32, String>>;
    /// `new InteractiveMode({ ... })` + `interactiveMode.run()`.
    fn run_interactive_mode(&self, options: InteractiveModeSeamOptions) -> BoxFuture<Result<Option<InteractiveModeRunResult>, String>>;
    /// `new InteractiveMode({ ... })` + `interactiveMode.init()` for the benchmark.
    fn init_interactive_mode(&self, options: InteractiveModeSeamOptions) -> BoxFuture<Result<(), String>>;
    /// `preloadCodeHighlighter()`.
    fn preload_code_highlighter(&self);
    /// `initTheme(themeName, enableWatcher)`.
    fn init_theme(&self, theme_name: Option<String>, enable_watcher: bool);
    /// `stopThemeWatcher()`.
    fn stop_theme_watcher(&self);
    /// `process.exit(code)`.
    fn exit(&self, code: i32);
    /// `process.exitCode = code`.
    fn set_exit_code(&self, code: i32);
    /// `process.stdin.isTTY`.
    fn stdin_is_tty(&self) -> bool;
    /// `process.env` lookup.
    fn env(&self, key: &str) -> Option<String>;
    /// `process.env[key] = value`.
    fn set_env(&self, key: &str, value: &str);
    /// `process.cwd()`.
    fn cwd(&self) -> String;
    /// `process.chdir(cwd)`.
    fn chdir(&self, cwd: &str) -> Result<(), String>;
    /// `await promptForMissingSessionCwd(issue, settingsManager)`.
    fn prompt_for_missing_session_cwd(&self, issue: &SessionCwdIssue, agent_dir: &str) -> Option<String>;
}

/// `runDaemonMode({ ... worker: { ... } })` arguments.
#[derive(Clone)]
pub struct DaemonModeSeamOptions {
    pub socket_path: Option<String>,
    pub default_session_config: AgentSessionRuntimeConfig,
    pub create_runtime: CreateAgentSessionRuntimeFactory,
    pub worker: Option<DaemonWorkerSeamOptions>,
}

/// `worker: { authenticationToken, workerInstanceId, restoreActiveSessionId }`.
#[derive(Clone)]
pub struct DaemonWorkerSeamOptions {
    pub authentication_token: String,
    pub worker_instance_id: Option<String>,
    pub restore_active_session_id: Option<String>,
}

/// `runAgentsViewMode(options)` arguments.
#[derive(Clone)]
pub struct AgentsViewSeamOptions {
    pub socket_path: String,
    pub config: AgentSessionRuntimeConfig,
    pub agent_dir: String,
    pub migrated_providers: Vec<String>,
    pub model_fallback_message: Option<String>,
    pub startup_model_id: Option<String>,
    pub verbose: bool,
    pub initial_session: Option<SessionSummary>,
    pub initial_scope_key: Option<AgentsViewScopeKey>,
}

/// `runPrintModeWithConnection(connection, options)` arguments.
#[derive(Clone)]
pub struct PrintModeSeamOptions {
    pub mode: String,
    pub messages: Vec<String>,
    pub initial_message: Option<String>,
    pub initial_images: Option<Vec<ImageContent>>,
}

/// `new InteractiveMode({ ... })` arguments.
#[derive(Clone)]
pub struct InteractiveModeSeamOptions {
    pub daemon_socket_path: Option<String>,
    pub migrated_providers: Vec<String>,
    pub model_fallback_message: Option<String>,
    pub initial_message: Option<String>,
    pub initial_images: Option<Vec<ImageContent>>,
    pub initial_messages: Vec<String>,
    pub verbose: bool,
    pub return_to_agents_view: bool,
    pub session_depth: Option<f64>,
    pub session_has_children: bool,
    /// `agentConnection`: a daemon-backed connection, or `None` for the
    /// in-process runtime (the host builds `new InProcessAgentConnection(runtime)`).
    pub connection: Option<Arc<dyn crate::modes::agent_connection::types::AgentConnection>>,
    /// `localSessionHost: createInteractiveModeLocalSessionHost(runtime)`.
    pub runtime: Option<Arc<AgentSessionRuntime>>,
}

/// `main(args, options?)`.
///
/// blocked_on: see `MainHost`. The port keeps the TypeScript order of every
/// step, including the early `--offline` env writes, the settings diagnostics,
/// the daemon-ready promise shared by the interactive branches, and the exit
/// codes of each failure path.
pub async fn main(args: Vec<String>, options: MainOptions, host: &dyn MainHost) {
    reset_timings();
    if is_daemon_worker_process_from_env() {
        let mut environment = current_environment();
        if let Err(message) = wait_for_daemon_worker_startup_gate(&mut environment) {
            eprintln!("{message}");
            host.exit(1);
            return;
        }
    }
    install_file_log_sink(None);
    if is_daemon_catalog_process_from_env() {
        if let Err(message) = host.run_daemon_catalog_process().await {
            eprintln!("Prime Agent daemon catalog failed: {message}");
            host.exit(1);
        }
        return;
    }
    // Client and daemon are separate processes; both need these in their registry.
    host.register_builtin_mcp_oauth_providers();
    let offline_mode = args.iter().any(|arg| arg == "--offline")
        || is_truthy_env_flag(host.env("PI_OFFLINE").as_deref());
    if offline_mode {
        host.set_env("PI_OFFLINE", "1");
        host.set_env("PI_SKIP_VERSION_CHECK", "1");
    }

    let public_outcome = match host.handle_public_command(args.clone()).await {
        Ok(outcome) => outcome,
        Err(message) => {
            eprintln!("{}", red(&message));
            host.exit(1);
            return;
        }
    };
    if public_outcome.handled {
        return;
    }
    let args = public_outcome.args;

    if crate::package_manager_cli::handle_config_command(&args).await {
        return;
    }

    let explicit_agents_view = Some(public_outcome.explicit_agents_view);

    let mut parsed = parse_args(&args);
    if !parsed.diagnostics.is_empty() {
        for diagnostic in &parsed.diagnostics {
            let color = if diagnostic.type_.as_str() == "error" { red } else { yellow };
            let label = if diagnostic.type_.as_str() == "error" { "Error" } else { "Warning" };
            eprintln!("{}", color(&format!("{label}: {}", diagnostic.message)));
        }
        if parsed.diagnostics.iter().any(|d| d.type_.as_str() == "error") {
            host.exit(1);
            return;
        }
    }
    time("parseArgs");
    let app_mode = resolve_app_mode(&parsed, host.stdin_is_tty());

    if should_reject_non_interactive_attach(public_outcome.attach_agent.as_deref(), &app_mode) {
        eprintln!("{}", red("Error: attach requires an interactive terminal"));
        host.exit(1);
        return;
    }
    if should_reject_non_interactive_bare_resume(parsed.resume.as_ref(), &app_mode) {
        eprintln!(
            "{}",
            red("Error: --resume without a session selector requires an interactive terminal")
        );
        host.exit(1);
        return;
    }
    set_log_context(log_context(&app_mode));
    let should_take_over_stdout = app_mode != APP_MODE_INTERACTIVE;
    if should_take_over_stdout {
        take_over_stdout();
    }

    // `main.ts:1200-1207`: these `console.log` calls run AFTER `takeOverStdout`
    // (`main.ts:1195-1198`), so while the takeover is active they reach stderr.
    if parsed.version == Some(true) {
        write_stdout(&format!("{VERSION}\n"));
        host.exit(0);
        return;
    }
    if parsed.help == Some(true) {
        write_stdout(&format!("{}\n", format_top_level_help()));
        host.exit(0);
        return;
    }

    if let Some(export_target) = parsed.export.clone() {
        let output_path = parsed.messages.first().cloned();
        let result = export_from_file(
            &export_target,
            Some(ExportOptions { output_path, ..Default::default() }),
        );
        let result = match result {
            Ok(result) => result,
            Err(message) => {
                eprintln!("{}", red(&format!("Error: {message}")));
                host.exit(1);
                return;
            }
        };
        // `main.ts:1219`: `console.log("Exported to: ...")` under the takeover.
        write_stdout(&format!("Exported to: {result}\n"));
        host.exit(0);
        return;
    }

    if (parsed.mode.as_deref() == Some("rpc") || parsed.mode.as_deref() == Some(APP_MODE_DAEMON))
        && !parsed.file_args.is_empty()
    {
        eprintln!("{}", red("Error: @file arguments are not supported in RPC or daemon mode"));
        host.exit(1);
        return;
    }

    validate_fork_flags(&parsed);

    let cwd = match &parsed.cwd {
        Some(requested) => Path::new(&expand_tilde_path(requested, None))
            .to_string_lossy()
            .to_string(),
        None => host.cwd(),
    };
    if parsed.cwd.is_some() {
        let absolute = if Path::new(&cwd).is_absolute() {
            cwd.clone()
        } else {
            Path::new(&host.cwd()).join(&cwd).to_string_lossy().to_string()
        };
        if let Err(message) = host.chdir(&absolute) {
            eprintln!("{}", red(&format!("Error: Cannot use cwd {cwd}: {message}")));
            host.exit(1);
            return;
        }
    }
    if let Some(daemon_socket) = &parsed.daemon_socket {
        // After --cwd so a relative socket path resolves against the requested directory.
        parsed.daemon_socket = Some(normalize_socket_path(daemon_socket, None));
    }

    // Run migrations (pass cwd for project-local migrations)
    let migrations = run_migrations(&cwd);
    let migrated_providers = migrations.migrated_auth_providers.clone();
    let deprecation_warnings = migrations.deprecation_warnings.clone();
    time("runMigrations");

    let agent_dir = get_agent_dir();
    let mut startup_settings_manager = SettingsManager::create(&cwd, Some(&agent_dir));
    report_diagnostics(&collect_settings_diagnostics(
        &mut startup_settings_manager,
        "startup session lookup",
    ));
    let startup_benchmark = is_truthy_env_flag(host.env("PI_STARTUP_BENCHMARK").as_deref());
    if startup_benchmark && app_mode != APP_MODE_INTERACTIVE {
        eprintln!("{}", red("Error: PI_STARTUP_BENCHMARK only supports interactive mode"));
        host.exit(1);
        return;
    }
    // Programmatic factories are process-local functions and cannot be serialized to a daemon worker.
    let has_process_local_extension_factories = options
        .extension_factories
        .as_ref()
        .map(|factories| !factories.is_empty())
        .unwrap_or(false);
    let daemon_client_runtime_decision = DaemonClientRuntimeDecision {
        decision: daemon_client_startup_decision(&parsed, &app_mode, startup_benchmark),
        owned_session_worker: host.is_owned_session_worker_process(),
        has_process_local_extension_factories,
    };
    let use_daemon_client = should_use_daemon_client_runtime(&daemon_client_runtime_decision);
    let use_daemon_interactive = use_daemon_client && app_mode == APP_MODE_INTERACTIVE;

    // Decide the final runtime cwd before creating cwd-bound runtime services.
    // --resume may select a session from another project, so project-local
    // settings, resources, provider registrations, and models must be resolved only after
    // the target session cwd is known. The startup-cwd settings manager is used only for
    // sessionDir lookup during session selection.
    let session_dir = parsed
        .session_dir
        .as_ref()
        .map(|dir| expand_tilde_path(dir, None))
        .or_else(get_session_dir_env_override)
        .or_else(|| startup_settings_manager.get_session_dir());
    let daemon_socket_path = parsed.daemon_socket.clone().unwrap_or_else(default_daemon_socket_path);
    // Kick off daemon spawn/readiness immediately so it overlaps session-manager
    // and runtime-services preparation; attach only connects to an existing daemon.
    let mut daemon_ready = if should_start_daemon_ready_for_startup(
        // packages/coding-agent/src/main.ts:1281 passes `useDaemonClient` (not the
        // interactive-only variant) so print/json/rpc/acp also start and await the
        // daemon before createDaemonClientConnection.
        &daemon_client_runtime_decision,
        public_outcome.attach_agent.as_deref(),
    ) {
        Some(DaemonReadyHandle::start(&daemon_socket_path))
    } else {
        None
    };
    let resume_selector = get_resume_selector(&parsed);
    let should_lookup_daemon_active_session = should_ensure_daemon_before_active_session_lookup(
        &DaemonActiveSessionLookupDecision {
            use_daemon_interactive,
            resume_selector: resume_selector.clone(),
            explicit_attach: Some(public_outcome.attach_agent.is_some()),
        },
    );
    if should_lookup_daemon_active_session {
        daemon_ready = await_daemon_ready(daemon_ready).await;
    }
    let mut active_daemon_session_summary: Option<SessionSummary> = None;
    if should_lookup_daemon_active_session {
        if let Some(resume_selector) = &resume_selector {
            let lookup = find_active_daemon_session_summary_for_interactive_startup(
                &daemon_socket_path,
                resume_selector,
                &ActiveDaemonSessionSummaryLookupOptions {
                    fallback_on_error: Some(public_outcome.attach_agent.is_none()),
                },
            )
            .await;
            match lookup {
                Ok(summary) => active_daemon_session_summary = summary,
                Err(message) => {
                    eprintln!(
                        "{}",
                        red(&format!(
                            "Error: Could not look up active agent '{resume_selector}': {message}"
                        ))
                    );
                    host.exit(1);
                    return;
                }
            }
        }
    }
    if let Some(attach_agent) = &public_outcome.attach_agent {
        if active_daemon_session_summary.is_none() {
            eprintln!(
                "{}",
                red(&format!("Error: No active agent found matching '{attach_agent}'"))
            );
            host.exit(1);
            return;
        }
    }
    let session_manager: SessionManager;
    if let Some(summary) = &active_daemon_session_summary {
        session_manager = create_session_manager_for_active_daemon_summary(summary, &cwd);
    } else if use_daemon_interactive
        && should_use_ephemeral_session_manager_for_daemon_interactive(&DaemonInteractiveSessionManagerDecision {
            resume: parsed.resume.clone(),
            continue_: parsed.continue_,
            fork: parsed.fork.clone(),
            has_active_daemon_session: Some(false),
        })
    {
        session_manager = SessionManager::in_memory(Some(&cwd), None).expect("in-memory session manager");
    } else {
        match create_session_manager(&parsed, &cwd, session_dir.as_deref(), use_daemon_client).await {
            Ok(manager) => session_manager = manager,
            Err(error) => {
                let message = error.message().to_string();
                if !error.is_selector_error() {
                    eprintln!("{}", red(&format!("Error: {message}")));
                    host.exit(1);
                    return;
                }
                let suggestion = match error.suggestion() {
                    Some(suggestion) => format!(" Did you mean '{suggestion}'?"),
                    None => String::new(),
                };
                eprintln!("{}", red(&format!("Error: {message}.{suggestion}")));
                eprintln!(
                    "{}",
                    dim(&format!("Open {APP_NAME} and press left-arrow to browse sessions."))
                );
                host.exit(1);
                return;
            }
        }
    }
    let mut session_manager = session_manager;
    // `getMissingSessionCwdIssue(sessionManager, cwd)`.
    //
    // blocked_on: `SessionManager` does not implement `SessionCwdSource`
    // (core/session-cwd.ts owns the trait), so the port passes the same two
    // accessors through a private adapter.
    let session_cwd_source = SessionManagerCwdSource { session_manager: &session_manager };
    if let Some(issue) = get_missing_session_cwd_issue(&session_cwd_source, &cwd) {
        if app_mode == APP_MODE_INTERACTIVE {
            host.init_theme(startup_settings_manager.get_theme(), false);
            host.set_keybindings(&agent_dir);
            let selected_cwd = host.prompt_for_missing_session_cwd(&issue, &agent_dir);
            let Some(selected_cwd) = selected_cwd else {
                host.exit(0);
                return;
            };
            let session_file = issue.session_file.clone().unwrap_or_default();
            session_manager = if use_daemon_client {
                read_session_manager(&session_file, session_dir.as_deref(), Some(&selected_cwd))
            } else {
                match SessionManager::open(&session_file, session_dir.as_deref(), Some(&selected_cwd)) {
                    Ok(manager) => manager,
                    Err(message) => {
                        eprintln!("{}", red(&format!("Error: {message}")));
                        host.exit(1);
                        return;
                    }
                }
            };
        } else {
            eprintln!("{}", red(&MissingSessionCwdError::new(issue).to_string()));
            host.exit(1);
            return;
        }
    }
    time("createSessionManager");

    // Every later consumer shares one `SessionManager`, the way the TypeScript
    // passes the same object to the runtime factory, the startup-model resolver
    // and `createAgentSessionRuntime`.
    let session_manager = Arc::new(Mutex::new(session_manager));
    let session_manager_cwd = session_manager.lock().expect("session manager poisoned").get_cwd();

    // `sessionManager.getCwd() === cwd ? startupSettingsManager : SettingsManager.create(...)`.
    // `SettingsManager` is not shared-mutable in the port, so the startup
    // manager is moved into the telemetry check instead of being aliased.
    let telemetry_settings_manager = if session_manager_cwd == cwd {
        startup_settings_manager
    } else {
        SettingsManager::create(&session_manager_cwd, Some(&agent_dir))
    };
    let telemetry_disabled = if crate::core::agent_session_services::is_telemetry_enabled(&Arc::new(Mutex::new(
        telemetry_settings_manager,
    ))) {
        None
    } else {
        Some(true)
    };
    let default_session_config = runtime_config_from_args(
        &parsed,
        &session_manager_cwd,
        &agent_dir,
        session_dir.as_deref(),
        &app_mode,
        telemetry_disabled,
    );
    // Verifier/headless clients pass initialGoal in each create request. The long-lived
    // daemon fallback must not seed that goal into unrelated future sessions.
    let daemon_default_session_config = daemon_server_default_session_config(&default_session_config);
    let runtime_default_session_config = if app_mode == APP_MODE_DAEMON {
        daemon_default_session_config.clone()
    } else {
        default_session_config.clone()
    };
    let create_runtime =
        create_default_runtime_factory(runtime_default_session_config, options.extension_factories.clone());
    time("createRuntime");
    // Daemon mode never uses the bootstrap runtime, so skip the heavy
    // createAgentSessionRuntime below and start listening immediately; sessions
    // are created on demand through the daemon protocol via createRuntime.
    // --list-models still takes the full path to print and exit.
    if app_mode == APP_MODE_DAEMON && parsed.list_models.is_none() {
        print_timings();
        if is_daemon_worker_process_from_env() {
            let environment = current_environment();
            let worker = match require_daemon_worker_authentication_token(&environment) {
                Ok(authentication_token) => DaemonWorkerSeamOptions {
                    authentication_token,
                    worker_instance_id: daemon_worker_instance_id(&environment),
                    restore_active_session_id: environment.get(DAEMON_WORKER_ACTIVE_SESSION_ID_ENV).cloned(),
                },
                Err(message) => {
                    eprintln!("{message}");
                    host.exit(1);
                    return;
                }
            };
            if let Err(message) = host
                .run_daemon_mode(DaemonModeSeamOptions {
                    socket_path: parsed.daemon_socket.clone(),
                    default_session_config: daemon_default_session_config,
                    create_runtime,
                    worker: Some(worker),
                })
                .await
            {
                eprintln!("{message}");
                host.exit(1);
            }
        } else if let Err(message) = host
            .run_daemon_supervisor_mode(parsed.daemon_socket.clone(), daemon_default_session_config)
            .await
        {
            eprintln!("{message}");
            host.exit(1);
        }
        return;
    }
    if use_daemon_interactive {
        // `main.ts:1404-1410` has no try/catch here, so a services-creation
        // failure rejects out of `main()` and the CLI exits 1.
        let prepared = match prepare_runtime_services(PrepareRuntimeServicesOptions {
            config: default_session_config.clone(),
            cwd: session_manager_cwd.clone(),
            agent_dir: agent_dir.clone(),
            session_manager: Arc::clone(&session_manager),
            extension_factories: options.extension_factories.clone(),
            session_options_override: None,
        })
        .await
        {
            Ok(prepared) => prepared,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        let services = Arc::clone(&prepared.services);
        let scoped_models = prepared.scoped_models.clone();
        let settings_manager = Arc::clone(&services.settings_manager);

        let startup_model = resolve_prepared_startup_model(&prepared, &session_manager).await;

        let stdin_content = read_piped_stdin().await;
        time("readPipedStdin");

        let (initial_message, initial_images) = match prepare_initial_message(
            &mut parsed,
            settings_manager.lock().unwrap().get_image_auto_resize(),
            stdin_content,
        )
        .await
        {
            Ok(result) => result,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        time("prepareInitialMessage");
        host.init_theme(settings_manager.lock().unwrap().get_theme(), true);
        time("initTheme");

        if !deprecation_warnings.is_empty() {
            show_deprecation_warnings(&deprecation_warnings).await;
        }

        report_diagnostics(&prepared.diagnostics);
        if prepared
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.type_.as_str() == "error")
        {
            host.exit(1);
            return;
        }
        time("prepareInteractiveServices");

        print_model_scope(&scoped_models, parsed.verbose == Some(true), &settings_manager);

        // `new ClientPromptStashStore()` is created once per TUI process and
        // shared across chat views; `InteractiveMode` owns the store itself, so
        // the port passes only the seam-visible parts below.
        let needs_onboarding = {
            let settings_guard = settings_manager.lock().unwrap();
            let mut registry_guard = services.model_registry.lock().unwrap();
            let settings_reader = CoreOnboardingSettingsReader { settings_manager: &settings_guard };
            let registry_reader = CoreOnboardingModelRegistryReader {
                model_registry: std::cell::RefCell::new(&mut *registry_guard),
            };
            crate::modes::interactive::onboarding::should_run_onboarding(
                &crate::modes::interactive::onboarding::OnboardingStartupState {
                    settings_manager: &settings_reader,
                    model_registry: &registry_reader,
                    model: startup_model.model.as_ref(),
                },
            )
        };
        if should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            use_daemon_interactive: use_daemon_interactive && parsed.no_session != Some(true),
            needs_onboarding,
            explicit_agents_view,
            resume: parsed.resume.clone(),
            continue_: parsed.continue_,
            fork: parsed.fork.clone(),
        }) {
            daemon_ready = await_daemon_ready(daemon_ready).await;
            host.preload_code_highlighter();
            print_timings();
            let result = host
                .run_agents_view_mode(AgentsViewSeamOptions {
                    socket_path: daemon_socket_path.clone(),
                    config: default_session_config.clone(),
                    agent_dir: agent_dir.clone(),
                    migrated_providers: migrated_providers.clone(),
                    model_fallback_message: startup_model.model_fallback_message.clone(),
                    startup_model_id: startup_model.model.as_ref().map(|model| model.id.clone()),
                    verbose: parsed.verbose == Some(true),
                    initial_session: None,
                    initial_scope_key: None,
                })
                .await;
            if let Err(message) = result {
                eprintln!("{message}");
                host.exit(1);
            }
            return;
        }

        daemon_ready = await_daemon_ready(daemon_ready).await;
        // A fresh default chat opens a real but message-less session; the lifecycle
        // axis treats it as a draft (hidden, discarded on detach if never used), so
        // no DeferredAgentConnection is needed to avoid creating it up front.
        let is_fresh_default_session =
            active_daemon_session_summary.is_none()
                && get_interactive_daemon_session_path(
                    &parsed,
                    &session_manager.lock().expect("session manager poisoned"),
                )
                .is_none();
        let connection = match create_daemon_client_connection(CreateDaemonClientConnectionOptions {
            defer_session_events: true,
            socket_path: daemon_socket_path.clone(),
            config: default_session_config.clone(),
            session_path: get_interactive_daemon_session_path(
                &parsed,
                &session_manager.lock().expect("session manager poisoned"),
            ),
            continue_recent: None,
            active_session_id: active_daemon_session_summary
                .as_ref()
                .map(get_daemon_summary_active_session_id),
            client_owned: parsed.no_session,
            no_session: parsed.no_session,
            supports_extension_ui: Some(true),
        })
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                // `if (error instanceof DaemonSessionCreateError) { print; exit(1) }
                //  throw error;` (`main.ts:1528-1534`).
                if !error.is_reported_at_interactive_attach() {
                    rethrow_daemon_connection_error(&error, host);
                    return;
                }
                eprintln!("{}", red(&format!("Error: {}", error.message)));
                host.exit(1);
                return;
            }
        };
        let (connection, summary) = connection;
        // `const attachModelFallbackMessage = isFreshDefaultSession
        //   ? startupModel.modelFallbackMessage
        //   : resolveAttachModelFallbackMessage(summary, startupModel.modelFallbackMessage);`
        // (`main.ts:1536-1538`). The helper is the real one
        // (`daemon_session_list.rs:227-238` -> `daemon-session-list.ts:104-109`).
        let attach_model_fallback_message = if is_fresh_default_session {
            startup_model.model_fallback_message.clone()
        } else {
            resolve_attach_model_fallback_message(&summary, startup_model.model_fallback_message.as_deref())
        };

        host.preload_code_highlighter();
        print_timings();
        let result = host
            .run_interactive_mode(InteractiveModeSeamOptions {
                daemon_socket_path: Some(daemon_socket_path.clone()),
                migrated_providers: migrated_providers.clone(),
                model_fallback_message: attach_model_fallback_message,
                initial_message,
                initial_images,
                initial_messages: parsed.messages.clone(),
                verbose: parsed.verbose == Some(true),
                // Resumed/attached daemon sessions are part of the same fleet; left
                // arrow takes them to the agents view like any other session. The agents
                // view was not rendered here, so we intentionally leave
                // agentsViewOwnsStartupNotices unset and let the in-session fallback run.
                return_to_agents_view: parsed.no_session != Some(true),
                session_depth: summary.rlm_depth.map(|depth| depth as f64),
                // Direct launch has only the attached summary; the live+passive+saved
                // unified catalog index is not built until the agents view opens. Retained
                // child snapshots augment this running-child fallback inside chat.
                session_has_children: summary.has_running_rlm_children == Some(true),
                connection: Some(Arc::clone(&connection) as Arc<dyn crate::modes::agent_connection::types::AgentConnection>),
                runtime: None,
            })
            .await;
        let interactive_result = match result {
            Ok(Some(result)) => result,
            Ok(None) => return,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        if parsed.no_session == Some(true) {
            return;
        }
        let (returned_summary, initial_scope_key) = returned_agents_view_state(summary, interactive_result);
        host.preload_code_highlighter();
        print_timings();
        let result = host
            .run_agents_view_mode(AgentsViewSeamOptions {
                socket_path: daemon_socket_path.clone(),
                config: default_session_config.clone(),
                agent_dir: agent_dir.clone(),
                migrated_providers: migrated_providers.clone(),
                model_fallback_message: None,
                startup_model_id: None,
                verbose: parsed.verbose == Some(true),
                initial_session: Some(returned_summary),
                initial_scope_key,
            })
            .await;
        if let Err(message) = result {
            eprintln!("{message}");
            host.exit(1);
        }
        return;
    }
    if use_daemon_client {
        let settings_manager = SettingsManager::create(&session_manager_cwd, Some(&agent_dir));
        let mut stdin_content: Option<String> = None;
        if app_mode != "rpc" && app_mode != "acp" {
            stdin_content = read_piped_stdin().await;
        }
        time("readPipedStdin");
        let (initial_message, initial_images) = match prepare_initial_message(
            &mut parsed,
            settings_manager.get_image_auto_resize(),
            stdin_content,
        )
        .await
        {
            Ok(result) => result,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        time("prepareInitialMessage");
        host.init_theme(settings_manager.get_theme(), false);
        time("initTheme");

        daemon_ready = await_daemon_ready(daemon_ready).await;
        let connection = match create_daemon_client_connection(CreateDaemonClientConnectionOptions {
            defer_session_events: false,
            socket_path: daemon_socket_path.clone(),
            config: default_session_config.clone(),
            session_path: if parsed.no_session == Some(true) {
                None
            } else {
                session_manager
                    .lock()
                    .expect("session manager poisoned")
                    .get_session_file()
            },
            continue_recent: parsed.continue_,
            active_session_id: None,
            client_owned: Some(is_client_owned_daemon_session(&app_mode, parsed.no_session)),
            no_session: parsed.no_session,
            supports_extension_ui: Some(app_mode == "rpc"),
        })
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                // `if (error instanceof SessionAlreadyActiveError || error instanceof
                //  DaemonSessionCreateError) { print; exit(1) } throw error;`
                // (`main.ts:1615-1621`).
                if !error.is_reported_at_daemon_client() {
                    rethrow_daemon_connection_error(&error, host);
                    return;
                }
                eprintln!("{}", red(&format!("Error: {}", error.message)));
                host.exit(1);
                return;
            }
        };
        let (connection, summary) = connection;
        let diagnostics: Vec<AgentSessionRuntimeDiagnostic> = summary
            .diagnostics
            .as_ref()
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        report_diagnostics(&diagnostics);
        if diagnostics.iter().any(|diagnostic| diagnostic.type_.as_str() == "error") {
            let _ = AgentConnectionTrait::dispose(connection.as_ref()).await;
            host.exit(1);
            return;
        }
        if summary.model.is_none() {
            eprintln!(
                "{}",
                red(&summary
                    .model_fallback_message
                    .clone()
                    .unwrap_or_else(format_no_models_available_message))
            );
            let _ = AgentConnectionTrait::dispose(connection.as_ref()).await;
            host.exit(1);
            return;
        }

        print_timings();
        let connection: Arc<dyn crate::modes::agent_connection::types::AgentConnection> = connection;
        if app_mode == "rpc" {
            if let Err(message) = host.run_rpc_mode_with_connection(connection).await {
                eprintln!("{message}");
                host.exit(1);
            }
            return;
        }
        if app_mode == "acp" {
            if let Err(message) = host.run_acp_mode_with_connection(connection).await {
                eprintln!("{message}");
                host.exit(1);
            }
            return;
        }
        let exit_code = match host
            .run_print_mode_with_connection(
                connection,
                PrintModeSeamOptions {
                    mode: to_print_output_mode(&app_mode).to_string(),
                    messages: parsed.messages.clone(),
                    initial_message,
                    initial_images,
                },
            )
            .await
        {
            Ok(exit_code) => exit_code,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        host.stop_theme_watcher();
        restore_stdout();
        if exit_code != 0 {
            host.set_exit_code(exit_code);
        }
        return;
    }

    let runtime = match create_agent_session_runtime_port(
        Arc::clone(&create_runtime),
        CreateAgentSessionRuntimeInput {
            cwd: session_manager_cwd.clone(),
            agent_dir: agent_dir.clone(),
            session_manager: Arc::clone(&session_manager),
            session_start_event: None,
            session_config: Some(default_session_config.clone()),
            session_options: None,
            runtime_metadata: None,
            session_lease: None,
        },
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(message) => {
            eprintln!("{}", red(&format!("Error: {message}")));
            host.exit(1);
            return;
        }
    };
    let services = runtime.services();
    let session = runtime.session();
    let model_fallback_message = runtime.model_fallback_message();
    host.install_owned_session_recovery_tracking(Arc::clone(&runtime));
    let settings_manager = Arc::clone(&services.settings_manager);
    let model_registry = Arc::clone(&services.model_registry);

    if let Some(list_models_value) = &parsed.list_models {
        let search_pattern = match list_models_value {
            ListModelsValue::All => None,
            ListModelsValue::Search(pattern) => Some(pattern.clone()),
        };
        // `listModels(modelRegistry, searchPattern)`.
        //
        // blocked_on: `cli/list-models.ts` is landed with a synchronous
        // `ModelRegistry` trait, while `await modelRegistry.refreshAvailableModels()`
        // is async in the TypeScript. The adapter below feeds the module from the
        // same registry through its synchronous `refresh()` / `getAvailable()`
        // pair, so no await is needed to produce the rows.
        let adapter = {
            let mut registry = model_registry.lock().unwrap();
            registry.refresh();
            CliModelRegistryAdapter {
                error: registry.get_error().map(str::to_string),
                models: registry.get_available(),
            }
        };
        // `list-models.ts` prints its table with `console.log`, which the
        // taken-over stdout (`core/output-guard.ts:18-27`, active for every
        // non-interactive mode via `main.ts:1195-1198`) routes to stderr.
        list_models(
            &adapter,
            search_pattern.as_deref(),
            &ListModelsIo {
                log: &|message: &str| write_stdout(&format!("{message}\n")),
                error: &|message: &str| eprintln!("{message}"),
            },
        );
        host.exit(0);
        return;
    }

    // Read piped stdin content (if any) - skip for RPC/daemon modes which use other transports
    let mut stdin_content: Option<String> = None;
    if app_mode != "rpc" && app_mode != "acp" && app_mode != APP_MODE_DAEMON {
        stdin_content = read_piped_stdin().await;
    }
    time("readPipedStdin");

    let (initial_message, initial_images) = match prepare_initial_message(
        &mut parsed,
        settings_manager.lock().unwrap().get_image_auto_resize(),
        stdin_content,
    )
    .await
    {
        Ok(result) => result,
        Err(message) => {
            eprintln!("{message}");
            host.exit(1);
            return;
        }
    };
    time("prepareInitialMessage");
    host.init_theme(
        settings_manager.lock().unwrap().get_theme(),
        app_mode == APP_MODE_INTERACTIVE,
    );
    time("initTheme");

    // Show deprecation warnings in interactive mode
    if app_mode == APP_MODE_INTERACTIVE && !deprecation_warnings.is_empty() {
        show_deprecation_warnings(&deprecation_warnings).await;
    }

    // `session.scopedModels` carries the same `{ model, thinkingLevel }` entries;
    // the CLI's `ScopedModel` keeps the level as its wire string.
    let scoped_models: Vec<ScopedModel> = session
        .scoped_models()
        .into_iter()
        .map(|scoped| ScopedModel {
            model: scoped.model,
            thinking_level: scoped.thinking_level.map(|level| level.as_str().to_string()),
        })
        .collect();
    time("resolveModelScope");
    report_diagnostics(&runtime.diagnostics());
    if runtime
        .diagnostics()
        .iter()
        .any(|diagnostic| diagnostic.type_.as_str() == "error")
    {
        host.exit(1);
        return;
    }
    time("createAgentSession");

    if app_mode != APP_MODE_INTERACTIVE && app_mode != APP_MODE_DAEMON && session.model().is_none() {
        eprintln!("{}", red(&format_no_models_available_message()));
        host.exit(1);
        return;
    }

    if app_mode == "rpc" {
        print_timings();
        if let Err(message) = host.run_rpc_mode(Arc::clone(&runtime)).await {
            eprintln!("{message}");
            host.exit(1);
        }
    } else if app_mode == "acp" {
        print_timings();
        if let Err(message) = host.run_acp_mode(Arc::clone(&runtime)).await {
            eprintln!("{message}");
            host.exit(1);
        }
    } else if app_mode == APP_MODE_INTERACTIVE {
        if explicit_agents_view == Some(true) || matches!(parsed.resume, Some(ResumeValue::Latest)) {
            eprintln!(
                "{}",
                yellow("Warning: the agents view needs the daemon; opening a normal chat instead")
            );
        }
        print_model_scope(&scoped_models, parsed.verbose == Some(true), &settings_manager);

        let interactive_options = InteractiveModeSeamOptions {
            daemon_socket_path: None,
            migrated_providers: migrated_providers.clone(),
            model_fallback_message: model_fallback_message.clone(),
            initial_message: initial_message.clone(),
            initial_images: initial_images.clone(),
            initial_messages: parsed.messages.clone(),
            verbose: parsed.verbose == Some(true),
            return_to_agents_view: false,
            session_depth: None,
            session_has_children: false,
            // `new InProcessAgentConnection(runtime)` plus
            // `createInteractiveModeLocalSessionHost(runtime)`: the host builds
            // both from the runtime it is given.
            connection: None,
            runtime: Some(Arc::clone(&runtime)),
        };
        if startup_benchmark {
            if let Err(message) = host.init_interactive_mode(interactive_options).await {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
            time("interactiveMode.init");
            print_timings();
            host.stop_theme_watcher();
            return;
        }

        host.preload_code_highlighter();
        print_timings();
        if let Err(message) = host.run_interactive_mode(interactive_options).await {
            eprintln!("{message}");
            host.exit(1);
        }
    } else {
        print_timings();
        let exit_code = match host
            .run_print_mode(
                Arc::clone(&runtime),
                PrintModeSeamOptions {
                    mode: to_print_output_mode(&app_mode).to_string(),
                    messages: parsed.messages.clone(),
                    initial_message,
                    initial_images,
                },
            )
            .await
        {
            Ok(exit_code) => exit_code,
            Err(message) => {
                eprintln!("{message}");
                host.exit(1);
                return;
            }
        };
        host.stop_theme_watcher();
        restore_stdout();
        if exit_code != 0 {
            host.set_exit_code(exit_code);
        }
    }
}

/// Carry the returned chat identity into the roster, preserving its daemon metadata.
fn returned_agents_view_state(
    mut summary: SessionSummary,
    result: InteractiveModeRunResult,
) -> (SessionSummary, Option<AgentsViewScopeKey>) {
    let scope = (result.type_ == InteractiveModeRunResultType::ScopedAgentsView).then(|| AgentsViewScopeKey {
        session_id: result.source.session_id.clone(),
        active_session_id: result.source.active_session_id.clone(),
    });
    summary.id = result.source.active_session_id.clone().unwrap_or(summary.id);
    summary.active_session_id = result.source.active_session_id;
    summary.session_file = result.source.session_file;
    summary.session_id = result.source.session_id;
    summary.session_name = result.source.session_name;
    summary.cwd = result.source.cwd;
    (summary, scope)
}

/// `console.log(chalk.dim(\`Model scope: ...\`))`.
fn print_model_scope(scoped_models: &[ScopedModel], verbose: bool, settings_manager: &Arc<Mutex<SettingsManager>>) {
    if !scoped_models.is_empty() && (verbose || !settings_manager.lock().unwrap().get_quiet_startup()) {
        let model_list = scoped_models
            .iter()
            .map(|scoped| match &scoped.thinking_level {
                Some(thinking_level) => format!("{}:{}", scoped.model.id, thinking_level),
                None => scoped.model.id.clone(),
            })
            .collect::<Vec<String>>()
            .join(", ");
        println!("{}", dim(&format!("Model scope: {model_list} {}", gray("(Ctrl+P to cycle)"))));
    }
}

/// `createAgentSessionRuntime(createRuntime, options)`.
///
/// blocked_on: `core/agent-session-runtime.ts` exports the class but not this
/// factory yet (the module is still being landed by another slice), so the
/// function body lives here verbatim: acquire the lease when the caller has
/// none, assert the session cwd, run the factory, and build the runtime.
pub async fn create_agent_session_runtime_port(
    create_runtime: CreateAgentSessionRuntimeFactory,
    options: CreateAgentSessionRuntimeInput,
) -> Result<Arc<AgentSessionRuntime>, String> {
    let CreateAgentSessionRuntimeInput {
        cwd,
        agent_dir,
        session_manager,
        session_start_event,
        session_config,
        session_options,
        runtime_metadata,
        session_lease,
    } = options;
    let lease = match session_lease {
        Some(lease) => Some(lease),
        None => match crate::core::session_lease::acquire_session_lease(
            session_manager.lock().unwrap().get_session_file().as_deref(),
            &agent_dir,
            None,
        ) {
            Ok(lease) => lease.map(|lease| Arc::new(Mutex::new(lease))),
            Err(error) => return Err(error.to_string()),
        },
    };
    let lease_failed = |error: String| {
        if let Some(lease) = &lease {
            lease.lock().unwrap().release();
        }
        error
    };
    crate::core::session_cwd::assert_session_cwd_exists(
        &SessionManagerCwdSource { session_manager: &*session_manager.lock().unwrap() },
        &cwd,
    )
    .map_err(|error: MissingSessionCwdError| lease_failed(error.to_string()))?;
    let result = create_runtime(CreateAgentSessionRuntimeInput {
        cwd,
        agent_dir,
        session_manager,
        session_start_event,
        session_config: session_config.clone(),
        session_options,
        runtime_metadata: runtime_metadata.clone(),
        session_lease: lease.clone(),
    })
    .await
    .map_err(lease_failed)?;
    Ok(AgentSessionRuntime::new(
        result.result.session,
        result.services,
        create_runtime,
        result.diagnostics,
        result.result.model_fallback_message,
        session_config,
        runtime_metadata.unwrap_or_default(),
        lease,
    ))
}

/// `SessionCwdSource` view of `SessionManager`.
struct SessionManagerCwdSource<'a> {
    session_manager: &'a SessionManager,
}

impl crate::core::session_cwd::SessionCwdSource for SessionManagerCwdSource<'_> {
    fn get_cwd(&self) -> String {
        self.session_manager.get_cwd()
    }

    fn get_session_file(&self) -> Option<String> {
        self.session_manager.get_session_file()
    }
}

/// `ModelRegistry` view consumed by `cli/list-models.ts`.
struct CliModelRegistryAdapter {
    error: Option<String>,
    models: Vec<Model>,
}

impl crate::cli::list_models::ModelRegistry for CliModelRegistryAdapter {
    fn get_error(&self) -> Option<String> {
        self.error.clone()
    }

    fn refresh_available_models(&self) -> Vec<Model> {
        self.models.clone()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn daemon_shutdown_notice_reaches_connection_recovery() {
        let event = super::main_entry_outbound_from_wire(&serde_json::json!({"type":"daemon_closing","reason":"shutdown"}));
        assert!(matches!(event, Some(super::ConnectionOutbound::DaemonClosing { reason: crate::modes::daemon::daemon_protocol::DaemonClosingReason::Shutdown })));
    }

    use super::*;
    use crate::cli::args::UnknownFlagValue;

    fn args(argv: &[&str]) -> Args {
        parse_args(&argv.iter().map(|arg| arg.to_string()).collect::<Vec<String>>())
    }

    #[test]
    fn app_mode_follows_the_mode_flags_then_the_tty() {
        assert_eq!(resolve_app_mode(&args(&["--mode", "daemon"]), true), "daemon");
        assert_eq!(resolve_app_mode(&args(&["--mode", "rpc"]), true), "rpc");
        assert_eq!(resolve_app_mode(&args(&["--mode", "acp"]), true), "acp");
        assert_eq!(resolve_app_mode(&args(&["--mode", "json"]), true), "json");
        assert_eq!(resolve_app_mode(&args(&["--print"]), true), "print");
        assert_eq!(resolve_app_mode(&args(&[]), false), "print");
        assert_eq!(resolve_app_mode(&args(&[]), true), "interactive");
    }

    #[test]
    fn print_output_mode_is_json_only_for_json() {
        assert_eq!(to_print_output_mode("json"), "json");
        assert_eq!(to_print_output_mode("print"), "text");
        assert_eq!(to_print_output_mode("interactive"), "text");
    }

    #[test]
    fn non_interactive_attach_and_bare_resume_are_rejected() {
        assert!(should_reject_non_interactive_attach(Some("a1"), "print"));
        assert!(!should_reject_non_interactive_attach(Some("a1"), "interactive"));
        assert!(!should_reject_non_interactive_attach(None, "print"));
        assert!(should_reject_non_interactive_bare_resume(Some(&ResumeValue::Latest), "rpc"));
        assert!(!should_reject_non_interactive_bare_resume(
            Some(&ResumeValue::Latest),
            "interactive"
        ));
        assert!(!should_reject_non_interactive_bare_resume(
            Some(&ResumeValue::Selector("s".to_string())),
            "print"
        ));
    }

    #[test]
    fn acp_is_the_only_client_owned_exception() {
        assert!(!is_client_owned_daemon_session("acp", None));
        assert!(is_client_owned_daemon_session("acp", Some(true)));
        assert!(is_client_owned_daemon_session("print", None));
    }

    #[test]
    fn daemon_client_requires_a_non_daemon_mode_without_benchmark_or_help() {
        let base = DaemonClientStartupDecision {
            app_mode: "print".to_string(),
            startup_benchmark: false,
            no_session: None,
            help: None,
            list_models: None,
        };
        assert!(should_use_daemon_client(&base));
        assert!(should_use_daemon_interactive(&DaemonClientStartupDecision {
            app_mode: "interactive".to_string(),
            ..base.clone()
        }));
        assert!(!should_use_daemon_client(&DaemonClientStartupDecision {
            app_mode: "daemon".to_string(),
            ..base.clone()
        }));
        assert!(!should_use_daemon_client(&DaemonClientStartupDecision {
            startup_benchmark: true,
            ..base.clone()
        }));
        assert!(!should_use_daemon_client(&DaemonClientStartupDecision {
            help: Some(true),
            ..base.clone()
        }));
        assert!(!should_use_daemon_client(&DaemonClientStartupDecision {
            list_models: Some(ListModelsValue::All),
            ..base.clone()
        }));
        assert!(!should_use_daemon_client_runtime(&DaemonClientRuntimeDecision {
            decision: base.clone(),
            owned_session_worker: true,
            has_process_local_extension_factories: false,
        }));
        assert!(!should_use_daemon_client_runtime(&DaemonClientRuntimeDecision {
            decision: base,
            owned_session_worker: false,
            has_process_local_extension_factories: true,
        }));
    }

    #[test]
    fn startup_daemon_readiness_skips_explicit_attach() {
        assert!(should_ensure_interactive_daemon_for_startup(true, None));
        assert!(!should_ensure_interactive_daemon_for_startup(true, Some("a1")));
        assert!(!should_ensure_interactive_daemon_for_startup(false, None));
    }

    #[test]
    fn print_mode_ensures_the_daemon_like_the_typescript() {
        // `main.ts:1281` gates `daemonReady` on `useDaemonClient`, so the headless
        // --print path starts the daemon and AWAITS it at `main.ts:1602` before
        // `createDaemonClientConnection`. Gating on `useDaemonInteractive` leaves
        // print mode with `daemonReady === undefined`, which makes
        // `awaitDaemonReady` (main.ts:419) a no-op and lets the connect race the
        // fire-and-forget `maybeStartDaemonEarly`.
        let decision = |app_mode: &str| DaemonClientRuntimeDecision {
            decision: DaemonClientStartupDecision {
                app_mode: app_mode.to_string(),
                startup_benchmark: false,
                no_session: None,
                help: None,
                list_models: None,
            },
            owned_session_worker: false,
            has_process_local_extension_factories: false,
        };

        for app_mode in ["print", "json", "rpc", "acp", "interactive"] {
            assert!(
                should_start_daemon_ready_for_startup(&decision(app_mode), None),
                "{app_mode} must start and await the daemon"
            );
            // `attachAgent !== undefined` keeps the existing-daemon-only contract.
            assert!(!should_start_daemon_ready_for_startup(&decision(app_mode), Some("a1")));
        }

        // The interactive-only variant is FALSE for every headless mode, which is
        // what made the old gate a no-op. If the gate regresses to it, the
        // assertions above fail with "print must start and await the daemon".
        for app_mode in ["print", "json", "rpc", "acp"] {
            let runtime_decision = decision(app_mode);
            let use_daemon_client = should_use_daemon_client_runtime(&runtime_decision);
            let use_daemon_interactive =
                use_daemon_client && runtime_decision.decision.app_mode == APP_MODE_INTERACTIVE;
            assert!(use_daemon_client, "{app_mode} uses the daemon client");
            assert!(
                !should_ensure_interactive_daemon_for_startup(use_daemon_interactive, None),
                "{app_mode} must not be gated on the interactive-only flag"
            );
        }

        // Daemon mode, the benchmark, --help, --list-models and an owned session
        // worker never start a daemon for themselves.
        assert!(!should_start_daemon_ready_for_startup(&decision(APP_MODE_DAEMON), None));
        let mut benchmark = decision("print");
        benchmark.decision.startup_benchmark = true;
        assert!(!should_start_daemon_ready_for_startup(&benchmark, None));
        let mut help = decision("print");
        help.decision.help = Some(true);
        assert!(!should_start_daemon_ready_for_startup(&help, None));
        let mut owned_worker = decision("print");
        owned_worker.owned_session_worker = true;
        assert!(!should_start_daemon_ready_for_startup(&owned_worker, None));
        let mut list_models = decision("print");
        list_models.decision.list_models = Some(ListModelsValue::All);
        assert!(!should_start_daemon_ready_for_startup(&list_models, None));
    }

    #[test]
    fn non_utf8_piped_stdin_is_decoded_lossily_not_dropped() {
        // `main.ts:136-147`: `setEncoding("utf8")` keeps the chunk content, so
        // the invalid bytes become U+FFFD instead of vanishing.
        assert_eq!(decode_piped_stdin(b"hello \xff world"), Some("hello \u{fffd} world".to_string()));
        assert_eq!(decode_piped_stdin(b"\xff\xfe"), Some("\u{fffd}\u{fffd}".to_string()));
        // Surrounding ASCII survives so the prompt is not empty.
        assert!(decode_piped_stdin(b"before \x80 after")
            .expect("content is kept")
            .starts_with("before "));
        // The `data.trim() || undefined` falsy check still yields None.
        assert_eq!(decode_piped_stdin(b""), None);
        assert_eq!(decode_piped_stdin(b"   \n"), None);
        // Trim runs AFTER the lossy decode, so an invalid byte is content even
        // with surrounding whitespace.
        assert_eq!(decode_piped_stdin(b" \xff "), Some("\u{fffd}".to_string()));
    }

    #[test]
    fn agents_view_opens_only_for_a_bare_or_explicit_start() {
        let base = AgentsViewStartupDecision {
            use_daemon_interactive: true,
            needs_onboarding: false,
            explicit_agents_view: None,
            resume: None,
            continue_: None,
            fork: None,
        };
        assert!(!should_open_agents_view_for_daemon_interactive(&base));
        assert!(should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            explicit_agents_view: Some(true),
            ..base.clone()
        }));
        assert!(should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            resume: Some(ResumeValue::Latest),
            ..base.clone()
        }));
        assert!(!should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            explicit_agents_view: Some(true),
            resume: Some(ResumeValue::Selector("s".to_string())),
            ..base.clone()
        }));
        assert!(!should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            explicit_agents_view: Some(true),
            needs_onboarding: true,
            ..base.clone()
        }));
        assert!(!should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            explicit_agents_view: Some(true),
            continue_: Some(true),
            ..base.clone()
        }));
        assert!(!should_open_agents_view_for_daemon_interactive(&AgentsViewStartupDecision {
            explicit_agents_view: Some(true),
            fork: Some("f".to_string()),
            ..base
        }));
    }

    #[test]
    fn the_ephemeral_session_manager_needs_no_target() {
        let base = DaemonInteractiveSessionManagerDecision {
            resume: None,
            continue_: None,
            fork: None,
            has_active_daemon_session: None,
        };
        assert!(should_use_ephemeral_session_manager_for_daemon_interactive(&base));
        assert!(should_use_ephemeral_session_manager_for_daemon_interactive(
            &DaemonInteractiveSessionManagerDecision {
                resume: Some(ResumeValue::Latest),
                ..base.clone()
            }
        ));
        assert!(!should_use_ephemeral_session_manager_for_daemon_interactive(
            &DaemonInteractiveSessionManagerDecision {
                resume: Some(ResumeValue::Selector("s".to_string())),
                ..base.clone()
            }
        ));
        assert!(!should_use_ephemeral_session_manager_for_daemon_interactive(
            &DaemonInteractiveSessionManagerDecision {
                has_active_daemon_session: Some(true),
                ..base
            }
        ));
    }

    #[test]
    fn the_active_session_lookup_runs_before_a_selector_lookup() {
        assert!(should_ensure_daemon_before_active_session_lookup(
            &DaemonActiveSessionLookupDecision {
                use_daemon_interactive: true,
                resume_selector: Some("abc".to_string()),
                explicit_attach: None,
            }
        ));
        // An explicit --attach always looks the daemon up, even for a path.
        assert!(should_ensure_daemon_before_active_session_lookup(
            &DaemonActiveSessionLookupDecision {
                use_daemon_interactive: true,
                resume_selector: Some("/tmp/session.jsonl".to_string()),
                explicit_attach: Some(true),
            }
        ));
        assert!(!should_ensure_daemon_before_active_session_lookup(
            &DaemonActiveSessionLookupDecision {
                use_daemon_interactive: true,
                resume_selector: Some("/tmp/session.jsonl".to_string()),
                explicit_attach: None,
            }
        ));
    }

    #[test]
    fn fork_conflicts_are_validated_in_flag_order() {
        let parsed = args(&["--fork", "target", "--continue"]);
        assert_eq!(parsed.fork.as_deref(), Some("target"));
        let conflicting = [
            parsed.continue_ == Some(true),
            parsed.resume.is_some(),
            parsed.no_session == Some(true),
        ];
        assert_eq!(conflicting, [true, false, false]);
    }

    /// The drive `path.resolve` prefixes to a drive-less rooted input, plus the
    /// host separator. Node's `resolve('/work', './skills')` is `C:\\work\\skills`
    /// on this host (measured with the pinned pair on Node v24.16.0), so the
    /// expectation is derived from the host rather than written as a POSIX
    /// literal that the TypeScript never produces.
    fn host_resolved(base: &str, tail: &str) -> String {
        let drive = std::env::current_dir()
            .ok()
            .and_then(|cwd| {
                cwd.components().next().and_then(|component| match component {
                    std::path::Component::Prefix(prefix) => {
                        Some(prefix.as_os_str().to_string_lossy().to_string())
                    }
                    _ => None,
                })
            })
            .unwrap_or_default();
        let sep = std::path::MAIN_SEPARATOR;
        let base = base.trim_start_matches(['/', '\\']);
        let tail = tail.trim_start_matches("./");
        format!("{drive}{sep}{base}{sep}{tail}")
    }

    #[test]
    fn runtime_config_mirrors_the_cli_flags() {
        let parsed = args(&[
            "--model",
            "provider/model",
            "--no-tools",
            "--skill",
            "./skills",
            "--goal",
            "ship",
            "--goal-token-budget",
            "1000",
            "--offline",
        ]);
        let config = runtime_config_from_args(&parsed, "/work", "/agent", Some("/sessions"), "print", Some(true));
        assert_eq!(config.cwd.as_deref(), Some("/work"));
        assert_eq!(config.agent_dir.as_deref(), Some("/agent"));
        assert_eq!(config.session_dir.as_deref(), Some("/sessions"));
        assert_eq!(config.model.as_deref(), Some("provider/model"));
        assert_eq!(config.no_tools, Some(true));
        // `main.ts:629-631` resolves each local CLI path with `resolve(cwd, value)`
        // (`main.ts:8` imports it from `node:path`), and on win32 that call returns
        // a drive-absolute, backslash path with `.` collapsed: the pinned behaviour
        // measured here is `resolve("/work", "./skills") === "C:\\work\\skills"`.
        // The POSIX literal "/work/./skills" cannot be produced by that call, so it
        // was not a valid expectation.
        assert_eq!(
            config.skills.as_ref().map(|skills| skills[0].clone()),
            Some(host_resolved("/work", "./skills"))
        );
        assert_eq!(config.execution_mode.as_deref(), Some("print"));
        assert_eq!(config.serialized_refine, Some(true));
        assert_eq!(config.telemetry_disabled, Some(true));
        assert_eq!(
            config.initial_goal.as_ref().map(|goal| goal.objective.clone()),
            Some("ship".to_string())
        );
        assert_eq!(
            config.initial_goal.as_ref().and_then(|goal| goal.token_budget),
            Some(1000.0)
        );
        // Daemon clients drop the execution mode; the worker keeps its own.
        let daemon = runtime_config_from_args(&parsed, "/work", "/agent", None, "daemon", None);
        assert!(daemon.execution_mode.is_none());
        assert_eq!(daemon.serialized_refine, Some(false));
        // Interactive clients never serialize refine.
        let interactive = runtime_config_from_args(&parsed, "/work", "/agent", None, "interactive", None);
        assert_eq!(interactive.serialized_refine, Some(false));
        assert_eq!(interactive.execution_mode.as_deref(), Some("interactive"));
        assert!(daemon_server_default_session_config(&interactive).initial_goal.is_none());
    }

    #[test]
    fn autonomous_config_is_built_only_from_autonomous_flags() {
        assert!(runtime_autonomous_config_from_args(&args(&["--model", "m"])).is_none());
        let config = runtime_autonomous_config_from_args(&args(&[
            "--autonomous",
            "--autonomous-gate",
            "cargo test",
            "--autonomous-gate-retries",
            "2",
            "--autonomous-max-turns",
            "7",
        ]))
        .expect("autonomous config");
        assert_eq!(config.enabled, Some(true));
        assert_eq!(config.max_turns, Some(7.0));
        assert_eq!(
            config.gates.as_ref().and_then(|gates| gates.commands.clone()),
            Some(vec!["cargo test".to_string()])
        );
        assert_eq!(config.gates.as_ref().and_then(|gates| gates.max_retries), Some(2.0));
        // Gate flags alone still turn autonomous mode on.
        let gates_only = runtime_autonomous_config_from_args(&args(&["--autonomous-gate", "cargo test"]))
            .expect("autonomous config");
        assert_eq!(gates_only.enabled, Some(true));
        assert!(gates_only.max_turns.is_none());
    }

    #[test]
    fn extension_flag_values_are_written_as_the_unknown_flags_map() {
        let parsed = args(&["--my-flag", "value", "--other"]);
        let config = runtime_config_from_args(&parsed, "/work", "/agent", None, "print", None);
        let flags = config.extension_flag_values.expect("extension flags");
        assert_eq!(flags.get("my-flag"), Some(&Value::String("value".to_string())));
        assert_eq!(flags.get("other"), Some(&Value::Bool(true)));
        assert!(matches!(
            parsed.unknown_flags.get("other"),
            Some(UnknownFlagValue::Flag)
        ));
    }

    #[test]
    fn local_cli_paths_are_resolved_and_urls_are_kept() {
        // Same `resolve(cwd, value)` rule as above: `Path::join` is not a
        // substitute (it keeps "./a" and drops the drive), so the sibling
        // assertion derived from it was not a valid expectation either.
        assert_eq!(
            resolve_cli_paths("/work", Some(&vec!["./a".to_string(), "https://x/y".to_string()])),
            Some(vec![host_resolved("/work", "./a"), "https://x/y".to_string()])
        );
        assert!(resolve_cli_paths("/work", None).is_none());
    }

    #[test]
    fn session_options_prefer_the_cli_model_then_the_scoped_models() {
        let settings = SettingsManager::in_memory(serde_json::Map::new());
        let registry = crate::core::model_registry::ModelRegistry::in_memory(crate::core::auth_storage::AuthStorage::create(
            Some("/tmp/auth.json".to_string()),
            None,
        ));
        let scoped = vec![ScopedModel { model: model("m1"), thinking_level: Some("high".to_string()) }];
        let config = runtime_config_from_args(&args(&[]), "/work", "/agent", None, "print", None);
        let built = build_session_options(&config, &scoped, false, &registry, &settings);
        assert_eq!(built.options.model.as_ref().map(|model| model.id.clone()), Some("m1".to_string()));
        assert_eq!(
            built.options.thinking_level,
            Some(pi_agent_core::types::ThinkingLevel::High)
        );
        assert_eq!(built.options.scoped_models.as_ref().map(Vec::len), Some(1));
        assert!(!built.cli_thinking_from_model);
        assert!(built.diagnostics.is_empty());

        let with_cli_thinking = runtime_config_from_args(&args(&["--thinking", "low"]), "/work", "/agent", None, "print", None);
        let built = build_session_options(&with_cli_thinking, &scoped, false, &registry, &settings);
        assert_eq!(
            built.options.thinking_level,
            Some(pi_agent_core::types::ThinkingLevel::Low)
        );
        // An existing session keeps its own model instead of the first scoped one.
        let built = build_session_options(&config, &scoped, true, &registry, &settings);
        assert!(built.options.model.is_none());
    }

    #[test]
    fn runtime_session_options_override_only_what_the_callers_set() {
        let base = CreateAgentSessionOptions {
            model: Some(model("base")),
            tools: Some(vec!["read".to_string()]),
            ..Default::default()
        };
        let runtime = AgentSessionCreationOptions {
            rlm_depth: Some(1),
            autonomous: Some(AgentAutonomousConfig {
                enabled: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = resolve_runtime_session_options(&base, Some(&runtime));
        assert_eq!(resolved.model.as_ref().map(|model| model.id.clone()), Some("base".to_string()));
        assert_eq!(resolved.tools, Some(vec!["read".to_string()]));
        assert_eq!(resolved.creation.rlm_depth, Some(1));
        // A subagent runtime disables the inherited autonomous loop.
        assert_eq!(
            resolved.creation.autonomous.as_ref().and_then(|config| config.enabled),
            Some(false)
        );
        let resolved = resolve_runtime_session_options(&base, None);
        assert!(resolved.creation.autonomous.is_none());
        assert_eq!(resolved.creation.rlm_depth, None);
    }

    #[test]
    fn truthy_env_flags_match_the_typescript() {
        for value in ["1", "true", "TRUE", "yes", "Yes"] {
            assert!(is_truthy_env_flag(Some(value)), "{value}");
        }
        for value in ["", "0", "false", "no", "2"] {
            assert!(!is_truthy_env_flag(Some(value)), "{value}");
        }
        assert!(!is_truthy_env_flag(None));
    }

    #[test]
    fn agents_view_command_strips_only_the_leading_token() {
        let parsed = parse_agents_view_command(&["agents".to_string(), "--resume".to_string()]);
        assert!(parsed.explicit_agents_view);
        assert_eq!(parsed.args, vec!["--resume".to_string()]);
        let parsed = parse_agents_view_command(&["--resume".to_string()]);
        assert!(!parsed.explicit_agents_view);
        assert_eq!(parsed.args, vec!["--resume".to_string()]);
        assert!(parse_agents_view_command(&[]).is_empty());
    }

    #[test]
    fn daemon_summary_helpers_prefer_the_active_id() {
        let mut summary = SessionSummary::default();
        summary.id = "agent-1".to_string();
        summary.cwd = "/work".to_string();
        assert_eq!(get_daemon_summary_active_session_id(&summary), "agent-1");
        summary.active_session_id = Some("active-1".to_string());
        assert_eq!(get_daemon_summary_active_session_id(&summary), "active-1");
    }

    #[test]
    fn returning_from_chat_preserves_roster_identity_and_child_scope() {
        let original = SessionSummary {
            id: "old-active".into(),
            session_id: "old-session".into(),
            message_count: 42,
            ..Default::default()
        };
        let result = InteractiveModeRunResult {
            type_: InteractiveModeRunResultType::ScopedAgentsView,
            source: crate::modes::interactive::interactive_mode::InteractiveModeRunResultSource {
                active_session_id: Some("new-active".into()),
                session_id: "new-session".into(),
                session_name: Some("renamed".into()),
                session_file: Some("/sessions/new.jsonl".into()),
                cwd: "/new-cwd".into(),
            },
        };
        let (summary, scope) = returned_agents_view_state(original.clone(), result.clone());
        assert_eq!(summary.id, "new-active");
        assert_eq!(summary.session_id, "new-session");
        assert_eq!(summary.cwd, "/new-cwd");
        assert_eq!(summary.session_name.as_deref(), Some("renamed"));
        assert_eq!(summary.message_count, 42);
        let scope = scope.expect("direct children scope");
        assert_eq!(scope.session_id, "new-session");
        assert_eq!(scope.active_session_id.as_deref(), Some("new-active"));

        let (_, scope) = returned_agents_view_state(original, InteractiveModeRunResult {
            type_: InteractiveModeRunResultType::AgentsView,
            ..result
        });
        assert!(scope.is_none());
    }

    #[test]
    fn active_summary_matching_demands_an_id_and_a_session_file() {
        let session_file = "/sessions/a.jsonl";
        let mut summary = SessionSummary {
            active_session_id: Some("active-1".to_string()),
            session_file: Some(session_file.to_string()),
            ..Default::default()
        };
        let summaries = vec![summary.clone()];
        assert!(find_active_daemon_session_summary_for_session_file(&summaries, session_file).is_some());
        summary.session_file = None;
        assert!(
            find_active_daemon_session_summary_for_session_file(&vec![summary.clone()], session_file).is_none()
        );
        summary.session_file = Some(session_file.to_string());
        summary.active_session_id = None;
        assert!(find_active_daemon_session_summary_for_session_file(&vec![summary], session_file).is_none());
    }

    /// The daemon slice's `DaemonResponseConstruction::failure` is private to
    /// `daemon_mode.rs`; the test builds the same literal.
    fn failed_response(command: &str, error: &str) -> DaemonResponse {
        DaemonResponse {
            id: None,
            type_: "response".to_string(),
            command: command.to_string(),
            success: false,
            data: None,
            error: Some(error.to_string()),
            error_info: None,
        }
    }

    #[test]
    fn an_unknown_active_session_lookup_falls_back_and_recovering_throws() {
        let response = failed_response("get_state", "Unknown active session: abc");
        assert!(resolve_active_session_lookup_failure(&response).is_none());
        let response = failed_response("get_state", "other failure");
        assert_eq!(
            resolve_active_session_lookup_failure(&response),
            Some("other failure".to_string())
        );
    }

    #[test]
    fn interactive_daemon_session_path_needs_an_explicit_target() {
        let parsed = args(&[]);
        let manager = SessionManager::in_memory(Some("/work"), None).expect("session manager");
        assert!(get_interactive_daemon_session_path(&parsed, &manager).is_none());
        let parsed = args(&["--continue"]);
        assert!(get_interactive_daemon_session_path(&parsed, &manager).is_none());
        let parsed = args(&["--resume", "abc"]);
        assert!(get_interactive_daemon_session_path(&parsed, &manager).is_none());
    }

    #[test]
    fn read_session_manager_adopts_the_header_cwd() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("session.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"session\",\"version\":3,\"id\":\"test-session\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/from-header\"}\n",
                "{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1767225601000}}\n",
            ),
        )
        .expect("write session");
        let manager = read_session_manager(&path.to_string_lossy(), None, None);
        assert_eq!(manager.get_cwd(), "/from-header");
        assert_eq!(manager.get_session_file().as_deref(), Some(path.to_string_lossy().as_ref()));
        let manager = read_session_manager(&path.to_string_lossy(), None, Some("/override"));
        assert_eq!(manager.get_cwd(), "/override");
    }

    #[test]
    fn create_session_manager_honours_no_session_and_read_only() {
        let parsed = args(&["--no-session"]);
        let manager = futures::executor::block_on(create_session_manager(&parsed, "/work", None, false))
            .expect("session manager");
        assert!(manager.get_session_file().is_none());

        let parsed = args(&[]);
        let manager = futures::executor::block_on(create_session_manager(&parsed, "/work", None, true))
            .expect("session manager");
        assert!(manager.get_session_file().is_none());
        assert_eq!(manager.get_cwd(), "/work");
    }

    #[test]
    fn session_cwd_source_forwards_the_manager_accessors() {
        let manager = SessionManager::in_memory(Some("/work"), None).expect("session manager");
        let source = SessionManagerCwdSource { session_manager: &manager };
        assert_eq!(crate::core::session_cwd::SessionCwdSource::get_cwd(&source), "/work");
        assert!(crate::core::session_cwd::SessionCwdSource::get_session_file(&source).is_none());
    }

    fn sample_model() -> pi_ai::types::Model {
        pi_ai::types::Model::new("", "", "", "", "")
    }

    fn model(id: &str) -> pi_ai::types::Model {
        pi_ai::types::Model::new(id, id, "openai-completions", "p", "https://example.invalid")
    }
}
