//! Port of packages/coding-agent/src/cli/owned-session-worker.ts
//!
//! The owned session worker runs a headless `--print`/`--json`/`--rpc` session in
//! a child process so the frontend survives a worker crash and can replay the
//! last persisted session.
//!
//! The frontend takes the same explicit host seam the other CLI/mode modules use
//! (`RpcModeHost` in modes/rpc/rpc-mode.ts, `DaemonCommandIo` in cli/daemon-command.ts).
//! Every host method mirrors exactly one Node call the TypeScript makes; the
//! `OwnedSessionWorkerHost` trait documents each one. `native_owned_worker`
//! provides the process, stream, signal, and owner-channel implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use pi_ai::types::BoxFuture;

use crate::cli::command_registry::{
    is_help_command_request, public_command_names, removed_command_names,
};
use crate::cli::subprocess_launch::{
    create_cli_subprocess_launch_spec, current_entrypoint, current_exec_args, current_exec_path,
    CliSubprocessLaunchSpec, ProcessEnv,
};
use crate::core::agent_session::{AgentSession, AgentSessionEvent};
use crate::core::agent_session_runtime::AgentSessionRuntime;
use crate::core::orphan_process_journal::{
    clear_orphan_process_journal, kill_orphan_process, read_active_orphan_processes,
    should_reap_orphan_process, ORPHAN_PROCESS_JOURNAL_ENV,
};
use crate::core::session_lease::{SESSION_LEASE_OWNER_ID_ENV, SESSION_LEASES_ENABLED_ENV};

const OWNED_WORKER_ENV: &str = "PRIME_AGENT_INTERNAL_OWNED_WORKER";
const OWNED_RECOVERY_DESCRIPTOR_ENV: &str = "PRIME_AGENT_INTERNAL_OWNED_RECOVERY_DESCRIPTOR";
const OWNED_PROFILE_ENV: &str = "PRIME_AGENT_INTERNAL_OWNED_PROFILE";
/// `process.env.PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND`.
const LEGACY_OWNED_WORKER_FRONTEND_ENV: &str =
    "PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND";

/// `let closeOwnerWatch: (() => void) | undefined`.
fn close_owner_watch() -> &'static Mutex<Option<Arc<dyn Fn() + Send + Sync>>> {
    static CLOSE_OWNER_WATCH: std::sync::OnceLock<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>> =
        std::sync::OnceLock::new();
    CLOSE_OWNER_WATCH.get_or_init(|| Mutex::new(None))
}

/// `OwnedSessionWorkerProfile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OwnedSessionWorkerProfile {
    Print,
    Json,
    Rpc,
    InteractiveEphemeral,
}

impl OwnedSessionWorkerProfile {
    /// The TypeScript literal.
    pub fn as_str(self) -> &'static str {
        match self {
            OwnedSessionWorkerProfile::Print => "print",
            OwnedSessionWorkerProfile::Json => "json",
            OwnedSessionWorkerProfile::Rpc => "rpc",
            OwnedSessionWorkerProfile::InteractiveEphemeral => "interactive-ephemeral",
        }
    }

    pub fn from_value(value: &str) -> Option<Self> {
        match value {
            "print" => Some(OwnedSessionWorkerProfile::Print),
            "json" => Some(OwnedSessionWorkerProfile::Json),
            "rpc" => Some(OwnedSessionWorkerProfile::Rpc),
            "interactive-ephemeral" => Some(OwnedSessionWorkerProfile::InteractiveEphemeral),
            _ => None,
        }
    }
}

impl Default for OwnedSessionWorkerProfile {
    fn default() -> Self {
        OwnedSessionWorkerProfile::Print
    }
}

/// `isOwnedSessionWorkerProcess(environment = process.env)`.
pub fn is_owned_session_worker_process(environment: &ProcessEnv) -> bool {
    environment.get(OWNED_WORKER_ENV).map(String::as_str) == Some("1")
}

/// `OwnedSessionRecoveryDescriptor`.
///
/// `profile` and `updatedAt` stay unchecked strings: `readOwnedRecoveryDescriptor`
/// validates only `version`, `sessionId`, `cwd` and `sessionFile`, exactly like
/// the TypeScript cast it performs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OwnedSessionRecoveryDescriptor {
    pub version: i64,
    pub profile: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    pub cwd: String,
    pub updated_at: String,
}

/// `NON_SESSION_FLAGS`.
fn non_session_flags() -> [&'static str; 6] {
    ["--help", "-h", "--version", "-v", "--list-models", "--export"]
}

/// `NON_SESSION_COMMANDS = new Set([...PUBLIC_COMMAND_NAMES, ...REMOVED_COMMAND_NAMES])`.
fn is_non_session_command(name: &str) -> bool {
    public_command_names().contains(name) || removed_command_names().contains(name)
}

/// `valueAfter(args, flag)`.
fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1).cloned())
}

/// `hasNonSessionOperation(args)`.
fn has_non_session_operation(args: &[String]) -> bool {
    if args
        .iter()
        .any(|arg| non_session_flags().contains(&arg.as_str()) || arg.starts_with("--export="))
    {
        return true;
    }
    let Some(first) = args.first() else {
        return false;
    };
    if first == "help" {
        let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
        return is_help_command_request(&rest);
    }
    is_non_session_command(first)
}

/// `isStartupBenchmark(environment)`.
fn is_startup_benchmark(environment: &ProcessEnv) -> bool {
    let value = environment
        .get("PI_STARTUP_BENCHMARK")
        .map(|value| value.to_lowercase())
        .unwrap_or_default();
    value == "1" || value == "true" || value == "yes"
}

/// `classifyOwnedSessionWorkerInvocation(args, stdinIsTTY, environment = process.env)`.
pub fn classify_owned_session_worker_invocation(
    args: &[String],
    stdin_is_tty: Option<bool>,
    environment: &ProcessEnv,
) -> Option<OwnedSessionWorkerProfile> {
    if is_owned_session_worker_process(environment) || has_non_session_operation(args) {
        return None;
    }

    let mode = value_after(args, "--mode");
    if mode.as_deref() == Some("daemon") {
        return None;
    }
    if mode.as_deref() == Some("rpc") {
        return Some(OwnedSessionWorkerProfile::Rpc);
    }
    if mode.as_deref() == Some("json") {
        return Some(OwnedSessionWorkerProfile::Json);
    }
    if args.iter().any(|arg| arg == "--print" || arg == "-p") || stdin_is_tty == Some(false) {
        return Some(OwnedSessionWorkerProfile::Print);
    }
    if args.iter().any(|arg| arg == "--no-session") || is_startup_benchmark(environment) {
        return Some(OwnedSessionWorkerProfile::InteractiveEphemeral);
    }
    None
}

/// `OwnedWorkerLaunchSpec = CliSubprocessLaunchSpec`.
pub type OwnedWorkerLaunchSpec = CliSubprocessLaunchSpec;

/// `createOwnedWorkerLaunchSpec(args, executable = process.execPath, execArgs = process.execArgv, entrypoint = process.argv[1])`.
///
/// `None` for `executable`/`exec_args`/`entrypoint` means "use the runtime
/// default", exactly like the TypeScript default parameters.
pub fn create_owned_worker_launch_spec(
    args: &[String],
    executable: Option<&str>,
    exec_args: Option<&[String]>,
    entrypoint: Option<&str>,
) -> OwnedWorkerLaunchSpec {
    let executable = executable.map(str::to_string).unwrap_or_else(current_exec_path);
    let exec_args = exec_args.map(|args| args.to_vec()).unwrap_or_else(current_exec_args);
    let entrypoint = entrypoint.map(str::to_string).unwrap_or_else(current_entrypoint);
    create_cli_subprocess_launch_spec(args, Some(&executable), &exec_args, Some(&entrypoint))
}

/// `readOwnedRecoveryDescriptor(path)`.
fn read_owned_recovery_descriptor(path: &str) -> Option<OwnedSessionRecoveryDescriptor> {
    // The worker may have stopped before creating a recoverable session.
    let contents = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let version = value.get("version").and_then(serde_json::Value::as_i64);
    if version != Some(1) {
        return None;
    }
    let session_id = value.get("sessionId")?.as_str()?.to_string();
    let cwd = value.get("cwd")?.as_str()?.to_string();
    let session_file = match value.get("sessionFile") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(session_file)) => Some(session_file.clone()),
        Some(_) => return None,
    };
    let profile = value
        .get("profile")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let updated_at = value
        .get("updatedAt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(OwnedSessionRecoveryDescriptor {
        version: 1,
        profile,
        session_id,
        session_file,
        cwd,
        updated_at,
    })
}

/// `writeOwnedRecoveryDescriptor(path, profile, session)`.
fn write_owned_recovery_descriptor(
    path: &str,
    profile: OwnedSessionWorkerProfile,
    session: &AgentSession,
) {
    let descriptor = OwnedSessionRecoveryDescriptor {
        version: 1,
        profile: profile.as_str().to_string(),
        session_id: session.session_id(),
        session_file: session.session_file(),
        cwd: session.session_manager.lock().expect("session manager poisoned").get_cwd(),
        updated_at: now_iso8601(),
    };
    let temp_path = format!("{}.{}.tmp", path, std::process::id());
    let payload = format!(
        "{}\n",
        serde_json::to_string(&descriptor).unwrap_or_else(|_| "{}".to_string())
    );
    if std::fs::write(&temp_path, payload).is_err() {
        return;
    }
    restrict_to_owner(&temp_path);
    let _ = std::fs::rename(&temp_path, path);
}

/// `writeFileSync(..., { mode: 0o600 })` + `chmodSync(tempPath, 0o600)`.
#[cfg(unix)]
fn restrict_to_owner(path: &str) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

/// Windows has no `0o600` mode; Node ignores `mode` there, so the port does too.
#[cfg(not(unix))]
fn restrict_to_owner(_path: &str) {}

/// `new Date().toISOString()`.
fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// `installOwnedSessionRecoveryTracking(runtime)`.
pub fn install_owned_session_recovery_tracking(runtime: &Arc<AgentSessionRuntime>) {
    let path = std::env::var(OWNED_RECOVERY_DESCRIPTOR_ENV).ok();
    let profile = std::env::var(OWNED_PROFILE_ENV)
        .ok()
        .and_then(|value| OwnedSessionWorkerProfile::from_value(&value));
    let (Some(path), Some(profile)) = (path, profile) else {
        return;
    };

    let unsubscribe_session: Arc<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>> =
        Arc::new(Mutex::new(None));
    // `const bind = (session: AgentSession) => { ... }` returns void in the
    // TypeScript; `onSessionReplaced` takes an async listener, so the port wraps
    // the same body in a future.
    let bind: Arc<dyn Fn(Arc<AgentSession>) + Send + Sync> = Arc::new({
        let unsubscribe_session = unsubscribe_session.clone();
        move |session: Arc<AgentSession>| {
            if let Some(unsubscribe) = unsubscribe_session.lock().expect("session listener poisoned").take() {
                unsubscribe();
            }
            write_owned_recovery_descriptor(&path, profile, &session);
            let last_session_file = Arc::new(Mutex::new(session.session_file()));
            let path = path.clone();
            let last_session_file_for_listener = last_session_file.clone();
            let session_for_listener = session.clone();
            let unsubscribe = session.subscribe(Arc::new(move |event: AgentSessionEvent| {
                if event.type_name() != "message_start" && event.type_name() != "session_info_changed"
                {
                    return;
                }
                let current = session_for_listener.session_file();
                let mut last_session_file = last_session_file_for_listener
                    .lock()
                    .expect("last session file poisoned");
                if current != *last_session_file {
                    *last_session_file = current;
                    write_owned_recovery_descriptor(&path, profile, &session_for_listener);
                }
            }));
            *unsubscribe_session.lock().expect("session listener poisoned") = Some(unsubscribe);
        }
    });

    bind(runtime.session());
    let bind_for_async = bind.clone();
    runtime.on_session_replaced(Arc::new(move |session: Arc<AgentSession>| {
        bind_for_async(session);
        let done: BoxFuture<()> = Box::pin(async {});
        done
    }));
}

/// `createRpcRecoveryArgs(args, sessionPath)`.
pub fn create_rpc_recovery_args(args: &[String], session_path: &str) -> Vec<String> {
    let mut recovered: Vec<String> = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].clone();
        if arg == "--resume" || arg == "-r" || arg == "--fork" {
            index += 2;
            continue;
        }
        if arg.starts_with("--resume=") {
            index += 1;
            continue;
        }
        if arg == "--continue" || arg == "-c" {
            index += 1;
            continue;
        }
        recovered.push(arg);
        index += 1;
    }
    recovered.push("--resume".to_string());
    recovered.push(session_path.to_string());
    recovered
}

/// `exitCodeForSignal(signal)`.
fn exit_code_for_signal(signal: Option<&str>) -> i32 {
    if signal == Some("SIGHUP") {
        return 129;
    }
    if signal == Some("SIGINT") {
        return 130;
    }
    if signal == Some("SIGTERM") {
        return 143;
    }
    1
}

/// The Node process/child-process surface this module needs.
///
/// The port takes an explicit host seam like `modes/rpc/rpc-mode.ts`. Every method
/// mirrors exactly one Node call the TypeScript makes.
pub trait OwnedSessionWorkerHost: Send + Sync {
    /// `process.platform`.
    fn platform(&self) -> String;
    /// `process.stdin.isTTY`
    fn stdin_is_tty(&self) -> Option<bool>;
    /// `process.cwd()`.
    fn cwd(&self) -> String;
    /// `process.env` used for the child environment and the profile checks.
    fn env(&self) -> ProcessEnv;
    /// `process.pid`.
    fn pid(&self) -> i64;
    /// `tmpdir()`.
    fn tmp_dir(&self) -> String;
    /// `randomUUID()`.
    fn random_uuid(&self) -> String;
    /// `Date.now()`.
    fn now_ms(&self) -> f64;

    /// `attachJsonlLineReader(process.stdin, onLine)`; returns the detach function.
    fn attach_stdin_lines(
        &self,
        on_line: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.stdin.once("end", handler)`; returns the `off` function.
    fn on_stdin_end(&self, handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.stdin.pause()`.
    fn pause_stdin(&self);
    /// `process.stdin.resume()`.
    fn resume_stdin(&self);
    /// `process.stdin.pipe(childInput)`.
    fn pipe_stdin(&self, child_stdin: &Arc<dyn OwnedWorkerChildStdin>);
    /// `process.stdin.unpipe(childInput)`.
    fn unpipe_stdin(&self, child_stdin: &Arc<dyn OwnedWorkerChildStdin>);

    /// `process.stdout.write(text)`; `false` means backpressure.
    fn write_stdout(&self, text: &str) -> bool;
    /// `process.stdout.once("drain", handler)`.
    fn on_stdout_drain(&self, handler: Arc<dyn Fn() + Send + Sync>);
    /// `process.stderr` sink for `childError.pipe(process.stderr, { end: false })`.
    fn write_stderr(&self, text: &str);

    /// `process.on(signal, handler)`; returns the cleanup function.
    fn on_signal(&self, signal: &str, handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `spawnHidden(command, args, options)`.
    fn spawn_worker(
        &self,
        launch: &OwnedWorkerLaunchSpec,
        options: OwnedWorkerSpawnOptions,
    ) -> Result<Arc<dyn OwnedWorkerChild>, String>;

    /// `process.kill(-pid, "SIGKILL")`; `true` also when the group is already absent.
    /// `false` means delivery failed or the outcome is unknown.
    fn kill_process_group(&self, pid: i64, signal: &str) -> bool;
    /// `setTimeout(handler, ms).unref()`.
    fn schedule_unref_timeout(&self, ms: u64, handler: Arc<dyn Fn() + Send + Sync>);
    /// `process.exit(code)`.
    fn exit(&self, code: i32);
    /// `process.exitCode = code`.
    fn set_exit_code(&self, code: i32);
    /// `process.kill(process.pid, signal)`.
    fn kill_self(&self, signal: &str);
    /// `process.channel` is present.
    fn has_owner_channel(&self) -> bool;
    /// `process.channel.unref()`.
    fn unref_owner_channel(&self);
    /// `process.once("disconnect", handler)`; returns the `off` function.
    fn once_owner_disconnect(&self, handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.connected`.
    fn owner_channel_connected(&self) -> bool;
    /// `process.disconnect()`.
    fn disconnect_owner_channel(&self);
}

/// `SpawnOptions` restricted to the fields `spawnWorker` sets.
///
/// `stdio` is collapsed into the four booleans: the TypeScript builds
/// `["inherit"|"pipe", "pipe"|"inherit", "pipe"|"inherit", "ipc"]` and the port
/// reproduces the same three descriptors plus the IPC channel.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedWorkerSpawnOptions {
    pub cwd: String,
    /// `detached: process.platform !== "win32"`.
    pub detached: bool,
    pub env: ProcessEnv,
    /// `interactive`: all three descriptors are `"inherit"`.
    pub inherit_stdio: bool,
    pub pipe_stdin: bool,
    pub pipe_stdout: bool,
    pub pipe_stderr: bool,
}

/// `child.once("close", (code, signal) => ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedWorkerExit {
    /// `code`; `None` is Node's `null`.
    pub code: Option<i32>,
    /// `signal`; `None` is Node's `null`.
    pub signal: Option<String>,
}

/// Node's writable stream end that the worker's stdin is.
pub trait OwnedWorkerChildStdin: Send + Sync {
    /// `input.writable`.
    fn writable(&self) -> bool;
    /// `input.write(text)`; `false` means backpressure.
    fn write(&self, text: &str) -> bool;
    /// Raw pipe writes preserve bytes, including UTF-8 sequences split across reads.
    fn write_bytes(&self, bytes: &[u8]) -> bool {
        self.write(&String::from_utf8_lossy(bytes))
    }
    /// `input.once("drain", handler)`.
    fn once_drain(&self, handler: Arc<dyn Fn() + Send + Sync>);
    /// `input.end()`.
    fn end(&self);
}

/// Node's readable stream end that the worker's stdout is.
pub trait OwnedWorkerChildStdout: Send + Sync {
    /// `childOutput.pause()`.
    fn pause(&self);
    /// `childOutput.resume()`.
    fn resume(&self);
    /// `attachJsonlLineReader(childOutput, onLine)`; returns the detach function.
    fn attach_lines(&self, on_line: Arc<dyn Fn(String) + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `childOutput.pipe(process.stdout, { end: false })`.
    fn pipe_to_stdout(&self);
}

/// Node's readable stream end that the worker's stderr is.
pub trait OwnedWorkerChildStderr: Send + Sync {
    /// `childError.pipe(process.stderr, { end: false })`.
    fn pipe_to_stderr(&self);
}

/// `ChildProcess` restricted to the members this module touches.
pub trait OwnedWorkerChild: Send + Sync {
    /// `child.pid`.
    fn pid(&self) -> Option<i64>;
    /// `child.stdin ?? undefined`.
    fn stdin(&self) -> Option<Arc<dyn OwnedWorkerChildStdin>>;
    /// `child.stdout ?? undefined`.
    fn stdout(&self) -> Option<Arc<dyn OwnedWorkerChildStdout>>;
    /// `child.stderr ?? undefined`.
    fn stderr(&self) -> Option<Arc<dyn OwnedWorkerChildStderr>>;
    /// `child.exitCode`.
    fn exit_code(&self) -> Option<i32>;
    /// `child.signalCode`.
    fn signal_code(&self) -> Option<String>;
    /// `child.kill(signal)`.
    fn kill(&self, signal: &str);
    /// `child.once("error", reject)` / `child.once("close", resolve)`.
    fn wait(&self) -> BoxFuture<Result<OwnedWorkerExit, String>>;
    /// `child.connected`.
    fn connected(&self) -> bool;
    /// `child.disconnect()`.
    fn disconnect(&self);
}

/// One in-flight anonymous RPC command: `{ publicId?, command }`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingRpcCommand {
    public_id: Option<String>,
    command: String,
}

/// The error the frontend fabricates for a command that was in flight when the
/// worker stopped.
const STOPPED_WORKER_ERROR: &str =
    "The isolated session worker stopped during this command; its result is uncertain and was not replayed";

/// `runOwnedSessionWorkerFrontend(args, profile)`.
pub async fn run_owned_session_worker_frontend(
    args: &[String],
    profile: OwnedSessionWorkerProfile,
    host: Arc<dyn OwnedSessionWorkerHost>,
) -> Result<i32, String> {
    let frontend = Arc::new(OwnedSessionWorkerFrontend::new(args, profile, host));
    let result = frontend.clone().run().await;
    // `finally`: detach the readers and drop the descriptor. Resource reaping
    // clears the journal only after all required kills have succeeded.
    frontend.cleanup();
    result
}

/// The frontend state machine. The TypeScript keeps this in closure variables;
/// the port keeps it on a struct so the same order is preserved while the
/// callbacks stay `Send + Sync`.
struct OwnedSessionWorkerFrontend {
    args: Vec<String>,
    profile: OwnedSessionWorkerProfile,
    host: Arc<dyn OwnedSessionWorkerHost>,
    interactive: bool,
    recovery_descriptor_path: String,
    orphan_process_journal_path: String,
    current_child: Mutex<Option<Arc<dyn OwnedWorkerChild>>>,
    terminating: AtomicBool,
    termination_signal: Mutex<Option<String>>,
    stdin_ended: AtomicBool,
    // Node delivers stdin and child callbacks serially. Native readers run on
    // separate threads, so installing a new input and flushing buffered records
    // must be atomic with forwarding new input and its EOF.
    rpc_input_bridge: Mutex<()>,
    current_rpc_input: Mutex<Option<Arc<dyn OwnedWorkerChildStdin>>>,
    current_rpc_output: Mutex<Option<Arc<dyn OwnedWorkerChildStdout>>>,
    rpc_stdout_paused: AtomicBool,
    detach_rpc_input: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    detach_rpc_output: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    buffered_rpc_input: Mutex<Vec<String>>,
    pending_rpc_commands: Mutex<indexmap::IndexMap<String, PendingRpcCommand>>,
    anonymous_rpc_id_prefix: String,
    anonymous_rpc_command_id: Mutex<u64>,
}

impl OwnedSessionWorkerFrontend {
    fn new(
        args: &[String],
        profile: OwnedSessionWorkerProfile,
        host: Arc<dyn OwnedSessionWorkerHost>,
    ) -> Self {
        let random_uuid = host.random_uuid();
        let recovery_descriptor_path = join_path(
            &host.tmp_dir(),
            &format!(
                "prime-agent-owned-{}-{}.json",
                host.pid(),
                &random_uuid.chars().take(12).collect::<String>()
            ),
        );
        let orphan_process_journal_path = format!("{recovery_descriptor_path}.orphans.jsonl");
        Self {
            args: args.to_vec(),
            profile,
            host: host.clone(),
            interactive: profile == OwnedSessionWorkerProfile::InteractiveEphemeral,
            recovery_descriptor_path,
            orphan_process_journal_path,
            current_child: Mutex::new(None),
            terminating: AtomicBool::new(false),
            termination_signal: Mutex::new(None),
            stdin_ended: AtomicBool::new(false),
            rpc_input_bridge: Mutex::new(()),
            current_rpc_input: Mutex::new(None),
            current_rpc_output: Mutex::new(None),
            rpc_stdout_paused: AtomicBool::new(false),
            detach_rpc_input: Mutex::new(None),
            detach_rpc_output: Mutex::new(None),
            buffered_rpc_input: Mutex::new(Vec::new()),
            pending_rpc_commands: Mutex::new(indexmap::IndexMap::new()),
            anonymous_rpc_id_prefix: format!("prime-agent-owned-{random_uuid}"),
            anonymous_rpc_command_id: Mutex::new(0),
        }
    }

    /// `prepareRpcInput(line)`.
    fn prepare_rpc_input(&self, line: &str) -> String {
        let Ok(command) = serde_json::from_str::<serde_json::Value>(line) else {
            // The worker preserves the existing parse-error response contract.
            return format!("{line}\n");
        };
        if !command.is_object() {
            return format!("{line}\n");
        }
        let Some(command_type) = command.get("type").and_then(serde_json::Value::as_str) else {
            return format!("{line}\n");
        };
        if command_type == "extension_ui_response" || command_type == "ack_result" {
            return format!("{line}\n");
        }
        let public_id = command
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let internal_id = match &public_id {
            Some(public_id) => public_id.clone(),
            None => {
                let mut counter = self.anonymous_rpc_command_id.lock().expect("rpc id poisoned");
                *counter += 1;
                format!("{}-{}", self.anonymous_rpc_id_prefix, *counter)
            }
        };
        self.pending_rpc_commands.lock().expect("rpc pending poisoned").insert(
            internal_id.clone(),
            PendingRpcCommand {
                public_id: public_id.clone(),
                command: command_type.to_string(),
            },
        );
        if public_id.is_some() {
            return format!("{line}\n");
        }
        let mut object = command.as_object().cloned().unwrap_or_default();
        object.insert("id".to_string(), serde_json::Value::String(internal_id));
        crate::modes::rpc::jsonl::serialize_json_line(&serde_json::Value::Object(object))
    }

    /// `observeRpcOutput(line)`.
    fn observe_rpc_output(self: &Arc<Self>, line: &str) {
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) else {
            return;
        };
        if !parsed.is_object() {
            return;
        }
        let id = parsed.get("id").and_then(serde_json::Value::as_str).map(str::to_string);
        let pending = id
            .as_ref()
            .and_then(|id| self.pending_rpc_commands.lock().expect("rpc pending poisoned").get(id).cloned());
        let response_type = parsed.get("type").and_then(serde_json::Value::as_str).map(str::to_string);
        let mut framed = format!("{line}\n");
        if response_type.as_deref() == Some("response") && pending.as_ref().map(|p| p.public_id.is_none()).unwrap_or(true) {
            // `const { id: _internalId, ...publicResponse } = response`.
            let mut public_response = parsed.as_object().cloned().unwrap_or_default();
            public_response.shift_remove("id");
            framed = crate::modes::rpc::jsonl::serialize_json_line(&serde_json::Value::Object(public_response));
        }
        if !self.host.write_stdout(&framed) && !self.rpc_stdout_paused.load(Ordering::SeqCst) {
            self.rpc_stdout_paused.store(true, Ordering::SeqCst);
            if let Some(output) = self.current_rpc_output.lock().expect("rpc output poisoned").clone() {
                output.pause();
            }
            let frontend = self.clone();
            self.host.on_stdout_drain(Arc::new(move || {
                frontend.rpc_stdout_paused.store(false, Ordering::SeqCst);
                if let Some(output) = frontend.current_rpc_output.lock().expect("rpc output poisoned").clone()
                {
                    output.resume();
                }
            }));
        }
        if response_type.as_deref() != Some("response") {
            return;
        }
        let Some(command) = parsed.get("command").and_then(serde_json::Value::as_str) else {
            return;
        };
        if let Some(id) = &id {
            let mut pending_commands = self.pending_rpc_commands.lock().expect("rpc pending poisoned");
            if pending_commands.get(id).map(|pending| pending.command.as_str()) == Some(command) {
                pending_commands.shift_remove(id);
            }
        }
    }

    /// `failPendingRpcCommands()`.
    fn fail_pending_rpc_commands(&self) {
        let pending: Vec<PendingRpcCommand> = self
            .pending_rpc_commands
            .lock()
            .expect("rpc pending poisoned")
            .values()
            .cloned()
            .collect();
        for command in pending {
            let mut object = serde_json::Map::new();
            if let Some(public_id) = &command.public_id {
                object.insert("id".to_string(), serde_json::Value::String(public_id.clone()));
            }
            object.insert("type".to_string(), serde_json::Value::String("response".to_string()));
            object.insert("command".to_string(), serde_json::Value::String(command.command));
            object.insert("success".to_string(), serde_json::Value::Bool(false));
            object.insert("error".to_string(), serde_json::Value::String(STOPPED_WORKER_ERROR.to_string()));
            let _ = self
                .host
                .write_stdout(&crate::modes::rpc::jsonl::serialize_json_line(&serde_json::Value::Object(object)));
        }
        self.pending_rpc_commands.lock().expect("rpc pending poisoned").clear();
    }

    /// `reapWorkerResources(workerPid)`.
    fn reap_worker_resources(&self, worker_pid: Option<i64>) {
        self.reap_worker_resources_with_kill(worker_pid, kill_orphan_process);
    }

    fn reap_worker_resources_with_kill<F>(&self, worker_pid: Option<i64>, kill_orphan: F)
    where
        F: Fn(i64) -> bool,
    {
        let Some(worker_pid) = worker_pid.filter(|pid| *pid != 0) else {
            return;
        };
        let mut cleanup_complete = true;
        if self.host.platform() != "win32"
            && !self.host.kill_process_group(worker_pid, "SIGKILL")
        {
            cleanup_complete = false;
        }
        match read_active_orphan_processes(&self.orphan_process_journal_path, worker_pid) {
            Ok(orphans) => {
                for orphan in orphans {
                    if !should_reap_orphan_process(&orphan) {
                        if orphan.process_start_id.is_none() {
                            // Identity-free evidence cannot prove the original process is absent.
                            cleanup_complete = false;
                        }
                        // A known start id that no longer matches retires the original record.
                        continue;
                    }
                    if !kill_orphan(orphan.pid) {
                        cleanup_complete = false;
                    }
                }
            }
            // An unreadable journal can still name surviving processes.
            Err(_) => cleanup_complete = false,
        }
        if cleanup_complete {
            clear_orphan_process_journal(&self.orphan_process_journal_path);
        }
    }

    /// `spawnWorker(workerArgs)`.
    fn spawn_worker(self: &Arc<Self>, worker_args: &[String]) -> Result<Arc<dyn OwnedWorkerChild>, String> {
        let launch = create_owned_worker_launch_spec(worker_args, None, None, None);
        let bridge_stdin =
            self.profile == OwnedSessionWorkerProfile::Rpc || self.host.stdin_is_tty() != Some(true);
        let mut env = self.host.env();
        env.insert(OWNED_WORKER_ENV.to_string(), "1".to_string());
        env.insert(
            OWNED_RECOVERY_DESCRIPTOR_ENV.to_string(),
            self.recovery_descriptor_path.clone(),
        );
        env.insert(OWNED_PROFILE_ENV.to_string(), self.profile.as_str().to_string());
        env.insert(
            ORPHAN_PROCESS_JOURNAL_ENV.to_string(),
            self.orphan_process_journal_path.clone(),
        );
        env.insert(SESSION_LEASES_ENABLED_ENV.to_string(), "1".to_string());
        env.insert(
            SESSION_LEASE_OWNER_ID_ENV.to_string(),
            format!("owned-{}", self.host.random_uuid()),
        );
        let child = self.host.spawn_worker(
            &launch,
            OwnedWorkerSpawnOptions {
                cwd: self.host.cwd(),
                detached: self.host.platform() != "win32",
                env,
                inherit_stdio: self.interactive,
                pipe_stdin: !self.interactive && bridge_stdin,
                pipe_stdout: !self.interactive,
                pipe_stderr: !self.interactive,
            },
        )?;
        *self.current_child.lock().expect("current child poisoned") = Some(child.clone());
        if !self.interactive {
            let child_input = child.stdin();
            let child_output = child.stdout();
            let child_error = child.stderr();
            if (bridge_stdin && child_input.is_none()) || child_output.is_none() || child_error.is_none()
            {
                child.kill("SIGTERM");
                return Err("Owned session worker did not expose bridged stdio".to_string());
            }
            child_error.expect("checked above").pipe_to_stderr();
            if self.profile == OwnedSessionWorkerProfile::Rpc {
                let Some(child_input) = child_input else {
                    return Err("Owned RPC worker did not expose stdin".to_string());
                };
                let child_output = child_output.expect("checked above");
                *self.current_rpc_output.lock().expect("rpc output poisoned") = Some(child_output.clone());
                if self.rpc_stdout_paused.load(Ordering::SeqCst) {
                    child_output.pause();
                }
                let frontend = self.clone();
                let detach = child_output.attach_lines(Arc::new(move |line: String| {
                    frontend.observe_rpc_output(&line);
                }));
                *self.detach_rpc_output.lock().expect("rpc detach poisoned") = Some(detach);
                let _bridge = self.rpc_input_bridge.lock().expect("rpc input bridge poisoned");
                *self.current_rpc_input.lock().expect("rpc input poisoned") = Some(child_input.clone());
                let buffered: Vec<String> = self
                    .buffered_rpc_input
                    .lock()
                    .expect("buffered rpc poisoned")
                    .drain(..)
                    .collect();
                for buffered in buffered {
                    child_input.write(&buffered);
                }
                if self.stdin_ended.load(Ordering::SeqCst) {
                    child_input.end();
                }
            } else {
                if let Some(child_input) = &child_input {
                    self.host.pipe_stdin(child_input);
                }
                child_output.expect("checked above").pipe_to_stdout();
            }
        }
        Ok(child)
    }

    /// `attachJsonlLineReader(process.stdin, ...)` for the `rpc` profile.
    fn attach_rpc_input(self: &Arc<Self>) {
        let frontend = self.clone();
        let detach = self.host.attach_stdin_lines(Arc::new(move |line: String| {
            let _bridge = frontend.rpc_input_bridge.lock().expect("rpc input bridge poisoned");
            let framed = frontend.prepare_rpc_input(&line);
            let input = frontend.current_rpc_input.lock().expect("rpc input poisoned").clone();
            match input {
                Some(input) if input.writable() => {
                    if !input.write(&framed) {
                        frontend.host.pause_stdin();
                        let frontend_for_drain = frontend.clone();
                        let input_for_drain = input.clone();
                        input.once_drain(Arc::new(move || {
                            let current = frontend_for_drain
                                .current_rpc_input
                                .lock()
                                .expect("rpc input poisoned")
                                .clone();
                            let same = match &current {
                                Some(current) => Arc::ptr_eq(current, &input_for_drain),
                                None => false,
                            };
                            if same && !frontend_for_drain.stdin_ended.load(Ordering::SeqCst) {
                                frontend_for_drain.host.resume_stdin();
                            }
                        }));
                    }
                }
                _ => frontend.buffered_rpc_input.lock().expect("buffered rpc poisoned").push(framed),
            }
        }));
        *self.detach_rpc_input.lock().expect("rpc detach poisoned") = Some(detach);
        let frontend = self.clone();
        self.host.on_stdin_end(Arc::new(move || {
            let _bridge = frontend.rpc_input_bridge.lock().expect("rpc input bridge poisoned");
            frontend.stdin_ended.store(true, Ordering::SeqCst);
            if let Some(input) = frontend.current_rpc_input.lock().expect("rpc input poisoned").clone() {
                input.end();
            }
        }));
    }

    /// The `while (true)` recovery loop of `runOwnedSessionWorkerFrontend`.
    async fn run(self: Arc<Self>) -> Result<i32, String> {
        if self.profile == OwnedSessionWorkerProfile::Rpc {
            self.attach_rpc_input();
        }

        let mut signals: Vec<String> = vec!["SIGINT".to_string(), "SIGTERM".to_string()];
        if self.host.platform() != "win32" {
            signals.push("SIGHUP".to_string());
        }
        let mut signal_handlers: Vec<(String, Arc<dyn Fn() + Send + Sync>, Arc<dyn Fn() + Send + Sync>)> =
            Vec::new();
        for signal in signals {
            let frontend = self.clone();
            let signal_for_handler = signal.clone();
            let handler: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                frontend.terminating.store(true, Ordering::SeqCst);
                let mut termination_signal = frontend.termination_signal.lock().expect("signal poisoned");
                if termination_signal.is_none() {
                    *termination_signal = Some(signal_for_handler.clone());
                }
                drop(termination_signal);
                if let Some(child) = frontend.current_child.lock().expect("current child poisoned").clone() {
                    forward_signal(child.as_ref(), &signal_for_handler);
                }
            });
            let cleanup = self.host.on_signal(&signal, handler.clone());
            signal_handlers.push((signal, handler, cleanup));
        }

        let result = self.run_recovery_loop().await;

        for (signal, handler, cleanup) in signal_handlers {
            let _ = signal;
            let _ = handler;
            cleanup();
        }
        result
    }

    /// The recovery loop body.
    async fn run_recovery_loop(self: &Arc<Self>) -> Result<i32, String> {
        let mut worker_args = self.args.clone();
        let mut recovery_attempt = 0usize;
        loop {
            if self.terminating.load(Ordering::SeqCst) {
                let signal = self.termination_signal.lock().expect("signal poisoned").clone();
                return Ok(exit_code_for_signal(signal.as_deref()));
            }
            let worker_started_at = self.host.now_ms();
            let child = self.spawn_worker(&worker_args)?;
            let worker_pid = child.pid();
            let exit = child.wait().await?;
            // `code ?? exitCodeForSignal(signal)`.
            let exit_signal = exit.signal.clone();
            let exit = OwnedWorkerExit {
                code: Some(exit.code.unwrap_or_else(|| exit_code_for_signal(exit_signal.as_deref()))),
                signal: exit_signal,
            };
            *self.current_child.lock().expect("current child poisoned") = None;
            {
                let _bridge = self.rpc_input_bridge.lock().expect("rpc input bridge poisoned");
                *self.current_rpc_input.lock().expect("rpc input poisoned") = None;
            }
            *self.current_rpc_output.lock().expect("rpc output poisoned") = None;
            if self.profile == OwnedSessionWorkerProfile::Rpc && !self.stdin_ended.load(Ordering::SeqCst) {
                self.host.resume_stdin();
            }
            if let Some(detach) = self.detach_rpc_output.lock().expect("rpc detach poisoned").take() {
                detach();
            }
            if !self.interactive && self.profile != OwnedSessionWorkerProfile::Rpc {
                if let Some(child_stdin) = child.stdin() {
                    self.host.unpipe_stdin(&child_stdin);
                }
            }
            if child.connected() {
                child.disconnect();
            }
            self.reap_worker_resources(worker_pid);
            let rpc_crashed = self.profile == OwnedSessionWorkerProfile::Rpc
                && !self.terminating.load(Ordering::SeqCst)
                && (exit.code != Some(0)
                    || exit.signal.is_some()
                    || !self.pending_rpc_commands.lock().expect("rpc pending poisoned").is_empty());
            let worker_exit_code = if rpc_crashed && exit.code == Some(0) {
                1
            } else {
                exit.code.unwrap_or(0)
            };
            if self.host.now_ms() - worker_started_at >= 60_000.0 {
                recovery_attempt = 0;
            }
            if rpc_crashed {
                self.fail_pending_rpc_commands();
            }
            let should_recover =
                rpc_crashed && !self.stdin_ended.load(Ordering::SeqCst) && recovery_attempt < 3;
            if !should_recover {
                let signal = self.termination_signal.lock().expect("signal poisoned").clone();
                return Ok(match signal {
                    Some(signal) => exit_code_for_signal(Some(&signal)),
                    None => worker_exit_code,
                });
            }
            let descriptor = read_owned_recovery_descriptor(&self.recovery_descriptor_path);
            let descriptor = match descriptor.filter(|descriptor| descriptor.session_file.is_some()) {
                Some(descriptor) => descriptor,
                None => {
                    let signal = self.termination_signal.lock().expect("signal poisoned").clone();
                    return Ok(match signal {
                        Some(signal) => exit_code_for_signal(Some(&signal)),
                        None => worker_exit_code,
                    });
                }
            };
            worker_args = create_rpc_recovery_args(
                &self.args,
                descriptor.session_file.as_deref().unwrap_or_default(),
            );
            let retry_delay = [250u64, 1000, 5000].get(recovery_attempt).copied().unwrap_or(5000);
            recovery_attempt += 1;
            tokio::time::sleep(std::time::Duration::from_millis(retry_delay)).await;
            if self.terminating.load(Ordering::SeqCst) {
                let signal = self.termination_signal.lock().expect("signal poisoned").clone();
                return Ok(exit_code_for_signal(signal.as_deref()));
            }
        }
    }

    /// The `finally` block: detach the readers and drop the descriptor.
    fn cleanup(&self) {
        if let Some(detach) = self.detach_rpc_input.lock().expect("rpc detach poisoned").take() {
            detach();
        }
        if let Some(detach) = self.detach_rpc_output.lock().expect("rpc detach poisoned").take() {
            detach();
        }
        let _ = std::fs::remove_file(&self.recovery_descriptor_path);
        // A failed or incomplete reap must leave readable evidence for a retry.
    }
}

/// `forwardSignal(child, signal)`.
fn forward_signal(child: &dyn OwnedWorkerChild, signal: &str) {
    if child.exit_code().is_none() && child.signal_code().is_none() {
        child.kill(signal);
    }
}

/// `join(dir, name)` for the two segments this module builds.
fn join_path(dir: &str, name: &str) -> String {
    let separator = if dir.ends_with('/') || dir.ends_with('\\') { "" } else { std::path::MAIN_SEPARATOR_STR };
    format!("{dir}{separator}{name}")
}

/// `maybeRunOwnedSessionWorkerFrontend(args, forceLegacyFrontend = false)`.
pub async fn maybe_run_owned_session_worker_frontend(
    args: &[String],
    force_legacy_frontend: bool,
    host: Arc<dyn OwnedSessionWorkerHost>,
) -> Result<bool, String> {
    if !force_legacy_frontend
        && host.env().get(LEGACY_OWNED_WORKER_FRONTEND_ENV).map(String::as_str) != Some("1")
    {
        return Ok(false);
    }
    let profile = classify_owned_session_worker_invocation(args, host.stdin_is_tty(), &host.env());
    let Some(profile) = profile else {
        return Ok(false);
    };
    let exit_code = run_owned_session_worker_frontend(args, profile, host.clone()).await?;
    host.set_exit_code(exit_code);
    Ok(true)
}

/// `installOwnedSessionWorkerOwnerWatch()`.
pub fn install_owned_session_worker_owner_watch(host: Arc<dyn OwnedSessionWorkerHost>) -> Result<(), String> {
    if !is_owned_session_worker_process(&host.env()) {
        return Ok(());
    }
    if !host.has_owner_channel() {
        return Err("Owned session worker is missing its owner channel".to_string());
    }

    let owner_gone = Arc::new(AtomicBool::new(false));
    let terminate: Arc<dyn Fn() + Send + Sync> = Arc::new({
        let owner_gone = owner_gone.clone();
        let host = host.clone();
        move || {
            if owner_gone.swap(true, Ordering::SeqCst) {
                return;
            }
            *close_owner_watch().lock().expect("owner watch poisoned") = None;
            let host_for_timer = host.clone();
            host.schedule_unref_timeout(
                5000,
                Arc::new(move || {
                    if host_for_timer.platform() != "win32" {
                        // Fall through to terminating only this process.
                        if host_for_timer.kill_process_group(host_for_timer.pid(), "SIGKILL") {
                            return;
                        }
                    }
                    host_for_timer.exit(143);
                }),
            );
            host.kill_self("SIGTERM");
        }
    });
    let detach_disconnect = host.once_owner_disconnect(terminate.clone());
    host.unref_owner_channel();
    *close_owner_watch().lock().expect("owner watch poisoned") = Some(Arc::new({
        let owner_gone = owner_gone.clone();
        let host = host.clone();
        move || {
            if owner_gone.swap(true, Ordering::SeqCst) {
                return;
            }
            detach_disconnect();
            if host.owner_channel_connected() {
                host.disconnect_owner_channel();
            }
            *close_owner_watch().lock().expect("owner watch poisoned") = None;
        }
    }));
    Ok(())
}

/// `closeOwnedSessionWorkerOwnerWatch()`.
pub fn close_owned_session_worker_owner_watch() {
    let closer = close_owner_watch().lock().expect("owner watch poisoned").clone();
    if let Some(closer) = closer {
        closer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> ProcessEnv {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn leaves_resident_interactive_and_non_session_operations_in_the_frontend() {
        let empty = ProcessEnv::new();
        assert_eq!(classify_owned_session_worker_invocation(&[], Some(true), &empty), None);
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--mode", "daemon"]), Some(false), &empty),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--help"]), Some(false), &empty),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--version"]), Some(false), &empty),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--list-models"]), Some(false), &empty),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(
                &strings(&["--export", "session.jsonl"]),
                Some(false),
                &empty
            ),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["help"]), Some(false), &empty),
            None
        );
        assert_eq!(
            classify_owned_session_worker_invocation(
                &strings(&["help", "me", "fix", "this"]),
                Some(false),
                &empty
            ),
            Some(OwnedSessionWorkerProfile::Print)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["daemon", "list"]), Some(false), &empty),
            None
        );
    }

    #[test]
    fn every_public_command_stays_in_the_frontend() {
        let empty = ProcessEnv::new();
        for command in public_command_names() {
            assert_eq!(
                classify_owned_session_worker_invocation(&[command.to_string()], Some(false), &empty),
                None,
                "public command {command} must stay in the frontend"
            );
        }
    }

    #[test]
    fn does_not_recursively_route_an_owned_worker() {
        assert!(is_owned_session_worker_process(&env(&[(OWNED_WORKER_ENV, "1")])));
        assert!(!is_owned_session_worker_process(&env(&[])));
        assert_eq!(
            classify_owned_session_worker_invocation(
                &strings(&["--mode", "rpc"]),
                Some(true),
                &env(&[(OWNED_WORKER_ENV, "1")])
            ),
            None
        );
    }

    #[test]
    fn profiles_follow_mode_print_and_tty_rules() {
        let empty = ProcessEnv::new();
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["-p", "hello"]), Some(true), &empty),
            Some(OwnedSessionWorkerProfile::Print)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&[], Some(false), &empty),
            Some(OwnedSessionWorkerProfile::Print)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--mode", "json"]), Some(true), &empty),
            Some(OwnedSessionWorkerProfile::Json)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--mode", "rpc"]), Some(true), &empty),
            Some(OwnedSessionWorkerProfile::Rpc)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--no-session"]), Some(true), &empty),
            Some(OwnedSessionWorkerProfile::InteractiveEphemeral)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&[], Some(true), &env(&[("PI_STARTUP_BENCHMARK", "TRUE")])),
            Some(OwnedSessionWorkerProfile::InteractiveEphemeral)
        );
        assert_eq!(
            classify_owned_session_worker_invocation(&strings(&["--export=a.jsonl"]), Some(false), &empty),
            None
        );
    }

    #[test]
    fn preserves_the_current_runtime_flags_and_cli_entrypoint() {
        let launch = create_owned_worker_launch_spec(
            &strings(&["--mode", "rpc"]),
            Some("/node"),
            Some(&strings(&["--loader", "tsx"])),
            Some("/cli.ts"),
        );
        assert_eq!(launch.command, "/node");
        assert_eq!(launch.args, strings(&["--loader", "tsx", "/cli.ts", "--mode", "rpc"]));
    }

    #[test]
    fn restarts_rpc_against_the_exact_persisted_session_without_changing_public_flags() {
        assert_eq!(
            create_rpc_recovery_args(
                &strings(&[
                    "--mode",
                    "rpc",
                    "--continue",
                    "--resume",
                    "/old.jsonl",
                    "--model",
                    "openai/gpt-5"
                ]),
                "/current.jsonl"
            ),
            strings(&["--mode", "rpc", "--model", "openai/gpt-5", "--resume", "/current.jsonl"])
        );
        assert_eq!(
            create_rpc_recovery_args(&strings(&["-c", "-r", "/old.jsonl", "--fork=x"]), "/new.jsonl"),
            strings(&["--fork=x", "--resume", "/new.jsonl"])
        );
    }

    #[test]
    fn exit_codes_follow_the_signal_table() {
        assert_eq!(exit_code_for_signal(Some("SIGHUP")), 129);
        assert_eq!(exit_code_for_signal(Some("SIGINT")), 130);
        assert_eq!(exit_code_for_signal(Some("SIGTERM")), 143);
        assert_eq!(exit_code_for_signal(None), 1);
        assert_eq!(exit_code_for_signal(Some("SIGUSR1")), 1);
    }

    #[test]
    fn reads_a_recovery_descriptor_written_by_this_module() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("descriptor.json");
        let path_string = path.to_string_lossy().to_string();
        let descriptor = OwnedSessionRecoveryDescriptor {
            version: 1,
            profile: "rpc".to_string(),
            session_id: "session-1".to_string(),
            session_file: Some("/tmp/session.jsonl".to_string()),
            cwd: "/work".to_string(),
            updated_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        std::fs::write(&path, serde_json::to_string(&descriptor).unwrap()).unwrap();
        assert_eq!(read_owned_recovery_descriptor(&path_string), Some(descriptor));
    }

    #[test]
    fn rejects_descriptors_without_the_required_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("descriptor.json");
        let path_string = path.to_string_lossy().to_string();
        let write = |value: &str| std::fs::write(&path, value).unwrap();

        write(r#"{"version":2,"sessionId":"a","cwd":"/work"}"#);
        assert_eq!(read_owned_recovery_descriptor(&path_string), None);
        write(r#"{"version":1,"cwd":"/work"}"#);
        assert_eq!(read_owned_recovery_descriptor(&path_string), None);
        write(r#"{"version":1,"sessionId":"a"}"#);
        assert_eq!(read_owned_recovery_descriptor(&path_string), None);
        write(r#"{"version":1,"sessionId":"a","cwd":"/work","sessionFile":7}"#);
        assert_eq!(read_owned_recovery_descriptor(&path_string), None);
        write("not json");
        assert_eq!(read_owned_recovery_descriptor(&path_string), None);
        assert_eq!(read_owned_recovery_descriptor(&format!("{path_string}.missing")), None);

        // `sessionFile` is optional.
        write(r#"{"version":1,"sessionId":"a","cwd":"/work"}"#);
        let descriptor = read_owned_recovery_descriptor(&path_string).unwrap();
        assert_eq!(descriptor.session_file, None);
        assert_eq!(descriptor.session_id, "a");
        assert_eq!(descriptor.cwd, "/work");
    }

    #[test]
    fn rpc_input_framing_adds_internal_ids_only_for_anonymous_commands() {
        let host = RecordingHost::default();
        let frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Rpc,
            Arc::new(host.clone()),
        );

        // A command with a public id is forwarded verbatim.
        let with_id = frontend.prepare_rpc_input(r#"{"id":"request-1","type":"get_state"}"#);
        assert_eq!(with_id, "{\"id\":\"request-1\",\"type\":\"get_state\"}\n");

        // An anonymous command gains a generated internal id.
        let anonymous = frontend.prepare_rpc_input(r#"{"type":"get_state","marker":"first"}"#);
        assert!(anonymous.ends_with("\n"));
        let parsed: serde_json::Value = serde_json::from_str(anonymous.trim()).unwrap();
        let internal_id = parsed.get("id").unwrap().as_str().unwrap().to_string();
        assert!(internal_id.starts_with("prime-agent-owned-"));
        assert_eq!(parsed.get("marker").unwrap().as_str(), Some("first"));

        // Acknowledgements and extension responses are never tracked.
        let ack = frontend.prepare_rpc_input(r#"{"type":"ack_result","commandId":"c1"}"#);
        assert_eq!(ack, "{\"type\":\"ack_result\",\"commandId\":\"c1\"}\n");
        let extension = frontend.prepare_rpc_input(r#"{"type":"extension_ui_response","id":"e1"}"#);
        assert_eq!(extension, "{\"type\":\"extension_ui_response\",\"id\":\"e1\"}\n");

        // Malformed input keeps the parse-error contract.
        assert_eq!(frontend.prepare_rpc_input("nope"), "nope\n");
        assert_eq!(frontend.prepare_rpc_input("[]"), "[]\n");

        let pending = frontend.pending_rpc_commands.lock().unwrap().clone();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending.get("request-1").unwrap().command, "get_state");
        assert!(pending.get(&internal_id).unwrap().public_id.is_none());
    }

    #[test]
    fn rpc_output_strips_internal_ids_and_clears_settled_commands() {
        let host = RecordingHost::default();
        let frontend = Arc::new(OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Rpc,
            Arc::new(host.clone()),
        ));
        let anonymous = frontend.prepare_rpc_input(r#"{"type":"get_state"}"#);
        let internal_id = serde_json::from_str::<serde_json::Value>(anonymous.trim())
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        frontend.observe_rpc_output(&format!(
            r#"{{"id":"{internal_id}","type":"response","command":"get_state","success":true}}"#
        ));
        assert!(frontend.pending_rpc_commands.lock().unwrap().is_empty());
        let written = host.stdout_text();
        assert_eq!(written, "{\"type\":\"response\",\"command\":\"get_state\",\"success\":true}\n");
    }

    #[test]
    fn rpc_output_keeps_public_ids_and_drops_malformed_lines() {
        let host = RecordingHost::default();
        let frontend = Arc::new(OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Rpc,
            Arc::new(host.clone()),
        ));
        frontend.prepare_rpc_input(r#"{"id":"request-1","type":"get_state"}"#);
        frontend.observe_rpc_output("not json");
        assert_eq!(host.stdout_text(), "");
        frontend.observe_rpc_output(r#"{"id":"request-1","type":"response","command":"get_state","success":true}"#);
        assert_eq!(
            host.stdout_text(),
            "{\"id\":\"request-1\",\"type\":\"response\",\"command\":\"get_state\",\"success\":true}\n"
        );
        assert!(frontend.pending_rpc_commands.lock().unwrap().is_empty());
    }

    #[test]
    fn fail_pending_rpc_commands_reports_every_unfinished_command() {
        let host = RecordingHost::default();
        let frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Rpc,
            Arc::new(host.clone()),
        );
        frontend.prepare_rpc_input(r#"{"id":"request-1","type":"get_state"}"#);
        frontend.prepare_rpc_input(r#"{"type":"get_state"}"#);
        frontend.fail_pending_rpc_commands();
        assert!(frontend.pending_rpc_commands.lock().unwrap().is_empty());
        let written = host.stdout_text();
        let lines: Vec<&str> = written.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first.get("id").unwrap().as_str(), Some("request-1"));
        assert_eq!(first.get("success").unwrap().as_bool(), Some(false));
        assert_eq!(first.get("error").unwrap().as_str(), Some(STOPPED_WORKER_ERROR));
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert!(second.get("id").is_none());
        assert_eq!(second.get("command").unwrap().as_str(), Some("get_state"));
    }

    fn write_orphan_record(
        path: &std::path::Path,
        owner_pid: i64,
        pid: i64,
        process_start_id: Option<String>,
    ) {
        std::fs::write(
            path,
            format!(
                "{}\n",
                serde_json::json!({
                    "version": 1,
                    "pid": pid,
                    "ownerPid": owner_pid,
                    "processStartId": process_start_id,
                    "active": true,
                    "recordedAt": "2026-01-01T00:00:00.000Z"
                })
            ),
        )
        .unwrap();
    }

    fn write_active_orphan(path: &std::path::Path, owner_pid: i64, pid: i64) {
        let process_start_id = crate::core::session_lease::get_process_start_id(pid)
            .expect("the current test process must have an observable start id");
        write_orphan_record(path, owner_pid, pid, Some(process_start_id));
    }

    #[test]
    fn failed_orphan_kill_keeps_the_journal_until_a_later_kill_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("orphans.jsonl");
        let host = RecordingHost::default();
        let worker_pid = 4242;
        let orphan_pid = std::process::id() as i64;
        let mut frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Print,
            Arc::new(host),
        );
        frontend.recovery_descriptor_path = dir.path().join("recovery.json").to_string_lossy().into_owned();
        frontend.orphan_process_journal_path = journal.to_string_lossy().into_owned();
        write_active_orphan(&journal, worker_pid, orphan_pid);

        let calls = Arc::new(Mutex::new(Vec::new()));
        let results = Arc::new(Mutex::new(std::collections::VecDeque::from([false, true])));
        let kill_orphan = {
            let calls = calls.clone();
            let results = results.clone();
            move |pid| {
                calls.lock().unwrap().push(pid);
                results.lock().unwrap().pop_front().unwrap()
            }
        };

        frontend.reap_worker_resources_with_kill(Some(worker_pid), &kill_orphan);
        assert_eq!(*calls.lock().unwrap(), vec![orphan_pid]);
        assert!(journal.exists(), "a false kill result must retain recovery evidence");
        assert_eq!(
            read_active_orphan_processes(&frontend.orphan_process_journal_path, worker_pid)
                .unwrap()
                .len(),
            1
        );

        frontend.cleanup();
        assert!(journal.exists(), "frontend cleanup must not erase a failed reap");

        frontend.reap_worker_resources_with_kill(Some(worker_pid), &kill_orphan);
        assert_eq!(*calls.lock().unwrap(), vec![orphan_pid, orphan_pid]);
        assert!(!journal.exists(), "a successful retry must clear the journal");
    }

    #[test]
    fn failed_group_kill_still_reaches_orphans_and_allows_a_complete_retry() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("orphans.jsonl");
        let host = RecordingHost::default();
        *host.platform.lock().unwrap() = Some("linux".to_string());
        *host.process_group_kill_results.lock().unwrap() =
            std::collections::VecDeque::from([false, true]);
        let worker_pid = 4242;
        let orphan_pid = std::process::id() as i64;
        let mut frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Print,
            Arc::new(host.clone()),
        );
        frontend.orphan_process_journal_path = journal.to_string_lossy().into_owned();
        write_active_orphan(&journal, worker_pid, orphan_pid);

        let orphan_calls = Arc::new(Mutex::new(Vec::new()));
        let kill_orphan = {
            let orphan_calls = orphan_calls.clone();
            move |pid| {
                orphan_calls.lock().unwrap().push(pid);
                true
            }
        };

        frontend.reap_worker_resources_with_kill(Some(worker_pid), &kill_orphan);
        assert_eq!(host.process_group_kills.lock().unwrap().len(), 1);
        assert_eq!(*orphan_calls.lock().unwrap(), vec![orphan_pid]);
        assert!(journal.exists(), "a false group kill must retain the journal");

        frontend.reap_worker_resources_with_kill(Some(worker_pid), &kill_orphan);
        assert_eq!(host.process_group_kills.lock().unwrap().len(), 2);
        assert_eq!(*orphan_calls.lock().unwrap(), vec![orphan_pid, orphan_pid]);
        assert!(!journal.exists(), "a complete retry must clear the journal");
    }

    #[test]
    fn absent_process_group_and_no_orphans_clear_the_journal() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("orphans.jsonl");
        std::fs::write(&journal, "").unwrap();
        let host = RecordingHost::default();
        *host.platform.lock().unwrap() = Some("linux".to_string());
        *host.process_group_kill_results.lock().unwrap() =
            std::collections::VecDeque::from([true]);
        let worker_pid = 4242;
        let mut frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Print,
            Arc::new(host.clone()),
        );
        frontend.orphan_process_journal_path = journal.to_string_lossy().into_owned();

        frontend.reap_worker_resources_with_kill(Some(worker_pid), |_| {
            panic!("an empty journal must not reach the orphan kill path")
        });

        assert_eq!(
            *host.process_group_kills.lock().unwrap(),
            vec![(worker_pid, "SIGKILL".to_string())]
        );
        assert!(!journal.exists(), "an absent group and no orphans are a successful no-op");
    }

    #[test]
    fn stale_and_dead_known_identity_orphans_retire_without_kills() {
        let dir = tempfile::tempdir().unwrap();
        let worker_pid = 4242;
        for orphan_pid in [std::process::id() as i64, i64::MAX] {
            let journal = dir.path().join(format!("orphans-{orphan_pid}.jsonl"));
            write_orphan_record(
                &journal,
                worker_pid,
                orphan_pid,
                Some(format!("not-the-current-start-id-{orphan_pid}")),
            );
            let host = RecordingHost::default();
            let mut frontend = OwnedSessionWorkerFrontend::new(
                &[],
                OwnedSessionWorkerProfile::Print,
                Arc::new(host),
            );
            frontend.orphan_process_journal_path = journal.to_string_lossy().into_owned();

            frontend.reap_worker_resources_with_kill(Some(worker_pid), |pid| {
                panic!("absent known identity {pid} must not reach the kill path")
            });

            assert!(
                !journal.exists(),
                "stale or dead known identity {orphan_pid} must retire its record"
            );
        }
    }

    #[test]
    fn identity_free_orphan_with_unknown_outcome_is_retained() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("orphans.jsonl");
        let host = RecordingHost::default();
        let worker_pid = 4242;
        let orphan_pid = i64::MAX;
        let mut frontend = OwnedSessionWorkerFrontend::new(
            &[],
            OwnedSessionWorkerProfile::Print,
            Arc::new(host),
        );
        frontend.orphan_process_journal_path = journal.to_string_lossy().into_owned();
        write_orphan_record(&journal, worker_pid, orphan_pid, None);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let kill_orphan = {
            let calls = calls.clone();
            move |pid| {
                calls.lock().unwrap().push(pid);
                false
            }
        };

        frontend.reap_worker_resources_with_kill(Some(worker_pid), kill_orphan);

        #[cfg(windows)]
        assert!(calls.lock().unwrap().is_empty());
        #[cfg(not(windows))]
        assert_eq!(*calls.lock().unwrap(), vec![orphan_pid]);
        assert!(journal.exists(), "unknown identity and kill evidence must remain retryable");
    }

    #[test]
    fn forwards_a_signal_only_while_the_child_is_still_running() {
        let running = RecordingChild::default();
        forward_signal(&running, "SIGTERM");
        assert_eq!(running.killed.lock().unwrap().clone(), vec!["SIGTERM".to_string()]);

        let exited = RecordingChild::default();
        *exited.exit_code_value.lock().unwrap() = Some(0);
        forward_signal(&exited, "SIGTERM");
        assert!(exited.killed.lock().unwrap().is_empty());

        let signalled = RecordingChild::default();
        *signalled.signal_code_value.lock().unwrap() = Some("SIGKILL".to_string());
        forward_signal(&signalled, "SIGTERM");
        assert!(signalled.killed.lock().unwrap().is_empty());
    }

    // -----------------------------------------------------------------
    // Test doubles for `OwnedSessionWorkerHost`.
    // -----------------------------------------------------------------

    #[derive(Default, Clone)]
    struct RecordingHost {
        stdout: Arc<Mutex<Vec<String>>>,
        stderr: Arc<Mutex<Vec<String>>>,
        exit_codes: Arc<Mutex<Vec<i32>>>,
        platform: Arc<Mutex<Option<String>>>,
        process_group_kills: Arc<Mutex<Vec<(i64, String)>>>,
        process_group_kill_results: Arc<Mutex<std::collections::VecDeque<bool>>>,
    }

    impl RecordingHost {
        fn stdout_text(&self) -> String {
            self.stdout.lock().unwrap().concat()
        }
    }

    impl OwnedSessionWorkerHost for RecordingHost {
        fn platform(&self) -> String {
            self.platform
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "win32".to_string())
        }
        fn stdin_is_tty(&self) -> Option<bool> {
            Some(true)
        }
        fn cwd(&self) -> String {
            "/work".to_string()
        }
        fn env(&self) -> ProcessEnv {
            ProcessEnv::new()
        }
        fn pid(&self) -> i64 {
            4242
        }
        fn tmp_dir(&self) -> String {
            "/tmp".to_string()
        }
        fn random_uuid(&self) -> String {
            "00000000-0000-4000-8000-000000000000".to_string()
        }
        fn now_ms(&self) -> f64 {
            0.0
        }
        fn attach_stdin_lines(
            &self,
            _on_line: Arc<dyn Fn(String) + Send + Sync>,
        ) -> Arc<dyn Fn() + Send + Sync> {
            Arc::new(|| {})
        }
        fn on_stdin_end(&self, _handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync> {
            Arc::new(|| {})
        }
        fn pause_stdin(&self) {}
        fn resume_stdin(&self) {}
        fn pipe_stdin(&self, _child_stdin: &Arc<dyn OwnedWorkerChildStdin>) {}
        fn unpipe_stdin(&self, _child_stdin: &Arc<dyn OwnedWorkerChildStdin>) {}
        fn write_stdout(&self, text: &str) -> bool {
            self.stdout.lock().unwrap().push(text.to_string());
            true
        }
        fn on_stdout_drain(&self, _handler: Arc<dyn Fn() + Send + Sync>) {}
        fn write_stderr(&self, text: &str) {
            self.stderr.lock().unwrap().push(text.to_string());
        }
        fn on_signal(
            &self,
            _signal: &str,
            _handler: Arc<dyn Fn() + Send + Sync>,
        ) -> Arc<dyn Fn() + Send + Sync> {
            Arc::new(|| {})
        }
        fn spawn_worker(
            &self,
            _launch: &OwnedWorkerLaunchSpec,
            _options: OwnedWorkerSpawnOptions,
        ) -> Result<Arc<dyn OwnedWorkerChild>, String> {
            Ok(Arc::new(RecordingChild::default()))
        }
        fn kill_process_group(&self, pid: i64, signal: &str) -> bool {
            self.process_group_kills
                .lock()
                .unwrap()
                .push((pid, signal.to_string()));
            self.process_group_kill_results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(false)
        }
        fn schedule_unref_timeout(&self, _ms: u64, _handler: Arc<dyn Fn() + Send + Sync>) {}
        fn exit(&self, code: i32) {
            self.exit_codes.lock().unwrap().push(code);
        }
        fn set_exit_code(&self, code: i32) {
            self.exit_codes.lock().unwrap().push(code);
        }
        fn kill_self(&self, _signal: &str) {}
        fn has_owner_channel(&self) -> bool {
            true
        }
        fn unref_owner_channel(&self) {}
        fn once_owner_disconnect(
            &self,
            _handler: Arc<dyn Fn() + Send + Sync>,
        ) -> Arc<dyn Fn() + Send + Sync> {
            Arc::new(|| {})
        }
        fn owner_channel_connected(&self) -> bool {
            true
        }
        fn disconnect_owner_channel(&self) {}
    }

    #[derive(Default)]
    struct RecordingChild {
        killed: Mutex<Vec<String>>,
        exit_code_value: Mutex<Option<i32>>,
        signal_code_value: Mutex<Option<String>>,
    }

    impl OwnedWorkerChild for RecordingChild {
        fn pid(&self) -> Option<i64> {
            Some(1)
        }
        fn stdin(&self) -> Option<Arc<dyn OwnedWorkerChildStdin>> {
            None
        }
        fn stdout(&self) -> Option<Arc<dyn OwnedWorkerChildStdout>> {
            None
        }
        fn stderr(&self) -> Option<Arc<dyn OwnedWorkerChildStderr>> {
            None
        }
        fn exit_code(&self) -> Option<i32> {
            *self.exit_code_value.lock().unwrap()
        }
        fn signal_code(&self) -> Option<String> {
            self.signal_code_value.lock().unwrap().clone()
        }
        fn kill(&self, signal: &str) {
            self.killed.lock().unwrap().push(signal.to_string());
        }
        fn wait(&self) -> BoxFuture<Result<OwnedWorkerExit, String>> {
            Box::pin(async { Ok(OwnedWorkerExit { code: Some(0), signal: None }) })
        }
        fn connected(&self) -> bool {
            false
        }
        fn disconnect(&self) {}
    }
}
