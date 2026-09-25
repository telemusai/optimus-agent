//! Port of packages/coding-agent/src/core/tools/ipython.ts

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::kernel::boot_gate::with_kernel_boot_permit;
use crate::core::kernel::shared::{
    create_kernel_startup_abort_error, ExecuteOptions, ExecuteResult, ExecuteStatus, KernelBootstrapProgressHandler,
    KernelDiffDisplay, KernelError, KernelManagerOptions, KernelSentAgentMessage, KernelStartOptions, StreamName,
    AbortSignal, KernelPythonSkill, PerformanceMetricRecorder,
};
use crate::core::kernel::state_snapshot::{
    cas_snapshot_root_in, manifest_path_in, snapshot_path_in, snapshot_state_exists_in, KernelSnapshotFormat,
    RestoreResult,
};
use crate::core::model_tool_output_policy::{
    persist_model_tool_output_artifact, resolve_model_tool_output_policy, ModelToolOutputArtifactV1,
    ModelToolOutputPolicy, ModelToolOutputScope, MODEL_TOOL_OUTPUT_MIN_BYTES, REPEATED_LARGE_TEXT_POLICY,
};
use crate::utils::mime::is_image_mime_type;

use super::tool_definition_wrapper::wrap_tool_definition;
use super::{ExtensionContext, ToolDefinition, ToolExecuteFn};

const UNSAFE_WINDOWS_CAPTURED_LAUNCHER_MESSAGE: &str =
    "A persistent PowerShell launcher must not use subprocess.run(..., capture_output=True) on Windows: \
a long-lived child can inherit the captured pipes and prevent the Python call from returning. \
Use stdout=subprocess.DEVNULL and stderr=subprocess.DEVNULL, or explicit log files, \
with a hard timeout and a separate bounded process/health check.";

fn unsafe_launcher_subprocess_pattern() -> &'static regex::Regex {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| regex::Regex::new(r"\bsubprocess\s*\.\s*run\s*\(").expect("valid subprocess pattern"))
}

fn unsafe_launcher_capture_pattern() -> &'static regex::Regex {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| regex::Regex::new(r"capture_output\s*=\s*True").expect("valid capture pattern"))
}

fn unsafe_launcher_script_pattern() -> &'static regex::Regex {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| {
        regex::Regex::new(r"(?:^|[^A-Za-z0-9_-])(?:start|serve|launch|run)\.ps1(?:[^A-Za-z0-9_-]|$)")
            .expect("valid launcher script pattern")
    })
}

/// NodeJS `process.platform` for the `platform` parameter default.
fn process_platform() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

pub fn is_unsafe_windows_captured_launcher(code: &str) -> bool {
    is_unsafe_windows_captured_launcher_on(code, process_platform())
}

pub fn is_unsafe_windows_captured_launcher_on(code: &str, platform: &str) -> bool {
    platform == "win32"
        && unsafe_launcher_subprocess_pattern().is_match(code)
        && unsafe_launcher_capture_pattern().is_match(code)
        && unsafe_launcher_script_pattern().is_match(code)
}

const RLM_BOOTSTRAP_HEADER_CODE: &str = r#"import asyncio
import os as _prime_agent_os

_prime_agent_os.environ["NO_COLOR"] = "1""#;

const RLM_BOOTSTRAP_RUNTIME_CODE: &str = r#"try:
    import rlm as _prime_agent_rlm_module
    rlm = _prime_agent_rlm_module.rlm
    bash = _prime_agent_rlm_module.bash
    import rlm.mcp as mcp
except Exception as _prime_agent_rlm_error:
    _PRIME_AGENT_RLM_IMPORT_ERROR = str(_prime_agent_rlm_error)

    class _PrimeAgentMissingRlm:
        def _raise_missing(self):
            raise RuntimeError(
                "prime-agent-runtime is not installed in this kernel. "
                "Remove ~/.prime/agent/kernel-venv so prime-agent can rebuild it, or set "
                "PRIME_AGENT_KERNEL_PYTHON to a kernel environment with prime-agent-runtime installed. "
                f"Import error: {_PRIME_AGENT_RLM_IMPORT_ERROR}"
            )

        async def run(self, prompt, **kwargs):
            self._raise_missing()

        async def find_models(self, query="", limit=8):
            self._raise_missing()

        async def create_session(self, prompt, **kwargs):
            self._raise_missing()

        async def list_subagents(self):
            self._raise_missing()

        async def delete_subagent(self, target):
            self._raise_missing()

        async def __call__(self, prompt, **kwargs):
            return await self.run(prompt, **kwargs)

    rlm = _PrimeAgentMissingRlm()

    def bash(command):
        rlm._raise_missing()"#;

/// Port of `buildRlmBootstrapCode`.
pub fn build_rlm_bootstrap_code(python_skills: &[KernelPythonSkill]) -> String {
    let base_code = format!("{RLM_BOOTSTRAP_HEADER_CODE}\n\n{RLM_BOOTSTRAP_RUNTIME_CODE}");
    let mut import_names: Vec<String> = Vec::new();
    for skill in python_skills {
        if !import_names.contains(&skill.import_name) {
            import_names.push(skill.import_name.clone());
        }
    }
    if import_names.is_empty() {
        return base_code;
    }

    let names_json = serde_json::to_string(&import_names).unwrap_or_else(|_| "[]".to_string());
    format!(
        r#"{base_code}

import importlib as _prime_agent_importlib
import inspect as _prime_agent_inspect
import sys as _prime_agent_sys
import types as _prime_agent_types

class _PrimeAgentCallableSkillModule(_prime_agent_types.ModuleType):
    async def __call__(self, *args, **kwargs):
        result = self.run(*args, **kwargs)
        if _prime_agent_inspect.isawaitable(result):
            return await result
        return result

class _PrimeAgentUnavailableSkill:
    def __init__(self, name, error):
        self.__name__ = name
        self._prime_agent_import_error = error
        self.__doc__ = f"Python skill {{name}} is unavailable: {{error}}"

    async def run(self, *args, **kwargs):
        raise RuntimeError(
            f"Python skill {{self.__name__}} is unavailable in this kernel. "
            f"Import error: {{self._prime_agent_import_error}}"
        )

    async def __call__(self, *args, **kwargs):
        return await self.run(*args, **kwargs)

    def __repr__(self):
        return f"<unavailable Python skill {{self.__name__!r}}: {{self._prime_agent_import_error}}>"

def _prime_agent_wrap_skill_module(module):
    run = getattr(module, "run", None)
    if not callable(run):
        return module
    if isinstance(module, _PrimeAgentCallableSkillModule):
        return module
    wrapped = _PrimeAgentCallableSkillModule(module.__name__)
    wrapped.__dict__.update(module.__dict__)
    try:
        wrapped.__signature__ = _prime_agent_inspect.signature(run)
    except Exception:
        pass
    doc = getattr(run, "__doc__", None)
    if doc:
        wrapped.__doc__ = doc
    _prime_agent_sys.modules[module.__name__] = wrapped
    return wrapped

_PRIME_AGENT_SKILL_IMPORT_ERRORS = {{}}

for _prime_agent_skill_name in {names_json}:
    try:
        globals()[_prime_agent_skill_name] = _prime_agent_wrap_skill_module(
            _prime_agent_importlib.import_module(_prime_agent_skill_name)
        )
    except Exception as _prime_agent_skill_error:
        _PRIME_AGENT_SKILL_IMPORT_ERRORS[_prime_agent_skill_name] = str(_prime_agent_skill_error)
        globals()[_prime_agent_skill_name] = _PrimeAgentUnavailableSkill(
            _prime_agent_skill_name,
            str(_prime_agent_skill_error),
        )"#
    )
}

pub const IPYTHON_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "description": "Python code to execute in the persistent Python REPL. Use the target project's own environment for project imports, tests, scripts, CLIs, and dependency checks instead of direct kernel imports."
    }
  },
  "required": ["code"]
}"#;

pub const BUSY_KERNEL_WAIT_CHOICE: &str = "Wait and preserve state";
pub const BUSY_KERNEL_KILL_CHOICE: &str = "Kill kernel and restart";
pub const BUSY_KERNEL_PROMPT: &str = "Python kernel is still busy\n\
Prime requested an interrupt for the previous Python operation, but the kernel has not yet confirmed completion.\n\
Wait to preserve state, or explicitly kill and restart the kernel. Killing loses in-memory variables, imports, and running tasks.";
pub const KERNEL_RESTART_NOTICE: &str = "<ipython_kernel_reset>\n\
The Python kernel was restarted after a previous interrupted cell kept running. Variables, imports, async tasks, and open resources from before the restart are no longer available; recreate them before using them.\n\
</ipython_kernel_reset>";

fn create_abort_error() -> KernelError {
    KernelError::new("Python execution aborted")
}

/// Port of `raceWithAbort`.
async fn race_with_abort<F, T>(
    promise: F,
    signal: Option<AbortSignal>,
    // Held across the `select!` await, so the callback must be `Send + Sync` for
    // the enclosing spawned startup task to stay `Send`.
    on_abort: Option<&(dyn Fn() + Send + Sync)>,
) -> Result<T, KernelError>
where
    F: std::future::Future<Output = Result<T, KernelError>>,
{
    let Some(signal) = signal else {
        return promise.await;
    };
    if signal.is_aborted() {
        if let Some(on_abort) = on_abort {
            on_abort();
        }
        return Err(create_abort_error());
    }
    tokio::select! {
        biased;
        value = promise => value,
        _ = signal.wait() => {
            if let Some(on_abort) = on_abort {
                on_abort();
            }
            Err(create_abort_error())
        }
    }
}

/// Port of `createLinkedAbortSignal`.
pub struct LinkedAbortSignal {
    pub signal: AbortSignal,
    listeners: Vec<crate::core::kernel::shared::AbortListener>,
}

impl LinkedAbortSignal {
    pub fn new(sources: Vec<Option<AbortSignal>>) -> Self {
        let controller = AbortSignal::new();
        let mut listeners = Vec::new();
        for source in sources {
            let Some(source) = source else {
                continue;
            };
            if source.is_aborted() {
                controller.abort(None);
                continue;
            }
            let target = controller.clone();
            listeners.push(source.add_listener(move || target.abort(None)));
        }
        Self {
            signal: controller,
            listeners,
        }
    }

    pub fn cleanup(&self) {
        for listener in &self.listeners {
            listener.remove();
        }
    }
}

fn set_working_message(ctx: Option<&ExtensionContext>, message: Option<&str>) {
    // Stale UI context; cosmetic only.
    if let Some(ctx) = ctx {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.ui.set_working_message(message);
        }));
    }
}

/// TypeScript `type IpythonToolInput = Static<typeof ipythonSchema>`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IpythonToolInput {
    pub code: String,
}

/// TypeScript `interface IpythonToolDetails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IpythonToolDetails {
    #[serde(rename = "durationMs", skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(rename = "errorEname", skip_serializing_if = "Option::is_none")]
    pub error_ename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Output that arrived without this cell's id (threads, other cells' leftovers), shown separately from stdout.
    #[serde(rename = "backgroundOutput", skip_serializing_if = "Option::is_none")]
    pub background_output: Option<String>,
    /// Diffs streamed from file edits, rendered by the cell view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffs: Option<Vec<KernelDiffDisplay>>,
    /// Media attachments loaded into context (e.g. by the attach-image skill).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<crate::core::kernel::shared::KernelAttachment>>,
    /// Agent messages sent from this cell.
    #[serde(rename = "sentAgentMessages", skip_serializing_if = "Option::is_none")]
    pub sent_agent_messages: Option<Vec<KernelSentAgentMessage>>,
    /// Durable, session-scoped source for an opt-in model-facing repeated-output reference.
    #[serde(rename = "modelOutputArtifact", skip_serializing_if = "Option::is_none")]
    pub model_output_artifact: Option<ModelToolOutputArtifactV1>,
    /// True when this result came after killing and restarting a busy kernel.
    #[serde(rename = "kernelRestarted", skip_serializing_if = "Option::is_none")]
    pub kernel_restarted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ExecErrorShape>,
    #[serde(rename = "executionReports", default, skip_serializing_if = "Option::is_none")]
    pub execution_reports: Option<Vec<crate::core::kernel::shared::ScriptExecutionReport>>,
}

/// `{ ename, evalue, traceback }` as serialized in the tool details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecErrorShape {
    pub ename: String,
    pub evalue: String,
    pub traceback: Vec<String>,
}

/// TypeScript `interface IpythonToolOptions`.
#[derive(Clone, Default)]
pub struct IpythonToolOptions {
    /// Python override. Must have prime-agent-runtime installed.
    pub python: Option<String>,
    pub env: Option<Vec<(String, String)>>,
    /// Command prefix prepended to every bash() command.
    pub command_prefix: Option<String>,
    /// Shell used by bash().
    pub shell_path: Option<String>,
    pub session_id: Option<String>,
    /// Typed host request handlers for the kernel<->host bridge (rlm.run, goal.*, ...).
    pub host_handlers: Option<crate::core::kernel::shared::HostRequestHandlers>,
    pub python_skills: Option<Vec<KernelPythonSkill>>,
    /// Per-session artifact dir where the kernel namespace snapshot is stored. Omit to disable snapshots.
    pub snapshot_dir: Option<String>,
    /// Explicit snapshot writer opt-in. Omitted preserves legacy/default continuation behavior.
    pub snapshot_format: Option<KernelSnapshotFormat>,
    /// Content-free snapshot timings routed through the owning session's live monitor.
    pub performance_metrics: Option<Arc<dyn PerformanceMetricRecorder>>,
    /// Opt-in model-facing output policy. Execution itself is never cached.
    pub model_tool_output_policy: Option<ModelToolOutputPolicy>,
    /// Resolves before this kernel starts - e.g. the previous provisioner's dispose.
    pub ready_gate: Option<Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>>,
    /// Fires once per kernel start when a previous session's namespace was revived.
    pub on_restore: Option<Arc<dyn Fn(RestoreResult) + Send + Sync>>,
    pub on_background_work_settled: Option<Arc<dyn Fn() + Send + Sync>>,
    pub on_late_sent_agent_message:
        Option<Arc<dyn Fn(String, KernelSentAgentMessage) + Send + Sync>>,
    /// Shared provisioner owning the kernel lifecycle. When provided, the remaining options are ignored.
    pub provisioner: Option<Arc<IpythonKernelProvisioner>>,
}

/// The kernel client surface the ipython tool drives.
///
/// The native adapter forwards these operations to `ReplKernelManager`; the
/// provisioner only depends on the operations below.
pub trait KernelClient: Send + Sync {
    fn is_running(&self) -> bool;
    /// A terminally dead kernel: `startedManager?.isDefunct`.
    fn is_defunct(&self) -> bool;
    fn start(&self, options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>>;
    fn execute(
        &self,
        code: &str,
        signal: Option<AbortSignal>,
        on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
    ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>>;
    /// [`KernelClient::execute`] plus the late-sent-agent-message handler.
    ///
    /// TypeScript passes one options object to `KernelClient.execute`
    /// (`packages/coding-agent/src/core/tools/ipython.ts:644-650`), which is how
    /// `onLateSentAgentMessage` reaches the kernel session; the manager then
    /// re-registers it under the request id once the execution settles
    /// (`packages/coding-agent/src/core/kernel/repl-manager.ts:1153-1155`).
    /// The Rust trait splits that object so callers that never pass a handler keep
    /// the three argument form (`packages/coding-agent/src/core/tools/acp-mcp.ts:47`
    /// passes only `signal`).
    ///
    /// The default drops the handler and defers to [`KernelClient::execute`]. An
    /// implementation that owns a real manager must override this: the native
    /// adapter does, so `ExecuteOptions.on_late_sent_agent_message` stays set and
    /// the kernel session's late-message dispatch is genuinely reached.
    fn execute_with_late_sent_agent_message(
        &self,
        code: &str,
        signal: Option<AbortSignal>,
        on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        on_late_sent_agent_message: Option<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>>,
    ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
        let _ = on_late_sent_agent_message;
        self.execute(code, signal, on_stream)
    }
    fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>>;
    fn shutdown(
        &self,
        snapshot: bool,
        drain_host_requests: bool,
    ) -> BoxFuture<'static, Result<(), KernelError>>;
    fn shutdown_and_settle(&self, _owner_session_id: &str, _timeout_ms: u64)
        -> BoxFuture<'static, Result<crate::core::kernel::shared::KernelSettlement, KernelError>> {
        Box::pin(async { Ok(crate::core::kernel::shared::KernelSettlement::unsupported("unowned kernel adapter")) })
    }
    fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>>;
    fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>>;
    fn list_namespace_names(
        &self,
        signal: Option<AbortSignal>,
    ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>>;
}

/// `{ pruned?: string[] }` from `pruneOversizedVariables`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PruneResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruned: Option<Vec<String>>,
}

/// Factory building the kernel manager for one startup.
pub type KernelClientFactory = Arc<
    dyn Fn(KernelManagerOptions) -> Arc<dyn KernelClient> + Send + Sync,
>;

/// Owns the lazy create+start+runtime-bootstrap of one session's Python kernel.
///
/// Concurrent ensure() calls await the same in-flight startup, a failed startup
/// clears the memo so the next call retries fresh, and progress listeners can
/// attach mid-flight (a tool call racing a background prewarm()).
pub struct IpythonKernelProvisioner {
    cwd: String,
    options: Option<IpythonToolOptions>,
    factory: KernelClientFactory,
    manager_promise: Mutex<Option<Arc<StartupHandle>>>,
    started_manager: Mutex<Option<Arc<dyn KernelClient>>>,
    startup_listeners: Mutex<Vec<KernelBootstrapProgressHandler>>,
    last_startup_message: Mutex<Option<String>>,
    last_restore: Mutex<Option<RestoreResult>>,
    dispose_controller: AbortSignal,
    /// Snapshot policy of the dispose that aborted a startup, honored by startKernel's failure teardown.
    dispose_snapshot: Mutex<bool>,
    ownership: Mutex<ProvisionerOwnership>,
    owned_tasks: pi_agent_core::execution_scope::ExecutionScope,
}
#[derive(Default)]
struct ProvisionerOwnership {
    fenced: bool,
    clients: Vec<Arc<dyn KernelClient>>,
    startups: Vec<Arc<StartupHandle>>,
}

struct StartupHandle {
    result: tokio::sync::Mutex<Option<Result<Arc<dyn KernelClient>, KernelError>>>,
    notify: tokio::sync::Notify,
}

impl StartupHandle {
    fn new() -> Self {
        Self {
            result: tokio::sync::Mutex::new(None),
            notify: tokio::sync::Notify::new(),
        }
    }

    async fn settle(&self, value: Result<Arc<dyn KernelClient>, KernelError>) {
        *self.result.lock().await = Some(value);
        self.notify.notify_waiters();
    }

    async fn wait(&self) -> Result<Arc<dyn KernelClient>, KernelError> {
        loop {
            if let Some(value) = self.result.lock().await.clone() {
                return value;
            }
            let notified = self.notify.notified();
            if let Some(value) = self.result.lock().await.clone() {
                return value;
            }
            notified.await;
        }
    }
}

impl IpythonKernelProvisioner {
    /// TypeScript `new IpythonKernelProvisioner(cwd, options)`.
    pub fn new(
        cwd: &str,
        options: Option<IpythonToolOptions>,
        factory: KernelClientFactory,
    ) -> Arc<Self> {
        Arc::new(Self {
            cwd: cwd.to_string(),
            options,
            factory,
            manager_promise: Mutex::new(None),
            started_manager: Mutex::new(None),
            startup_listeners: Mutex::new(Vec::new()),
            last_startup_message: Mutex::new(None),
            last_restore: Mutex::new(None),
            dispose_controller: AbortSignal::new(),
            dispose_snapshot: Mutex::new(true),
            ownership: Mutex::new(ProvisionerOwnership::default()),
            owned_tasks: Default::default(),
        })
    }

    /// Test-only view of the last recorded restore result (the revive path).
    #[doc(hidden)]
    pub fn last_restore_for_tests(&self) -> Option<crate::core::kernel::state_snapshot::RestoreResult> {
        self.last_restore.lock().expect("last restore lock").clone()
    }

    /// The kernel manager, once a startup has completed successfully.
    pub fn manager(&self) -> Option<Arc<dyn KernelClient>> {
        self.started_manager.lock().expect("started manager lock").clone()
    }

    /// Result of reviving a prior session's namespace on the last kernel start, if any.
    pub fn last_restore(&self) -> Option<RestoreResult> {
        self.last_restore.lock().expect("last restore lock").clone()
    }

    /// Session scope used only for opt-in durable model-output artifacts.
    pub fn model_tool_output_scope(&self) -> Option<ModelToolOutputScope> {
        let session_id = self.options.as_ref().and_then(|options| options.session_id.clone());
        let session_artifact_dir = self.options.as_ref().and_then(|options| options.snapshot_dir.clone());
        match (session_id, session_artifact_dir) {
            (Some(session_id), Some(session_artifact_dir)) => Some(ModelToolOutputScope {
                session_id,
                session_artifact_dir,
            }),
            _ => None,
        }
    }

    /// Resolved once from the owner provisioner so wrapper options cannot silently disable it.
    pub fn model_tool_output_policy(&self) -> ModelToolOutputPolicy {
        resolve_model_tool_output_policy(
            self.options
                .as_ref()
                .and_then(|options| options.model_tool_output_policy),
        )
    }

    /// Start the kernel in the background. Failures are swallowed here and surface on the next ensure().
    pub fn prewarm(self: &Arc<Self>) {
        let provisioner = self.clone();
        let _ = self.owned_tasks.spawn(async move {
            let _ = provisioner.ensure(None, None).await;
        }, true, true);
    }

    /// Whether a kernel has finished starting and is currently running.
    pub fn has_running_kernel(&self) -> bool {
        self.manager().map(|manager| manager.is_running()).unwrap_or(false)
    }

    /// Remove live variables above the snapshot's per-variable size limit.
    pub async fn prune_oversized_variables(&self) -> Option<Vec<String>> {
        let manager = self.current_manager().await?;
        let result = manager.prune_oversized_variables().await.ok()?;
        result.map(|result| result.pruned.unwrap_or_default())
    }

    /// Live user-defined names in the kernel namespace, or null if listing failed / no kernel.
    pub async fn list_namespace_names(&self, signal: Option<AbortSignal>) -> Option<Vec<String>> {
        let manager = self.current_manager().await?;
        manager.list_namespace_names(signal).await.ok().flatten()
    }

    async fn current_manager(&self) -> Option<Arc<dyn KernelClient>> {
        if let Some(manager) = self.manager() {
            return Some(manager);
        }
        let handle = self.manager_promise.lock().expect("manager promise lock").clone();
        match handle {
            Some(handle) => handle.wait().await.ok(),
            None => None,
        }
    }

    /// Retained stop disposes execution, not history or snapshots. One deadline
    /// covers startup plus every manager generation, including failed startup.
    pub async fn shutdown_and_settle(&self, owner: &str, timeout_ms: u64)
        -> Result<crate::core::kernel::shared::KernelSettlement, KernelError> {
        use crate::core::kernel::shared::KernelSettlement;
        if owner.is_empty()
            || self.options.as_ref().and_then(|options| options.session_id.as_deref()) != Some(owner)
        {
            return Err(KernelError::new("Kernel owner mismatch"));
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms.min(10000));
        let (startups, clients) = {
            let mut ownership = self.ownership.lock().unwrap();
            ownership.fenced = true;
            (ownership.startups.clone(), ownership.clients.clone())
        };
        *self.dispose_snapshot.lock().unwrap() = false;
        self.dispose_controller.abort(None);
        self.owned_tasks.request_cancel();
        let mut result = KernelSettlement {
            supported: cfg!(windows), settled: cfg!(windows), kernel_exited: true,
            descendants_exited: true, local_tasks_settled: true, errors: Vec::new(),
            ownership_scope: "native-kernel-job-members".to_string(),
        };
        if self.options.as_ref().and_then(|options| options.ready_gate.as_ref()).is_some() {
            result.supported = false;
            result.errors.push("unproved predecessor provisioner".to_string());
        }
        // Admit every known generation before waiting for any one result. A
        // failed or slow adapter cannot skip another client's terminal stop.
        let settlements = clients.into_iter().enumerate().map(|(index, client)| async move {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now()).as_millis() as u64;
            let operation = std::panic::AssertUnwindSafe(async move {
                client.shutdown_and_settle(owner, remaining).await
            }).catch_unwind();
            (index, tokio::time::timeout_at(deadline, operation).await)
        });
        for (index, outcome) in futures::future::join_all(settlements).await {
            match outcome {
                Ok(Ok(Ok(receipt))) => {
                    result.supported &= receipt.supported;
                    result.settled &= receipt.settled;
                    result.kernel_exited &= receipt.kernel_exited;
                    result.descendants_exited &= receipt.descendants_exited;
                    result.local_tasks_settled &= receipt.local_tasks_settled;
                    result.errors.extend(receipt.errors);
                }
                failure => {
                    result.supported = false;
                    result.settled = false;
                    result.kernel_exited = false;
                    result.descendants_exited = false;
                    result.local_tasks_settled = false;
                    let reason = match failure {
                        Ok(Ok(Err(error))) => error.to_string(),
                        Ok(Err(_)) => "adapter panicked".to_string(),
                        Err(_) => "deadline expired".to_string(),
                        Ok(Ok(Ok(_))) => unreachable!(),
                    };
                    result.errors.push(format!("Kernel client {index} settlement failed: {reason}"));
                }
            }
        }
        for startup in startups {
            // Startup failure is not exit proof; every created client was included above.
            if tokio::time::timeout_at(deadline, startup.wait()).await.is_err() {
                result.local_tasks_settled = false;
                result.errors.push("Kernel startup settlement timed out".to_string());
            }
        }
        let local = self.owned_tasks.settle(deadline.saturating_duration_since(tokio::time::Instant::now())).await;
        result.supported &= local.supported;
        result.local_tasks_settled &= local.tools_settled && local.model_settled && !local.failed;
        if !local.supported {
            result.errors.push("Kernel provisioner has unproved local work ownership".to_string());
        }
        if !local.tools_settled || !local.model_settled || local.failed {
            result.errors.push("Kernel provisioner owned tasks or streams failed or did not settle".to_string());
        }
        result.settled &= result.supported && result.kernel_exited && result.descendants_exited
            && result.local_tasks_settled && result.errors.is_empty();
        Ok(result)
    }

    /// Dispose the kernel owned by this provisioner, including one still starting up.
    pub async fn dispose(&self, snapshot: Option<bool>) {
        *self.dispose_snapshot.lock().expect("dispose snapshot lock") = snapshot.unwrap_or(true);
        // Drops a still-queued boot out of the semaphore and short-circuits an
        // in-flight startKernel before it spawns, so a disposed session's boot
        // doesn't waste a slot during a fan-out.
        self.dispose_controller.abort(None);
        // A replacement may never start, but still owns the old kernel's shutdown.
        if let Some(ready_gate) = self.options.as_ref().and_then(|options| options.ready_gate.clone()) {
            ready_gate().await;
        }
        let pending = self.manager_promise.lock().expect("manager promise lock").take();
        *self.started_manager.lock().expect("started manager lock") = None;
        let Some(pending) = pending else {
            return;
        };
        // a failed startup already cleaned up after itself
        if let Ok(manager) = pending.wait().await {
            let snapshot = *self.dispose_snapshot.lock().expect("dispose snapshot lock");
            let _ = manager.shutdown(snapshot, true).await;
        }
    }

    /// Begin disposal immediately, while allowing every replacement to await the same flush.
    pub(crate) fn replacement_ready_gate(
        self: &Arc<Self>,
    ) -> Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync> {
        let previous = self.clone();
        let disposal = async move { previous.dispose(None).await }.boxed().shared();
        tokio::spawn(disposal.clone());
        Arc::new(move || disposal.clone().boxed())
    }

    pub async fn kill(&self) {
        let pending = self.manager_promise.lock().expect("manager promise lock").take();
        *self.started_manager.lock().expect("started manager lock") = None;
        let Some(pending) = pending else {
            return;
        };
        // a failed startup already cleaned up after itself
        if let Ok(manager) = pending.wait().await {
            let _ = manager.kill().await;
        }
    }

    pub async fn ensure(
        self: &Arc<Self>,
        on_progress: Option<KernelBootstrapProgressHandler>,
        signal: Option<AbortSignal>,
    ) -> Result<Arc<dyn KernelClient>, KernelError> {
        if signal.as_ref().map(|signal| signal.is_aborted()).unwrap_or(false) {
            return Err(create_abort_error());
        }
        let (handle, launch_startup) = {
            let mut ownership = self.ownership.lock().unwrap();
            if ownership.fenced {
                return Err(KernelError::new("Kernel provisioner retained-stop fence"));
            }
            // Only terminally dead kernels drop the memo; repair still reuses it.
            let started_defunct = self.manager().map(|manager| manager.is_defunct()).unwrap_or(false);
            let mut memo = self.manager_promise.lock().expect("manager promise lock");
            if started_defunct {
                *memo = None;
                *self.started_manager.lock().expect("started manager lock") = None;
            }
            match memo.as_ref() {
                Some(handle) => (handle.clone(), false),
                None => {
                    let handle = Arc::new(StartupHandle::new());
                    *memo = Some(handle.clone());
                    ownership.startups.push(handle.clone());
                    (handle, true)
                }
            }
        };

        let mut cleanup_progress_listener: Option<crate::core::kernel::shared::AbortListener> = None;
        if let Some(on_progress) = on_progress.as_ref() {
            if self.manager().is_none() {
                self.startup_listeners.lock().expect("listeners lock").push(on_progress.clone());
                if let Some(signal) = signal.clone() {
                    let provisioner = self.clone();
                    let listener = signal.add_listener(move || {
                        provisioner.startup_listeners.lock().expect("listeners lock").clear();
                    });
                    cleanup_progress_listener = Some(listener);
                }
                // Joining an in-flight startup: replay the current stage.
                let has_promise = self.manager_promise.lock().expect("manager promise lock").is_some();
                let last_message = self.last_startup_message.lock().expect("last message lock").clone();
                if has_promise {
                    if let Some(last_message) = last_message {
                        on_progress(&last_message);
                    }
                }
            }
        }

        if launch_startup {
            let provisioner = self.clone();
            let startup_handle = handle.clone();
            let startup_signal = signal.clone();
            let startup_task = self.owned_tasks.spawn(async move {
                let outcome = provisioner.start_kernel(startup_signal).await;
                let failed = outcome.is_err();
                if !failed {
                    if let Ok(manager) = outcome.as_ref() {
                        let is_current = provisioner
                            .manager_promise
                            .lock()
                            .expect("manager promise lock")
                            .as_ref()
                            .map(|current| Arc::ptr_eq(current, &startup_handle))
                            .unwrap_or(false);
                        if is_current {
                            *provisioner.started_manager.lock().expect("started manager lock") =
                                Some(manager.clone());
                        }
                    }
                } else {
                    // Clear the memo so the next ensure() retries instead of
                    // rethrowing a cached rejection forever.
                    let is_current = provisioner
                        .manager_promise
                        .lock()
                        .expect("manager promise lock")
                        .as_ref()
                        .map(|current| Arc::ptr_eq(current, &startup_handle))
                        .unwrap_or(false);
                    if is_current {
                        *provisioner.manager_promise.lock().expect("manager promise lock") = None;
                    }
                }
                provisioner.settle_startup();
                startup_handle.settle(outcome).await;
            }, false, true);
            if let Err(error) = startup_task {
                handle.settle(Err(KernelError::new(error))).await;
                self.settle_startup();
                if let Some(listener) = cleanup_progress_listener { listener.remove(); }
                return Err(KernelError::new(error));
            }
        }

        let result = race_with_abort(
            handle.wait().map(|value| value),
            signal,
            None,
        )
        .await;
        if let Some(listener) = cleanup_progress_listener {
            listener.remove();
        }
        if self.ownership.lock().unwrap().fenced {
            return Err(KernelError::new("Kernel provisioner retained-stop fence"));
        }
        result
    }

    fn settle_startup(&self) {
        self.startup_listeners.lock().expect("listeners lock").clear();
        *self.last_startup_message.lock().expect("last message lock") = None;
    }

    fn emit_startup_progress(&self, message: &str) {
        *self.last_startup_message.lock().expect("last message lock") = Some(message.to_string());
        let listeners = self.startup_listeners.lock().expect("listeners lock").clone();
        for listener in listeners {
            listener(message);
        }
    }

    async fn start_kernel(self: &Arc<Self>, signal: Option<AbortSignal>) -> Result<Arc<dyn KernelClient>, KernelError> {
        let startup_abort = LinkedAbortSignal::new(vec![Some(self.dispose_controller.clone()), signal]);
        let startup_signal = startup_abort.signal.clone();
        let outcome = self.start_kernel_inner(&startup_signal).await;
        startup_abort.cleanup();
        outcome
    }

    async fn start_kernel_inner(
        self: &Arc<Self>,
        startup_signal: &AbortSignal,
    ) -> Result<Arc<dyn KernelClient>, KernelError> {
        // Wait for a previous provisioner (e.g. on /reload) to finish disposing -
        // and flushing its final snapshot - before we read that snapshot back.
        if let Some(ready_gate) = self.options.as_ref().and_then(|options| options.ready_gate.clone()) {
            let gate = ready_gate();
            let _ = race_with_abort(gate.map(Ok), Some(startup_signal.clone()), None).await;
        }

        let snapshot_dir = self.options.as_ref().and_then(|options| options.snapshot_dir.clone());
        // Always inject an absolute trusted shell (None only on win32 without
        // bash, where the runtime's teaching error fires instead).
        let shell_path = resolve_kernel_bash_shell(
            self.options.as_ref().and_then(|options| options.shell_path.as_deref()),
        );
        let command_prefix = self.options.as_ref().and_then(|options| options.command_prefix.clone());
        let bootstrap_code = build_rlm_bootstrap_code(
            self.options
                .as_ref()
                .and_then(|options| options.python_skills.as_deref())
                .unwrap_or(&[]),
        );

        let mut env = kernel_shell_env(
            crate::utils::shell::get_shell_env(),
            self.options.as_ref().and_then(|options| options.env.as_deref()).unwrap_or(&[]),
            cfg!(windows),
        );
        if let Some(shell_path) = shell_path.as_ref() {
            env.insert("PRIME_AGENT_BASH_SHELL".to_string(), shell_path.clone());
        }
        if let Some(command_prefix) = command_prefix.as_ref() {
            env.insert("PRIME_AGENT_BASH_COMMAND_PREFIX".to_string(), command_prefix.clone());
        }

        let mut host_handlers = crate::core::memory::service::create_memory_host_handlers(
            self.cwd.clone(),
            self.options
                .as_ref()
                .and_then(|options| options.env.as_ref())
                .and_then(|env| {
                    env.iter()
                        .find(|(key, _)| key == "PRIME_AGENT_CODING_AGENT_DIR")
                        .map(|(_, value)| value.clone())
                }),
            snapshot_dir.clone(),
            self.options.as_ref()
                .and_then(|options| options.host_handlers.as_ref())
                .and_then(|handlers| handlers.get("model.info").cloned()),
        );
        if let Some(extra) = self.options.as_ref().and_then(|options| options.host_handlers.clone()) {
            for (key, value) in extra {
                host_handlers.insert(key, value);
            }
        }

        let manager_options = KernelManagerOptions {
            python: self.options.as_ref().and_then(|options| options.python.clone()),
            cwd: Some(self.cwd.clone()),
            env: Some(env),
            session_id: self.options.as_ref().and_then(|options| options.session_id.clone()),
            host_handlers: Some(host_handlers),
            python_skills: self.options.as_ref().and_then(|options| options.python_skills.clone()),
            performance_metrics: self.options.as_ref().and_then(|options| options.performance_metrics.clone()),
            // Only persistent sessions (which have an artifact dir) get a revivable snapshot.
            snapshot: snapshot_dir.as_ref().map(|snapshot_dir| crate::core::kernel::shared::KernelSnapshotConfig {
                path: snapshot_path_in(snapshot_dir),
                manifest_path: manifest_path_in(snapshot_dir),
                cas_root_path: Some(cas_snapshot_root_in(snapshot_dir)),
                format: self.options.as_ref().and_then(|options| options.snapshot_format),
                max_bytes: None,
                max_variable_bytes: None,
                debounce_ms: None,
            }),
            on_background_work_settled: self.options.as_ref().and_then(|options| options.on_background_work_settled.clone()),
            bootstrap_code: Some(bootstrap_code.clone()),
            stderr_log_path: snapshot_dir
                .as_ref()
                .map(|snapshot_dir| format!("{}/kernel-stderr.log", snapshot_dir.trim_end_matches(['/', '\\']))),
        };

        let manager = {
            let mut ownership = self.ownership.lock().unwrap();
            if ownership.fenced { return Err(KernelError::new("Kernel provisioner retained-stop fence")); }
            let manager = (self.factory)(manager_options);
            ownership.clients.push(manager.clone());
            manager
        };
        let mut pending_restore: Option<RestoreResult> = None;
        let startup_result: Result<(), KernelError> = async {
            // Emitted synchronously (before the permit await) so a listener
            // attaching mid-flight can replay the current stage.
            self.emit_startup_progress("Starting Python kernel...");
            // Only the process spawn + port resolve contends for OS resources under a
            // fan-out, and it is bounded by start()'s own timeouts.
            let manager_for_start = manager.clone();
            let progress_provisioner = self.clone();
            with_kernel_boot_permit(
                move || {
                    // Disposed while queued for the permit - don't spawn a kernel nobody wants.
                    if startup_signal.is_aborted() {
                        return futures::future::ready(Err(KernelError::new(
                            "Kernel provisioner disposed before start",
                        )))
                        .boxed();
                    }
                    let progress: KernelBootstrapProgressHandler = Arc::new(move |message: &str| {
                        progress_provisioner.emit_startup_progress(message);
                    });
                    manager_for_start.start(KernelStartOptions {
                        on_bootstrap_progress: Some(progress),
                        signal: Some(startup_signal.clone()),
                    })
                },
                Some(startup_signal.clone()),
            )
            .await?;

            // Revive a prior session's namespace before the bootstrap, so the bootstrap
            // then overwrites live handles (rlm, skills) on top of anything restored.
            if let Some(snapshot_dir) = snapshot_dir.as_ref() {
                let snapshot_existed = snapshot_state_exists_in(snapshot_dir);
                self.emit_startup_progress("Restoring Python state...");
                let restore = race_with_abort(manager.restore_state().map(|value| value), Some(startup_signal.clone()), None)
                    .await?;
                if snapshot_existed {
                    pending_restore = Some(restore.unwrap_or_else(|| RestoreResult {
                        restored: Vec::new(),
                        failed: Vec::new(),
                        format: None,
                        generation: None,
                        rolled_back: None,
                        unsaved_work_possible: None,
                        legacy_recovery: None,
                        path: snapshot_path_in(snapshot_dir),
                    }));
                }
            }
            self.emit_startup_progress("Preparing Python runtime...");
            let bootstrap = manager
                .execute(&bootstrap_code, Some(startup_signal.clone()), None)
                .await?;
            if bootstrap.status != ExecuteStatus::Ok {
                let details = [bootstrap.stderr, bootstrap
                    .error
                    .as_ref()
                    .map(|error| error.traceback.join("\n"))
                    .unwrap_or_default()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<String>>()
                .join("\n");
                return Err(KernelError::new(format!(
                    "Failed to initialize rlm runtime in the Python kernel:\n{details}"
                )));
            }
            Ok(())
        }
        .await;

        if let Err(error) = startup_result {
            // Never leak the kernel process if startup fails after spawn - and never
            // surface the failure before the teardown (final snapshot flush included)
            // finished.
            let snapshot = *self.dispose_snapshot.lock().expect("dispose snapshot lock");
            let _ = manager.shutdown(snapshot, true).await;
            return Err(error);
        }

        // Only tell the model what was revived once the kernel is actually usable.
        if let Some(pending_restore) = pending_restore {
            *self.last_restore.lock().expect("last restore lock") = Some(pending_restore.clone());
            if let Some(on_restore) = self.options.as_ref().and_then(|options| options.on_restore.clone()) {
                on_restore(pending_restore);
            }
        }
        Ok(manager)
    }
}

/// Pass only the local shell PATH default; explicit kernel overrides remain authoritative.
fn kernel_shell_env(
    shell_env: Vec<(String, String)>,
    overrides: &[(String, String)],
    windows: bool,
) -> std::collections::HashMap<String, String> {
    let is_path = |key: &str| if windows { key.eq_ignore_ascii_case("PATH") } else { key == "PATH" };
    let mut env = std::collections::HashMap::new();
    for (key, value) in shell_env.into_iter().filter(|(key, _)| is_path(key)) {
        env.insert(if windows { "PATH".to_string() } else { key }, value);
    }
    for (key, value) in overrides {
        env.insert(if is_path(key) { "PATH".to_string() } else { key.clone() }, value.clone());
    }
    env
}

/// Port of `utils/shell.ts resolveKernelBashShell`.
///
/// Absolute default shell for the kernel's bash(): explicit shellPath wins; POSIX
/// uses /bin/bash else /bin/sh (absolute, never PATH - the kernel inherits a
/// user-influenced PATH); win32 uses only the canonical Git Bash install paths.
/// `None` = no shell found: kernel startup must not fail, bash() raises its
/// teaching error.
fn resolve_kernel_bash_shell(custom_shell_path: Option<&str>) -> Option<String> {
    if let Some(explicit) = custom_shell_path.map(str::trim).filter(|path| !path.is_empty()) {
        return Some(explicit.to_string());
    }
    if !cfg!(windows) {
        return Some(if std::path::Path::new("/bin/bash").exists() {
            "/bin/bash".to_string()
        } else {
            "/bin/sh".to_string()
        });
    }
    // Hardcoded literals: ProgramFiles env vars are ambient attacker-influenceable
    // input, the same trust-laundering class as PATH.
    for path in [
        "C:\\Program Files\\Git\\bin\\bash.exe",
        "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
    ] {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

async fn choose_busy_kernel_action(ctx: Option<&ExtensionContext>, signal: Option<AbortSignal>) -> &'static str {
    let Some(ctx) = ctx else {
        return "cancel";
    };
    if !ctx.has_ui {
        return "cancel";
    }
    // TS passes the tool signal into `ui.select` (ipython.ts:616-618): abort
    // closes the dialog, the selection resolves to cancel, and the loop keeps
    // the original busy error. A stale selection never resumes the loop.
    let open_dialog = || {
        ctx.ui.select(
            BUSY_KERNEL_PROMPT.to_string(),
            vec![
                BUSY_KERNEL_WAIT_CHOICE.to_string(),
                BUSY_KERNEL_KILL_CHOICE.to_string(),
            ],
            super::ExtensionUiDialogOptions {
                signal: None,
                timeout: None,
            },
        )
    };
    let choice = match signal.as_ref() {
        Some(signal) if signal.is_aborted() => None,
        Some(signal) => {
            tokio::select! {
                choice = open_dialog() => choice,
                _ = signal.wait() => None,
            }
        }
        None => open_dialog().await,
    };
    match choice.as_deref() {
        Some(choice) if choice == BUSY_KERNEL_WAIT_CHOICE => "wait",
        Some(choice) if choice == BUSY_KERNEL_KILL_CHOICE => "kill",
        _ => "cancel",
    }
}

pub struct BusyKernelExecution {
    pub result: ExecuteResult,
    pub kernel_restarted: bool,
}

/// Port of `executeWithBusyKernelChoice`.
#[allow(clippy::too_many_arguments)]
async fn execute_with_busy_kernel_choice(
    provisioner: Arc<IpythonKernelProvisioner>,
    report_startup_progress: KernelBootstrapProgressHandler,
    tool_call_id: &str,
    code: &str,
    signal: Option<AbortSignal>,
    on_stream: Arc<dyn Fn(&str, StreamName) + Send + Sync>,
    on_working_message: Arc<dyn Fn(Option<&str>) + Send + Sync>,
    on_late_sent_agent_message: Option<Arc<dyn Fn(String, KernelSentAgentMessage) + Send + Sync>>,
    ctx: Option<&ExtensionContext>,
) -> Result<BusyKernelExecution, KernelError> {
    let mut kernel_restarted = false;
    loop {
        let manager = provisioner
            .ensure(Some(report_startup_progress.clone()), signal.clone())
            .await?;
        // `onLateSentAgentMessage: onLateSentAgentMessage ? (message) =>
        // onLateSentAgentMessage(toolCallId, message) : undefined`
        // (packages/coding-agent/src/core/tools/ipython.ts:647-649). The tool-call id is
        // bound here so the kernel callback only carries the message, and the bound
        // handler is handed to the kernel session for every attempt of the loop -
        // including a retry after a kill/restart, which builds a fresh session.
        let late_handler: Option<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>> =
            on_late_sent_agent_message.as_ref().map(|handler| {
                let handler = handler.clone();
                let tool_call_id = tool_call_id.to_string();
                Arc::new(move |message: KernelSentAgentMessage| {
                    handler(tool_call_id.clone(), message)
                }) as Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>
            });
        let result = manager
            .execute_with_late_sent_agent_message(
                code,
                signal.clone(),
                Some(on_stream.clone()),
                late_handler,
            )
            .await;
        match result {
            Ok(result) => {
                return Ok(BusyKernelExecution {
                    result,
                    kernel_restarted,
                })
            }
            Err(error) => {
                let aborted = signal.as_ref().map(|signal| signal.is_aborted()).unwrap_or(false);
                if error != KernelError::BusyAfterInterrupt || aborted {
                    return Err(error);
                }
                let action = choose_busy_kernel_action(ctx, signal.clone()).await;
                if action == "wait" {
                    on_working_message(Some("Waiting for Python kernel..."));
                    continue;
                }
                if action == "kill" {
                    on_working_message(Some("Restarting Python kernel..."));
                    provisioner.kill().await;
                    kernel_restarted = true;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

/// Turn kernel image attachments into `ImageContent` blocks; non-image types are dropped.
pub fn image_blocks_from_attachments(
    attachments: Option<&[crate::core::kernel::shared::KernelAttachment]>,
) -> Vec<pi_ai::types::ImageContent> {
    let Some(attachments) = attachments else {
        return Vec::new();
    };
    attachments
        .iter()
        .filter(|attachment| is_image_mime_type(&attachment.mime_type))
        .map(|attachment| pi_ai::types::ImageContent::new(attachment.data.clone(), attachment.mime_type.clone()))
        .collect()
}

pub const IPYTHON_TOOL_DESCRIPTION: &str =
    "Execute Python code in a persistent Python REPL. Top-level `await` is supported. Variables, imports, and loaded data persist across calls, and are revived on a best-effort basis when a session is resumed (objects that cannot be serialized are dropped and reported). Run shell commands with `bash('cmd')` / `await bash('cmd')`. Project imports, tests, scripts, CLIs, and dependency checks should run through the target project's own environment.";

pub const IPYTHON_TOOL_PROMPT_SNIPPET: &str =
    "ipython - persistent Python REPL for code, state, and bash() orchestration";

/// Port of `createIpythonToolDefinition`'s execute body.
#[allow(clippy::too_many_arguments)]
pub async fn execute_ipython(
    provisioner: Arc<IpythonKernelProvisioner>,
    tool_call_id: &str,
    params: &IpythonToolInput,
    signal: Option<AbortSignal>,
    on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    // `options?.onLateSentAgentMessage` (packages/coding-agent/src/core/tools/ipython.ts:739),
    // threaded into the kernel session by executeWithBusyKernelChoice.
    on_late_sent_agent_message: Option<Arc<dyn Fn(String, KernelSentAgentMessage) + Send + Sync>>,
    ctx: Option<&ExtensionContext>,
) -> Result<(Vec<pi_agent_core::types::ContentBlock>, IpythonToolDetails, bool), KernelError> {
    if is_unsafe_windows_captured_launcher(&params.code) {
        return Ok((
            vec![pi_agent_core::types::ContentBlock::text(UNSAFE_WINDOWS_CAPTURED_LAUNCHER_MESSAGE)],
            IpythonToolDetails {
                duration_ms: Some(0.0),
                status: Some("error".to_string()),
                error_ename: Some("UnsafeWindowsCapturedLauncher".to_string()),
                stdout: Some(String::new()),
                stderr: Some(String::new()),
                ..IpythonToolDetails::default()
            },
            true,
        ));
    }

    let has_working_message = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let set_tool_working_message: Arc<dyn Fn(Option<&str>) + Send + Sync> = {
        let ctx = ctx.cloned();
        let has_working_message = has_working_message.clone();
        Arc::new(move |message: Option<&str>| {
            set_working_message(ctx.as_ref(), message);
            has_working_message.store(message.is_some(), std::sync::atomic::Ordering::SeqCst);
        })
    };

    let report_startup_progress: KernelBootstrapProgressHandler = {
        let set_tool_working_message = set_tool_working_message.clone();
        let on_update = on_update.clone();
        Arc::new(move |message: &str| {
            set_tool_working_message(Some(message));
            if let Some(on_update) = on_update.as_ref() {
                on_update(pi_agent_core::types::AgentToolResult::new(
                    vec![pi_agent_core::types::ContentBlock::text(message)],
                    serde_json::json!({ "status": "starting" }),
                ));
            }
        })
    };

    let stream_update = {
        let on_update = on_update.clone();
        Arc::new(move |chunk: &str, _name: StreamName| {
            if let Some(on_update) = on_update.as_ref() {
                on_update(pi_agent_core::types::AgentToolResult::new(
                    vec![pi_agent_core::types::ContentBlock::text(chunk)],
                    serde_json::json!({ "status": "ok" }),
                ));
            }
        }) as Arc<dyn Fn(&str, StreamName) + Send + Sync>
    };

    let model_tool_output_scope = provisioner.model_tool_output_scope();
    let model_tool_output_policy = provisioner.model_tool_output_policy();

    let execution = execute_with_busy_kernel_choice(
        provisioner,
        report_startup_progress,
        tool_call_id,
        &params.code,
        signal,
        stream_update,
        set_tool_working_message.clone(),
        on_late_sent_agent_message,
        ctx,
    )
    .await;

    if has_working_message.load(std::sync::atomic::Ordering::SeqCst) {
        set_tool_working_message(None);
    }

    let BusyKernelExecution {
        result: r,
        kernel_restarted,
    } = execution?;

    let mut text = r.stdout.clone();
    if !r.stderr.is_empty() {
        text += &format!("{}{}", if text.is_empty() { "" } else { "\n" }, r.stderr);
    }
    if let Some(result) = r.result.clone() {
        if !result.is_empty() {
            text += &format!("{}{}", if text.is_empty() { "" } else { "\n" }, result);
        }
    }
    if r.status == ExecuteStatus::Error {
        if let Some(error) = r.error.as_ref() {
            text += &format!(
                "{}{}",
                if text.is_empty() { "" } else { "\n" },
                error.traceback.join("\n")
            );
        }
    }
    if let Some(background_output) = r.background_output.as_ref() {
        if !background_output.is_empty() {
            text += &format!(
                "{}[background output (unattributed)]\n{}",
                if text.is_empty() { "" } else { "\n" },
                background_output
            );
        }
    }
    if kernel_restarted {
        text = if text.is_empty() {
            KERNEL_RESTART_NOTICE.to_string()
        } else {
            format!("{KERNEL_RESTART_NOTICE}\n\n{text}")
        };
    }

    if let Some(reports) = &r.execution_reports {
        for report in reports {
            text += &format!("{}[script result] {}", if text.is_empty() { "" } else { "\n" },
                serde_json::json!({"schema": report.schema, "stage": report.stage,
                    "scriptId": report.script_id, "exitCode": report.exit_code, "isError": report.failed()}));
        }
    }
    let image_blocks = image_blocks_from_attachments(r.attachments.as_deref());
    let mut content: Vec<pi_agent_core::types::ContentBlock> =
        vec![pi_agent_core::types::ContentBlock::text(text.clone())];
    for image in &image_blocks {
        content.push(pi_agent_core::types::ContentBlock::Image(image.clone()));
    }
    let is_error = r.status == ExecuteStatus::Error || r.status == ExecuteStatus::Aborted
        || r.execution_reports.as_ref().is_some_and(|reports| reports.iter().any(|report| report.failed()));
    let mut model_output_artifact: Option<ModelToolOutputArtifactV1> = None;
    let background_output_empty = r
        .background_output
        .as_ref()
        .map(|output| output.is_empty())
        .unwrap_or(true);
    let no_diffs = r.diffs.as_ref().map(|diffs| diffs.is_empty()).unwrap_or(true);
    let no_sent_messages = r
        .sent_agent_messages
        .as_ref()
        .map(|messages| messages.is_empty())
        .unwrap_or(true);
    if model_tool_output_policy == REPEATED_LARGE_TEXT_POLICY
        && model_tool_output_scope.is_some()
        && r.status == ExecuteStatus::Ok
        && !is_error
        && image_blocks.is_empty()
        && r.stderr.is_empty()
        && background_output_empty
        && !kernel_restarted
        && no_diffs
        && no_sent_messages
        && text.len() >= MODEL_TOOL_OUTPUT_MIN_BYTES
    {
        let scope = model_tool_output_scope.expect("scope checked");
        // Artifact persistence is optional. Keep the complete inline result on any failure.
        model_output_artifact = persist_model_tool_output_artifact(&text, &scope).ok();
    }

    let details = IpythonToolDetails {
        duration_ms: Some(r.duration_ms),
        status: Some(
            match r.status {
                ExecuteStatus::Ok => "ok",
                ExecuteStatus::Error => "error",
                ExecuteStatus::Aborted => "aborted",
            }
            .to_string(),
        ),
        execution_reports: r.execution_reports.clone(),
        error_ename: r.error.as_ref().map(|error| error.ename.clone()),
        stdout: Some(r.stdout.clone()),
        stderr: Some(r.stderr.clone()),
        result: r.result.clone(),
        background_output: r.background_output.clone(),
        diffs: r.diffs.clone(),
        attachments: r.attachments.clone(),
        sent_agent_messages: r.sent_agent_messages.clone(),
        model_output_artifact,
        kernel_restarted: Some(kernel_restarted),
        error: r.error.as_ref().map(|error| ExecErrorShape {
            ename: error.ename.clone(),
            evalue: error.evalue.clone(),
            traceback: error.traceback.clone(),
        }),
    };

    Ok((content, details, is_error))
}

/// Port of `createIpythonToolDefinition`.
pub fn create_ipython_tool_definition(
    cwd: &str,
    options: Option<IpythonToolOptions>,
) -> ToolDefinition<IpythonToolDetails> {
    let options = options.unwrap_or_default();
    let provisioner = options.provisioner.clone().unwrap_or_else(|| {
        IpythonKernelProvisioner::new(cwd, Some(options.clone()), default_kernel_client_factory())
    });
    // `options?.onLateSentAgentMessage` is read per tool call from the captured options
    // (packages/coding-agent/src/core/tools/ipython.ts:739), so the handler the session
    // registered in agent-session.ts:10099-10100 is not dropped at the tool boundary.
    let on_late_sent_agent_message = options.on_late_sent_agent_message.clone();

    let execute: ToolExecuteFn<IpythonToolDetails> = Arc::new(
        move |tool_call_id: String,
              params: Value,
              signal: Option<tokio_util::sync::CancellationToken>,
              on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
              ctx: ExtensionContext| {
            let provisioner = provisioner.clone();
            let on_late_sent_agent_message = on_late_sent_agent_message.clone();
            Box::pin(async move {
                let input: IpythonToolInput = serde_json::from_value(params)
                    .map_err(|error| anyhow::anyhow!("ipython tool input is invalid. {error}"))?;
                let abort_signal = signal.map(abort_signal_from_token);
                let (content, details, is_error) = execute_ipython(
                    provisioner,
                    &tool_call_id,
                    &input,
                    abort_signal,
                    on_update,
                    on_late_sent_agent_message,
                    Some(&ctx),
                )
                .await
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
                Ok(pi_agent_core::types::AgentToolResult::new(
                    content,
                    serde_json::to_value(details).unwrap_or(Value::Null),
                ).with_error(is_error))
            })
        },
    );

    ToolDefinition {
        name: "ipython".to_string(),
        label: "ipython".to_string(),
        description: IPYTHON_TOOL_DESCRIPTION.to_string(),
        prompt_snippet: Some(IPYTHON_TOOL_PROMPT_SNIPPET.to_string()),
        parameters: serde_json::from_str(IPYTHON_SCHEMA).expect("valid ipython schema"),
        // The kernel is single-threaded - pi must not run two ipython calls in parallel within a batch.
        execution_mode: Some(pi_agent_core::types::ToolExecutionMode::Sequential),
        execute,
        ..ToolDefinition::default()
    }
}

pub fn create_ipython_tool(cwd: &str, options: Option<IpythonToolOptions>) -> pi_agent_core::types::AgentTool {
    wrap_tool_definition(&create_ipython_tool_definition(cwd, options), None)
}

pub(crate) fn abort_signal_from_token(token: tokio_util::sync::CancellationToken) -> AbortSignal {
    let signal = AbortSignal::new();
    if token.is_cancelled() {
        signal.abort(None);
        return signal;
    }
    let target = signal.clone();
    pi_agent_core::execution_scope::spawn_auxiliary(async move {
        token.cancelled_owned().await;
        target.abort(None);
    });
    signal
}

/// Adapt the shared kernel lifecycle to the tool's owned-future interface.
struct ReplKernelClient {
    manager: Arc<dyn crate::core::kernel::shared::KernelClient>,
}

impl KernelClient for ReplKernelClient {
    fn shutdown_and_settle(&self, owner_session_id: &str, timeout_ms: u64)
        -> BoxFuture<'static, Result<crate::core::kernel::shared::KernelSettlement, KernelError>> {
        let manager = self.manager.clone();
        let owner = owner_session_id.to_string();
        Box::pin(async move { manager.shutdown_and_settle(&owner, timeout_ms).await })
    }
    fn is_running(&self) -> bool { self.manager.is_running() }
    fn is_defunct(&self) -> bool { self.manager.is_defunct() }
    fn start(&self, options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { manager.start(options).await })
    }
    fn execute(&self, code: &str, signal: Option<AbortSignal>, on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
        self.execute_with_late_sent_agent_message(code, signal, on_stream, None)
    }
    fn execute_with_late_sent_agent_message(
        &self,
        code: &str,
        signal: Option<AbortSignal>,
        on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        on_late_sent_agent_message: Option<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>>,
    ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
        let manager = self.manager.clone();
        let code = code.to_string();
        // The TypeScript passes one options object to the kernel client
        // (packages/coding-agent/src/core/tools/ipython.ts:644-650), so
        // `onLateSentAgentMessage` lands on `ExecuteOptions` and the kernel session
        // registers it once the cell settles
        // (packages/coding-agent/src/core/kernel/repl-manager.ts:1153-1155, reached in
        // Rust at core/kernel/repl_manager.rs:2222 through
        // `register_late_sent_agent_message_handler`). `..Default::default()` here
        // used to leave that field unset, so a kernel-sent agent message emitted after
        // the tool result was silently dropped.
        Box::pin(async move {
            manager
                .execute(
                    code,
                    ExecuteOptions {
                        signal,
                        on_stream,
                        on_late_sent_agent_message,
                        ..Default::default()
                    },
                )
                .await
        })
    }
    fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { Ok(manager.restore_state(Default::default()).await) })
    }
    fn shutdown(&self, snapshot: bool, drain_host_requests: bool) -> BoxFuture<'static, Result<(), KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { manager.shutdown(crate::core::kernel::shared::KernelShutdownOptions { snapshot, drain_host_requests }).await.map(|_| ()) })
    }
    fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { manager.kill().await; Ok(()) })
    }
    fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { Ok(manager.prune_oversized_variables().await.map(|result| PruneResult { pruned: result.pruned })) })
    }
    fn list_namespace_names(&self, signal: Option<AbortSignal>) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
        let manager = self.manager.clone();
        Box::pin(async move { Ok(manager.list_namespace_names(signal).await) })
    }
}

static KERNEL_CLIENT_FACTORY_OVERRIDE: Mutex<Option<KernelClientFactory>> = Mutex::new(None);

/// Test-only seam: replace the default kernel client factory so tests can drive
/// the real provisioner with a recording fake kernel. `None` restores the
/// native factory (the port of `newReplKernelManager`).
#[doc(hidden)]
pub fn set_kernel_client_factory_override(factory: Option<KernelClientFactory>) {
    *KERNEL_CLIENT_FACTORY_OVERRIDE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = factory;
}

pub(crate) fn default_kernel_client_factory() -> KernelClientFactory {
    if let Some(factory) = KERNEL_CLIENT_FACTORY_OVERRIDE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return factory;
    }
    Arc::new(|options: KernelManagerOptions| -> Arc<dyn KernelClient> {
        Arc::new(ReplKernelClient {
            manager: crate::core::kernel::repl_manager::new_repl_kernel_manager(options),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kernel::shared::{ExecError, KernelAttachment};

    // The launcher-guard markers are assembled from fragments so this test file
    // never stores the blocked pattern as one contiguous literal.
    fn launcher_capture_code() -> String {
        concat!("subprocess", ".run(['./", "run", ".ps1'], ", "capture_output", "=True)").to_string()
    }

    fn launcher_capture_code_without_script() -> String {
        concat!("subprocess", ".run(['ls'], ", "capture_output", "=True)").to_string()
    }

    fn python_skill(import_name: &str) -> KernelPythonSkill {
        KernelPythonSkill {
            import_name: import_name.to_string(),
            package_path: String::new(),
            pyproject_path: String::new(),
            name: import_name.to_string(),
        }
    }

    #[test]
    fn unsafe_windows_captured_launcher_detects_all_three_markers() {
        let code = launcher_capture_code();
        assert!(is_unsafe_windows_captured_launcher_on(&code, "win32"));
        assert!(!is_unsafe_windows_captured_launcher_on(&code, "linux"));
        assert!(!is_unsafe_windows_captured_launcher_on(&launcher_capture_code_without_script(), "win32"));
        assert!(!is_unsafe_windows_captured_launcher_on(
            &format!("{}", "subprocess"),
            "win32"
        ));
    }

    #[test]
    fn managed_kernel_path_honors_explicit_overrides_and_platform_keys() {
        for windows in [false, true] {
            let base = vec![("PATH".into(), "/managed:/system".into()), ("SECRET".into(), "not-forwarded".into())];
            let defaults = kernel_shell_env(base.clone(), &[], windows);
            assert_eq!(defaults.len(), 1);
            assert_eq!(defaults["PATH"], "/managed:/system");
            for path in ["", "/explicit"] {
                let env = kernel_shell_env(base.clone(), &[("PATH".into(), path.into()), ("OTHER".into(), "value".into())], windows);
                assert_eq!(env["PATH"], path);
                assert_eq!(env["OTHER"], "value");
                assert!(!env.contains_key("SECRET"));
            }
            let env = kernel_shell_env(base, &[("Path".into(), "override".into())], windows);
            assert_eq!(env["PATH"], if windows { "override" } else { "/managed:/system" });
            assert_eq!(env.contains_key("Path"), !windows);
        }
    }

    // Re-exec isolates PATH/profile changes from parallel tests and never provisions tools.
    #[cfg(unix)]
    #[tokio::test]
    async fn managed_kernel_path_resolves_both_tools_from_clean_environment() {
        const CHILD: &str = "OPTIMUS_MANAGED_PATH_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            use std::os::unix::fs::PermissionsExt;
            let root = tempfile::tempdir().unwrap();
            let bin = root.path().join("managed tools");
            std::fs::create_dir(&bin).unwrap();
            for name in ["rg", "fd"] {
                let tool = bin.join(name);
                std::fs::write(&tool, format!("#!/bin/sh\nprintf '{name}-fixture\\n'\n")).unwrap();
                std::fs::set_permissions(tool, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "core::tools::ipython::tests::managed_kernel_path_resolves_both_tools_from_clean_environment", "--nocapture"])
                .env_clear()
                .env(CHILD, "1")
                .env("HOME", root.path())
                .env("PI_BIN_DIR", &bin)
                .env("PI_OFFLINE", "1")
                .env("PATH", "")
                .output().unwrap();
            assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        assert_eq!(std::env::var("PATH").unwrap(), "");
        let captured = Arc::new(Mutex::new(None));
        let target = captured.clone();
        let cwd = std::env::var("HOME").unwrap();
        let provisioner = IpythonKernelProvisioner::new(&cwd, None, Arc::new(move |options| {
            *target.lock().unwrap() = options.env;
            Arc::new(StubKernelClient)
        }));
        provisioner.ensure(None, None).await.unwrap();
        let env = captured.lock().unwrap().clone().unwrap();
        assert_eq!(env["PATH"], std::env::var("PI_BIN_DIR").unwrap());
        let output = std::process::Command::new(&env["PRIME_AGENT_BASH_SHELL"])
            .args(["-c", "rg --version && fd --version"])
            .env_clear().envs(&env).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(String::from_utf8_lossy(&output.stdout), "rg-fixture\nfd-fixture\n");
        assert_eq!(std::env::var("PATH").unwrap(), "", "host PATH is not mutated");
        provisioner.dispose(Some(false)).await;
    }

    #[test]
    fn bootstrap_code_without_skills_is_the_base_code() {
        let code = build_rlm_bootstrap_code(&[]);
        assert!(code.starts_with("import asyncio"));
        assert!(code.contains("NO_COLOR"));
        assert!(code.contains("_PrimeAgentMissingRlm"));
        assert!(!code.contains("_PRIME_AGENT_SKILL_IMPORT_ERRORS"));
    }

    #[test]
    fn bootstrap_code_dedupes_skill_import_names() {
        let code = build_rlm_bootstrap_code(&[python_skill("agent_message"), python_skill("agent_message")]);
        assert!(code.contains("_PRIME_AGENT_SKILL_IMPORT_ERRORS"));
        assert_eq!(code.matches("for _prime_agent_skill_name in").count(), 1);
        assert!(code.contains("[\"agent_message\"]"));
    }

    #[test]
    fn image_blocks_drop_non_image_attachments() {
        let attachments = vec![
            KernelAttachment {
                mime_type: "image/png".to_string(),
                data: "AAAA".to_string(),
                path: None,
            },
            KernelAttachment {
                mime_type: "application/json".to_string(),
                data: "{}".to_string(),
                path: None,
            },
        ];
        let blocks = image_blocks_from_attachments(Some(&attachments));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].mime_type, "image/png");
        assert_eq!(blocks[0].data, "AAAA");
        assert!(image_blocks_from_attachments(None).is_empty());
    }

    #[test]
    fn busy_kernel_constants_match_typescript() {
        assert_eq!(BUSY_KERNEL_WAIT_CHOICE, "Wait and preserve state");
        assert_eq!(BUSY_KERNEL_KILL_CHOICE, "Kill kernel and restart");
        assert!(BUSY_KERNEL_PROMPT.starts_with("Python kernel is still busy\n"));
        assert!(KERNEL_RESTART_NOTICE.starts_with("<ipython_kernel_reset>\n"));
        assert!(KERNEL_RESTART_NOTICE.ends_with("</ipython_kernel_reset>"));
    }

    #[test]
    fn ipython_schema_matches_typescript_shape() {
        let schema: Value = serde_json::from_str(IPYTHON_SCHEMA).expect("schema");
        assert_eq!(schema["required"], serde_json::json!(["code"]));
        assert_eq!(schema["properties"]["code"]["type"], serde_json::json!("string"));
    }

    struct StubKernelClient;

    impl KernelClient for StubKernelClient {
        fn is_running(&self) -> bool {
            true
        }
        fn is_defunct(&self) -> bool {
            false
        }
        fn start(&self, _options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn execute(
            &self,
            _code: &str,
            _signal: Option<AbortSignal>,
            _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            Box::pin(async {
                Ok(ExecuteResult {
                    stdout: "ok".to_string(),
                    stderr: String::new(),
                    result: None,
                    diffs: None,
                    attachments: None,
                    sent_agent_messages: None,
                    background_output: None,
                    status: ExecuteStatus::Ok,
                    error: None,
                    execution_reports: None,
                    duration_ms: 3.0,
                })
            })
        }
        fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn shutdown(&self, _snapshot: bool, _drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
            Box::pin(async {
                Ok(Some(PruneResult {
                    pruned: Some(vec!["x".to_string()]),
                }))
            })
        }
        fn list_namespace_names(
            &self,
            _signal: Option<AbortSignal>,
        ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
            Box::pin(async { Ok(Some(vec!["x".to_string()])) })
        }
    }

    struct FailingKernelClient;

    impl KernelClient for FailingKernelClient {
        fn is_running(&self) -> bool {
            false
        }
        fn is_defunct(&self) -> bool {
            false
        }
        fn start(&self, _options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Err(KernelError::new("start failed")) })
        }
        fn execute(
            &self,
            _code: &str,
            _signal: Option<AbortSignal>,
            _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            Box::pin(async { Err(KernelError::new("execute failed")) })
        }
        fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn shutdown(&self, _snapshot: bool, _drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn list_namespace_names(
            &self,
            _signal: Option<AbortSignal>,
        ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
    }

    struct ErroringKernelClient;

    impl KernelClient for ErroringKernelClient {
        fn is_running(&self) -> bool {
            true
        }
        fn is_defunct(&self) -> bool {
            false
        }
        fn start(&self, _options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn execute(
            &self,
            code: &str,
            _signal: Option<AbortSignal>,
            _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            if code == build_rlm_bootstrap_code(&[]) {
                return StubKernelClient.execute(code, _signal, _on_stream);
            }
            Box::pin(async {
                Ok(ExecuteResult {
                    stdout: "partial".to_string(),
                    stderr: "warn".to_string(),
                    result: Some("value".to_string()),
                    diffs: None,
                    attachments: None,
                    sent_agent_messages: None,
                    background_output: Some("bg".to_string()),
                    status: ExecuteStatus::Error,
                    error: Some(ExecError {
                        ename: "ValueError".to_string(),
                        evalue: "bad".to_string(),
                        traceback: vec!["line 1".to_string(), "line 2".to_string()],
                    }),
                    execution_reports: None,
                    duration_ms: 9.0,
                })
            })
        }
        fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn shutdown(&self, _snapshot: bool, _drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn list_namespace_names(
            &self,
            _signal: Option<AbortSignal>,
        ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
    }

    /// Records whether the busy-kernel path forwarded a late-message handler.
    ///
    /// Only an override of `execute_with_late_sent_agent_message` can set
    /// `captured`, so the test proves the three-argument `execute` is no longer the
    /// call the ipython tool makes when a handler exists.
    struct LateHandlerCapturingKernelClient {
        captured: Mutex<Vec<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>>>,
    }

    impl KernelClient for LateHandlerCapturingKernelClient {
        fn is_running(&self) -> bool {
            true
        }
        fn is_defunct(&self) -> bool {
            false
        }
        fn start(&self, _options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        /// The provisioner's own bootstrap cell (and any other handler-free call)
        /// still goes through the plain `execute`, so that path must stay usable.
        fn execute(
            &self,
            code: &str,
            _signal: Option<AbortSignal>,
            _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
        ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            if code == build_rlm_bootstrap_code(&[]) {
                return StubKernelClient.execute(code, _signal, _on_stream);
            }
            Box::pin(async {
                Err(KernelError::new(
                    "a tool cell with a handler must not use plain execute",
                ))
            })
        }
        fn execute_with_late_sent_agent_message(
            &self,
            _code: &str,
            _signal: Option<AbortSignal>,
            _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
            on_late_sent_agent_message: Option<Arc<dyn Fn(KernelSentAgentMessage) + Send + Sync>>,
        ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            self.captured
                .lock()
                .unwrap()
                .push(on_late_sent_agent_message.expect("handler forwarded to the kernel session"));
            Box::pin(async {
                Ok(ExecuteResult {
                    stdout: "ok".to_string(),
                    stderr: String::new(),
                    result: None,
                    diffs: None,
                    attachments: None,
                    sent_agent_messages: None,
                    background_output: None,
                    status: ExecuteStatus::Ok,
                    error: None,
                    execution_reports: None,
                    duration_ms: 1.0,
                })
            })
        }
        fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn shutdown(&self, _snapshot: bool, _drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
            Box::pin(async { Ok(()) })
        }
        fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
        fn list_namespace_names(
            &self,
            _signal: Option<AbortSignal>,
        ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
            Box::pin(async { Ok(None) })
        }
    }

    fn sent_agent_message(id: &str, message: &str) -> KernelSentAgentMessage {
        KernelSentAgentMessage {
            id: id.to_string(),
            message: message.to_string(),
            delivery_status: crate::core::kernel::shared::KernelDeliveryStatus::Delivered,
            receiver_role: Some(crate::core::kernel::shared::KernelReceiverRole::Parent),
            target: crate::core::kernel::shared::KernelSentAgentMessageTarget {
                active_session_id: "active".to_string(),
                session_id: "session".to_string(),
                session_name: None,
            },
        }
    }

    /// `onLateSentAgentMessage: (message) => onLateSentAgentMessage(toolCallId, message)`
    /// (packages/coding-agent/src/core/tools/ipython.ts:647-649): the tool-call id is
    /// bound at the call site and the bound handler reaches the kernel session.
    #[tokio::test]
    async fn busy_kernel_path_binds_the_tool_call_id_and_forwards_the_late_handler() {
        let client = Arc::new(LateHandlerCapturingKernelClient {
            captured: Mutex::new(Vec::new()),
        });
        let provisioner = provisioner_with(client.clone());
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_for_handler = seen.clone();
        let handler: Arc<dyn Fn(String, KernelSentAgentMessage) + Send + Sync> = Arc::new(
            move |tool_call_id: String, message: KernelSentAgentMessage| {
                seen_for_handler.lock().unwrap().push((tool_call_id, message.id));
            },
        );

        let execution = execute_with_busy_kernel_choice(
            provisioner,
            Arc::new(|_message: &str| {}),
            "call-late",
            "print(1)",
            None,
            Arc::new(|_chunk: &str, _name: StreamName| {}),
            Arc::new(|_message: Option<&str>| {}),
            Some(handler),
            None,
        )
        .await
        .expect("executed");
        assert_eq!(execution.result.stdout, "ok");

        let captured = client.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        captured[0](sent_agent_message("msg-1", "hello"));
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [("call-late".to_string(), "msg-1".to_string())]
        );
    }

    /// A caller that passes no handler keeps the plain `execute` shape - the
    /// default trait method must not invent one (ipython.ts:647-649).
    #[tokio::test]
    async fn busy_kernel_path_without_a_handler_uses_plain_execute() {
        let provisioner = provisioner_with(Arc::new(StubKernelClient));
        let execution = execute_with_busy_kernel_choice(
            provisioner,
            Arc::new(|_message: &str| {}),
            "call-plain",
            "print(1)",
            None,
            Arc::new(|_chunk: &str, _name: StreamName| {}),
            Arc::new(|_message: Option<&str>| {}),
            None,
            None,
        )
        .await
        .expect("executed");
        assert_eq!(execution.result.stdout, "ok");
    }

    fn provisioner_with(client: Arc<dyn KernelClient>) -> Arc<IpythonKernelProvisioner> {
        IpythonKernelProvisioner::new("/tmp", None, Arc::new(move |_options| client.clone()))
    }

    #[tokio::test]
    async fn replacement_disposes_previous_without_starting_new_kernel() {
        let previous = provisioner_with(Arc::new(StubKernelClient));
        previous.ensure(None, None).await.expect("old kernel started");
        assert!(previous.has_running_kernel());
        let ready_gate = previous.replacement_ready_gate();

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !previous.dispose_controller.is_aborted() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reload must dispose the previous kernel without another tool call");
        tokio::time::timeout(std::time::Duration::from_secs(2), ready_gate())
            .await
            .expect("old kernel flush finished");
        assert!(!previous.has_running_kernel());
        assert!(previous.manager_promise.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn unstarted_replacement_dispose_waits_for_previous_flush() {
        let (release, receiver) = tokio::sync::watch::channel(false);
        let replacement = IpythonKernelProvisioner::new(
            "/tmp",
            Some(IpythonToolOptions {
                ready_gate: Some(Arc::new(move || {
                    let mut receiver = receiver.clone();
                    Box::pin(async move {
                        receiver.wait_for(|released| *released).await.expect("flush released");
                    })
                })),
                ..Default::default()
            }),
            Arc::new(|_| panic!("disposing an unused replacement must not start a kernel")),
        );
        let disposal = replacement.dispose(None);
        tokio::pin!(disposal);
        tokio::select! {
            _ = &mut disposal => panic!("dispose returned before the previous snapshot flush"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
        }
        release.send(true).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), disposal)
            .await
            .expect("dispose must finish after the previous snapshot flush");
        assert!(replacement.manager_promise.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn default_kernel_factory_constructs_a_lazy_real_manager() {
        let directory = tempfile::tempdir().unwrap();
        let manager = default_kernel_client_factory()(KernelManagerOptions {
            cwd: Some(directory.path().to_string_lossy().into_owned()),
            // Construction must not run the interpreter or bootstrap a venv.
            python: Some(directory.path().join("not-started-python").to_string_lossy().into_owned()),
            ..Default::default()
        });
        assert!(!manager.is_running());
        assert!(!manager.is_defunct());
        manager.shutdown(false, true).await.unwrap();
        assert!(!manager.is_running());
    }

    #[test]
    fn ipython_tool_definition_uses_sequential_execution() {
        let definition = create_ipython_tool_definition(
            "/tmp",
            Some(IpythonToolOptions {
                provisioner: Some(provisioner_with(Arc::new(StubKernelClient))),
                ..IpythonToolOptions::default()
            }),
        );
        assert_eq!(definition.name, "ipython");
        assert_eq!(definition.label, "ipython");
        assert_eq!(
            definition.execution_mode,
            Some(pi_agent_core::types::ToolExecutionMode::Sequential)
        );
        assert_eq!(
            definition.parameters,
            serde_json::from_str::<Value>(IPYTHON_SCHEMA).expect("schema")
        );
    }

    #[tokio::test]
    async fn provisioner_starts_once_and_reuses_the_manager() {
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts_for_factory = starts.clone();
        let provisioner = IpythonKernelProvisioner::new(
            "/tmp",
            None,
            Arc::new(move |_options| -> Arc<dyn KernelClient> {
                starts_for_factory.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Arc::new(StubKernelClient)
            }),
        );
        let first = provisioner.ensure(None, None).await.expect("first");
        let second = provisioner.ensure(None, None).await.expect("second");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(starts.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(provisioner.has_running_kernel());
        assert_eq!(provisioner.manager().map(|manager| manager.is_running()), Some(true));
    }

    #[tokio::test]
    async fn provisioner_clears_the_memo_after_a_failed_startup() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempts_for_factory = attempts.clone();
        let provisioner = IpythonKernelProvisioner::new(
            "/tmp",
            None,
            Arc::new(move |_options| -> Arc<dyn KernelClient> {
                let attempt = attempts_for_factory.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if attempt == 0 {
                    Arc::new(FailingKernelClient)
                } else {
                    Arc::new(StubKernelClient)
                }
            }),
        );
        assert!(provisioner.ensure(None, None).await.is_err());
        assert!(provisioner.ensure(None, None).await.is_ok());
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn provisioner_rejects_an_already_aborted_signal() {
        let provisioner = provisioner_with(Arc::new(StubKernelClient));
        let signal = AbortSignal::new();
        signal.abort(None);
        let error = provisioner
        .ensure(None, Some(signal))
        .await
        .err()
        .expect("aborted");
        assert_eq!(error.to_string(), "Python execution aborted");
    }

    #[tokio::test]
    async fn provisioner_reports_namespace_and_prune_results() {
        let provisioner = provisioner_with(Arc::new(StubKernelClient));
        provisioner.ensure(None, None).await.expect("started");
        assert_eq!(
            provisioner.prune_oversized_variables().await,
            Some(vec!["x".to_string()])
        );
        assert_eq!(
            provisioner.list_namespace_names(None).await,
            Some(vec!["x".to_string()])
        );
    }

    #[tokio::test]
    async fn execute_ipython_returns_details_for_a_successful_cell() {
        let provisioner = provisioner_with(Arc::new(StubKernelClient));
        let (content, details, is_error) = execute_ipython(
            provisioner,
            "call-1",
            &IpythonToolInput {
                code: "print(1)".to_string(),
            },
            None,
            None,
            None,
            None,
        )
        .await
        .expect("executed");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0].as_text(), Some("ok"));
        assert_eq!(details.status.as_deref(), Some("ok"));
        assert_eq!(details.kernel_restarted, Some(false));
        assert_eq!(details.stdout.as_deref(), Some("ok"));
        assert!(!is_error);
    }

    #[tokio::test]
    async fn execute_ipython_joins_error_sections_and_marks_error() {
        let provisioner = provisioner_with(Arc::new(ErroringKernelClient));
        let (content, details, is_error) = execute_ipython(
            provisioner,
            "call-2",
            &IpythonToolInput {
                code: "raise ValueError".to_string(),
            },
            None,
            None,
            None,
            None,
        )
        .await
        .expect("executed");
        assert_eq!(
            content[0].as_text(),
            Some("partial\nwarn\nvalue\nline 1\nline 2\n[background output (unattributed)]\nbg")
        );
        assert_eq!(details.status.as_deref(), Some("error"));
        assert_eq!(details.error_ename.as_deref(), Some("ValueError"));
        assert_eq!(details.error.as_ref().map(|error| error.traceback.len()), Some(2));
        assert!(is_error);
    }

    #[tokio::test]
    async fn execute_ipython_runs_the_launcher_guard_per_platform() {
        let provisioner = provisioner_with(Arc::new(StubKernelClient));
        let (content, details, is_error) = execute_ipython(
            provisioner,
            "call-3",
            &IpythonToolInput {
                code: launcher_capture_code(),
            },
            None,
            None,
            None,
            None,
        )
        .await
        .expect("executed");
        // The guard is win32-only; on other platforms the cell runs normally.
        if cfg!(windows) {
            assert_eq!(details.error_ename.as_deref(), Some("UnsafeWindowsCapturedLauncher"));
            assert!(is_error);
        } else {
            assert_eq!(content[0].as_text(), Some("ok"));
            assert!(!is_error);
        }
    }

    #[test]
    fn linked_abort_signal_propagates_and_cleans_up() {
        let source = AbortSignal::new();
        let linked = LinkedAbortSignal::new(vec![Some(source.clone())]);
        assert!(!linked.signal.is_aborted());
        source.abort(None);
        assert!(linked.signal.is_aborted());
        linked.cleanup();
    }

    #[test]
    fn model_tool_output_scope_requires_session_and_snapshot_dir() {
        let provisioner = IpythonKernelProvisioner::new(
            "/tmp",
            Some(IpythonToolOptions {
                session_id: Some("session-1".to_string()),
                ..IpythonToolOptions::default()
            }),
            Arc::new(|_options| -> Arc<dyn KernelClient> { Arc::new(StubKernelClient) }),
        );
        assert!(provisioner.model_tool_output_scope().is_none());
        let with_dir = IpythonKernelProvisioner::new(
            "/tmp",
            Some(IpythonToolOptions {
                session_id: Some("session-1".to_string()),
                snapshot_dir: Some("/tmp/artifacts".to_string()),
                ..IpythonToolOptions::default()
            }),
            Arc::new(|_options| -> Arc<dyn KernelClient> { Arc::new(StubKernelClient) }),
        );
        let scope = with_dir.model_tool_output_scope().expect("scope");
        assert_eq!(scope.session_id, "session-1");
        assert_eq!(scope.session_artifact_dir, "/tmp/artifacts");
        assert_eq!(with_dir.model_tool_output_policy(), "off");
    }

    #[test]
    fn resolve_kernel_bash_shell_prefers_explicit_path() {
        assert_eq!(
            resolve_kernel_bash_shell(Some("  /custom/bash  ")).as_deref(),
            Some("/custom/bash")
        );
        if !cfg!(windows) {
            assert!(resolve_kernel_bash_shell(None).is_some());
        }
    }

    /// G2-08: the busy-kernel dialog must honor the tool abort signal (TS
    /// ipython.ts:616-618 passes the signal; abort closes the dialog and the
    /// loop keeps the original error). A stale selection must not resume the
    /// abandoned loop.
    #[tokio::test]
    async fn busy_dialog_abort_cancels_prompt_and_stays_cancelled() {
        use super::super::{ExtensionUiContext, ExtensionUiDialogOptions};
        use std::time::Duration;

        let signal = AbortSignal::new();
        let ctx = ExtensionContext {
            has_ui: true,
            cwd: String::new(),
            ui: ExtensionUiContext {
                select: Some(Arc::new(
                    |_title: String, _options: Vec<String>, _opts: ExtensionUiDialogOptions| {
                        Box::pin(async move {
                            // A real user is still deciding; only the abort may end this.
                            futures::future::pending::<Option<String>>().await
                        }) as BoxFuture<'static, Option<String>>
                    },
                )),
                ..Default::default()
            },
        };
        let ctx_ref = &ctx;
        let action = tokio::time::timeout(Duration::from_secs(10), async {
            let fut = choose_busy_kernel_action(Some(ctx_ref), Some(signal.clone()));
            tokio::pin!(fut);
            tokio::select! {
                action = &mut fut => panic!("dialog resolved without the abort: {action}"),
                _ = tokio::time::sleep(Duration::from_millis(300)) => {}
            }
            signal.abort(None);
            fut.await
        })
        .await
        .expect("busy dialog did not close on abort: the kernel dialog ignores the signal");
        assert_eq!(action, "cancel", "abort must cancel the busy prompt");
    }

    /// Negative controls: a real selection still resolves wait/kill, and an
    /// already-aborted signal cancels without opening a dialog.
    #[tokio::test]
    async fn busy_dialog_selections_still_resolve_and_preflight_abort_cancels() {
        use super::super::{ExtensionUiContext, ExtensionUiDialogOptions};
        use std::time::Duration;

        for (choice, expected) in [
            (BUSY_KERNEL_WAIT_CHOICE.to_string(), "wait"),
            (BUSY_KERNEL_KILL_CHOICE.to_string(), "kill"),
        ] {
            let ctx = ExtensionContext {
                has_ui: true,
                cwd: String::new(),
                ui: ExtensionUiContext {
                    select: Some(Arc::new({
                        let choice = choice.clone();
                        move |_title: String, _options: Vec<String>, _opts: ExtensionUiDialogOptions| {
                            let choice = choice.clone();
                            Box::pin(async move { Some(choice) }) as BoxFuture<'static, Option<String>>
                        }
                    })),
                    ..Default::default()
                },
            };
            let action = choose_busy_kernel_action(Some(&ctx), None).await;
            assert_eq!(action, expected, "selection {choice} must resolve {expected}");
        }

        let signal = AbortSignal::new();
        signal.abort(None);
        let ctx = ExtensionContext {
            has_ui: true,
            cwd: String::new(),
            ui: ExtensionUiContext::default(),
        };
        let action = tokio::time::timeout(Duration::from_secs(2), choose_busy_kernel_action(Some(&ctx), Some(signal.clone())))
            .await
            .expect("preflight abort must not hang");
        assert_eq!(action, "cancel");
    }


    type SettlementCallback = Arc<dyn Fn(String, u64) -> BoxFuture<'static,
        Result<crate::core::kernel::shared::KernelSettlement, KernelError>> + Send + Sync>;

    struct SettlementKernelClient {
        on_settle: SettlementCallback,
        startup_entered: Option<Arc<tokio::sync::Notify>>,
    }

    impl KernelClient for SettlementKernelClient {
        fn is_running(&self) -> bool { true }
        fn is_defunct(&self) -> bool { false }
        fn start(&self, options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
            match self.startup_entered.clone() {
                None => StubKernelClient.start(options),
                Some(entered) => Box::pin(async move {
                    entered.notify_one();
                    options.signal.expect("controlled startup signal").wait().await;
                    Err(create_kernel_startup_abort_error())
                }),
            }
        }
        fn execute(&self, code: &str, signal: Option<AbortSignal>,
            on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>)
            -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
            StubKernelClient.execute(code, signal, on_stream)
        }
        fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
            StubKernelClient.restore_state()
        }
        fn shutdown(&self, snapshot: bool, drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
            StubKernelClient.shutdown(snapshot, drain)
        }
        fn shutdown_and_settle(&self, owner: &str, timeout_ms: u64)
            -> BoxFuture<'static, Result<crate::core::kernel::shared::KernelSettlement, KernelError>> {
            (self.on_settle)(owner.to_string(), timeout_ms)
        }
        fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> { StubKernelClient.kill() }
        fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
            StubKernelClient.prune_oversized_variables()
        }
        fn list_namespace_names(&self, signal: Option<AbortSignal>)
            -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
            StubKernelClient.list_namespace_names(signal)
        }
    }

    fn settlement_ok() -> crate::core::kernel::shared::KernelSettlement {
        crate::core::kernel::shared::KernelSettlement {
            ownership_scope: "native-kernel-job-members".into(), supported: true,
            settled: true, kernel_exited: true, descendants_exited: true,
            local_tasks_settled: true, errors: Vec::new(),
        }
    }

    fn settlement_provisioner(owner: &str) -> Arc<IpythonKernelProvisioner> {
        IpythonKernelProvisioner::new("/tmp", Some(IpythonToolOptions {
            session_id: Some(owner.into()), ..Default::default()
        }), Arc::new(|_| panic!("settlement-only fixture must not create a kernel")))
    }

    #[tokio::test]
    async fn provisioner_settlement_bounds_all_clients_and_aggregates_failures() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicUsize>);
        impl Drop for Dropped {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }
        let provisioner = settlement_provisioner("owned");
        let entered = Arc::new(Mutex::new(Vec::new()));
        let dropped = Arc::new(AtomicUsize::new(0));
        for index in 0..4 {
            let entered = entered.clone();
            let dropped = dropped.clone();
            provisioner.ownership.lock().unwrap().clients.push(Arc::new(SettlementKernelClient {
                startup_entered: None,
                on_settle: Arc::new(move |owner, timeout_ms| {
                    let entered = entered.clone();
                    let guard = Dropped(dropped.clone());
                    Box::pin(async move {
                        let _guard = guard;
                        assert_eq!(owner, "owned");
                        assert_eq!(timeout_ms, 0);
                        entered.lock().unwrap().push(index);
                        match index {
                            0 => std::future::pending().await,
                            1 => Err(KernelError::new("controlled adapter failure")),
                            2 => Ok(settlement_ok()),
                            _ => panic!("controlled adapter panic"),
                        }
                    })
                }),
            }));
        }
        let report = tokio::time::timeout(std::time::Duration::from_secs(1),
            provisioner.shutdown_and_settle("owned", 0)).await.unwrap().unwrap();
        let mut seen = entered.lock().unwrap().clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3], "failure must not skip later generations");
        assert_eq!(dropped.load(Ordering::SeqCst), 4, "no timed-out adapter future is detached");
        assert!(!report.settled && !report.kernel_exited && !report.descendants_exited && !report.local_tasks_settled);
        assert_eq!(report.errors.len(), 3, "{report:?}");
        for expected in ["deadline expired", "controlled adapter failure", "adapter panicked"] {
            assert!(report.errors.iter().any(|error| error.contains(expected)), "{report:?}");
        }
    }

    #[tokio::test]
    async fn provisioner_settlement_admits_clients_concurrently() {
        let provisioner = settlement_provisioner("owned");
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        for _ in 0..2 {
            let barrier = barrier.clone();
            provisioner.ownership.lock().unwrap().clients.push(Arc::new(SettlementKernelClient {
                startup_entered: None,
                on_settle: Arc::new(move |_, _| {
                    let barrier = barrier.clone();
                    Box::pin(async move { barrier.wait().await; Ok(settlement_ok()) })
                }),
            }));
        }
        let report = provisioner.shutdown_and_settle("owned", 1000).await.unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(report.kernel_exited && report.descendants_exited && report.local_tasks_settled);
        assert_eq!(report.settled, cfg!(windows));
    }

    #[tokio::test]
    async fn provisioner_settlement_rejects_empty_or_wrong_owner_before_mutation() {
        for (configured, supplied) in [("", ""), ("owned", "other"), ("owned", "")] {
            let provisioner = settlement_provisioner(configured);
            assert!(provisioner.shutdown_and_settle(supplied, 1000).await.is_err());
            assert!(!provisioner.ownership.lock().unwrap().fenced);
            assert!(!provisioner.dispose_controller.is_aborted());
            assert!(!provisioner.owned_tasks.status().fenced);
        }
    }

    #[tokio::test]
    async fn provisioner_settlement_late_ensure_cannot_publish_a_hanging_memo() {
        let provisioner = settlement_provisioner("owned");
        provisioner.shutdown_and_settle("owned", 1000).await.unwrap();
        for _ in 0..2 {
            let result = tokio::time::timeout(std::time::Duration::from_secs(1),
                provisioner.ensure(None, None)).await.expect("late ensure must decline promptly");
            assert!(result.err().unwrap().to_string().contains("retained-stop fence"));
            assert!(provisioner.manager_promise.lock().unwrap().is_none());
            assert!(provisioner.ownership.lock().unwrap().startups.is_empty());
        }
        assert!(provisioner.current_manager().await.is_none());
    }

    #[tokio::test]
    async fn provisioner_settlement_startup_race_keeps_registered_client() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let entered = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(AtomicUsize::new(0));
        let stop_count = stopped.clone();
        let client: Arc<dyn KernelClient> = Arc::new(SettlementKernelClient {
            startup_entered: Some(entered.clone()),
            on_settle: Arc::new(move |_, _| {
                let stopped = stop_count.clone();
                Box::pin(async move { stopped.fetch_add(1, Ordering::SeqCst); Ok(settlement_ok()) })
            }),
        });
        let provisioner = IpythonKernelProvisioner::new("/tmp", Some(IpythonToolOptions {
            session_id: Some("owned".into()), ..Default::default()
        }), Arc::new(move |_| client.clone()));
        let starter = provisioner.clone();
        let starting = tokio::spawn(async move { starter.ensure(None, None).await });
        tokio::time::timeout(std::time::Duration::from_secs(1), entered.notified()).await.unwrap();
        let report = provisioner.shutdown_and_settle("owned", 1000).await.unwrap();
        assert_eq!(stopped.load(Ordering::SeqCst), 1);
        assert!(report.errors.is_empty() && report.local_tasks_settled, "{report:?}");
        assert_eq!(report.settled, cfg!(windows));
        assert!(tokio::time::timeout(std::time::Duration::from_secs(1), starting).await.unwrap().unwrap().is_err());
        assert!(provisioner.ensure(None, None).await.is_err());
        assert_eq!(provisioner.ownership.lock().unwrap().clients.len(), 1);
    }

    #[tokio::test]
    async fn provisioner_settlement_uncovered_completed_work_is_unsupported() {
        let provisioner = settlement_provisioner("owned");
        provisioner.owned_tasks.spawn(async {}, false, false).unwrap().await.unwrap();
        let report = provisioner.shutdown_and_settle("owned", 1000).await.unwrap();
        assert!(!report.supported && !report.settled, "{report:?}");
        assert!(report.errors.iter().any(|error| error.contains("unproved local work ownership")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provisioner_settlement_requires_owned_stream_acknowledgement() {
        let provisioner = settlement_provisioner("owned");
        let stream = pi_ai::utils::event_stream::AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        let (release, blocked) = std::sync::mpsc::channel();
        let (entered, started) = tokio::sync::oneshot::channel();
        stream.spawn(async move {
            entered.send(()).unwrap();
            let _ = blocked.recv(); // controlled non-cooperative task, released below
        });
        started.await.unwrap();
        provisioner.owned_tasks.register_stream(receipt.clone()).unwrap();
        let report = provisioner.shutdown_and_settle("owned", 0).await.unwrap();
        let release_result = release.send(());
        assert!(!report.settled && !report.local_tasks_settled, "{report:?}");
        assert!(report.errors.iter().any(|error| error.contains("owned tasks or streams")));
        release_result.unwrap();
        assert!(receipt.settle(std::time::Duration::from_secs(1)).await.settled);
    }
}
