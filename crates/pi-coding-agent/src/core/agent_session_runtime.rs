//! Port of packages/coding-agent/src/core/agent-session-runtime.ts

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::vec::Vec;

use futures::future::Shared;
use futures::FutureExt;
use serde_json::Value;

use crate::core::agent_session::{AgentSession, ExtensionBindings};
use crate::core::agent_session_config::AgentSessionRuntimeConfig;
use crate::core::agent_session_services::{
    AgentSessionCreationOptions, AgentSessionRuntimeDiagnostic, AgentSessionServices,
};
use crate::core::auth_guidance::is_obsolete_model_fallback_message;
use crate::core::extensions::types::{
    ExtensionEvent, SessionBeforeForkPayload, SessionBeforeSwitchPayload, SessionShutdownPayload,
};
use crate::core::agent_session::RLM_CHILD_AGENT_STATUS_CANCELLED;
use crate::core::agent_session::ScopedModel;
use crate::core::rlm_runtime::{
    CreateRlmSubagentRuntimeOptions, RlmSubagentRuntime, SubagentRuntimeHost,
};
use crate::core::sdk::CreateAgentSessionResult;
use crate::core::session_cwd::{assert_session_cwd_exists, SessionCwdSource};
use crate::core::session_import_errors::SessionImportFileNotFoundError;
use crate::core::session_lease::{acquire_session_lease, canonical_session_path, SessionLease};
use crate::core::session_manager::{NewSessionOptions, SessionManager};

pub type BoxFuture<T> = pi_ai::types::BoxFuture<T>;

mod daemon_adapter;
mod in_process_adapter;

pub(crate) use daemon_adapter::{AgentSessionDaemonAdapter, run_native_daemon_mode};
pub(crate) use in_process_adapter::InProcessRuntimeHostAdapter;


// ---------------------------------------------------------------------------
// Re-exports (`export { ... } from "./agent-session-services.js"`)
// ---------------------------------------------------------------------------

pub use crate::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
};
pub use crate::core::agent_session_services::{
    AgentSessionRuntimeDiagnostic as RuntimeDiagnostic, AgentSessionServices as RuntimeServices,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};

/// `interface CreateAgentSessionRuntimeResult extends CreateAgentSessionResult`.
pub struct CreateAgentSessionRuntimeResult {
    /// `...CreateAgentSessionResult`.
    pub result: CreateAgentSessionResult,
    pub services: Arc<AgentSessionServices>,
    pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// The argument object of `CreateAgentSessionRuntimeFactory`.
#[derive(Clone)]
pub struct CreateAgentSessionRuntimeInput {
    pub cwd: String,
    pub agent_dir: String,
    pub session_manager: Arc<Mutex<SessionManager>>,
    pub session_start_event: Option<Value>,
    pub session_config: Option<AgentSessionRuntimeConfig>,
    pub session_options: Option<AgentSessionCreationOptions>,
    pub runtime_metadata: Option<AgentSessionRuntimeMetadata>,
    pub session_lease: Option<Arc<Mutex<SessionLease>>>,
}

/// `type CreateAgentSessionRuntimeFactory = (options) => Promise<CreateAgentSessionRuntimeResult>`.
pub type CreateAgentSessionRuntimeFactory = Arc<
    dyn Fn(CreateAgentSessionRuntimeInput) -> BoxFuture<Result<CreateAgentSessionRuntimeResult, String>>
        + Send
        + Sync,
>;

/// `type AgentSessionRuntimeKind = "top-level" | "subagent"`.
pub const AGENT_SESSION_RUNTIME_KIND_TOP_LEVEL: &str = "top-level";
pub const AGENT_SESSION_RUNTIME_KIND_SUBAGENT: &str = "subagent";

/// `interface AgentSessionRuntimeMetadata`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionRuntimeMetadata {
    pub kind: String,
    pub created_at: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_active_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_child_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_parent_node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rehydrated_completed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spawn_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<String>,
}

impl AgentSessionRuntimeMetadata {
    /// `{ kind: "top-level", createdAt: Date.now() }`.
    pub fn top_level() -> Self {
        Self {
            kind: AGENT_SESSION_RUNTIME_KIND_TOP_LEVEL.to_string(),
            created_at: now_millis(),
            parent_active_session_id: None,
            parent_session_id: None,
            parent_session_file: None,
            rlm_child_id: None,
            rlm_parent_node_id: None,
            rehydrated_completed: None,
            prompt: None,
            spawn_code: None,
            session_dir: None,
        }
    }
}

/// `extractUserMessageText(content)`.
pub fn extract_user_message_text(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    content
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<&str>>()
                .join("")
        })
        .unwrap_or_default()
}

/// `Date.now()`.
fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

/// `interface AgentSessionRuntimeDisposeOptions`.
#[derive(Debug, Clone, Default)]
pub struct AgentSessionRuntimeDisposeOptions {
    /// `kernelSnapshot?: boolean` - default true.
    pub kernel_snapshot: Option<bool>,
}

/// `type CreateAgentSessionRuntimeFactory` options object, kept as a plain struct.
pub type RuntimeEnvScope = Arc<
    dyn Fn(
            BoxFuture<Result<CreateAgentSessionRuntimeResult, String>>,
        ) -> BoxFuture<Result<CreateAgentSessionRuntimeResult, String>>
        + Send
        + Sync,
>;

/// `(ctx: ReplacedSessionContext) => Promise<void>` used by `withSession`.
pub type WithSessionCallback = Arc<
    dyn Fn(crate::core::agent_session::ReplacedSessionContext) -> BoxFuture<()> + Send + Sync,
>;

/// `(sessionManager: SessionManager) => Promise<void>` used by `newSession({ setup })`.
pub type NewSessionSetupCallback = Arc<dyn Fn(Arc<Mutex<SessionManager>>) -> BoxFuture<()> + Send + Sync>;

/// `{ cancelled: boolean; selectedText?: string }`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionReplacementResult {
    pub cancelled: bool,
    pub selected_text: Option<String>,
}

/// Return value of `switchSession`/`newSession`/`importFromJsonl`
/// (`{ cancelled: boolean }`). The `session_before_switch` payload uses
/// `cancel`; this is the wrapper result.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeEventResult {
    pub cancelled: bool,
}

/// `this.session.extensionRunner`, the surface `agent-session-runtime.ts` drives.
///
/// The landed `core::agent_session::ExtensionRunner` seam exposes typed emitters
/// but not the generic `emit(event)` this file drives, so `hasHandlers`/`emit`
/// arrive through an injected seam. `bindExtensions`, `setSessionName`,
/// `getRlmChildRunStatus` and `createReplacedSessionContext` are called on the
/// session itself, as in the TypeScript. Until a runner is attached the results
/// match the TypeScript `hasHandlers(...) === false` branch: nothing is emitted
/// and nothing is cancelled. Recorded in `blocked_on`.
pub trait RuntimeExtensionRunner: Send + Sync {
    fn has_handlers(&self, event_type: &str) -> bool;
    /// `runner.emit(event)` - `session_before_switch`, `session_before_fork`,
    /// `session_shutdown`.
    fn emit_event(&self, event: Value) -> BoxFuture<Option<Value>>;
}

/// Adapter over the session's live `extensionRunner`.
///
/// `agent-session-runtime.ts` reads `this.session.extensionRunner` on every emit
/// (lines 170, 187, 201, 692), so the runtime resolves the CURRENT session's
/// runner on each call instead of holding a captured handle. An attached seam
/// still wins so an embedder can override the routing.
struct SessionRunnerAdapter {
    runner: crate::core::agent_session::ExtensionRunner,
}

impl RuntimeExtensionRunner for SessionRunnerAdapter {
    fn has_handlers(&self, event_type: &str) -> bool {
        self.runner.has_handlers(event_type)
    }

    fn emit_event(&self, event: Value) -> BoxFuture<Option<Value>> {
        let runner = Arc::clone(&self.runner);
        Box::pin(async move {
            // The runtime passes serialized `{ type, ... }` payloads; they decode
            // back into the typed `ExtensionEvent` the runner dispatches on.
            match serde_json::from_value::<crate::core::extensions::types::ExtensionEvent>(event) {
                Ok(event) => runner.emit(event).await,
                Err(_) => None,
            }
        })
    }
}

impl AgentSessionRuntime {
    /// Attaches the `ExtensionRunner` seam for the events this file emits.
    pub fn set_session_extension_runner(&self, runner: Option<Arc<dyn RuntimeExtensionRunner>>) {
        *self
            .session_runner
            .lock()
            .expect("session runner poisoned") = runner;
    }

    /// `this.session.extensionRunner` (agent-session-runtime.ts:170).
    ///
    /// The session's own runner is resolved per call; the injected seam is only a
    /// fallback for embedders that route these events elsewhere.
    fn session_extension_runner(&self) -> Option<Arc<dyn RuntimeExtensionRunner>> {
        if let Some(runner) = self
            .session_runner
            .lock()
            .expect("session runner poisoned")
            .clone()
        {
            return Some(runner);
        }
        self.session()
            .extension_runner()
            .map(|runner| Arc::new(SessionRunnerAdapter { runner }) as Arc<dyn RuntimeExtensionRunner>)
    }
}

/// `class AgentSessionRuntime implements SubagentRuntimeHost`.
pub struct AgentSessionRuntime {
    session: Mutex<Arc<AgentSession>>,
    services: Mutex<Arc<AgentSessionServices>>,
    create_runtime: CreateAgentSessionRuntimeFactory,
    diagnostics: Mutex<Vec<AgentSessionRuntimeDiagnostic>>,
    model_fallback_message: Mutex<Option<String>>,
    session_config: Option<AgentSessionRuntimeConfig>,
    metadata: AgentSessionRuntimeMetadata,
    session_lease: Mutex<Option<Arc<Mutex<SessionLease>>>>,
    rebind_session: Mutex<Option<Arc<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>>>,
    session_replaced_listeners:
        Mutex<Vec<Arc<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>>>,
    runtime_env_scope: Mutex<Option<RuntimeEnvScope>>,
    before_session_invalidate: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    subagent_runtime_host: Mutex<Option<Arc<dyn SubagentRuntimeHost>>>,
    subagent_runtimes: Mutex<HashMap<String, Arc<AgentSessionRuntime>>>,
    dispose_promise: Mutex<Option<Shared<BoxFuture<Result<(), String>>>>>,
    /// Lets the `SubagentRuntimeHost` impl (which receives `&self`) recover the
    /// `Arc` that the session host holds.
    self_weak: Mutex<Weak<AgentSessionRuntime>>,
    /// `this.session.extensionRunner`, for the events this file emits.
    session_runner: Mutex<Option<Arc<dyn RuntimeExtensionRunner>>>,
}

impl AgentSessionRuntime {
    /// `constructor(session, services, createRuntime, diagnostics = [],
    /// modelFallbackMessage?, sessionConfig?, metadata = { kind: "top-level",
    /// createdAt: Date.now() }, sessionLease?)`.
    pub fn new(
        session: Arc<AgentSession>,
        services: Arc<AgentSessionServices>,
        create_runtime: CreateAgentSessionRuntimeFactory,
        diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
        model_fallback_message: Option<String>,
        session_config: Option<AgentSessionRuntimeConfig>,
        metadata: AgentSessionRuntimeMetadata,
        session_lease: Option<Arc<Mutex<SessionLease>>>,
    ) -> Arc<Self> {
        let runtime = Arc::new(Self {
            session: Mutex::new(session),
            services: Mutex::new(services),
            create_runtime,
            diagnostics: Mutex::new(diagnostics),
            model_fallback_message: Mutex::new(model_fallback_message),
            session_config,
            metadata,
            session_lease: Mutex::new(session_lease),
            rebind_session: Mutex::new(None),
            session_replaced_listeners: Mutex::new(Vec::new()),
            runtime_env_scope: Mutex::new(None),
            before_session_invalidate: Mutex::new(None),
            subagent_runtime_host: Mutex::new(None),
            subagent_runtimes: Mutex::new(HashMap::new()),
            dispose_promise: Mutex::new(None),
            self_weak: Mutex::new(Weak::new()),
            session_runner: Mutex::new(None),
        });
        *runtime.self_weak.lock().expect("self weak poisoned") = Arc::downgrade(&runtime);
        runtime.bind_runtime_host();
        runtime
    }

    fn self_arc(&self) -> Option<Arc<Self>> {
        self.self_weak
            .lock()
            .expect("self weak poisoned")
            .upgrade()
    }

    /// `get services()`.
    pub fn services(&self) -> Arc<AgentSessionServices> {
        self.services.lock().expect("services poisoned").clone()
    }

    /// `get session()`.
    pub fn session(&self) -> Arc<AgentSession> {
        self.session.lock().expect("session poisoned").clone()
    }

    /// `get cwd()`.
    pub fn cwd(&self) -> String {
        self.services().cwd.clone()
    }

    /// `get diagnostics()`.
    pub fn diagnostics(&self) -> Vec<AgentSessionRuntimeDiagnostic> {
        self.diagnostics.lock().expect("diagnostics poisoned").clone()
    }

    /// `get modelFallbackMessage()`.
    ///
    /// A restore warning must not keep claiming an old fallback after model selection.
    pub fn model_fallback_message(&self) -> Option<String> {
        let message = self
            .model_fallback_message
            .lock()
            .expect("model fallback message poisoned")
            .clone();
        let model = self.session().model();
        if is_obsolete_model_fallback_message(message.as_deref(), model.as_ref().map(|model| (model.provider.as_str(), model.id.as_str()))) {
            return None;
        }
        message
    }

    /// `get metadata()`.
    pub fn metadata(&self) -> AgentSessionRuntimeMetadata {
        self.metadata.clone()
    }

    /// `get runtimeConfig()`.
    pub fn runtime_config(&self) -> Option<AgentSessionRuntimeConfig> {
        self.session_config.clone()
    }

    /// `setRebindSession(rebindSession?)`.
    pub fn set_rebind_session(
        &self,
        rebind_session: Option<Arc<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>>,
    ) {
        *self.rebind_session.lock().expect("rebind session poisoned") = rebind_session;
    }

    /// `onSessionReplaced(listener)` - returns an unsubscribe function.
    pub fn on_session_replaced(
        &self,
        listener: Arc<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>,
    ) -> Arc<dyn Fn() + Send + Sync> {
        self.session_replaced_listeners
            .lock()
            .expect("session replaced listeners poisoned")
            .push(Arc::clone(&listener));
        let weak = self
            .self_weak
            .lock()
            .expect("self weak poisoned")
            .clone();
        Arc::new(move || {
            if let Some(runtime) = weak.upgrade() {
                let mut listeners = runtime
                    .session_replaced_listeners
                    .lock()
                    .expect("session replaced listeners poisoned");
                if let Some(index) = listeners
                    .iter()
                    .position(|candidate| Arc::ptr_eq(candidate, &listener))
                {
                    listeners.remove(index);
                }
            }
        })
    }

    /**
     * Host-installed scope wrapping every runtime rebuild (new/switch/fork/
     * import and subagent creation), during which extensions re-load. The
     * daemon uses it to apply the session's client env for load-time captures.
     */
    pub fn set_runtime_env_scope(&self, scope: Option<RuntimeEnvScope>) {
        *self.runtime_env_scope.lock().expect("runtime env scope poisoned") = scope;
    }

    /// `scopedBuild(fn)`.
    async fn scoped_build<F, Fut>(&self, build: F) -> Result<CreateAgentSessionRuntimeResult, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<CreateAgentSessionRuntimeResult, String>> + Send + 'static,
    {
        let scope = self
            .runtime_env_scope
            .lock()
            .expect("runtime env scope poisoned")
            .clone();
        match scope {
            Some(scope) => {
                let pending: BoxFuture<Result<CreateAgentSessionRuntimeResult, String>> =
                    Box::pin(build());
                scope(pending).await
            }
            None => build().await,
        }
    }

    /// `setSubagentRuntimeHost(host?)`.
    pub fn set_subagent_runtime_host(self: &Arc<Self>, host: Option<Arc<dyn SubagentRuntimeHost>>) {
        *self
            .subagent_runtime_host
            .lock()
            .expect("subagent runtime host poisoned") = host;
        self.bind_runtime_host();
    }

    /**
     * Set a synchronous callback that runs after `session_shutdown` handlers finish
     * but before the current session is invalidated.
     *
     * This is for host-owned UI teardown that must not yield to the event loop,
     * such as detaching extension-provided TUI components before the old extension
     * context becomes stale.
     */
    pub fn set_before_session_invalidate(&self, before_session_invalidate: Option<Arc<dyn Fn() + Send + Sync>>) {
        *self
            .before_session_invalidate
            .lock()
            .expect("before session invalidate poisoned") = before_session_invalidate;
    }

    /// `emitBeforeSwitch(reason, targetSessionFile?)`.
    async fn emit_before_switch(
        &self,
        reason: &str,
        target_session_file: Option<String>,
    ) -> Result<SessionBeforeEventResult, String> {
        let runner = self.session_extension_runner();
        let Some(runner) = runner else {
            return Ok(SessionBeforeEventResult { cancelled: false });
        };
        if !runner.has_handlers("session_before_switch") {
            return Ok(SessionBeforeEventResult { cancelled: false });
        }

        let result = runner
            .emit_event(
                serde_json::to_value(ExtensionEvent::SessionBeforeSwitch(SessionBeforeSwitchPayload {
                    reason: reason.to_string(),
                    target_session_file,
                }))
                .unwrap_or(Value::Null),
            )
            .await;
        Ok(SessionBeforeEventResult {
            cancelled: result
                .as_ref()
                .and_then(|result| result.get("cancel"))
                .and_then(Value::as_bool)
                == Some(true),
        })
    }

    /// `emitBeforeFork(entryId, options)`.
    async fn emit_before_fork(
        &self,
        entry_id: &str,
        position: &str,
    ) -> Result<SessionBeforeEventResult, String> {
        let runner = self.session_extension_runner();
        let Some(runner) = runner else {
            return Ok(SessionBeforeEventResult { cancelled: false });
        };
        if !runner.has_handlers("session_before_fork") {
            return Ok(SessionBeforeEventResult { cancelled: false });
        }

        let result = runner
            .emit_event(
                serde_json::to_value(ExtensionEvent::SessionBeforeFork(SessionBeforeForkPayload {
                    entry_id: entry_id.to_string(),
                    position: position.to_string(),
                }))
                .unwrap_or(Value::Null),
            )
            .await;
        Ok(SessionBeforeEventResult {
            cancelled: result
                .as_ref()
                .and_then(|result| result.get("cancel"))
                .and_then(Value::as_bool)
                == Some(true),
        })
    }

    /// `teardownCurrent(reason, targetSessionFile?)`.
    ///
    /// Child disposal can fail after an earlier child was irreversibly disposed.
    /// Finish old-session teardown in that case so the caller can commit the new
    /// candidate instead of retaining a partially destroyed old session.
    async fn teardown_current(
        self: &Arc<Self>,
        reason: &str,
        target_session_file: Option<String>,
    ) -> Result<(), String> {
        let child_dispose_error = self.dispose_hosted_subagent_runtimes().await.err();

        let runner = self.session_extension_runner();
        if let Some(runner) = runner {
            // `emitSessionShutdownEvent(runner, event)` emits only with handlers.
            if runner.has_handlers("session_shutdown") {
                runner
                    .emit_event(
                        serde_json::to_value(ExtensionEvent::SessionShutdown(SessionShutdownPayload {
                            reason: reason.to_string(),
                            target_session_file,
                        }))
                        .unwrap_or(Value::Null),
                    )
                    .await;
            }
        }
        if let Some(before_session_invalidate) = self
            .before_session_invalidate
            .lock()
            .expect("before session invalidate poisoned")
            .clone()
        {
            before_session_invalidate();
        }
        // Await the kernel's final snapshot flush before invalidating the session.
        self.session().dispose_async(None).await;
        match child_dispose_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `bindRuntimeHost()`.
    fn bind_runtime_host(self: &Arc<Self>) {
        let host = self
            .subagent_runtime_host
            .lock()
            .expect("subagent runtime host poisoned")
            .clone();
        // `this._session.setSubagentRuntimeHost(this.subagentRuntimeHost ?? this)`:
        // without an installed host the runtime hosts subagents itself, so the
        // session gets an adapter that forwards to this runtime.
        let session_host: Arc<dyn SubagentRuntimeHost> = host
            .unwrap_or_else(|| Arc::clone(self) as Arc<dyn SubagentRuntimeHost>);
        self.session().set_subagent_runtime_host(Some(session_host));
    }

    /// `apply(result)`.
    fn apply(self: &Arc<Self>, result: CreateAgentSessionRuntimeResult) {
        *self.session.lock().expect("session poisoned") = Arc::clone(&result.result.session);
        *self.services.lock().expect("services poisoned") = Arc::clone(&result.services);
        *self.diagnostics.lock().expect("diagnostics poisoned") = result.diagnostics;
        *self
            .model_fallback_message
            .lock()
            .expect("model fallback message poisoned") = result.result.model_fallback_message;
        self.bind_runtime_host();
    }

    /// `acquireReplacementLease(sessionPath)`.
    fn acquire_replacement_lease(
        &self,
        session_path: Option<&str>,
    ) -> Result<Option<Arc<Mutex<SessionLease>>>, String> {
        if let Some(session_path) = session_path {
            let existing = self
                .session_lease
                .lock()
                .expect("session lease poisoned")
                .clone();
            if let Some(existing) = existing {
                let matches = existing
                    .lock()
                    .expect("session lease poisoned")
                    .session_path
                    == canonical_session_path(session_path);
                if matches {
                    return Ok(Some(existing));
                }
            }
        }
        acquire_session_lease(session_path, &self.services().agent_dir, None)
            .map(|lease| lease.map(|lease| Arc::new(Mutex::new(lease))))
            .map_err(|error| error.to_string())
    }

    /// `releaseUncommittedLease(lease)`.
    fn release_uncommitted_lease(&self, lease: Option<Arc<Mutex<SessionLease>>>) {
        let current = self
            .session_lease
            .lock()
            .expect("session lease poisoned")
            .clone();
        let is_current = match (current, &lease) {
            (Some(current), Some(lease)) => Arc::ptr_eq(&current, lease),
            _ => false,
        };
        if !is_current {
            if let Some(lease) = lease {
                lease.lock().expect("session lease poisoned").release();
            }
        }
    }

    /// `releaseSessionLease()`.
    fn release_session_lease(&self) {
        let lease = self.session_lease.lock().expect("session lease poisoned").take();
        if let Some(lease) = lease {
            lease.lock().expect("session lease poisoned").release();
        }
    }

    /// `commitReplacementLease(lease)`.
    fn commit_replacement_lease(&self, lease: Option<Arc<Mutex<SessionLease>>>) {
        let mut slot = self.session_lease.lock().expect("session lease poisoned");
        let is_current = match (&*slot, &lease) {
            (Some(current), Some(lease)) => Arc::ptr_eq(current, lease),
            _ => false,
        };
        if is_current {
            return;
        }
        let previous = slot.take();
        *slot = lease;
        if let Some(previous) = previous {
            previous.lock().expect("session lease poisoned").release();
        }
    }

    /// `buildAndApplyReplacement(build, lease)`.
    ///
    /// An outer `Err` means the candidate build failed before any old-session
    /// effects. `Ok(Some(error))` means child cleanup failed after irreversible
    /// effects began: the candidate is committed, and the caller must finish the
    /// replacement before reporting that deferred cleanup error.
    async fn build_and_apply_replacement<F, Fut>(
        self: &Arc<Self>,
        build: F,
        lease: Option<Arc<Mutex<SessionLease>>>,
        reason: &str,
        target_session_file: Option<String>,
    ) -> Result<Option<String>, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<CreateAgentSessionRuntimeResult, String>>,
    {
        let result = match build().await {
            Ok(result) => result,
            Err(error) => {
                self.release_uncommitted_lease(lease);
                return Err(error);
            }
        };
        // Keep the working session installed until the candidate is complete.
        // Once child disposal begins, never roll back to its potentially partial
        // old session: finish teardown, apply the candidate, and commit its lease.
        let deferred_cleanup_error = self
            .teardown_for_replacement(reason, target_session_file)
            .await
            .err();
        self.apply(result);
        self.commit_replacement_lease(lease);
        Ok(deferred_cleanup_error)
    }

    /// `teardownForReplacement(reason, targetSessionFile)`.
    async fn teardown_for_replacement(
        self: &Arc<Self>,
        reason: &str,
        target_session_file: Option<String>,
    ) -> Result<(), String> {
        self.teardown_current(reason, target_session_file).await
    }

    /// `disposeSubagentRuntimes()`.
    ///
    /// Returns a boxed future: this function awaits `dispose`, which awaits
    /// `dispose_once`, which awaits `dispose_hosted_subagent_runtimes`, which awaits
    /// this function again. Boxing the recursive link breaks that layout/Send cycle
    /// so the spawned `BoxFuture` cast in `dispose` stays `Send`.
    fn dispose_subagent_runtimes(self: &Arc<Self>) -> BoxFuture<Result<(), String>> {
        let this = Arc::clone(self);
        Box::pin(async move { this.dispose_subagent_runtimes_inner().await })
    }

    async fn dispose_subagent_runtimes_inner(self: &Arc<Self>) -> Result<(), String> {
        let runtimes: Vec<Arc<AgentSessionRuntime>> = {
            let mut map = self
                .subagent_runtimes
                .lock()
                .expect("subagent runtimes poisoned");
            let runtimes: Vec<Arc<AgentSessionRuntime>> = map.values().cloned().collect();
            map.clear();
            runtimes
        };
        let mut dispose_error: Option<String> = None;
        for runtime in runtimes {
            if let Err(error) = runtime.dispose(None).await {
                if dispose_error.is_none() {
                    dispose_error = Some(error);
                }
            }
        }
        match dispose_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `disposeHostedSubagentRuntimes()`.
    async fn dispose_hosted_subagent_runtimes(self: &Arc<Self>) -> Result<(), String> {
        let mut dispose_error: Option<String> = None;
        let host = self
            .subagent_runtime_host
            .lock()
            .expect("subagent runtime host poisoned")
            .clone();
        if let Some(host) = host {
            if let Err(error) = host.dispose_rlm_subagent_runtimes().await {
                if dispose_error.is_none() {
                    dispose_error = Some(error);
                }
            }
        }
        if let Err(error) = self.dispose_subagent_runtimes().await {
            if dispose_error.is_none() {
                dispose_error = Some(error);
            }
        }
        match dispose_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `listSubagentRuntimes()`.
    pub fn list_subagent_runtimes(&self) -> Vec<Arc<AgentSessionRuntime>> {
        self.subagent_runtimes
            .lock()
            .expect("subagent runtimes poisoned")
            .values()
            .cloned()
            .collect()
    }
}

/// `createRlmSubagentRuntime`, `switchSession`, `newSession`, `fork`,
/// `importFromJsonl` and `dispose` - the rest of the class body.
impl AgentSessionRuntime {

    /// `createRlmSubagentRuntime(options)` body: the class returns the runtime
    /// itself (TypeScript casts `this` to `RlmSubagentRuntime`, whose only field
    /// is `session`), so the trait impl at the bottom of this file wraps it.
    async fn build_rlm_subagent_runtime(
        self: &Arc<Self>,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> Result<Arc<AgentSessionRuntime>, String> {
        // `SessionManager.create(options.parentSession.sessionManager.getCwd(), options.sessionDir)`.
        let parent_cwd = options
            .parent_session
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .get_cwd();
        let mut session_manager = SessionManager::create(&parent_cwd, Some(&options.session_dir))?;
        let parent_session_file = options
            .parent_session
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file();
        if let Some(parent_session_file) = parent_session_file {
            session_manager.new_session(Some(&NewSessionOptions {
                id: None,
                parent_session: Some(parent_session_file),
                rlm_depth: Some(options.rlm_depth as i64),
            }))?;
        }
        let session_manager = Arc::new(Mutex::new(session_manager));
        let cwd = session_manager.lock().expect("session manager poisoned").get_cwd();
        let agent_dir = self.services().agent_dir.clone();
        let session_config = self.session_config.clone();
        let runtime = self
            .scoped_build(|| {
                let create_runtime = Arc::clone(&self.create_runtime);
                let session_manager = Arc::clone(&session_manager);
                let input = CreateAgentSessionRuntimeInput {
                    cwd,
                    agent_dir,
                    session_manager,
                    session_start_event: Some(serde_json::json!({
                        "type": "session_start",
                        "reason": "startup",
                    })),
                    session_config,
                    session_options: Some(AgentSessionCreationOptions {
                        model: Some(options.model.clone()),
                        thinking_level: Some(options.thinking_level.clone()),
                        service_tier: Some(options.service_tier.clone()),
                        scoped_models: Some(
                            options
                                .scoped_models
                                .iter()
                                .map(|scoped| ScopedModel {
                                    model: scoped.model.clone(),
                                    thinking_level: scoped.thinking_level.clone(),
                                })
                                .collect(),
                        ),
                        initial_active_tool_names: Some(options.active_tool_names.clone()),
                        allowed_tool_names: options.allowed_tool_names.clone(),
                        custom_tools: Some(options.custom_tools.clone()),
                        include_goals: Some(options.include_goals),
                        include_compact_skill: Some(options.include_compact_skill),
                        rlm_depth: Some(options.rlm_depth as i64),
                        rlm_max_depth: Some(options.rlm_max_depth as i64),
                        rlm_session_dir: Some(options.session_dir.clone()),
                        rlm_parent_node_id: Some(options.rlm_parent_node_id.clone()),
                        rlm_parent_agent: Some(
                            options
                                .parent_session
                                .session_name()
                                .unwrap_or_else(|| options.parent_session.session_id()),
                        ),
                        semantic_parent_session_id: Some(options.parent_session.session_id()),
                        semantic_spawned_by_request_id: options.spawned_by_request_id.clone(),
                        ..Default::default()
                    }),
                    runtime_metadata: Some(AgentSessionRuntimeMetadata {
                        kind: AGENT_SESSION_RUNTIME_KIND_SUBAGENT.to_string(),
                        created_at: now_millis(),
                        parent_active_session_id: None,
                        parent_session_id: Some(options.parent_session.session_id()),
                        parent_session_file: options
                            .parent_session
                            .session_file(),
                        rlm_child_id: Some(options.id.clone()),
                        rlm_parent_node_id: Some(options.rlm_parent_node_id.clone()),
                        rehydrated_completed: None,
                        prompt: Some(options.prompt.clone()),
                        spawn_code: options.spawn_code.clone(),
                        session_dir: Some(options.session_dir.clone()),
                    }),
                    session_lease: None,
                };
                async move { create_runtime(input).await }
            })
            .await?;
        let runtime = AgentSessionRuntime::new(
            runtime.result.session,
            runtime.services,
            Arc::clone(&self.create_runtime),
            runtime.diagnostics,
            runtime.result.model_fallback_message,
            self.session_config.clone(),
            AgentSessionRuntimeMetadata {
                kind: AGENT_SESSION_RUNTIME_KIND_SUBAGENT.to_string(),
                created_at: now_millis(),
                parent_session_id: Some(options.parent_session.session_id()),
                parent_session_file: options.parent_session.session_file(),
                rlm_child_id: Some(options.id.clone()),
                rlm_parent_node_id: Some(options.rlm_parent_node_id.clone()),
                prompt: Some(options.prompt.clone()),
                spawn_code: options.spawn_code.clone(),
                session_dir: Some(options.session_dir.clone()),
                ..Default::default()
            },
            None,
        );
        self.subagent_runtimes
            .lock()
            .expect("subagent runtimes poisoned")
            .insert(options.id.clone(), Arc::clone(&runtime));
        let outcome = self
            .finish_subagent_startup(&runtime, &options)
            .await;
        if let Err(error) = outcome {
            self.subagent_runtimes
                .lock()
                .expect("subagent runtimes poisoned")
                .remove(&options.id);
            let _ = runtime.dispose(None).await;
            return Err(error);
        }
        Ok(runtime)
    }

    /// The post-build startup steps of `createRlmSubagentRuntime`.
    async fn finish_subagent_startup(
        &self,
        runtime: &Arc<AgentSessionRuntime>,
        options: &CreateRlmSubagentRuntimeOptions,
    ) -> Result<(), String> {
        let session = runtime.session();
        session
            .bind_extensions(&ExtensionBindings {
                ui_context: None,
                command_context_actions: None,
                shutdown_handler: None,
                on_error: None,
            })
            .await?;
        let run_status = session.get_rlm_child_run_status(&options.id);
        if run_status.as_deref() == Some(RLM_CHILD_AGENT_STATUS_CANCELLED) {
            return Err("RLM subagent startup was cancelled".to_string());
        }
        if session.session_name().as_deref() != Some(options.session_name.as_str()) {
            session.set_session_name(&options.session_name)?;
        }
        if let Some(on_session_published) = &options.on_session_published {
            on_session_published(&session);
        }
        Ok(())
    }

    /// `deleteRlmSubagentRuntime(childId, session?)`.
    async fn dispose_subagent_runtime(
        &self,
        child_id: &str,
        session: Option<&Arc<AgentSession>>,
    ) -> Result<(), String> {
        let runtime = self
            .subagent_runtimes
            .lock()
            .expect("subagent runtimes poisoned")
            .get(child_id)
            .cloned();
        let Some(runtime) = runtime else {
            if let Some(session) = session {
                session.dispose_async(None).await;
            }
            return Ok(());
        };
        self.subagent_runtimes
            .lock()
            .expect("subagent runtimes poisoned")
            .remove(child_id);
        let should_dispose_stale_session = session
            .map(|session| !Arc::ptr_eq(&runtime.session(), session))
            .unwrap_or(false);
        let result = runtime.dispose(None).await;
        if should_dispose_stale_session {
            if let Some(session) = session {
                session.dispose_async(None).await;
            }
        }
        result
    }

    /// `finishSessionReplacement(withSession?)`.
    async fn finish_session_replacement(&self, with_session: Option<WithSessionCallback>) -> Result<(), String> {
        let rebind_session = self.rebind_session.lock().expect("rebind session poisoned").clone();
        if let Some(rebind_session) = rebind_session {
            rebind_session(self.session()).await;
        }
        // Match JavaScript Set iteration: deleting the current or an earlier entry
        // does not skip the next one, deleting a later entry suppresses it, and an
        // entry added during notification is visited once in this same dispatch.
        let mut visited: Vec<std::sync::Weak<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>> = Vec::new();
        loop {
            let listener = self
                .session_replaced_listeners
                .lock()
                .expect("session replaced listeners poisoned")
                .iter()
                .find(|candidate| {
                    !visited.iter().any(|seen| {
                        seen.upgrade().is_some_and(|seen| Arc::ptr_eq(&seen, candidate))
                    })
                })
                .cloned();
            let Some(listener) = listener else { break; };
            visited.push(Arc::downgrade(&listener));
            listener(self.session()).await;
        }
        if let Some(with_session) = with_session {
            with_session(self.session().create_replaced_session_context()).await;
        }
        Ok(())
    }

    /// `switchSession(sessionPath, options?)`.
    pub async fn switch_session(
        self: &Arc<Self>,
        session_path: &str,
        options: Option<SwitchSessionOptions>,
    ) -> Result<SessionBeforeEventResult, String> {
        let before_result = self
            .emit_before_switch("resume", Some(session_path.to_string()))
            .await?;
        if before_result.cancelled {
            return Ok(before_result);
        }

        let previous_session_file = self.session().session_file();
        let lease = self.acquire_replacement_lease(Some(session_path))?;
        let opened = Self::open_session_manager(
            session_path,
            None,
            options.as_ref().and_then(|options| options.cwd_override.as_deref()),
        );
        let session_manager = match opened {
            Ok(session_manager) => session_manager,
            Err(error) => {
                self.release_uncommitted_lease(lease);
                return Err(error);
            }
        };
        let session_manager_file = session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file();
        let cwd_ok = assert_session_cwd_exists(
            &SessionManagerCwdSource { session_manager: &session_manager },
            &self.cwd(),
        );
        if let Err(error) = cwd_ok {
            self.release_uncommitted_lease(lease);
            return Err(error.to_string());
        }
        let agent_dir = self.services().agent_dir.clone();
        let session_config = self.session_config.clone();
        let cwd = session_manager.lock().expect("session manager poisoned").get_cwd();
        let previous_for_event = previous_session_file.clone();
        let deferred_cleanup_error = self
            .build_and_apply_replacement(
                {
                    let this = Arc::clone(self);
                    move || {
                        let this = Arc::clone(&this);
                        let session_manager = Arc::clone(&session_manager);
                        async move {
                            this.clone().scoped_build(move || {
                                let create_runtime = Arc::clone(&this.create_runtime);
                                let session_manager = Arc::clone(&session_manager);
                                async move {
                                    create_runtime(CreateAgentSessionRuntimeInput {
                                        cwd,
                                        agent_dir,
                                        session_manager,
                                        session_start_event: Some(serde_json::json!({
                                            "type": "session_start",
                                            "reason": "resume",
                                            "previousSessionFile": previous_for_event,
                                        })),
                                        session_config,
                                        session_options: None,
                                        runtime_metadata: None,
                                        session_lease: None,
                                    })
                                    .await
                                }
                            })
                            .await
                        }
                    }
                },
                lease,
                "resume",
                session_manager_file,
            )
            .await?;
        self.finish_session_replacement(options.and_then(|options| options.with_session))
            .await?;
        if let Some(error) = deferred_cleanup_error {
            return Err(error);
        }
        Ok(SessionBeforeEventResult { cancelled: false })
    }

    /// `newSession(options?)`.
    pub async fn new_session(
        self: &Arc<Self>,
        options: Option<NewSessionOptionsInput>,
    ) -> Result<SessionBeforeEventResult, String> {
        let before_result = self.emit_before_switch("new", None).await?;
        if before_result.cancelled {
            return Ok(before_result);
        }

        let previous_session_file = self.session().session_file();
        let session_dir = self
            .session()
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_dir();
        let mut new_manager = SessionManager::create(&self.cwd(), Some(&session_dir))?;
        if let Some(parent_session) = options.as_ref().and_then(|options| options.parent_session.clone()) {
            let parent = self.session();
            let rlm_depth = parent
                .session_manager
                .lock()
                .expect("session manager poisoned")
                .get_header()
                .and_then(|header| header.get("rlmDepth").and_then(Value::as_i64))
                .unwrap_or_else(|| parent.rlm_depth());
            new_manager.new_session(Some(&NewSessionOptions {
                id: None,
                parent_session: Some(parent_session),
                rlm_depth: Some(rlm_depth),
            }))?;
        }
        let session_manager = Arc::new(Mutex::new(new_manager));
        let session_manager_file = session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file();
        let lease = self.acquire_replacement_lease(session_manager_file.as_deref())?;

        let new_session_file = session_manager_file.clone();
        let agent_dir = self.services().agent_dir.clone();
        let session_config = self.session_config.clone();
        let cwd = self.cwd();
        let deferred_cleanup_error = self
            .build_and_apply_replacement(
                {
                    let this = Arc::clone(self);
                    move || {
                        let this = Arc::clone(&this);
                        let session_manager = Arc::clone(&session_manager);
                        async move {
                            this.clone().scoped_build(move || {
                                let create_runtime = Arc::clone(&this.create_runtime);
                                let session_manager = Arc::clone(&session_manager);
                                async move {
                                    create_runtime(CreateAgentSessionRuntimeInput {
                                        cwd,
                                        agent_dir,
                                        session_manager,
                                        session_start_event: Some(serde_json::json!({
                                            "type": "session_start",
                                            "reason": "new",
                                            "previousSessionFile": previous_session_file,
                                        })),
                                        session_config,
                                        session_options: None,
                                        runtime_metadata: None,
                                        session_lease: None,
                                    })
                                    .await
                                }
                            })
                            .await
                        }
                    }
                },
                lease,
                "new",
                new_session_file,
            )
            .await?;
        if let Some(setup) = options.as_ref().and_then(|options| options.setup.clone()) {
            let session = self.session();
            setup(Arc::clone(&session.session_manager)).await;
            let messages = session
                .session_manager
                .lock()
                .expect("session manager poisoned")
                .build_session_context(None)
                .messages;
            let mut state = session.agent.state();
            state.messages = messages;
            session.agent.set_state(state);
        }
        self.finish_session_replacement(options.and_then(|options| options.with_session))
            .await?;
        if let Some(error) = deferred_cleanup_error {
            return Err(error);
        }
        Ok(SessionBeforeEventResult { cancelled: false })
    }

    /// `fork(entryId, options?)`.
    pub async fn fork(
        self: &Arc<Self>,
        entry_id: &str,
        options: Option<ForkOptionsInput>,
    ) -> Result<SessionReplacementResult, String> {
        let position = options
            .as_ref()
            .and_then(|options| options.position.clone())
            .unwrap_or_else(|| "before".to_string());
        let before_result = self.emit_before_fork(entry_id, &position).await?;
        if before_result.cancelled {
            return Ok(SessionReplacementResult {
                cancelled: true,
                selected_text: None,
            });
        }

        let selected_entry = self
            .session()
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .get_entry(entry_id);
        let Some(selected_entry) = selected_entry else {
            return Err("Invalid entry ID for forking".to_string());
        };

        let (target_leaf_id, selected_text) = if position == "at" {
            (
                selected_entry
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                None,
            )
        } else {
            let entry_type = selected_entry.get("type").and_then(Value::as_str);
            let role = selected_entry
                .get("message")
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str);
            if entry_type != Some("message") || role != Some("user") {
                return Err("Invalid entry ID for forking".to_string());
            }
            let text = selected_entry
                .get("message")
                .and_then(|message| message.get("content"))
                .map(extract_user_message_text);
            (
                selected_entry
                    .get("parentId")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                text,
            )
        };

        let previous_session_file = self.session().session_file();
        if self
            .session()
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .is_persisted()
        {
            let current_session_file = self.session().session_file();
            let Some(current_session_file) = current_session_file else {
                return Err("Persisted session is missing a session file".to_string());
            };
            let session_dir = self
                .session()
                .session_manager
                .lock()
                .expect("session manager poisoned")
                .get_session_dir();
            if target_leaf_id.is_none() {
                let source_header = self
                    .session()
                    .session_manager
                    .lock()
                    .expect("session manager poisoned")
                    .get_header();
                let mut new_manager = SessionManager::create(&self.cwd(), Some(&session_dir))?;
                let rlm_depth = source_header
                    .as_ref()
                    .and_then(|header| header.get("rlmDepth").and_then(Value::as_i64))
                    .unwrap_or_else(|| self.session().rlm_depth());
                new_manager.new_session(Some(&NewSessionOptions {
                    id: None,
                    parent_session: Some(current_session_file),
                    rlm_depth: Some(rlm_depth),
                }))?;
                let session_manager = Arc::new(Mutex::new(new_manager));
                let file = session_manager
                    .lock()
                    .expect("session manager poisoned")
                    .get_session_file();
                let lease = self.acquire_replacement_lease(file.as_deref())?;
                self.rebuild_after_fork(
                    session_manager,
                    "fork",
                    previous_session_file,
                    lease,
                    options.and_then(|options| options.with_session),
                )
                .await?;
                return Ok(SessionReplacementResult {
                    cancelled: false,
                    selected_text,
                });
            }

            let target_leaf_id = target_leaf_id.expect("target leaf checked above");
            let source_manager = Self::open_session_manager(
                &current_session_file,
                Some(&session_dir),
                None,
            )?;
            let forked_session_path = {
                let mut manager = source_manager.lock().expect("session manager poisoned");
                manager.create_branched_session(&target_leaf_id)?
            };
            let Some(forked_session_path) = forked_session_path else {
                return Err("Failed to create forked session".to_string());
            };
            let session_manager =
                Self::open_session_manager(&forked_session_path, Some(&session_dir), None)?;
            let file = session_manager
                .lock()
                .expect("session manager poisoned")
                .get_session_file();
            let lease = self.acquire_replacement_lease(file.as_deref())?;
            self.rebuild_after_fork(
                session_manager,
                "fork",
                previous_session_file,
                lease,
                options.and_then(|options| options.with_session),
            )
            .await?;
            return Ok(SessionReplacementResult {
                cancelled: false,
                selected_text,
            });
        }

        // Non-persisted: the current `SessionManager` is reused, exactly like the
        // TypeScript `const sessionManager = this.session.sessionManager;`.
        let session_manager = {
            let current = Arc::clone(&self.session().session_manager);
            let staged = current
                .lock()
                .expect("session manager poisoned")
                .clone_for_nonpersisted_replacement();
            Arc::new(Mutex::new(staged))
        };
        match &target_leaf_id {
            None => {
                let source_header = session_manager
                    .lock()
                    .expect("session manager poisoned")
                    .get_header();
                let rlm_depth = source_header
                    .as_ref()
                    .and_then(|header| header.get("rlmDepth").and_then(Value::as_i64))
                    .unwrap_or_else(|| self.session().rlm_depth());
                let parent_file = self.session().session_file();
                let mut manager = session_manager.lock().expect("session manager poisoned");
                manager.new_session(Some(&NewSessionOptions {
                    id: None,
                    parent_session: parent_file,
                    rlm_depth: Some(rlm_depth),
                }))?;
            }
            Some(target_leaf_id) => {
                let mut manager = session_manager.lock().expect("session manager poisoned");
                manager.create_branched_session(target_leaf_id)?;
            }
        }
        let file = session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file();
        let lease = self.acquire_replacement_lease(file.as_deref())?;
        self.rebuild_after_fork(
            session_manager,
            "fork",
            previous_session_file,
            lease,
            options.and_then(|options| options.with_session),
        )
        .await?;
        Ok(SessionReplacementResult {
            cancelled: false,
            selected_text,
        })
    }

    /// The shared `buildAndApplyReplacement` + `finishSessionReplacement` tail of
    /// the three `fork` branches (same `session_start` event in each).
    async fn rebuild_after_fork(
        self: &Arc<Self>,
        session_manager: Arc<Mutex<SessionManager>>,
        reason: &str,
        previous_session_file: Option<String>,
        lease: Option<Arc<Mutex<SessionLease>>>,
        with_session: Option<WithSessionCallback>,
    ) -> Result<(), String> {
        let file = session_manager.lock().expect("session manager poisoned").get_session_file();
        let agent_dir = self.services().agent_dir.clone();
        let session_config = self.session_config.clone();
        let cwd = session_manager.lock().expect("session manager poisoned").get_cwd();
        let reason = reason.to_string();
        let teardown_reason = reason.clone();
        let deferred_cleanup_error = self.build_and_apply_replacement(
            {
                let this = Arc::clone(self);
                move || {
                    let this = Arc::clone(&this);
                    let session_manager = Arc::clone(&session_manager);
                    async move {
                        this.clone().scoped_build(move || {
                            let create_runtime = Arc::clone(&this.create_runtime);
                            let session_manager = Arc::clone(&session_manager);
                            async move {
                                create_runtime(CreateAgentSessionRuntimeInput {
                                    cwd,
                                    agent_dir,
                                    session_manager,
                                    session_start_event: Some(serde_json::json!({
                                        "type": "session_start",
                                        "reason": reason.clone(),
                                        "previousSessionFile": previous_session_file,
                                    })),
                                    session_config,
                                    session_options: None,
                                    runtime_metadata: None,
                                    session_lease: None,
                                })
                                .await
                            }
                        })
                        .await
                    }
                }
            },
            lease,
            &teardown_reason,
            file,
        )
        .await?;
        self.finish_session_replacement(with_session).await?;
        match deferred_cleanup_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `importFromJsonl(inputPath, cwdOverride?)`.
    pub async fn import_from_jsonl(
        self: &Arc<Self>,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> Result<SessionBeforeEventResult, String> {
        let resolved_path = resolve_path(input_path);
        if !std::path::Path::new(&resolved_path).exists() {
            return Err(SessionImportFileNotFoundError::new(resolved_path).to_string());
        }

        let session_dir = self
            .session()
            .session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_dir();
        if !std::path::Path::new(&session_dir).exists() {
            std::fs::create_dir_all(&session_dir).map_err(|error| error.to_string())?;
        }

        let destination_path = join_path(
            &session_dir,
            std::path::Path::new(&resolved_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
                .as_str(),
        );
        let before_result = self
            .emit_before_switch("resume", Some(destination_path.clone()))
            .await?;
        if before_result.cancelled {
            return Ok(before_result);
        }

        let previous_session_file = self.session().session_file();
        let lease = self.acquire_replacement_lease(Some(&destination_path))?;
        if resolve_path(&destination_path) != resolved_path {
            if let Err(error) = std::fs::copy(&resolved_path, &destination_path) {
                self.release_uncommitted_lease(lease);
                return Err(error.to_string());
            }
        }
        let session_manager = match Self::open_session_manager(
            &destination_path,
            Some(&session_dir),
            cwd_override,
        ) {
            Ok(session_manager) => session_manager,
            Err(error) => {
                self.release_uncommitted_lease(lease);
                return Err(error);
            }
        };
        let cwd_ok = assert_session_cwd_exists(
            &SessionManagerCwdSource { session_manager: &session_manager },
            &self.cwd(),
        );
        if let Err(error) = cwd_ok {
            self.release_uncommitted_lease(lease);
            return Err(error.to_string());
        }
        let file = session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file();
        let agent_dir = self.services().agent_dir.clone();
        let session_config = self.session_config.clone();
        let cwd = session_manager.lock().expect("session manager poisoned").get_cwd();
        let deferred_cleanup_error = self.build_and_apply_replacement(
            {
                let this = Arc::clone(self);
                move || {
                    let this = Arc::clone(&this);
                    let session_manager = Arc::clone(&session_manager);
                    async move {
                        this.clone().scoped_build(move || {
                            let create_runtime = Arc::clone(&this.create_runtime);
                            let session_manager = Arc::clone(&session_manager);
                            async move {
                                create_runtime(CreateAgentSessionRuntimeInput {
                                    cwd,
                                    agent_dir,
                                    session_manager,
                                    session_start_event: Some(serde_json::json!({
                                        "type": "session_start",
                                        "reason": "resume",
                                        "previousSessionFile": previous_session_file,
                                    })),
                                    session_config,
                                    session_options: None,
                                    runtime_metadata: None,
                                    session_lease: None,
                                })
                                .await
                            }
                        })
                        .await
                    }
                }
            },
            lease,
            "resume",
            file,
        )
        .await?;
        // `finishSessionReplacement()` with no `withSession`.
        self.finish_session_replacement(None).await?;
        if let Some(error) = deferred_cleanup_error {
            return Err(error);
        }
        Ok(SessionBeforeEventResult { cancelled: false })
    }

    /// `disposeOnce(options)`.
    async fn dispose_once(self: &Arc<Self>, options: AgentSessionRuntimeDisposeOptions) -> Result<(), String> {
        let mut dispose_error: Option<String> = None;

        // `await emitSessionShutdownEvent(this.session.extensionRunner, {...})`.
        // The emit leg cannot reject in the port (the seam returns no `Result`),
        // so the TypeScript `catch` around it has nothing to collect.
        let runner = self.session_extension_runner();
        if let Some(runner) = runner {
            if runner.has_handlers("session_shutdown") {
                runner
                    .emit_event(serde_json::json!({
                        "type": "session_shutdown",
                        "reason": "quit",
                    }))
                    .await;
            }
        }

        if let Some(before_session_invalidate) = self
            .before_session_invalidate
            .lock()
            .expect("before session invalidate poisoned")
            .clone()
        {
            before_session_invalidate();
        }
        // Await the kernel's final snapshot flush before tearing the session down.
        self.session()
            .dispose_async(Some(options.kernel_snapshot.unwrap_or(true)))
            .await;
        if let Err(error) = self.dispose_hosted_subagent_runtimes().await {
            if dispose_error.is_none() {
                dispose_error = Some(error);
            }
        }

        self.release_session_lease();
        match dispose_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `dispose(options?)`.
    ///
    /// `if (!this.disposePromise) this.disposePromise = this.disposeOnce(options ?? {});`
    /// then `await this.disposePromise`. Every caller shares one run.
    pub async fn dispose(
        self: &Arc<Self>,
        options: Option<AgentSessionRuntimeDisposeOptions>,
    ) -> Result<(), String> {
        let shared = {
            let mut slot = self.dispose_promise.lock().expect("dispose promise poisoned");
            match slot.clone() {
                Some(shared) => shared,
                None => {
                    let this = Arc::clone(self);
                    let options = options.unwrap_or_default();
                    let future: BoxFuture<Result<(), String>> = Box::pin(async move { this.dispose_once(options).await });
                    let shared = future.shared();
                    *slot = Some(shared.clone());
                    shared
                }
            }
        };
        shared.await
    }

    /// `SessionManager.open(path, sessionDir, cwdOverride)`.
    fn open_session_manager(
        path: &str,
        session_dir: Option<&str>,
        cwd_override: Option<&str>,
    ) -> Result<Arc<Mutex<SessionManager>>, String> {
        SessionManager::open(path, session_dir, cwd_override).map(|manager| Arc::new(Mutex::new(manager)))
    }
}

/// `SessionCwdSource` view of `SessionManager` (`assertSessionCwdExists` input).
struct SessionManagerCwdSource<'a> {
    session_manager: &'a Arc<Mutex<SessionManager>>,
}

impl crate::core::session_cwd::SessionCwdSource for SessionManagerCwdSource<'_> {
    fn get_cwd(&self) -> String {
        self.session_manager
            .lock()
            .expect("session manager poisoned")
            .get_cwd()
    }

    fn get_session_file(&self) -> Option<String> {
        self.session_manager
            .lock()
            .expect("session manager poisoned")
            .get_session_file()
    }
}

/// `switchSession` options (`{ cwdOverride?, withSession? }`).
#[derive(Clone, Default)]
pub struct SwitchSessionOptions {
    pub cwd_override: Option<String>,
    pub with_session: Option<WithSessionCallback>,
}

/// `newSession` options (`{ parentSession?, setup?, withSession? }`).
#[derive(Clone, Default)]
pub struct NewSessionOptionsInput {
    pub parent_session: Option<String>,
    pub setup: Option<NewSessionSetupCallback>,
    pub with_session: Option<WithSessionCallback>,
}

/// `fork` options (`{ position?, withSession? }`).
#[derive(Clone, Default)]
pub struct ForkOptionsInput {
    pub position: Option<String>,
    pub with_session: Option<WithSessionCallback>,
}

/// `path.resolve(value)` (`agent-session-runtime.ts:2` imports `resolve` from
/// `node:path`; `importFromJsonl` calls it at `agent-session-runtime.ts:190`).
///
/// Node's win32 `isAbsolute` treats a bare rooted path as absolute
/// (`path.isAbsolute("/tmp/x.jsonl") === true`, measured on Node v24.16.0), so
/// `resolve` returns it unchanged. `std::path::Path::is_absolute` needs a drive
/// or UNC prefix and reports false, and `cwd.join(...)` then prefixes the
/// current drive: `"/tmp/x.jsonl"` became `"C:/tmp/x.jsonl"`.
fn resolve_path(value: &str) -> String {
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        let bytes = value.as_bytes();
        let drive_absolute = bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/');
        if !value.is_empty() && (drive_absolute || bytes[0] == b'\\' || bytes[0] == b'/') {
            return value.to_string();
        }
    } else if std::path::Path::new(value).is_absolute() {
        return value.to_string();
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    std::fs::canonicalize(cwd.join(value))
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| cwd.join(value).to_string_lossy().to_string())
}

/// `join(agentDir, name)`.
fn join_path(base: &str, name: &str) -> String {
    std::path::Path::new(base)
        .join(name)
        .to_string_lossy()
        .to_string()
}

/// `export async function createAgentSessionRuntime(createRuntime, options)`.
pub async fn create_agent_session_runtime(
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
        None => {
            let path = session_manager
                .lock()
                .expect("session manager poisoned")
                .get_session_file();
            match acquire_session_lease(path.as_deref(), &agent_dir, None) {
                Ok(lease) => lease.map(|lease| Arc::new(Mutex::new(lease))),
                Err(error) => return Err(error.to_string()),
            }
        }
    };
    let lease_on_error = lease.clone();
    let result = async {
        assert_session_cwd_exists(
            &SessionManagerCwdSource { session_manager: &session_manager },
            &cwd,
        )
        .map_err(|error| error.to_string())?;
        create_runtime(CreateAgentSessionRuntimeInput {
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
    }
    .await;

    match result {
        Ok(result) => {
            let session = Arc::clone(&result.result.session);
            Ok(AgentSessionRuntime::new(
                session,
                result.services,
                create_runtime,
                result.diagnostics,
                result.result.model_fallback_message.clone(),
                session_config,
                runtime_metadata.unwrap_or_else(AgentSessionRuntimeMetadata::top_level),
                lease,
            ))
        }
        Err(error) => {
            if let Some(lease) = lease_on_error {
                lease.lock().expect("session lease poisoned").release();
            }
            Err(error)
        }
    }
}

/// `class AgentSessionRuntime implements SubagentRuntimeHost`.
impl SubagentRuntimeHost for AgentSessionRuntime {
    fn create_rlm_subagent_runtime(
        &self,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> BoxFuture<Result<RlmSubagentRuntime, String>> {
        let runtime = self.self_arc();
        Box::pin(async move {
            let Some(runtime) = runtime else {
                return Err("AgentSessionRuntime has been dropped".to_string());
            };
            let created = runtime.build_rlm_subagent_runtime(options).await?;
            Ok(RlmSubagentRuntime {
                session: created.session(),
            })
        })
    }

    /// `deleteRlmSubagentRuntime(childId, session?)`.
    fn delete_rlm_subagent_runtime(
        &self,
        child_id: &str,
        session: Option<&Arc<AgentSession>>,
    ) -> BoxFuture<Result<(), String>> {
        let runtime = self.self_arc();
        let child_id = child_id.to_string();
        let session = session.cloned();
        Box::pin(async move {
            let runtime = runtime.ok_or_else(|| "AgentSessionRuntime has been dropped".to_string())?;
            runtime.dispose_subagent_runtime(&child_id, session.as_ref()).await
        })
    }

    /// `disposeRlmSubagentRuntimes?()` - `disposeSubagentRuntimes()` in the class.
    fn dispose_rlm_subagent_runtimes(&self) -> BoxFuture<Result<(), String>> {
        let runtime = self.self_arc();
        Box::pin(async move {
            let runtime = runtime.ok_or_else(|| "AgentSessionRuntime has been dropped".to_string())?;
            runtime.dispose_subagent_runtimes().await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_user_message_text_joins_text_parts() {
        assert_eq!(extract_user_message_text(&serde_json::json!("hello")), "hello");
        assert_eq!(
            extract_user_message_text(&serde_json::json!([
                { "type": "text", "text": "a" },
                { "type": "image", "data": "x" },
                { "type": "text", "text": "b" },
            ])),
            "ab"
        );
    }

    #[test]
    fn extract_user_message_text_ignores_partless_content() {
        assert_eq!(extract_user_message_text(&serde_json::json!({ "type": "text" })), "");
    }

    #[test]
    fn top_level_metadata_defaults_to_the_top_level_kind() {
        let metadata = AgentSessionRuntimeMetadata::top_level();
        assert_eq!(metadata.kind, AGENT_SESSION_RUNTIME_KIND_TOP_LEVEL);
        assert!(metadata.created_at > 0.0);
        assert!(metadata.rlm_child_id.is_none());
    }

    #[test]
    fn session_before_event_result_serializes_cancelled() {
        let value = serde_json::to_value(SessionBeforeEventResult { cancelled: true }).unwrap();
        assert_eq!(value, serde_json::json!({ "cancelled": true }));
    }

    #[test]
    fn metadata_uses_camel_case_wire_names() {
        let value = serde_json::to_value(AgentSessionRuntimeMetadata {
            kind: AGENT_SESSION_RUNTIME_KIND_SUBAGENT.to_string(),
            created_at: 1.0,
            parent_session_id: Some("parent".to_string()),
            rlm_child_id: Some("child".to_string()),
            session_dir: Some("/tmp".to_string()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(value["createdAt"], serde_json::json!(1.0));
        assert_eq!(value["parentSessionId"], serde_json::json!("parent"));
        assert_eq!(value["rlmChildId"], serde_json::json!("child"));
        assert!(value.get("parentSessionFile").is_none());
    }

    #[test]
    fn join_path_puts_the_name_under_the_base() {
        assert_eq!(
            join_path("/tmp/sessions", "a.jsonl"),
            std::path::Path::new("/tmp/sessions")
                .join("a.jsonl")
                .to_string_lossy()
                .to_string()
        );
    }

    #[test]
    fn resolve_path_keeps_absolute_paths() {
        let absolute = std::path::Path::new("/tmp/x.jsonl");
        assert_eq!(resolve_path("/tmp/x.jsonl"), absolute.to_string_lossy().to_string());
    }

    #[test]
    fn session_before_event_result_default_is_not_cancelled() {
        assert!(!SessionBeforeEventResult::default().cancelled);
    }
}
