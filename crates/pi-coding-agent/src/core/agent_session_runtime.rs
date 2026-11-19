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
        #[cfg(test)]
        if target_session_file
            .as_deref()
            .is_some_and(import_teardown_fault)
        {
            return Err("synthetic import teardown fault before old-session disposal".to_string());
        }
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
            // The selected entry came from the live manager, so its accepted
            // buffered bytes must reach disk before the fork re-opens the file.
            self.session()
                .session_manager
                .lock()
                .expect("session manager poisoned")
                .flush_now()?;
            let source_manager = Self::open_session_manager(
                &current_session_file,
                Some(&session_dir),
                None,
            )?;
            let forked_session_path = {
                let mut manager = source_manager.lock().expect("session manager poisoned");
                let forked = manager.create_branched_session(&target_leaf_id)?;
                if forked.is_some() {
                    // A user-only branch otherwise follows the lazy first-
                    // assistant rule and names a file that does not exist yet.
                    // Persist the accepted source entries before reopening it.
                    manager.flush_now()?;
                }
                forked
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

        // Always validate and build against a side file. In particular, an
        // import whose basename names the live (or another existing) session
        // must not truncate that destination before every fallible check wins.
        let staged_path = format!(
            "{destination_path}.import-{}.{}.tmp",
            std::process::id(),
            uuid::Uuid::new_v4()
        );
        let mut staged = StagedImportFile { path: staged_path, published: false };
        if let Err(error) = std::fs::copy(&resolved_path, &staged.path) {
            self.release_uncommitted_lease(lease);
            return Err(error.to_string());
        }
        let session_manager = match Self::open_session_manager(
            &staged.path,
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
            &ImportSessionCwdSource {
                session_manager: &session_manager,
                destination_path: &destination_path,
            },
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
        let create_runtime = Arc::clone(&self.create_runtime);
        let candidate_manager = Arc::clone(&session_manager);
        let built = self
            .scoped_build(move || async move {
                create_runtime(CreateAgentSessionRuntimeInput {
                    cwd,
                    agent_dir,
                    session_manager: candidate_manager,
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
            })
            .await;
        let result = match built {
            Ok(result) => result,
            Err(error) => {
                self.release_uncommitted_lease(lease);
                return Err(error);
            }
        };

        // Publish before any irreversible old-session teardown. Candidate and
        // old-live buffers drain at their current paths, then the rename and
        // retarget run under the same manager/buffer locks. Candidate appends
        // accepted after this handoff can only target the destination.
        let replaces_live_file = self
            .session()
            .session_file()
            .as_deref()
            .map(resolve_path)
            == Some(resolve_path(&destination_path));
        let old_manager = replaces_live_file
            .then(|| Arc::clone(&self.session().session_manager));
        let backup_path = (replaces_live_file || std::path::Path::new(&destination_path).exists())
            .then(|| import_backup_path(&destination_path));
        let handoff = {
            // Lock order is candidate -> optional old-live manager. No guard or
            // buffer lock crosses the teardown await below.
            let mut candidate = session_manager.lock().expect("session manager poisoned");
            if let Some(old_manager) = &old_manager {
                let mut old = old_manager.lock().expect("session manager poisoned");
                candidate.import_handoff(&destination_path, || {
                    old.import_handoff(
                        backup_path
                            .as_deref()
                            .expect("a live import destination always has a backup target"),
                        || {
                            publish_staged_import(
                                &staged.path,
                                &destination_path,
                                backup_path.as_deref(),
                            )
                        },
                    )
                })
            } else {
                candidate.import_handoff(&destination_path, || {
                    publish_staged_import(&staged.path, &destination_path, backup_path.as_deref())
                })
            }
        };
        let published_backup = match handoff {
            Ok(backup) => backup,
            Err(error) => {
                // A failed handoff leaves candidate targeting staged and the
                // old live runtime targeting its original destination.
                result.result.session.dispose_async(None).await;
                self.release_uncommitted_lease(lease);
                return Err(error);
            }
        };
        staged.published = true;

        let teardown_error = self
            .teardown_for_replacement("resume", Some(destination_path.clone()))
            .await
            .err();
        #[cfg(test)]
        let rollback_safe_error = teardown_error
            .as_ref()
            .filter(|error| {
                error.as_str() == "synthetic import teardown fault before old-session disposal"
            })
            .cloned();
        #[cfg(not(test))]
        let rollback_safe_error: Option<String> = None;
        if let Some(error) = rollback_safe_error {
            // Only the synthetic pre-disposal fault is rollback-safe. Production
            // child-cleanup errors arrive after irreversible disposal and must
            // commit the candidate, matching the other replacement operations.
            let mut rollback_failure = None;
            let rollback_error = {
                let mut candidate = session_manager.lock().expect("session manager poisoned");
                let rollback = if let Some(old_manager) = &old_manager {
                    let mut old = old_manager.lock().expect("session manager poisoned");
                    candidate.import_handoff(&staged.path, || {
                        old.import_handoff(&destination_path, || {
                            match rollback_published_import(
                                &staged.path,
                                &destination_path,
                                published_backup.as_deref(),
                            ) {
                                Ok(()) => Ok(()),
                                Err(failure) => {
                                    let message = failure.message.clone();
                                    rollback_failure = Some(failure);
                                    Err(message)
                                }
                            }
                        })
                    })
                } else {
                    candidate.import_handoff(&staged.path, || {
                        match rollback_published_import(
                            &staged.path,
                            &destination_path,
                            published_backup.as_deref(),
                        ) {
                            Ok(()) => Ok(()),
                            Err(failure) => {
                                let message = failure.message.clone();
                                rollback_failure = Some(failure);
                                Err(message)
                            }
                        }
                    })
                };
                match rollback {
                    Ok(()) => {
                        staged.published = false;
                        None
                    }
                    Err(error) => {
                        if rollback_failure
                            .as_ref()
                            .is_some_and(|failure| failure.staged_reclaimed)
                        {
                            candidate.retarget_imported_session_file(&staged.path);
                            staged.published = false;
                        }
                        Some(error)
                    }
                }
            };
            result.result.session.dispose_async(None).await;
            self.release_uncommitted_lease(lease);
            return match rollback_error {
                Some(rollback_error) => Err(format!(
                    "{error}; import rollback also failed: {rollback_error}"
                )),
                None => Err(error),
            };
        }

        let deferred_cleanup_error = teardown_error;
        if let Some(old_manager) = &old_manager {
            // All old dispose writes belong to the backup inode. Draining is
            // best-effort here because the old runtime is already disposed;
            // reporting an Err now would falsely promise a live-session rollback.
            let mut old = old_manager.lock().expect("session manager poisoned");
            let _ = old.flush_now();
            old.retire_import_replaced_session();
        }
        if let Some(backup) = published_backup {
            let _ = std::fs::remove_file(backup);
        }

        self.apply(result);
        self.commit_replacement_lease(lease);
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

/// CWD validation reads the staged manager but reports the public destination
/// in errors, not a random staging filename.
struct ImportSessionCwdSource<'a> {
    session_manager: &'a Arc<Mutex<SessionManager>>,
    destination_path: &'a str,
}

impl crate::core::session_cwd::SessionCwdSource for ImportSessionCwdSource<'_> {
    fn get_cwd(&self) -> String {
        self.session_manager
            .lock()
            .expect("session manager poisoned")
            .get_cwd()
    }

    fn get_session_file(&self) -> Option<String> {
        Some(self.destination_path.to_string())
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

/// Removes an unpublished staged import on every early-return path.
struct StagedImportFile {
    path: String,
    published: bool,
}

impl Drop for StagedImportFile {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImportRenameStage {
    DestinationBackup,
    StagedPublish,
    BackupRestore,
    RollbackCandidate,
}

#[cfg(test)]
#[derive(Default)]
struct ImportFaultState {
    missing_directory: String,
    failures: Vec<ImportRenameStage>,
    attempts: Vec<ImportRenameStage>,
    injected: Vec<ImportRenameStage>,
    staged_path: Option<String>,
    fail_teardown: bool,
}

#[cfg(test)]
fn import_fault_registry(
) -> &'static Mutex<HashMap<String, Arc<Mutex<ImportFaultState>>>> {
    static REGISTRY: std::sync::OnceLock<Mutex<HashMap<String, Arc<Mutex<ImportFaultState>>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
fn import_fault_for(destination_path: &str) -> Option<Arc<Mutex<ImportFaultState>>> {
    import_fault_registry()
        .lock()
        .expect("import fault registry poisoned")
        .get(&resolve_path(destination_path))
        .cloned()
}

#[cfg(test)]
fn inject_import_rename_fault(source: &str, destination: &str) -> Option<std::io::Result<()>> {
    let (destination_path, stage) = if destination.contains(".import-backup-") {
        (source, ImportRenameStage::DestinationBackup)
    } else if source.contains(".import-backup-") {
        (destination, ImportRenameStage::BackupRestore)
    } else if source.contains(".import-") {
        (destination, ImportRenameStage::StagedPublish)
    } else if destination.contains(".import-") {
        (source, ImportRenameStage::RollbackCandidate)
    } else {
        return None;
    };
    let fault = import_fault_for(destination_path)?;
    let missing_directory = {
        let mut state = fault.lock().expect("import fault state poisoned");
        state.attempts.push(stage);
        match stage {
            ImportRenameStage::StagedPublish => state.staged_path = Some(source.to_string()),
            _ => {}
        }
        if !state.failures.contains(&stage) {
            return None;
        }
        state.missing_directory.clone()
    };
    // The injected error comes from the operating system, not a fabricated
    // string. The missing parent also guarantees that `source` is not moved.
    let blocked = std::path::Path::new(&missing_directory).join("rename-target");
    let result = std::fs::rename(source, blocked.to_string_lossy().as_ref());
    assert!(result.is_err(), "the missing rename parent must produce a real OS fault");
    fault.lock().expect("import fault state poisoned").injected.push(stage);
    Some(result)
}

#[cfg(test)]
fn import_teardown_fault(target_session_file: &str) -> bool {
    let Some(fault) = import_fault_for(target_session_file) else {
        return false;
    };
    let mut state = fault.lock().expect("import fault state poisoned");
    let fail = state.fail_teardown;
    state.fail_teardown = false;
    fail
}

fn import_backup_path(destination_path: &str) -> String {
    format!(
        "{destination_path}.import-backup-{}.{}.tmp",
        std::process::id(),
        uuid::Uuid::new_v4()
    )
}

/// One indirection keeps production on the real Windows/Unix rename while
/// deterministic tests can inject an actual failing filesystem operation.
fn rename_import_file(source: &str, destination: &str) -> std::io::Result<()> {
    #[cfg(test)]
    if let Some(result) = inject_import_rename_fault(source, destination) {
        return result;
    }
    std::fs::rename(source, destination)
}

/// Restore the original inode when possible. If a rename rollback itself is
/// obstructed, a same-filesystem copy is a second recovery path for the exact
/// bytes; the backup is retained and named in the error if both methods fail.
fn restore_import_backup(backup: &str, destination_path: &str) -> Result<(), String> {
    match rename_import_file(backup, destination_path) {
        Ok(()) => Ok(()),
        Err(rename_error) => match std::fs::copy(backup, destination_path) {
            Ok(_) => {
                std::fs::remove_file(backup).map_err(|remove_error| {
                    format!(
                        "restore rename failed ({rename_error}); copied the original bytes to {destination_path}, but retained backup {backup} could not be removed: {remove_error}"
                    )
                })?;
                Ok(())
            }
            Err(copy_error) => Err(format!(
                "restore rename failed ({rename_error}); restore copy failed ({copy_error}); original bytes remain recoverable at {backup}"
            )),
        },
    }
}

/// Publish beside the destination. An existing destination is moved aside so
/// the same rename protocol works on Windows as on Unix; a failed publish puts
/// those exact old bytes back before reporting the error.
fn publish_staged_import(
    staged_path: &str,
    destination_path: &str,
    backup: Option<&str>,
) -> Result<Option<String>, String> {
    let destination_exists = std::path::Path::new(destination_path).exists();
    if destination_exists {
        let backup = backup.ok_or_else(|| "Existing import destination has no backup path".to_string())?;
        rename_import_file(destination_path, backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = rename_import_file(staged_path, destination_path) {
        let restore = if destination_exists {
            backup.map(|backup| restore_import_backup(backup, destination_path))
        } else {
            None
        };
        return match restore {
            Some(Err(restore_error)) => Err(format!(
                "{error}; additionally failed to restore the existing session: {restore_error}"
            )),
            _ => Err(error.to_string()),
        };
    }
    Ok(if destination_exists {
        backup.map(str::to_string)
    } else {
        None
    })
}

struct ImportRollbackError {
    message: String,
    staged_reclaimed: bool,
}

/// Undo a successful handoff when old-runtime teardown still fails. Original
/// bytes stay in the backup and the old manager remains targeted there on an
/// obstructed restore, keeping that runtime operable and recovery explicit.
fn rollback_published_import(
    staged_path: &str,
    destination_path: &str,
    backup: Option<&str>,
) -> Result<(), ImportRollbackError> {
    if let Err(error) = rename_import_file(destination_path, staged_path) {
        return Err(ImportRollbackError {
            message: format!(
                "could not reclaim the imported destination: {error}; original bytes remain at {}",
                backup.unwrap_or("their original current-session path")
            ),
            staged_reclaimed: false,
        });
    }
    if let Some(backup) = backup {
        if let Err(error) = restore_import_backup(backup, destination_path) {
            return Err(ImportRollbackError {
                message: error,
                staged_reclaimed: true,
            });
        }
    }
    Ok(())
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

    struct PersistenceFixture {
        _temp: tempfile::TempDir,
        cwd: String,
        session_dir: String,
        runtime: Arc<AgentSessionRuntime>,
        fail_build: Arc<std::sync::atomic::AtomicBool>,
        prepare_candidate: Arc<std::sync::atomic::AtomicBool>,
        candidate_managers: Arc<Mutex<Vec<Arc<Mutex<SessionManager>>>>>,
        candidate_disposals: Arc<std::sync::atomic::AtomicUsize>,
    }

    async fn build_persistence_candidate(
        input: CreateAgentSessionRuntimeInput,
        settings_manager: Arc<Mutex<crate::core::settings_manager::SettingsManager>>,
        model_registry: Arc<Mutex<crate::core::model_registry::ModelRegistry>>,
        model: pi_ai::types::Model,
        fail_build: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<CreateAgentSessionRuntimeResult, String> {
        if fail_build.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("synthetic import candidate failure".to_string());
        }
        let services = crate::core::agent_session_services::create_agent_session_services(
            crate::core::agent_session_services::CreateAgentSessionServicesOptions {
                cwd: input.cwd.clone(),
                agent_dir: Some(input.agent_dir.clone()),
                auth_storage: None,
                settings_manager: Some(Arc::clone(&settings_manager)),
                model_registry: Some(Arc::clone(&model_registry)),
                extension_flag_values: None,
                no_builtin_herdr_reporter: Some(true),
                telemetry_disabled: Some(true),
                resource_loader_options: Some(
                    crate::core::resource_loader::DefaultResourceLoaderOptions {
                        cwd: input.cwd.clone(),
                        agent_dir: input.agent_dir.clone(),
                        settings_manager: Some(settings_manager),
                        no_extensions: true,
                        no_skills: true,
                        no_prompt_templates: true,
                        no_themes: true,
                        no_context_files: true,
                        bundled_skills_dir: Some(None),
                        ..Default::default()
                    },
                ),
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let services = Arc::new(services);
        let result = crate::core::agent_session_services::create_agent_session_from_services(
            crate::core::agent_session_services::CreateAgentSessionFromServicesOptions {
                services: Arc::clone(&services),
                session_manager: input.session_manager,
                session_start_event: input.session_start_event,
                creation: AgentSessionCreationOptions {
                    model: Some(model),
                    no_tools: Some("all".to_string()),
                    prewarm_ipython_kernel: Some(false),
                    telemetry_disabled: Some(true),
                    rlm_depth: Some(0),
                    serialized_refine: Some(false),
                    ..Default::default()
                },
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        Ok(CreateAgentSessionRuntimeResult {
            result,
            services,
            diagnostics: Vec::new(),
        })
    }

    impl PersistenceFixture {
        async fn new(case: &str) -> Self {
            let temp = tempfile::tempdir().unwrap();
            let cwd_path = temp.path().join("workspace");
            let session_path = temp.path().join("sessions");
            let agent_path = temp.path().join(case);
            std::fs::create_dir_all(&cwd_path).unwrap();
            std::fs::create_dir_all(&session_path).unwrap();
            std::fs::create_dir_all(&agent_path).unwrap();
            let cwd = cwd_path.to_string_lossy().into_owned();
            let session_dir = session_path.to_string_lossy().into_owned();
            let agent_dir = agent_path.to_string_lossy().into_owned();

            let provider = pi_ai::providers::faux::register_faux_provider(Some(
                pi_ai::providers::faux::RegisterFauxProviderOptions {
                    provider: Some(format!("persistence-{}", uuid::Uuid::new_v4())),
                    tokens_per_second: Some(0.0),
                    ..Default::default()
                },
            ));
            let model = provider.get_model();
            let settings_manager = Arc::new(Mutex::new(
                crate::core::settings_manager::SettingsManager::in_memory(
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
                ),
            ));
            let model_registry = Arc::new(Mutex::new(
                crate::core::model_registry::ModelRegistry::in_memory(
                    crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
                ),
            ));
            let current_file = session_path.join("current.jsonl").to_string_lossy().into_owned();
            let mut manager = SessionManager::create(&cwd, Some(&session_dir)).unwrap();
            manager.set_session_file(&current_file, None, None).unwrap();
            manager.flush_now().unwrap();
            let session_manager = Arc::new(Mutex::new(manager));
            let fail_build = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let prepare_candidate = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let candidate_managers = Arc::new(Mutex::new(Vec::new()));
            let candidate_disposals = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let factory: CreateAgentSessionRuntimeFactory = {
                let settings_manager = Arc::clone(&settings_manager);
                let model_registry = Arc::clone(&model_registry);
                let model = model.clone();
                let fail_build = Arc::clone(&fail_build);
                let prepare_candidate = Arc::clone(&prepare_candidate);
                let candidate_managers = Arc::clone(&candidate_managers);
                let candidate_disposals = Arc::clone(&candidate_disposals);
                Arc::new(move |input| {
                    let settings_manager = Arc::clone(&settings_manager);
                    let model_registry = Arc::clone(&model_registry);
                    let model = model.clone();
                    let fail_build = Arc::clone(&fail_build);
                    let prepare_candidate = Arc::clone(&prepare_candidate);
                    let candidate_managers = Arc::clone(&candidate_managers);
                    let candidate_disposals = Arc::clone(&candidate_disposals);
                    let candidate_manager = Arc::clone(&input.session_manager);
                    Box::pin(async move {
                        let built = build_persistence_candidate(
                            input,
                            settings_manager,
                            model_registry,
                            model,
                            fail_build,
                        )
                        .await?;
                        if prepare_candidate.swap(false, std::sync::atomic::Ordering::SeqCst) {
                            candidate_manager.lock().unwrap().hold_write_buffer_for_test();
                            candidate_manager
                                .lock()
                                .unwrap()
                                .append_message(pi_agent_core::types::AgentMessage::Message(
                                    pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                                        pi_ai::types::UserContent::Text(
                                            "candidate-pre-publish-entry".to_string(),
                                        ),
                                        31,
                                    )),
                                ))
                                .unwrap();
                            let callback_manager = Arc::clone(&candidate_manager);
                            let candidate_disposals = Arc::clone(&candidate_disposals);
                            built.result.session.register_dispose_callback(Arc::new(move || {
                                let callback_manager = Arc::clone(&callback_manager);
                                let candidate_disposals = Arc::clone(&candidate_disposals);
                                Box::pin(async move {
                                    candidate_disposals.fetch_add(
                                        1,
                                        std::sync::atomic::Ordering::SeqCst,
                                    );
                                    callback_manager
                                        .lock()
                                        .unwrap()
                                        .append_message(pi_agent_core::types::AgentMessage::Message(
                                            pi_ai::types::Message::User(
                                                pi_ai::types::UserMessage::new(
                                                    pi_ai::types::UserContent::Text(
                                                        "candidate-dispose-callback-entry".to_string(),
                                                    ),
                                                    32,
                                                ),
                                            ),
                                        ))
                                        .unwrap();
                                })
                            }));
                            candidate_managers.lock().unwrap().push(candidate_manager);
                        }
                        Ok(built)
                    })
                })
            };
            let runtime = create_agent_session_runtime(
                factory,
                CreateAgentSessionRuntimeInput {
                    cwd: cwd.clone(),
                    agent_dir: agent_dir.clone(),
                    session_manager,
                    session_start_event: None,
                    session_config: None,
                    session_options: None,
                    runtime_metadata: None,
                    session_lease: None,
                },
            )
            .await
            .unwrap();
            Self {
                _temp: temp,
                cwd,
                session_dir,
                runtime,
                fail_build,
                prepare_candidate,
                candidate_managers,
                candidate_disposals,
            }
        }

        fn path(&self, name: &str) -> String {
            std::path::Path::new(&self.session_dir)
                .join(name)
                .to_string_lossy()
                .into_owned()
        }

        fn input_path(&self, name: &str) -> String {
            let directory = self._temp.path().join("external");
            std::fs::create_dir_all(&directory).unwrap();
            directory.join(name).to_string_lossy().into_owned()
        }

        fn collision_destination(
            &self,
            current: bool,
            name: &str,
        ) -> (String, Arc<Mutex<SessionManager>>) {
            let destination = self.path(name);
            let manager = if current {
                let manager = Arc::clone(&self.runtime.session().session_manager);
                let live_file = self.runtime.session().session_file().unwrap();
                std::fs::rename(&live_file, &destination).unwrap();
                manager
                    .lock()
                    .unwrap()
                    .set_session_file(&destination, None, None)
                    .unwrap();
                self.runtime.release_session_lease();
                let lease = self.runtime.acquire_replacement_lease(Some(&destination)).unwrap();
                self.runtime.commit_replacement_lease(lease);
                manager
            } else {
                let mut manager = SessionManager::create(&self.cwd, Some(&self.session_dir)).unwrap();
                manager.set_session_file(&destination, None, None).unwrap();
                manager.flush_now().unwrap();
                Arc::new(Mutex::new(manager))
            };
            (destination, manager)
        }

        fn candidate_manager(&self) -> Arc<Mutex<SessionManager>> {
            self.candidate_managers
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("the prepared candidate retained its real session manager")
        }

        fn prepare_import_candidate(&self) {
            self.prepare_candidate
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn install_import_fault(
        fixture: &PersistenceFixture,
        destination: &str,
        failures: Vec<ImportRenameStage>,
        fail_teardown: bool,
    ) -> Arc<Mutex<ImportFaultState>> {
        let state = Arc::new(Mutex::new(ImportFaultState {
            missing_directory: fixture
                ._temp
                .path()
                .join("missing-rename-parent")
                .to_string_lossy()
                .into_owned(),
            failures,
            fail_teardown,
            ..Default::default()
        }));
        import_fault_registry()
            .lock()
            .unwrap()
            .insert(resolve_path(destination), Arc::clone(&state));
        state
    }

    fn clear_import_fault(destination: &str) {
        import_fault_registry().lock().unwrap().remove(&resolve_path(destination));
    }

    fn import_leftovers(destination: &str) -> Vec<String> {
        let destination = std::path::Path::new(destination);
        let prefix = format!(
            "{}.import-",
            destination.file_name().unwrap().to_string_lossy()
        );
        std::fs::read_dir(destination.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(&prefix))
            .collect()
    }

    fn append_buffered_user(
        manager: &Arc<Mutex<SessionManager>>,
        text: &str,
    ) -> (String, Vec<u8>) {
        let file = manager.lock().unwrap().get_session_file().unwrap();
        manager.lock().unwrap().hold_write_buffer_for_test();
        let before = std::fs::read(&file).unwrap();
        let id = manager
            .lock()
            .unwrap()
            .append_message(pi_agent_core::types::AgentMessage::Message(
                pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                    pi_ai::types::UserContent::Text(text.to_string()),
                    1,
                )),
            ))
            .unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), before, "fixture must hold the append");
        (id, before)
    }

    fn write_import(path: &str, id: &str, cwd: &str) {
        std::fs::write(
            path,
            format!(
                "{{\"type\":\"session\",\"version\":3,\"id\":{},\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":{}}}\n",
                serde_json::to_string(id).unwrap(),
                serde_json::to_string(cwd).unwrap(),
            ),
        )
        .unwrap();
    }

    fn assert_html_includes_pending(
        fixture: &PersistenceFixture,
        output: &str,
        newest_id: &str,
        export: Result<String, String>,
    ) {
        let output_path = export.unwrap();
        assert_eq!(output_path, output);
        let html = std::fs::read_to_string(output).unwrap();
        const DATA_OPEN: &str = r#"<script id="session-data" type="application/json">"#;
        let data_start = html.find(DATA_OPEN).expect("embedded session-data start") + DATA_OPEN.len();
        let data_end = html[data_start..]
            .find("</script>")
            .map(|offset| data_start + offset)
            .expect("embedded session-data end");
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&html[data_start..data_end])
            .expect("session-data base64");
        let session_data: Value = serde_json::from_slice(&decoded).expect("session-data JSON");
        assert!(session_data
            .get("entries")
            .and_then(Value::as_array)
            .is_some_and(|entries| entries.iter().any(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(newest_id)
                    && serde_json::to_string(entry)
                        .is_ok_and(|entry| entry.contains("pending-html-export-visibility"))
            })), "pending user entry missing from exported session data");
        let source = std::fs::read_to_string(fixture.runtime.session().session_file().unwrap()).unwrap();
        assert!(source.contains(&format!("\"id\":\"{newest_id}\"")), "export must drain the pending id");
    }

    fn assert_fork_contains_every_source_entry(
        source_entries: Vec<serde_json::Map<String, Value>>,
        forked_file: String,
        newest_id: &str,
    ) {
        let forked = std::fs::read_to_string(forked_file).unwrap();
        assert!(forked.contains(&format!("\"id\":\"{newest_id}\"")), "newest pending id missing from fork");
        for source in source_entries {
            let id = source.get("id").and_then(Value::as_str).unwrap();
            assert!(forked.contains(&format!("\"id\":\"{id}\"")), "source entry {id} missing from fork");
        }
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn daemon_html_export_drains_live_pending_entries() {
        let fixture = PersistenceFixture::new("daemon-html").await;
        let (id, _) = append_buffered_user(
            &fixture.runtime.session().session_manager,
            "pending-html-export-visibility",
        );
        let output = fixture._temp.path().join("daemon.html").to_string_lossy().into_owned();
        let adapter = AgentSessionDaemonAdapter::new(Arc::clone(&fixture.runtime));
        use crate::modes::daemon::daemon_mode::DaemonSession;
        let exported = adapter.export_to_html(Some(&output)).await;
        assert_html_includes_pending(&fixture, &output, &id, exported);
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_process_html_export_drains_live_pending_entries() {
        let fixture = PersistenceFixture::new("in-process-html").await;
        let (id, _) = append_buffered_user(
            &fixture.runtime.session().session_manager,
            "pending-html-export-visibility",
        );
        let output = fixture._temp.path().join("in-process.html").to_string_lossy().into_owned();
        let adapter = InProcessRuntimeHostAdapter::new(Arc::clone(&fixture.runtime));
        use crate::modes::agent_connection::in_process_agent_connection::InProcessRuntimeHost;
        let exported = adapter.session_export_to_html(Some(&output)).await;
        assert_html_includes_pending(&fixture, &output, &id, exported);
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn daemon_persisted_fork_drains_every_live_pending_entry() {
        let fixture = PersistenceFixture::new("daemon-fork").await;
        let manager = Arc::clone(&fixture.runtime.session().session_manager);
        let (id, _) = append_buffered_user(&manager, "pending-daemon-fork-entry");
        let source_entries = manager.lock().unwrap().get_entries();
        let adapter = AgentSessionDaemonAdapter::new(Arc::clone(&fixture.runtime));
        use crate::modes::daemon::daemon_mode::{DaemonSession, ForkOptions};
        adapter
            .fork(&id, ForkOptions { position: Some("at".to_string()) })
            .await
            .unwrap();
        let forked_file = fixture.runtime.session().session_file().unwrap();
        assert_fork_contains_every_source_entry(source_entries, forked_file, &id);
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_process_persisted_fork_drains_every_live_pending_entry() {
        let fixture = PersistenceFixture::new("in-process-fork").await;
        let manager = Arc::clone(&fixture.runtime.session().session_manager);
        let (id, _) = append_buffered_user(&manager, "pending-in-process-fork-entry");
        let source_entries = manager.lock().unwrap().get_entries();
        let adapter = InProcessRuntimeHostAdapter::new(Arc::clone(&fixture.runtime));
        use crate::modes::agent_connection::in_process_agent_connection::InProcessRuntimeHost;
        use crate::modes::agent_connection::types::AgentConnectionForkOptions;
        adapter
            .runtime_fork(&id, Some(AgentConnectionForkOptions { position: Some("at".to_string()) }))
            .await
            .unwrap();
        let forked_file = fixture.runtime.session().session_file().unwrap();
        assert_fork_contains_every_source_entry(source_entries, forked_file, &id);
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn successful_current_path_import_excludes_old_pending_and_teardown_entries() {
        let fixture = PersistenceFixture::new("success-current-import").await;
        let old_manager = Arc::clone(&fixture.runtime.session().session_manager);
        let destination = fixture.runtime.session().session_file().unwrap();
        let (old_pending_id, _) = append_buffered_user(
            &old_manager,
            "old-entry-that-must-not-cross-import-commit",
        );
        let old_session = fixture.runtime.session();
        let teardown_entry_id = Arc::new(Mutex::new(None::<String>));
        let callback_manager = Arc::clone(&old_manager);
        let callback_entry_id = Arc::clone(&teardown_entry_id);
        old_session.register_dispose_callback(Arc::new(move || {
            let callback_manager = Arc::clone(&callback_manager);
            let callback_entry_id = Arc::clone(&callback_entry_id);
            Box::pin(async move {
                let id = callback_manager
                    .lock()
                    .unwrap()
                    .append_message(pi_agent_core::types::AgentMessage::Message(
                        pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                            pi_ai::types::UserContent::Text(
                                "old-dispose-callback-entry".to_string(),
                            ),
                            2,
                        )),
                    ))
                    .unwrap();
                *callback_entry_id.lock().unwrap() = Some(id);
            })
        }));
        let input = fixture.input_path("current.jsonl");
        write_import(&input, "successful-current-import", &fixture.cwd);

        fixture.runtime.import_from_jsonl(&input, None).await.unwrap();
        let teardown_entry_id = teardown_entry_id
            .lock()
            .unwrap()
            .clone()
            .expect("dispose callback must append the teardown entry");
        assert!(!Arc::ptr_eq(&fixture.runtime.session().session_manager, &old_manager));
        assert_eq!(fixture.runtime.session().session_file().as_deref(), Some(destination.as_str()));
        let imported = std::fs::read_to_string(&destination).unwrap();
        assert!(imported.contains("successful-current-import"));
        assert!(
            !imported.contains(&format!("\"id\":\"{old_pending_id}\"")),
            "old pre-import bytes crossed into the imported inode"
        );
        assert!(
            !imported.contains(&format!("\"id\":\"{teardown_entry_id}\"")),
            "old teardown bytes crossed into the imported inode"
        );

        // Simulate a late timed flush from the retained old manager. Backup
        // retargeting, the post-teardown drain and manager retirement make it
        // a no-op: old entries cannot contaminate the imported destination or
        // recreate an expired backup after cleanup.
        old_manager.lock().unwrap().flush_now().unwrap();
        let after_old_flush = std::fs::read_to_string(&destination).unwrap();
        for old_id in [&old_pending_id, &teardown_entry_id] {
            assert!(
                !after_old_flush.contains(&format!("\"id\":\"{old_id}\"")),
                "late old-manager drain contaminated the imported path with {old_id}"
            );
        }

        fixture
            .runtime
            .session()
            .session_manager
            .lock()
            .unwrap()
            .flush_now()
            .unwrap();
        let after_new_flush = std::fs::read_to_string(&destination).unwrap();
        assert!(after_new_flush.contains("successful-current-import"));
        for old_id in [&old_pending_id, &teardown_entry_id] {
            assert!(
                !after_new_flush.contains(&format!("\"id\":\"{old_id}\"")),
                "old manager appended after the new manager took over the path: {old_id}"
            );
        }
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_faults_preserve_current_and_inactive_destinations_with_pending_entries() {
        for current_destination in [true, false] {
            for fault in ["header", "cwd", "candidate"] {
                let fixture = PersistenceFixture::new(&format!("import-{fault}-{current_destination}")).await;
                let destination = fixture.path("collision.jsonl");
                let old_manager = if current_destination {
                    let manager = Arc::clone(&fixture.runtime.session().session_manager);
                    let live_file = fixture.runtime.session().session_file().unwrap();
                    std::fs::rename(&live_file, &destination).unwrap();
                    manager
                        .lock()
                        .unwrap()
                        .set_session_file(&destination, None, None)
                        .unwrap();
                    manager
                } else {
                    let mut manager = SessionManager::create(&fixture.cwd, Some(&fixture.session_dir)).unwrap();
                    manager.set_session_file(&destination, None, None).unwrap();
                    manager.flush_now().unwrap();
                    Arc::new(Mutex::new(manager))
                };
                let (old_pending_id, old_bytes) = append_buffered_user(
                    &old_manager,
                    &format!("old-pending-{fault}-{current_destination}"),
                );
                let input = fixture.input_path("collision.jsonl");
                match fault {
                    "header" => write_import(&input, "../header-fault", &fixture.cwd),
                    "cwd" => write_import(
                        &input,
                        "cwd-fault",
                        &fixture._temp.path().join("missing-cwd").to_string_lossy(),
                    ),
                    "candidate" => {
                        write_import(&input, "candidate-fault", &fixture.cwd);
                        fixture.fail_build.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    _ => unreachable!(),
                }
                let error = fixture.runtime.import_from_jsonl(&input, None).await.unwrap_err();
                assert!(!error.is_empty());
                assert_eq!(std::fs::read(&destination).unwrap(), old_bytes, "{fault} changed {current_destination} destination bytes");
                assert!(old_manager.lock().unwrap().get_entry(&old_pending_id).is_some(), "old pending entry was lost");
                old_manager.lock().unwrap().flush_now().unwrap();
                let recovered = std::fs::read_to_string(&destination).unwrap();
                assert!(recovered.contains(&format!("\"id\":\"{old_pending_id}\"")), "old pending entry did not drain");
                assert!(!recovered.contains("candidate-fault"), "imported candidate contaminated old bytes");
                assert!(!recovered.contains("cwd-fault"), "imported cwd candidate contaminated old bytes");
                assert!(!recovered.contains("header-fault"), "invalid import contaminated old bytes");
                if current_destination {
                    assert!(Arc::ptr_eq(&fixture.runtime.session().session_manager, &old_manager));
                }
                drop(old_manager);
                fixture.runtime.dispose(Some(Default::default())).await.unwrap();
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_rename_faults_preserve_current_and_inactive_runtime_and_bytes() {
        for current_destination in [true, false] {
            for (label, failures) in [
                ("backup", vec![ImportRenameStage::DestinationBackup]),
                ("staged", vec![ImportRenameStage::StagedPublish]),
                (
                    "restore",
                    vec![ImportRenameStage::StagedPublish, ImportRenameStage::BackupRestore],
                ),
            ] {
                let fixture = PersistenceFixture::new(&format!(
                    "rename-fault-{label}-{current_destination}"
                ))
                .await;
                let (destination, collision_manager) = fixture
                    .collision_destination(current_destination, &format!("{label}.jsonl"));
                let (old_entry_id, _) = append_buffered_user(
                    &collision_manager,
                    &format!("old-before-{label}-{current_destination}"),
                );
                collision_manager.lock().unwrap().flush_now().unwrap();
                let old_bytes = std::fs::read(&destination).unwrap();
                let current_session = fixture.runtime.session();
                let current_manager = Arc::clone(&current_session.session_manager);
                let current_lease = fixture
                    .runtime
                    .session_lease
                    .lock()
                    .unwrap()
                    .clone()
                    .expect("fixture owns its current-session lease");
                let old_disposals = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let callback_disposals = Arc::clone(&old_disposals);
                current_session.register_dispose_callback(Arc::new(move || {
                    let callback_disposals = Arc::clone(&callback_disposals);
                    Box::pin(async move {
                        callback_disposals.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    })
                }));
                fixture.prepare_import_candidate();
                let fault = install_import_fault(&fixture, &destination, failures, false);
                let input = fixture.input_path(&format!("{label}.jsonl"));
                write_import(&input, &format!("imported-{label}"), &fixture.cwd);

                let error = fixture.runtime.import_from_jsonl(&input, None).await.unwrap_err();
                assert!(!error.is_empty());
                let (attempts, injected, fault_staged_path) = {
                    let state = fault.lock().unwrap();
                    (
                        state.attempts.clone(),
                        state.injected.clone(),
                        state.staged_path.clone(),
                    )
                };
                clear_import_fault(&destination);
                assert!(attempts.contains(&match label {
                    "backup" => ImportRenameStage::DestinationBackup,
                    _ => ImportRenameStage::StagedPublish,
                }));
                let expected_fault = match label {
                    "backup" => ImportRenameStage::DestinationBackup,
                    _ => ImportRenameStage::StagedPublish,
                };
                assert!(injected.contains(&expected_fault));
                if label == "restore" {
                    assert!(attempts.contains(&ImportRenameStage::BackupRestore));
                    assert!(injected.contains(&ImportRenameStage::BackupRestore));
                }

                assert_eq!(std::fs::read(&destination).unwrap(), old_bytes);
                assert!(Arc::ptr_eq(&fixture.runtime.session(), &current_session));
                assert!(Arc::ptr_eq(
                    &fixture.runtime.session().session_manager,
                    &current_manager
                ));
                let after_lease = fixture.runtime.session_lease.lock().unwrap().clone().unwrap();
                assert!(Arc::ptr_eq(&after_lease, &current_lease));
                assert_eq!(old_disposals.load(std::sync::atomic::Ordering::SeqCst), 0);
                let candidate_manager = fixture.candidate_manager();
                let candidate_target = candidate_manager
                    .lock()
                    .unwrap()
                    .get_session_file()
                    .unwrap();
                assert!(candidate_target.contains(&format!("{label}.jsonl.import-")));
                if let Some(fault_staged_path) = fault_staged_path {
                    assert_eq!(candidate_target, fault_staged_path);
                }
                assert_eq!(
                    fixture
                        .candidate_disposals
                        .load(std::sync::atomic::Ordering::SeqCst),
                    1,
                    "candidate must be disposed exactly once"
                );
                assert!(import_leftovers(&destination).is_empty());

                // The retained old runtime still accepts and persists a real entry.
                let operable_id = current_manager
                    .lock()
                    .unwrap()
                    .append_message(pi_agent_core::types::AgentMessage::Message(
                        pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                            pi_ai::types::UserContent::Text(
                                "old-runtime-operable-after-import-fault".to_string(),
                            ),
                            41,
                        )),
                    ))
                    .unwrap();
                current_manager.lock().unwrap().flush_now().unwrap();
                assert!(collision_manager.lock().unwrap().get_entry(&old_entry_id).is_some());
                assert!(current_manager.lock().unwrap().get_entry(&operable_id).is_some());
                let current_file = current_manager.lock().unwrap().get_session_file().unwrap();
                let current_bytes = std::fs::read_to_string(current_file).unwrap();
                assert!(current_bytes.contains("old-runtime-operable-after-import-fault"));
                assert!(!current_bytes.contains(&format!("imported-{label}")));

                fixture.runtime.dispose(Some(Default::default())).await.unwrap();
                assert_eq!(old_disposals.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_teardown_fault_rolls_back_publish_and_restore_faults() {
        for current_destination in [true, false] {
            for inject_restore_fault in [false, true] {
                let fixture = PersistenceFixture::new(&format!(
                    "teardown-rollback-{current_destination}-{inject_restore_fault}"
                ))
                .await;
                let (destination, collision_manager) = fixture.collision_destination(
                    current_destination,
                    &format!("teardown-{inject_restore_fault}.jsonl"),
                );
                collision_manager.lock().unwrap().flush_now().unwrap();
                let old_bytes = std::fs::read(&destination).unwrap();
                let current_session = fixture.runtime.session();
                let current_manager = Arc::clone(&current_session.session_manager);
                let current_lease = fixture
                    .runtime
                    .session_lease
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap();
                let old_disposals = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let callback_disposals = Arc::clone(&old_disposals);
                current_session.register_dispose_callback(Arc::new(move || {
                    let callback_disposals = Arc::clone(&callback_disposals);
                    Box::pin(async move {
                        callback_disposals.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    })
                }));
                fixture.prepare_import_candidate();
                let failures = if inject_restore_fault {
                    vec![ImportRenameStage::BackupRestore]
                } else {
                    Vec::new()
                };
                let fault = install_import_fault(&fixture, &destination, failures, true);
                let input = fixture.input_path(&format!(
                    "teardown-{inject_restore_fault}.jsonl"
                ));
                write_import(&input, "must-not-survive-teardown-rollback", &fixture.cwd);

                let error = fixture.runtime.import_from_jsonl(&input, None).await.unwrap_err();
                assert!(error.contains("synthetic import teardown fault"));
                let (attempts, injected) = {
                    let state = fault.lock().unwrap();
                    (state.attempts.clone(), state.injected.clone())
                };
                clear_import_fault(&destination);
                assert!(attempts.contains(&ImportRenameStage::RollbackCandidate));
                assert!(attempts.contains(&ImportRenameStage::BackupRestore));
                if inject_restore_fault {
                    assert!(injected.contains(&ImportRenameStage::BackupRestore));
                }
                assert_eq!(std::fs::read(&destination).unwrap(), old_bytes);
                assert!(Arc::ptr_eq(&fixture.runtime.session(), &current_session));
                assert!(Arc::ptr_eq(
                    &fixture.runtime.session().session_manager,
                    &current_manager
                ));
                let after_lease = fixture.runtime.session_lease.lock().unwrap().clone().unwrap();
                assert!(Arc::ptr_eq(&after_lease, &current_lease));
                assert_eq!(old_disposals.load(std::sync::atomic::Ordering::SeqCst), 0);
                assert_eq!(
                    fixture
                        .candidate_disposals
                        .load(std::sync::atomic::Ordering::SeqCst),
                    1
                );
                assert!(import_leftovers(&destination).is_empty());
                assert!(!std::fs::read_to_string(&destination)
                    .unwrap()
                    .contains("must-not-survive-teardown-rollback"));
                fixture.runtime.dispose(Some(Default::default())).await.unwrap();
                assert_eq!(old_disposals.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candidate_late_append_and_dispose_callback_use_destination_without_staging_files() {
        for target in ["current", "inactive", "unique"] {
            let fixture = PersistenceFixture::new(&format!("candidate-handoff-{target}")).await;
            let destination = if target == "unique" {
                fixture.path("candidate-unique.jsonl")
            } else {
                let (destination, old_manager) = fixture
                    .collision_destination(target == "current", "candidate-collision.jsonl");
                if target == "current" {
                    old_manager.lock().unwrap().flush_now().unwrap();
                }
                destination
            };
            fixture.prepare_import_candidate();
            let input = fixture.input_path(std::path::Path::new(&destination)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .as_ref());
            write_import(&input, &format!("candidate-success-{target}"), &fixture.cwd);

            fixture.runtime.import_from_jsonl(&input, None).await.unwrap();
            let candidate_manager = fixture.candidate_manager();
            assert_eq!(
                candidate_manager.lock().unwrap().get_session_file().as_deref(),
                Some(destination.as_str())
            );
            assert!(Arc::ptr_eq(
                &fixture.runtime.session().session_manager,
                &candidate_manager
            ));
            candidate_manager
                .lock()
                .unwrap()
                .append_message(pi_agent_core::types::AgentMessage::Message(
                    pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                        pi_ai::types::UserContent::Text(
                            "candidate-late-append-after-publish".to_string(),
                        ),
                        51,
                    )),
                ))
                .unwrap();
            candidate_manager.lock().unwrap().flush_now().unwrap();
            let published = std::fs::read_to_string(&destination).unwrap();
            assert!(published.contains("candidate-pre-publish-entry"));
            assert!(published.contains("candidate-late-append-after-publish"));
            assert!(import_leftovers(&destination).is_empty());

            fixture.runtime.dispose(Some(Default::default())).await.unwrap();
            candidate_manager.lock().unwrap().flush_now().unwrap();
            let disposed = std::fs::read_to_string(&destination).unwrap();
            assert!(disposed.contains("candidate-dispose-callback-entry"));
            assert_eq!(
                fixture
                    .candidate_disposals
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
            assert!(import_leftovers(&destination).is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_still_publishes_a_valid_unique_candidate() {
        let fixture = PersistenceFixture::new("unique-import").await;
        let input = fixture.input_path("unique.jsonl");
        write_import(&input, "unique-imported-session", &fixture.cwd);
        let destination = fixture.path("unique.jsonl");
        assert!(!std::path::Path::new(&destination).exists());
        let result = fixture.runtime.import_from_jsonl(&input, None).await.unwrap();
        assert!(!result.cancelled);
        assert_eq!(fixture.runtime.session().session_file().as_deref(), Some(destination.as_str()));
        let imported = std::fs::read_to_string(&destination).unwrap();
        assert!(imported.contains("unique-imported-session"));
        let leftovers = std::fs::read_dir(&fixture.session_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains("unique.jsonl.import-"))
            .count();
        assert_eq!(leftovers, 0, "staged and backup files must be cleaned up");
        fixture.runtime.dispose(Some(Default::default())).await.unwrap();
    }

}
