//! Owner parity-validation tests for T08 (daemon admission, completion and
//! message receipts): findings C-01, C-02, C-09, H-05.
//!
//! These run inside the crate so they can drive the real production entry points
//! (`AgentDaemon::handle_line`, `AgentDaemon::accept_agent_session_message`,
//! `AgentSessionDaemonAdapter::{prompt_until_accepted,prompt_and_wait}`) without
//! re-implementing any of them. All writable state lives under
//! `PARITY_DAEMON_STATE_ROOT` (default: a process-private temporary directory);
//! nothing here touches the live daemon, its pipe, or the user profile.

use std::sync::atomic::{AtomicBool, Ordering};

use std::pin::Pin;

use tokio::sync::{mpsc, oneshot};

use super::*;

/// Isolated state root for this owner's fixtures (V00).
pub(super) fn state_root(case: &str) -> std::path::PathBuf {
    static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let base = std::env::var_os("PARITY_DAEMON_STATE_ROOT")
        .map(|raw| {
            assert!(!raw.is_empty(), "PARITY_DAEMON_STATE_ROOT must not be empty");
            std::path::absolute(raw).expect("absolute state root")
        })
        .unwrap_or_else(|| {
            ROOT.get_or_init(|| tempfile::Builder::new().prefix("optimus-daemon-t08-").tempdir().expect("private state root"))
                .path().to_path_buf()
        });
    assert!(!base.to_string_lossy().to_lowercase().contains(".prime"), "state root must not touch .prime");
    std::fs::create_dir_all(&base).expect("state root");
    assert!(!std::fs::canonicalize(&base).expect("canonical state root").to_string_lossy().to_lowercase().contains(".prime"), "state root must not resolve into .prime");
    let root = base.join(case);
    std::fs::create_dir_all(root.join("agent")).expect("case root");
    std::fs::create_dir_all(root.join("workspace")).expect("case workspace");
    root
}

/// A test barrier the extension command's dialog waits on.
#[derive(Default)]
pub(super) struct DialogBarrier {
    pub started: AtomicBool,
    pub finished: AtomicBool,
    release: StdMutex<Option<oneshot::Sender<()>>>,
    pub release_rx: StdMutex<Option<oneshot::Receiver<()>>>,
}

impl DialogBarrier {
    pub fn new() -> Arc<Self> {
        let (tx, rx) = oneshot::channel();
        Arc::new(Self {
            started: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            release: StdMutex::new(Some(tx)),
            release_rx: StdMutex::new(Some(rx)),
        })
    }

    pub fn started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// Let the dialog return (the "user completed the dialog" edge).
    pub fn complete(&self) {
        if let Some(release) = self.release.lock().unwrap().take() {
            let _ = release.send(());
        }
    }

    /// Wait until the dialog has actually been entered, with a bound.
    pub async fn wait_started(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if self.started() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.started()
    }

    pub async fn wait_finished(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if self.finished() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.finished()
    }
}

/// The in-crate fixture: one real `AgentSession` over the faux provider, behind the
/// real daemon adapter and registered in a real `AgentDaemon`.
pub(super) struct ParityFixture {
    /// The fixture directory, removed on drop.
    pub root: std::path::PathBuf,
    pub daemon: Arc<AgentDaemon>,
    pub state: Arc<StdMutex<ActiveSessionState>>,
    pub session: Arc<crate::core::agent_session::AgentSession>,
    pub barrier: Arc<DialogBarrier>,
    pub provider: pi_ai::providers::faux::FauxProviderRegistration,
    /// Outbound frames the daemon wrote to the fake client.
    pub outbound: mpsc::UnboundedReceiver<Vec<u8>>,
    pub client: Arc<DaemonClientHandle>,
    pub active_session_id: String,
}

/// The extension command this fixture registers.
pub(super) const DIALOG_COMMAND: &str = "parity-dialog";

pub(super) fn dialog_extension_factory(
    barrier: Arc<DialogBarrier>,
) -> crate::core::extensions::types::ExtensionFactory {
    Arc::new(move |api: Arc<dyn crate::core::extensions::types::ExtensionApi>| {
        let barrier = Arc::clone(&barrier);
        Box::pin(async move {
            let handler: Arc<
                dyn Fn(
                        String,
                        Arc<dyn crate::core::extensions::types::ExtensionCommandContext>,
                    )
                        -> Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
                    + Send
                    + Sync,
            > = Arc::new(move |_args, _ctx| {
                let barrier = Arc::clone(&barrier);
                Box::pin(async move {
                    barrier.started.store(true, Ordering::SeqCst);
                    let receiver = barrier.release_rx.lock().unwrap().take();
                    match receiver {
                        Some(receiver) => {
                            let _ = receiver.await;
                        }
                        None => tokio::time::sleep(Duration::from_millis(50)).await,
                    }
                    barrier.finished.store(true, Ordering::SeqCst);
                    Ok(())
                })
            });
            api.register_command(
                DIALOG_COMMAND.to_string(),
                crate::core::extensions::types::RegisterCommandOptions {
                    description: Some("parity dialog".to_string()),
                    get_argument_completions: None,
                    handler: Some(handler),
                },
            );
            Ok(())
        })
    })
}

impl ParityFixture {
    pub async fn new(case: &str) -> Self {
        let root = state_root(case);
        let cwd = root.join("workspace").to_string_lossy().into_owned();
        let agent_dir = root.join("agent").to_string_lossy().into_owned();
        let barrier = DialogBarrier::new();

        let provider = pi_ai::providers::faux::register_faux_provider(Some(
            pi_ai::providers::faux::RegisterFauxProviderOptions {
                provider: Some(format!("parity-{}", uuid::Uuid::new_v4())),
                tokens_per_second: Some(0.0),
                ..Default::default()
            },
        ));
        let model = provider.get_model();

        let settings = Arc::new(StdMutex::new(
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
        let session_manager = Arc::new(StdMutex::new(
            crate::core::session_manager::SessionManager::in_memory(Some(&cwd), Some(&agent_dir))
                .expect("session manager"),
        ));
        let auth_storage = Arc::new(tokio::sync::Mutex::new(
            crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
        ));
        let model_registry = Arc::new(StdMutex::new(
            crate::core::model_registry::ModelRegistry::in_memory(
                crate::core::auth_storage::AuthStorage::in_memory(Default::default(), None),
            ),
        ));
        auth_storage
            .lock()
            .await
            .set_runtime_api_key(&model.provider, "parity-faux-key");
        model_registry
            .lock()
            .unwrap()
            .set_runtime_api_key(&model.provider, "parity-faux-key");

        let services = crate::core::agent_session_services::create_agent_session_services(
            crate::core::agent_session_services::CreateAgentSessionServicesOptions {
                cwd: cwd.clone(),
                agent_dir: Some(agent_dir.clone()),
                auth_storage: Some(Arc::clone(&auth_storage)),
                settings_manager: Some(Arc::clone(&settings)),
                model_registry: Some(Arc::clone(&model_registry)),
                extension_flag_values: None,
                no_builtin_herdr_reporter: Some(true),
                telemetry_disabled: Some(true),
                resource_loader_options: Some(
                    crate::core::resource_loader::DefaultResourceLoaderOptions {
                    cwd: cwd.clone(),
                    agent_dir: agent_dir.clone(),
                    settings_manager: Some(Arc::clone(&settings)),
                    extension_factories: vec![dialog_extension_factory(Arc::clone(&barrier))],
                    // Inline factories still load with `no_extensions` (resource_loader.rs:790);
                    // the built-ins stay off so nothing else can claim the command name.
                    no_extensions: true,
                    no_skills: true,
                    no_prompt_templates: true,
                    no_themes: true,
                    no_context_files: true,
                    bundled_skills_dir: Some(None),
                    ..Default::default()
                }),
            },
        )
        .await
        .expect("session services");
        let services = Arc::new(services);

        let created = crate::core::agent_session_services::create_agent_session_from_services(
            crate::core::agent_session_services::CreateAgentSessionFromServicesOptions {
                services: Arc::clone(&services),
                session_manager: Arc::clone(&session_manager),
                session_start_event: None,
                creation: crate::core::agent_session_services::AgentSessionCreationOptions {
                    model: Some(model.clone()),
                    no_tools: Some("all".to_string()),
                    prewarm_ipython_kernel: Some(false),
                    telemetry_disabled: Some(true),
                    ..Default::default()
                },
            },
        )
        .await
        .expect("agent session");
        let session = created.session;
        assert!(
            session
                .extension_runner()
                .map(|runner| runner.get_command(DIALOG_COMMAND).is_some())
                .unwrap_or(false),
            "fixture guard: the dialog command must be registered on the session"
        );

        let refuse: crate::core::agent_session_runtime::CreateAgentSessionRuntimeFactory =
            Arc::new(
                |_input: crate::core::agent_session_runtime::CreateAgentSessionRuntimeInput| {
                    Box::pin(
                        async move { Err("parity fixture does not create runtimes".to_string()) },
                    )
                },
            );
        // The runtime takes the canonical metadata; the daemon handle takes the
        // daemon-side projection of the same record.
        let metadata = crate::core::agent_session_runtime::AgentSessionRuntimeMetadata::top_level();
        let handle_metadata = AgentSessionRuntimeMetadata {
            kind: Some("top-level".to_string()),
            ..AgentSessionRuntimeMetadata::default()
        };
        let runtime = crate::core::agent_session_runtime::AgentSessionRuntime::new(
            Arc::clone(&session),
            Arc::clone(&services),
            refuse,
            Vec::new(),
            None,
            None,
            metadata.clone(),
            None,
        );
        let adapter: Arc<dyn DaemonSession> =
            Arc::new(crate::core::agent_session_runtime::AgentSessionDaemonAdapter::new(runtime));

        let socket_path = root.join("daemon-parity.sock").to_string_lossy().into_owned();
        let daemon = AgentDaemon::new(
            socket_path.clone(),
            DaemonModeOptions {
                socket_path: Some(socket_path),
                default_session_config: AgentSessionRuntimeConfig {
                    cwd: Some(cwd.clone()),
                    agent_dir: Some(agent_dir.clone()),
                    ..Default::default()
                },
                create_runtime: Arc::new(|_| {
                    Box::pin(async { Err("parity fixture does not create runtimes".to_string()) })
                }),
                worker: None,
            },
        );
        let state = daemon
            .add_runtime(
                AgentSessionRuntimeHandle {
                    session: adapter,
                    metadata: handle_metadata,
                    model_fallback_message: None,
                    new_session: None,
                    switch_session: None,
                    fork: None,
                    import_from_jsonl: None,
                },
                None,
                None,
                None,
            )
            .await
            .expect("daemon registers the runtime");
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();

        let (sender, outbound) = mpsc::unbounded_channel();
        let client = Arc::new(DaemonClientHandle::new(
            "parity-client",
            Arc::new(DaemonClientWriter::new(sender)),
            Arc::new(|| {}),
        ));
        {
            let mut client_state = client.state.lock().expect("daemon client poisoned");
            client_state.authenticated = Some(true);
            client_state.transport = Some("jsonl".to_string());
            client_state.capabilities = normalize_client_capabilities(None, None);
        }
        daemon
            .clients
            .lock()
            .expect("clients poisoned")
            .push(Arc::clone(&client));

        Self {
            root,
            daemon,
            state,
            session,
            barrier,
            provider,
            outbound,
            client,
            active_session_id,
        }
    }

    pub fn command(&self, value: Value) -> String {
        serialize_json_line(&value)
    }

    /// The next response frame for `id`, skipping every non-response frame.
    pub async fn wait_response(&mut self, id: &str, timeout: Duration) -> Option<Value> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let frame = match tokio::time::timeout(remaining, self.outbound.recv()).await {
                Ok(Some(frame)) => frame,
                _ => return None,
            };
            if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                if value.get("type").and_then(Value::as_str) == Some("response")
                    && value.get("id").and_then(Value::as_str) == Some(id)
                {
                    return Some(value);
                }
            }
        }
    }

    /// Every frame received so far, without waiting.
    pub fn drain(&mut self) -> Vec<Value> {
        let mut frames = Vec::new();
        while let Ok(frame) = self.outbound.try_recv() {
            if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                frames.push(value);
            }
        }
        frames
    }
}

impl Drop for ParityFixture {
    fn drop(&mut self) {
        self.provider.unregister();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The fixture guard: the daemon `prompt` path must run the dialog (the existing
/// prompt-adapter repair reroutes extension commands into the full `session.prompt`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t08_daemon_prompt_runs_the_extension_dialog() {
    let mut fixture = ParityFixture::new("t08-extension").await;
    let command = serde_json::json!({
        "type": "prompt",
        "id": "prompt-1",
        "activeSessionId": fixture.active_session_id,
        "message": format!("/{DIALOG_COMMAND} now"),
    });
    fixture
        .daemon
        .handle_line(Arc::clone(&fixture.client), fixture.command(command))
        .await;
    assert!(
        fixture.barrier.wait_started(Duration::from_secs(5)).await,
        "the dialog never started, so the daemon prompt path is not on the extension route"
    );
    let response = fixture.wait_response("prompt-1", Duration::from_secs(10)).await;
    fixture.barrier.complete();
    let response = response.expect("daemon prompt must acknowledge the extension command");
    assert_eq!(response["success"], true, "{response}");
    assert!(
        fixture.barrier.wait_finished(Duration::from_secs(5)).await,
        "the dialog must finish once the user completes it"
    );
}

/// C-01: the daemon `prompt_and_wait` RPC must drive the extension dialog and settle
/// after the documented completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t08_extension_command_dialog_runs_through_daemon_prompt_and_wait() {
    let mut fixture = ParityFixture::new("t08-extension").await;
    let command = serde_json::json!({
        "type": "prompt_and_wait",
        "id": "wait-1",
        "activeSessionId": fixture.active_session_id,
        "message": format!("/{DIALOG_COMMAND} now"),
    });
    // The daemon serves `prompt_and_wait` inline (`handle_line` awaits it), so a request
    // that never settles must not wedge the test: drive it on a task and bound every
    // observation. The hang is part of the finding, not a test-harness problem.
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let line = fixture.command(command);
    let request = tokio::spawn(async move {
        daemon.handle_line(client, line).await;
    });

    let started = fixture.barrier.wait_started(Duration::from_secs(5)).await;
    fixture.barrier.complete();
    let response = fixture.wait_response("wait-1", Duration::from_secs(10)).await;
    assert!(
        started,
        "prompt_and_wait dropped the lazy extension completion: the dialog never ran"
    );
    let response =
        response.expect("prompt_and_wait never settled: the RPC is left hanging forever");
    assert_eq!(response["success"], true, "{response}");
    assert!(
        fixture.barrier.wait_finished(Duration::from_secs(5)).await,
        "the dialog must finish once the user completes it"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(5), request)
            .await
            .is_ok(),
        "the daemon never finished serving prompt_and_wait"
    );
}


// ---------------------------------------------------------------------------
// C-02 / H-05: a daemon that holds one scripted session.
// ---------------------------------------------------------------------------

/// A daemon with one resident, scripted session plus a fake attached client whose
/// outbound frames the test reads. No process, no pipe, no real session.
pub(super) struct ScriptedDaemonFixture {
    pub root: std::path::PathBuf,
    pub daemon: Arc<AgentDaemon>,
    pub state: Arc<StdMutex<ActiveSessionState>>,
    pub active_session_id: String,
    pub session_id: String,
    pub outbound: mpsc::UnboundedReceiver<Vec<u8>>,
    pub client: Arc<DaemonClientHandle>,
}

impl ScriptedDaemonFixture {
    pub fn new(case: &str, session: Arc<ReportingSession>) -> Self {
        let root = state_root(case);
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let session_file = root.join("sessions").join("scripted.jsonl");
        std::fs::create_dir_all(session_file.parent().unwrap()).expect("sessions dir");
        std::fs::write(&session_file, "{}\n").expect("session file");
        let socket_path = root.join("daemon-parity.sock").to_string_lossy().into_owned();
        let daemon = AgentDaemon::new(
            socket_path.clone(),
            DaemonModeOptions {
                socket_path: Some(socket_path),
                default_session_config: AgentSessionRuntimeConfig {
                    cwd: Some(root.join("workspace").to_string_lossy().into_owned()),
                    agent_dir: Some(agent_dir.to_string_lossy().into_owned()),
                    ..Default::default()
                },
                create_runtime: Arc::new(|_| {
                    Box::pin(async { Err("parity fixture does not create runtimes".to_string()) })
                }),
                worker: None,
            },
        );
        let active_session_id = session.active_session_id.clone();
        let session_id = session.session_id.clone();
        let state = Arc::new(StdMutex::new(ActiveSessionState::new(
            active_session_id.clone(),
            crate::modes::daemon::active_session_state::AgentSessionRuntime {
                session: ActiveSessionRuntimeSession {
                    session_id: session_id.clone(),
                    session_name: Some(active_session_id.clone()),
                    session_file: session.session_file(),
                    ..ActiveSessionRuntimeSession::default()
                },
                metadata: Some(AgentSessionRuntimeMetadata {
                    kind: Some("top-level".to_string()),
                    ..AgentSessionRuntimeMetadata::default()
                }),
                model_fallback_message: None,
            },
        )));
        daemon.sessions.lock().expect("sessions poisoned").insert(
            active_session_id.clone(),
            Arc::new(DaemonSessionState {
                state: Arc::clone(&state),
                session: Arc::clone(&session) as Arc<dyn DaemonSession>,
                runtime_metadata: AgentSessionRuntimeMetadata {
                    kind: Some("top-level".to_string()),
                    ..AgentSessionRuntimeMetadata::default()
                },
                snapshot_boundary: StdMutex::new(None),
            }),
        );
        let (sender, outbound) = mpsc::unbounded_channel();
        let client = Arc::new(DaemonClientHandle::new(
            "parity-client",
            Arc::new(DaemonClientWriter::new(sender)),
            Arc::new(|| {}),
        ));
        {
            let mut client_state = client.state.lock().expect("daemon client poisoned");
            client_state.authenticated = Some(true);
            client_state.transport = Some("jsonl".to_string());
            client_state.capabilities = normalize_client_capabilities(None, None);
        }
        daemon
            .clients
            .lock()
            .expect("clients poisoned")
            .push(Arc::clone(&client));
        Self {
            root,
            daemon,
            state,
            active_session_id,
            session_id,
            outbound,
            client,
        }
    }

    /// Drive one command line through the real daemon dispatcher.
    pub async fn send(&self, value: Value) {
        self.daemon
            .handle_line(Arc::clone(&self.client), serialize_json_line(&value))
            .await;
    }

    pub async fn wait_response(&mut self, id: &str, timeout: Duration) -> Option<Value> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let frame = match tokio::time::timeout(remaining, self.outbound.recv()).await {
                Ok(Some(frame)) => frame,
                _ => return None,
            };
            if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                if value.get("type").and_then(Value::as_str) == Some("response")
                    && value.get("id").and_then(Value::as_str) == Some(id)
                {
                    return Some(value);
                }
            }
        }
    }
}

impl Drop for ScriptedDaemonFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}


// ---------------------------------------------------------------------------
// C-02 / H-05 fixture: a `DaemonSession` whose admission call the test controls.
// Every other member delegates to `MissingSession` (the daemon's own
// not-resident double), so only the admission seam is scripted.
// ---------------------------------------------------------------------------

pub(super) struct ReportingSession {
    pub inner: MissingSession,
    pub active_session_id: String,
    pub session_id: String,
    pub session_file: String,
    /// Overrides `is_session_active` for the recovery-busy predicate test; the
    /// default (`true`) keeps every existing fixture resident.
    pub session_active: bool,
    /// Overrides `has_running_rlm_children` for the recovery-busy predicate test.
    pub running_children: bool,
    /// `(reportSucceeded, reportQueued)` - what the session reports at its own preflight.
    pub report: Option<(bool, bool)>,
    /// The admission call's own outcome (a thrown admission error when `Some`).
    pub error: Option<String>,
    pub calls: Arc<AtomicU64>,
    /// Admission gate for the disconnect test: when `gate` is set, the admission
    /// blocks until the test notifies it.
    pub gate: AtomicBool,
    pub gate_notify: StdMutex<Option<Arc<tokio::sync::Notify>>>,
    /// Counters the disconnect test reads.
    pub entered: Arc<AtomicU64>,
    pub finished: Arc<AtomicU64>,
}

/// Calls the daemon-side preflight callback whatever arity the tree's
/// `PromptInvocation.preflight_result` has: the baseline seam reports only the
/// outcome (`Fn(bool)`), the widened one also reports the queue disposition
/// (`Fn(bool, bool)`). Keeping the double arity-agnostic lets one test text run in
/// both phases.
pub(super) trait ReportPreflight {
    fn report(&self, succeeded: bool, queued: bool);
}

impl ReportPreflight for Arc<dyn Fn(bool) + Send + Sync> {
    fn report(&self, succeeded: bool, _queued: bool) {
        (*self)(succeeded);
    }
}

impl ReportPreflight for Arc<dyn Fn(bool, bool) + Send + Sync> {
    fn report(&self, succeeded: bool, queued: bool) {
        (*self)(succeeded, queued);
    }
}

impl ReportingSession {
    /// Report the recorded preflight verdict (when the fixture models one) and then
    /// return the recorded outcome. The real session reports at its own once-guarded
    /// preflight; the daemon must translate both flags.
    fn scripted_admission(
        &self,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let report = self.report;
        let error = self.error.clone();
        let calls = Arc::clone(&self.calls);
        let gate = self.gate.load(Ordering::SeqCst);
        let gate_notify = self.gate_notify.lock().unwrap().clone();
        let entered = Arc::clone(&self.entered);
        let finished = Arc::clone(&self.finished);
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            entered.fetch_add(1, Ordering::SeqCst);
            if gate {
                if let Some(gate_notify) = gate_notify {
                    gate_notify.notified().await;
                }
            }
            if let (Some(preflight), Some(report)) = (&options.preflight_result, report) {
                preflight.report(report.0, report.1);
            }
            finished.fetch_add(1, Ordering::SeqCst);
            match error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        })
    }

    pub fn arc(
        active_session_id: &str,
        session_id: &str,
        session_file: &str,
        report: Option<(bool, bool)>,
        error: Option<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: MissingSession::new(active_session_id),
            active_session_id: active_session_id.to_string(),
            session_id: session_id.to_string(),
            session_file: session_file.to_string(),
            session_active: true,
            running_children: false,
            report,
            error,
            calls: Arc::new(AtomicU64::new(0)),
            gate: AtomicBool::new(false),
            gate_notify: StdMutex::new(None),
            entered: Arc::new(AtomicU64::new(0)),
            finished: Arc::new(AtomicU64::new(0)),
        })
    }
}

impl DaemonSession for ReportingSession {
    fn session_id(&self) -> String {
        self.session_id.clone()
    }
    fn session_name(&self) -> Option<String> {
        Some(self.active_session_id.clone())
    }
    fn session_file(&self) -> Option<String> {
        Some(self.session_file.clone())
    }
    fn is_session_active(&self) -> bool {
        self.session_active
    }
    fn unfinished_action_count(&self) -> f64 {
        0.0
    }
    /// The admission seam under test: reports the recorded preflight verdict (when the
    /// fixture models a session that reports one) and then returns the recorded outcome.
    fn accept_agent_message_prompt(
        &self,
        _message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        self.scripted_admission(options)
    }

    /// The daemon `prompt` command routes non-agent-message prompts here
    /// (daemon-mode.rs `handle_prompt_command`), so the same scripted report applies.
    fn prompt_until_accepted(
        &self,
        _message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        self.scripted_admission(options)
    }

    /// The daemon `prompt_and_wait` command routes here.
    fn prompt_and_wait(
        &self,
        _message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        self.scripted_admission(options)
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
        self.inner.set_exec_env_provider(client_env);
    }
    fn set_runtime_env_scope(&self, client_env: Option<HashMap<String, String>>) {
        self.inner.set_runtime_env_scope(client_env);
    }
    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {
        self.inner.set_subagent_runtime_host(host);
    }
    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {
        self.inner.set_rebind_session(rebind);
    }
    fn bind_extensions(&self, binding: crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput) -> BoxFuture<'static, Result<(), String>> {
        self.inner.bind_extensions(binding)
    }
    fn abort_for_update_restart(&self) {
        self.inner.abort_for_update_restart();
    }
    fn is_streaming(&self) -> bool {
        self.inner.is_streaming()
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
        self.inner.has_running_rlm_children() || self.running_children
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
        self.inner.set_current_recap(recap);
    }
    fn set_session_name(&self, name: &str) {
        self.inner.set_session_name(name);
    }
    fn get_rlm_child_run_status(&self, child_id: &str) -> Option<String> {
        self.inner.get_rlm_child_run_status(child_id)
    }
    fn register_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> bool {
        self.inner.register_rlm_child_session(child_id, session)
    }
    fn remove_queued_follow_up(&self, key: &str) {
        self.inner.remove_queued_follow_up(key);
    }
    fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.inner.subscribe(listener)
    }
    fn prompt_heartbeat(&self, job: &AgentCronJob, options: PromptInvocation) -> BoxFuture<'static, Result<(), String>> {
        self.inner.prompt_heartbeat(job, options)
    }
    fn steer(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<(), String>> {
        self.inner.steer(message, images, options)
    }
    fn follow_up(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<bool, String>> {
        self.inner.follow_up(message, images, options)
    }
    fn restore_steering_message(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<(), String>> {
        self.inner.restore_steering_message(message, images, options)
    }
    fn restore_follow_up_message(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<bool, String>> {
        self.inner.restore_follow_up_message(message, images, options)
    }
    fn restore_pending_next_turn_messages(&self, messages: &Value) {
        self.inner.restore_pending_next_turn_messages(messages);
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
    fn mutate_queued_message(&self, lane: &str, index: f64, expected_text: &str, mutation: &Value) -> Value {
        self.inner.mutate_queued_message(lane, index, expected_text, mutation)
    }
    fn get_steering_message_previews(&self) -> Vec<Value> {
        self.inner.get_steering_message_previews()
    }
    fn get_follow_up_message_previews(&self) -> Vec<Value> {
        self.inner.get_follow_up_message_previews()
    }
    fn request_abort(&self) {
        self.inner.request_abort();
    }
    fn cancel_rlm_child_run(&self, child_id: &str) -> bool {
        self.inner.cancel_rlm_child_run(child_id)
    }
    fn delete_inactive_rlm_subagent(&self, child_id: &str, is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>,) -> BoxFuture<'static, Result<String, String>> {
        self.inner.delete_inactive_rlm_subagent(child_id, is_resident_child_running)
    }
    fn run_user_bash(&self, command: &str, options: RunUserBashOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.run_user_bash(command, options)
    }
    fn execute_bash(&self, command: &str) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.execute_bash(command)
    }
    fn abort_bash(&self) {
        self.inner.abort_bash();
    }
    fn acquire_session_input_pause(&self) -> SessionInputPause {
        self.inner.acquire_session_input_pause()
    }
    fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        self.inner.wait_for_idle()
    }
    fn wait_for_headless_completion(&self, options: HeadlessCompletionOptions) -> BoxFuture<'static, Result<Value, String>> {
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
    fn set_model(&self, model: &pi_ai::types::Model, wait_for_extensions: bool) -> BoxFuture<'static, Result<(), String>> {
        self.inner.set_model(model, wait_for_extensions)
    }
    fn cycle_model(&self, direction: &str, wait_for_extensions: bool) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>> {
        self.inner.cycle_model(direction, wait_for_extensions)
    }
    fn set_scoped_models(&self, scoped_models: &Value) {
        self.inner.set_scoped_models(scoped_models);
    }
    fn set_thinking_level(&self, level: &str) {
        self.inner.set_thinking_level(level);
    }
    fn set_service_tier(&self, service_tier: &str) {
        self.inner.set_service_tier(service_tier);
    }
    fn cycle_thinking_level(&self) -> Option<String> {
        self.inner.cycle_thinking_level()
    }
    fn set_transport(&self, transport: &str) {
        self.inner.set_transport(transport);
    }
    fn set_steering_mode(&self, mode: &str) {
        self.inner.set_steering_mode(mode);
    }
    fn set_follow_up_mode(&self, mode: &str) {
        self.inner.set_follow_up_mode(mode);
    }
    fn set_auto_compaction_enabled(&self, enabled: bool) {
        self.inner.set_auto_compaction_enabled(enabled);
    }
    fn set_auto_retry_enabled(&self, enabled: bool) {
        self.inner.set_auto_retry_enabled(enabled);
    }
    fn compact(&self, custom_instructions: Option<&str>) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.compact(custom_instructions)
    }
    fn refine(&self, options: RefineOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.refine(options)
    }
    fn abort_compaction(&self) {
        self.inner.abort_compaction();
    }
    fn abort_branch_summary(&self) {
        self.inner.abort_branch_summary();
    }
    fn abort_retry(&self) {
        self.inner.abort_retry();
    }
    fn reload(&self) -> BoxFuture<'static, Result<(), String>> {
        self.inner.reload()
    }
    fn get_rlm_max_depth_status(&self) -> Value {
        self.inner.get_rlm_max_depth_status()
    }
    fn set_rlm_max_depth(&self, max_depth: Value, global: bool) -> BoxFuture<'static, Result<Value, String>> {
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
    fn export_to_html(&self, output_path: Option<&str>) -> BoxFuture<'static, Result<String, String>> {
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
    fn navigate_tree(&self, target_id: &str, options: NavigateTreeOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.navigate_tree(target_id, options)
    }
    fn start_side_question(&self, question: &str, options: SideQuestionOptions) -> BoxFuture<'static, Result<(), String>> {
        self.inner.start_side_question(question, options)
    }
    fn abort_side_question(&self, side_question_id: &str) {
        self.inner.abort_side_question(side_question_id);
    }
    fn release_acp_mcp_servers(&self, owner_id: &str, server_names: &[String]) -> BoxFuture<'static, Result<(), String>> {
        self.inner.release_acp_mcp_servers(owner_id, server_names)
    }
    fn replace_acp_mcp_servers(&self, servers: &[Value], owner_id: &str) -> BoxFuture<'static, Result<(), String>> {
        self.inner.replace_acp_mcp_servers(servers, owner_id)
    }
    fn new_session(&self, options: Option<NewSessionRuntimeOptions>) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.new_session(options)
    }
    fn release_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> Option<Box<dyn FnOnce() + Send>> {
        self.inner.release_rlm_child_session(child_id, session)
    }
    fn replied_to_parent_since_task(&self) -> Option<bool> {
        self.inner.replied_to_parent_since_task()
    }
    fn switch_session(&self, session_path: &str, options: SessionPathOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.switch_session(session_path, options)
    }
    fn fork(&self, entry_id: &str, options: ForkOptions) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.fork(entry_id, options)
    }
    fn import_from_jsonl(&self, input_path: &str, cwd_override: Option<&str>) -> BoxFuture<'static, Result<Value, String>> {
        self.inner.import_from_jsonl(input_path, cwd_override)
    }
    fn dispose(&self) -> BoxFuture<'static, ()> {
        self.inner.dispose()
    }
}


// ---------------------------------------------------------------------------
// C-02: receipt outcomes are truthful
// ---------------------------------------------------------------------------

/// C-02 (a): a session that reports `preflight(false, false)` (rejected admission)
/// must not be recorded as delivered. TS `acceptAgentSessionMessage`
/// (daemon-mode.ts:6401-6409) throws `"Agent message was not accepted"`; the port
/// ignores the report and returns DELIVERED.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t08_receipt_outcomes_are_truthful_queued_is_not_delivered() {
    let session = ReportingSession::arc(
        "active-queued",
        "session-queued",
        "C:/parity/session-queued.jsonl",
        Some((true, true)),
        None,
    );
    let mut fixture = ScriptedDaemonFixture::new("t08-receipts", Arc::clone(&session));

    let message = fixture
        .daemon
        .send_agent_session_message(SendAgentMessageInput {
            target_selector: fixture.active_session_id.clone(),
            message: "queued case".to_string(),
            from_state: None,
            sender: None,
            client_id: Some("parity-client".to_string()),
            sender_key: None,
            origin: "cli".to_string(),
        })
        .await
        .expect("the send itself must return a receipt");

    assert_eq!(
        message.delivery_status,
        DELIVERY_STATUS_QUEUED,
        "a session that reported (true, true) queued the message; the receipt must say queued"
    );
    assert!(
        message.delivered_at.is_none(),
        "a queued message must not carry deliveredAt: {message:?}"
    );
    assert!(
        message.queued_at.is_some(),
        "a queued message must carry queuedAt: {message:?}"
    );
}

/// C-02 (b): a session that reports `preflight(false, false)` (rejected admission)
/// must fail the send instead of returning a fabricated delivery.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t08_receipt_outcomes_are_truthful_rejected_admission_fails() {
    let session = ReportingSession::arc(
        "active-rejected",
        "session-rejected",
        "C:/parity/session-rejected.jsonl",
        Some((false, false)),
        None,
    );
    let _fixture = ScriptedDaemonFixture::new("t08-receipts", session);

    let result = _fixture
        .daemon
        .send_agent_session_message(SendAgentMessageInput {
            target_selector: _fixture.active_session_id.clone(),
            message: "rejected case".to_string(),
            from_state: None,
            sender: None,
            client_id: Some("parity-client".to_string()),
            sender_key: None,
            origin: "cli".to_string(),
        })
        .await;

    match result {
        Ok(message) => panic!(
            "a rejected admission was reported as a successful receipt: {message:?}"
        ),
        Err(error) => assert!(
            error.contains("not accepted"),
            "the rejection reason must reach the sender, got: {error}"
        ),
    }
}

/// C-02 (c): a genuinely thrown admission error must not be reported as queued.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t08_receipt_outcomes_are_truthful_thrown_admission_error_is_an_error() {
    let session = ReportingSession::arc(
        "active-thrown",
        "session-thrown",
        "C:/parity/session-thrown.jsonl",
        None,
        Some("Cannot admit a session action while session input admission is paused.".to_string()),
    );
    let fixture = ScriptedDaemonFixture::new("t08-receipts", session);

    let result = fixture
        .daemon
        .send_agent_session_message(SendAgentMessageInput {
            target_selector: fixture.active_session_id.clone(),
            message: "thrown case".to_string(),
            from_state: None,
            sender: None,
            client_id: Some("parity-client".to_string()),
            sender_key: None,
            origin: "cli".to_string(),
        })
        .await;

    match result {
        Ok(message) => panic!(
            "a thrown admission error was reported as a receipt ({:?}) instead of an error",
            message.delivery_status
        ),
        Err(error) => assert!(
            error.contains("paused"),
            "the thrown admission error must reach the sender, got: {error}"
        ),
    }
}

/// C-02 (d) + H-05: a `prompt` the session rejects must send the requesting client a
/// failure response (TS daemon-mode.ts:4545-4558), not silence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t08_rejected_prompt_reports_failure_to_the_requesting_client() {
    let session = ReportingSession::arc(
        "active-prompt-rejected",
        "session-prompt-rejected",
        "C:/parity/session-prompt-rejected.jsonl",
        Some((false, false)),
        None,
    );
    // The daemon `prompt` command path dispatches through `prompt_until_accepted`, so
    // the fixture's admission seam is reached from there.
    let mut fixture = ScriptedDaemonFixture::new("t08-receipts", session);
    fixture
        .send(serde_json::json!({
            "type": "prompt",
            "id": "prompt-rejected",
            "activeSessionId": fixture.active_session_id,
            "message": "hello",
        }))
        .await;

    let response = fixture.wait_response("prompt-rejected", Duration::from_secs(5)).await;
    match response {
        Some(response) => assert_eq!(
            response["success"], false,
            "a rejected prompt must be reported as a failure: {response}"
        ),
        None => panic!(
            "the requesting client received no response for a rejected prompt (H-05: the \
             preflight rejection is swallowed and the client waits for the timeout)"
        ),
    }
}


// ---------------------------------------------------------------------------
// C-09: delivery is reported only after the ticket's delivery settles
// ---------------------------------------------------------------------------

/// Build one assistant reply for the faux provider.
fn assistant_text(model: &pi_ai::types::Model, text: &str) -> pi_ai::types::AssistantMessage {
    let mut message = pi_ai::types::AssistantMessage::default();
    message.api = model.api.clone();
    message.provider = model.provider.clone();
    message.model = model.id.clone();
    message.content = vec![pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new(
        text,
    ))];
    message.stop_reason = pi_ai::types::STOP_REASON_STOP.to_string();
    message
}

/// C-09: `reportPreflight(true, false)` asserts the message was delivered. TS
/// (agent-session.ts:5636-5643) only reports that verdict from inside
/// `result.ticket.delivered.then(...)`, so delivery must already have settled.
/// The port reports immediately after admission (agent_session.rs:8359-8363).
///
/// This is a race-window finding, so it is measured as a rate over bounded
/// repetitions; a single pass is not evidence either way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t08_delivery_reporting_waits_for_ticket_delivery() {
    const ROUNDS: usize = 10;
    let fixture = ParityFixture::new("t08-receipts").await;
    let model = fixture.provider.get_model();
    let mut reports = 0usize;
    let mut violations = 0usize;
    let mut unfinished_rounds = 0usize;

    for round in 0..ROUNDS {
        fixture
            .provider
            .append_responses(vec![pi_ai::providers::faux::FauxResponseStep::Message(
                assistant_text(&model, "ack"),
            )]);
        let id = format!("c09-probe-{round}");
        let delivery = fixture
            .session
            .wait_for_agent_message_prompt_delivery(&id);
        let delivery_at_report = Arc::new(StdMutex::new(None));
        let observed = Arc::clone(&delivery_at_report);
        let probe = delivery.clone();
        let result = fixture
            .session
            .prompt_until_accepted(
                "delivery probe",
                Some(crate::core::agent_session::PromptOptions {
                    agent_message_id: Some(id.clone()),
                    expand_prompt_templates: Some(false),
                    skip_input_handlers: Some(true),
                    preflight_result: Some(Arc::new(move |succeeded: bool, queued: bool| {
                        if succeeded && !queued {
                            *observed.lock().unwrap() = Some(probe.is_settled());
                        }
                    })),
                    ..Default::default()
                }),
            )
            .await;
        assert!(
            result.is_ok(),
            "round {round}: the probe prompt did not run, so the measurement is not on the bug path: {result:?}"
        );
        // A truthful implementation reports from `delivered.then(...)`, so the verdict can
        // arrive AFTER `prompt_until_accepted` returned. Wait for it with a bound instead of
        // sampling once: sampling too early would hide the bug's own signature (the immediate
        // report) or report nothing at all for the fixed tree.
        let report_deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while tokio::time::Instant::now() < report_deadline
            && delivery_at_report.lock().unwrap().is_none()
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        match delivery_at_report.lock().unwrap().clone() {
            Some(settled) => {
                reports += 1;
                if !settled {
                    violations += 1;
                }
            }
            None => unfinished_rounds += 1,
        }
        let _ = tokio::time::timeout(
            Duration::from_secs(10),
            fixture.session.wait_for_session_input_idle(),
        )
        .await;
    }

    println!(
        "T08_C09_PROBE rounds={ROUNDS} reports={reports} violations={violations} \
         unobserved={unfinished_rounds}"
    );
    assert_eq!(
        unfinished_rounds, 0,
        "the delivery report was never observed; this measurement cannot see the bug path"
    );
    assert_eq!(
        violations, 0,
        "reported delivery {violations}/{reports} times before the ticket's delivery settled; \
         the verdict must come from delivered.then(...) (agent-session.ts:5639-5642)"
    );
}

// ---------------------------------------------------------------------------
// T08: disconnect during admission
// ---------------------------------------------------------------------------

/// T08 `disconnect_during_admission`: a client that goes away while an admission is
/// still in flight must not lose the admission, double-deliver, or leave the daemon
/// wedged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t08_disconnect_during_admission_keeps_admission_observable() {
    let session = ReportingSession::arc(
        "active-disconnect",
        "session-disconnect",
        "C:/parity/session-disconnect.jsonl",
        Some((true, false)),
        None,
    );
    // Admission waits at the gate until the test releases it.
    let gate = Arc::new(tokio::sync::Notify::new());
    session.gate.store(true, Ordering::SeqCst);
    session.gate_notify.lock().unwrap().replace(Arc::clone(&gate));
    let mut fixture = ScriptedDaemonFixture::new("t08-disconnect", Arc::clone(&session));

    fixture
        .send(serde_json::json!({
            "type": "prompt",
            "id": "disconnect-1",
            "activeSessionId": fixture.active_session_id,
            "admissionId": "adm-disconnect-1",
            "message": "in flight",
        }))
        .await;
    // Wait until the spawned handler really entered the session admission.
    let entered = {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline
            && session.entered.load(Ordering::SeqCst) == 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        session.entered.load(Ordering::SeqCst)
    };
    assert_eq!(entered, 1, "the prompt handler never reached the session admission");

    // Controlled disconnect: the client's socket is gone before the response is due.
    fixture.client.writer.end();
    gate.notify_waiters();
    gate.notify_one();

    let finished = {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline
            && session.finished.load(Ordering::SeqCst) == 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        session.finished.load(Ordering::SeqCst)
    };
    assert_eq!(
        finished, 1,
        "the admission future was dropped without being driven to completion (unobserved future)"
    );
    assert_eq!(
        session.calls.load(Ordering::SeqCst),
        1,
        "the admission ran more than once for one command"
    );
    // The admission bookkeeping must be released even though the client left.
    assert!(
        fixture
            .daemon
            .prompt_admissions
            .lock()
            .expect("prompt admissions poisoned")
            .is_empty(),
        "the prompt admission leaked after the client disconnected"
    );

    // The daemon must still serve a later command through the same state.
    let (sender, mut outbound) = mpsc::unbounded_channel();
    let client = Arc::new(DaemonClientHandle::new(
        "parity-client-2",
        Arc::new(DaemonClientWriter::new(sender)),
        Arc::new(|| {}),
    ));
    {
        let mut client_state = client.state.lock().expect("daemon client poisoned");
        client_state.authenticated = Some(true);
        client_state.transport = Some("jsonl".to_string());
        client_state.capabilities = normalize_client_capabilities(None, None);
    }
    fixture
        .daemon
        .clients
        .lock()
        .expect("clients poisoned")
        .push(Arc::clone(&client));
    fixture
        .daemon
        .handle_line(
            Arc::clone(&client),
            serialize_json_line(&serde_json::json!({
                "type": "agent_messages_status",
                "id": "after-disconnect",
            })),
        )
        .await;
    let response = tokio::time::timeout(Duration::from_secs(5), outbound.recv())
        .await
        .expect("the daemon stopped answering after a client disconnect")
        .expect("a frame");
    let value: Value = serde_json::from_slice(&response).expect("json frame");
    assert_eq!(
        value["id"], "after-disconnect",
        "the post-disconnect command was not answered: {value}"
    );
}

/// The recovery-journal busy predicate must match the TypeScript
/// `recordWorkerRecoveryState` busy expression (daemon-mode.ts:7375-7380):
/// `hasLiveSessionWork(state) || isRetrying || hasAcceptedPromptInFlight`,
/// where `hasLiveSessionWork` is `isSessionActive || hasRunningRlmChildren`.
/// A settled session with running children is busy; a settled session with no
/// children and no foreground work is not (audit BUSY-FLAG-01 parity delta).
#[test]
fn recovery_busy_matches_the_typescript_predicate() {
    let file = "parity-recovery-busy.jsonl";
    let active = ReportingSession::arc("active", "saved", file, None, None);
    assert!(
        crate::modes::daemon::daemon_mode::worker_recovery_busy(active.as_ref()),
        "a session-active session is busy"
    );

    let settled_with_children = {
        // The `arc` constructor pins `session_active`/`running_children` for the
        // existing fixtures; the predicate test builds the settled-with-children
        // variant directly.
        let variant = ReportingSession {
            inner: MissingSession::new("active"),
            active_session_id: "active".to_string(),
            session_id: "saved".to_string(),
            session_file: file.to_string(),
            session_active: false,
            running_children: true,
            report: None,
            error: None,
            calls: Arc::new(AtomicU64::new(0)),
            gate: AtomicBool::new(false),
            gate_notify: StdMutex::new(None),
            entered: Arc::new(AtomicU64::new(0)),
            finished: Arc::new(AtomicU64::new(0)),
        };
        variant
    };
    assert!(
        !settled_with_children.is_session_active(),
        "the fixture must model a settled session"
    );
    assert!(
        crate::modes::daemon::daemon_mode::worker_recovery_busy(&settled_with_children),
        "running rlm children keep the recovery record busy even while the parent is settled"
    );

    let settled = {
        ReportingSession {
            inner: MissingSession::new("active"),
            active_session_id: "active".to_string(),
            session_id: "saved".to_string(),
            session_file: file.to_string(),
            session_active: false,
            running_children: false,
            report: None,
            error: None,
            calls: Arc::new(AtomicU64::new(0)),
            gate: AtomicBool::new(false),
            gate_notify: StdMutex::new(None),
            entered: Arc::new(AtomicU64::new(0)),
            finished: Arc::new(AtomicU64::new(0)),
        }
    };
    assert!(
        !crate::modes::daemon::daemon_mode::worker_recovery_busy(&settled),
        "a fully settled session with no children is not busy"
    );
}

// ---------------------------------------------------------------------------
// B1 (UI-002/UI-003 rev4): after EVERY replacement/resync delivery the daemon
// republishes the authoritative Jev footer. The regressions below drive the
// REAL daemon seams (`queue_client_catchup`, `catch_up_backpressured_client`,
// `broadcast_to_session`, `stream_worker_snapshot`) over the real
// `ParityFixture`, so every frame asserted is a real wire frame.
// ---------------------------------------------------------------------------

/// The four env-mutating B1 regressions share the process environment:
/// `JevFooterEnvGuard` re-points `PRIME_AGENT_CODING_AGENT_DIR`, so their
/// set/restore windows take turns (the migrations.rs `ENV_LOCK` pattern).
static B1_ENV_LOCK: StdMutex<()> = StdMutex::new(());

/// Points `crate::config::get_agent_dir()` — and therefore the Jev settings
/// store the footer forms read — at this fixture's private agent dir for the
/// duration of one test, then restores the previous value.
struct JevFooterEnvGuard {
    previous: Option<std::ffi::OsString>,
}

impl JevFooterEnvGuard {
    fn new(agent_dir: &std::path::Path) -> Self {
        let previous = std::env::var_os("PRIME_AGENT_CODING_AGENT_DIR");
        std::fs::create_dir_all(agent_dir).expect("agent dir");
        std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", agent_dir);
        Self { previous }
    }
}

impl Drop for JevFooterEnvGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => std::env::set_var("PRIME_AGENT_CODING_AGENT_DIR", value),
            None => std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR"),
        }
    }
}

/// Writes `<agent>/jev/jev-settings.json` with explicit per-session Jev
/// overrides (decision mode + INDEPENDENT compaction), merged onto the
/// crate's own default settings so the persisted schema stays valid.
fn b1_write_jev_settings(
    agent_dir: &std::path::Path,
    overrides: &[(&str, Option<pi_jev::config::JevMode>, bool)],
) {
    let mut settings = pi_jev::config::JevSettings::default();
    for (session_id, mode, compaction_enabled) in overrides {
        let entry = settings.sessions.entry(session_id.to_string()).or_default();
        entry.mode = *mode;
        entry.compaction_enabled = Some(*compaction_enabled);
    }
    let jev_dir = agent_dir.join("jev");
    std::fs::create_dir_all(&jev_dir).expect("jev dir");
    std::fs::write(
        jev_dir.join(pi_jev::config::SETTINGS_FILE_NAME),
        serde_json::to_string_pretty(&settings).expect("settings serialize"),
    )
    .expect("settings write");
}

/// The decision segment only turns green when a saved credential PRESENCE
/// file exists (`footer_decision_state`); the value is never read.
fn b1_write_credential_presence(agent_dir: &std::path::Path) {
    let jev_dir = agent_dir.join("jev");
    std::fs::create_dir_all(&jev_dir).expect("jev dir");
    std::fs::write(
        jev_dir.join(format!(
            "{}.{}",
            pi_jev::config::DEFAULT_KEY_ID,
            pi_jev::credential::CREDENTIAL_FILE_NAME
        )),
        "{\"presenceOnly\":true}\n",
    )
    .expect("credential presence write");
}

/// Registers a resident scripted session (this file's own `ReportingSession`)
/// into the fixture daemon, mirroring `ScriptedDaemonFixture`'s registration,
/// and returns the daemon state for it.
fn b1_register_scripted_session(
    daemon: &Arc<AgentDaemon>,
    root: &std::path::Path,
    active_session_id: &str,
    session_id: &str,
) -> Arc<StdMutex<ActiveSessionState>> {
    let sessions_dir = root.join("sessions");
    std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
    let session_file = sessions_dir.join(format!("{active_session_id}.jsonl"));
    std::fs::write(&session_file, "{}\n").expect("session file");
    let session = ReportingSession::arc(
        active_session_id,
        session_id,
        &session_file.to_string_lossy(),
        None,
        None,
    );
    let state = Arc::new(StdMutex::new(ActiveSessionState::new(
        active_session_id.to_string(),
        crate::modes::daemon::active_session_state::AgentSessionRuntime {
            session: ActiveSessionRuntimeSession {
                session_id: session_id.to_string(),
                session_name: Some(active_session_id.to_string()),
                session_file: session.session_file(),
                ..ActiveSessionRuntimeSession::default()
            },
            metadata: Some(AgentSessionRuntimeMetadata {
                kind: Some("top-level".to_string()),
                ..AgentSessionRuntimeMetadata::default()
            }),
            model_fallback_message: None,
        },
    )));
    daemon.sessions.lock().expect("sessions poisoned").insert(
        active_session_id.to_string(),
        Arc::new(DaemonSessionState {
            state: Arc::clone(&state),
            session: Arc::clone(&session) as Arc<dyn DaemonSession>,
            runtime_metadata: AgentSessionRuntimeMetadata {
                kind: Some("top-level".to_string()),
                ..AgentSessionRuntimeMetadata::default()
            },
            snapshot_boundary: StdMutex::new(None),
        }),
    );
    state
}

/// Attaches the fixture viewer to one session with `extension_ui` (plus any
/// extra capabilities), mirroring the attach arm's own bookkeeping
/// (`state.clients` + `attached_active_session_ids` + session capabilities).
fn b1_attach_viewer(
    client: &Arc<DaemonClientHandle>,
    state: &Arc<StdMutex<ActiveSessionState>>,
    active_session_id: &str,
    capabilities: &[&str],
) {
    {
        let mut guard = state.lock().expect("active session poisoned");
        if !guard
            .clients
            .iter()
            .any(|candidate| Arc::ptr_eq(candidate, &client.state))
        {
            guard.clients.push(Arc::clone(&client.state));
        }
    }
    client
        .state
        .lock()
        .expect("daemon client poisoned")
        .attached_active_session_ids
        .insert(active_session_id.to_string());
    let requested: HashSet<String> = capabilities.iter().map(|name| name.to_string()).collect();
    set_daemon_client_session_capabilities(
        client,
        active_session_id,
        normalize_client_capabilities(Some(&requested), Some(true)),
    );
}

/// Decodes the private frames a private-framed fixture client received into
/// their JSON payloads (the wire format `encode_private_frame` writes).
fn b1_decode_private_frames(buffers: &[Vec<u8>]) -> Vec<Value> {
    let mut frames = Vec::new();
    for buffer in buffers {
        let mut offset = 0usize;
        while buffer.len().saturating_sub(offset) >= 8 {
            let header_length =
                u32::from_be_bytes(buffer[offset..offset + 4].try_into().expect("4 bytes"))
                    as usize;
            let payload_length =
                u32::from_be_bytes(buffer[offset + 4..offset + 8].try_into().expect("4 bytes"))
                    as usize;
            let payload_start = offset + 8 + header_length;
            if buffer.len() < payload_start + payload_length {
                break;
            }
            if let Ok(value) = serde_json::from_slice::<Value>(
                &buffer[payload_start..payload_start + payload_length],
            ) {
                frames.push(value);
            }
            offset = payload_start + payload_length;
        }
    }
    frames
}

/// Every `setStatus` footer frame a client received, as
/// (statusKey, statusText, statusCompactText) tuples, in wire order.
fn b1_footer_pairs(frames: &[Value]) -> Vec<(String, String, String)> {
    frames
        .iter()
        .filter(|frame| {
            frame.get("type").and_then(Value::as_str) == Some("extension_ui_request")
                && frame.get("method").and_then(Value::as_str) == Some("setStatus")
        })
        .filter_map(|frame| {
            let payload = frame.get("payload")?;
            Some((
                payload
                    .get("statusKey")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                payload
                    .get("statusText")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                payload
                    .get("statusCompactText")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ))
        })
        .collect()
}

fn b1_frame_index(frames: &[Value], frame_type: &str) -> Option<usize> {
    frames
        .iter()
        .position(|frame| frame.get("type").and_then(Value::as_str) == Some(frame_type))
}

fn b1_footer_positions(frames: &[Value]) -> Vec<usize> {
    frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| {
            frame.get("type").and_then(Value::as_str) == Some("extension_ui_request")
        })
        .map(|(index, _)| index)
        .collect()
}

/// B1a: a backpressured viewer converges through catch-up, and the repushed
/// footer carries the NEW session's effective values. Session A (Compare +
/// compaction on) is replaced by session B (Off + compaction off): A's pair
/// must show A's labels, B's pair must show B's labels and none of A's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_backpressured_replacement_catchup_republishes_the_new_session_footer() {
    let _env_lock = B1_ENV_LOCK.lock().unwrap();
    crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let case = "b1-replacement-catchup";
    let root = state_root(case);
    let _guard = JevFooterEnvGuard::new(&root.join("agent"));
    let mut fixture = ParityFixture::new(case).await;
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let session_a_id = daemon.session_of(&fixture.state).session_id();
    let session_b_active = "b1-session-b";
    let session_b_id = "b1-session-b-durable";
    b1_write_jev_settings(
        &root.join("agent"),
        &[
            (&session_a_id, Some(pi_jev::config::JevMode::Compare), true),
            (session_b_id, Some(pi_jev::config::JevMode::Off), false),
        ],
    );
    b1_write_credential_presence(&root.join("agent"));
    let state_b = b1_register_scripted_session(&daemon, &root, session_b_active, session_b_id);
    b1_attach_viewer(
        &client,
        &fixture.state,
        &fixture.active_session_id,
        &["extension_ui"],
    );
    b1_attach_viewer(&client, &state_b, session_b_active, &["extension_ui"]);

    // The OLD session's catch-up first: its pair must carry A's values.
    client.set_backpressured(true);
    daemon.queue_client_catchup(&client, &fixture.active_session_id, "replacement");
    client.set_backpressured(false);
    daemon
        .catch_up_backpressured_client(Arc::clone(&client))
        .await
        .expect("catch-up drained");
    let frames_a = fixture.drain();
    let replaced_a = b1_frame_index(&frames_a, "session_replaced")
        .expect("A catch-up delivered a replace frame");
    let pairs_a = b1_footer_pairs(&frames_a);
    assert_eq!(
        pairs_a.len(),
        2,
        "decision + compaction segments: {frames_a:?}"
    );
    assert_eq!(
        b1_footer_positions(&frames_a),
        vec![replaced_a + 1, replaced_a + 2],
        "the footer pair must follow the replace frame: {frames_a:?}"
    );
    assert_eq!(pairs_a[0].0, "jev");
    assert!(
        pairs_a[0].1.contains("Jev On (Compare)"),
        "A full decision text: {:?}",
        pairs_a[0].1
    );
    assert!(
        pairs_a[0].2.contains("Jev C On"),
        "A compact decision text: {:?}",
        pairs_a[0].2
    );
    assert_eq!(pairs_a[1].0, "jev-compact");
    assert!(
        pairs_a[1].1.contains("Jev compact on"),
        "A compaction text: {:?}",
        pairs_a[1].1
    );
    assert!(
        pairs_a[1].2.contains("Jev Cmp on"),
        "A compact compaction: {:?}",
        pairs_a[1].2
    );

    // The NEW session's replacement catch-up: B's pair replaces A's labels.
    client.set_backpressured(true);
    daemon.queue_client_catchup(&client, session_b_active, "replacement");
    client.set_backpressured(false);
    daemon
        .catch_up_backpressured_client(Arc::clone(&client))
        .await
        .expect("catch-up drained");
    let frames_b = fixture.drain();
    let replaced_b = b1_frame_index(&frames_b, "session_replaced")
        .expect("B catch-up delivered a replace frame");
    assert_eq!(
        frames_b[replaced_b]
            .get("activeSessionId")
            .and_then(Value::as_str),
        Some(session_b_active)
    );
    let pairs_b = b1_footer_pairs(&frames_b);
    assert_eq!(
        pairs_b.len(),
        2,
        "decision + compaction segments: {frames_b:?}"
    );
    assert_eq!(
        b1_footer_positions(&frames_b),
        vec![replaced_b + 1, replaced_b + 2],
        "the footer pair must follow the replace frame: {frames_b:?}"
    );
    assert_eq!(pairs_b[0].0, "jev");
    assert!(
        pairs_b[0].1.contains("Jev Off"),
        "B full decision text: {:?}",
        pairs_b[0].1
    );
    assert!(
        pairs_b[0].2.contains("Jev Off"),
        "B compact decision text: {:?}",
        pairs_b[0].2
    );
    assert!(
        !pairs_b[0].1.contains("Jev On (Compare)"),
        "no A decision label may leak into B: {:?}",
        pairs_b[0].1
    );
    assert!(
        !pairs_b[0].2.contains("Jev C On"),
        "no A compact decision label may leak into B: {:?}",
        pairs_b[0].2
    );
    assert_eq!(pairs_b[1].0, "jev-compact");
    assert!(
        pairs_b[1].1.contains("Jev compact off"),
        "B compaction text: {:?}",
        pairs_b[1].1
    );
    assert!(
        pairs_b[1].2.contains("Jev Cmp off"),
        "B compact compaction: {:?}",
        pairs_b[1].2
    );
    assert!(
        !pairs_b[1].1.contains("Jev compact on"),
        "no A compaction label may leak into B: {:?}",
        pairs_b[1].1
    );
}

/// B1b: a resync catch-up republishes the footer with the two segments
/// INDEPENDENT: decision Off (red) while the session's own compaction
/// setting is ON (green).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_resync_catchup_republishes_independent_decision_and_compaction() {
    let _env_lock = B1_ENV_LOCK.lock().unwrap();
    crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let case = "b1-resync-catchup";
    let root = state_root(case);
    let _guard = JevFooterEnvGuard::new(&root.join("agent"));
    let mut fixture = ParityFixture::new(case).await;
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let session_c_active = "b1-session-c";
    let session_c_id = "b1-session-c-durable";
    b1_write_jev_settings(
        &root.join("agent"),
        &[(session_c_id, Some(pi_jev::config::JevMode::Off), true)],
    );
    b1_write_credential_presence(&root.join("agent"));
    let state_c = b1_register_scripted_session(&daemon, &root, session_c_active, session_c_id);
    b1_attach_viewer(&client, &state_c, session_c_active, &["extension_ui"]);

    client.set_backpressured(true);
    daemon.queue_client_catchup(&client, session_c_active, "resync");
    client.set_backpressured(false);
    daemon
        .catch_up_backpressured_client(Arc::clone(&client))
        .await
        .expect("catch-up drained");
    let frames = fixture.drain();
    let resynced =
        b1_frame_index(&frames, "session_resynced").expect("the catch-up delivered a resync frame");
    assert_eq!(
        frames[resynced]
            .get("activeSessionId")
            .and_then(Value::as_str),
        Some(session_c_active)
    );
    let pairs = b1_footer_pairs(&frames);
    assert_eq!(pairs.len(), 2, "decision + compaction segments: {frames:?}");
    assert_eq!(
        b1_footer_positions(&frames),
        vec![resynced + 1, resynced + 2],
        "the footer pair must follow the resync frame: {frames:?}"
    );
    assert_eq!(pairs[0].0, "jev");
    assert!(
        pairs[0].1.contains("Jev Off"),
        "decision stays Off: {:?}",
        pairs[0].1
    );
    assert!(
        !pairs[0].1.contains("Jev On"),
        "decision off must never claim on: {:?}",
        pairs[0].1
    );
    assert_eq!(pairs[1].0, "jev-compact");
    assert!(
        pairs[1].1.contains("Jev compact on"),
        "compaction is independently ON: {:?}",
        pairs[1].1
    );
    assert!(
        pairs[1].2.contains("Jev Cmp on"),
        "compact compaction is independently ON: {:?}",
        pairs[1].2
    );
    assert!(
        !pairs[1].2.contains("Jev Cmp off"),
        "compaction on must never render off: {:?}",
        pairs[1].2
    );
}

/// B1c: a chunked-snapshot client receives the footer only AFTER the
/// replacement snapshot stream completes (the client applies the chunked
/// snapshot as the replacement when `session_snapshot_end` lands, and that
/// application resets the status surface).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_chunked_replacement_snapshot_republishes_the_footer_after_the_stream() {
    let _env_lock = B1_ENV_LOCK.lock().unwrap();
    crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let case = "b1-chunked-replacement";
    let root = state_root(case);
    let _guard = JevFooterEnvGuard::new(&root.join("agent"));
    let mut fixture = ParityFixture::new(case).await;
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let session_b_active = "b1-chunked-session-b";
    let session_b_id = "b1-chunked-b-durable";
    b1_write_jev_settings(
        &root.join("agent"),
        &[(session_b_id, Some(pi_jev::config::JevMode::Off), false)],
    );
    b1_write_credential_presence(&root.join("agent"));
    let state_b = b1_register_scripted_session(&daemon, &root, session_b_active, session_b_id);
    {
        let mut client_state = client.state.lock().expect("daemon client poisoned");
        client_state.transport = Some("private-framed".to_string());
    }
    b1_attach_viewer(
        &client,
        &state_b,
        session_b_active,
        &["extension_ui", "chunked_snapshot"],
    );

    client.set_backpressured(true);
    daemon.queue_client_catchup(&client, session_b_active, "replacement");
    client.set_backpressured(false);
    daemon
        .catch_up_backpressured_client(Arc::clone(&client))
        .await
        .expect("catch-up drained");

    let mut buffers = Vec::new();
    while let Ok(frame) = fixture.outbound.try_recv() {
        buffers.push(frame);
    }
    let frames = b1_decode_private_frames(&buffers);
    let replaced = b1_frame_index(&frames, "session_replaced")
        .expect("the catch-up delivered a replace frame announcing the snapshot");
    assert_eq!(
        frames[replaced]
            .get("snapshotFollows")
            .and_then(Value::as_bool),
        Some(true)
    );
    let begin =
        b1_frame_index(&frames, "session_snapshot_begin").expect("the snapshot stream began");
    assert_eq!(
        frames[begin].get("purpose").and_then(Value::as_str),
        Some("replacement")
    );
    let end =
        b1_frame_index(&frames, "session_snapshot_end").expect("the snapshot stream completed");
    assert!(replaced < begin && begin < end, "stream order: {frames:?}");
    let pairs = b1_footer_pairs(&frames);
    assert_eq!(pairs.len(), 2, "decision + compaction segments: {frames:?}");
    assert_eq!(
        b1_footer_positions(&frames),
        vec![end + 1, end + 2],
        "the footer pair must follow the completed snapshot stream: {frames:?}"
    );
    assert_eq!(pairs[0].0, "jev");
    assert!(
        pairs[0].1.contains("Jev Off") && pairs[0].2.contains("Jev Off"),
        "B decision full + compact: {:?} / {:?}",
        pairs[0].1,
        pairs[0].2
    );
    assert_eq!(pairs[1].0, "jev-compact");
    assert!(
        pairs[1].1.contains("Jev compact off") && pairs[1].2.contains("Jev Cmp off"),
        "B compaction full + compact: {:?} / {:?}",
        pairs[1].1,
        pairs[1].2
    );
}

/// B1d: a viewer that is mid-snapshot-stream when the daemon-side replacement
/// lands gets NOTHING inline (the deferred frames are cleared and the
/// replacement converges into the catch-up queue); once the stream finishes,
/// the catch-up delivers the replace frame and the footer pair.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_snapshot_streaming_client_converges_to_catchup_footer() {
    let _env_lock = B1_ENV_LOCK.lock().unwrap();
    crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let case = "b1-deferred-convergence";
    let root = state_root(case);
    let _guard = JevFooterEnvGuard::new(&root.join("agent"));
    let mut fixture = ParityFixture::new(case).await;
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let session_b_active = "b1-deferred-session-b";
    let session_b_id = "b1-deferred-b-durable";
    b1_write_jev_settings(
        &root.join("agent"),
        &[(session_b_id, Some(pi_jev::config::JevMode::Off), false)],
    );
    b1_write_credential_presence(&root.join("agent"));
    let state_b = b1_register_scripted_session(&daemon, &root, session_b_active, session_b_id);
    b1_attach_viewer(&client, &state_b, session_b_active, &["extension_ui"]);

    mark_client_snapshot_streaming(&client, session_b_active);
    let entry = daemon
        .sessions
        .lock()
        .expect("sessions poisoned")
        .get(session_b_active)
        .cloned()
        .expect("registered entry");
    daemon.broadcast_to_session(
        &entry,
        DaemonOutbound::SessionReplaced {
            active_session_id: session_b_active.to_string(),
            state: Value::Null,
            messages: Vec::new(),
        },
    );
    assert!(
        fixture.drain().is_empty(),
        "nothing may be written while the client is snapshot-streaming"
    );
    {
        let state = client.state.lock().expect("daemon client poisoned");
        assert!(
            state
                .catchup_active_session_ids
                .as_ref()
                .is_some_and(|ids| ids.contains(session_b_active)),
            "the replacement must converge into the catch-up queue"
        );
    }
    let mut client_mut = Arc::clone(&client);
    finish_client_snapshot_streaming(&mut client_mut, session_b_active);
    daemon
        .catch_up_backpressured_client(Arc::clone(&client))
        .await
        .expect("catch-up drained");
    let frames = fixture.drain();
    let replaced = b1_frame_index(&frames, "session_replaced")
        .expect("the catch-up delivered the replace frame");
    assert_eq!(
        frames[replaced]
            .get("activeSessionId")
            .and_then(Value::as_str),
        Some(session_b_active)
    );
    let pairs = b1_footer_pairs(&frames);
    assert_eq!(pairs.len(), 2, "decision + compaction segments: {frames:?}");
    assert_eq!(
        b1_footer_positions(&frames),
        vec![replaced + 1, replaced + 2],
        "the footer pair must follow the replace frame: {frames:?}"
    );
    assert!(
        pairs[0].1.contains("Jev Off") && pairs[0].2.contains("Jev Off"),
        "B decision full + compact: {:?} / {:?}",
        pairs[0].1,
        pairs[0].2
    );
    assert!(
        pairs[1].1.contains("Jev compact off") && pairs[1].2.contains("Jev Cmp off"),
        "B compaction full + compact: {:?} / {:?}",
        pairs[1].1,
        pairs[1].2
    );
}

/// B1e (fail-closed): an ABORTED replacement transfer delivers its failure
/// frame but must NOT publish a premature footer pair — the trailing catch-up
/// delivers fresh state and pushes after its own delivery — and a cancelled
/// transfer must not strand a catch-up either.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn b1_aborted_snapshot_transfer_never_publishes_a_premature_footer() {
    let case = "b1-aborted-transfer";
    let mut fixture = ParityFixture::new(case).await;
    let signal = tokio_util::sync::CancellationToken::new();
    signal.cancel();
    let snapshot = serde_json::json!({
        "lastEventSequence": 1,
        "lastEventCursor": { "generation": "b1", "sequence": 1 },
        "messages": [],
    });
    let transcript = create_snapshot_transcript_chunks(CreateSnapshotTranscriptChunksOptions {
        active_session_id: fixture.active_session_id.clone(),
        snapshot_id: "b1-aborted-snapshot".to_string(),
        messages: Vec::new(),
        target_chunk_bytes: Some(SNAPSHOT_TARGET_CHUNK_BYTES),
        aborted: signal.is_cancelled(),
    });
    // R1 (test validity): fixture defaults grant only {attach_snapshot,
    // event_sequence} and `capabilities_for_session` falls back to the global
    // set, which would mask the completed-transfer footer gate — without an
    // explicit extension_ui grant the no-footer assert below is vacuous.
    b1_attach_viewer(
        &fixture.client,
        &fixture.state,
        &fixture.active_session_id,
        &["extension_ui"],
    );
    fixture
        .daemon
        .stream_worker_snapshot(
            &fixture.client,
            &fixture.state,
            "b1-aborted-snapshot",
            snapshot,
            0,
            transcript,
            "replacement",
            signal,
            false,
        )
        .await
        .expect("the aborted transfer completes without error");
    let frames = fixture.drain();
    assert!(
        frames
            .iter()
            .any(|frame| frame.get("type").and_then(Value::as_str)
                == Some("session_snapshot_failed")),
        "the aborted transfer reports its failure: {frames:?}"
    );
    assert!(
        b1_footer_pairs(&frames).is_empty(),
        "no footer pair may be published for an aborted transfer: {frames:?}"
    );
    let state = fixture.client.state.lock().expect("daemon client poisoned");
    assert!(
        state
            .catchup_active_session_ids
            .as_ref()
            .map_or(true, |ids| ids.is_empty()),
        "a cancelled transfer must not strand a catch-up"
    );
}
