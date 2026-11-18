//! OS adapters for the ported CLI. The startup decisions remain in main_entry.

use std::future::Future;
use std::io::{IsTerminal, Write};
use std::pin::Pin;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use crate::cli::owned_session_worker;
use crate::cli_main_entry::{CliMainHost, OwnedSessionWorkerEntry};
use crate::core::agent_session_runtime::{AgentSessionRuntime, InProcessRuntimeHostAdapter};
use crate::main_entry::{
    self, AgentsViewSeamOptions, BoxFuture, DaemonModeSeamOptions, InteractiveModeSeamOptions,
    MainHost, MainOptions, PrintModeSeamOptions, PublicCommandOutcome,
};
use crate::modes::agent_connection::in_process_agent_connection::InProcessAgentConnection;
use crate::modes::agent_connection::types::AgentConnection;
use crate::modes::interactive::theme::theme;
use crate::modes::print_mode::{self, PrintModeHost, PrintModeOptions};

#[derive(Clone)]
pub(crate) struct NativeMainHost {
    argv: Vec<String>,
    exit_code: Arc<AtomicI32>,
}

impl NativeMainHost {
    pub(crate) fn new(args: Vec<String>) -> Self {
        let executable = args
            .first()
            .cloned()
            .unwrap_or_else(|| "optimus-rust".to_string());
        // run_cli consumes Node's argv layout. The native executable replaces
        // both the Node interpreter and script, without consuming a user flag.
        let mut argv = vec![executable.clone(), executable];
        argv.extend(args.into_iter().skip(1));
        Self {
            argv,
            exit_code: Arc::new(AtomicI32::new(0)),
        }
    }

    pub(crate) fn exit_code(&self) -> i32 {
        self.exit_code.load(Ordering::SeqCst)
    }
}

// Console/file adapters may hold thread-local or synchronous UI values across
// awaits. Keep those futures on their owning thread while Tokio continues
// driving sockets and provider events on its worker threads.
fn on_owner_thread<T: Send + 'static>(
    start: impl FnOnce() -> Pin<Box<dyn Future<Output = T>>> + Send + 'static,
) -> BoxFuture<T> {
    let runtime = tokio::runtime::Handle::current();
    Box::pin(async move {
        tokio::task::spawn_blocking(move || runtime.block_on(start()))
            .await
            .expect("CLI owner thread panicked")
    })
}

impl OwnedSessionWorkerEntry for NativeMainHost {
    fn install_owner_watch(&self) {
        if let Err(error) = crate::cli::native_owned_worker::install_owner_watch() {
            eprintln!("{error}");
            self.exit_code.store(1, Ordering::SeqCst);
        }
    }

    fn close_owner_watch(&self) {
        owned_session_worker::close_owned_session_worker_owner_watch();
    }

    fn is_owned_session_worker_process(&self) -> bool {
        owned_session_worker::is_owned_session_worker_process(&std::env::vars().collect())
    }

    fn maybe_run_frontend(&self, args: Vec<String>) -> BoxFuture<bool> {
        let exit_code = self.exit_code.clone();
        Box::pin(async move {
            if exit_code.load(Ordering::SeqCst) != 0 {
                return true;
            }
            match crate::cli::native_owned_worker::maybe_run_frontend(args).await {
                Ok(Some(code)) => {
                    exit_code.store(code, Ordering::SeqCst);
                    true
                }
                Ok(None) => false,
                Err(error) => {
                    eprintln!("{error}");
                    exit_code.store(1, Ordering::SeqCst);
                    true
                }
            }
        })
    }
}

impl CliMainHost for NativeMainHost {
    fn run_main(&self, args: Vec<String>) -> BoxFuture<Result<(), String>> {
        let host = self.clone();
        on_owner_thread(move || {
            Box::pin(async move {
                if host.exit_code() != 0 {
                    return Ok(());
                }
                main_entry::main(args, MainOptions::default(), &host).await;
                Ok(())
            })
        })
    }

    fn argv(&self) -> Vec<String> {
        self.argv.clone()
    }
    fn set_process_title(&self, _title: &str) {
        // A native executable already has an OS-visible process name.
    }
    fn set_env(&self, key: &str, value: &str) {
        std::env::set_var(key, value);
    }
    fn silence_emit_warning(&self) {
        // Rust has no Node emitWarning handler.
    }
    fn install_global_dispatcher(&self) {
        // Providers own reqwest clients and request deadlines.
    }
}

impl MainHost for NativeMainHost {
    fn handle_public_command(
        &self,
        args: Vec<String>,
    ) -> BoxFuture<Result<PublicCommandOutcome, String>> {
        let exit_code = self.exit_code.clone();
        on_owner_thread(move || {
            Box::pin(async move {
                let set_exit_code = |code| {
                    exit_code.store(code, Ordering::SeqCst);
                };
                let result = crate::cli::public_command::handle_public_command(
                    &args,
                    &crate::cli::public_command::PublicCommandIo {
                        log: &|message| println!("{message}"),
                        error: &|message| eprintln!("{message}"),
                        set_exit_code: &set_exit_code,
                        stdin_is_tty: Some(std::io::stdin().is_terminal()),
                        prompt_yes_no: &main_entry::prompt_confirm,
                        cwd: std::env::current_dir()
                            .map_err(|error| error.to_string())?
                            .to_string_lossy()
                            .into_owned(),
                    },
                )
                .await;
                Ok(PublicCommandOutcome {
                    handled: result.handled,
                    args: result.args,
                    explicit_agents_view: result.explicit_agents_view,
                    attach_agent: result.attach_agent,
                })
            })
        })
    }

    fn is_owned_session_worker_process(&self) -> bool {
        OwnedSessionWorkerEntry::is_owned_session_worker_process(self)
    }
    fn install_owned_session_recovery_tracking(&self, runtime: Arc<AgentSessionRuntime>) {
        owned_session_worker::install_owned_session_recovery_tracking(&runtime);
    }
    fn register_builtin_mcp_oauth_providers(&self) {
        pi_ai::mcp::catalog::register_builtin_mcp_oauth_providers();
    }
    fn set_keybindings(&self, agent_dir: &str) {
        crate::core::keybindings::KeybindingsManager::create(Some(agent_dir));
    }
    fn run_daemon_catalog_process(&self) -> BoxFuture<Result<(), String>> {
        Box::pin(crate::modes::daemon::daemon_catalog_entry::run_native_catalog_process())
    }
    fn run_daemon_supervisor_mode(
        &self,
        socket_path: Option<String>,
        config: crate::core::agent_session_config::AgentSessionRuntimeConfig,
    ) -> BoxFuture<Result<(), String>> {
        Box::pin(
            crate::modes::daemon::daemon_supervisor::run_daemon_supervisor_mode(
                socket_path,
                config,
            ),
        )
    }
    fn run_daemon_mode(&self, options: DaemonModeSeamOptions) -> BoxFuture<Result<(), String>> {
        Box::pin(crate::core::agent_session_runtime::run_native_daemon_mode(
            options,
        ))
    }
    fn run_agents_view_mode(
        &self,
        options: AgentsViewSeamOptions,
    ) -> BoxFuture<Result<(), String>> {
        on_owner_thread(move || {
            Box::pin(crate::modes::agents_view::native_host::run_agents_view_mode(options))
        })
    }
    fn run_rpc_mode_with_connection(
        &self,
        connection: Arc<dyn AgentConnection>,
    ) -> BoxFuture<Result<(), String>> {
        Box::pin(crate::modes::rpc::native_host::run_native_rpc_mode(
            connection,
        ))
    }
    fn run_acp_mode_with_connection(
        &self,
        connection: Arc<dyn AgentConnection>,
    ) -> BoxFuture<Result<(), String>> {
        on_owner_thread(move || {
            Box::pin(crate::modes::acp::acp_mode::run_acp_mode_with_connection(
                connection,
                Default::default(),
            ))
        })
    }
    fn run_print_mode_with_connection(
        &self,
        connection: Arc<dyn AgentConnection>,
        options: PrintModeSeamOptions,
    ) -> BoxFuture<Result<i32, String>> {
        Box::pin(print_mode::run_print_mode_with_connection(
            connection,
            Arc::new(NativePrintHost),
            print_options(options),
        ))
    }
    fn run_rpc_mode(&self, runtime: Arc<AgentSessionRuntime>) -> BoxFuture<Result<(), String>> {
        let connection = Arc::new(InProcessAgentConnection::new(Arc::new(
            InProcessRuntimeHostAdapter::new(runtime),
        )));
        Box::pin(crate::modes::rpc::native_host::run_native_rpc_mode(
            connection,
        ))
    }
    fn run_acp_mode(&self, runtime: Arc<AgentSessionRuntime>) -> BoxFuture<Result<(), String>> {
        on_owner_thread(move || {
            Box::pin(crate::modes::acp::acp_mode::run_acp_mode(Arc::new(
                InProcessRuntimeHostAdapter::new(runtime),
            )))
        })
    }
    fn run_print_mode(
        &self,
        runtime: Arc<AgentSessionRuntime>,
        options: PrintModeSeamOptions,
    ) -> BoxFuture<Result<i32, String>> {
        Box::pin(print_mode::run_print_mode(
            Arc::new(InProcessRuntimeHostAdapter::new(runtime)),
            Arc::new(NativePrintHost),
            print_options(options),
        ))
    }
    fn run_interactive_mode(
        &self,
        options: InteractiveModeSeamOptions,
    ) -> BoxFuture<
        Result<
            Option<crate::modes::interactive::interactive_mode::InteractiveModeRunResult>,
            String,
        >,
    > {
        on_owner_thread(move || {
            Box::pin(
                crate::modes::interactive::native_host::run_interactive_mode_for_agents(options),
            )
        })
    }
    fn init_interactive_mode(
        &self,
        options: InteractiveModeSeamOptions,
    ) -> BoxFuture<Result<(), String>> {
        on_owner_thread(move || {
            Box::pin(crate::modes::interactive::native_host::init_interactive_mode(options))
        })
    }
    fn preload_code_highlighter(&self) {
        theme::preload_code_highlighter();
    }
    fn init_theme(&self, name: Option<String>, watch: bool) {
        theme::init_theme(name.as_deref(), watch);
    }
    fn stop_theme_watcher(&self) {
        theme::stop_theme_watcher();
    }
    fn exit(&self, code: i32) {
        self.exit_code.store(code, Ordering::SeqCst);
    }
    fn set_exit_code(&self, code: i32) {
        self.exit_code.store(code, Ordering::SeqCst);
    }
    fn stdin_is_tty(&self) -> bool {
        std::io::stdin().is_terminal()
    }
    fn env(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
    fn set_env(&self, key: &str, value: &str) {
        std::env::set_var(key, value);
    }
    fn cwd(&self) -> String {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
    fn chdir(&self, cwd: &str) -> Result<(), String> {
        std::env::set_current_dir(cwd).map_err(|error| error.to_string())
    }
    fn prompt_for_missing_session_cwd(
        &self,
        issue: &crate::core::session_cwd::SessionCwdIssue,
        _agent_dir: &str,
    ) -> Option<String> {
        eprintln!(
            "{}",
            crate::core::session_cwd::format_missing_session_cwd_prompt(issue)
        );
        eprint!("Working directory (leave empty to cancel): ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok()?;
        let directory = line.trim();
        if directory.is_empty() {
            return None;
        }
        let directory = crate::config::expand_tilde_path(directory, None);
        if !std::path::Path::new(&directory).is_dir() {
            eprintln!("Directory does not exist: {directory}");
            return None;
        }
        Some(directory)
    }
}

fn print_options(options: PrintModeSeamOptions) -> PrintModeOptions {
    PrintModeOptions {
        mode: options.mode,
        messages: options.messages,
        initial_message: options.initial_message,
        initial_images: options.initial_images,
    }
}

#[cfg(any(not(unix), test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonUnixPrintSignal {
    CtrlC,
    CtrlClose,
}

#[cfg(any(not(unix), test))]
fn nonunix_print_signal(signal: &str, windows: bool) -> Option<NonUnixPrintSignal> {
    match signal {
        "SIGINT" => Some(NonUnixPrintSignal::CtrlC),
        "SIGTERM" if windows => Some(NonUnixPrintSignal::CtrlClose),
        _ => None,
    }
}

#[cfg(any(not(unix), test))]
fn listen_for_nonunix_print_signal_with(
    signal: &str,
    windows: bool,
    handler: Arc<dyn Fn() + Send + Sync>,
    subscribe: impl FnOnce(NonUnixPrintSignal) -> BoxFuture<bool> + Send + 'static,
) -> Arc<dyn Fn() + Send + Sync> {
    let Some(event) = nonunix_print_signal(signal, windows) else {
        return Arc::new(|| {});
    };
    let task = tokio::spawn(async move {
        if subscribe(event).await {
            handler();
        }
    });
    Arc::new(move || task.abort())
}

struct NativePrintHost;

impl PrintModeHost for NativePrintHost {
    fn on_signal(
        &self,
        signal: &str,
        handler: Arc<dyn Fn() + Send + Sync>,
    ) -> Arc<dyn Fn() + Send + Sync> {
        #[cfg(unix)]
        {
            let signal = signal.to_string();
            let task = tokio::spawn(async move {
                use tokio::signal::unix::{signal as subscribe, SignalKind};
                let kind = match signal.as_str() {
                    "SIGINT" => SignalKind::interrupt(),
                    "SIGHUP" => SignalKind::hangup(),
                    _ => SignalKind::terminate(),
                };
                if let Ok(mut stream) = subscribe(kind) {
                    if stream.recv().await.is_some() {
                        handler();
                    }
                }
            });
            Arc::new(move || task.abort())
        }
        #[cfg(not(unix))]
        {
            listen_for_nonunix_print_signal_with(signal, cfg!(windows), handler, |event| {
                Box::pin(async move {
                    match event {
                        NonUnixPrintSignal::CtrlC => tokio::signal::ctrl_c().await.is_ok(),
                        NonUnixPrintSignal::CtrlClose => {
                            #[cfg(windows)]
                            {
                                match tokio::signal::windows::ctrl_close() {
                                    Ok(mut stream) => stream.recv().await.is_some(),
                                    Err(_) => false,
                                }
                            }
                            #[cfg(not(windows))]
                            {
                                false
                            }
                        }
                    }
                })
            })
        }
    }
    fn exit(&self, code: i32) {
        std::process::exit(code);
    }
    fn platform(&self) -> String {
        crate::utils::pi_user_agent::process_platform().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;

    #[tokio::test]
    async fn windows_print_signals_use_only_their_matching_console_events() {
        for (signal, expected) in [
            ("SIGINT", NonUnixPrintSignal::CtrlC),
            ("SIGTERM", NonUnixPrintSignal::CtrlClose),
        ] {
            let subscriptions = Arc::new(Mutex::new(Vec::new()));
            let calls = Arc::new(AtomicUsize::new(0));
            let source = subscriptions.clone();
            let handler_calls = calls.clone();
            let cleanup = listen_for_nonunix_print_signal_with(
                signal,
                true,
                Arc::new(move || {
                    handler_calls.fetch_add(1, Ordering::SeqCst);
                }),
                move |event| {
                    source.lock().unwrap().push(event);
                    Box::pin(async { true })
                },
            );
            tokio::task::yield_now().await;
            assert_eq!(*subscriptions.lock().unwrap(), vec![expected]);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            cleanup();
        }
    }

    #[tokio::test]
    async fn unsupported_nonunix_print_signals_install_no_listener() {
        for (signal, windows) in [("SIGHUP", true), ("SIGTERM", false), ("SIGBOGUS", true)] {
            let subscriptions = Arc::new(AtomicUsize::new(0));
            let source = subscriptions.clone();
            let cleanup = listen_for_nonunix_print_signal_with(
                signal,
                windows,
                Arc::new(|| panic!("unsupported signal invoked the handler")),
                move |_| {
                    source.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async { true })
                },
            );
            cleanup();
            assert_eq!(subscriptions.load(Ordering::SeqCst), 0);
        }
    }
}
