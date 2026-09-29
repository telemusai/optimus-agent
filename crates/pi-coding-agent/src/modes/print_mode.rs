//! Port of packages/coding-agent/src/modes/print-mode.ts
//!
//! Print mode (single-shot): Send prompts, output result, exit.
//!
//! Used for:
//! - `pi -p "prompt"` - text output
//! - `pi --mode json "prompt"` - JSON event stream

use std::sync::Arc;

use pi_ai::types::{BoxFuture, ImageContent, STOP_REASON_ABORTED, STOP_REASON_ERROR};
use serde_json::Value;

use crate::core::output_guard::{flush_raw_stdout, write_raw_stdout};
use crate::modes::headless_completion::{
    latest_autonomous_gate_attempt, select_headless_terminal_result, HeadlessTerminalResultMessage,
};
use crate::modes::agent_connection::in_process_agent_connection::{
    InProcessAgentConnection, InProcessHeadlessExtensionOptions, InProcessRuntimeHost,
};
use crate::modes::agent_connection::types::*;
use crate::utils::shell::kill_tracked_detached_children;

/// `PrintModeOptions`.
#[derive(Debug, Clone, Default)]
pub struct PrintModeOptions {
    /// Output mode: "text" for final response only, "json" for all events.
    pub mode: String,
    /// Array of additional prompts to send after initialMessage.
    pub messages: Vec<String>,
    /// First message to send (may contain @file content).
    pub initial_message: Option<String>,
    /// Images to attach to the initial message.
    pub initial_images: Option<Vec<ImageContent>>,
}

/// `MODE_TEXT` / `MODE_JSON`.
pub const PRINT_MODE_TEXT: &str = "text";
pub const PRINT_MODE_JSON: &str = "json";

/// `describeAutonomousLimit(status, reason)`.
fn describe_autonomous_limit(status: &AgentAutonomousStatus, reason: AutonomousLimitReason) -> String {
    if reason == LIMIT_MAX_CONTINUATIONS {
        return format!(
            "maxContinuations reached ({}/{})",
            status.continuations_used, status.limits.max_continuations
        );
    }
    if reason == LIMIT_MAX_TURNS {
        return format!("maxTurns reached ({}/{})", status.turns_used, status.limits.max_turns);
    }
    if reason == LIMIT_MAX_TOKENS {
        return format!("maxTokens reached ({}/{})", status.tokens_used, status.limits.max_tokens);
    }
    let elapsed = match status.started_at {
        None => 0.0,
        Some(started_at) => (now_ms() as f64 - started_at).max(0.0),
    };
    format!("timeoutMs reached ({elapsed}/{})", status.limits.timeout_ms)
}

/// `Date.now()`.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// The Node process surface print mode touches.
///
/// blocked_on: a library crate cannot subscribe to OS signals or call
/// `process.exit`, so the port takes an explicit host seam. Every method mirrors
/// exactly the Node call the TypeScript makes.
pub trait PrintModeHost: Send + Sync {
    /// `process.on(signal, handler)`; returns the cleanup function.
    fn on_signal(&self, signal: &str, handler: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync>;
    /// `process.exit(code)`.
    fn exit(&self, code: i32);
    /// `process.platform`.
    fn platform(&self) -> String;
}

/// `runPrintMode(runtimeHost, options)`.
///
/// blocked_on: `AgentSessionRuntime` (core/agent-session-runtime.ts) belongs to
/// another slice, so the port takes the runtime host seam the in-process
/// connection already consumes.
pub async fn run_print_mode(
    host: Arc<dyn InProcessRuntimeHost>,
    process_host: Arc<dyn PrintModeHost>,
    options: PrintModeOptions,
) -> Result<i32, String> {
    let connection = Arc::new(InProcessAgentConnection::new(host));
    let bind = {
        let connection = connection.clone();
        Arc::new(move || connection.bind_headless_extensions(InProcessHeadlessExtensionOptions::default()))
            as Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>
    };
    run_print_mode_with_connection_internal(
        connection as Arc<dyn AgentConnection>,
        process_host,
        options,
        Some(bind),
    )
    .await
}

/// `runPrintModeWithConnection(connection, options)`.
pub async fn run_print_mode_with_connection(
    connection: Arc<dyn AgentConnection>,
    process_host: Arc<dyn PrintModeHost>,
    options: PrintModeOptions,
) -> Result<i32, String> {
    run_print_mode_with_connection_internal(connection, process_host, options, None).await
}

/// `runPrintModeWithConnectionInternal(connection, options, bindHeadlessExtensions?)`.
async fn run_print_mode_with_connection_internal(
    connection: Arc<dyn AgentConnection>,
    process_host: Arc<dyn PrintModeHost>,
    options: PrintModeOptions,
    bind_headless_extensions: Option<Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>>,
) -> Result<i32, String> {
    let PrintModeOptions {
        mode,
        messages,
        initial_message,
        initial_images,
    } = options;
    let mut exit_code = 0i32;
    let disposed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let unsubscribe: Arc<std::sync::Mutex<Option<Arc<dyn Fn() + Send + Sync>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let mut signal_cleanup_handlers: Vec<Arc<dyn Fn() + Send + Sync>> = Vec::new();

    // `disposeConnection()`.
    let dispose_connection = {
        let disposed = disposed.clone();
        let unsubscribe = unsubscribe.clone();
        let connection = connection.clone();
        Arc::new(move || {
            let disposed = disposed.clone();
            let unsubscribe = unsubscribe.clone();
            let connection = connection.clone();
            Box::pin(async move {
                if disposed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
                if let Some(unsubscribe) = unsubscribe.lock().expect("unsubscribe poisoned").take() {
                    unsubscribe();
                }
                let _ = connection.dispose().await;
            }) as BoxFuture<()>
        }) as Arc<dyn Fn() -> BoxFuture<()> + Send + Sync>
    };

    for signal in print_mode_signal_names(&process_host.platform()) {
        let handler = {
            let dispose_connection = dispose_connection.clone();
            let process_host = process_host.clone();
            let signal = signal.to_string();
            Arc::new(move || {
                let dispose_connection = dispose_connection.clone();
                let process_host = process_host.clone();
                let signal = signal.clone();
                kill_tracked_detached_children();
                tokio::spawn(async move {
                    dispose_connection().await;
                    process_host.exit(print_mode_signal_exit_code(&signal));
                });
            }) as Arc<dyn Fn() + Send + Sync>
        };
        let cleanup = process_host.on_signal(signal, handler);
        signal_cleanup_handlers.push(cleanup);
    }

    let body = async {
        if mode == PRINT_MODE_JSON {
            let header = connection.get_session_header().await.unwrap_or(None);
            if let Some(header) = header {
                write_raw_stdout(&format!(
                    "{}\n",
                    serde_json::to_string(&header).unwrap_or_else(|_| "null".to_string())
                ));
            }
        }

        let listener = {
            let mode = mode.clone();
            Arc::new(move |event: AgentConnectionEvent| {
                let mode = mode.clone();
                Box::pin(async move {
                    match &event {
                        AgentConnectionEvent::SessionEvent { event } => {
                            if mode == PRINT_MODE_JSON {
                                write_raw_stdout(&format!(
                                    "{}\n",
                                    serde_json::to_string(event).unwrap_or_else(|_| "null".to_string())
                                ));
                            }
                        }
                        AgentConnectionEvent::ExtensionError {
                            extension_path,
                            error,
                            ..
                        } => {
                            eprintln!("Extension error ({extension_path}): {error}");
                        }
                        _ => {}
                    }
                }) as BoxFuture<()>
            }) as AgentConnectionEventListener
        };
        let detach = connection.subscribe(listener);
        *unsubscribe.lock().expect("unsubscribe poisoned") = Some(Arc::from(detach));
        if let Some(bind) = &bind_headless_extensions {
            bind().await?;
        }

        if let Some(initial_message) = initial_message {
            let prompt_options = AgentConnectionPromptOptions {
                images: initial_images,
                ..Default::default()
            };
            connection
                .prompt_and_wait(&initial_message, Some(prompt_options))
                .await?;
        }
        for message in &messages {
            connection.prompt_and_wait(message, None).await?;
        }

        let autonomous_status = connection.wait_for_headless_completion(None).await?;
        {
            // Output format must not change whether a completed request failed.
            let terminal = select_headless_terminal_result(&connection.get_messages().await?);
            match &terminal.primary {
                Some(HeadlessTerminalResultMessage::Assistant(primary)) => {
                    if primary.stop_reason == STOP_REASON_ERROR || primary.stop_reason == STOP_REASON_ABORTED {
                        let error_message = primary.error_message.clone().unwrap_or_default();
                        if error_message.is_empty() {
                            eprintln!("Request {}", primary.stop_reason);
                        } else {
                            eprintln!("{error_message}");
                        }
                        exit_code = 1;
                    } else if mode == PRINT_MODE_TEXT {
                        for content in &primary.content {
                            if let pi_ai::types::ContentBlock::Text(text) = content {
                                write_raw_stdout(&format!("{}\n", text.text));
                            }
                        }
                    }
                }
                Some(HeadlessTerminalResultMessage::SessionSlashCommandResult(primary)) => {
                    if mode == PRINT_MODE_TEXT {
                        let content = terminal_value_text(primary);
                        write_raw_stdout(&format!("{content}\n"));
                    }
                    let details = primary.get("details").cloned().unwrap_or(Value::Null);
                    let success = details.get("success").and_then(Value::as_bool).unwrap_or(false);
                    let severity = details.get("severity").and_then(Value::as_str).unwrap_or_default();
                    if !success || severity == "error" {
                        exit_code = 1;
                    }
                }
                None => {}
            }
            for outcome in &terminal.compaction_outcomes {
                if mode == PRINT_MODE_TEXT {
                    eprintln!("{}", terminal_value_text(outcome));
                }
                let is_failed = outcome
                    .get("details")
                    .and_then(|details| details.get("outcome"))
                    .and_then(Value::as_str)
                    == Some("failed");
                if is_failed {
                    exit_code = 1;
                }
            }
        }

        let autonomous_limit = autonomous_limit_reason(&autonomous_status);
        if autonomous_status.enabled
            && !autonomous_status.gates.commands.is_empty()
            && autonomous_status.last_gate_failure.is_some()
        {
            let limit_text = match autonomous_limit {
                Some(limit) => format!(
                    "; autonomous limit reached: {}",
                    describe_autonomous_limit(&autonomous_status, limit)
                ),
                None => String::new(),
            };
            let exit_text = autonomous_status
                .last_gate_failure
                .as_ref()
                .map(|failure| failure.exit_text.clone())
                .unwrap_or_default();
            eprintln!(
                "Autonomous quality gate still failing after attempt {}/{}: {exit_text}{limit_text}",
                latest_autonomous_gate_attempt(&autonomous_status),
                autonomous_status.gates.max_retries
            );
            exit_code = 1;
        } else if autonomous_status.enabled && autonomous_status.gates.commands.is_empty() && autonomous_limit.is_some()
        {
            let limit = autonomous_limit.unwrap_or_default();
            eprintln!(
                "Autonomous run stopped before terminal evidence; {}",
                describe_autonomous_limit(&autonomous_status, limit)
            );
            exit_code = 1;
        }

        Ok::<i32, String>(exit_code)
    }
    .await;

    for cleanup in signal_cleanup_handlers.drain(..) {
        cleanup();
    }
    dispose_connection().await;
    flush_raw_stdout().await;

    match body {
        Ok(exit_code) => Ok(exit_code),
        Err(error) => {
            eprintln!("{error}");
            Ok(1)
        }
    }
}

/// `message.content` of a session-slash-command result / compaction outcome,
/// where `content` is always a plain string in the TypeScript.
fn terminal_value_text(message: &Value) -> String {
    message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `["SIGINT","SIGTERM", ...(process.platform === "win32" ? [] : ["SIGHUP"])]`.
fn print_mode_signal_names(platform: &str) -> Vec<&'static str> {
    let mut signals = vec!["SIGINT", "SIGTERM"];
    if platform != "win32" {
        signals.push("SIGHUP");
    }
    signals
}

/// `signal === "SIGINT" ? 130 : signal === "SIGHUP" ? 129 : 143`.
fn print_mode_signal_exit_code(signal: &str) -> i32 {
    if signal == "SIGINT" {
        130
    } else if signal == "SIGHUP" {
        129
    } else {
        143
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autonomous_limit_text_matches_the_typescript() {
        let status = AgentAutonomousStatus {
            enabled: true,
            continuations_used: 2.0,
            turns_used: 5.0,
            tokens_used: 80_000.0,
            started_at: Some(0.0),
            limits: AgentAutonomousLimits {
                max_continuations: 3.0,
                max_turns: 12.0,
                max_tokens: 80_000.0,
                timeout_ms: 1_800_000.0,
            },
            gates: AgentAutonomousGateStatus {
                commands: vec!["npm test".to_string()],
                max_retries: 3.0,
                timeout_ms: 300_000.0,
            },
            gate_attempts: Default::default(),
            last_gate_failure: None,
        };
        assert_eq!(
            describe_autonomous_limit(&status, LIMIT_MAX_CONTINUATIONS),
            "maxContinuations reached (2/3)"
        );
        assert_eq!(describe_autonomous_limit(&status, LIMIT_MAX_TURNS), "maxTurns reached (5/12)");
        assert_eq!(
            describe_autonomous_limit(&status, LIMIT_MAX_TOKENS),
            "maxTokens reached (80000/80000)"
        );
        assert!(describe_autonomous_limit(&status, LIMIT_TIMEOUT_MS).ends_with("/1800000)"));
    }

    #[test]
    fn signal_names_and_exit_codes_match_node() {
        assert_eq!(print_mode_signal_names("win32"), vec!["SIGINT", "SIGTERM"]);
        assert_eq!(print_mode_signal_names("linux"), vec!["SIGINT", "SIGTERM", "SIGHUP"]);
        assert_eq!(print_mode_signal_exit_code("SIGINT"), 130);
        assert_eq!(print_mode_signal_exit_code("SIGHUP"), 129);
        assert_eq!(print_mode_signal_exit_code("SIGTERM"), 143);
    }

    #[test]
    fn terminal_value_text_reads_the_content_string() {
        assert_eq!(terminal_value_text(&serde_json::json!({ "content": "hi" })), "hi");
        assert_eq!(terminal_value_text(&serde_json::json!({})), "");
    }
}
