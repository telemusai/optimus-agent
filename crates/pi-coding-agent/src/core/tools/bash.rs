//! Port of packages/coding-agent/src/core/tools/bash.ts

use std::sync::Arc;

use pi_agent_core::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pi_agent_core::types::ContentBlock as AgentContentBlock;
// `core/tools/bash.ts:9-15` imports `getShellConfig`, `getShellEnv` and
// `killProcessTree` from `utils/shell.js`; use the same owners here.
use crate::utils::shell::{
    get_shell_config, get_shell_env, kill_process_tree, track_detached_child_pid,
    untrack_detached_child_pid,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::code_preview::preview_bash_command;
use super::output_accumulator::{OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot};
use super::render_utils::{get_text_output, invalid_arg_text, str_value, RenderContentBlock, RenderResultLike, TextOutputOptions, ToolTheme};
use super::tool_definition_wrapper::wrap_tool_definition;
use super::truncate::{format_size, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use super::{ExtensionContext, ToolDefinition, ToolExecuteFn};

pub const BASH_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "command": { "type": "string", "description": "Bash command to execute" },
    "timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" }
  },
  "required": ["command"]
}"#;

/// TypeScript `type BashToolInput = Static<typeof bashSchema>`.
#[derive(Debug, Clone, Default)]
pub struct BashToolInput {
    pub command: String,
    pub timeout: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BashToolDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<super::truncate::TruncationResult>,
    #[serde(rename = "fullOutputPath", skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// Execution options passed to [`BashOperations::exec`].
#[derive(Clone)]
pub struct BashExecOptions {
    /// Receives each output chunk in arrival order.
    pub on_data: Arc<dyn Fn(&[u8]) + Send + Sync>,
    pub signal: Option<CancellationToken>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
    pub env: Option<Vec<(String, String)>>,
}

/// Pluggable operations for the bash tool.
/// Override these to delegate command execution to remote systems (for example SSH).
pub trait BashOperations: Send + Sync {
    /// Execute a command and stream output.
    /// Resolves to the exit code (`None` if killed).
    fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> futures::future::BoxFuture<'static, Result<BashExecResult, String>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BashExecResult {
    pub exit_code: Option<i32>,
}

/// Create bash operations using pi's built-in local shell execution backend.
///
/// This is useful for extensions that intercept user_bash and still want pi's
/// standard local shell behavior while wrapping or rewriting commands.
pub fn create_local_bash_operations(options: Option<LocalBashOperationsOptions>) -> Arc<dyn BashOperations> {
    Arc::new(LocalBashOperations {
        shell_path: options.and_then(|options| options.shell_path),
    })
}

#[derive(Debug, Clone, Default)]
pub struct LocalBashOperationsOptions {
    pub shell_path: Option<String>,
}

/// Local shell backend.
///
/// The TypeScript version spawns `getShellConfig(shellPath)` through
/// `spawnHidden` and tracks detached child pids. The port runs the same shell
/// command and streams stdout/stderr in arrival order; process-tree tracking is
/// owned by `utils/child-process.rs` in another slice.
pub struct LocalBashOperations {
    pub shell_path: Option<String>,
}

// Tool futures can be dropped by the agent's cancellation race before their
// own select observes the token. Keep process/reader cleanup owned by RAII too.
struct BashChild(tokio::process::Child);
impl std::ops::Deref for BashChild {
    type Target = tokio::process::Child;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl std::ops::DerefMut for BashChild {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}
impl Drop for BashChild {
    fn drop(&mut self) {
        if let Some(pid) = self.0.id() {
            kill_process_tree(pid as i32);
            let _ = self.0.start_kill();
            untrack_detached_child_pid(pid as i32);
        }
    }
}
struct BashReaders(Vec<tokio::task::AbortHandle>);
impl Drop for BashReaders {
    fn drop(&mut self) { for reader in &self.0 { reader.abort(); } }
}

impl BashOperations for LocalBashOperations {
    fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> futures::future::BoxFuture<'static, Result<BashExecResult, String>> {
        let command = command.to_string();
        let cwd = cwd.to_string();
        let shell_path = self.shell_path.clone();
        Box::pin(async move {
            // TS order (`core/tools/bash.ts:69-73`): resolve the shell config
            // first - `getShellConfig` throws `Custom shell path not found: ...`
            // or the `No bash shell found. Options: ...` teaching error - and
            // only then check the working directory. `?` rejects with the same
            // string, mirroring the throw inside the Promise executor.
            let config = get_shell_config(shell_path.as_deref())?;
            let shell = config.shell;
            let args = config.args;

            if !std::path::Path::new(&cwd).exists() {
                return Err(format!(
                    "Working directory does not exist: {cwd}\nCannot execute bash commands."
                ));
            }

            let mut command_builder = tokio::process::Command::new(&shell);
            for arg in &args {
                command_builder.arg(arg);
            }
            command_builder.arg(&command);
            command_builder.current_dir(&cwd);
            command_builder.stdin(std::process::Stdio::null());
            command_builder.stdout(std::process::Stdio::piped());
            command_builder.stderr(std::process::Stdio::piped());
            command_builder.env_clear();
            for (key, value) in options.env.clone().unwrap_or_else(get_shell_env) {
                command_builder.env(key, value);
            }
            #[cfg(unix)]
            command_builder.process_group(0);

            let mut child = match command_builder.spawn() {
                Ok(child) => BashChild(child),
                Err(error) => return Err(error.to_string()),
            };

            // TS bash.ts:80: `if (child.pid) trackDetachedChildPid(child.pid);`
            let tracked_child_pid = child.id().map(|pid| pid as i32);
            if let Some(pid) = tracked_child_pid {
                track_detached_child_pid(pid);
            }
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let on_data = options.on_data.clone();
            let on_data_err = options.on_data.clone();

            let stdout_task = stdout.map(|stdout| {
                let on_data = on_data.clone();
                tokio::spawn(async move { pump(stdout, on_data).await })
            });
            let stderr_task = stderr.map(|stderr| {
                let on_data = on_data_err.clone();
                tokio::spawn(async move { pump(stderr, on_data).await })
            });

            let _readers = BashReaders([&stdout_task, &stderr_task].into_iter()
                .filter_map(|task| task.as_ref().map(tokio::task::JoinHandle::abort_handle)).collect());
            let mut timed_out = false;
            let mut aborted = false;
            // `wait` holds a mutable borrow of `child` for as long as it lives, so
            // the pid is captured before the wait starts.
            let child_pid = child.id();
            let result = {
                let wait = child.wait();
                tokio::pin!(wait);

                let signal = options.signal.clone();
                let timeout = options.timeout;
                loop {
                    let deadline = async {
                        match timeout {
                            Some(seconds) if seconds > 0.0 => {
                                tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
                                "timeout"
                            }
                            _ => {
                                futures::future::pending::<&'static str>().await
                            }
                        }
                    };
                    let abort = async {
                        match signal.as_ref() {
                            Some(token) => {
                                token.cancelled().await;
                                "abort"
                            }
                            None => futures::future::pending::<&'static str>().await,
                        }
                    };
                    tokio::select! {
                        status = &mut wait => break Some(status),
                        reason = deadline => {
                            if reason == "timeout" { timed_out = true; }
                            break None;
                        }
                        reason = abort => {
                            if reason == "abort" { aborted = true; }
                            break None;
                        }
                    }
                }
            };

            if result.is_none() {
                if let Some(pid) = child_pid {
                    // `utils/shell.ts killProcessTree(pid: number)`.
                    kill_process_tree(pid as i32);
                }
            }

            // Shell exit does not imply EOF: a background descendant may still
            // hold either pipe. Never wait without a bound after exit or kill.
            if result.is_none() {
                let _ = child.start_kill();
                let _ = tokio::time::timeout(std::time::Duration::from_secs(3), child.wait()).await;
            }
            let drain_deadline = tokio::time::Instant::now()
                + std::time::Duration::from_millis(crate::utils::child_process::EXIT_STDIO_GRACE_MS);
            let mut detached_output = false;
            for mut task in [stdout_task, stderr_task].into_iter().flatten() {
                if tokio::time::timeout_at(drain_deadline, &mut task).await.is_err() {
                    task.abort();
                    let _ = task.await;
                    detached_output = true;
                }
            }
            if detached_output && result.is_some() {
                (options.on_data)(b"\n[Shell exited; closed output pipes held by a background process. Redirect background output to a file to retain it.]\n");
            }

            // TS bash.ts:102/:116: the child settled (exit or kill), so the
            // journal record goes inactive in both the success and error paths.
            if let Some(pid) = tracked_child_pid {
                untrack_detached_child_pid(pid);
            }

            if aborted {
                return Err("aborted".to_string());
            }
            if timed_out {
                return Err(format!("timeout:{}", options.timeout.unwrap_or(0.0)));
            }

            match result {
                Some(Ok(status)) => Ok(BashExecResult {
                    exit_code: status.code(),
                }),
                Some(Err(error)) => Err(error.to_string()),
                None => Ok(BashExecResult { exit_code: None }),
            }
        })
    }
}

async fn pump<R>(mut reader: R, on_data: Arc<dyn Fn(&[u8]) + Send + Sync>)
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut buffer = vec![0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => on_data(&buffer[..read]),
            Err(_) => break,
        }
    }
}

// Shell resolution is NOT re-implemented here: `core/tools/bash.ts:9-15`
// imports `getShellConfig`/`getShellEnv`/`killProcessTree` from `utils/shell.js`,
// and this module does the same through `crate::utils::shell`. The previous
// local copies diverged from `utils/shell.ts`:
//   - unix preferred `$SHELL` (zsh/fish) where TS always uses `/bin/bash`
//     (`utils/shell.ts:131-135`), then bash on PATH, then `sh`;
//   - win32 used a bare `"bash"` from PATH, which CreateProcess can resolve to
//     the `System32\bash.exe` WSL launcher that `orderWindowsBashCandidates`
//     (`utils/shell.ts:14-19`) deliberately de-prioritizes;
//   - neither the `Custom shell path not found: ...` nor the multi-line
//     `No bash shell found. Options: ...` teaching error existed
//     (`utils/shell.ts:66-69`, `utils/shell.ts:118-124`);
//   - `get_shell_env` returned the ambient env without the `getBinDir()` PATH
//     prepend (`utils/shell.ts:147-161`).


/// TypeScript `interface BashSpawnContext`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BashSpawnContext {
    pub command: String,
    pub cwd: String,
    pub env: Vec<(String, String)>,
}

/// TypeScript `type BashSpawnHook`.
pub type BashSpawnHook = Arc<dyn Fn(BashSpawnContext) -> BashSpawnContext + Send + Sync>;

fn resolve_spawn_context(command: &str, cwd: &str, spawn_hook: Option<&BashSpawnHook>) -> BashSpawnContext {
    let base_context = BashSpawnContext {
        command: command.to_string(),
        cwd: cwd.to_string(),
        // `core/tools/bash.ts:135`: `env: { ...getShellEnv() }` - the bin-dir
        // PATH prepend (`utils/shell.ts:147-161`) must reach tool commands.
        env: get_shell_env(),
    };
    match spawn_hook {
        Some(hook) => hook(base_context),
        None => base_context,
    }
}

/// TypeScript `interface BashToolOptions`.
#[derive(Clone, Default)]
pub struct BashToolOptions {
    /// Custom operations for command execution. Default: local shell
    pub operations: Option<Arc<dyn BashOperations>>,
    /// Command prefix prepended to every command (for example shell setup commands)
    pub command_prefix: Option<String>,
    /// Optional explicit shell path from settings
    pub shell_path: Option<String>,
    /// Hook to adjust command, cwd, or env before execution
    pub spawn_hook: Option<BashSpawnHook>,
}

pub const BASH_PREVIEW_LINES: usize = 5;
pub const BASH_UPDATE_THROTTLE_MS: u64 = 100;

/// TypeScript `type BashRenderState`.
#[derive(Default)]
pub struct BashRenderState {
    pub started_at: Option<f64>,
    pub ended_at: Option<f64>,
}

/// TypeScript `type BashResultRenderState`.
#[derive(Default)]
pub struct BashResultRenderState {
    pub cached_width: Option<usize>,
    pub cached_lines: Option<Vec<String>>,
    pub cached_skipped: Option<usize>,
}

pub fn format_duration(ms: f64) -> String {
    format!("{:.1}s", ms / 1000.0)
}

pub fn format_bash_call(args: Option<&BashToolInput>, theme: &dyn ToolTheme) -> String {
    let command_value = args.map(|args| Value::String(args.command.clone()));
    let command = str_value(command_value.as_ref());
    let timeout = args.and_then(|args| args.timeout);
    // `core/tools/bash.ts:180` gates on JS truthiness: `timeout ? ... : ""`, so
    // `timeout: 0` renders just the command. `NaN` is falsy in JS too.
    let timeout_suffix = match timeout {
        Some(timeout) if timeout != 0.0 && !timeout.is_nan() => {
            theme.fg("muted", &format!(" (timeout {timeout}s)"))
        }
        _ => String::new(),
    };
    let command_display = match command {
        None => invalid_arg_text(theme),
        Some(command) if !command.is_empty() => {
            let preview = preview_bash_command(&command);
            let label = if preview.language == super::code_preview::CodePreviewLanguage::Bash {
                String::new()
            } else {
                format!("{}: ", preview.language.as_str())
            };
            if !preview.text.is_empty() {
                format!("{label}{}", preview.text)
            } else {
                command
            }
        }
        Some(_) => theme.fg("toolOutput", "..."),
    };
    format!("{}{}", theme.fg("toolTitle", &theme.bold(&format!("$ {command_display}"))), timeout_suffix)
}

/// TypeScript `type BashResultRenderComponent`.
///
/// The TypeScript component is a `Container` of child rows; the port keeps the
/// same children as a `Vec<String>` plus the cached preview state the renderer
/// reuses between frames.
#[derive(Default)]
pub struct BashResultRenderComponent {
    pub state: BashResultRenderState,
    pub children: Vec<String>,
}

impl BashResultRenderComponent {
    pub fn clear(&mut self) {
        self.children.clear();
    }

    pub fn add_child(&mut self, text: String) {
        self.children.push(text);
    }
}

/// Port of `rebuildBashResultRenderComponent`.
pub fn rebuild_bash_result_render_component(
    component: &mut BashResultRenderComponent,
    result: Option<&RenderResultLike>,
    details: Option<&BashToolDetails>,
    options: super::ToolRenderResultOptions,
    show_images: bool,
    include_image_dimensions: bool,
    show_expand_hint: bool,
    started_at: Option<f64>,
    ended_at: Option<f64>,
    theme: &dyn ToolTheme,
) {
    component.clear();

    let output = get_text_output(
        result,
        show_images,
        TextOutputOptions {
            include_image_dimensions: Some(include_image_dimensions),
        },
    )
    .trim()
    .to_string();

    if !output.is_empty() {
        let styled_output = output
            .split('\n')
            .map(|line| theme.fg("toolOutput", line))
            .collect::<Vec<String>>()
            .join("\n");

        if options.expanded {
            component.add_child(format!("\n{styled_output}"));
        } else {
            let state = &mut component.state;
            let lines: Vec<String> = styled_output.split('\n').map(str::to_string).collect();
            let width = BASH_PREVIEW_LINES;
            let skipped = lines.len().saturating_sub(width);
            let visual_lines = lines[skipped.min(lines.len())..].to_vec();
            state.cached_width = Some(width);
            state.cached_lines = Some(visual_lines.clone());
            state.cached_skipped = Some(skipped);
            if skipped > 0 {
                // `expandCollapseHint("app.tools.expand", false)` renders the
                // shortcut suffix; without the keybinding manager it stays empty.
                let hint = if show_expand_hint {
                    theme.fg("muted", &format!("... {skipped} earlier lines"))
                } else {
                    theme.fg("muted", &format!("... ({skipped} earlier lines)"))
                };
                component.add_child(String::new());
                component.add_child(hint);
            } else {
                component.add_child(String::new());
            }
            for line in visual_lines {
                component.add_child(line);
            }
        }
    }

    let truncation = details.and_then(|details| details.truncation.as_ref());
    let full_output_path = details.and_then(|details| details.full_output_path.clone());
    let is_truncated = truncation.map(|truncation| truncation.truncated).unwrap_or(false);
    if is_truncated || full_output_path.is_some() {
        let mut warnings: Vec<String> = Vec::new();
        if let Some(full_output_path) = full_output_path {
            warnings.push(format!("Full output: {full_output_path}"));
        }
        if is_truncated {
            let truncation = truncation.expect("truncation present");
            if truncation.truncated_by == Some(super::truncate::TruncatedBy::Lines) {
                warnings.push(format!(
                    "Truncated: showing {} of {} lines",
                    truncation.output_lines, truncation.total_lines
                ));
            } else {
                warnings.push(format!(
                    "Truncated: {} lines shown ({} limit)",
                    truncation.output_lines,
                    format_size(truncation.max_bytes)
                ));
            }
        }
        component.add_child(format!(
            "\n{}",
            theme.fg("warning", &format!("[{}]", warnings.join(". ")))
        ));
    }

    if let Some(started_at) = started_at {
        let label = if options.is_partial { "Elapsed" } else { "Took" };
        let end_time = ended_at.unwrap_or_else(now_ms);
        component.add_child(format!(
            "\n{}",
            theme.fg("muted", &format!("{label} {}", format_duration(end_time - started_at)))
        ));
    }
}

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

pub const BASH_TOOL_DESCRIPTION_TEMPLATE: &str =
    "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {lines} lines or {kb}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.";

pub fn bash_tool_description() -> String {
    BASH_TOOL_DESCRIPTION_TEMPLATE
        .replace("{lines}", &DEFAULT_MAX_LINES.to_string())
        .replace("{kb}", &(DEFAULT_MAX_BYTES / 1024).to_string())
}

/// TS `lastUpdateAt` + `updateDirty` + `updateTimer` for `scheduleOutputUpdate`
/// (`core/tools/bash.ts:295-333`).
#[derive(Default)]
struct BashUpdateThrottle {
    /// TS `updateDirty`.
    dirty: bool,
    /// TS `lastUpdateAt`, in epoch milliseconds.
    last_update_at: f64,
    /// TS `updateTimer !== undefined`.
    timer_pending: bool,
    /// Stale-timer guard: a bumped id invalidates an armed trailing task, which
    /// stands in for the `clearTimeout` call TS uses to cancel it.
    timer_id: u64,
}

/// TS `emitOutputUpdate` (`core/tools/bash.ts:299-311`): dirty-gated, so a
/// snapshot that has not changed since the last emission is not re-sent.
fn emit_output_update(
    throttle: &Arc<std::sync::Mutex<BashUpdateThrottle>>,
    output: &Arc<std::sync::Mutex<OutputAccumulator>>,
    on_update: &AgentToolUpdateCallback,
) {
    {
        let mut state = throttle.lock().expect("update lock");
        if !state.dirty {
            return;
        }
        state.dirty = false;
        state.last_update_at = now_ms();
    }
    let snapshot = output.lock().expect("output lock").snapshot();
    on_update(update_result(&snapshot));
}

/// TS `clearUpdateTimer` (`core/tools/bash.ts:313-318`).
fn clear_update_timer(throttle: &Arc<std::sync::Mutex<BashUpdateThrottle>>) {
    let mut state = throttle.lock().expect("update lock");
    if state.timer_pending {
        state.timer_pending = false;
        state.timer_id += 1;
    }
}

/// TS `updateTimer` callback: emit only if this task is still the armed timer.
fn fire_update_timer(
    throttle: &Arc<std::sync::Mutex<BashUpdateThrottle>>,
    output: &Arc<std::sync::Mutex<OutputAccumulator>>,
    on_update: &AgentToolUpdateCallback,
    timer_id: u64,
) {
    let still_armed = {
        let mut state = throttle.lock().expect("update lock");
        if !state.timer_pending || state.timer_id != timer_id {
            false
        } else {
            state.timer_pending = false;
            true
        }
    };
    if still_armed {
        emit_output_update(throttle, output, on_update);
    }
}

/// TS `scheduleOutputUpdate` (`core/tools/bash.ts:320-333`).
///
/// A throttled chunk is no longer dropped: the pending trailing task emits it
/// when the window elapses (`setTimeout`), so the last chunk of a burst always
/// reaches the update stream.
fn schedule_output_update(
    throttle: &Arc<std::sync::Mutex<BashUpdateThrottle>>,
    output: &Arc<std::sync::Mutex<OutputAccumulator>>,
    on_update: &AgentToolUpdateCallback,
) {
    let delay = {
        let mut state = throttle.lock().expect("update lock");
        state.dirty = true;
        BASH_UPDATE_THROTTLE_MS as f64 - (now_ms() - state.last_update_at)
    };
    if delay <= 0.0 {
        clear_update_timer(throttle);
        emit_output_update(throttle, output, on_update);
        return;
    }
    let timer_id = {
        let mut state = throttle.lock().expect("update lock");
        if state.timer_pending {
            // `updateTimer ??= setTimeout(...)` - one armed timer at a time.
            return;
        }
        state.timer_pending = true;
        state.timer_id += 1;
        state.timer_id
    };
    let delay = std::time::Duration::from_millis(delay.ceil() as u64);
    let (state_for_task, output_for_task, update_for_task) =
        (throttle.clone(), output.clone(), on_update.clone());
    let (state_for_thread, output_for_thread, update_for_thread) =
        (throttle.clone(), output.clone(), on_update.clone());
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(async move {
                tokio::time::sleep(delay).await;
                fire_update_timer(&state_for_task, &output_for_task, &update_for_task, timer_id);
            });
        }
        // No reactor available (a foreign-thread `onData`): sleep on a helper
        // thread so the trailing emission still happens.
        Err(_) => {
            std::thread::spawn(move || {
                std::thread::sleep(delay);
                fire_update_timer(&state_for_thread, &output_for_thread, &update_for_thread, timer_id);
            });
        }
    }
}

/// Port of `createBashToolDefinition`'s execute body.
///
/// The TypeScript closure mutates one `OutputAccumulator` from the data handler
/// and from the finishing path; the shared `Arc<Mutex<_>>` keeps that single
/// instance without changing the emission order.
pub async fn execute_bash(
    cwd: &str,
    operations: Arc<dyn BashOperations>,
    command_prefix: Option<&str>,
    spawn_hook: Option<&BashSpawnHook>,
    input: &BashToolInput,
    signal: Option<CancellationToken>,
    on_update: Option<AgentToolUpdateCallback>,
) -> Result<(String, Option<BashToolDetails>), String> {
    let resolved_command = match command_prefix {
        Some(prefix) => format!("{prefix}\n{}", input.command),
        None => input.command.clone(),
    };
    let spawn_context = resolve_spawn_context(&resolved_command, cwd, spawn_hook);
    let output = Arc::new(std::sync::Mutex::new(OutputAccumulator::with_temp_file_prefix(
        OutputAccumulatorOptions::default(),
        "pi-bash",
    )));
    let update_throttle = Arc::new(std::sync::Mutex::new(BashUpdateThrottle::default()));

    if let Some(on_update) = on_update.as_ref() {
        // `core/tools/bash.ts:335-337`: `onUpdate({ content: [], details: undefined })`.
        on_update(AgentToolResult::new(Vec::new(), Value::Null));
    }

    // TS `handleData` (`core/tools/bash.ts:339-342`): append, then schedule.
    let on_data: Arc<dyn Fn(&[u8]) + Send + Sync> = {
        let output = output.clone();
        let on_update = on_update.clone();
        let update_throttle = update_throttle.clone();
        Arc::new(move |data: &[u8]| {
            if output.lock().expect("output lock").append(data).is_err() {
                return;
            }
            let Some(on_update) = on_update.as_ref() else {
                return;
            };
            schedule_output_update(&update_throttle, &output, on_update);
        })
    };

    let exec_options = BashExecOptions {
        on_data,
        signal: signal.clone(),
        timeout: input.timeout,
        env: Some(spawn_context.env.clone()),
    };

    let result = operations
        .exec(&spawn_context.command, &spawn_context.cwd, exec_options)
        .await;

    let snapshot = finish_output(&output, on_update.as_ref(), &update_throttle);
    // `finally { clearUpdateTimer(); }` (`core/tools/bash.ts:408-410`) runs after
    // the final snapshot, cancelling any trailing timer a late chunk armed.
    clear_update_timer(&update_throttle);
    let (text, details) = format_output(&snapshot, &output, if result.is_err() { "" } else { "(no output)" });

    match result {
        Ok(exec_result) => {
            if let Some(exit_code) = exec_result.exit_code {
                if exit_code != 0 {
                    return Err(append_status(&text, &format!("Command exited with code {exit_code}")));
                }
            }
            Ok((text, details))
        }
        Err(error) => {
            if error == "aborted" {
                return Err(append_status(&text, "Command aborted"));
            }
            if let Some(timeout_secs) = error.strip_prefix("timeout:") {
                return Err(append_status(
                    &text,
                    &format!("Command timed out after {timeout_secs} seconds"),
                ));
            }
            Err(error)
        }
    }
}

fn update_result(snapshot: &OutputSnapshot) -> AgentToolResult {
    AgentToolResult::new(
        vec![AgentContentBlock::text(snapshot.content.clone())],
        serde_json::json!({
            "truncation": if snapshot.truncation.truncated {
                serde_json::to_value(&snapshot.truncation).unwrap_or(Value::Null)
            } else {
                Value::Null
            },
            "fullOutputPath": snapshot.full_output_path,
        }),
    )
}

/// TS `finishOutput` (`core/tools/bash.ts:344-351`): finish, cancel the pending
/// trailing timer, emit the dirty-gated update, then snapshot only after the
/// spill settled.
fn finish_output(
    output: &Arc<std::sync::Mutex<OutputAccumulator>>,
    on_update: Option<&AgentToolUpdateCallback>,
    update_throttle: &Arc<std::sync::Mutex<BashUpdateThrottle>>,
) -> OutputSnapshot {
    output.lock().expect("output lock").finish();
    clear_update_timer(update_throttle);
    if let Some(on_update) = on_update {
        // `std::sync::Mutex` is not reentrant: the accumulator lock is released
        // before `emit_output_update` takes it to read the snapshot.
        emit_output_update(update_throttle, output, on_update);
    }
    // Snapshot only after the spill settled: the advertised path is terminal.
    let mut accumulator = output.lock().expect("output lock");
    accumulator.close_temp_file();
    accumulator.snapshot()
}

fn format_output(snapshot: &OutputSnapshot, output: &Arc<std::sync::Mutex<OutputAccumulator>>, empty_text: &str) -> (String, Option<BashToolDetails>) {
    let truncation = snapshot.truncation.clone();
    let mut text = if snapshot.content.is_empty() {
        empty_text.to_string()
    } else {
        snapshot.content.clone()
    };
    let mut details: Option<BashToolDetails> = None;
    if truncation.truncated {
        details = Some(BashToolDetails {
            truncation: Some(truncation.clone()),
            full_output_path: snapshot.full_output_path.clone(),
        });
        let start_line = truncation.total_lines - truncation.output_lines + 1;
        let end_line = truncation.total_lines;
        // A degraded spill has no path; never advertise "Full output: undefined".
        let location = match snapshot.full_output_path.as_ref() {
            Some(path) => format!(". Full output: {path}"),
            None => String::new(),
        };
        if truncation.last_line_partial {
            // The partial line is the first SHOWN line; trailing blanks can follow it.
            let last_line_bytes = output.lock().expect("output lock").get_last_line_bytes();
            let line_size = if last_line_bytes > 0 {
                format!(" (line is {})", format_size(last_line_bytes))
            } else {
                String::new()
            };
            text += &format!(
                "\n\n[Showing last {} of line {start_line}{line_size}{location}]",
                format_size(truncation.output_bytes)
            );
        } else if truncation.truncated_by == Some(super::truncate::TruncatedBy::Lines) {
            text += &format!(
                "\n\n[Showing lines {start_line}-{end_line} of {}{location}]",
                truncation.total_lines
            );
        } else {
            text += &format!(
                "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit){location}]",
                truncation.total_lines,
                format_size(DEFAULT_MAX_BYTES)
            );
        }
    }
    (text, details)
}

fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// Port of `createBashToolDefinition`.
pub fn create_bash_tool_definition(cwd: &str, options: Option<&BashToolOptions>) -> ToolDefinition<BashToolDetails> {
    let operations = options
        .and_then(|options| options.operations.clone())
        .unwrap_or_else(|| create_local_bash_operations(Some(LocalBashOperationsOptions {
            shell_path: options.and_then(|options| options.shell_path.clone()),
        })));
    let command_prefix = options.and_then(|options| options.command_prefix.clone());
    let spawn_hook = options.and_then(|options| options.spawn_hook.clone());
    let cwd = cwd.to_string();

    let execute: ToolExecuteFn<BashToolDetails> = Arc::new(
        move |_tool_call_id: String,
              params: Value,
              signal: Option<CancellationToken>,
              on_update: Option<AgentToolUpdateCallback>,
              _ctx: ExtensionContext| {
            let cwd = cwd.clone();
            let operations = operations.clone();
            let command_prefix = command_prefix.clone();
            let spawn_hook = spawn_hook.clone();
            Box::pin(async move {
                let input = BashToolInput {
                    command: params
                        .get("command")
                        .and_then(|command| command.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    timeout: params.get("timeout").and_then(|timeout| timeout.as_f64()),
                };
                let (text, details) =
                    execute_bash(&cwd, operations, command_prefix.as_deref(), spawn_hook.as_ref(), &input, signal, on_update)
                        .await
                        .map_err(anyhow::Error::msg)?;
                Ok(AgentToolResult::new(
                    vec![AgentContentBlock::text(text)],
                    serde_json::to_value(details).unwrap_or(Value::Null),
                ))
            })
        },
    );

    ToolDefinition {
        name: "bash".to_string(),
        label: "bash".to_string(),
        description: bash_tool_description(),
        prompt_snippet: Some("Execute bash commands (ls, grep, find, etc.)".to_string()),
        parameters: serde_json::from_str(BASH_SCHEMA).expect("valid bash schema"),
        replay_built_in_tool_name: Some("bash".to_string()),
        execute,
        ..ToolDefinition::default()
    }
}

pub fn create_bash_tool(cwd: &str, options: Option<&BashToolOptions>) -> AgentTool {
    wrap_tool_definition(&create_bash_tool_definition(cwd, options), None)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tools::render_utils::PlainTheme;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingOperations {
        calls: Arc<AtomicUsize>,
    }

    impl BashOperations for RecordingOperations {
        fn exec(
            &self,
            command: &str,
            _cwd: &str,
            options: BashExecOptions,
        ) -> futures::future::BoxFuture<'static, Result<BashExecResult, String>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let command = command.to_string();
            let on_data = options.on_data.clone();
            let signal = options.signal.clone();
            Box::pin(async move {
                if command.contains("fail") {
                    on_data(b"boom");
                    return Ok(BashExecResult { exit_code: Some(1) });
                }
                if command.contains("abort") {
                    return Err("aborted".to_string());
                }
                if command.contains("timeout") {
                    return Err("timeout:2".to_string());
                }
                on_data(b"hello\n");
                if let Some(token) = signal {
                    token.cancelled().await;
                }
                Ok(BashExecResult { exit_code: Some(0) })
            })
        }
    }

    fn recording(calls: Arc<AtomicUsize>) -> Arc<dyn BashOperations> {
        Arc::new(RecordingOperations { calls })
    }

    #[test]
    fn description_uses_default_limits() {
        let description = bash_tool_description();
        assert!(description.contains("last 2000 lines"));
        assert!(description.contains("50KB"));
    }

    #[test]
    fn format_bash_call_previews_runner_commands() {
        let args = BashToolInput {
            command: "npx tsx ../../node_modules/vitest/dist/cli.js --run test/a.test.ts".to_string(),
            timeout: Some(30.0),
        };
        let text = format_bash_call(Some(&args), &PlainTheme);
        assert!(text.contains("$ vitest --run test/a.test.ts"));
        assert!(text.contains("(timeout 30s)"));
    }

    #[test]
    fn format_bash_call_marks_empty_command() {
        let empty = BashToolInput {
            command: String::new(),
            timeout: None,
        };
        let text = format_bash_call(Some(&empty), &PlainTheme);
        assert!(text.contains("$ ..."));
    }

    // `core/tools/bash.ts:180`: `timeout ? theme.fg(...) : ""` - `timeout: 0`
    // is falsy, so no suffix is rendered.
    #[test]
    fn format_bash_call_omits_the_suffix_for_a_zero_timeout() {
        let zero = BashToolInput {
            command: "echo hi".to_string(),
            timeout: Some(0.0),
        };
        let text = format_bash_call(Some(&zero), &PlainTheme);
        assert!(!text.contains("timeout"), "rendered: {text}");

        let absent = BashToolInput {
            command: "echo hi".to_string(),
            timeout: None,
        };
        assert!(!format_bash_call(Some(&absent), &PlainTheme).contains("timeout"));
    }

    #[test]
    fn format_duration_uses_one_decimal() {
        assert_eq!(format_duration(1500.0), "1.5s");
    }

    // `core/tools/bash.ts:320-333` `scheduleOutputUpdate`: a chunk inside the
    // throttle window is queued, not dropped, and is emitted when the window
    // elapses.
    #[tokio::test]
    async fn schedule_output_update_emits_the_trailing_chunk() {
        let output = Arc::new(std::sync::Mutex::new(OutputAccumulator::with_temp_file_prefix(
            OutputAccumulatorOptions::default(),
            "pi-bash-test-trailing",
        )));
        let throttle = Arc::new(std::sync::Mutex::new(BashUpdateThrottle::default()));
        let updates: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let collected = updates.clone();
        let on_update: AgentToolUpdateCallback = Arc::new(move |result: AgentToolResult| {
            let text = result
                .content
                .iter()
                .filter_map(|block| block.as_text())
                .collect::<Vec<&str>>()
                .join("");
            collected.lock().expect("updates lock").push(text);
        });

        schedule_output_update(&throttle, &output, &on_update);
        let first = updates.lock().expect("updates lock").len();
        // `updateTimer ??= setTimeout(...)`: a second chunk inside the window
        // must not arm a second timer.
        schedule_output_update(&throttle, &output, &on_update);
        assert_eq!(updates.lock().expect("updates lock").len(), first);

        tokio::time::sleep(std::time::Duration::from_millis(BASH_UPDATE_THROTTLE_MS + 300)).await;
        assert_eq!(
            updates.lock().expect("updates lock").len(),
            first + 1,
            "the trailing chunk must be emitted at throttle resolution"
        );
        // Trailing emission is dirty-gated: nothing changed since, so no more.
        assert!(
            !throttle.lock().expect("update lock").timer_pending,
            "the trailing timer must disarm itself"
        );
    }

    // `core/tools/bash.ts:299-311` `emitOutputUpdate` gate: a clean throttle
    // emits nothing.
    #[tokio::test]
    async fn emit_output_update_is_dirty_gated() {
        let output = Arc::new(std::sync::Mutex::new(OutputAccumulator::with_temp_file_prefix(
            OutputAccumulatorOptions::default(),
            "pi-bash-test-dirty",
        )));
        let throttle = Arc::new(std::sync::Mutex::new(BashUpdateThrottle::default()));
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let on_update: AgentToolUpdateCallback = Arc::new(move |_result: AgentToolResult| {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        emit_output_update(&throttle, &output, &on_update);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "clean throttle must not emit");

        throttle.lock().expect("update lock").dirty = true;
        emit_output_update(&throttle, &output, &on_update);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!throttle.lock().expect("update lock").dirty);
    }

    #[tokio::test]
    async fn execute_bash_streams_output_and_returns_details() {
        let calls = Arc::new(AtomicUsize::new(0));
        let input = BashToolInput {
            command: "echo hello".to_string(),
            timeout: None,
        };
        let (text, details) = execute_bash("/", recording(calls.clone()), None, None, &input, None, None)
            .await
            .expect("executed");
        assert_eq!(text, "hello\n");
        assert!(details.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn execute_bash_reports_non_zero_exit_code() {
        let calls = Arc::new(AtomicUsize::new(0));
        let input = BashToolInput {
            command: "fail".to_string(),
            timeout: None,
        };
        let error = execute_bash("/", recording(calls), None, None, &input, None, None)
            .await
            .expect_err("must fail");
        assert_eq!(error, "boom\n\nCommand exited with code 1");
    }

    #[tokio::test]
    async fn execute_bash_reports_abort_and_timeout_statuses() {
        let calls = Arc::new(AtomicUsize::new(0));
        let abort_input = BashToolInput {
            command: "abort".to_string(),
            timeout: None,
        };
        let abort_error = execute_bash("/", recording(calls.clone()), None, None, &abort_input, None, None)
            .await
            .expect_err("aborted");
        assert_eq!(abort_error, "Command aborted");

        let timeout_input = BashToolInput {
            command: "timeout".to_string(),
            timeout: Some(2.0),
        };
        let timeout_error = execute_bash("/", recording(calls), None, None, &timeout_input, None, None)
            .await
            .expect_err("timed out");
        assert_eq!(timeout_error, "Command timed out after 2 seconds");
    }

    #[tokio::test]
    async fn execute_bash_applies_command_prefix() {
        let calls = Arc::new(AtomicUsize::new(0));
        let input = BashToolInput {
            command: "echo hello".to_string(),
            timeout: None,
        };
        let (text, _) = execute_bash("/", recording(calls), Some("set -e"), None, &input, None, None)
            .await
            .expect("executed");
        assert_eq!(text, "hello\n");
    }

    #[tokio::test]
    async fn execute_bash_rejects_missing_working_directory() {
        let calls = Arc::new(AtomicUsize::new(0));
        let input = BashToolInput {
            command: "echo hello".to_string(),
            timeout: None,
        };
        let error = execute_bash(
            "/definitely/missing/dir",
            create_local_bash_operations(None),
            None,
            None,
            &input,
            None,
            None,
        )
        .await
        .expect_err("must reject");
        assert!(error.starts_with(
            "Working directory does not exist: /definitely/missing/dir\nCannot execute bash commands."
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    // `test/tools.test.ts:322-343` "should pass shellPath through to shell
    // resolution" plus `utils/shell.ts:66-69`: the tool must go through the real
    // owner, so a bad shellPath surfaces the teaching error (not a spawn error).
    #[tokio::test]
    async fn local_operations_reject_a_missing_custom_shell_path() {
        let operations = create_local_bash_operations(Some(LocalBashOperationsOptions {
            shell_path: Some("/custom/bash".to_string()),
        }));
        let error = operations
            .exec(
                "echo test",
                ".",
                BashExecOptions {
                    on_data: Arc::new(|_| {}),
                    signal: None,
                    timeout: None,
                    env: None,
                },
            )
            .await
            .expect_err("must reject");
        assert_eq!(error, "Custom shell path not found: /custom/bash");
    }

    // `utils/shell.ts:64-130` never consults `$SHELL`: unix always uses
    // `/bin/bash` first. Regression guard for the removed `$SHELL` shadow.
    #[test]
    fn shell_resolution_ignores_the_shell_env_var_on_unix() {
        if cfg!(windows) {
            return;
        }
        let config = get_shell_config(None).expect("resolved shell");
        assert_eq!(config.shell, "/bin/bash");
        assert_eq!(config.args, vec!["-c".to_string()]);
    }

    // End-to-end guard for THIS host: the shell the tool now resolves through
    // the `utils/shell.ts` owner must exist and must actually run a command.
    // On win32 the old shadow used a bare `"bash"` from PATH and no Git Bash
    // candidate was consulted (`utils/shell.ts:100-116`).
    #[tokio::test]
    async fn local_operations_run_the_resolved_shell() {
        // Honest skip: a host with no bash at all gets TS's teaching error from
        // `getLocalShellConfig`, which the test above already covers.
        let Ok(config) = get_shell_config(None) else {
            return;
        };
        assert!(
            std::path::Path::new(&config.shell).exists(),
            "resolved shell must exist: {}",
            config.shell
        );
        assert_eq!(config.args, vec!["-c".to_string()]);

        let operations = create_local_bash_operations(None);
        let received: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = received.clone();
        let result = operations
            .exec(
                "echo pi-bash-shell-ok",
                ".",
                BashExecOptions {
                    on_data: Arc::new(move |data: &[u8]| {
                        sink.lock().expect("sink lock").extend_from_slice(data);
                    }),
                    signal: None,
                    timeout: Some(30.0),
                    env: None,
                },
            )
            .await
            .expect("executed");
        assert_eq!(result.exit_code, Some(0));
        let stdout = String::from_utf8_lossy(&received.lock().expect("sink lock")).to_string();
        assert!(stdout.contains("pi-bash-shell-ok"), "stdout: {stdout}");
    }

    // `core/tools/bash.ts:135` (`resolveSpawnContext`) and
    // `utils/shell.ts:147-161`: the spawn env carries the `getBinDir()` prepend.
    #[test]
    fn spawn_context_env_uses_the_shell_env_owner() {
        let context = resolve_spawn_context("echo hi", ".", None);
        let bin_dir = crate::utils::tools_manager::get_bin_dir().to_string_lossy().to_string();
        let path_entry = context
            .env
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .expect("PATH entry");
        assert!(path_entry.starts_with(&bin_dir));
        assert_eq!(context.command, "echo hi");
    }

    #[test]
    fn render_component_shows_preview_hint_and_duration() {
        let result = RenderResultLike {
            content: vec![RenderContentBlock::from_text("l1\nl2\nl3\nl4\nl5\nl6\nl7")],
        };
        let mut component = BashResultRenderComponent::default();
        rebuild_bash_result_render_component(
            &mut component,
            Some(&result),
            None,
            super::super::ToolRenderResultOptions::default(),
            true,
            true,
            true,
            Some(0.0),
            Some(1000.0),
            &PlainTheme,
        );
        assert_eq!(component.state.cached_skipped, Some(2));
        assert!(component.children.iter().any(|child| child.contains("... 2 earlier lines")));
        assert!(component.children.iter().any(|child| child.contains("Took 1.0s")));
    }

    #[test]
    fn render_component_reports_truncation_warnings() {
        let result = RenderResultLike {
            content: vec![RenderContentBlock::from_text("out")],
        };
        let details = BashToolDetails {
            truncation: Some(super::super::truncate::TruncationResult {
                content: "out".to_string(),
                truncated: true,
                truncated_by: Some(super::super::truncate::TruncatedBy::Lines),
                total_lines: 10,
                total_bytes: 100,
                output_lines: 2,
                output_bytes: 8,
                last_line_partial: false,
                first_line_exceeds_limit: false,
                max_lines: 2,
                max_bytes: 1024,
            }),
            full_output_path: Some("/tmp/full.log".to_string()),
        };
        let mut component = BashResultRenderComponent::default();
        rebuild_bash_result_render_component(
            &mut component,
            Some(&result),
            Some(&details),
            super::super::ToolRenderResultOptions::default(),
            true,
            true,
            true,
            None,
            None,
            &PlainTheme,
        );
        let warning = component
            .children
            .iter()
            .find(|child| child.contains("Full output: /tmp/full.log"))
            .expect("warning row");
        assert!(warning.contains("Truncated: showing 2 of 10 lines"));
    }
}
