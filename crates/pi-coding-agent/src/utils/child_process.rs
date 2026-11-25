//! Port of packages/coding-agent/src/utils/child-process.ts

use std::io::Read;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(windows)]
#[path = "child_process/kernel_job.rs"]
pub mod kernel_job;
#[cfg(windows)]
pub use kernel_job::{spawn_kernel, KernelJob, KernelProcess};
#[cfg(not(windows))]
pub type KernelProcess = OwnedChild;

#[cfg(not(windows))]
pub fn spawn_kernel(command: &str, args: &[String], options: SpawnOptions) -> std::io::Result<KernelProcess> {
    Ok(spawn_hidden(command, args, options)?.child)
}

pub const EXIT_STDIO_GRACE_MS: u64 = 100;

/// `windowsHide` for every non-interactive spawn.
#[derive(Debug, Clone, Default)]
pub struct SpawnOptions {
    pub cwd: Option<String>,
    /// `env`. Node's `spawn` replaces the child environment with this map; Rust's
    /// `Command::envs` only adds to the parent's. `replace_env` selects the Node
    /// behaviour while every existing caller keeps the additive default.
    pub env: Option<Vec<(String, String)>>,
    /// `false` (default): `env` is added on top of the inherited environment
    /// (Rust `Command::envs` semantics). `true`: `env` is the child's whole
    /// environment, as Node's `spawn(env)` does (env_clear + envs).
    pub replace_env: bool,
    pub detached: bool,
    /// Run through the platform shell as Node's `shell: true` does.
    pub shell: bool,
    pub capture_stdout: bool,
    pub capture_stderr: bool,
    /// `stdio[0] = "pipe"`: the caller writes the child's stdin.
    pub stdin_piped: bool,
}

/// Owned asynchronous child. Non-detached children remain owned until an
/// explicit wait consumes them; cancelling that wait kills the process tree
/// and Tokio's kill-on-drop reaper schedules the final wait. A deliberately
/// detached spawn has no drop kill, matching the caller's detach contract.
pub struct OwnedChild {
    child: tokio::process::Child,
    detached: bool,
    process: Option<crate::utils::shell::OwnedChildProcess>,
}

impl OwnedChild {
    pub(crate) fn new(child: tokio::process::Child, detached: bool) -> Self {
        // Capture ownership at the real construction boundary, before a later
        // detached-registry call or cancellation can act on a recycled PID.
        let process = child
            .id()
            .and_then(|pid| crate::utils::shell::register_owned_child_process(pid as i32));
        Self { child, detached, process }
    }

    pub(crate) fn owned_process(&self) -> Option<&crate::utils::shell::OwnedChildProcess> {
        self.process.as_ref()
    }
}

impl std::ops::Deref for OwnedChild {
    type Target = tokio::process::Child;

    fn deref(&self) -> &Self::Target {
        &self.child
    }
}

impl std::ops::DerefMut for OwnedChild {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.child
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        let Some(process) = self.process.as_ref() else {
            return;
        };
        if self.detached {
            crate::utils::shell::release_owned_child_process(process);
            return;
        }
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            // Poll errors fail closed: the inner kill-on-drop still owns
            // the root, while this path also reaches its descendants.
            crate::utils::shell::kill_owned_child_process_tree(process);
            let _ = self.child.start_kill();
        }
        crate::utils::shell::release_owned_child_process(process);
    }
}

pub struct ChildProcessHandle {
    pub child: OwnedChild,
}

pub fn spawn_hidden(command: &str, args: &[String], options: SpawnOptions) -> std::io::Result<ChildProcessHandle> {
    let mut builder = build_tokio_command(command, args, &options);
    builder.kill_on_drop(!options.detached);
    Ok(ChildProcessHandle {
        child: OwnedChild::new(builder.spawn()?, options.detached),
    })
}

/// Test-only seam (parity-validation fixture): lets a test intercept the
/// synchronous spawn boundary used by `taskkill` (kernel cleanup) and `where`
/// (bash resolution) without touching those call sites. Returning `Some`
/// overrides the real spawn with a caller-provided child handle (a test can
/// hand back a long-lived child to model a hung runner); `None` falls through
/// unchanged. Inert when unset.
type SyncSpawnOverrideFn = fn(&str, &[String], &SpawnOptions) -> Option<std::io::Result<std::process::Child>>;
static SYNC_SPAWN_OVERRIDE: Mutex<Option<SyncSpawnOverrideFn>> = Mutex::new(None);

#[doc(hidden)]
pub fn set_sync_spawn_override_for_tests(override_fn: Option<SyncSpawnOverrideFn>) {
    *SYNC_SPAWN_OVERRIDE.lock().unwrap() = override_fn;
}

fn drain_child_pipe<P: std::io::Read>(pipe: Option<P>) -> Vec<u8> {
    let mut buffer = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut buffer);
    }
    buffer
}

fn wait_sync_child(
    mut child: std::process::Child,
) -> std::io::Result<std::process::Output> {
    let status = child.wait()?;
    Ok(std::process::Output {
        status,
        stdout: drain_child_pipe(child.stdout.take()),
        stderr: drain_child_pipe(child.stderr.take()),
    })
}

pub fn spawn_sync_hidden(
    command: &str,
    args: &[String],
    options: SpawnOptions,
) -> std::io::Result<std::process::Output> {
    if let Some(override_fn) = SYNC_SPAWN_OVERRIDE.lock().unwrap().as_ref() {
        if let Some(child) = override_fn(command, args, &options) {
            // The unbounded path: an overridden child is waited on without a
            // deadline, which is exactly the defect these tests exercise.
            return wait_sync_child(child?);
        }
    }
    let mut builder = build_std_command(command, args, &options);
    builder.output()
}

/// `spawnSync` with a Node-style timeout (child-process.ts: `execFileSync`
/// `{ timeout }`): on expiry the child is killed and the call errors, so a
/// hung runner can never wedge a synchronous teardown path.
pub fn spawn_sync_hidden_with_timeout(
    command: &str,
    args: &[String],
    options: SpawnOptions,
    timeout_ms: u64,
) -> std::io::Result<std::process::Output> {
    if let Some(override_fn) = SYNC_SPAWN_OVERRIDE.lock().unwrap().as_ref() {
        if let Some(child) = override_fn(command, args, &options) {
            return wait_for_sync_child_with_timeout(child?, timeout_ms);
        }
    }
    let mut builder = build_std_command(command, args, &options);
    let child = builder.spawn()?;
    wait_for_sync_child_with_timeout(child, timeout_ms)
        .map_err(|error| sync_timeout_error(error, command, timeout_ms))
}

fn sync_timeout_error(error: std::io::Error, command: &str, timeout_ms: u64) -> std::io::Error {
    if error.kind() == std::io::ErrorKind::TimedOut {
        std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("command timed out after {}ms: {}", timeout_ms, command),
        )
    } else {
        error
    }
}

struct SyncPipeDrain {
    receiver: std::sync::mpsc::Receiver<Vec<u8>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn start_sync_pipe_drain<P: Read + Send + 'static>(mut pipe: P) -> SyncPipeDrain {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = pipe.read_to_end(&mut buffer);
        let _ = sender.send(buffer);
    });
    SyncPipeDrain {
        receiver,
        thread: Some(thread),
    }
}

impl SyncPipeDrain {
    /// Receive and join only inside the caller's original deadline. If a
    /// descendant inherited the pipe, the blocked reader owns it and is left
    /// detached rather than wedging process cleanup or accumulating a join.
    fn finish(mut self, deadline: Instant) -> Vec<u8> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match self.receiver.recv_timeout(remaining) {
            Ok(buffer) => {
                if let Some(thread) = self.thread.take() {
                    let _ = thread.join();
                }
                buffer
            }
            Err(_) => {
                drop(self.thread.take());
                Vec::new()
            }
        }
    }
}

/// Wait, reap, and drain a synchronous child under one deadline. Both captured
/// pipes are drained while the child runs, preventing a full-pipe deadlock.
pub fn wait_for_sync_child_with_timeout(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> std::io::Result<std::process::Output> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let stdout = child.stdout.take().map(start_sync_pipe_drain);
    let stderr = child.stderr.take().map(start_sync_pipe_drain);
    let status = wait_with_timeout(&mut child, timeout_ms)?
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "command timed out"))?;
    let stdout = stdout.map(|pipe| pipe.finish(deadline)).unwrap_or_default();
    let stderr = stderr.map(|pipe| pipe.finish(deadline)).unwrap_or_default();
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

pub fn exec_sync_hidden(command: &str, options: SpawnOptions) -> std::io::Result<std::process::Output> {
    // Node runs the string through the shell; cmd.exe /c and /bin/sh -c are the equivalents.
    let mut builder = shell_command(command);
    apply_std_options(&mut builder, &options);
    builder.output()
}

pub fn exec_file_hidden(
    file: &str,
    args: &[String],
    options: SpawnOptions,
) -> std::io::Result<std::process::Output> {
    let mut builder = build_std_command(file, args, &options);
    builder.output()
}

pub fn exec_file_sync_hidden(
    file: &str,
    args: &[String],
    options: SpawnOptions,
) -> std::io::Result<std::process::Output> {
    let mut builder = build_std_command(file, args, &options);
    builder.output()
}

/// `execSync(command, { input, timeout, stdio: ["pipe", "ignore", "ignore"] })`.
pub fn exec_sync_hidden_with_input(
    command: &str,
    input: &str,
    options: SpawnOptions,
    timeout_ms: u64,
) -> std::io::Result<std::process::ExitStatus> {
    use std::io::Write;
    let mut builder = shell_command(command);
    let mut options = options;
    options.stdin_piped = true;
    apply_std_options(&mut builder, &options);
    let mut child = builder.spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input.as_bytes());
        let _ = stdin.flush();
        drop(stdin);
    }
    match wait_with_timeout(&mut child, timeout_ms)? {
        Some(status) => Ok(status),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("command timed out after {}ms: {}", timeout_ms, command),
        )),
    }
}

fn shell_command(command: &str) -> std::process::Command {
    if cfg!(windows) {
        let mut builder = std::process::Command::new("cmd");
        builder.arg("/c").arg(command);
        builder
    } else {
        let mut builder = std::process::Command::new("/bin/sh");
        builder.arg("-c").arg(command);
        builder
    }
}

fn apply_std_options(builder: &mut std::process::Command, options: &SpawnOptions) {
    if let Some(cwd) = &options.cwd {
        builder.current_dir(cwd);
    }
    if let Some(env) = &options.env {
        if options.replace_env {
            builder.env_clear();
        }
        builder.envs(env.iter().map(|(key, value)| (key.clone(), value.clone())));
    }
    builder.stdin(if options.stdin_piped {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    builder.stdout(if options.capture_stdout {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    builder.stderr(if options.capture_stderr {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    hide_console_window_std(builder);
}

fn build_std_command(command: &str, args: &[String], options: &SpawnOptions) -> std::process::Command {
    let mut builder = if options.shell && cfg!(windows) {
        let mut builder = std::process::Command::new("cmd");
        builder.arg("/c").arg(command);
        builder.args(args);
        builder
    } else if options.shell {
        // Node joins command and arguments before handing them to /bin/sh.
        shell_command(&std::iter::once(command).chain(args.iter().map(String::as_str)).collect::<Vec<_>>().join(" "))
    } else {
        let mut builder = std::process::Command::new(command);
        builder.args(args);
        builder
    };
    apply_std_options(&mut builder, options);
    builder
}

#[cfg(windows)]
use std::os::windows::process::CommandExt as _;

fn build_tokio_command(command: &str, args: &[String], options: &SpawnOptions) -> tokio::process::Command {
    let mut builder = if options.shell && cfg!(windows) {
        let mut builder = tokio::process::Command::new("cmd");
        builder.arg("/c").arg(command);
        builder.args(args);
        builder
    } else if options.shell {
        tokio::process::Command::from(shell_command(
            &std::iter::once(command).chain(args.iter().map(String::as_str)).collect::<Vec<_>>().join(" "),
        ))
    } else {
        let mut builder = tokio::process::Command::new(command);
        builder.args(args);
        builder
    };
    if let Some(cwd) = &options.cwd {
        builder.current_dir(cwd);
    }
    if let Some(env) = &options.env {
        if options.replace_env {
            builder.env_clear();
        }
        builder.envs(env.iter().map(|(key, value)| (key.clone(), value.clone())));
    }
    builder.stdin(if options.stdin_piped {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    builder.stdout(if options.capture_stdout {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    builder.stderr(if options.capture_stderr {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    if options.detached {
        #[cfg(unix)]
        builder.process_group(0);
    }
    hide_console_window_tokio(&mut builder);
    builder
}

#[cfg(windows)]
fn hide_console_window_std(builder: &mut std::process::Command) {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    builder.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console_window_std(_builder: &mut std::process::Command) {}

#[cfg(windows)]
fn hide_console_window_tokio(builder: &mut tokio::process::Command) {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    builder.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console_window_tokio(_builder: &mut tokio::process::Command) {}

const WINDOWS_SHELL_COMMANDS: [&str; 6] = ["npm", "npx", "pnpm", "yarn", "yarnpkg", "corepack"];

pub fn should_use_windows_shell(command: &str) -> bool {
    if !cfg!(windows) {
        return false;
    }
    let command_name = Path::new(command)
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| command.to_lowercase());
    command_name.ends_with(".cmd")
        || command_name.ends_with(".bat")
        || WINDOWS_SHELL_COMMANDS.contains(&command_name.as_str())
}

/// Cheap kill(0) existence probe; counts zombies as existing.
pub fn process_id_exists(pid: i32) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if handle.is_null() {
                // ERROR_ACCESS_DENIED means the process exists but is not ours.
                return std::io::Error::last_os_error().raw_os_error() == Some(5);
            }
            CloseHandle(handle);
            true
        }
    }
    #[cfg(not(windows))]
    {
        unsafe {
            if libc::kill(pid, 0) == 0 {
                return true;
            }
            std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
        }
    }
}

/// A zombie has already exited; it only lingers until its parent reaps it.
pub fn is_zombie_process(pid: i32) -> bool {
    if cfg!(windows) {
        return false;
    }

    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        if let Some(index) = stat.rfind(')') {
            let state = stat[index + 1..].trim_start().chars().next();
            return state == Some('Z');
        }
    }

    // Fall through to the portable process listing used on macOS and BSD.
    match exec_file_sync_hidden(
        "ps",
        &[
            "-p".to_string(),
            pid.to_string(),
            "-o".to_string(),
            "stat=".to_string(),
        ],
        SpawnOptions {
            capture_stdout: true,
            ..Default::default()
        },
    ) {
        Ok(output) => String::from_utf8_lossy(&output.stdout).trim().starts_with('Z'),
        Err(_) => false,
    }
}

/// True only for a process that is actually running: zombies do not count.
pub fn is_process_alive(pid: i32) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};

        // An exited process can still be opened by PID while a handle is held.
        // Its process object is signaled even if its exit code is STILL_ACTIVE.
        unsafe {
            let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid as u32);
            if handle.is_null() {
                return process_id_exists(pid);
            }
            let exited = WaitForSingleObject(handle, 0) == WAIT_OBJECT_0;
            CloseHandle(handle);
            !exited
        }
    }
    #[cfg(not(windows))]
    {
        process_id_exists(pid) && !is_zombie_process(pid)
    }
}

/// True while the group has any member left, zombies included; a group can outlive its leader.
pub fn process_group_exists(pgid: i32) -> bool {
    #[cfg(windows)]
    {
        let _ = pgid;
        false
    }
    #[cfg(not(windows))]
    unsafe {
        if libc::kill(-pgid, 0) == 0 {
            return true;
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

/// True while the group has a RUNNING member; unreaped zombies have exited and must not block a group stop.
pub fn process_group_has_live_member(pgid: i32) -> bool {
    if !process_group_exists(pgid) {
        return false;
    }

    #[cfg(windows)]
    {
        let _ = pgid;
        return false;
    }

    #[cfg(not(windows))]
    {
        let listing = exec_file_sync_hidden(
            "ps",
            &[
                "-A".to_string(),
                "-o".to_string(),
                "pgid=".to_string(),
                "-o".to_string(),
                "stat=".to_string(),
            ],
            SpawnOptions {
                capture_stdout: true,
                ..Default::default()
            },
        );
        match listing {
            Ok(output) => {
                for line in String::from_utf8_lossy(&output.stdout).split('\n') {
                    let fields: Vec<&str> = line.split_whitespace().collect();
                    if fields.len() < 2 {
                        continue;
                    }
                    if fields[0].parse::<i32>() == Ok(pgid) && !fields[1].starts_with('Z') {
                        return true;
                    }
                }
                false
            }
            // Unverifiable listing reads alive: callers keep escalating instead of dropping records over live descendants.
            Err(_) => true,
        }
    }
}

/// Signal the group only while it is provably still the target: the leader process (even a zombie)
/// anchors its pgid against reuse; once the leader is gone, a live member must hold the pgid at
/// signal time, narrowing reuse exposure to the inherent kill() TOCTOU of any single-pid signal.
pub fn signal_process_group_if_held(pgid: i32, signal: Signal) -> bool {
    if !process_id_exists(pgid) && !process_group_has_live_member(pgid) {
        return false;
    }
    signal_process_group_or_process(pgid, signal);
    true
}

/// Returns whether a signal was actually delivered (Node's `child.kill()`
/// result); the fallback journal stays active when no signal landed.
pub fn signal_process_group_or_process(pid: i32, signal: Signal) -> bool {
    if send_signal(-pid, signal) {
        return true;
    }
    // Fall back when process groups are unavailable or the group already exited.
    send_signal(pid, signal)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Hup,
    Int,
    Term,
    Kill,
    Usr1,
    Usr2,
}

impl Signal {
    pub fn name(self) -> &'static str {
        match self {
            Signal::Hup => "SIGHUP",
            Signal::Int => "SIGINT",
            Signal::Term => "SIGTERM",
            Signal::Kill => "SIGKILL",
            Signal::Usr1 => "SIGUSR1",
            Signal::Usr2 => "SIGUSR2",
        }
    }

    /// Node's `os.constants.signals` numbers for the signals used in this crate.
    pub fn number(self) -> i32 {
        match self {
            Signal::Hup => 1,
            Signal::Int => 2,
            Signal::Kill => 9,
            Signal::Usr1 => 10,
            Signal::Usr2 => 12,
            Signal::Term => 15,
        }
    }
}

#[cfg(not(windows))]
fn send_signal(pid: i32, signal: Signal) -> bool {
    let number = match signal {
        Signal::Hup => libc::SIGHUP,
        Signal::Int => libc::SIGINT,
        Signal::Term => libc::SIGTERM,
        Signal::Kill => libc::SIGKILL,
        Signal::Usr1 => libc::SIGUSR1,
        Signal::Usr2 => libc::SIGUSR2,
    };
    unsafe { libc::kill(pid, number) == 0 }
}

#[cfg(windows)]
fn send_signal(pid: i32, _signal: Signal) -> bool {
    // Windows has no POSIX signals; terminating the process is the only equivalent.
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid as u32);
        if handle.is_null() {
            return false;
        }
        let ok = TerminateProcess(handle, 1) != 0;
        CloseHandle(handle);
        ok
    }
}

pub fn signal_exit_code(signal: Option<Signal>) -> Option<i32> {
    let signal = signal?;
    Some(128 + signal.number())
}

pub fn normalized_exit_code(code: Option<i32>, signal: Option<Signal>) -> Option<i32> {
    code.or_else(|| signal_exit_code(signal))
}

/// Captured output from an asynchronous child whose pipes were drained while
/// the process ran. Pipes still held by descendants are closed after the grace
/// period. Output from a pipe that does not close in time may be omitted.
pub struct ChildProcessOutput {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

async fn read_async_pipe<P>(pipe: Option<P>) -> Vec<u8>
where
    P: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = tokio::io::AsyncReadExt::read_to_end(&mut pipe, &mut buffer).await;
    }
    buffer
}

async fn finish_async_pipe(
    mut task: tokio::task::JoinHandle<Vec<u8>>,
    deadline: tokio::time::Instant,
) -> Vec<u8> {
    match tokio::time::timeout_at(deadline, &mut task).await {
        Ok(Ok(buffer)) => buffer,
        Ok(Err(_)) => Vec::new(),
        Err(_) => {
            task.abort();
            let _ = task.await;
            Vec::new()
        }
    }
}

/// Wait for a child while concurrently draining both captured pipes. Waiting
/// for exit before starting a reader can deadlock as soon as either pipe fills.
pub async fn wait_for_child_process_with_output(child: &mut OwnedChild) -> std::io::Result<ChildProcessOutput> {
    let stdout = tokio::spawn(read_async_pipe(child.stdout.take()));
    let stderr = tokio::spawn(read_async_pipe(child.stderr.take()));
    let status = match child.wait().await {
        Ok(status) => status,
        Err(error) => {
            stdout.abort();
            stderr.abort();
            return Err(error);
        }
    };

    // Shell exit does not imply EOF. Drain both readers under one shared,
    // post-exit deadline instead of giving inherited handles sequential grace.
    let drain_deadline = tokio::time::Instant::now() + Duration::from_millis(EXIT_STDIO_GRACE_MS);
    let (stdout, stderr) = tokio::join!(
        finish_async_pipe(stdout, drain_deadline),
        finish_async_pipe(stderr, drain_deadline),
    );
    Ok(ChildProcessOutput { status, stdout, stderr })
}

pub async fn wait_for_child_process(mut child: OwnedChild) -> std::io::Result<Option<i32>> {
    let output = wait_for_child_process_with_output(&mut child).await?;
    Ok(exit_code_for_status(&output.status))
}

/// Node-compatible exit code from the status itself. Do not map unknown Unix
/// signals through the small public `Signal` enum: SIGUSR1/SIGABRT and future
/// signal values must retain their actual platform numbers.
#[cfg(unix)]
pub fn exit_code_for_status(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    Some(status.code().unwrap_or_else(|| 128 + status.signal().unwrap_or(0)))
}

#[cfg(not(unix))]
pub fn exit_code_for_status(status: &std::process::ExitStatus) -> Option<i32> {
    status.code()
}

/// Reads a child's captured stdout after the process finished.
pub fn read_child_stdout(output: &mut std::process::Child) -> String {
    let mut buffer = String::new();
    if let Some(stdout) = output.stdout.as_mut() {
        let _ = stdout.read_to_string(&mut buffer);
    }
    buffer
}

trait SyncChildWait {
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>>;
    fn kill(&mut self) -> std::io::Result<()>;
    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus>;
}

impl SyncChildWait for std::process::Child {
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        std::process::Child::try_wait(self)
    }

    fn kill(&mut self) -> std::io::Result<()> {
        std::process::Child::kill(self)
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        std::process::Child::wait(self)
    }
}

fn wait_with_timeout_inner<C: SyncChildWait>(
    child: &mut C,
    timeout_ms: u64,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(None) => {
                // A failed kill is not a timeout success: report it immediately
                // rather than either discarding the error or waiting forever.
                child.kill().map_err(|error| {
                    std::io::Error::new(
                        error.kind(),
                        format!("failed to kill child after {timeout_ms}ms timeout: {error}"),
                    )
                })?;
                // A successful kill must be reaped before the caller returns.
                child.wait().map_err(|error| {
                    std::io::Error::new(
                        error.kind(),
                        format!("failed to reap killed child after {timeout_ms}ms timeout: {error}"),
                    )
                })?;
                return Ok(None);
            }
            Err(error) => {
                return Err(std::io::Error::new(
                    error.kind(),
                    format!("failed to poll child status: {error}"),
                ));
            }
        }
    }
}

/// Bounded wait used by synchronous helpers. `Ok(None)` means the deadline
/// expired, the kill succeeded, and the child was reaped. Poll, kill, and reap
/// failures remain distinguishable `Err` values.
pub fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout_ms: u64,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    wait_with_timeout_inner(child, timeout_ms)
}


#[cfg(test)]
mod tests {
    use super::*;

    const HELPER_ENV: &str = "OPTIMUS_CHILD_PROCESS_TEST_HELPER";
    const HELPER_TEST: &str = "utils::child_process::tests::child_process_test_helper";
    const LARGE_OUTPUT_BYTES: usize = 1_100_000;

    fn helper_args() -> Vec<String> {
        vec![
            "--exact".to_string(),
            HELPER_TEST.to_string(),
            "--nocapture".to_string(),
        ]
    }

    fn helper_options(mode: &str, capture: bool) -> SpawnOptions {
        SpawnOptions {
            env: Some(vec![(HELPER_ENV.to_string(), mode.to_string())]),
            capture_stdout: capture,
            capture_stderr: capture,
            ..Default::default()
        }
    }

    fn spawn_async_helper(mode: &str, detached: bool, capture: bool) -> ChildProcessHandle {
        let mut options = helper_options(mode, capture);
        options.detached = detached;
        spawn_hidden(
            std::env::current_exe().unwrap().to_str().unwrap(),
            &helper_args(),
            options,
        )
        .unwrap()
    }

    fn spawn_sync_helper(mode: &str, capture: bool, timeout_ms: u64) -> std::io::Result<std::process::Output> {
        spawn_sync_hidden_with_timeout(
            std::env::current_exe().unwrap().to_str().unwrap(),
            &helper_args(),
            helper_options(mode, capture),
            timeout_ms,
        )
    }

    /// Test-owned process entrypoint. It is inert during the normal suite and
    /// active only in a process spawned with `HELPER_ENV` by these tests.
    #[test]
    fn child_process_test_helper() {
        let Ok(mode) = std::env::var(HELPER_ENV) else {
            return;
        };
        match mode.as_str() {
            "large-output" => {
                use std::io::Write;
                std::io::stdout().write_all(&vec![b'o'; LARGE_OUTPUT_BYTES]).unwrap();
                std::io::stdout().flush().unwrap();
                std::io::stderr().write_all(&vec![b'e'; LARGE_OUTPUT_BYTES]).unwrap();
                std::io::stderr().flush().unwrap();
            }
            "hold-pipes" => {
                let mut options = helper_options("sleep-brief", false);
                options.capture_stdout = false;
                options.capture_stderr = false;
                let mut builder = build_std_command(
                    std::env::current_exe().unwrap().to_str().unwrap(),
                    &helper_args(),
                    &options,
                );
                // Inherit this helper's captured output handles in the
                // short-lived descendant after the direct child exits.
                builder.stdout(Stdio::inherit()).stderr(Stdio::inherit());
                builder.spawn().unwrap();
            }
            "sleep-brief" => std::thread::sleep(Duration::from_secs(2)),
            "sleep" => std::thread::sleep(Duration::from_secs(30)),
            unexpected => panic!("unexpected helper mode: {unexpected}"),
        }
        std::process::exit(0);
    }

    #[cfg(unix)]
    fn wait_until_reaped(pid: i32) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if !is_process_alive(pid) && !is_zombie_process(pid) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("child {pid} was not reaped");
    }

    #[cfg(unix)]
    #[test]
    fn shell_spawn_interprets_the_joined_command_and_arguments() {
        let output = spawn_sync_hidden(
            "printf '%s'",
            &["'hello world'".to_string()],
            SpawnOptions { shell: true, capture_stdout: true, ..Default::default() },
        ).unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"hello world");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_spawn_reports_success_and_failure() {
        for code in [0, 7] {
            let handle = spawn_hidden(
                &format!("exit {code}"),
                &[],
                SpawnOptions { shell: true, ..Default::default() },
            ).unwrap();
            assert_eq!(wait_for_child_process(handle.child).await.unwrap(), Some(code));
        }
    }

    #[test]
    fn windows_shell_detection_matches_the_typescript() {
        if !cfg!(windows) {
            assert!(!should_use_windows_shell("npm"));
            return;
        }
        assert!(should_use_windows_shell("npm"));
        assert!(should_use_windows_shell("C:\\tools\\npx.cmd"));
        assert!(should_use_windows_shell("build.bat"));
        assert!(!should_use_windows_shell("git"));
    }

    #[test]
    fn own_process_exists_and_is_not_a_zombie() {
        let pid = std::process::id() as i32;
        assert!(process_id_exists(pid));
        assert!(!is_zombie_process(pid));
        assert!(is_process_alive(pid));
    }

    #[test]
    fn signal_exit_codes_match_node() {
        assert_eq!(signal_exit_code(None), None);
        assert_eq!(signal_exit_code(Some(Signal::Term)), Some(128 + 15));
        assert_eq!(normalized_exit_code(Some(3), Some(Signal::Kill)), Some(3));
        assert_eq!(normalized_exit_code(None, Some(Signal::Kill)), Some(128 + 9));
    }

    #[test]
    fn spawn_sync_hidden_captures_stdout() {
        let command = if cfg!(windows) { "cmd" } else { "sh" };
        let args = if cfg!(windows) {
            vec!["/c".to_string(), "echo hi".to_string()]
        } else {
            vec!["-c".to_string(), "echo hi".to_string()]
        };
        let output = spawn_sync_hidden(
            command,
            &args,
            SpawnOptions {
                capture_stdout: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(String::from_utf8_lossy(&output.stdout).contains("hi"));
    }

    #[tokio::test]
    async fn wait_for_child_process_reports_the_exit_code() {
        let command = if cfg!(windows) { "cmd" } else { "sh" };
        let args = if cfg!(windows) {
            vec!["/c".to_string(), "exit 7".to_string()]
        } else {
            vec!["-c".to_string(), "exit 7".to_string()]
        };
        let handle = spawn_hidden(
            command,
            &args,
            SpawnOptions {
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(wait_for_child_process(handle.child).await.unwrap(), Some(7));
    }

    #[test]
    fn process_group_probes_are_false_on_windows() {
        if cfg!(windows) {
            assert!(!process_group_exists(1234));
            assert!(!process_group_has_live_member(1234));
        }
    }

    /// G2-01: the bounded sync spawn must kill a hanging child at the timeout
    /// and report TimedOut instead of blocking forever.
    #[cfg(windows)]
    #[test]
    fn spawn_sync_hidden_with_timeout_bounds_a_hanging_child() {
        let started = std::time::Instant::now();
        let result = spawn_sync_hidden_with_timeout(
            "ping",
            &[
                "-n".to_string(),
                "30".to_string(),
                "127.0.0.1".to_string(),
            ],
            SpawnOptions::default(),
            500,
        );
        match result {
            Err(error) => {
                assert!(error.kind() == std::io::ErrorKind::TimedOut, "expected TimedOut, got {error}");
                assert!(started.elapsed() < std::time::Duration::from_secs(5), "the hang must be bounded");
            }
            Ok(_) => panic!("a 30s ping must not complete within a 500ms timeout"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn spawn_sync_hidden_with_timeout_returns_completed_output() {
        let result = spawn_sync_hidden_with_timeout(
            "cmd",
            &["/c".to_string(), "echo parity-ok".to_string()],
            SpawnOptions {
                capture_stdout: true,
                ..Default::default()
            },
            5000,
        )
        .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), "parity-ok");
    }

    #[tokio::test]
    async fn async_wait_drains_large_captured_pipes_without_external_readers() {
        let mut handle = spawn_async_helper("large-output", false, true);
        let result = tokio::time::timeout(Duration::from_secs(10), async {
            wait_for_child_process_with_output(&mut handle.child).await.unwrap()
        })
        .await
        .expect("full pipes must not deadlock waiting for child exit");
        assert!(result.status.success());
        assert!(result.stdout.len() >= LARGE_OUTPUT_BYTES);
        assert!(result.stderr.len() >= LARGE_OUTPUT_BYTES);
    }

    #[tokio::test]
    async fn async_wait_also_drains_when_output_is_not_returned() {
        let handle = spawn_async_helper("large-output", false, true);
        let code = tokio::time::timeout(
            Duration::from_secs(10),
            wait_for_child_process(handle.child),
        )
        .await
        .expect("uncaptured large output must still be drained")
        .unwrap();
        assert_eq!(code, Some(0));
    }

    #[test]
    fn sync_wait_drains_large_captured_pipes_during_the_wait() {
        let output = spawn_sync_helper("large-output", true, 10_000).unwrap();
        assert!(output.status.success());
        assert!(output.stdout.len() >= LARGE_OUTPUT_BYTES);
        assert!(output.stderr.len() >= LARGE_OUTPUT_BYTES);
    }

    #[test]
    fn sync_wait_binds_inherited_pipe_drains_to_the_original_deadline() {
        let started = Instant::now();
        let output = spawn_sync_helper("hold-pipes", true, 1_000).unwrap();
        assert!(output.status.success());
        assert!(
            started.elapsed() < Duration::from_millis(1_800),
            "inherited pipes outlasted the deadline: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn async_wait_binds_inherited_pipe_drains_to_one_grace_period() {
        let mut handle = spawn_async_helper("hold-pipes", false, true);
        let started = Instant::now();
        let output = tokio::time::timeout(
            Duration::from_secs(1),
            wait_for_child_process_with_output(&mut handle.child),
        )
        .await
        .expect("inherited handles must not hold the wait open")
        .unwrap();
        assert!(output.status.success());
        assert!(started.elapsed() < Duration::from_millis(900));
    }

    struct KillFailure;

    impl SyncChildWait for KillFailure {
        fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
            Ok(None)
        }

        fn kill(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected kill failure",
            ))
        }

        fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
            panic!("a failed kill must not start an unbounded wait")
        }
    }

    #[test]
    fn timeout_propagates_kill_failure_without_waiting() {
        let started = Instant::now();
        let error = wait_with_timeout_inner(&mut KillFailure, 0).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("failed to kill child"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    struct PollFailure;

    impl SyncChildWait for PollFailure {
        fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
            Err(std::io::Error::other("injected status failure"))
        }

        fn kill(&mut self) -> std::io::Result<()> {
            panic!("a status failure must not be converted into a timeout kill")
        }

        fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
            unreachable!()
        }
    }

    #[test]
    fn timeout_preserves_status_errors() {
        let error = wait_with_timeout_inner(&mut PollFailure, 100).unwrap_err();
        assert!(error.to_string().contains("failed to poll child status"));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_and_reaps_a_sleeper_without_a_zombie() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id() as i32;
        assert_eq!(wait_with_timeout(&mut child, 25).unwrap(), None);
        assert!(matches!(child.try_wait(), Ok(Some(_))));
        assert!(!is_process_alive(pid));
        assert!(!is_zombie_process(pid));
    }

    #[tokio::test]
    async fn aborted_wait_kills_and_reaps_a_non_detached_child() {
        let handle = spawn_async_helper("sleep", false, false);
        let pid = handle.child.id().unwrap() as i32;
        let task = tokio::spawn(wait_for_child_process(handle.child));
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        #[cfg(unix)]
        wait_until_reaped(pid);
        #[cfg(not(unix))]
        {
            let deadline = Instant::now() + Duration::from_secs(3);
            while is_process_alive(pid) && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(!is_process_alive(pid));
        }
    }

    #[tokio::test]
    async fn detached_child_survives_owner_drop() {
        let handle = spawn_async_helper("sleep-brief", true, false);
        let pid = handle.child.id().unwrap() as i32;
        drop(handle);
        assert!(is_process_alive(pid));
        let deadline = Instant::now() + Duration::from_secs(4);
        while is_process_alive(pid) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!is_process_alive(pid));
        #[cfg(unix)]
        unsafe {
            // A deliberately detached tokio child has no async reaper. Reap
            // this test-owned process directly after proving it survived drop.
            let mut status = 0;
            assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
        }
    }

    #[cfg(unix)]
    #[test]
    fn exit_codes_use_the_actual_unix_status_signal() {
        use std::os::unix::process::ExitStatusExt;
        for signal in [libc::SIGUSR1, libc::SIGABRT] {
            let status = std::process::ExitStatus::from_raw(signal);
            assert_eq!(status.signal(), Some(signal));
            assert_eq!(exit_code_for_status(&status), Some(128 + signal));
        }
    }


    #[tokio::test]
    async fn cancelled_borrowed_wait_preserves_owner_for_graceful_termination() {
        let mut handle = spawn_async_helper("sleep", false, false);
        let pid = handle.child.id().unwrap() as i32;
        let first = tokio::time::timeout(
            Duration::from_millis(25),
            wait_for_child_process_with_output(&mut handle.child),
        )
        .await;
        assert!(first.is_err(), "the sleeper must trigger the first bound");
        assert!(
            is_process_alive(pid),
            "dropping a borrowed wait must not preempt the graceful signal"
        );

        assert!(signal_process_group_or_process(pid, Signal::Term));
        let output = tokio::time::timeout(
            Duration::from_secs(2),
            wait_for_child_process_with_output(&mut handle.child),
        )
        .await
        .expect("successful termination must be reaped")
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(output.status.signal(), Some(libc::SIGTERM));
            assert_eq!(exit_code_for_status(&output.status), Some(128 + libc::SIGTERM));
        }
        assert!(!is_process_alive(pid));
    }

}
