//! Native process adapter for the legacy owned-worker frontend.
//!
//! The owner channel is an authenticated loopback connection. Both ends keep
//! the connection open for the process lifetime; EOF has the same meaning as
//! Node IPC disconnect. Reader threads do not keep the process/runtime alive.

use std::io::{self, IsTerminal, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use pi_ai::types::BoxFuture;

use super::owned_session_worker::*;
use super::subprocess_launch::ProcessEnv;
use crate::modes::rpc::jsonl::{JsonlLineReader, JsonlLineReaderOptions, StringDecoder};

const OWNER_ADDRESS_ENV: &str = "PRIME_AGENT_INTERNAL_OWNED_OWNER_ADDRESS";
const OWNER_TOKEN_ENV: &str = "PRIME_AGENT_INTERNAL_OWNED_OWNER_TOKEN";
type Callback = Arc<dyn Fn() + Send + Sync>;
type LineCallback = Arc<dyn Fn(String) + Send + Sync>;

pub(crate) fn install_owner_watch() -> Result<(), String> {
    let host = Arc::new(NativeOwnedWorkerHost::new()?);
    install_owned_session_worker_owner_watch(host)
}

pub(crate) async fn maybe_run_frontend(args: Vec<String>) -> Result<Option<i32>, String> {
    // Avoid acquiring stdin or an owner connection when the legacy route is off.
    if std::env::var("PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND").as_deref() != Ok("1") {
        return Ok(None);
    }
    let host = Arc::new(NativeOwnedWorkerHost::frontend());
    if !maybe_run_owned_session_worker_frontend(&args, false, host.clone()).await? {
        return Ok(None);
    }
    if let Some(error) = host
        .error
        .lock()
        .expect("native host error poisoned")
        .take()
    {
        return Err(error);
    }
    Ok(Some(host.exit_code.load(Ordering::SeqCst)))
}

#[derive(Clone)]
enum OutputTarget {
    Lines(LineCallback),
    Stdout,
    Stderr,
    Child(Arc<dyn OwnedWorkerChildStdin>),
}

#[derive(Default)]
struct ReaderState {
    target: Option<OutputTarget>,
    paused: bool,
    ended: bool,
    finished: bool,
    on_end: Vec<Callback>,
    error: Option<String>,
}

/// Each pipe has one reader, and callbacks run without holding its state lock.
struct NativeReader {
    source: Mutex<Option<Box<dyn Read + Send>>>,
    state: Mutex<ReaderState>,
    changed: Condvar,
    done: tokio::sync::Notify,
}

impl NativeReader {
    fn new(source: impl Read + Send + 'static) -> Arc<Self> {
        Arc::new(Self {
            source: Mutex::new(Some(Box::new(source))),
            state: Mutex::new(ReaderState::default()),
            changed: Condvar::new(),
            done: tokio::sync::Notify::new(),
        })
    }

    fn set_target(self: &Arc<Self>, target: OutputTarget) {
        let ended = {
            let mut state = self.state.lock().expect("reader state poisoned");
            state.target = Some(target.clone());
            state.ended
        };
        if ended {
            if let OutputTarget::Child(input) = target {
                input.end();
            }
            return;
        }
        self.changed.notify_all();
        let source = self.source.lock().expect("reader source poisoned").take();
        if let Some(source) = source {
            let reader = self.clone();
            std::thread::spawn(move || reader.read(source));
        }
    }

    fn read(self: Arc<Self>, mut source: Box<dyn Read + Send>) {
        let weak = Arc::downgrade(&self);
        let mut lines = JsonlLineReader::new(
            Arc::new(move |line| {
                let callback = weak.upgrade().and_then(|reader| {
                    let state = reader.state.lock().expect("reader state poisoned");
                    match &state.target {
                        Some(OutputTarget::Lines(callback)) => Some(callback.clone()),
                        _ => None,
                    }
                });
                if let Some(callback) = callback {
                    callback(line);
                }
            }),
            JsonlLineReaderOptions::default(),
        );
        let mut decoder = StringDecoder::new();
        let mut buffer = [0u8; 8192];
        let result = loop {
            {
                let mut state = self.state.lock().expect("reader state poisoned");
                while state.paused || state.target.is_none() {
                    state = self.changed.wait(state).expect("reader state poisoned");
                }
            }
            let length = match source.read(&mut buffer) {
                Ok(0) => break Ok(()),
                Ok(length) => length,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => break Err(error),
            };
            let target = self
                .state
                .lock()
                .expect("reader state poisoned")
                .target
                .clone();
            let write = match target {
                Some(OutputTarget::Lines(_)) => {
                    lines.push(&decoder.write(&buffer[..length]));
                    Ok(())
                }
                Some(OutputTarget::Stdout) => write_stdout_bytes(&buffer[..length]),
                Some(OutputTarget::Stderr) => {
                    let mut stderr = io::stderr().lock();
                    stderr
                        .write_all(&buffer[..length])
                        .and_then(|_| stderr.flush())
                }
                Some(OutputTarget::Child(input)) => {
                    if input.write_bytes(&buffer[..length]) {
                        Ok(())
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "worker stdin closed",
                        ))
                    }
                }
                None => Ok(()),
            };
            if let Err(error) = write {
                break Err(error);
            }
        };
        lines.push(&decoder.end());
        lines.end();
        let (callbacks, target) = {
            let mut state = self.state.lock().expect("reader state poisoned");
            state.ended = true;
            state.error = result.err().map(|error| error.to_string());
            (std::mem::take(&mut state.on_end), state.target.clone())
        };
        if let Some(OutputTarget::Child(input)) = target {
            input.end();
        }
        for callback in callbacks {
            callback();
        }
        self.state.lock().expect("reader state poisoned").finished = true;
        self.done.notify_waiters();
    }

    fn attach(self: &Arc<Self>, callback: LineCallback) -> Callback {
        self.set_target(OutputTarget::Lines(callback.clone()));
        let reader = self.clone();
        Arc::new(move || {
            let mut state = reader.state.lock().expect("reader state poisoned");
            if matches!(&state.target, Some(OutputTarget::Lines(current)) if Arc::ptr_eq(current, &callback))
            {
                state.target = None;
            }
        })
    }

    fn on_end(self: &Arc<Self>, callback: Callback) -> Callback {
        let ended = {
            let mut state = self.state.lock().expect("reader state poisoned");
            if !state.ended {
                state.on_end.push(callback.clone());
            }
            state.ended
        };
        if ended {
            callback();
        }
        let reader = self.clone();
        Arc::new(move || {
            reader
                .state
                .lock()
                .expect("reader state poisoned")
                .on_end
                .retain(|current| !Arc::ptr_eq(current, &callback))
        })
    }

    fn set_paused(&self, paused: bool) {
        self.state.lock().expect("reader state poisoned").paused = paused;
        self.changed.notify_all();
    }

    async fn wait(&self) -> Result<(), String> {
        loop {
            let notified = self.done.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.state.lock().expect("reader state poisoned");
                if state.finished {
                    return state.error.clone().map_or(Ok(()), Err);
                }
            }
            notified.await;
        }
    }
}

struct NativeChildInput(Mutex<Option<ChildStdin>>);

impl OwnedWorkerChildStdin for NativeChildInput {
    fn writable(&self) -> bool {
        self.0.lock().expect("child stdin poisoned").is_some()
    }
    fn write(&self, text: &str) -> bool {
        self.write_bytes(text.as_bytes())
    }
    fn write_bytes(&self, bytes: &[u8]) -> bool {
        let mut input = self.0.lock().expect("child stdin poisoned");
        let written = input
            .as_mut()
            .is_some_and(|input| input.write_all(bytes).is_ok());
        if !written {
            input.take();
        }
        written
    }
    // Writes are synchronous and bounded by the OS pipe; they finish draining
    // before returning. They never create an unbounded queue on the frontend.
    fn once_drain(&self, handler: Callback) {
        handler();
    }
    fn end(&self) {
        self.0.lock().expect("child stdin poisoned").take();
    }
}

struct NativeChildOutput(Arc<NativeReader>);
impl OwnedWorkerChildStdout for NativeChildOutput {
    fn pause(&self) {
        self.0.set_paused(true);
    }
    fn resume(&self) {
        self.0.set_paused(false);
    }
    fn attach_lines(&self, callback: LineCallback) -> Callback {
        self.0.attach(callback)
    }
    fn pipe_to_stdout(&self) {
        self.0.set_target(OutputTarget::Stdout);
    }
}
impl OwnedWorkerChildStderr for NativeChildOutput {
    fn pipe_to_stderr(&self) {
        self.0.set_target(OutputTarget::Stderr);
    }
}

#[derive(Default)]
struct OwnerState {
    stream: Option<TcpStream>,
    disconnected: bool,
    callback: Option<Callback>,
}

#[derive(Default)]
struct OwnerLink(Mutex<OwnerState>);

impl OwnerLink {
    fn adopt(self: &Arc<Self>, stream: TcpStream) -> io::Result<()> {
        let reader = stream.try_clone()?;
        {
            let mut state = self.0.lock().expect("owner state poisoned");
            if state.disconnected {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "owner disconnected",
                ));
            }
            state.stream = Some(stream);
        }
        let link = self.clone();
        std::thread::spawn(move || {
            let mut reader = reader;
            let mut byte = [0];
            loop {
                match reader.read(&mut byte) {
                    Ok(0) => break,
                    Ok(_) => continue,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            link.disconnect();
        });
        Ok(())
    }

    fn connected(&self) -> bool {
        !self.0.lock().expect("owner state poisoned").disconnected
    }

    fn disconnect(&self) {
        let callback = {
            let mut state = self.0.lock().expect("owner state poisoned");
            if state.disconnected {
                return;
            }
            state.disconnected = true;
            if let Some(stream) = state.stream.take() {
                let _ = stream.shutdown(Shutdown::Both);
            }
            state.callback.take()
        };
        if let Some(callback) = callback {
            callback();
        }
    }

    fn once_disconnect(self: &Arc<Self>, callback: Callback) -> Callback {
        let disconnected = {
            let mut state = self.0.lock().expect("owner state poisoned");
            if !state.disconnected {
                state.callback = Some(callback.clone());
            }
            state.disconnected
        };
        if disconnected {
            callback();
        }
        let link = self.clone();
        Arc::new(move || {
            let mut state = link.0.lock().expect("owner state poisoned");
            if state
                .callback
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &callback))
            {
                state.callback = None;
            }
        })
    }
}

fn connect_owner(environment: &ProcessEnv) -> Result<Arc<OwnerLink>, String> {
    let address: SocketAddr = environment
        .get(OWNER_ADDRESS_ENV)
        .ok_or("Owned session worker is missing its owner channel")?
        .parse()
        .map_err(|_| "Invalid owned-worker owner address")?;
    if !address.ip().is_loopback() {
        return Err("Owned-worker owner address must be loopback".into());
    }
    let token = environment
        .get(OWNER_TOKEN_ENV)
        .filter(|token| token.len() == 36)
        .ok_or("Owned session worker is missing its owner token")?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))
        .map_err(|error| format!("Cannot connect to owned-worker owner: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|error| error.to_string())?;
    stream
        .write_all(token.as_bytes())
        .map_err(|error| error.to_string())?;
    let mut acknowledgement = [0];
    stream
        .read_exact(&mut acknowledgement)
        .map_err(|error| format!("Owner handshake failed: {error}"))?;
    if acknowledgement != [1] {
        return Err("Owner handshake rejected".into());
    }
    stream
        .set_read_timeout(None)
        .map_err(|error| error.to_string())?;
    let link = Arc::new(OwnerLink::default());
    link.adopt(stream).map_err(|error| error.to_string())?;
    Ok(link)
}

fn accept_owner(listener: TcpListener, token: String, link: Arc<OwnerLink>) {
    std::thread::spawn(move || {
        while link.connected() {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                    let mut presented = [0u8; 36];
                    if stream.read_exact(&mut presented).is_err() || presented != token.as_bytes() {
                        continue;
                    }
                    if stream.write_all(&[1]).is_err() {
                        continue;
                    }
                    if stream.set_read_timeout(None).is_err() {
                        continue;
                    }
                    if link.adopt(stream).is_err() {
                        link.disconnect();
                    }
                    return;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {
                    link.disconnect();
                    return;
                }
            }
        }
    });
}

struct NativeChild {
    pid: i64,
    process: Arc<Mutex<Child>>,
    input: Option<Arc<NativeChildInput>>,
    output: Option<Arc<NativeChildOutput>>,
    error: Option<Arc<NativeChildOutput>>,
    status: Arc<Mutex<Option<OwnedWorkerExit>>>,
    owner: Arc<OwnerLink>,
}

impl Drop for NativeChild {
    fn drop(&mut self) {
        self.owner.disconnect();
        let mut process = self.process.lock().expect("child process poisoned");
        if matches!(process.try_wait(), Ok(None)) {
            // Also reap on an early frontend error or a cancelled wait future.
            if send_signal(self.pid, "SIGKILL") {
                let _ = process.wait();
            }
        }
    }
}

impl OwnedWorkerChild for NativeChild {
    fn pid(&self) -> Option<i64> {
        Some(self.pid)
    }
    fn stdin(&self) -> Option<Arc<dyn OwnedWorkerChildStdin>> {
        self.input
            .clone()
            .map(|input| input as Arc<dyn OwnedWorkerChildStdin>)
    }
    fn stdout(&self) -> Option<Arc<dyn OwnedWorkerChildStdout>> {
        self.output
            .clone()
            .map(|output| output as Arc<dyn OwnedWorkerChildStdout>)
    }
    fn stderr(&self) -> Option<Arc<dyn OwnedWorkerChildStderr>> {
        self.error
            .clone()
            .map(|error| error as Arc<dyn OwnedWorkerChildStderr>)
    }
    fn exit_code(&self) -> Option<i32> {
        self.status
            .lock()
            .expect("child status poisoned")
            .as_ref()
            .and_then(|exit| exit.code)
    }
    fn signal_code(&self) -> Option<String> {
        self.status
            .lock()
            .expect("child status poisoned")
            .as_ref()
            .and_then(|exit| exit.signal.clone())
    }
    fn kill(&self, signal: &str) {
        let _ = send_signal(self.pid, signal);
    }
    fn wait(&self) -> BoxFuture<Result<OwnedWorkerExit, String>> {
        let process = self.process.clone();
        let status = self.status.clone();
        let output = self.output.clone();
        let error = self.error.clone();
        Box::pin(async move {
            let exit = loop {
                let result = process
                    .lock()
                    .expect("child process poisoned")
                    .try_wait()
                    .map_err(|error| error.to_string())?;
                if let Some(result) = result {
                    #[cfg(unix)]
                    let signal = {
                        use std::os::unix::process::ExitStatusExt;
                        result.signal().map(signal_name)
                    };
                    #[cfg(not(unix))]
                    let signal = None;
                    let exit = OwnedWorkerExit {
                        code: result.code(),
                        signal,
                    };
                    *status.lock().expect("child status poisoned") = Some(exit.clone());
                    break exit;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            };
            // Node's close event follows stdout/stderr EOF, not just process exit.
            if let Some(output) = output {
                output.0.wait().await?;
            }
            if let Some(error) = error {
                error.0.wait().await?;
            }
            Ok(exit)
        })
    }
    fn connected(&self) -> bool {
        self.owner.connected()
    }
    fn disconnect(&self) {
        self.owner.disconnect();
    }
}

struct NativeOwnedWorkerHost {
    stdin: Arc<NativeReader>,
    owner: Option<Arc<OwnerLink>>,
    exit_code: AtomicI32,
    error: Mutex<Option<String>>,
}

impl NativeOwnedWorkerHost {
    fn frontend() -> Self {
        Self {
            stdin: NativeReader::new(io::stdin()),
            owner: None,
            exit_code: AtomicI32::new(0),
            error: Mutex::new(None),
        }
    }
    fn new() -> Result<Self, String> {
        let environment: ProcessEnv = std::env::vars().collect();
        let mut host = Self::frontend();
        if is_owned_session_worker_process(&environment) {
            host.owner = Some(connect_owner(&environment)?);
        }
        Ok(host)
    }
}

impl OwnedSessionWorkerHost for NativeOwnedWorkerHost {
    fn platform(&self) -> String {
        if cfg!(windows) {
            "win32"
        } else {
            std::env::consts::OS
        }
        .into()
    }
    fn stdin_is_tty(&self) -> Option<bool> {
        Some(io::stdin().is_terminal())
    }
    fn cwd(&self) -> String {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
    fn env(&self) -> ProcessEnv {
        std::env::vars().collect()
    }
    fn pid(&self) -> i64 {
        i64::from(std::process::id())
    }
    fn tmp_dir(&self) -> String {
        std::env::temp_dir().to_string_lossy().into_owned()
    }
    fn random_uuid(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }
    fn now_ms(&self) -> f64 {
        chrono::Utc::now().timestamp_millis() as f64
    }
    fn attach_stdin_lines(&self, callback: LineCallback) -> Callback {
        self.stdin.attach(callback)
    }
    fn on_stdin_end(&self, callback: Callback) -> Callback {
        self.stdin.on_end(callback)
    }
    fn pause_stdin(&self) {
        self.stdin.set_paused(true);
    }
    fn resume_stdin(&self) {
        self.stdin.set_paused(false);
    }
    fn pipe_stdin(&self, input: &Arc<dyn OwnedWorkerChildStdin>) {
        self.stdin.set_target(OutputTarget::Child(input.clone()));
    }
    fn unpipe_stdin(&self, input: &Arc<dyn OwnedWorkerChildStdin>) {
        let mut state = self.stdin.state.lock().expect("stdin state poisoned");
        if matches!(&state.target, Some(OutputTarget::Child(current)) if Arc::ptr_eq(current, input))
        {
            state.target = None;
        }
    }
    fn write_stdout(&self, text: &str) -> bool {
        if let Err(error) = write_stdout_bytes(text.as_bytes()) {
            *self.error.lock().expect("native host error poisoned") = Some(error.to_string());
            return false;
        }
        true
    }
    fn on_stdout_drain(&self, callback: Callback) {
        callback();
    }
    fn write_stderr(&self, text: &str) {
        let _ = io::stderr().write_all(text.as_bytes());
    }
    fn on_signal(&self, signal: &str, callback: Callback) -> Callback {
        match subscribe_signal(signal, callback) {
            Ok(cleanup) => cleanup,
            Err(error) => {
                *self.error.lock().expect("native host error poisoned") = Some(error);
                Arc::new(|| {})
            }
        }
    }
    fn spawn_worker(
        &self,
        launch: &OwnedWorkerLaunchSpec,
        options: OwnedWorkerSpawnOptions,
    ) -> Result<Arc<dyn OwnedWorkerChild>, String> {
        if let Some(error) = self
            .error
            .lock()
            .expect("native host error poisoned")
            .clone()
        {
            return Err(error);
        }
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| error.to_string())?;
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        let token = uuid::Uuid::new_v4().to_string();
        let mut command = Command::new(&launch.command);
        command
            .args(&launch.args)
            .current_dir(&options.cwd)
            .env_clear()
            .envs(&options.env)
            .env(OWNER_ADDRESS_ENV, address.to_string())
            .env(OWNER_TOKEN_ENV, &token)
            .stdin(if options.pipe_stdin && !options.inherit_stdio {
                Stdio::piped()
            } else {
                Stdio::inherit()
            })
            .stdout(if options.pipe_stdout && !options.inherit_stdio {
                Stdio::piped()
            } else {
                Stdio::inherit()
            })
            .stderr(if options.pipe_stderr && !options.inherit_stdio {
                Stdio::piped()
            } else {
                Stdio::inherit()
            });
        #[cfg(unix)]
        if options.detached {
            use std::os::unix::process::CommandExt;
            // setsid is async-signal-safe; no allocations or locks in pre_exec.
            unsafe {
                command.pre_exec(|| {
                    if libc::setsid() < 0 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW, like spawnHidden.
        }
        let mut process = command
            .spawn()
            .map_err(|error| format!("Cannot spawn owned worker: {error}"))?;
        let owner = Arc::new(OwnerLink::default());
        accept_owner(listener, token, owner.clone());
        Ok(Arc::new(NativeChild {
            pid: i64::from(process.id()),
            input: process
                .stdin
                .take()
                .map(|input| Arc::new(NativeChildInput(Mutex::new(Some(input))))),
            output: process
                .stdout
                .take()
                .map(|output| Arc::new(NativeChildOutput(NativeReader::new(output)))),
            error: process
                .stderr
                .take()
                .map(|error| Arc::new(NativeChildOutput(NativeReader::new(error)))),
            process: Arc::new(Mutex::new(process)),
            status: Arc::new(Mutex::new(None)),
            owner,
        }))
    }
    fn kill_process_group(&self, pid: i64, signal: &str) -> bool {
        #[cfg(unix)]
        {
            pid > 0 && send_process_group_signal(-pid, signal)
        }
        #[cfg(not(unix))]
        {
            let _ = (pid, signal);
            false
        }
    }
    fn schedule_unref_timeout(&self, ms: u64, callback: Callback) {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            callback();
        });
    }
    fn exit(&self, code: i32) {
        std::process::exit(code);
    }
    fn set_exit_code(&self, code: i32) {
        self.exit_code.store(code, Ordering::SeqCst);
    }
    fn kill_self(&self, signal: &str) {
        let _ = send_signal(self.pid(), signal);
    }
    fn has_owner_channel(&self) -> bool {
        self.owner.is_some()
    }
    // Unlike a Tokio task awaiting stdin, the detached owner reader does not
    // prevent runtime shutdown. No event-loop reference needs to be removed.
    fn unref_owner_channel(&self) {}
    fn once_owner_disconnect(&self, callback: Callback) -> Callback {
        match &self.owner {
            Some(owner) => owner.once_disconnect(callback),
            None => Arc::new(|| {}),
        }
    }
    fn owner_channel_connected(&self) -> bool {
        self.owner.as_ref().is_some_and(|owner| owner.connected())
    }
    fn disconnect_owner_channel(&self) {
        if let Some(owner) = &self.owner {
            owner.disconnect();
        }
    }
}

fn write_stdout_bytes(bytes: &[u8]) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(bytes).and_then(|_| stdout.flush())
}

#[cfg(unix)]
fn signal_number(signal: &str) -> Option<i32> {
    match signal {
        "SIGINT" => Some(libc::SIGINT),
        "SIGTERM" => Some(libc::SIGTERM),
        "SIGHUP" => Some(libc::SIGHUP),
        "SIGKILL" => Some(libc::SIGKILL),
        _ => None,
    }
}

#[cfg(unix)]
fn signal_name(signal: i32) -> String {
    match signal {
        libc::SIGINT => "SIGINT".into(),
        libc::SIGTERM => "SIGTERM".into(),
        libc::SIGHUP => "SIGHUP".into(),
        libc::SIGKILL => "SIGKILL".into(),
        _ => format!("SIG{signal}"),
    }
}

#[cfg(unix)]
fn process_group_signal_succeeded(result: i32, error: Option<i32>) -> bool {
    result == 0 || (result == -1 && error == Some(libc::ESRCH))
}

#[cfg(unix)]
fn send_process_group_signal(pid: i64, signal: &str) -> bool {
    let (Ok(pid), Some(signal)) = (i32::try_from(pid), signal_number(signal)) else {
        return false;
    };
    // This adapter signals only a specific process group, never every permitted process.
    if pid >= -1 {
        return false;
    }
    let result = unsafe { libc::kill(pid, signal) };
    let error = if result == 0 {
        None
    } else {
        io::Error::last_os_error().raw_os_error()
    };
    // ESRCH proves the complete group is already gone; other errors remain retryable.
    process_group_signal_succeeded(result, error)
}

fn send_signal(pid: i64, signal: &str) -> bool {
    #[cfg(unix)]
    {
        let (Ok(pid), Some(signal)) = (i32::try_from(pid), signal_number(signal)) else {
            return false;
        };
        if pid == 0 || pid == -1 {
            return false;
        }
        unsafe { libc::kill(pid, signal) == 0 }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, TerminateProcess, PROCESS_TERMINATE,
        };
        if !matches!(signal, "SIGTERM" | "SIGINT" | "SIGKILL") {
            return false;
        }
        let Ok(pid) = u32::try_from(pid) else {
            return false;
        };
        unsafe {
            let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
            if handle.is_null() {
                return false;
            }
            let terminated = TerminateProcess(handle, 1) != 0;
            CloseHandle(handle);
            terminated
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (pid, signal);
        false
    }
}

fn subscribe_signal(signal: &str, callback: Callback) -> Result<Callback, String> {
    #[cfg(unix)]
    let task = {
        let number = signal_number(signal).ok_or_else(|| format!("Unsupported signal {signal}"))?;
        let mut stream =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::from_raw(number))
                .map_err(|error| error.to_string())?;
        tokio::spawn(async move {
            while stream.recv().await.is_some() {
                callback();
            }
        })
    };
    #[cfg(windows)]
    let task = match signal {
        "SIGINT" => {
            let mut stream = tokio::signal::windows::ctrl_c().map_err(|error| error.to_string())?;
            tokio::spawn(async move {
                while stream.recv().await.is_some() {
                    callback();
                }
            })
        }
        "SIGTERM" => {
            let mut stream =
                tokio::signal::windows::ctrl_close().map_err(|error| error.to_string())?;
            tokio::spawn(async move {
                while stream.recv().await.is_some() {
                    callback();
                }
            })
        }
        _ => return Err(format!("Unsupported signal {signal}")),
    };
    #[cfg(not(any(unix, windows)))]
    return Err(format!(
        "Owned-worker signals unsupported on {}: {signal}",
        std::env::consts::OS
    ));
    #[cfg(any(unix, windows))]
    Ok(Arc::new(move || task.abort()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn process_group_esrch_is_success_but_other_failures_remain_retryable() {
        assert!(process_group_signal_succeeded(0, None));
        assert!(process_group_signal_succeeded(-1, Some(libc::ESRCH)));
        assert!(!process_group_signal_succeeded(-1, Some(libc::EPERM)));
        assert!(!process_group_signal_succeeded(-1, None));
    }

    #[test]
    fn explicit_worker_requires_a_real_owner_channel() {
        assert!(connect_owner(&ProcessEnv::new())
            .err()
            .unwrap()
            .contains("missing its owner channel"));
        let mut environment = ProcessEnv::new();
        environment.insert(OWNER_ADDRESS_ENV.into(), "192.0.2.1:4321".into());
        environment.insert(OWNER_TOKEN_ENV.into(), uuid::Uuid::new_v4().to_string());
        assert!(connect_owner(&environment)
            .err()
            .unwrap()
            .contains("must be loopback"));
    }

    #[test]
    fn authenticated_owner_connection_observes_disconnect() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut environment = ProcessEnv::new();
        environment.insert(
            OWNER_ADDRESS_ENV.into(),
            listener.local_addr().unwrap().to_string(),
        );
        let token = uuid::Uuid::new_v4().to_string();
        environment.insert(OWNER_TOKEN_ENV.into(), token.clone());
        let owner = Arc::new(OwnerLink::default());
        accept_owner(listener, token, owner.clone());
        let child = connect_owner(&environment).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let _detach = child.once_disconnect(Arc::new(move || {
            sender.send(()).unwrap();
        }));
        owner.disconnect();
        receiver.recv_timeout(Duration::from_secs(3)).unwrap();
        assert!(!child.connected());
    }

    #[tokio::test]
    async fn raw_stdin_pipe_preserves_bytes_and_ends_the_child_input() {
        #[derive(Default)]
        struct Capture {
            bytes: Mutex<Vec<u8>>,
            ended: std::sync::atomic::AtomicBool,
        }
        impl OwnedWorkerChildStdin for Capture {
            fn writable(&self) -> bool {
                !self.ended.load(Ordering::SeqCst)
            }
            fn write(&self, text: &str) -> bool {
                self.write_bytes(text.as_bytes())
            }
            fn write_bytes(&self, bytes: &[u8]) -> bool {
                self.bytes.lock().unwrap().extend_from_slice(bytes);
                true
            }
            fn once_drain(&self, callback: Callback) {
                callback();
            }
            fn end(&self) {
                self.ended.store(true, Ordering::SeqCst);
            }
        }
        let bytes = vec![0, 0xff, 0xe2, 0x82, 0xac, b'\n'];
        let reader = NativeReader::new(io::Cursor::new(bytes.clone()));
        let capture = Arc::new(Capture::default());
        reader.set_target(OutputTarget::Child(capture.clone()));
        tokio::time::timeout(Duration::from_secs(3), reader.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*capture.bytes.lock().unwrap(), bytes);
        assert!(capture.ended.load(Ordering::SeqCst));
    }

    #[cfg(unix)]
    fn child_options(pipe_output: bool) -> OwnedWorkerSpawnOptions {
        OwnedWorkerSpawnOptions {
            cwd: std::env::current_dir()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            detached: true,
            env: ProcessEnv::new(),
            inherit_stdio: false,
            pipe_stdin: true,
            pipe_stdout: pipe_output,
            pipe_stderr: pipe_output,
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_child_exit_waits_for_all_output_and_trailing_line() {
        let host = NativeOwnedWorkerHost::frontend();
        let child = host
            .spawn_worker(
                &OwnedWorkerLaunchSpec {
                    command: "/bin/sh".into(),
                    args: vec!["-c".into(), "printf '€\\ntrailing'; exit 7".into()],
                },
                child_options(true),
            )
            .unwrap();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = lines.clone();
        let _detach = child
            .stdout()
            .unwrap()
            .attach_lines(Arc::new(move |line| sink.lock().unwrap().push(line)));
        child.stderr().unwrap().pipe_to_stderr();
        child.stdin().unwrap().end();
        let exit = tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .unwrap()
            .unwrap();
        child.disconnect();
        assert_eq!(
            exit,
            OwnedWorkerExit {
                code: Some(7),
                signal: None
            }
        );
        assert_eq!(*lines.lock().unwrap(), vec!["€", "trailing"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_child_receives_forwarded_termination_signal() {
        let host = NativeOwnedWorkerHost::frontend();
        let child = host
            .spawn_worker(
                &OwnedWorkerLaunchSpec {
                    command: "/bin/sleep".into(),
                    args: vec!["30".into()],
                },
                child_options(false),
            )
            .unwrap();
        child.kill("SIGTERM");
        let exit = tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .unwrap()
            .unwrap();
        child.disconnect();
        assert_eq!(exit.code, None);
        assert_eq!(exit.signal.as_deref(), Some("SIGTERM"));
    }
}
