//! The actual CLI, supervisor and worker must reach the real HTTP provider adapter.
//! All configuration, sockets and sessions are private; the model endpoint is loopback.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use pi_coding_agent::modes::daemon::daemon_client::{DaemonClient, DaemonClientRequestOptions};
use pi_coding_agent::modes::daemon::daemon_protocol::daemon_protocol_info;
use serde_json::{json, Value};

const REPLY: &str = "LOCAL_HTTP_FIXTURE_RESPONSE";

struct LocalModel {
    address: std::net::SocketAddr,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<Result<(), String>>>,
}

impl LocalModel {
    fn start() -> Self {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stopped = stop.clone();
        let received = requests.clone();
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        if !peer.ip().is_loopback() {
                            return Err("Non-loopback model client".into());
                        }
                        // Windows accept() makes the accepted socket inherit the listener's
                        // non-blocking mode (Unix does not). Leaving it non-blocking makes
                        // serve_completion fail with WSAEWOULDBLOCK (os error 10035) whenever the
                        // request bytes have not landed yet, which kills this accept loop and
                        // fails the test intermittently. The fixture is a blocking HTTP server.
                        stream.set_nonblocking(false).unwrap();
                        let body = serve_completion(stream)?;
                        received.lock().unwrap().push(body);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            Ok(())
        });
        Self {
            address,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn finish(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap().unwrap();
        }
    }
}
impl Drop for LocalModel {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn serve_completion(mut stream: TcpStream) -> Result<Value, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
        if bytes.len() >= 64 * 1024 {
            return Err("HTTP headers exceed fixture limit".into());
        }
        read_more(&mut stream, &mut bytes)?;
    };
    let headers = std::str::from_utf8(&bytes[..header_end]).map_err(|error| error.to_string())?;
    if !headers.starts_with("POST /v1/chat/completions HTTP/1.1\r\n") {
        return Err(format!(
            "Unexpected request: {}",
            headers.lines().next().unwrap_or_default()
        ));
    }
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .ok_or_else(|| "Missing HTTP content length".to_string())?;
    if content_length > 1024 * 1024 {
        return Err("HTTP body exceeds fixture limit".into());
    }
    while bytes.len() < header_end + content_length {
        read_more(&mut stream, &mut bytes)?;
    }
    let body: Value = serde_json::from_slice(&bytes[header_end..header_end + content_length])
        .map_err(|error| error.to_string())?;
    if body["model"] != "fixture-model" || body["stream"] != true {
        return Err(format!("Unexpected completion body: {body}"));
    }
    let text = json!({"id":"fixture-1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{"role":"assistant","content":REPLY},"finish_reason":null}]});
    let end = json!({"id":"fixture-1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}});
    let response = format!("data: {text}\n\ndata: {end}\n\ndata: [DONE]\n\n");
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).map_err(|error| error.to_string())?;
    Ok(body)
}

fn read_more(stream: &mut TcpStream, bytes: &mut Vec<u8>) -> Result<(), String> {
    let mut buffer = [0u8; 4096];
    let count = stream
        .read(&mut buffer)
        .map_err(|error| error.to_string())?;
    if count == 0 {
        return Err("HTTP client closed before complete request".into());
    }
    bytes.extend_from_slice(&buffer[..count]);
    Ok(())
}

struct PrivateCli {
    root: tempfile::TempDir,
    agent_dir: PathBuf,
    workspace: PathBuf,
    socket: String,
}
impl PrivateCli {
    fn new(base_url: &str) -> Self {
        // Keep the fixture independent of checkout depth: the daemon also puts
        // worker sockets under this root's tmp directory. Deep checkout paths
        // can exceed sockaddr_un's limit before a session is created.
        let scratch = std::env::temp_dir().canonicalize().unwrap();
        let root = tempfile::Builder::new()
            .prefix("native-cli-")
            .tempdir_in(scratch)
            .unwrap();
        let agent_dir = root.path().join("agent");
        let workspace = root.path().join("workspace");
        for directory in [
            &agent_dir,
            &workspace,
            &root.path().join("home"),
            &root.path().join("sessions"),
            &root.path().join("tmp"),
        ] {
            fs::create_dir_all(directory).unwrap();
        }
        fs::write(agent_dir.join("settings.json"), serde_json::to_vec(&json!({
            "defaultProvider":"local-cli-fixture", "defaultModel":"fixture-model", "defaultThinkingLevel":"off",
            "telemetry":{"enabled":false}, "autoCompaction":{"enabled":false}, "quietStartup":true,
        })).unwrap()).unwrap();
        // This literal is fixture data, never a credential or environment lookup.
        fs::write(agent_dir.join("models.json"), serde_json::to_vec(&json!({"providers":{"local-cli-fixture":{
            "baseUrl":base_url, "apiKey":"local-fixture-not-a-secret", "api":"openai-completions",
            "models":[{"id":"fixture-model","name":"Local fixture","reasoning":false,"input":["text"],"contextWindow":32768,"maxTokens":128,
                "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}]
        }}})).unwrap()).unwrap();
        let socket = if cfg!(windows) {
            format!(r"\\.\pipe\optimus-native-cli-{}", uuid::Uuid::new_v4())
        } else {
            root.path()
                .join("daemon.sock")
                .to_string_lossy()
                .into_owned()
        };
        Self {
            root,
            agent_dir,
            workspace,
            socket,
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_optimus-rust"));
        command
            .env_clear()
            .current_dir(&self.workspace)
            .stdin(Stdio::null())
            .env("HOME", self.root.path().join("home"))
            .env("USERPROFILE", self.root.path().join("home"))
            .env("TMPDIR", self.root.path().join("tmp"))
            .env("TMP", self.root.path().join("tmp"))
            .env("TEMP", self.root.path().join("tmp"))
            .env("XDG_CONFIG_HOME", self.root.path().join("home/config"))
            .env("XDG_CACHE_HOME", self.root.path().join("home/cache"))
            .env("PRIME_AGENT_CODING_AGENT_DIR", &self.agent_dir)
            .env("PRIME_AGENT_SESSION_DIR", self.root.path().join("sessions"))
            .env("PI_OFFLINE", "1")
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("TERM", "dumb");
        // Executable/tool discovery is needed by the real supervisor's worker launcher.
        if let Some(path) = std::env::var_os("PATH") {
            command.env("PATH", path);
        }
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            command.env("SystemRoot", system_root);
        }
        command
    }
    fn spawn(&self, label: &str, args: &[&str]) -> OwnedChild {
        let stdout = self.root.path().join(format!("{label}.stdout"));
        let stderr = self.root.path().join(format!("{label}.stderr"));
        let mut command = self.command();
        command
            .args(args)
            .args(["--daemon-socket", &self.socket])
            .stdout(File::create(&stdout).unwrap())
            .stderr(File::create(&stderr).unwrap());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        OwnedChild {
            child: command.spawn().expect("launch real optimus-rust"),
            stdout,
            stderr,
        }
    }
}

struct OwnedChild {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}
impl OwnedChild {
    fn wait(&mut self, timeout: Duration) -> (ExitStatus, String, String) {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return (
                    status,
                    fs::read_to_string(&self.stdout).unwrap(),
                    fs::read_to_string(&self.stderr).unwrap(),
                );
            }
            assert!(
                Instant::now() < deadline,
                "CLI timed out; stdout: {}; stderr: {}",
                fs::read_to_string(&self.stdout).unwrap(),
                fs::read_to_string(&self.stderr).unwrap()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            #[cfg(unix)]
            unsafe {
                // This process group was created by this test's Command only.
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct OwnedDaemon {
    process: OwnedChild,
    socket: String,
}
impl Drop for OwnedDaemon {
    fn drop(&mut self) {
        if self.process.child.try_wait().ok().flatten().is_some() {
            return;
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let client = DaemonClient::create(&self.socket);
            if client.connect(300).await.is_ok() {
                let _ = client
                    .request(
                        json!({"type":"shutdown"}).as_object().unwrap().clone(),
                        Some(1000),
                        DaemonClientRequestOptions::default(),
                    )
                    .await;
                client.close().await;
            }
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if self.process.child.try_wait().ok().flatten().is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        // OwnedChild supplies the bounded fallback if protocol shutdown failed.
    }
}

fn wait_for_daemon(daemon: &mut OwnedDaemon) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        assert!(
            daemon.process.child.try_wait().unwrap().is_none(),
            "daemon exited: {}",
            fs::read_to_string(&daemon.process.stderr).unwrap()
        );
        let ready = runtime.block_on(async {
            let client = DaemonClient::create(&daemon.socket);
            let result = if client.connect(200).await.is_ok() {
                client.wait_for_hello(200).await.ok()
            } else {
                None
            };
            client.close().await;
            result
        });
        if let Some(hello) = ready {
            assert_eq!(hello.protocol, daemon_protocol_info());
            return;
        }
        assert!(
            Instant::now() < deadline,
            "daemon handshake timed out: {}",
            fs::read_to_string(&daemon.process.stderr).unwrap()
        );
        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn actual_print_cli_streams_local_http_through_the_daemon_worker() {
    let mut model = LocalModel::start();
    let fixture = PrivateCli::new(&format!("http://{}/v1", model.address));
    let mut daemon = OwnedDaemon {
        process: fixture.spawn("daemon", &["--mode", "daemon", "--offline"]),
        socket: fixture.socket.clone(),
    };
    wait_for_daemon(&mut daemon);
    let mut text = fixture.spawn(
        "text",
        &[
            "--print",
            "--offline",
            "--no-tools",
            "--provider",
            "local-cli-fixture",
            "--model",
            "fixture-model",
            "cli-text-prompt",
        ],
    );
    let (status, stdout, stderr) = text.wait(Duration::from_secs(45));
    assert!(
        status.success(),
        "print failed: {status}; stdout: {stdout}; stderr: {stderr}"
    );
    assert_eq!(
        stdout.trim(),
        REPLY,
        "print must expose the real provider response; stderr: {stderr}"
    );
    let mut structured = fixture.spawn(
        "json",
        &[
            "--print",
            "--mode",
            "json",
            "--offline",
            "--no-tools",
            "--provider",
            "local-cli-fixture",
            "--model",
            "fixture-model",
            "cli-json-prompt",
        ],
    );
    let (status, stdout, stderr) = structured.wait(Duration::from_secs(45));
    assert!(
        status.success(),
        "JSON print failed: {status}; stdout: {stdout}; stderr: {stderr}"
    );
    let frames: Vec<Value> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("Non-JSON stdout: {line}: {error}"))
        })
        .collect();
    assert!(
        frames.iter().any(|frame| frame["type"] == "message_end"
            && frame["message"]["role"] == "assistant"
            && frame["message"]["content"].to_string().contains(REPLY)),
        "missing final assistant event: {stdout}; stderr: {stderr}"
    );
    let requests = model.requests.lock().unwrap().clone();
    for prompt in ["cli-text-prompt", "cli-json-prompt"] {
        assert!(
            requests
                .iter()
                .any(|request| request["messages"].to_string().contains(prompt)),
            "CLI prompt never reached the HTTP adapter: {prompt}"
        );
    }
    drop(daemon);
    model.finish();
}

#[test]
fn execution_mode_survives_real_daemon_worker_resume_and_changes_http_requests() {
    let mut model = LocalModel::start();
    let fixture = PrivateCli::new(&format!("http://{}/v1", model.address));
    let mut daemon = OwnedDaemon {
        process: fixture.spawn("mode-daemon", &["--mode", "daemon", "--offline"]),
        socket: fixture.socket.clone(),
    };
    wait_for_daemon(&mut daemon);
    async fn request(client: &Arc<DaemonClient>, body: Value) -> Value {
        let response = client.request(body.as_object().unwrap().clone(), Some(30_000), Default::default()).await.unwrap();
        assert!(response.success, "{:?}", response.error);
        response.data.unwrap_or(Value::Null)
    }
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    runtime.block_on(async {
        let client = DaemonClient::create(&fixture.socket);
        client.connect(1000).await.unwrap();
        assert!(client.wait_for_hello(1000).await.unwrap().supports("execution_mode"));
        let created = request(&client, json!({"type":"create", "config":{
            "cwd":fixture.workspace, "agentDir":fixture.agent_dir, "sessionDir":fixture.root.path().join("sessions"),
            "provider":"local-cli-fixture", "model":"fixture-model", "noSkills":true, "noExtensions":true
        }})).await;
        let active = created.get("activeSessionId").or_else(|| created.get("id")).unwrap().as_str().unwrap();
        for message in ["HTTP_MODE_HISTORY", "/mode direct", "HTTP_DIRECT_REQUEST", "/mode ipython", "HTTP_IPYTHON_REQUEST", "/mode direct"] {
            request(&client, json!({"type":"prompt_and_wait", "activeSessionId":active, "message":message})).await;
        }
        let state = request(&client, json!({"type":"get_connection_state", "activeSessionId":active})).await;
        assert_eq!(state["activeToolNames"], json!(["bash", "edit"]));
        let saved = state["sessionFile"].as_str().unwrap().to_string();
        client.close().await;
        // A new CLI client resumes the durable chat, through the real supervisor.
        let mut resumed = fixture.spawn("mode-resumed", &["--print", "--offline", "--resume", &saved, "HTTP_RESUMED_DIRECT"]);
        let (status, stdout, stderr) = resumed.wait(Duration::from_secs(45));
        assert!(status.success(), "{status}; {stdout}; {stderr}");
    });
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 4, "mode switches must not call a model");
    for (index, direct) in [(0, false), (1, true), (2, false), (3, true)] {
        let request = &requests[index];
        let names: Vec<_> = request["tools"].as_array().unwrap().iter()
            .filter_map(|tool| tool["function"]["name"].as_str()).collect();
        assert_eq!(names.contains(&"ipython"), !direct, "{names:?}");
        assert_eq!(names.contains(&"bash"), direct, "{names:?}");
        assert_eq!(names.contains(&"edit"), direct, "{names:?}");
        let messages = request["messages"].as_array().unwrap();
        let system = messages.iter().filter(|m| m["role"] == "system" || m["role"] == "developer")
            .map(|m| m["content"].to_string()).collect::<String>();
        assert_eq!(system.contains("Execution mode: Direct tools"), direct);
        assert_eq!(system.contains("Python is the orchestration language"), !direct);
        assert!(request["messages"].to_string().contains("HTTP_MODE_HISTORY"));
    }
    drop(requests);
    drop(daemon);
    model.finish();
}
