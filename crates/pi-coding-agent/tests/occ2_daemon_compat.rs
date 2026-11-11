//! Daemon-owned OCC session preferences: compatibility and end-to-end tests.
//!
//! Offline scenarios (no daemon process): the schema/capability contract that
//! separates v0.1.34+ daemons from v0.1.33 peers, and the preference
//! resolution/inheritance semantics of `core::occ::session_settings`.
//!
//! Live scenarios: a real `optimus-rust` daemon behind a named pipe, driven by
//! `DaemonClient`, proving the daemon persists and reads the per-chat toggle
//! itself (foreign agentDir, relative paths and remote clients can no longer
//! write a preference the daemon never reads), that independent chats stay
//! isolated, that corrupt or oversized storage surfaces as a failure, and that
//! the preference survives a daemon restart. See reports/occ2opt/DAEMON-COMPAT.md
//! for the failure modes these tests pin.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use pi_coding_agent::core::occ::session_settings;
use pi_coding_agent::modes::daemon::daemon_client::{DaemonClient, DaemonClientRequestOptions};
use pi_coding_agent::modes::daemon::daemon_mode::DAEMON_COMMAND_TYPES;
use pi_coding_agent::modes::daemon::daemon_protocol::{
    daemon_command_compatibility, daemon_command_plane, daemon_protocol_info, DaemonCommandCompatibility,
    DaemonServerCapability, DAEMON_DEFAULT_SERVER_CAPABILITIES, DAEMON_SCHEMA_ID,
    DAEMON_SCHEMA_REVISION, READ_ONLY_DAEMON_COMMANDS,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Offline: protocol contract
// ---------------------------------------------------------------------------

#[test]
fn current_revision_advertises_daemon_owned_occ_session_preferences() {
    assert!(DAEMON_SCHEMA_REVISION >= 41, "daemon-owned OCC preferences require revision 41 or later");
    assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::OccSessionPreferences));
    // The pre-41 capability described global-settings storage semantics, so it
    // is no longer advertised; the variant stays for parsing old hellos.
    assert!(!DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::Occ2Communication));
    for command in ["occ_get_preferences", "occ_set_preferences"] {
        assert_eq!(
            daemon_command_compatibility(command),
            DaemonCommandCompatibility::capability(DaemonServerCapability::OccSessionPreferences)
        );
        assert_eq!(daemon_command_plane(command), Some("session"));
        // Registered commands: an old daemon that does not know them is the
        // only failure mode left, and it answers "Unknown daemon command".
        assert!(DAEMON_COMMAND_TYPES.contains(&command));
    }
    assert!(READ_ONLY_DAEMON_COMMANDS.contains(&"occ_get_preferences"));
    assert!(!READ_ONLY_DAEMON_COMMANDS.contains(&"occ_set_preferences"));
    // The worker list is not enough: the supervisor fronts every client
    // connection and refuses unlisted commands with "Unknown daemon command"
    // before the worker that implements them is ever reached. Both lists must
    // carry the OCC commands or the live daemon path is dead on arrival.
    use pi_coding_agent::modes::daemon::daemon_supervisor;
    for command in ["occ_get_preferences", "occ_set_preferences"] {
        assert!(
            daemon_supervisor::DAEMON_COMMAND_TYPES.contains(&command),
            "the supervisor must forward {command}"
        );
    }
}

#[test]
fn hello_frames_distinguish_the_storage_semantics_across_versions() {
    use pi_coding_agent::modes::daemon::daemon_client::DaemonHello;
    // A v0.1.33 daemon: OCC-2 encoding support, but no session-preference
    // commands. A new client must route /occ to the legacy local path and
    // never claim silent success.
    let legacy = DaemonHello::from_value(&json!({
        "type": "daemon_hello",
        "socketPath": "/tmp/legacy.sock",
        "protocol": daemon_protocol_info(),
        "schemaId": "protocol-7-schema-40-occ2-communication",
        "schemaRevision": 40,
        "appVersion": "0.1.33",
        "serverCapabilities": ["occ2_communication", "execution_mode"],
    }))
    .expect("legacy daemon_hello frame parses");
    assert_eq!(legacy.schema_revision, Some(40));
    assert!(legacy.supports("occ2_communication"));
    assert!(!legacy.supports("occ_session_preferences"));
    // A v0.1.34+ daemon: the daemon owns the preference get/set and no longer
    // advertises the pre-41 capability, so an old client warns instead of
    // writing stale global settings.
    let current = DaemonHello::from_value(&json!({
        "type": "daemon_hello",
        "socketPath": "/tmp/current.sock",
        "protocol": daemon_protocol_info(),
        "schemaId": DAEMON_SCHEMA_ID,
        "schemaRevision": DAEMON_SCHEMA_REVISION,
        "appVersion": "0.1.34",
        "serverCapabilities": DAEMON_DEFAULT_SERVER_CAPABILITIES
            .iter()
            .map(|capability| serde_json::to_value(capability).unwrap().as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
    }))
    .expect("current daemon_hello frame parses");
    assert_eq!(current.schema_revision, Some(DAEMON_SCHEMA_REVISION));
    assert_eq!(current.raw.get("schemaId").and_then(Value::as_str), Some(DAEMON_SCHEMA_ID));
    assert!(current.supports("occ_session_preferences"));
    assert!(!current.supports("occ2_communication"));
}


#[test]
fn occ31_schema47_is_optional_and_keeps_old_startup_and_occ2_contracts() {
    use pi_coding_agent::modes::daemon::daemon_client::DaemonHello;
    use pi_coding_agent::modes::daemon::daemon_protocol::{
        DaemonCompatibilityHello, daemon_outbound_compatibility, meets_daemon_command_compatibility,
    };
    use pi_coding_agent::modes::daemon::daemon_supervisor;
    assert_eq!(DAEMON_SCHEMA_REVISION, 47);
    assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::Occ31Textmas));
    let current = DaemonCompatibilityHello {
        protocol: daemon_protocol_info(), schema_revision: Some(47),
        server_capabilities: Some(DAEMON_DEFAULT_SERVER_CAPABILITIES.into_iter().chain([DaemonServerCapability::AgentRoster]).collect()),
    };
    let mut old = current.clone();
    old.schema_revision = Some(46);
    old.server_capabilities.as_mut().unwrap().retain(|cap| *cap != DaemonServerCapability::Occ31Textmas);
    for command in ["occ31_get_preferences", "occ31_set_mode", "occ32_set_enabled"] {
        assert_eq!(daemon_command_compatibility(command), DaemonCommandCompatibility::gated(47, DaemonServerCapability::Occ31Textmas));
        assert_eq!(daemon_command_plane(command), Some("session"));
        assert!(DAEMON_COMMAND_TYPES.contains(&command));
        assert!(daemon_supervisor::DAEMON_COMMAND_TYPES.contains(&command));
        assert!(!meets_daemon_command_compatibility(&old, &daemon_command_compatibility(command)));
        assert!(meets_daemon_command_compatibility(&current, &daemon_command_compatibility(command)));
    }
    assert!(READ_ONLY_DAEMON_COMMANDS.contains(&"occ31_get_preferences"));
    assert!(!READ_ONLY_DAEMON_COMMANDS.contains(&"occ31_set_mode"));
    assert!(!READ_ONLY_DAEMON_COMMANDS.contains(&"occ32_set_enabled"));
    for command in ["create", "attach", "reattach", "list", "get_state", "roster_subscribe", "occ_get_preferences", "occ_set_preferences"] {
        assert!(meets_daemon_command_compatibility(&old, &daemon_command_compatibility(command)));
        assert!(meets_daemon_command_compatibility(&current, &daemon_command_compatibility(command)));
    }
    for event in ["daemon_hello", "response", "session_attached", "session_resynced", "roster_update"] {
        assert!(meets_daemon_command_compatibility(&old, &daemon_outbound_compatibility(event)));
        assert!(meets_daemon_command_compatibility(&current, &daemon_outbound_compatibility(event)));
    }
    // Both gates matter: even a mistakenly advertised capability on schema46 is unavailable.
    for revision in [None, Some(46), Some(47)] {
        let hello = DaemonHello::from_value(&json!({"type":"daemon_hello", "protocol":daemon_protocol_info(),
            "schemaRevision":revision,"serverCapabilities":["occ31_textmas","occ_session_preferences"]})).unwrap();
        assert_eq!(hello.supports("occ31_textmas"), revision == Some(47));
        assert!(hello.supports("occ_session_preferences"));
    }
    // Old clients use string capabilities and ignore added optional greeting fields.
    #[derive(serde::Deserialize)]
    #[serde(rename_all="camelCase")]
    struct LegacyHello { protocol: pi_coding_agent::modes::daemon::daemon_protocol::DaemonProtocolInfo, server_capabilities: Vec<String> }
    let old_reader: LegacyHello = serde_json::from_value(json!({
        "protocol":daemon_protocol_info(), "schemaRevision":47, "serverCapabilities":DAEMON_DEFAULT_SERVER_CAPABILITIES.as_slice(),
    })).unwrap();
    assert_eq!(old_reader.protocol.version, 7);
    assert!(old_reader.server_capabilities.iter().any(|cap| cap == "occ_session_preferences"));
}

// ---------------------------------------------------------------------------
// Offline: preference resolution semantics
// ---------------------------------------------------------------------------

fn write_session(sessions_dir: &Path, id: &str, parent: Option<&str>, depth: u64) -> String {
    fs::create_dir_all(sessions_dir).unwrap();
    let path = sessions_dir.join(format!("{id}.jsonl"));
    fs::write(
        &path,
        format!(
            "{}\n",
            json!({"type":"session","id":id,"rlmDepth":depth,"parentSession":parent})
        ),
    )
    .unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn preference_resolution_isolates_chats_and_inherits_down_subagent_chains() {
    let root = tempfile::tempdir().unwrap();
    let sessions = root.path().join("sessions");
    let a = write_session(&sessions, "chat-a", None, 0);
    let b = write_session(&sessions, "chat-b", None, 0);
    let child = write_session(&sessions, "child", Some("chat-a.jsonl"), 1);
    let nested = write_session(&sessions, "nested", Some("child.jsonl"), 2);
    let fork = write_session(&sessions, "fork", Some("chat-a.jsonl"), 0);

    // Defaults: on with no source.
    assert_eq!(
        session_settings::preferences(&a).unwrap(),
        json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": null})
    );
    // The owning chat is not inherited; other chats and forks stay isolated.
    session_settings::set_preferences(&a, true).unwrap();
    assert_eq!(
        session_settings::preferences(&a).unwrap(),
        json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "chat-a"})
    );
    assert_eq!(session_settings::preferences(&b).unwrap()["occ2Enabled"], json!(true));
    assert_eq!(session_settings::preferences(&fork).unwrap()["occ2Enabled"], json!(true));
    // Subagents inherit the nearest explicit preference, dynamically.
    assert_eq!(
        session_settings::preferences(&child).unwrap(),
        json!({"occ2Enabled": true, "inherited": true, "sourceSessionId": "chat-a"})
    );
    assert_eq!(
        session_settings::preferences(&nested).unwrap()["sourceSessionId"],
        json!("chat-a")
    );
    // A child override wins for its own subtree and stops there.
    session_settings::set_preferences(&child, false).unwrap();
    assert_eq!(session_settings::preferences(&child).unwrap()["occ2Enabled"], json!(false));
    assert_eq!(session_settings::preferences(&nested).unwrap()["occ2Enabled"], json!(false));
    assert_eq!(session_settings::preferences(&a).unwrap()["occ2Enabled"], json!(true));
    // The set response is the storage's own echo, never the request blindly.
    let set = session_settings::set_preferences(&b, true).unwrap();
    assert_eq!(set["occ2Enabled"], json!(true));
    assert_eq!(set["applied"], json!(true));
    assert_eq!(set["scope"], json!("this chat"));
    assert_eq!(set["effectiveForNewMessages"], json!(true));
    assert_eq!(set["sourceSessionId"], json!("chat-b"));
}

// ---------------------------------------------------------------------------
// Live daemon harness (same shape as tests/native_cli.rs)
// ---------------------------------------------------------------------------

const REPLY: &str = "OCC_FIXTURE_RESPONSE";

fn serve_completion(mut stream: TcpStream) -> Result<Value, String> {
    stream.set_read_timeout(Some(Duration::from_secs(3))).map_err(|error| error.to_string())?;
    stream.set_write_timeout(Some(Duration::from_secs(3))).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
        if bytes.len() >= 64 * 1024 {
            return Err("HTTP headers exceed fixture limit".into());
        }
        let mut buffer = [0u8; 4096];
        let count = stream.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("HTTP client closed before complete request".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    };
    let headers = std::str::from_utf8(&bytes[..header_end]).map_err(|error| error.to_string())?;
    if !headers.starts_with("POST /v1/chat/completions HTTP/1.1\r\n") {
        return Err(format!("Unexpected request: {}", headers.lines().next().unwrap_or_default()));
    }
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .ok_or_else(|| "Missing HTTP content length".to_string())?;
    while bytes.len() < header_end + content_length {
        let mut buffer = [0u8; 4096];
        let count = stream.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("HTTP client closed before complete request".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let body: Value = serde_json::from_slice(&bytes[header_end..header_end + content_length])
        .map_err(|error| error.to_string())?;
    let text = json!({"id":"occ-fixture-1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{"role":"assistant","content":REPLY},"finish_reason":null}]});
    let end = json!({"id":"occ-fixture-1","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}});
    let response = format!("data: {text}\n\ndata: {end}\n\ndata: [DONE]\n\n");
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len())
        .map_err(|error| error.to_string())?;
    Ok(body)
}

struct LocalModel {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl LocalModel {
    fn start() -> Self {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        // Windows accept() inherits the listener's non-blocking
                        // mode; the fixture server is blocking.
                        stream.set_nonblocking(false).unwrap();
                        let _ = serve_completion(stream);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(_) => return,
                }
            }
        });
        Self { stop, worker: Some(worker) }
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

struct PrivateDaemon {
    root: tempfile::TempDir,
    agent_dir: PathBuf,
    workspace: PathBuf,
    socket: String,
}

impl PrivateDaemon {
    fn new() -> Self {
        let scratch = std::env::temp_dir().canonicalize().unwrap();
        let root = tempfile::Builder::new()
            .prefix("occ-daemon-")
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
        // A loopback model endpoint keeps the fixture self-contained; the OCC
        // tests never prompt, so no completion is ever requested.
        let sink = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base_url = format!("http://{}/v1", sink.local_addr().unwrap());
        drop(sink);
        fs::write(agent_dir.join("settings.json"), serde_json::to_vec(&json!({
            "defaultProvider":"local-cli-fixture", "defaultModel":"fixture-model", "defaultThinkingLevel":"off",
            "telemetry":{"enabled":false}, "autoCompaction":{"enabled":false}, "quietStartup":true,
        })).unwrap()).unwrap();
        fs::write(agent_dir.join("models.json"), serde_json::to_vec(&json!({"providers":{"local-cli-fixture":{
            "baseUrl":base_url, "apiKey":"local-fixture-not-a-secret", "api":"openai-completions",
            "models":[{"id":"fixture-model","name":"Local fixture","reasoning":false,"input":["text"],"contextWindow":32768,"maxTokens":128,
                "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}]
        }}})).unwrap()).unwrap();
        let socket = if cfg!(windows) {
            format!(r"\\.\pipe\optimus-occ-daemon-{}", uuid::Uuid::new_v4())
        } else {
            root.path().join("daemon.sock").to_string_lossy().into_owned()
        };
        Self { root, agent_dir, workspace, socket }
    }

    fn command(&self) -> Command {
        let binary = std::env::var_os("OPTIMUS_TEST_BINARY")
            .unwrap_or_else(|| env!("CARGO_BIN_EXE_optimus-rust").into());
        let mut command = Command::new(binary);
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
        if let Some(path) = std::env::var_os("PATH") {
            command.env("PATH", path);
        }
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            command.env("SystemRoot", system_root);
        }
        command
    }

    fn spawn(&self, label: &str) -> OwnedChild {
        let stdout = self.root.path().join(format!("{label}.stdout"));
        let stderr = self.root.path().join(format!("{label}.stderr"));
        let mut command = self.command();
        command
            .args(["--mode", "daemon", "--offline"])
            .args(["--daemon-socket", &self.socket])
            .stdout(File::create(&stdout).unwrap())
            .stderr(File::create(&stderr).unwrap());
        OwnedChild { child: command.spawn().expect("launch real optimus-rust"), stdout, stderr }
    }
}

struct OwnedChild {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl OwnedChild {
    fn stderr(&self) -> String {
        fs::read_to_string(&self.stderr).unwrap()
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
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
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
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
            daemon.process.stderr()
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
            daemon.process.stderr()
        );
        thread::sleep(Duration::from_millis(25));
    }
}

async fn request(client: &Arc<DaemonClient>, body: Value) -> Value {
    let response = client
        .request(body.as_object().unwrap().clone(), Some(30_000), DaemonClientRequestOptions::default())
        .await
        .unwrap();
    assert!(response.success, "{:?}", response.error);
    response.data.unwrap_or(Value::Null)
}

async fn request_result(client: &Arc<DaemonClient>, body: Value) -> Result<Value, String> {
    let response = client
        .request(body.as_object().unwrap().clone(), Some(30_000), DaemonClientRequestOptions::default())
        .await
        .unwrap();
    if response.success {
        Ok(response.data.unwrap_or(Value::Null))
    } else {
        Err(response.error.unwrap_or_else(|| "Daemon request failed".to_string()))
    }
}

async fn create_session(client: &Arc<DaemonClient>, fixture: &PrivateDaemon, session_path: Option<&str>) -> String {
    let mut body = json!({"type":"create", "config":{
        "cwd":fixture.workspace, "agentDir":fixture.agent_dir, "sessionDir":fixture.root.path().join("sessions"),
        "provider":"local-cli-fixture", "model":"fixture-model", "noSkills":true, "noExtensions":true
    }});
    if let Some(session_path) = session_path {
        body["sessionPath"] = json!(session_path);
    }
    let created = request(client, body).await;
    created
        .get("activeSessionId")
        .or_else(|| created.get("id"))
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

fn artifact_preference_path(session_file: &str) -> PathBuf {
    // dirname(dirname(file))/session-artifacts/<id>/occ.json, the daemon's own
    // derivation (core/occ/session_settings.rs::preference_path).
    Path::new(session_file)
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join("session-artifacts")
        .join(Path::new(session_file).file_stem().unwrap())
        .join("occ.json")
}

// ---------------------------------------------------------------------------
// Live daemon: the daemon owns the preference
// ---------------------------------------------------------------------------

// Plain #[test] + an explicit runtime, matching tests/native_cli.rs: the
// daemon handles (OwnedDaemon::drop) block on their own current-thread
// runtime, which aborts the process ("cannot start a runtime from within a
// runtime") if they drop inside a #[tokio::test] runtime.
#[test]
fn daemon_owns_occ_preferences_across_chats_failures_and_restart() {
    let _model = LocalModel::start();
    let fixture = PrivateDaemon::new();
    let mut daemon = OwnedDaemon { process: fixture.spawn("occ-daemon"), socket: fixture.socket.clone() };
    wait_for_daemon(&mut daemon);

    let first_file = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
    let client = Arc::new(DaemonClient::create(&fixture.socket));
    client.connect(1000).await.unwrap();
    let hello = client.wait_for_hello(1000).await.unwrap();
    assert_eq!(hello.schema_revision, Some(DAEMON_SCHEMA_REVISION), "the shipped daemon must advertise the current revision");
    assert_eq!(hello.raw.get("schemaId").and_then(Value::as_str), Some(DAEMON_SCHEMA_ID));
    assert!(hello.supports("occ_session_preferences"));
    assert!(!hello.supports("occ2_communication"), "the pre-41 capability must not be advertised");

    // A chat starts with OCC-2 enabled and no preference recorded.
    let first = create_session(&client, &fixture, None).await;
    let first_state = request(&client, json!({"type":"get_connection_state", "activeSessionId":first})).await;
    let first_file = first_state["sessionFile"].as_str().unwrap().to_string();
    assert!(!first_file.is_empty());
    assert_eq!(
        request(&client, json!({"type":"occ_get_preferences", "activeSessionId":first})).await,
        json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": null})
    );

    // Setting the toggle is daemon-owned: the write lands in the daemon's own
    // session artifacts, never in the client process or global settings.
    let set = request(&client, json!({"type":"occ_set_preferences", "activeSessionId":first, "occ2Enabled":true})).await;
    assert_eq!(set["occ2Enabled"], json!(true));
    assert_eq!(set["applied"], json!(true));
    assert_eq!(set["scope"], json!("this chat"));
    assert_eq!(set["effectiveForNewMessages"], json!(true));
    assert!(set["sourceSessionId"].is_string());
    let first_occ = artifact_preference_path(&first_file);
    assert!(first_occ.is_file(), "the daemon wrote {first_occ:?} itself");
    assert!(first_occ.starts_with(fixture.root.path()), "the artifact stays inside the daemon profile");
    let settings = fs::read_to_string(fixture.agent_dir.join("settings.json")).unwrap();
    let settings: Value = serde_json::from_str(&settings).unwrap();
    assert!(settings.get("agentCommunication").is_none(), "global settings must stay untouched");

    // The read is daemon-authoritative and immediately visible (no restart).
    assert_eq!(
        request(&client, json!({"type":"occ_get_preferences", "activeSessionId":first})).await,
        json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": set["sourceSessionId"].clone()})
    );

    // An explicit off survives the new default; independent chats stay on.
    request(&client, json!({"type":"occ_set_preferences", "activeSessionId":first, "occ2Enabled":false})).await;
    assert_eq!(request(&client, json!({"type":"occ_get_preferences", "activeSessionId":first})).await["occ2Enabled"], json!(false));
    // Independent chats on the same daemon and profile stay isolated.
    let second = create_session(&client, &fixture, None).await;
    assert_eq!(
        request(&client, json!({"type":"occ_get_preferences", "activeSessionId":second})).await["occ2Enabled"],
        json!(true)
    );

    // Missing or unknown sessions fail loudly instead of returning a default.
    assert!(request_result(&client, json!({"type":"occ_get_preferences", "activeSessionId":"occ-no-such-session"})).await.is_err());
    assert!(request_result(&client, json!({"type":"occ_set_preferences", "activeSessionId":"occ-no-such-session", "occ2Enabled":true})).await.is_err());

    // Invalid payloads fail loudly too.
    assert!(request_result(&client, json!({"type":"occ_set_preferences", "activeSessionId":first})).await.is_err());

    // Corrupt and oversized preference storage surface as failures, never as
    // silent values.
    fs::write(&first_occ, "not json").unwrap();
    assert!(request_result(&client, json!({"type":"occ_get_preferences", "activeSessionId":first})).await.is_err());
    fs::write(&first_occ, "x".repeat(4097)).unwrap();
    assert!(request_result(&client, json!({"type":"occ_get_preferences", "activeSessionId":first})).await.is_err());
    // A set heals the storage and reports the stored value back.
    assert_eq!(
        request(&client, json!({"type":"occ_set_preferences", "activeSessionId":first, "occ2Enabled":false})).await["occ2Enabled"],
        json!(false)
    );

    // The preference survives a daemon restart and reattach. The first
    // connection closes inside the runtime; the daemon drops outside it.
    client.close().await;
    first_file
        });
    drop(daemon);
    let mut restarted = OwnedDaemon { process: fixture.spawn("occ-daemon-restarted"), socket: fixture.socket.clone() };
    wait_for_daemon(&mut restarted);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
    let client = Arc::new(DaemonClient::create(&fixture.socket));
    client.connect(1000).await.unwrap();
    client.wait_for_hello(1000).await.unwrap();
    let resumed = create_session(&client, &fixture, Some(&first_file)).await;
    assert_eq!(
        request(&client, json!({"type":"occ_get_preferences", "activeSessionId":resumed})).await["occ2Enabled"],
        json!(false)
    );
    assert_eq!(
        request(&client, json!({"type":"occ_set_preferences", "activeSessionId":resumed, "occ2Enabled":true})).await["applied"],
        json!(true)
    );
    client.close().await;
        });
}

// Plain #[test] + an explicit runtime (same reason as above: the daemon
// handle must drop outside any async runtime).
#[test]
fn daemon_reports_inherited_preferences_for_subagent_chats() {
    let _model = LocalModel::start();
    let fixture = PrivateDaemon::new();
    let mut daemon = OwnedDaemon { process: fixture.spawn("occ-child-daemon"), socket: fixture.socket.clone() };
    wait_for_daemon(&mut daemon);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
    let client = Arc::new(DaemonClient::create(&fixture.socket));
    client.connect(1000).await.unwrap();
    client.wait_for_hello(1000).await.unwrap();

    // A parent chat with the toggle on.
    let parent = create_session(&client, &fixture, None).await;
    let parent_state = request(&client, json!({"type":"get_connection_state", "activeSessionId":parent})).await;
    let parent_file = parent_state["sessionFile"].as_str().unwrap().to_string();
    request(&client, json!({"type":"occ_set_preferences", "activeSessionId":parent, "occ2Enabled":true})).await;

    // A subagent chat: same sessions directory, rlmDepth 1, parentSession
    // pointing at the parent's file. The daemon loads it through the same
    // sessionPath path a resumed chat uses.
    let parent_name = Path::new(&parent_file).file_name().unwrap().to_string_lossy().to_string();
    let header_line = fs::read_to_string(&parent_file).unwrap().lines().next().unwrap().to_string();
    let mut header: Value = serde_json::from_str(&header_line).unwrap();
    header["rlmDepth"] = json!(1);
    header["parentSession"] = json!(parent_name);
    let child_file = fixture
        .root
        .path()
        .join("sessions")
        .join("occ-child-chat.jsonl")
        .to_string_lossy()
        .into_owned();
    fs::write(&child_file, format!("{header}\n")).unwrap();

    let child = create_session(&client, &fixture, Some(&child_file)).await;
    let child_preferences = request(&client, json!({"type":"occ_get_preferences", "activeSessionId":child})).await;
    assert_eq!(child_preferences["occ2Enabled"], json!(true), "subagents inherit the parent chat's toggle");
    assert_eq!(child_preferences["inherited"], json!(true));
    assert_eq!(
        child_preferences["sourceSessionId"],
        json!(Path::new(&parent_file).file_stem().unwrap().to_string_lossy().into_owned())
    );
    client.close().await;
        });
}

// Isolated fake-provider profile, no prompts or model requests. The daemon is
// authoritative even though the test client has no matching local profile.
#[test]
fn daemon_owns_independent_occ31_and_occ32_preferences_without_latent_model_support() {
    let fixture = PrivateDaemon::new();
    let settings_before = fs::read(fixture.agent_dir.join("settings.json")).unwrap();
    let models_before = fs::read(fixture.agent_dir.join("models.json")).unwrap();
    let mut daemon = OwnedDaemon { process: fixture.spawn("occ31-daemon"), socket: fixture.socket.clone() };
    wait_for_daemon(&mut daemon);
    let file = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let client = Arc::new(DaemonClient::create(&fixture.socket));
        client.connect(1000).await.unwrap();
        assert!(client.wait_for_hello(1000).await.unwrap().supports("occ31_textmas"));
        let id = create_session(&client, &fixture, None).await;
        let other = create_session(&client, &fixture, None).await;
        let state = request(&client, json!({"type":"get_connection_state","activeSessionId":id})).await;
        assert_eq!(state["model"]["provider"], "local-cli-fixture");
        assert_eq!(state["occ3Modes"], json!({"textmasMode":"off","latentEnabled":false,"latentActive":false,"latentAvailable":false}));
        let file = state["sessionFile"].as_str().unwrap().to_string();
        let pref = pi_coding_agent::core::occ::textmas_settings::preference_path(&file);
        assert!(pref.starts_with(fixture.root.path()));
        assert!(!pref.exists());
        assert_eq!(request(&client, json!({"type":"occ31_get_preferences","activeSessionId":id})).await,
            json!({"occ31Mode":"off","inherited":false,"sourceSessionId":null}));
        request(&client, json!({"type":"occ_set_preferences","activeSessionId":id,"occ2Enabled":false})).await;
        let latent = request(&client, json!({"type":"occ32_set_enabled","activeSessionId":id,"enabled":true})).await;
        assert_eq!(latent["occ32Enabled"], true);
        assert_eq!(latent["applied"], true);
        let old_lane_bytes = fs::read(session_settings::preference_path(&file)).unwrap();
        for mode in ["sequential", "off", "hierarchical"] {
            let echo = request(&client, json!({"type":"occ31_set_mode","activeSessionId":id,"mode":mode})).await;
            assert_eq!(echo["occ31Mode"], mode);
            assert_eq!(echo["applied"], true);
            assert_eq!(request(&client, json!({"type":"occ31_get_preferences","activeSessionId":id})).await["occ31Mode"], mode);
            assert_eq!(fs::read(session_settings::preference_path(&file)).unwrap(), old_lane_bytes);
        }
        let state = request(&client, json!({"type":"get_connection_state","activeSessionId":id})).await;
        assert_eq!(state["occ3Modes"], json!({"textmasMode":"hierarchical","latentEnabled":true,"latentActive":false,"latentAvailable":false}));
        assert_eq!(request(&client, json!({"type":"occ31_get_preferences","activeSessionId":other})).await["occ31Mode"], "off");
        for mode in [Value::Null, json!("on"), json!(true), json!("hierarchal")] {
            assert!(request_result(&client, json!({"type":"occ31_set_mode","activeSessionId":id,"mode":mode})).await.is_err());
        }
        assert!(request_result(&client, json!({"type":"occ32_set_enabled","activeSessionId":id,"enabled":"yes"})).await.is_err());
        assert!(request_result(&client, json!({"type":"occ31_set_mode","activeSessionId":"missing","mode":"off"})).await.is_err());
        for corrupt in ["not json".to_string(), "x".repeat(4097)] {
            fs::write(&pref, corrupt).unwrap();
            assert!(request_result(&client, json!({"type":"occ31_get_preferences","activeSessionId":id})).await.is_err());
            assert_eq!(request(&client, json!({"type":"occ_get_preferences","activeSessionId":id})).await["occ2Enabled"], false);
            assert!(request(&client, json!({"type":"get_connection_state","activeSessionId":id})).await.get("occ3Modes").is_none());
        }
        request(&client, json!({"type":"occ31_set_mode","activeSessionId":id,"mode":"hierarchical"})).await;
        request(&client, json!({"type":"occ_set_preferences","activeSessionId":id,"occ2Enabled":true})).await;
        assert!(session_settings::resolve(&file).unwrap().occ3_enabled, "OCC2 setter must preserve latent ON");
        request(&client, json!({"type":"occ32_set_enabled","activeSessionId":id,"enabled":false})).await;
        assert!(session_settings::load(&file).unwrap(), "latent setter must preserve OCC2 ON");
        client.close().await;
        file
    });
    drop(daemon);
    let mut daemon = OwnedDaemon { process: fixture.spawn("occ31-restart"), socket: fixture.socket.clone() };
    wait_for_daemon(&mut daemon);
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let client = Arc::new(DaemonClient::create(&fixture.socket));
        client.connect(1000).await.unwrap();
        let resumed = create_session(&client, &fixture, Some(&file)).await;
        assert_eq!(request(&client, json!({"type":"occ31_get_preferences","activeSessionId":resumed})).await["occ31Mode"], "hierarchical");
        assert_eq!(request(&client, json!({"type":"get_connection_state","activeSessionId":resumed})).await["occ3Modes"]["latentEnabled"], false);
        client.close().await;
    });
    assert_eq!(fs::read(fixture.agent_dir.join("settings.json")).unwrap(), settings_before);
    assert_eq!(fs::read(fixture.agent_dir.join("models.json")).unwrap(), models_before);
}
