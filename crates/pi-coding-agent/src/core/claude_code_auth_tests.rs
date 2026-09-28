use super::*;
use crate::core::auth_storage::AuthCredential;
use crate::core::model_registry::{ModelRegistry, ProviderConfigInput};
use std::sync::Arc;

fn blob(access: &str, expires: i64) -> String {
    json!({"claudeAiOauth": {"accessToken": access, "refreshToken": "synthetic-refresh",
        "expiresAt": expires, "subscriptionType": "max", "scopes": ["user:inference"]},
        "mcpOAuth": {"keep": "untouched"}})
    .to_string()
}

fn fixture(directory: &Path) -> ClaudeCodeAuth {
    ClaudeCodeAuth {
        directory: directory.into(),
        service: None,
        program: directory.join("no-such-claude"),
        version: Mutex::new(Some((Instant::now(), Ok(Some("2.1.999".into()))))),
        cache: Mutex::new(None),
        last_version: Mutex::new(None),
        refresh_url: "http://127.0.0.1:1/must-not-contact-real-provider".into(),
    }
}

fn storage(auth: ClaudeCodeAuth) -> AuthStorage {
    let mut storage = AuthStorage::in_memory(IndexMap::new(), None);
    storage.claude_code = Some(Arc::new(auth));
    storage.set(
        "anthropic",
        AuthCredential::ApiKey {
            key: "synthetic-paid-key".into(),
            prime_team: None,
        },
    );
    storage.set_runtime_api_key("anthropic", "synthetic-runtime-key");
    storage
}

fn write_login(directory: &Path, expires: i64) {
    std::fs::write(
        directory.join(".credentials.json"),
        blob("sk-ant-oat01-synthetic", expires),
    )
    .unwrap();
}

async fn server<F>(handler: F) -> (String, tokio::task::JoinHandle<Value>)
where
    F: FnOnce(&Value) -> (u16, String) + Send + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let (header_end, content_length) = loop {
            let mut chunk = [0; 4096];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&bytes[..end]);
                let length = head
                    .lines()
                    .find_map(|line| {
                        line.to_lowercase()
                            .strip_prefix("content-length:")
                            .map(|value| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (end + 4, length);
            }
        };
        while bytes.len() < header_end + content_length {
            let mut chunk = [0; 4096];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&chunk[..n]);
        }
        let mut request: Value =
            serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap();
        request["fixture_headers"] =
            json!(String::from_utf8_lossy(&bytes[..header_end]).to_string());
        let (status, body) = handler(&request);
        let reply = format!(
            "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(reply.as_bytes()).await.unwrap();
        request
    });
    (url, task)
}

fn refresh_body() -> String {
    json!({"access_token":"sk-ant-oat01-rotated", "refresh_token":"synthetic-rotated-refresh", "expires_in":3600}).to_string()
}

#[test]
fn parses_only_subscription_login_and_never_echoes_invalid_secrets() {
    assert!(parse_credentials(r#"{"apiKey":"synthetic","mcpOAuth":{}}"#)
        .unwrap()
        .is_none());
    for raw in [
        "secret-invalid-json",
        r#"{"claudeAiOauth":{"accessToken":"secret-invalid-token","expiresAt":1000}}"#,
        r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-synthetic","expiresAt":"secret"}}"#,
    ] {
        let error = parse_credentials(raw).unwrap_err();
        assert!(!error.contains("secret"));
        assert!(!error.contains("sk-ant"));
    }
    assert!(parse_credentials(&blob("sk-ant-oat01-synthetic", 1000))
        .unwrap()
        .is_some());
}

#[test]
fn refresh_preserves_missing_refresh_token_and_rejects_invalid_response() {
    let credentials = parse_refresh_response(
        br#"{"access_token":"sk-ant-oat01-new","expires_in":3600}"#,
        "keep-refresh",
        1000,
    )
    .unwrap();
    assert_eq!(credentials.refresh, "keep-refresh");
    assert_eq!(credentials.expires, 3_601_000.0);
    for response in [
        "secret-invalid",
        r#"{"access_token":"secret","expires_in":3600}"#,
        r#"{"access_token":"sk-ant-oat01-new","expires_in":-1}"#,
        r#"{"access_token":"sk-ant-oat01-new","expires_in":3600,"refresh_token":""}"#,
    ] {
        assert!(!parse_refresh_response(response.as_bytes(), "old", 1)
            .unwrap_err()
            .contains("secret"));
    }
}

#[test]
fn keychain_profile_and_command_encoding_keep_accounts_separate() {
    assert_eq!(keychain_service(None), "Claude Code-credentials");
    assert_eq!(keychain_service(Some("")), "Claude Code-credentials");
    assert_ne!(
        keychain_service(Some("/work/claude")),
        keychain_service(Some("/personal/claude"))
    );
    assert_eq!(
        keychain_account("    \"acct\"<blob>=\"my account\"\n").as_deref(),
        Some("my account")
    );
    assert!(keychain_account("missing account").is_none());
    assert_eq!(security_quote("a\\b\"c").unwrap(), "\"a\\\\b\\\"c\"");
    assert!(security_quote("inject\ncommand").is_err());
}

#[test]
fn versions_are_strict_and_missing_cli_does_not_import_credentials() {
    assert_eq!(
        parse_version("2.1.283 (Claude Code)\n").as_deref(),
        Some("2.1.283")
    );
    for text in [
        "2.1.283",
        "fake",
        "2.1.2\r\nInjected: header (Claude Code)",
        "a.b.c (Claude Code)",
    ] {
        assert!(parse_version(text).is_none());
    }
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), super::super::now_millis() + 300_000);
    let auth = fixture(directory.path());
    *auth.version.lock().unwrap() = None;
    assert!(auth.snapshot(true).unwrap().is_none());
}

#[tokio::test]
async fn broken_cli_without_a_saved_login_keeps_normal_auth_available() {
    let directory = tempfile::tempdir().unwrap();
    let auth = fixture(directory.path());
    *auth.version.lock().unwrap() = Some((Instant::now(), Err("version failed".into())));
    assert!(auth.snapshot(true).unwrap().is_none());
    write_login(directory.path(), super::super::now_millis() + 300_000);
    assert!(auth.snapshot(true).is_err());
}

#[cfg(unix)]
#[test]
fn credential_pipe_is_rejected_without_blocking() {
    use std::os::unix::ffi::OsStrExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join(".credentials.json");
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(Source::File(path)
        .read()
        .unwrap_err()
        .contains("not a regular file"));
}

#[tokio::test]
async fn selects_claude_before_other_credentials_without_seeding_auth_json() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), super::super::now_millis() + 300_000);
    let mut storage = storage(fixture(directory.path()));
    let before = storage.get_all();
    let status = storage.get_auth_status("anthropic");
    assert_eq!(status.source.as_deref(), Some(AUTH_SOURCE_CLAUDE_CODE));
    assert_eq!(status.label.as_deref(), Some("Claude Code 2.1.999 OAuth"));
    let result = storage
        .get_api_key_with_source_token("anthropic", true)
        .await
        .unwrap();
    assert_eq!(result.api_key.as_deref(), Some("sk-ant-oat01-synthetic"));
    assert_eq!(result.source_token.unwrap().source, AUTH_SOURCE_CLAUDE_CODE);
    assert_eq!(storage.get_all(), before);
    storage
        .login("anthropic", Default::default())
        .await
        .unwrap();
    assert_eq!(storage.get_all(), before);
    assert!(storage
        .logout("anthropic")
        .unwrap_err()
        .contains("claude auth logout"));
    storage.set_runtime_api_key("openrouter", "synthetic-router");
    assert_eq!(
        storage
            .get_api_key("openrouter", false)
            .await
            .unwrap()
            .as_deref(),
        Some("synthetic-router")
    );
}

#[tokio::test]
async fn reads_peer_rotation_before_request_and_recovers_from_stale_fingerprint() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), super::super::now_millis() + 300_000);
    let mut storage = storage(fixture(directory.path()));
    let initial = storage
        .get_api_key_with_source_token("anthropic", false)
        .await
        .unwrap();
    storage.mark_auth_source_stale(initial.source_token.as_ref().unwrap());
    assert!(storage.get_api_key("anthropic", false).await.is_err());
    std::fs::write(
        directory.path().join(".credentials.json"),
        blob("sk-ant-oat01-peer", super::super::now_millis() + 300_000),
    )
    .unwrap();
    assert_eq!(
        storage
            .get_api_key("anthropic", false)
            .await
            .unwrap()
            .as_deref(),
        Some("sk-ant-oat01-peer")
    );
}

#[tokio::test]
async fn refreshes_once_across_two_instances_and_preserves_metadata() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), 1);
    let (url, server) = server(|request| {
        assert_eq!(request["grant_type"], "refresh_token");
        assert_eq!(request["refresh_token"], "synthetic-refresh");
        assert_eq!(request["client_id"], CLIENT_ID);
        (200, refresh_body())
    })
    .await;
    let mut first = fixture(directory.path());
    first.refresh_url = url.clone();
    let mut second = fixture(directory.path());
    second.refresh_url = url;
    let (first, second) = tokio::join!(first.resolve(), second.resolve());
    assert_eq!(
        first.unwrap().unwrap().credentials.access,
        "sk-ant-oat01-rotated"
    );
    assert_eq!(
        second.unwrap().unwrap().credentials.access,
        "sk-ant-oat01-rotated"
    );
    server.await.unwrap();
    let path = directory.path().join(".credentials.json");
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        saved["claudeAiOauth"]["refreshToken"],
        "synthetic-rotated-refresh"
    );
    assert_eq!(saved["claudeAiOauth"]["subscriptionType"], "max");
    assert_eq!(saved["mcpOAuth"]["keep"], "untouched");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[tokio::test]
async fn refresh_failure_never_uses_paid_key_or_logs_response_secrets() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), 1);
    let original = std::fs::read(directory.path().join(".credentials.json")).unwrap();
    let (url, server) = server(|_| (400, "synthetic-secret-response".into())).await;
    let mut auth = fixture(directory.path());
    auth.refresh_url = url;
    let mut storage = storage(auth);
    let error = storage.get_api_key("anthropic", true).await.unwrap_err();
    assert!(error.contains("HTTP 400"));
    assert!(!error.contains("synthetic-secret-response"));
    assert_eq!(
        std::fs::read(directory.path().join(".credentials.json")).unwrap(),
        original
    );
    server.await.unwrap();
}

#[tokio::test]
async fn does_not_overwrite_a_login_changed_by_claude_during_refresh() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), 1);
    let path = directory.path().join(".credentials.json");
    let peer = blob(
        "sk-ant-oat01-peer-login",
        super::super::now_millis() + 300_000,
    );
    let expected = peer.clone();
    let (url, server) = server(move |_| {
        std::fs::write(path, peer).unwrap();
        (200, refresh_body())
    })
    .await;
    let mut auth = fixture(directory.path());
    auth.refresh_url = url;
    assert_eq!(
        auth.resolve().await.unwrap().unwrap().credentials.access,
        "sk-ant-oat01-peer-login"
    );
    assert_eq!(
        std::fs::read_to_string(directory.path().join(".credentials.json")).unwrap(),
        expected
    );
    server.await.unwrap();
}

#[tokio::test]
async fn cancelled_request_finishes_refresh_write_back() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), 1);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut auth = fixture(directory.path());
    auth.refresh_url = format!("http://{}", listener.local_addr().unwrap());
    let storage = storage(auth);
    let request = tokio::spawn(async move { storage.resolve_claude_code_auth("anthropic").await });
    let (mut socket, _) = listener.accept().await.unwrap();
    let mut buffer = [0; 4096];
    assert!(socket.read(&mut buffer).await.unwrap() > 0);
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    let body = refresh_body();
    socket
        .write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let path = directory.path().join(".credentials.json");
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if std::fs::read_to_string(&path)
                .unwrap()
                .contains("synthetic-rotated-refresh")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("token rotation must be saved after caller cancellation");
}

#[tokio::test]
async fn malformed_store_and_write_failure_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join(".credentials.json"),
        "synthetic-invalid-json",
    )
    .unwrap();
    let mut auth = storage(fixture(directory.path()));
    assert!(auth
        .get_api_key("anthropic", true)
        .await
        .unwrap_err()
        .contains("Invalid Claude Code credential JSON"));
    assert_eq!(
        std::fs::read_to_string(directory.path().join(".credentials.json")).unwrap(),
        "synthetic-invalid-json"
    );
    let missing = Source::File(
        directory
            .path()
            .join("missing-parent")
            .join("credentials.json"),
    );
    assert!(missing
        .write("synthetic-secret")
        .unwrap_err()
        .contains("Cannot save"));
}

#[tokio::test]
async fn registry_pins_version_and_rejects_credential_forwarding() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), super::super::now_millis() + 300_000);
    let mut registry = ModelRegistry::in_memory(storage(fixture(directory.path())));
    registry
        .register_provider(
            "anthropic",
            ProviderConfigInput {
                headers: Some(IndexMap::from([
                    ("User-Agent".into(), "stale-version".into()),
                    ("X-Api-Key".into(), "synthetic-paid-key".into()),
                    ("Authorization".into(), "wrong-account".into()),
                ])),
                ..Default::default()
            },
        )
        .unwrap();
    let mut model = registry.find("anthropic", "claude-opus-5-5").unwrap();
    assert_eq!(model.context_window, 1_000_000.0);
    assert!(registry.has_configured_auth(&model));
    assert!(registry.is_using_oauth(&model));
    let resolved = registry.get_api_key_and_headers(&model).await;
    assert!(resolved.ok);
    assert_eq!(resolved.api_key.as_deref(), Some("sk-ant-oat01-synthetic"));
    let headers = resolved.headers.unwrap();
    assert_eq!(headers["user-agent"], "claude-cli/2.1.999");
    assert!(!headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization")
            || key.eq_ignore_ascii_case("x-api-key")));
    for endpoint in [
        "http://api.anthropic.com",
        "https://api.anthropic.com.evil.test",
        "https://proxy.test",
        "https://api.anthropic.com:8443",
        "https://user@api.anthropic.com",
        "https://api.anthropic.com/other",
    ] {
        model.base_url = endpoint.into();
        let resolved = registry.get_api_key_and_headers(&model).await;
        assert!(!resolved.ok, "{endpoint}");
        assert!(resolved.api_key.is_none());
        assert!(!registry.has_configured_auth(&model));
    }
}

#[tokio::test]
async fn native_provider_sends_resolved_oauth_and_version_to_local_fixture() {
    let directory = tempfile::tempdir().unwrap();
    write_login(directory.path(), super::super::now_millis() + 300_000);
    let mut registry = ModelRegistry::in_memory(storage(fixture(directory.path())));
    let mut model = registry.find("anthropic", "claude-opus-5-5").unwrap();
    model.headers = Some(IndexMap::from([
        ("Authorization".into(), "Bearer stale-model-token".into()),
        ("X-Api-Key".into(), "paid-model-key".into()),
        ("User-Agent".into(), "stale-model-version".into()),
        ("X-App".into(), "stale-app".into()),
    ]));
    let auth = registry.get_api_key_and_headers(&model).await;
    assert!(auth.ok);
    let (url, server) = server(|request| {
        let headers = request["fixture_headers"].as_str().unwrap();
        assert!(headers.contains("authorization: Bearer sk-ant-oat01-synthetic"));
        assert!(headers.contains("user-agent: claude-cli/2.1.999"));
        assert!(headers.contains("x-app: cli"));
        assert!(headers.contains("oauth-2025-04-20"));
        assert!(!headers.contains("x-api-key:"));
        assert_eq!(request["model"], "claude-opus-5-5");
        assert!(request["system"][0]["text"].as_str().unwrap().contains("Claude Code"));
        assert_eq!(request["messages"].as_array().unwrap().len(), 1);
        assert!(!request.to_string().contains("local-only authentication diagnostic"));
        (200, concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"fixture\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"fixture ok\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n").into())
    }).await;
    // Only the synthetic transport model is redirected; production registry checks remain strict.
    model.base_url = url;
    let stream = pi_ai::providers::anthropic::stream_anthropic(
        &model,
        &pi_ai::types::Context {
            system_prompt: Some("Optimus test".into()),
            messages: vec![
                pi_ai::types::Message::user(pi_ai::types::UserMessage::new(
                    pi_ai::types::UserContent::Text("fixture request".into()),
                    1,
                )),
                pi_ai::types::Message::assistant(pi_ai::types::AssistantMessage {
                    stop_reason: "error".into(),
                    error_message: Some("local-only authentication diagnostic".into()),
                    ..Default::default()
                }),
            ],
            tools: None,
        },
        Some(pi_ai::providers::anthropic::AnthropicOptions::from_base(
            &pi_ai::types::StreamOptions {
                api_key: auth.api_key,
                headers: auth.headers,
                ..Default::default()
            },
        )),
    );
    let response = tokio::time::timeout(Duration::from_secs(5), stream.result())
        .await
        .unwrap();
    assert_eq!(response.stop_reason, "stop", "{:?}", response.error_message);
    server.await.unwrap();
}

#[cfg(unix)]
#[test]
fn installed_cli_probe_is_bounded_and_rechecks_version() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let program = directory.path().join("claude");
    std::fs::write(&program, "#!/bin/sh\nprintf '2.1.123 (Claude Code)\\n'\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut auth = fixture(directory.path());
    auth.program = program.clone();
    *auth.version.lock().unwrap() = None;
    assert_eq!(auth.version().unwrap().as_deref(), Some("2.1.123"));
    std::fs::write(&program, "#!/bin/sh\nprintf '2.1.124 (Claude Code)\\n'\n").unwrap();
    *auth.version.lock().unwrap() = None;
    assert_eq!(auth.version().unwrap().as_deref(), Some("2.1.124"));
    let started = Instant::now();
    assert!(run_command(
        Path::new("/bin/sh"),
        &["-c", "sleep 3"],
        None,
        Duration::from_millis(100)
    )
    .is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn production_discovery_and_opt_out_use_an_isolated_profile() {
    let directory = tempfile::tempdir().unwrap();
    let profile = directory.path().join("optimus");
    let claude = directory.path().join("claude-config");
    std::fs::create_dir_all(&claude).unwrap();
    write_login(&claude, super::super::now_millis() + 300_000);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let executable = directory.path().join("claude");
        std::fs::write(
            &executable,
            "#!/bin/sh\nprintf '2.1.999 (Claude Code)\\n'\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    #[cfg(windows)]
    std::fs::write(
        directory.path().join("claude.cmd"),
        "@echo off\r\necho 2.1.999 (Claude Code)\r\n",
    )
    .unwrap();
    for opt_out in [false, true] {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "core::auth_storage::claude_code::tests::discovery_fixture",
                "--ignored",
            ])
            .env("HOME", directory.path())
            .env("USERPROFILE", directory.path())
            .env("PRIME_AGENT_CODING_AGENT_DIR", &profile)
            .env("CLAUDE_CONFIG_DIR", &claude)
            .env_remove("CLAUDE_SECURESTORAGE_CONFIG_DIR")
            .env("PATH", directory.path())
            .env("OPTIMUS_CLAUDE_CODE_AUTH", if opt_out { "0" } else { "1" })
            .env("ANTHROPIC_API_KEY", "synthetic-environment-key")
            .env_remove("ANTHROPIC_OAUTH_TOKEN")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
    }
}

#[test]
#[ignore = "subprocess fixture with private HOME, fake CLI, and synthetic credentials"]
fn discovery_fixture() {
    let mut storage = AuthStorage::create(None, None);
    let expected = if std::env::var("OPTIMUS_CLAUDE_CODE_AUTH").unwrap() == "0" {
        "synthetic-environment-key"
    } else {
        "sk-ant-oat01-synthetic"
    };
    let key = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(storage.get_api_key("anthropic", false))
        .unwrap()
        .unwrap();
    assert_eq!(key, expected);
}
