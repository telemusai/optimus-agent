//! Claude Code owns these credentials; Optimus never seeds a second token store.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use pi_ai::utils::oauth::anthropic::{CLIENT_ID, TOKEN_URL};
use pi_ai::utils::oauth::types::OAuthCredentials;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{
    AuthApiKeyResult, AuthSourceCandidate, AuthSourceToken, AuthStorage, AUTH_SOURCE_CLAUDE_CODE,
};
use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};
use crate::utils::store_lock::open_store_lock;

const MAX_BLOB: usize = 1024 * 1024;
const REFRESH_MARGIN_MS: f64 = 60_000.0;
const LOGIN_HELP: &str = "Claude Code OAuth is selected. Run `claude auth login` to renew it, or set OPTIMUS_CLAUDE_CODE_AUTH=0 to use Optimus credentials.";

#[derive(Clone)]
enum Source {
    File(PathBuf),
    Keychain(String),
}

impl Source {
    fn identity(&self) -> String {
        match self {
            Self::File(path) => format!("file:{}", path.display()),
            Self::Keychain(service) => format!("keychain:{service}"),
        }
    }

    fn read(&self) -> Result<Option<String>, String> {
        match self {
            Self::File(path) => {
                let mut options = std::fs::OpenOptions::new();
                options.read(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.custom_flags(libc::O_NONBLOCK);
                }
                let file = match options.open(path) {
                    Ok(file) => file,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(_) => return Err("Cannot read Claude Code credential file".into()),
                };
                if !file
                    .metadata()
                    .map_err(|_| "Cannot inspect Claude Code credential file")?
                    .is_file()
                {
                    return Err("Claude Code credential path is not a regular file".into());
                }
                let mut bytes = Vec::new();
                file.take(MAX_BLOB as u64 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|_| "Cannot read Claude Code credential file")?;
                decode_blob(bytes).map(Some)
            }
            Self::Keychain(service) => {
                let (status, bytes) = run_command(
                    Path::new("/usr/bin/security"),
                    &["find-generic-password", "-s", service, "-w"],
                    None,
                    Duration::from_secs(2),
                )?;
                match status {
                    Some(0) => decode_blob(bytes).map(Some),
                    Some(44) => Ok(None),
                    _ => Err("Cannot read Claude Code Keychain entry; unlock the Keychain and allow access".into()),
                }
            }
        }
    }

    fn write(&self, blob: &str) -> Result<(), String> {
        match self {
            Self::File(path) => write_file_atomic_sync(
                &path.to_string_lossy(),
                blob,
                WriteFileAtomicOptions {
                    mode: Some(0o600),
                    fsync: true,
                    ..Default::default()
                },
            )
            .map_err(|_| "Cannot save rotated Claude Code credentials".into()),
            Self::Keychain(service) => {
                let (status, bytes) = run_command(
                    Path::new("/usr/bin/security"),
                    &["find-generic-password", "-s", service],
                    None,
                    Duration::from_secs(2),
                )?;
                if status != Some(0) {
                    return Err(
                        "Cannot locate the Claude Code Keychain account for write-back".into(),
                    );
                }
                let metadata = String::from_utf8_lossy(&bytes);
                let account = keychain_account(&metadata)
                    .ok_or("Cannot identify the Claude Code Keychain account for write-back")?;
                // security's interactive input keeps secrets out of process arguments.
                let input = format!(
                    "add-generic-password -U -s {} -a {} -w {}\n",
                    security_quote(service)?,
                    security_quote(&account)?,
                    security_quote(blob)?
                );
                let (status, _) = run_command(
                    Path::new("/usr/bin/security"),
                    &["-i"],
                    Some(input),
                    Duration::from_secs(2),
                )?;
                if status != Some(0) {
                    return Err("Cannot save rotated Claude Code credentials to Keychain".into());
                }
                // Interactive security may exit successfully after a command fails.
                if self.read()?.as_deref().map(str::trim) != Some(blob.trim()) {
                    return Err("Claude Code Keychain write-back could not be verified".into());
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone)]
struct Snapshot {
    source: Source,
    credentials: OAuthCredentials,
    version: String,
}

type SnapshotResult = Result<Option<Snapshot>, String>;

pub(super) struct ClaudeCodeAuth {
    directory: PathBuf,
    service: Option<String>,
    program: PathBuf,
    version: Mutex<Option<(Instant, Result<Option<String>, String>)>>,
    cache: Mutex<Option<(Instant, SnapshotResult)>>,
    last_version: Mutex<Option<String>>,
    refresh_url: String,
}

impl ClaudeCodeAuth {
    pub(super) fn from_environment() -> Option<Self> {
        if std::env::var("OPTIMUS_CLAUDE_CODE_AUTH").ok().as_deref() == Some("0") {
            return None;
        }
        let home = dirs::home_dir()?;
        let selector = std::env::var("CLAUDE_SECURESTORAGE_CONFIG_DIR")
            .ok()
            .or_else(|| std::env::var("CLAUDE_CONFIG_DIR").ok());
        let directory = selector
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".claude"));
        let service = cfg!(target_os = "macos").then(|| keychain_service(selector.as_deref()));
        Some(Self {
            directory,
            service,
            program: PathBuf::from("claude"),
            version: Mutex::new(None),
            cache: Mutex::new(None),
            last_version: Mutex::new(None),
            refresh_url: TOKEN_URL.to_string(),
        })
    }

    fn version(&self) -> Result<Option<String>, String> {
        let mut cache = self.version.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((checked, result)) = cache.as_ref() {
            if checked.elapsed() < Duration::from_secs(300) {
                return result.clone();
            }
        }
        let result = match run_command(&self.program, &["--version"], None, Duration::from_secs(2))
        {
            Ok((Some(0), bytes)) => parse_version(&String::from_utf8_lossy(&bytes))
                .map(Some)
                .ok_or_else(|| "Cannot determine the installed Claude Code version".into()),
            Ok(_) => Err("Claude Code version check failed".into()),
            Err(error) if error == "Command not installed" => Ok(None),
            Err(_) => Err("Claude Code version check failed or timed out".into()),
        };
        *cache = Some((Instant::now(), result.clone()));
        result
    }

    fn snapshot(&self, force: bool) -> SnapshotResult {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if !force {
            if let Some((checked, result)) = cache.as_ref() {
                if checked.elapsed() < Duration::from_secs(30) {
                    return result.clone();
                }
            }
        }
        let result = self.read_snapshot();
        *cache = Some((Instant::now(), result.clone()));
        result
    }

    fn read_snapshot(&self) -> SnapshotResult {
        let version = match self.version() {
            Ok(Some(version)) => Ok(version),
            Ok(None) => return Ok(None),
            Err(error) => Err(error),
        };
        let file_path = self.directory.join(".credentials.json");
        let file = Source::File(std::fs::canonicalize(&file_path).unwrap_or(file_path));
        let mut keychain_error = None;
        if let Some(service) = &self.service {
            let source = Source::Keychain(service.clone());
            match read_credentials(&source) {
                Ok(Some(credentials)) => {
                    return Ok(Some(Snapshot {
                        source,
                        credentials,
                        version: version?,
                    }))
                }
                Ok(None) => {}
                Err(error) => keychain_error = Some(error),
            }
        }
        if let Some(credentials) = read_credentials(&file)? {
            return Ok(Some(Snapshot {
                source: file,
                credentials,
                version: version?,
            }));
        }
        match keychain_error {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    async fn resolve(&self) -> SnapshotResult {
        let Some(snapshot) = self.snapshot(true)? else {
            return Ok(None);
        };
        if fresh(&snapshot.credentials) {
            *self.last_version.lock().unwrap_or_else(|p| p.into_inner()) =
                Some(snapshot.version.clone());
            return Ok(Some(snapshot));
        }
        // Stable, separate lock: never interfere with Claude Code's own lock format.
        let lock_path = match &snapshot.source {
            Source::File(path) => path.with_extension("json.optimus-refresh"),
            Source::Keychain(_) => {
                super::create_dir_mode(&self.directory, 0o700)
                    .map_err(|_| "Cannot create Claude Code refresh lock directory")?;
                self.directory.join(".optimus-oauth-refresh")
            }
        };
        let lock = open_store_lock(&lock_path.to_string_lossy())
            .map_err(|_| "Cannot open Claude Code refresh lock")?;
        let deadline = Instant::now() + Duration::from_secs(35);
        loop {
            match lock.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await
                }
                _ => return Err("Claude Code credential refresh is busy; retry".into()),
            }
        }
        let result = self.refresh_locked(snapshot).await;
        *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = None;
        if let Ok(Some(snapshot)) = &result {
            *self.last_version.lock().unwrap_or_else(|p| p.into_inner()) =
                Some(snapshot.version.clone());
        }
        result
    }

    async fn refresh_locked(&self, mut snapshot: Snapshot) -> SnapshotResult {
        snapshot.credentials = read_credentials(&snapshot.source)?
            .ok_or("Claude Code login changed during refresh; retry")?;
        if fresh(&snapshot.credentials) {
            return Ok(Some(snapshot));
        }
        if snapshot.credentials.refresh.is_empty() {
            return Err(LOGIN_HELP.into());
        }
        let refreshed = refresh_token(&self.refresh_url, &snapshot.credentials.refresh).await;
        // Claude Code does not share our lock. Re-read before committing and never
        // overwrite another login, rotated token, logout, or unrelated metadata.
        let raw = snapshot
            .source
            .read()?
            .ok_or("Claude Code logged out during refresh; retry")?;
        let current =
            parse_credentials(&raw)?.ok_or("Claude Code login changed during refresh; retry")?;
        if current != snapshot.credentials {
            if fresh(&current) {
                snapshot.credentials = current;
                return Ok(Some(snapshot));
            }
            return Err("Claude Code credentials changed during refresh; retry".into());
        }
        let refreshed = refreshed?;
        let updated = update_blob(&raw, &refreshed)?;
        snapshot.source.write(&updated)?;
        snapshot.credentials = refreshed;
        Ok(Some(snapshot))
    }
}

fn fresh(credentials: &OAuthCredentials) -> bool {
    credentials.expires > super::now_millis() as f64 + REFRESH_MARGIN_MS
}

fn read_credentials(source: &Source) -> Result<Option<OAuthCredentials>, String> {
    source
        .read()?
        .map(|raw| parse_credentials(&raw))
        .transpose()
        .map(Option::flatten)
}

fn parse_credentials(raw: &str) -> Result<Option<OAuthCredentials>, String> {
    let document: Value =
        serde_json::from_str(raw).map_err(|_| "Invalid Claude Code credential JSON")?;
    let Some(value) = document.get("claudeAiOauth") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let access = value
        .get("accessToken")
        .and_then(Value::as_str)
        .filter(|value| {
            value.starts_with("sk-ant-oat") && !value.bytes().any(|c| c.is_ascii_whitespace())
        })
        .ok_or("Invalid Claude Code OAuth access token")?;
    let expires = value
        .get("expiresAt")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or("Invalid Claude Code OAuth expiry")?;
    let refresh = value
        .get("refreshToken")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(Some(OAuthCredentials {
        access: access.into(),
        refresh: refresh.into(),
        expires,
        extra: Default::default(),
    }))
}

fn update_blob(raw: &str, credentials: &OAuthCredentials) -> Result<String, String> {
    let mut document: Value =
        serde_json::from_str(raw).map_err(|_| "Invalid Claude Code credential JSON")?;
    let oauth = document
        .get_mut("claudeAiOauth")
        .and_then(Value::as_object_mut)
        .ok_or("Claude Code OAuth login is missing")?;
    oauth.insert("accessToken".into(), json!(credentials.access));
    oauth.insert("refreshToken".into(), json!(credentials.refresh));
    oauth.insert("expiresAt".into(), json!(credentials.expires));
    serde_json::to_string(&document).map_err(|_| "Cannot serialize Claude Code credentials".into())
}

async fn refresh_token(endpoint: &str, current_refresh: &str) -> Result<OAuthCredentials, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Cannot create OAuth client")?;
    let mut response = client
        .post(endpoint)
        .json(&json!({
            "grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": current_refresh,
        }))
        .send()
        .await
        .map_err(|_| "Claude Code OAuth refresh request failed; retry")?;
    if !response.status().is_success() {
        return Err(format!(
            "Claude Code OAuth refresh failed (HTTP {}). {LOGIN_HELP}",
            response.status().as_u16()
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "Cannot read OAuth refresh response")?
    {
        if bytes.len() + chunk.len() > MAX_BLOB {
            return Err("OAuth refresh response is too large".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    parse_refresh_response(&bytes, current_refresh, super::now_millis())
}

fn parse_refresh_response(
    bytes: &[u8],
    current_refresh: &str,
    now: i64,
) -> Result<OAuthCredentials, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| "Invalid OAuth refresh response JSON")?;
    let access = value
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| {
            value.starts_with("sk-ant-oat") && !value.bytes().any(|c| c.is_ascii_whitespace())
        })
        .ok_or("OAuth refresh response has no valid access token")?;
    let lifetime = value
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 60.0 && *value <= 366.0 * 86400.0)
        .ok_or("OAuth refresh response has no valid expiry")?;
    let refresh = match value.get("refresh_token") {
        None | Some(Value::Null) => current_refresh,
        Some(Value::String(value)) if !value.is_empty() => value,
        _ => return Err("OAuth refresh response has an invalid refresh token".into()),
    };
    Ok(OAuthCredentials {
        access: access.into(),
        refresh: refresh.into(),
        expires: now as f64 + lifetime * 1000.0,
        extra: Default::default(),
    })
}

fn parse_version(output: &str) -> Option<String> {
    let version = output.trim().strip_suffix(" (Claude Code)")?;
    let parts: Vec<_> = version.split('.').collect();
    (version.len() <= 40
        && parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())))
    .then(|| version.to_string())
}

fn keychain_service(selector: Option<&str>) -> String {
    match selector.filter(|value| !value.is_empty()) {
        Some(value) => format!(
            "Claude Code-credentials-{}",
            &format!("{:x}", Sha256::digest(value.as_bytes()))[..8]
        ),
        None => "Claude Code-credentials".into(),
    }
}

fn keychain_account(metadata: &str) -> Option<String> {
    let encoded = metadata
        .lines()
        .find_map(|line| line.trim().strip_prefix("\"acct\"<blob>="))?;
    serde_json::from_str(encoded).ok()
}

fn security_quote(value: &str) -> Result<String, String> {
    if value.chars().any(char::is_control) {
        return Err("Invalid Keychain command value".into());
    }
    Ok(format!(
        "\"{}\"",
        value.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}

fn decode_blob(bytes: Vec<u8>) -> Result<String, String> {
    if bytes.len() > MAX_BLOB {
        return Err("Claude Code credentials exceed the size limit".into());
    }
    String::from_utf8(bytes).map_err(|_| "Claude Code credentials are not UTF-8".into())
}

/// No shell, no inherited stdin, no stderr/secret echo, bounded output and lifetime.
fn run_command(
    program: &Path,
    args: &[&str],
    input: Option<String>,
    timeout: Duration,
) -> Result<(Option<i32>, Vec<u8>), String> {
    #[cfg(windows)]
    let resolved = resolve_windows_program(program);
    #[cfg(windows)]
    let program = resolved.as_path();
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let mut child = command.spawn().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            "Command not installed".to_string()
        } else {
            "Cannot start credential helper".to_string()
        }
    })?;
    if let (Some(mut pipe), Some(input)) = (child.stdin.take(), input) {
        std::thread::spawn(move || {
            let _ = pipe.write_all(input.as_bytes());
        });
    }
    let mut stdout = child
        .stdout
        .take()
        .ok_or("Cannot capture credential helper")?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout
            .by_ref()
            .take(MAX_BLOB as u64 + 1)
            .read_to_end(&mut bytes);
        let _ = tx.send(result.map(|_| bytes));
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Credential helper timed out or failed".into());
            }
        }
    };
    let bytes = rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_| "Credential helper output timed out")?
        .map_err(|_| "Cannot read credential helper output")?;
    if bytes.len() > MAX_BLOB {
        return Err("Credential helper output exceeds the size limit".into());
    }
    Ok((status.code(), bytes))
}

#[cfg(windows)]
fn resolve_windows_program(program: &Path) -> PathBuf {
    if program.is_absolute() || program.components().count() != 1 || program.extension().is_some() {
        return program.to_path_buf();
    }
    // Rust's native lookup adds .exe only; npm installations expose claude.cmd.
    if let Some(paths) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&paths).filter(|path| path.is_absolute()) {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = directory.join(program).with_extension(extension);
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    program.to_path_buf()
}

impl AuthStorage {
    fn claude_snapshot(&self, provider: &str, force: bool) -> SnapshotResult {
        if provider != "anthropic" {
            return Ok(None);
        }
        self.claude_code
            .as_ref()
            .map(|auth| auth.snapshot(force))
            .unwrap_or(Ok(None))
    }

    fn candidate_for_claude(&self, snapshot: &Snapshot) -> AuthSourceCandidate {
        self.create_auth_source_candidate(
            AUTH_SOURCE_CLAUDE_CODE,
            true,
            &snapshot.source.identity(),
            Some(&format!(
                "{}\0{}\0{}",
                snapshot.credentials.access,
                snapshot.credentials.refresh,
                snapshot.credentials.expires
            )),
            Some(&format!("Claude Code {} OAuth", snapshot.version)),
            None,
        )
    }

    pub(super) fn get_claude_code_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        match self.claude_snapshot(provider, false) {
            Ok(Some(snapshot)) => Some(self.candidate_for_claude(&snapshot)),
            Ok(None) => None,
            Err(_) => Some(self.create_auth_source_candidate(
                AUTH_SOURCE_CLAUDE_CODE,
                false,
                "unreadable",
                Some("unreadable"),
                Some("Claude Code OAuth unavailable"),
                None,
            )),
        }
    }

    pub(crate) fn has_claude_code_auth(&self, provider: &str) -> bool {
        self.get_claude_code_candidate(provider).is_some()
    }

    pub(super) async fn resolve_claude_code_auth(
        &self,
        provider: &str,
    ) -> Result<Option<AuthApiKeyResult>, String> {
        if provider != "anthropic" {
            return Ok(None);
        }
        let Some(auth) = &self.claude_code else {
            return Ok(None);
        };
        let auth = std::sync::Arc::clone(auth);
        // A cancelled model request must not discard already-rotated credentials.
        let result = tokio::spawn(async move { auth.resolve().await })
            .await
            .map_err(|_| "Claude Code credential resolution failed")?;
        let Some(snapshot) = result.map_err(|error| format!("{error}. {LOGIN_HELP}"))? else {
            return Ok(None);
        };
        let candidate = self.candidate_for_claude(&snapshot);
        if self.is_auth_source_stale(provider, &candidate) {
            return Err(LOGIN_HELP.into());
        }
        Ok(Some(AuthApiKeyResult {
            api_key: Some(snapshot.credentials.access.clone()),
            source_token: self.get_auth_source_token_for_candidate(provider, &candidate),
        }))
    }

    pub(crate) fn validate_claude_code_endpoint(
        &self,
        model: &pi_ai::types::Model,
    ) -> Result<(), String> {
        if !self.has_claude_code_auth(&model.provider) {
            return Ok(());
        }
        let valid = url::Url::parse(&model.base_url).ok().is_some_and(|url| {
            model.api == "anthropic-messages"
                && url.scheme() == "https"
                && url.host_str() == Some("api.anthropic.com")
                && url.port_or_known_default() == Some(443)
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && matches!(url.path().trim_end_matches('/'), "" | "/v1")
        });
        if valid {
            Ok(())
        } else {
            Err("Claude Code OAuth can only be used with Anthropic's direct Messages endpoint. For a custom endpoint, set OPTIMUS_CLAUDE_CODE_AUTH=0 and configure its credentials explicitly.".into())
        }
    }

    pub(crate) fn pin_claude_code_headers(
        &self,
        token: Option<&AuthSourceToken>,
        headers: &mut Option<IndexMap<String, String>>,
    ) {
        if token.is_none_or(|token| token.source != AUTH_SOURCE_CLAUDE_CODE) {
            return;
        }
        let Some(auth) = &self.claude_code else {
            return;
        };
        let Some(version) = auth
            .last_version
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
        else {
            return;
        };
        let headers = headers.get_or_insert_with(IndexMap::new);
        headers.retain(|name, _| {
            !["user-agent", "authorization", "x-api-key", "x-app"]
                .iter()
                .any(|reserved| name.eq_ignore_ascii_case(reserved))
        });
        headers.insert("user-agent".into(), format!("claude-cli/{version}"));
        headers.insert("x-app".into(), "cli".into());
    }
}

#[cfg(test)]
#[path = "claude_code_auth_tests.rs"]
mod tests;
