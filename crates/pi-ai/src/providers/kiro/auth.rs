//! Kiro CLI owns refresh tokens. Read its database without copying or rotating them.
use crate::utils::oauth::types::{OAuthCredentials, OAuthProviderInterface};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

pub const ENVELOPE: &str = "optimus-kiro-oauth:";

#[derive(Clone, Serialize, Deserialize)]
pub struct Credential {
    pub access: String,
    pub region: String,
    #[serde(default)]
    pub profile_arn: Option<String>,
}

impl Credential {
    pub fn encode(&self) -> String {
        format!(
            "{ENVELOPE}{}",
            serde_json::to_string(self).expect("Kiro credential serialization")
        )
    }
}

pub struct Snapshot {
    pub credential: Credential,
    pub expires: i64,
}

pub fn valid_region(region: &str) -> bool {
    !region.is_empty()
        && region.len() <= 32
        && region
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

pub fn default_cli_path() -> Option<PathBuf> {
    if std::env::var("OPTIMUS_KIRO_CLI_AUTH").ok().as_deref() == Some("0") {
        return None;
    }
    if let Some(path) = std::env::var_os("KIRO_CLI_DB_FILE").filter(|s| !s.is_empty()) {
        return Some(path.into());
    }
    native_cli_path()
}

fn native_cli_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let mut paths = vec![];
    if let Some(data) = dirs::data_local_dir() {
        paths.push(data.join("kiro-cli/data.sqlite3"));
    }
    paths.push(home.join(".local/share/kiro-cli/data.sqlite3"));
    paths.push(home.join("Library/Application Support/kiro-cli/data.sqlite3"));
    paths.into_iter().find(|path| path.is_file())
}

pub fn read_cli(path: &Path) -> Result<Option<Snapshot>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|_| "Cannot open Kiro CLI credential database".to_string())?;
    conn.busy_timeout(Duration::from_millis(250))
        .map_err(|_| "Kiro CLI database is busy")?;
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM auth_kv WHERE key = ?1",
            ["kirocli:odic:token"],
            |row| row.get(0),
        )
        .optional()
        .map_err(|_| "Cannot read Kiro CLI login; use `kiro-cli login`")?;
    let Some(raw) = raw else { return Ok(None) };
    if raw.len() > 65_536 {
        return Err("Kiro CLI credential record is too large".into());
    }
    let token: Value =
        serde_json::from_str(&raw).map_err(|_| "Invalid Kiro CLI credential record")?;
    let access = token["access_token"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("Kiro CLI login has no access token")?
        .to_string();
    let expires = chrono::DateTime::parse_from_rfc3339(token["expires_at"].as_str().unwrap_or(""))
        .map_err(|_| "Kiro CLI login has no valid expiry")?
        .timestamp_millis();
    let profile: Option<String> = conn
        .query_row(
            "SELECT CAST(value AS TEXT) FROM state WHERE key = ?1",
            ["api.codewhisperer.profile"],
            |row| row.get(0),
        )
        .optional()
        .map_err(|_| "Cannot read Kiro CLI profile")?;
    let profile_arn = profile
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v["arn"].as_str().map(str::to_owned));
    // The SSO region can differ from the Kiro service region (e.g. ap-southeast-2).
    let region = profile_arn
        .as_deref()
        .and_then(|s| s.split(':').nth(3))
        .filter(|s| valid_region(s))
        .map(str::to_owned)
        .unwrap_or_else(|| "us-east-1".into());
    Ok(Some(Snapshot {
        credential: Credential {
            access,
            region,
            profile_arn,
        },
        expires,
    }))
}

pub async fn resolve_cli(path: &Path) -> Result<Snapshot, String> {
    resolve_with_refresh(path, async {
        // A custom database may belong to another account. Never refresh the default
        // CLI account on behalf of an unrelated credential source.
        let native = native_cli_path().and_then(|p| p.canonicalize().ok());
        if native.is_none() || native != path.canonicalize().ok() {
            return Err("Refresh this custom Kiro CLI login in its owning CLI profile".into());
        }
        let mut command = tokio::process::Command::new("kiro-cli");
        command
            .args(["chat", "--list-models", "--format", "json"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        command.creation_flags(0x08000000);
        let status = command
            .status()
            .await
            .map_err(|_| "Cannot refresh Kiro CLI login; run `kiro-cli login`")?;
        if !status.success() {
            return Err("Kiro CLI login refresh failed; run `kiro-cli login`".into());
        }
        Ok(())
    })
    .await
}

async fn resolve_with_refresh(
    path: &Path,
    refresh: impl std::future::Future<Output = Result<(), String>>,
) -> Result<Snapshot, String> {
    let snapshot = read_cli(path)?.ok_or("No Kiro CLI login; run `kiro-cli login`")?;
    if snapshot.expires > crate::utils::now_ms() + 60_000 {
        return Ok(snapshot);
    }
    tokio::time::timeout(Duration::from_secs(20), refresh)
        .await
        .map_err(|_| "Kiro CLI login refresh timed out; run `kiro-cli login`")??;
    let snapshot = read_cli(path)?.ok_or("Kiro CLI login is no longer available")?;
    if snapshot.expires <= crate::utils::now_ms() + 60_000 {
        return Err("Kiro CLI login has expired; run `kiro-cli login`".into());
    }
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn refresh_rereads_owner_token_and_rejects_unchanged_expired_login() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.sqlite3");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE auth_kv(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE state(key TEXT PRIMARY KEY,value BLOB);").unwrap();
        conn.execute(
            "INSERT INTO auth_kv VALUES (?1,?2)",
            rusqlite::params![
                "kirocli:odic:token",
                json!({"access_token":"expired-fixture","expires_at":"2000-01-01T00:00:00Z"})
                    .to_string()
            ],
        )
        .unwrap();
        assert!(resolve_with_refresh(&path, async { Ok(()) })
            .await
            .err()
            .unwrap()
            .contains("expired"));
        assert_eq!(
            resolve_with_refresh(&path, async { Err("owner failure".into()) })
                .await
                .err()
                .unwrap(),
            "owner failure"
        );
        let snapshot = resolve_with_refresh(&path, async {
            conn.execute(
                "UPDATE auth_kv SET value=?1",
                [
                    json!({"access_token":"fresh-fixture","expires_at":"2099-01-01T00:00:00Z"})
                        .to_string(),
                ],
            )
            .unwrap();
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(snapshot.credential.access, "fresh-fixture");
        let snapshot =
            resolve_with_refresh(&path, async { panic!("fresh login must not invoke CLI") })
                .await
                .unwrap();
        assert_eq!(snapshot.credential.access, "fresh-fixture");
    }
}

fn oauth_credentials(snapshot: Snapshot, path: &Path) -> OAuthCredentials {
    OAuthCredentials {
        access: snapshot.credential.encode(),
        refresh: "managed-by-kiro-cli".into(),
        expires: (snapshot.expires - 60_000) as f64,
        extra: json!({"kiroCliDatabase":path}).as_object().unwrap().clone(),
    }
}

pub fn oauth_provider() -> OAuthProviderInterface {
    OAuthProviderInterface {
        id: "kiro".into(),
        name: "Kiro (existing CLI login)".into(),
        uses_callback_server: Some(false),
        login: Arc::new(|callbacks| {
            Box::pin(async move {
                let path = default_cli_path()
                    .ok_or("Run `kiro-cli login` first, or configure KIRO_API_KEY")?;
                let signal = callbacks.signal.unwrap_or_default();
                let snapshot = tokio::select! {
                    biased;
                    _ = signal.cancelled() => return Err("Kiro login cancelled".into()),
                    result = resolve_cli(&path) => result?,
                };
                Ok(oauth_credentials(snapshot, &path))
            })
        }),
        refresh_token: Arc::new(|credentials| {
            Box::pin(async move {
                let path = credentials
                    .extra
                    .get("kiroCliDatabase")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
                    .ok_or("Missing Kiro CLI database; run /login kiro")?;
                Ok(oauth_credentials(resolve_cli(&path).await?, &path))
            })
        }),
        get_api_key: Arc::new(|credentials| credentials.access.clone()),
        modify_models: None,
    }
}
