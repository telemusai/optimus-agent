//! Ambient Kiro auth is enabled only for the normal profile, after explicit keys.
use super::*;
use pi_ai::providers::kiro::auth::{read_cli, resolve_cli, Snapshot};
const SOURCE: &str = "kiro_cli";

impl AuthStorage {
    fn candidate_for_kiro(&self, snapshot: &Snapshot) -> AuthSourceCandidate {
        self.create_auth_source_candidate(
            SOURCE,
            false,
            snapshot
                .credential
                .profile_arn
                .as_deref()
                .unwrap_or("kiro-cli"),
            Some(&snapshot.credential.access),
            Some("Kiro CLI"),
            None,
        )
    }

    pub(super) fn get_kiro_cli_candidate(&self, provider: &str) -> Option<AuthSourceCandidate> {
        if provider != "kiro" {
            return None;
        }
        let snapshot = read_cli(self.kiro_cli_path.as_ref()?).ok()??;
        Some(self.candidate_for_kiro(&snapshot))
    }

    pub(super) async fn resolve_kiro_cli_auth(
        &self,
        provider: &str,
    ) -> Result<Option<AuthApiKeyResult>, String> {
        if provider != "kiro" {
            return Ok(None);
        }
        let Some(path) = self.kiro_cli_path.as_ref() else {
            return Ok(None);
        };
        let Some(snapshot) = read_cli(path)? else {
            return Ok(None);
        };
        if self.is_auth_source_stale(provider, &self.candidate_for_kiro(&snapshot)) {
            return Ok(None);
        }
        let snapshot = resolve_cli(path).await?;
        let candidate = self.candidate_for_kiro(&snapshot);
        Ok(Some(AuthApiKeyResult {
            api_key: Some(snapshot.credential.encode()),
            source_token: self.get_auth_source_token_for_candidate(provider, &candidate),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kiro_auth_uses_synthetic_profile_in_isolated_process() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "core::auth_storage::kiro_cli::tests::kiro_auth_fixture",
                "--ignored",
            ])
            .env_remove("KIRO_API_KEY")
            .env_remove("KIRO_CLI_DB_FILE")
            .env("OPTIMUS_KIRO_CLI_AUTH", "0")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    #[ignore = "parent test runs this with isolated auth environment"]
    async fn kiro_auth_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("data.sqlite3");
        let conn = rusqlite::Connection::open(&database).unwrap();
        conn.execute_batch("CREATE TABLE auth_kv(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE state(key TEXT PRIMARY KEY,value BLOB);").unwrap();
        conn.execute(
            "INSERT INTO auth_kv VALUES (?1, ?2)",
            rusqlite::params![
                "kirocli:odic:token",
                json!({"access_token":"fixture-token","expires_at":"2099-01-01T00:00:00Z"})
                    .to_string()
            ],
        )
        .unwrap();
        let auth_file = dir.path().join("auth.json").to_string_lossy().to_string();
        let mut storage = AuthStorage::create(Some(auth_file), None);
        assert!(
            storage.kiro_cli_path.is_none(),
            "isolated profiles must not borrow host credentials"
        );
        assert!(AuthStorage::in_memory(Default::default(), None)
            .kiro_cli_path
            .is_none());
        storage.kiro_cli_path = Some(database.clone());
        let before = std::fs::read(&database).unwrap();
        assert_eq!(
            storage.get_auth_status("kiro").source.as_deref(),
            Some("kiro_cli")
        );
        let resolved = storage
            .get_api_key_with_source_token("kiro", true)
            .await
            .unwrap();
        assert!(resolved
            .api_key
            .unwrap()
            .starts_with(pi_ai::providers::kiro::auth::ENVELOPE));
        assert!(
            storage.get_all().is_empty(),
            "ambient credentials are never copied to auth.json"
        );
        assert_eq!(std::fs::read(&database).unwrap(), before);
        storage.set_runtime_api_key("kiro", "ksk_runtime");
        assert_eq!(
            storage.get_api_key("kiro", true).await.unwrap().as_deref(),
            Some("ksk_runtime")
        );
        storage.remove_runtime_api_key("kiro");
        storage.set(
            "kiro",
            AuthCredential::ApiKey {
                key: "ksk_stored".into(),
                prime_team: None,
            },
        );
        assert_eq!(
            storage.get_api_key("kiro", true).await.unwrap().as_deref(),
            Some("ksk_stored")
        );
        storage.remove("kiro");
        assert!(storage.mark_auth_stale("kiro"));
        assert!(!storage.has_auth("kiro"));
        assert!(storage.get_api_key("kiro", true).await.unwrap().is_none());
        conn.execute(
            "UPDATE auth_kv SET value=?1",
            [
                json!({"access_token":"rotated-fixture-token","expires_at":"2099-01-01T00:00:00Z"})
                    .to_string(),
            ],
        )
        .unwrap();
        assert!(
            storage.has_auth("kiro"),
            "a fresh owner login recovers a stale source"
        );
        storage.login("kiro", Default::default()).await.unwrap();
        assert!(storage.get_all().is_empty());
        assert!(storage
            .logout("kiro")
            .unwrap_err()
            .contains("kiro-cli logout"));
    }
}
