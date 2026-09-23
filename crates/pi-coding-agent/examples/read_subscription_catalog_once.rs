//! Explicitly gated Copilot metadata read. Never part of normal tests.
//! Cached-only by default; a separate gate allows one normal provider refresh in memory.
//! No login, auth persistence, inference, catalog redirects/retries, or raw error logging.
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_ai::copilot_client_version::COPILOT_CLIENT_HEADERS;
use pi_ai::types::BoxFuture;
use pi_ai::utils::oauth::github_copilot::{
    get_github_copilot_base_url, github_copilot_oauth_provider,
};
use pi_ai::utils::oauth::types::{OAuthCredentials, OAuthProviderInterface};
use pi_coding_agent::core::auth_storage::{
    AuthCredential, AuthStorage, AuthStorageBackend, AuthStorageOptions, LockFn,
};
use serde_json::{json, Map, Value};

struct ReadOnlyAuth(PathBuf);

impl AuthStorageBackend for ReadOnlyAuth {
    fn with_lock(
        &self,
        callback: &mut dyn FnMut(Option<String>) -> Result<Option<String>, String>,
    ) -> Result<(), String> {
        let raw = std::fs::read_to_string(&self.0)
            .map_err(|_| "cached auth unavailable".to_string())?;
        let next = callback(Some(raw))?;
        if next.is_some() {
            return Err("auth writes prohibited".to_string());
        }
        Ok(())
    }

    fn with_lock_async(&self, _callback: LockFn) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Err("async auth and refresh prohibited".to_string()) })
    }
}

fn selected_metadata(value: &Value) -> Result<Vec<Value>, &'static str> {
    let rows = value.get("data").and_then(Value::as_array).ok_or("invalid_catalog_shape")?;
    let mut selected = Vec::new();
    for row in rows {
        let id = row.get("id").and_then(Value::as_str).unwrap_or("");
        let name = row.get("name").and_then(Value::as_str).unwrap_or("");
        let normalized_name = name.to_ascii_lowercase().replace(' ', "-");
        if !matches!(id, "gpt-6-sol" | "gpt-6-luna")
            && !matches!(normalized_name.as_str(), "gpt-6-sol" | "gpt-6-luna") {
            continue;
        }
        if id.is_empty() || id.len() > 128
            || !id.bytes().all(|c| c.is_ascii_alphanumeric() || b"-._:/".contains(&c)) {
            return Err("invalid_selected_model_id");
        }
        let mut output = Map::new();
        output.insert("id".to_string(), json!(id));
        for field in ["model_picker_enabled", "preview"] {
            if let Some(value) = row.get(field).and_then(Value::as_bool) {
                output.insert(field.to_string(), json!(value));
            }
        }
        if let Some(state @ ("enabled" | "disabled" | "unconfigured")) = row.pointer("/policy/state").and_then(Value::as_str) {
            output.insert("policyState".to_string(), json!(state));
        }
        let mut limits = Map::new();
        for field in ["max_context_window_tokens", "max_prompt_tokens", "max_output_tokens"] {
            if let Some(value) = row.pointer(&format!("/capabilities/limits/{field}")).and_then(Value::as_u64) {
                limits.insert(field.to_string(), json!(value));
            }
        }
        output.insert("limits".to_string(), json!(limits));
        let mut supports = Map::new();
        for field in ["vision", "tool_calls", "parallel_tool_calls", "streaming"] {
            if let Some(value) = row.pointer(&format!("/capabilities/supports/{field}")).and_then(Value::as_bool) {
                supports.insert(field.to_string(), json!(value));
            }
        }
        if let Some(levels) = row.pointer("/capabilities/supports/reasoning_effort").and_then(Value::as_array) {
            let levels: Vec<&str> = levels.iter().filter_map(Value::as_str)
                .filter(|level| matches!(*level, "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra"))
                .collect();
            supports.insert("reasoning_effort".to_string(), json!(levels));
        }
        output.insert("supports".to_string(), json!(supports));
        if let Some(endpoints) = row.get("supported_endpoints").and_then(Value::as_array) {
            let endpoints: Vec<&str> = endpoints.iter().filter_map(Value::as_str)
                .filter(|path| matches!(*path, "/responses" | "/chat/completions" | "/v1/messages"))
                .collect();
            output.insert("supported_endpoints".to_string(), json!(endpoints));
        }
        selected.push(json!(output));
    }
    Ok(selected)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogMode { CachedOnly, RefreshOnce }

fn parse_gate(value: &str) -> Option<CatalogMode> {
    match value {
        "github-copilot" => Some(CatalogMode::CachedOnly),
        "github-copilot-refresh-once" => Some(CatalogMode::RefreshOnce),
        _ => None,
    }
}

fn credentials_current(credentials: &OAuthCredentials) -> bool {
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    credentials.expires.is_finite() && credentials.expires > (now_ms + 60_000) as f64
}

fn credential_failure(status: &str, refresh_attempted: bool) -> Value {
    json!({"status": status, "networkAttempted": refresh_attempted,
        "refreshAttempted": refresh_attempted, "catalogAttempted": false})
}

async fn prepare_credentials(
    credentials: OAuthCredentials,
    mode: CatalogMode,
    receipt: &mut std::fs::File,
    provider: &OAuthProviderInterface,
) -> Result<OAuthCredentials, Value> {
    use std::io::Write;
    if mode == CatalogMode::CachedOnly {
        return if credentials_current(&credentials) { Ok(credentials) }
            else { Err(credential_failure("cached_oauth_expired_no_refresh", false)) };
    }
    if credentials.refresh.is_empty() {
        return Err(credential_failure("cached_refresh_token_missing", false));
    }
    // The reviewed normal public-subscription token endpoint is api.github.com.
    // Do not infer an enterprise refresh endpoint from an unreviewed cached value.
    if credentials.extra.get("enterpriseUrl").is_some_and(|value|
        value.as_str().map_or(true, |domain| domain != "github.com")) {
        return Err(credential_failure("unsupported_cached_refresh_endpoint", false));
    }
    if writeln!(receipt, "{}", json!({"status": "provider_refresh_started",
        "networkAttempted": true, "refreshAttempted": true, "catalogAttempted": false}))
        .and_then(|_| receipt.sync_all()).is_err() {
        return Err(credential_failure("receipt_write_failed", false));
    }
    // This is the existing provider implementation, called exactly once. It does
    // not run device login, model-policy enablement, or AuthStorage persistence.
    let refreshed = match tokio::time::timeout(Duration::from_secs(20),
        (provider.refresh_token)(credentials)).await {
        Ok(Ok(refreshed)) => refreshed,
        Ok(Err(error)) => {
            let mut failure = credential_failure("provider_refresh_failed", true);
            // Native errors may include response bodies. Keep only a leading
            // valid HTTP status number; never emit the error string itself.
            if let Some(status) = error.split_ascii_whitespace().next()
                .and_then(|value| value.parse::<u16>().ok()).filter(|status| (100..600).contains(status)) {
                failure["httpStatus"] = json!(status);
            }
            return Err(failure);
        }
        Err(_) => return Err(credential_failure("provider_refresh_timed_out", true)),
    };
    if refreshed.access.is_empty() || !credentials_current(&refreshed) {
        return Err(credential_failure("refreshed_oauth_invalid", true));
    }
    Ok(refreshed)
}

async fn read_once(auth_path: PathBuf, receipt: &mut std::fs::File, mode: CatalogMode) -> Value {
    let storage = AuthStorage::from_storage(Box::new(ReadOnlyAuth(auth_path)), Some(AuthStorageOptions {
        prime_cli_config_path: None, use_prime_cli_config: false,
    }));
    let Some(AuthCredential::OAuth { credentials }) = storage.get("github-copilot") else {
        return credential_failure("cached_oauth_missing_or_invalid", false);
    };
    let provider = github_copilot_oauth_provider();
    let credentials = match prepare_credentials(credentials, mode, receipt, &provider).await {
        Ok(credentials) => credentials,
        Err(failure) => return failure,
    };
    let mut result = read_catalog(&credentials, receipt, &provider).await;
    let catalog_attempted = result["networkAttempted"].as_bool().unwrap_or(false);
    let refresh_attempted = mode == CatalogMode::RefreshOnce;
    result["refreshAttempted"] = json!(refresh_attempted);
    result["catalogAttempted"] = json!(catalog_attempted);
    result["networkAttempted"] = json!(refresh_attempted || catalog_attempted);
    result
}

async fn read_catalog(credentials: &OAuthCredentials, receipt: &mut std::fs::File,
    provider: &OAuthProviderInterface) -> Value {
    use std::io::Write;
    let token = (provider.get_api_key)(credentials);
    if token.is_empty() {
        return json!({"status": "cached_oauth_empty", "networkAttempted": false});
    }
    let base = get_github_copilot_base_url(Some(&token),
        credentials.extra.get("enterpriseUrl").and_then(Value::as_str));
    let Ok(base) = url::Url::parse(&base) else {
        return json!({"status": "unsupported_cached_endpoint", "networkAttempted": false});
    };
    let allowed_host = base.host_str().is_some_and(|host|
        host == "api.githubcopilot.com" || host.ends_with(".githubcopilot.com"));
    if base.scheme() != "https" || !allowed_host || !base.username().is_empty()
        || base.password().is_some() || base.port().is_some() || base.query().is_some()
        || base.fragment().is_some() || base.path() != "/" {
        return json!({"status": "unsupported_cached_endpoint", "networkAttempted": false});
    }
    let Ok(endpoint) = base.join("models") else {
        return json!({"status": "unsupported_cached_endpoint", "networkAttempted": false});
    };
    let Ok(client) = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(15)).connect_timeout(Duration::from_secs(10)).build() else {
        return json!({"status": "http_client_unavailable", "networkAttempted": false});
    };
    // create_new receipt (in main) and this durable marker make accidental reruns visible.
    if writeln!(receipt, "{}", json!({"status": "catalog_get_started", "networkAttempted": true}))
        .and_then(|_| receipt.sync_all()).is_err() {
        return json!({"status": "receipt_write_failed", "networkAttempted": false});
    }
    let mut request = client.get(endpoint).bearer_auth(token)
        .header("OpenAI-Intent", "model-access")
        .header("X-GitHub-Api-Version", "2025-05-01")
        .header("Accept", "application/json");
    for (name, value) in COPILOT_CLIENT_HEADERS {
        request = request.header(name, value);
    }
    let Ok(mut response) = request.send().await else {
        return json!({"status": "catalog_transport_failed", "networkAttempted": true});
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return json!({"status": "catalog_http_error", "httpStatus": status, "networkAttempted": true});
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) if body.len() + chunk.len() <= 4 * 1024 * 1024 => body.extend_from_slice(&chunk),
            Ok(Some(_)) => return json!({"status": "catalog_too_large", "networkAttempted": true}),
            Ok(None) => break,
            Err(_) => return json!({"status": "catalog_body_failed", "networkAttempted": true}),
        }
    }
    let Ok(value) = serde_json::from_slice::<Value>(&body) else {
        return json!({"status": "catalog_json_invalid", "networkAttempted": true});
    };
    match selected_metadata(&value) {
        Ok(models) => json!({"status": "catalog_read", "provider": "github-copilot",
            "httpStatus": status, "networkAttempted": true, "selectedModels": models,
            "inferenceTested": false}),
        Err(reason) => json!({"status": reason, "networkAttempted": true}),
    }
}

#[tokio::main]
async fn main() {
    use std::io::Write;
    let mode = std::env::var("OPTIMUS_READONLY_CATALOG_ONCE").ok()
        .as_deref().and_then(parse_gate);
    let Some(mode) = mode else {
        eprintln!("blocked: explicit catalog-read gate absent");
        std::process::exit(2);
    };
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: read_subscription_catalog_once AUTH_PATH NEW_RECEIPT_PATH");
        std::process::exit(2);
    }
    let Ok(mut receipt) = std::fs::OpenOptions::new().write(true).create_new(true).open(&args[1]) else {
        eprintln!("blocked: receipt already exists or cannot be created; do not retry catalog lookup");
        std::process::exit(2);
    };
    let result = read_once(PathBuf::from(&args[0]), &mut receipt, mode).await;
    let _ = writeln!(receipt, "{result}");
    let _ = receipt.sync_all();
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_projection_keeps_only_selected_public_metadata() {
        let input = json!({"account": "PRIVATE", "data": [{
            "id": "gpt-6-sol", "name": "GPT-6 Sol", "model_picker_enabled": true,
            "requestHeaders": {"Authorization": "Bearer PRIVATE"},
            "capabilities": {"limits": {"max_output_tokens": 128000},
                "supports": {"vision": true, "reasoning_effort": ["low", "max", "PRIVATE"]}},
            "supported_endpoints": ["/responses", "https://PRIVATE"]
        }, {"id": "unrelated", "name": "Unrelated", "secret": "PRIVATE"}]});
        let output = selected_metadata(&input).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["id"], "gpt-6-sol");
        assert_eq!(output[0]["limits"]["max_output_tokens"], 128000);
        assert_eq!(output[0]["supports"]["reasoning_effort"], json!(["low", "max"]));
        assert!(!json!(output).to_string().contains("PRIVATE"));
        assert!(selected_metadata(&json!({})).is_err());
    }

    fn expired_fixture() -> OAuthCredentials {
        OAuthCredentials { access: "synthetic-expired".to_string(),
            refresh: "synthetic-refresh".to_string(), expires: 1.0, ..Default::default() }
    }

    #[test]
    fn refresh_requires_a_distinct_explicit_gate() {
        assert_eq!(parse_gate("github-copilot"), Some(CatalogMode::CachedOnly));
        assert_eq!(parse_gate("github-copilot-refresh-once"), Some(CatalogMode::RefreshOnce));
        for invalid in ["", "refresh", "github-copilot-refresh", "true"] {
            assert_eq!(parse_gate(invalid), None);
        }
    }

    #[tokio::test]
    async fn cached_mode_never_calls_refresh() {
        let mut provider = github_copilot_oauth_provider();
        provider.refresh_token = std::sync::Arc::new(|_| panic!("refresh forbidden"));
        let mut receipt = tempfile::tempfile().unwrap();
        let failure = prepare_credentials(expired_fixture(), CatalogMode::CachedOnly,
            &mut receipt, &provider).await.unwrap_err();
        assert_eq!(failure["status"], "cached_oauth_expired_no_refresh");
        assert_eq!(failure["networkAttempted"], false);
        assert_eq!(receipt.metadata().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn refresh_failure_is_once_and_drops_raw_provider_error() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut provider = github_copilot_oauth_provider();
        provider.refresh_token = Arc::new(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err("401 Unauthorized: PRIVATE_TOKEN_AND_BODY".to_string()) })
        });
        let mut receipt = tempfile::tempfile().unwrap();
        let failure = prepare_credentials(expired_fixture(), CatalogMode::RefreshOnce,
            &mut receipt, &provider).await.unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(failure["status"], "provider_refresh_failed");
        assert_eq!(failure["httpStatus"], 401);
        assert_eq!(failure["refreshAttempted"], true);
        assert_eq!(failure["catalogAttempted"], false);
        assert!(!failure.to_string().contains("PRIVATE"));
    }

    #[tokio::test]
    async fn refreshed_credentials_stay_in_memory_after_one_provider_call() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut provider = github_copilot_oauth_provider();
        provider.refresh_token = Arc::new(move |credentials| {
            observed.fetch_add(1, Ordering::SeqCst);
            assert_eq!(credentials.refresh, "synthetic-refresh");
            Box::pin(async {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
                Ok(OAuthCredentials { access: "synthetic-refreshed".to_string(),
                    expires: (now + 3_600_000) as f64, ..expired_fixture() })
            })
        });
        let mut receipt = tempfile::tempfile().unwrap();
        let refreshed = prepare_credentials(expired_fixture(), CatalogMode::RefreshOnce,
            &mut receipt, &provider).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(refreshed.access, "synthetic-refreshed");
        assert!(receipt.metadata().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn refresh_refuses_missing_token_and_unreviewed_endpoint_without_network() {
        let mut provider = github_copilot_oauth_provider();
        provider.refresh_token = std::sync::Arc::new(|_| panic!("refresh forbidden"));
        let mut missing = expired_fixture();
        missing.refresh.clear();
        let mut enterprise = expired_fixture();
        enterprise.extra.insert("enterpriseUrl".to_string(), json!("unreviewed.invalid"));
        for (credentials, expected) in [(missing, "cached_refresh_token_missing"),
            (enterprise, "unsupported_cached_refresh_endpoint")] {
            let mut receipt = tempfile::tempfile().unwrap();
            let failure = prepare_credentials(credentials, CatalogMode::RefreshOnce,
                &mut receipt, &provider).await.unwrap_err();
            assert_eq!(failure["status"], expected);
            assert_eq!(failure["networkAttempted"], false);
            assert_eq!(receipt.metadata().unwrap().len(), 0);
        }
    }

    #[test]
    fn readonly_auth_backend_never_persists_a_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic-auth.json");
        std::fs::write(&path, "{}").unwrap();
        let backend = ReadOnlyAuth(path.clone());
        assert!(backend.with_lock(&mut |_| Ok(Some("mutated".to_string()))).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "{}");
    }
}
