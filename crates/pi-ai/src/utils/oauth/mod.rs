//! Port of packages/ai/src/utils/oauth/index.ts
//!
//! OAuth credential management for AI providers.
//!
//! This module handles login, token refresh, and credential storage
//! for OAuth-based providers:
//! - Anthropic (Claude Pro/Max)
//! - GitHub Copilot

pub mod anthropic;
pub mod github_copilot;
pub mod oauth_page;
pub mod openai_codex;
pub mod pkce;
pub mod types;

pub use anthropic::{anthropic_oauth_provider, login_anthropic, refresh_anthropic_token};
pub use github_copilot::{
    get_github_copilot_base_url, github_copilot_oauth_provider, login_github_copilot, normalize_domain,
    refresh_github_copilot_token,
};
pub use openai_codex::{login_openai_codex, openai_codex_oauth_provider, refresh_openai_codex_token};
pub use types::*;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

fn oauth_provider_registry() -> &'static Mutex<HashMap<String, OAuthProviderInterface>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, OAuthProviderInterface>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = HashMap::new();
        for provider in built_in_oauth_providers() {
            registry.insert(provider.id.clone(), provider);
        }
        Mutex::new(registry)
    })
}

fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<String, OAuthProviderInterface>> {
    oauth_provider_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn built_in_oauth_providers() -> Vec<OAuthProviderInterface> {
    vec![
        crate::providers::kiro::auth::oauth_provider(),
        anthropic::anthropic_oauth_provider(),
        github_copilot::github_copilot_oauth_provider(),
        openai_codex::openai_codex_oauth_provider(),
    ]
}

pub fn get_oauth_provider(id: &str) -> Option<OAuthProviderInterface> {
    lock_registry().get(id).cloned()
}

pub fn register_oauth_provider(provider: OAuthProviderInterface) {
    lock_registry().insert(provider.id.clone(), provider);
}

/// Unregister an OAuth provider.
///
/// If the provider is built-in, restores the built-in implementation.
/// Custom providers are removed completely.
pub fn unregister_oauth_provider(id: &str) {
    let built_in_provider = built_in_oauth_providers()
        .into_iter()
        .find(|provider| provider.id == id);
    if let Some(built_in_provider) = built_in_provider {
        lock_registry().insert(id.to_string(), built_in_provider);
        return;
    }
    lock_registry().remove(id);
}

pub fn reset_oauth_providers() {
    let mut registry = lock_registry();
    registry.clear();
    for provider in built_in_oauth_providers() {
        registry.insert(provider.id.clone(), provider);
    }
}

pub fn get_oauth_providers() -> Vec<OAuthProviderInterface> {
    lock_registry().values().cloned().collect()
}

/// @deprecated Use get_oauth_providers() which returns OAuthProviderInterface[]
pub fn get_oauth_provider_info_list() -> Vec<OAuthProviderInfo> {
    get_oauth_providers()
        .into_iter()
        .map(|provider| OAuthProviderInfo {
            id: provider.id,
            name: provider.name,
            available: true,
        })
        .collect()
}

/// Refresh token for any OAuth provider.
/// @deprecated Use get_oauth_provider(id).refresh_token() instead
pub async fn refresh_oauth_token(
    provider_id: &str,
    credentials: OAuthCredentials,
) -> Result<OAuthCredentials, String> {
    let Some(provider) = get_oauth_provider(provider_id) else {
        return Err(format!("Unknown OAuth provider: {}", provider_id));
    };
    (provider.refresh_token)(credentials).await
}

/// Get API key for a provider from OAuth credentials.
/// Automatically refreshes expired tokens.
///
/// @returns API key string and updated credentials, or null if no credentials
/// @throws Error if refresh fails
pub async fn get_oauth_api_key(
    provider_id: &str,
    credentials: &HashMap<String, OAuthCredentials>,
) -> Result<Option<OAuthApiKey>, String> {
    let Some(provider) = get_oauth_provider(provider_id) else {
        return Err(format!("Unknown OAuth provider: {}", provider_id));
    };

    let Some(mut creds) = credentials.get(provider_id).cloned() else {
        return Ok(None);
    };

    if crate::utils::now_ms() as f64 >= creds.expires {
        match (provider.refresh_token)(creds.clone()).await {
            Ok(refreshed) => creds = refreshed,
            Err(_) => {
                return Err(format!("Failed to refresh OAuth token for {}", provider_id));
            }
        }
    }

    let api_key = (provider.get_api_key)(&creds);
    Ok(Some(OAuthApiKey {
        new_credentials: creds,
        api_key,
    }))
}

#[derive(Debug, Clone, PartialEq)]
pub struct OAuthApiKey {
    pub new_credentials: OAuthCredentials,
    pub api_key: String,
}

/// Helper kept for callers that store credentials as raw JSON.
pub fn credentials_from_value(value: &Value) -> Option<OAuthCredentials> {
    serde_json::from_value(value.clone()).ok()
}


/// Private plumbing that replaces the Node `http.createServer` callback server
/// used by the OAuth flows. Not a public API: it exists only because Rust links
/// the HTTP server at build time instead of `import("node:http")`.
pub(crate) mod plumbing {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A one-shot slot that settles with a value or a cancellation
    /// (`settleWait` / `cancelWait` / `waitForCode` in the TypeScript).
    pub(crate) struct CallbackSlot<T> {
        result: Arc<Mutex<Option<T>>>,
        notify: Arc<tokio::sync::Notify>,
        cancelled: Arc<AtomicBool>,
    }

    impl<T> Clone for CallbackSlot<T> {
        fn clone(&self) -> Self {
            Self {
                result: self.result.clone(),
                notify: self.notify.clone(),
                cancelled: self.cancelled.clone(),
            }
        }
    }

    impl<T> Default for CallbackSlot<T> {
        fn default() -> Self {
            Self {
                result: Arc::new(Mutex::new(None)),
                notify: Arc::new(tokio::sync::Notify::new()),
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl<T: Clone + Send + 'static> CallbackSlot<T> {
        /// `settleWait(value)` - first value wins.
        pub(crate) fn settle(&self, value: Option<T>) {
            {
                let mut result = self.result.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if result.is_none() {
                    *result = value;
                }
            }
            self.notify.notify_waiters();
        }

        /// `cancelWait()` - settles with `null`.
        pub(crate) fn cancel(&self) {
            self.cancelled.store(true, Ordering::SeqCst);
            self.settle(None);
        }

        /// `waitForCode()`.
        pub(crate) async fn wait(&self) -> Option<T> {
            loop {
                let notified = self.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                {
                    let result = self.result.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    if result.is_some() {
                        return result.clone();
                    }
                }
                if self.cancelled.load(Ordering::SeqCst) {
                    return None;
                }
                notified.await;
            }
        }
    }

    /// Serve one HTTP request at a time on `listener` and reply with the
    /// handler's `(status, body)`, mirroring `http.createServer((req, res) => ...)`.
    pub(crate) fn spawn_http_callback_server<F>(
        listener: TcpListener,
        handler: F,
    ) -> tokio::task::JoinHandle<()>
    where
        F: Fn(&str, &HashMap<String, String>) -> (u16, String) + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let handler = handler.clone();
                let mut buffer = vec![0u8; 8192];
                let Ok(read) = socket.read(&mut buffer).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                let request_line = request.lines().next().unwrap_or_default().to_string();
                let target = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (path, query) = match target.split_once('?') {
                    Some((path, query)) => (path.to_string(), query.to_string()),
                    None => (target.clone(), String::new()),
                };
                let params: HashMap<String, String> = url::form_urlencoded::parse(query.as_bytes())
                    .map(|(key, value)| (key.to_string(), value.to_string()))
                    .collect();
                let (status, body) = handler(&path, &params);
                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    if status == 200 { "OK" } else { "Bad Request" },
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        })
    }

    /// Bind the callback listener on `host:port` (Node `server.listen`).
    pub(crate) async fn bind_callback_listener(
        host: &str,
        port: u16,
    ) -> std::io::Result<TcpListener> {
        TcpListener::bind((host, port)).await
    }

    /// `process.env.PI_OAUTH_CALLBACK_HOST || "127.0.0.1"`
    pub(crate) fn oauth_callback_host() -> String {
        std::env::var("PI_OAUTH_CALLBACK_HOST")
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "127.0.0.1".to_string())
    }

    /// `const decode = (s: string) => atob(s)`.
    pub(crate) fn decode_base64(value: &str) -> String {
        use base64::Engine;
        let padded = match value.len() % 4 {
            2 => format!("{}==", value),
            3 => format!("{}=", value),
            _ => value.to_string(),
        };
        base64::engine::general_purpose::STANDARD
            .decode(padded)
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_providers_are_registered() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        reset_oauth_providers();
        let ids: Vec<String> = get_oauth_providers().into_iter().map(|provider| provider.id).collect();
        assert!(ids.contains(&"anthropic".to_string()));
        assert!(ids.contains(&"github-copilot".to_string()));
        assert!(ids.contains(&"openai-codex".to_string()));
        assert!(get_oauth_provider("anthropic").unwrap().uses_callback_server == Some(true));
    }

    #[test]
    fn unregister_restores_built_in_and_drops_custom() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        reset_oauth_providers();
        let custom = OAuthProviderInterface {
            id: "custom".to_string(),
            name: "Custom".to_string(),
            login: std::sync::Arc::new(|_| Box::pin(async { Ok(OAuthCredentials::default()) })),
            uses_callback_server: None,
            refresh_token: std::sync::Arc::new(|credentials| Box::pin(async move { Ok(credentials) })),
            get_api_key: std::sync::Arc::new(|credentials: &OAuthCredentials| credentials.access.clone()),
            modify_models: None,
        };
        register_oauth_provider(custom);
        assert!(get_oauth_provider("custom").is_some());
        unregister_oauth_provider("custom");
        assert!(get_oauth_provider("custom").is_none());

        let built_in = get_oauth_provider("anthropic").unwrap();
        unregister_oauth_provider("anthropic");
        let restored = get_oauth_provider("anthropic").unwrap();
        assert_eq!(restored.name, built_in.name);
        reset_oauth_providers();
    }

    #[test]
    fn provider_info_list_reports_available() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        reset_oauth_providers();
        let list = get_oauth_provider_info_list();
        assert!(list.iter().all(|info| info.available));
        assert!(list.iter().any(|info| info.id == "openai-codex"));
    }

    #[tokio::test]
    async fn get_oauth_api_key_returns_none_without_credentials() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        reset_oauth_providers();
        let credentials = HashMap::new();
        assert!(get_oauth_api_key("anthropic", &credentials).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_oauth_api_key_uses_stored_credentials_when_unexpired() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        reset_oauth_providers();
        let mut credentials = HashMap::new();
        credentials.insert(
            "anthropic".to_string(),
            OAuthCredentials {
                refresh: "r".to_string(),
                access: "access-token".to_string(),
                expires: (crate::utils::now_ms() as f64) + 3_600_000.0,
                extra: serde_json::Map::new(),
            },
        );
        let result = get_oauth_api_key("anthropic", &credentials).await.unwrap().unwrap();
        assert_eq!(result.api_key, "access-token");
        assert_eq!(result.new_credentials.access, "access-token");
    }

    #[tokio::test]
    async fn unknown_provider_errors() {
        let _providers = crate::test_env::ScopedOAuthProviders::new();
        let credentials = HashMap::new();
        assert_eq!(
            get_oauth_api_key("nope", &credentials).await.unwrap_err(),
            "Unknown OAuth provider: nope"
        );
        assert_eq!(
            refresh_oauth_token("nope", OAuthCredentials::default()).await.unwrap_err(),
            "Unknown OAuth provider: nope"
        );
    }
}
