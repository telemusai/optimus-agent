//! Port of packages/ai/src/mcp/oauth.ts
//!
//! Node `http.createServer` becomes a small tokio TCP listener that serves the
//! same HTML pages and the same callback paths. `fetch` becomes `reqwest`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use rand::RngCore;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::utils::oauth::oauth_page::{oauth_error_html, oauth_success_html};
use crate::utils::oauth::pkce::generate_pkce;
use crate::utils::oauth::types::{
    OAuthCredentials, OAuthLoginCallbacks, OAuthProviderInterface, OAuthPrompt,
};

pub const CALLBACK_PATH: &str = "/callback";
pub const CALLBACK_PORT_COUNT: u16 = 10;
pub const TOKEN_EXPIRY_BUFFER_MS: f64 = 5.0 * 60.0 * 1000.0;

/// `process.env.PI_OAUTH_CALLBACK_HOST || "127.0.0.1"`
pub fn callback_host() -> String {
    std::env::var("PI_OAUTH_CALLBACK_HOST")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

/// A range (not one port) so a leaked/concurrent login can't wedge all logins
/// with EADDRINUSE. Distinct from the Anthropic callback port (53692). All
/// candidates are registered as redirect URIs.
pub fn callback_port_base() -> u16 {
    std::env::var("PI_MCP_OAUTH_CALLBACK_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(53700)
}

pub fn callback_ports() -> Vec<u16> {
    let base = callback_port_base();
    (0..CALLBACK_PORT_COUNT)
        .map(|index| base.saturating_add(index))
        .collect()
}

pub fn redirect_uri_for(port: u16) -> String {
    format!("http://localhost:{}{}", port, CALLBACK_PATH)
}

pub fn all_redirect_uris() -> Vec<String> {
    callback_ports().into_iter().map(redirect_uri_for).collect()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AuthServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registration_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes_supported: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProtectedResourceMetadata {
    pub resource: String,
    pub authorization_servers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Discovery {
    pub metadata: AuthServerMetadata,
    pub resource: Option<String>,
    pub issuer: Option<String>,
}

/// `export interface McpOAuthConfig`
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct McpOAuthConfig {
    /// MCP server name; provider id becomes `mcp:<server>`.
    pub server: String,
    /// Human-readable label shown in OAuth UI; defaults to `server`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// MCP resource URL used for protected-resource and authorization-server discovery.
    pub url: String,
    /// Pre-registered client id (servers without DCR, e.g. Slack).
    #[serde(rename = "clientId", skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Requested OAuth scopes; defaults to the server's advertised scopes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes: Option<String>,
}

fn validated_https_url(value: &str, name: &str) -> Result<url::Url, String> {
    let url = url::Url::parse(value).map_err(|_| format!("{} must be an absolute HTTPS URL", name))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(format!(
            "{} must be an absolute HTTPS URL without credentials or a fragment",
            name
        ));
    }
    Ok(url)
}

fn canonical_resource(url: &url::Url) -> String {
    if url.path() == "/" && url.query().is_none() {
        return url.origin().ascii_serialization();
    }
    let mut canonical = format!("{}{}", url.origin().ascii_serialization(), url.path());
    if let Some(query) = url.query() {
        canonical.push('?');
        canonical.push_str(query);
    }
    canonical
}

fn authorization_server_metadata_urls(issuer: &str) -> Result<Vec<String>, String> {
    let url = validated_https_url(issuer, "Authorization server issuer")?;
    if url.query().is_some() {
        return Err("Authorization server issuer must not contain a query string".to_string());
    }
    let path = if url.path() == "/" {
        String::new()
    } else {
        url.path().trim_end_matches('/').to_string()
    };
    let origin = url.origin().ascii_serialization();
    Ok(vec![
        format!("{}/.well-known/oauth-authorization-server{}", origin, path),
        format!("{}{}/.well-known/openid-configuration", origin, path),
    ])
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        // `redirect: "error"` in the TypeScript: never follow a redirect.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())
}

async fn fetch_response(url: &str) -> Result<reqwest::Response, String> {
    http_client()?
        .get(url)
        .send()
        .await
        .map_err(|error| error.to_string())
}

async fn fetch_json(url: &str, init: Option<Value>) -> Result<Value, String> {
    let client = http_client()?;
    let method = init
        .as_ref()
        .and_then(|init| init.get("method"))
        .and_then(Value::as_str)
        .unwrap_or("GET")
        .to_string();
    let mut request = if method == "POST" {
        client.post(url)
    } else {
        client.get(url)
    };
    if let Some(init) = init {
        if let Some(body) = init.get("body").and_then(Value::as_str) {
            request = request.body(body.to_string());
        }
        if let Some(headers) = init.get("headers").and_then(Value::as_object) {
            for (name, value) in headers {
                if let Some(value) = value.as_str() {
                    request = request.header(name.as_str(), value);
                }
            }
        }
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "{} {} failed: {}",
            method,
            url,
            response.status().as_u16()
        ));
    }
    response.json::<Value>().await.map_err(|error| error.to_string())
}

fn authorization_server_metadata(
    value: &Value,
    issuer: &str,
    require_exact_issuer: bool,
) -> Result<AuthServerMetadata, String> {
    let Some(metadata) = value.as_object() else {
        return Err(format!("Authorization server metadata for {} is invalid", issuer));
    };
    let Some(advertised) = metadata.get("issuer").and_then(Value::as_str) else {
        return Err(format!(
            "Authorization server metadata for {} is missing its issuer",
            issuer
        ));
    };
    if require_exact_issuer {
        if advertised != issuer {
            return Err(format!(
                "Authorization server metadata issuer does not exactly match {}",
                issuer
            ));
        }
    } else {
        let advertised_issuer = validated_https_url(advertised, "Authorization server metadata issuer")?;
        let issuer_url = url::Url::parse(issuer).map_err(|error| error.to_string())?;
        if advertised_issuer.origin() != issuer_url.origin() || advertised_issuer.query().is_some() {
            return Err(format!(
                "Origin authorization server metadata issuer must stay on {}",
                issuer_url.origin().ascii_serialization()
            ));
        }
    }
    let (Some(authorization_endpoint), Some(token_endpoint)) = (
        metadata.get("authorization_endpoint").and_then(Value::as_str),
        metadata.get("token_endpoint").and_then(Value::as_str),
    ) else {
        return Err(format!(
            "Authorization server metadata for {} is missing required endpoints",
            issuer
        ));
    };
    validated_https_url(authorization_endpoint, "Authorization endpoint")?;
    validated_https_url(token_endpoint, "Token endpoint")?;
    let registration_endpoint = metadata
        .get("registration_endpoint")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(registration_endpoint) = registration_endpoint.as_deref() {
        validated_https_url(registration_endpoint, "Registration endpoint")?;
    }
    Ok(AuthServerMetadata {
        issuer: advertised.to_string(),
        authorization_endpoint: authorization_endpoint.to_string(),
        token_endpoint: token_endpoint.to_string(),
        registration_endpoint,
        scopes_supported: metadata.get("scopes_supported").and_then(Value::as_array).map(|scopes| {
            scopes
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        }),
    })
}

async fn json_metadata(response: reqwest::Response, url: &str) -> Result<Value, String> {
    let status = response.status().as_u16();
    if status != 200 {
        return Err(format!("GET {} failed: {}", url, status));
    }
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or_default().trim().to_lowercase())
        .unwrap_or_default();
    if content_type != "application/json" {
        return Err(format!("GET {} did not return application/json", url));
    }
    response.json::<Value>().await.map_err(|error| error.to_string())
}

async fn discover_authorization_server(
    issuer: &str,
    require_exact_issuer: bool,
) -> Result<AuthServerMetadata, String> {
    let candidates = authorization_server_metadata_urls(issuer)?;
    let mut last_error: Option<String> = None;
    for candidate in &candidates {
        match fetch_response(candidate).await {
            Ok(response) => {
                if response.status().as_u16() == 404 {
                    continue;
                }
                match json_metadata(response, candidate).await {
                    Ok(value) => {
                        return authorization_server_metadata(&value, issuer, require_exact_issuer);
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!(
        "Could not discover OAuth metadata for {}. Tried {}. Last error: {}",
        issuer,
        candidates.join(", "),
        last_error.unwrap_or_else(|| "none".to_string())
    ))
}

/// Random, URL-safe CSRF `state` value, independent of the PKCE verifier.
pub fn random_state() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    base64::engine::general_purpose::STANDARD
        .encode(bytes)
        .replace('+', "-")
        .replace('/', "_")
        .replace('=', "")
}

fn resource_metadata(value: &Value, resource: &str) -> Result<ProtectedResourceMetadata, String> {
    let Some(metadata) = value.as_object() else {
        return Err("Protected-resource metadata is invalid".to_string());
    };
    let Some(advertised) = metadata.get("resource").and_then(Value::as_str) else {
        return Err(format!(
            "Protected-resource metadata resource does not exactly match {}",
            resource
        ));
    };
    if advertised != resource {
        return Err(format!(
            "Protected-resource metadata resource does not exactly match {}",
            resource
        ));
    }
    let Some(servers) = metadata.get("authorization_servers").and_then(Value::as_array) else {
        return Err("Protected-resource metadata has no authorization_servers".to_string());
    };
    if servers.is_empty() {
        return Err("Protected-resource metadata has no authorization_servers".to_string());
    }
    let mut authorization_servers = Vec::new();
    for issuer in servers {
        let Some(issuer) = issuer.as_str() else {
            return Err("Protected-resource metadata has an invalid authorization server".to_string());
        };
        validated_https_url(issuer, "Authorization server issuer")?;
        authorization_servers.push(issuer.to_string());
    }
    Ok(ProtectedResourceMetadata {
        resource: advertised.to_string(),
        authorization_servers,
    })
}

fn resource_metadata_url(resource: &url::Url) -> String {
    let path = if resource.path() == "/" { "" } else { resource.path() };
    let mut url = format!(
        "{}/.well-known/oauth-protected-resource{}{}",
        resource.origin().ascii_serialization(),
        path,
        resource
            .query()
            .map(|query| format!("?{}", query))
            .unwrap_or_default()
    );
    if url.ends_with('/') && path.is_empty() {
        url = url.trim_end_matches('/').to_string();
    }
    url
}

/// The `resource_metadata="..."` parameter of a `WWW-Authenticate` header.
fn header_resource_metadata(value: Option<&str>) -> Option<String> {
    let value = value?;
    let regex = regex::Regex::new(r#"(?:^|[,\s])resource_metadata\s*=\s*"((?:[^"\\]|\\.)*)""#).ok()?;
    let captures = regex.captures(value)?;
    let matched = captures.get(1)?.as_str();
    let unescape = regex::Regex::new(r"\\(.)").ok()?;
    Some(unescape.replace_all(matched, "$1").to_string())
}

async fn try_protected_resource_metadata(url: &str) -> Result<Option<ProtectedResourceMetadata>, String> {
    let resource = validated_https_url(url, "MCP endpoint")?;
    let mut header_url: Option<String> = None;
    // This probe deliberately has no Authorization header. It must not leak an
    // existing token.
    if let Ok(response) = fetch_response(resource.as_str()).await {
        header_url = response
            .headers()
            .get("www-authenticate")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| header_resource_metadata(Some(value)));
    }

    let candidate = match header_url.as_deref() {
        Some(header_url) => validated_https_url(header_url, "resource_metadata")?.to_string(),
        None => resource_metadata_url(&resource),
    };
    let response = fetch_response(&candidate).await?;
    if response.status().as_u16() == 404 && header_url.is_none() {
        return Ok(None);
    }
    let value = json_metadata(response, &candidate).await?;
    Ok(Some(resource_metadata(&value, &canonical_resource(&resource))?))
}

/// Discover RFC 9728 protected-resource metadata before the origin-level
/// authorization server fallback.
async fn discover(url: &str) -> Result<Discovery, String> {
    if let Some(protected_resource) = try_protected_resource_metadata(url).await? {
        let issuer = protected_resource
            .authorization_servers
            .first()
            .cloned()
            .ok_or_else(|| "Protected-resource metadata has no authorization_servers".to_string())?;
        return Ok(Discovery {
            metadata: discover_authorization_server(&issuer, true).await?,
            resource: Some(protected_resource.resource),
            issuer: Some(issuer),
        });
    }
    let issuer = validated_https_url(url, "MCP endpoint")?
        .origin()
        .ascii_serialization();
    Ok(Discovery {
        metadata: discover_authorization_server(&issuer, false).await?,
        resource: None,
        issuer: None,
    })
}

async fn register_client(registration_endpoint: &str, label: &str) -> Result<String, String> {
    validated_https_url(registration_endpoint, "Registration endpoint")?;
    let body = json!({
        "client_name": label,
        "redirect_uris": all_redirect_uris(),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    });
    let data = fetch_json(
        registration_endpoint,
        Some(json!({
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": body.to_string(),
        })),
    )
    .await?;
    match data.get("client_id").and_then(Value::as_str) {
        Some(client_id) if !client_id.is_empty() => Ok(client_id.to_string()),
        _ => Err(format!(
            "Dynamic client registration at {} returned no client_id",
            registration_endpoint
        )),
    }
}

/// oauth.ts:445-446
/// `const authorizationUrl = new URL(meta.authorization_endpoint);`
/// `for (const [name, value] of authParams) authorizationUrl.searchParams.set(name, value);`
///
/// `searchParams.set` REPLACES the first occurrence of a same-name parameter and drops the
/// later ones; `append_pair` duplicated them, so an advertised endpoint that already carries
/// e.g. `response_type`, `state` or `redirect_uri` produced a URL many auth servers reject.
fn authorization_url_with_params(
    authorization_endpoint: &str,
    auth_params: &[(String, String)],
) -> Result<url::Url, String> {
    let mut authorization_url = url::Url::parse(authorization_endpoint).map_err(|error| error.to_string())?;
    let existing: Vec<(String, String)> = authorization_url
        .query_pairs()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    let mut replaced: Vec<String> = Vec::new();
    for (key, value) in &existing {
        match auth_params.iter().find(|(name, _)| name == key) {
            Some((name, replacement)) => {
                if !replaced.iter().any(|seen| seen == name) {
                    // `set` keeps the ORIGINAL position of an existing key.
                    serializer.append_pair(name, replacement);
                    replaced.push(name.clone());
                }
            }
            None => {
                serializer.append_pair(key, value);
            }
        }
    }
    for (name, value) in auth_params {
        if !replaced.iter().any(|seen| seen == name) {
            serializer.append_pair(name, value);
        }
    }
    let query = serializer.finish();
    authorization_url.set_query(if query.is_empty() { None } else { Some(query.as_str()) });
    Ok(authorization_url)
}

pub type CallbackResult = Option<(String, String)>;

pub struct CallbackServer {
    pub port: u16,
    pub redirect_uri: String,
    result: Arc<Mutex<CallbackResult>>,
    notify: Arc<tokio::sync::Notify>,
    cancel: Arc<Mutex<bool>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

/// Owns the callback waiter's cancel handle so the manual-paste branch can settle the waiter
/// through `cb.cancel()` (oauth.ts:468 and oauth.ts:481) without borrowing the server.
#[derive(Clone)]
struct CallbackCanceller {
    cancel: Arc<Mutex<bool>>,
    notify: Arc<tokio::sync::Notify>,
}

impl CallbackCanceller {
    fn cancel(&self) {
        *self.cancel.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        self.notify.notify_waiters();
    }
}

/// oauth.ts:466-483 manual-paste outcome.
struct ManualOutcome {
    /// The parsed `{ code, state }` paste, or None.
    result: CallbackResult,
    /// `manualError` (oauth.ts:475-476): a real paste that failed validation.
    error: Option<String>,
    /// `manualCancelled` (oauth.ts:477-479): a UI cancellation, not a failure.
    cancelled: bool,
}

/// oauth.ts:464-483. The paste prompt races the browser callback: a successful
/// `parseRedirectInput` calls `cb.cancel()` immediately (oauth.ts:468); a cancellation or a
/// validation error waits a 500 ms grace period so an in-flight redirect can still win, and
/// then calls `cb.cancel()` too (oauth.ts:481) so login cannot hang with no redirect.
fn manual_code_future(
    on_manual_code_input: crate::utils::oauth::types::OnManualCodeInput,
    state: String,
    canceller: CallbackCanceller,
) -> crate::types::BoxFuture<ManualOutcome> {
    Box::pin(async move {
        let mut outcome = ManualOutcome { result: None, error: None, cancelled: false };
        let input = on_manual_code_input().await;
        let parsed = match &input {
            Ok(input) => parse_redirect_input(input, &state),
            Err(error) => Err(error.clone()),
        };
        match parsed {
            Ok(parsed) => {
                outcome.result = Some(parsed);
                canceller.cancel();
            }
            Err(error) => {
                // A validation error on a real paste (bad state / no code) is a genuine
                // failure to surface; a UI cancellation is not.
                if error.contains("state mismatch") || error.contains("authorization code") {
                    outcome.error = Some(error);
                } else {
                    outcome.cancelled = true;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
                canceller.cancel();
            }
        }
        outcome
    })
}

/// oauth.ts:484-486
/// `const fromCallback = await cb.waitForCode(); result = fromCallback ?? (await manual);`
///
/// Both sides are polled concurrently, so the paste prompt is shown while the redirect is
/// awaited. Whichever completes first wins and the other is cancelled, not merely dropped: a
/// real paste cancels the callback waiter immediately (oauth.ts:468), a UI cancellation or
/// validation error cancels it after a 500 ms grace period (oauth.ts:480-481), and a winning
/// browser redirect ends the race without waiting for the paste prompt (the biased branch
/// order keeps `fromCallback ?? (await manual)` preferring the callback result).
async fn race_callback_and_manual(
    wait_for_code: impl std::future::Future<Output = CallbackResult>,
    mut manual: crate::types::BoxFuture<ManualOutcome>,
) -> ManualOutcome {
    let from_callback = wait_for_code;
    tokio::pin!(from_callback);
    tokio::select! {
        biased;
        from_callback = &mut from_callback => {
            if from_callback.is_some() {
                // oauth.ts:485 short-circuits on `fromCallback`, so the paste prompt is left
                // unresolved (and dropped here) instead of blocking the login.
                return ManualOutcome { result: from_callback, error: None, cancelled: false };
            }
        }
        outcome = &mut manual => return outcome,
    }
    // The callback waiter resolves as None only after the manual branch cancelled it and is
    // about to return, so this await continues `await manual` (oauth.ts:485) and is bounded.
    manual.await
}

impl CallbackServer {
    /// `cb.cancel()` for the manual-paste future (oauth.ts:468 and oauth.ts:481).
    fn canceller(&self) -> CallbackCanceller {
        CallbackCanceller {
            cancel: self.cancel.clone(),
            notify: self.notify.clone(),
        }
    }

    pub fn cancel(&self) {
        self.canceller().cancel();
    }

    pub async fn wait_for_code(&self) -> CallbackResult {
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
            let cancelled = *self.cancel.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancelled {
                return None;
            }
            notified.await;
        }
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

impl Drop for CallbackServer {
    fn drop(&mut self) {
        self.close();
    }
}

/// Minimal HTTP/1.1 handler serving the OAuth callback page.
fn spawn_callback_server(listener: TcpListener, label: String, state: String, shared: Arc<CallbackShared>) {
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let label = label.clone();
            let state = state.clone();
            let shared = shared.clone();
            tokio::spawn(async move {
                let mut buffer = vec![0u8; 8192];
                let Ok(read) = socket.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                let request_line = request.lines().next().unwrap_or_default().to_string();
                let target = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (path, query) = match target.split_once('?') {
                    Some((path, query)) => (path.to_string(), query.to_string()),
                    None => (target.clone(), String::new()),
                };
                let params: std::collections::HashMap<String, String> =
                    url::form_urlencoded::parse(query.as_bytes())
                        .map(|(key, value)| (key.to_string(), value.to_string()))
                        .collect();

                let respond = |status: u16, body: String| {
                    let response = format!(
                        "HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        status,
                        if status == 200 { "OK" } else { "Bad Request" },
                        body.len(),
                        body
                    );
                    response
                };

                if path != CALLBACK_PATH {
                    let body = oauth_error_html("Callback route not found.", None);
                    let _ = socket.write_all(respond(404, body).as_bytes()).await;
                    return;
                }

                let error = params.get("error").cloned();
                let code = params.get("code").cloned();
                let callback_state = params.get("state").cloned();
                let status = if error.is_some() || code.is_none() { 400 } else { 200 };
                let body = if let Some(error) = error {
                    let body = oauth_error_html(
                        &format!("{} authentication failed.", label),
                        Some(&format!("Error: {}", error)),
                    );
                    shared.settle(None);
                    body
                } else if code.is_none() || callback_state.is_none() {
                    let body = oauth_error_html("Missing code or state parameter.", None);
                    shared.settle(None);
                    body
                } else {
                    let body = oauth_success_html(&format!(
                        "{} authentication completed. You can close this window.",
                        label
                    ));
                    // state is validated by the caller after the callback settles.
                    shared.settle(Some((
                        code.clone().unwrap_or_default(),
                        callback_state.clone().unwrap_or_default(),
                    )));
                    body
                };
                let _ = socket.write_all(respond(status, body).as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
}

struct CallbackShared {
    result: Arc<Mutex<CallbackResult>>,
    notify: Arc<tokio::sync::Notify>,
}

impl CallbackShared {
    fn settle(&self, value: CallbackResult) {
        {
            let mut result = self.result.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if result.is_none() {
                *result = value;
            }
        }
        self.notify.notify_waiters();
    }
}

/// Try each candidate port with a FRESH listener (a listener that failed to bind
/// can't be reused), so a leaked/concurrent login can't block us with EADDRINUSE.
async fn start_callback_server(label: &str, state: &str) -> Result<CallbackServer, String> {
    let host = callback_host();
    let mut last_error: Option<String> = None;
    for port in callback_ports() {
        match TcpListener::bind((host.as_str(), port)).await {
            Ok(listener) => {
                let result: Arc<Mutex<CallbackResult>> = Arc::new(Mutex::new(None));
                let notify = Arc::new(tokio::sync::Notify::new());
                let shared = Arc::new(CallbackShared {
                    result: result.clone(),
                    notify: notify.clone(),
                });
                spawn_callback_server(listener, label.to_string(), state.to_string(), shared);
                return Ok(CallbackServer {
                    port,
                    redirect_uri: redirect_uri_for(port),
                    result,
                    notify,
                    cancel: Arc::new(Mutex::new(false)),
                    handle: None,
                });
            }
            Err(error) => last_error = Some(format!("port {} in use: {}", port, error)),
        }
    }
    Err(format!(
        "Could not start the OAuth callback server: ports {}-{} are all in use. Close other login attempts and retry. ({})",
        callback_port_base(),
        callback_port_base() + CALLBACK_PORT_COUNT - 1,
        last_error.unwrap_or_else(|| "unknown".to_string())
    ))
}

fn parse_redirect_input(input: &str, expected_state: &str) -> Result<(String, String), String> {
    let value = input.trim();
    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    match url::Url::parse(value) {
        Ok(url) => {
            for (key, value) in url.query_pairs() {
                match key.as_ref() {
                    "code" => code = Some(value.to_string()),
                    "state" => state = Some(value.to_string()),
                    _ => {}
                }
            }
        }
        Err(_) => {
            let params: std::collections::HashMap<String, String> =
                url::form_urlencoded::parse(value.as_bytes())
                    .map(|(key, value)| (key.to_string(), value.to_string()))
                    .collect();
            code = params
                .get("code")
                .cloned()
                .or_else(|| (!value.is_empty()).then(|| value.to_string()));
            state = params.get("state").cloned();
        }
    }
    if let Some(state) = state.as_deref() {
        if state != expected_state {
            return Err("OAuth state mismatch".to_string());
        }
    }
    let Some(code) = code else {
        return Err("Missing authorization code".to_string());
    };
    Ok((code, state.unwrap_or_else(|| expected_state.to_string())))
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<f64>,
}

async fn exchange_token(
    token_endpoint: &str,
    params: &[(String, String)],
) -> Result<TokenResponse, String> {
    validated_https_url(token_endpoint, "Token endpoint")?;
    let client = http_client()?;
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(params.iter().map(|(key, value)| (key.as_str(), value.as_str())))
        .finish();
    let response = client
        .post(token_endpoint)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status().as_u16();
    let text = response.text().await.unwrap_or_default();
    if !(200..300).contains(&status) {
        return Err(format!(
            "Token request to {} failed: {}",
            token_endpoint, status
        ));
    }
    let token: Value = serde_json::from_str(&text).map_err(|_| {
        format!("Token request to {} returned invalid JSON", token_endpoint)
    })?;
    let access_token = token
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Token request to {} returned no access_token", token_endpoint))?;
    if access_token.is_empty() {
        return Err(format!("Token request to {} returned no access_token", token_endpoint));
    }
    let refresh_token = match token.get("refresh_token") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => {
            return Err(format!(
                "Token request to {} returned an invalid refresh_token",
                token_endpoint
            ))
        }
    };
    let expires_in = match token.get("expires_in") {
        None | Some(Value::Null) => None,
        Some(Value::Number(number)) => number.as_f64().filter(|value| value.is_finite()),
        Some(_) => {
            return Err(format!(
                "Token request to {} returned an invalid expires_in",
                token_endpoint
            ))
        }
    };
    Ok(TokenResponse {
        access_token: access_token.to_string(),
        refresh_token,
        expires_in,
    })
}

fn now_ms() -> f64 {
    crate::utils::now_ms() as f64
}

fn to_credentials(
    token: &TokenResponse,
    token_endpoint: &str,
    client_id: &str,
    endpoint: Option<&str>,
    resource: Option<&str>,
    issuer: Option<&str>,
    previous_refresh: Option<&str>,
) -> OAuthCredentials {
    let expires = match token.expires_in {
        Some(expires_in) => now_ms() + expires_in * 1000.0 - TOKEN_EXPIRY_BUFFER_MS,
        None => now_ms() + 3600.0 * 1000.0 - TOKEN_EXPIRY_BUFFER_MS,
    };
    let mut credentials = OAuthCredentials {
        access: token.access_token.clone(),
        // Some servers omit refresh_token on refresh; keep the prior one.
        refresh: token
            .refresh_token
            .clone()
            .or_else(|| previous_refresh.map(str::to_string))
            .unwrap_or_default(),
        expires,
        extra: serde_json::Map::new(),
    };
    credentials
        .extra
        .insert("tokenEndpoint".to_string(), Value::String(token_endpoint.to_string()));
    credentials
        .extra
        .insert("clientId".to_string(), Value::String(client_id.to_string()));
    if let Some(endpoint) = endpoint {
        credentials
            .extra
            .insert("endpoint".to_string(), Value::String(endpoint.to_string()));
    }
    if let Some(resource) = resource {
        credentials
            .extra
            .insert("resource".to_string(), Value::String(resource.to_string()));
    }
    if let Some(issuer) = issuer {
        credentials
            .extra
            .insert("issuer".to_string(), Value::String(issuer.to_string()));
    }
    credentials
}

fn extra_str<'a>(credentials: &'a OAuthCredentials, key: &str) -> Option<&'a str> {
    credentials.extra.get(key).and_then(Value::as_str)
}

pub fn create_mcp_oauth_provider(config: McpOAuthConfig) -> OAuthProviderInterface {
    let label = config.label.clone().unwrap_or_else(|| config.server.clone());
    let config = Arc::new(config);

    let login_config = config.clone();
    let login_label = label.clone();
    let login = Arc::new(move |callbacks: OAuthLoginCallbacks| {
        let config = login_config.clone();
        let label = login_label.clone();
        Box::pin(async move {
            let discovery = discover(&config.url).await?;
            let meta = discovery.metadata.clone();
            if let Some(on_progress) = callbacks.on_progress.as_ref() {
                on_progress(format!(
                    "Discovered {}",
                    discovery.issuer.clone().unwrap_or_else(|| meta.issuer.clone())
                ));
            }

            let mut client_id = config.client_id.clone();
            if client_id.is_none() {
                let Some(registration_endpoint) = meta.registration_endpoint.clone() else {
                    return Err(format!(
                        "{} does not support dynamic client registration and no clientId was configured. Set a pre-registered client id for this server.",
                        label
                    ));
                };
                if let Some(on_progress) = callbacks.on_progress.as_ref() {
                    on_progress("Registering OAuth client\u{2026}".to_string());
                }
                client_id = Some(register_client(&registration_endpoint, &format!("Optimus by Telemus AI ({})", label)).await?);
            }
            let client_id = client_id.unwrap_or_default();

            let (verifier, challenge) = generate_pkce().await;
            // `state` must be independent of the PKCE verifier.
            let state = random_state();
            let scope = config
                .scopes
                .clone()
                .or_else(|| meta.scopes_supported.as_ref().map(|scopes| scopes.join(" ")));
            let callback = start_callback_server(&label, &state).await?;
            let redirect_uri = callback.redirect_uri.clone();

            let mut auth_params: Vec<(String, String)> = vec![
                ("client_id".to_string(), client_id.clone()),
                ("response_type".to_string(), "code".to_string()),
                ("redirect_uri".to_string(), redirect_uri.clone()),
                ("code_challenge".to_string(), challenge),
                ("code_challenge_method".to_string(), "S256".to_string()),
                ("state".to_string(), state.clone()),
            ];
            if let Some(scope) = scope.as_deref() {
                if !scope.is_empty() {
                    auth_params.push(("scope".to_string(), scope.to_string()));
                }
            }
            if let Some(resource) = discovery.resource.as_deref() {
                auth_params.push(("resource".to_string(), resource.to_string()));
            }

            let authorization_url = authorization_url_with_params(&meta.authorization_endpoint, &auth_params)?;

            if let Some(on_auth) = callbacks.on_auth.as_ref() {
                on_auth(crate::utils::oauth::types::OAuthAuthInfo {
                    url: authorization_url.to_string(),
                    instructions: Some(
                        "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                            .to_string(),
                    ),
                });
            }

            // Race the local callback server against a manual paste (oauth.ts:453-486). The
            // callback waiter is awaited FIRST; a real paste cancels it, and a UI cancellation
            // cancels it after a 500 ms grace period, so login always completes.
            let mut result: CallbackResult;
            let manual_cancelled: bool;
            if let Some(on_manual_code_input) = callbacks.on_manual_code_input.as_ref() {
                let outcome = race_callback_and_manual(
                    callback.wait_for_code(),
                    manual_code_future(
                        on_manual_code_input.clone(),
                        state.clone(),
                        callback.canceller(),
                    ),
                )
                .await;
                if outcome.result.is_none() {
                    // oauth.ts:486 `if (!result && manualError) throw manualError;`
                    if let Some(error) = outcome.error {
                        return Err(error);
                    }
                }
                manual_cancelled = outcome.cancelled;
                result = outcome.result;
            } else {
                manual_cancelled = false;
                result = callback.wait_for_code().await;
                if result.is_none() {
                    let Some(on_prompt) = callbacks.on_prompt.as_ref() else {
                        return Err("Missing authorization code".to_string());
                    };
                    let input = on_prompt(OAuthPrompt {
                        message: "Paste the authorization code or full redirect URL:".to_string(),
                        placeholder: Some(redirect_uri.clone()),
                        allow_empty: None,
                    })
                    .await;
                    result = Some(parse_redirect_input(&input, &state)?);
                }
            }
            let Some((code, result_state)) = result else {
                return Err(if manual_cancelled {
                    "Login cancelled".to_string()
                } else {
                    "Missing authorization code".to_string()
                });
            };
            if result_state != state {
                return Err("OAuth state mismatch".to_string());
            }

            if let Some(on_progress) = callbacks.on_progress.as_ref() {
                on_progress("Exchanging authorization code for tokens\u{2026}".to_string());
            }
            let mut params: Vec<(String, String)> = vec![
                ("grant_type".to_string(), "authorization_code".to_string()),
                ("code".to_string(), code),
                ("redirect_uri".to_string(), redirect_uri),
                ("client_id".to_string(), client_id.clone()),
                ("code_verifier".to_string(), verifier),
            ];
            if let Some(resource) = discovery.resource.as_deref() {
                params.push(("resource".to_string(), resource.to_string()));
            }
            let token = exchange_token(&meta.token_endpoint, &params).await?;
            Ok(to_credentials(
                &token,
                &meta.token_endpoint,
                &client_id,
                Some(&config.url),
                discovery.resource.as_deref(),
                discovery.issuer.as_deref(),
                None,
            ))
        }) as crate::types::BoxFuture<Result<OAuthCredentials, String>>
    });

    let refresh_config = config.clone();
    let refresh_label = label.clone();
    let refresh_token = Arc::new(move |credentials: OAuthCredentials| {
        let config = refresh_config.clone();
        let label = refresh_label.clone();
        Box::pin(async move {
            let stored_endpoint = extra_str(&credentials, "endpoint").map(str::to_string);
            if stored_endpoint.as_deref() != Some(config.url.as_str()) {
                return Err(format!(
                    "Stored OAuth credentials are not bound to {}; re-run /mcp login {}",
                    config.url, config.server
                ));
            }
            let configured_resource =
                canonical_resource(&validated_https_url(&config.url, "MCP endpoint")?);
            let stored_resource = extra_str(&credentials, "resource").map(str::to_string);
            if let Some(resource) = stored_resource.as_deref() {
                if resource != configured_resource {
                    return Err(format!(
                        "Stored OAuth credentials are not bound to {}; re-run /mcp login {}",
                        configured_resource, config.server
                    ));
                }
            }
            let stored_issuer = extra_str(&credentials, "issuer").map(str::to_string);
            if stored_resource.is_none() != stored_issuer.is_none() {
                return Err(format!(
                    "Stored OAuth credentials for {} have incomplete resource binding; re-run /mcp login {}",
                    label, config.server
                ));
            }
            if let Some(issuer) = stored_issuer.as_deref() {
                validated_https_url(issuer, "Stored authorization server issuer")?;
            }
            if credentials.refresh.is_empty() {
                return Err(format!(
                    "No refresh token stored for {}; re-run /mcp login {}",
                    label, config.server
                ));
            }
            let discovery = discover(&config.url).await?;
            if stored_resource.is_none() != discovery.resource.is_none() {
                return Err(format!(
                    "OAuth discovery mode changed for {}; re-run /mcp login {}",
                    config.url, config.server
                ));
            }
            if let Some(resource) = stored_resource.as_deref() {
                if discovery.resource.as_deref() != Some(resource)
                    || discovery.issuer.as_deref() != stored_issuer.as_deref()
                {
                    return Err(format!(
                        "Stored OAuth credentials do not match current protected-resource metadata for {}",
                        config.url
                    ));
                }
            }
            let stored_token_endpoint = extra_str(&credentials, "tokenEndpoint").map(str::to_string);
            let token_endpoint = stored_token_endpoint
                .clone()
                .unwrap_or_else(|| discovery.metadata.token_endpoint.clone());
            if let Some(stored_token_endpoint) = stored_token_endpoint.as_deref() {
                if discovery.metadata.token_endpoint != stored_token_endpoint {
                    return Err(format!(
                        "Stored OAuth token endpoint does not match current authorization-server metadata for {}",
                        config.url
                    ));
                }
            }
            let client_id = extra_str(&credentials, "clientId")
                .map(str::to_string)
                .or_else(|| config.client_id.clone());
            if token_endpoint.is_empty() {
                return Err(format!(
                    "No token endpoint stored for {}; re-run /mcp login {}",
                    label, config.server
                ));
            }
            let mut params: Vec<(String, String)> = vec![
                ("grant_type".to_string(), "refresh_token".to_string()),
                ("refresh_token".to_string(), credentials.refresh.clone()),
            ];
            if let Some(client_id) = client_id.as_deref() {
                params.push(("client_id".to_string(), client_id.to_string()));
            }
            if let Some(resource) = stored_resource.as_deref() {
                params.push(("resource".to_string(), resource.to_string()));
            }
            let token = exchange_token(&token_endpoint, &params).await?;
            Ok(to_credentials(
                &token,
                &token_endpoint,
                client_id.as_deref().unwrap_or_default(),
                stored_endpoint.as_deref(),
                stored_resource.as_deref(),
                stored_issuer.as_deref(),
                Some(&credentials.refresh),
            ))
        }) as crate::types::BoxFuture<Result<OAuthCredentials, String>>
    });

    OAuthProviderInterface {
        id: format!("mcp:{}", config.server),
        name: label,
        login,
        uses_callback_server: Some(true),
        refresh_token,
        get_api_key: Arc::new(|credentials: &OAuthCredentials| credentials.access.clone()),
        modify_models: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_range_and_redirect_uris_match_typescript() {
        assert_eq!(callback_ports().len(), 10);
        assert_eq!(callback_ports()[0], 53700);
        assert_eq!(callback_ports()[9], 53709);
        assert_eq!(redirect_uri_for(53700), "http://localhost:53700/callback");
        assert_eq!(all_redirect_uris().len(), 10);
    }

    #[test]
    fn validates_https_urls() {
        assert!(validated_https_url("https://example.com/mcp", "MCP endpoint").is_ok());
        assert!(validated_https_url("http://example.com", "MCP endpoint").is_err());
        assert!(validated_https_url("https://user:pass@example.com", "MCP endpoint").is_err());
        assert!(validated_https_url("https://example.com#frag", "MCP endpoint").is_err());
        assert!(validated_https_url("not a url", "MCP endpoint").is_err());
    }

    #[test]
    fn canonical_resource_drops_root_slash() {
        let root = url::Url::parse("https://mcp.linear.app/").unwrap();
        assert_eq!(canonical_resource(&root), "https://mcp.linear.app");
        let path = url::Url::parse("https://mcp.linear.app/mcp").unwrap();
        assert_eq!(canonical_resource(&path), "https://mcp.linear.app/mcp");
    }

    #[test]
    fn metadata_urls_follow_rfc_8414_and_oidc() {
        let urls = authorization_server_metadata_urls("https://auth.example.com").unwrap();
        assert_eq!(
            urls,
            vec![
                "https://auth.example.com/.well-known/oauth-authorization-server".to_string(),
                "https://auth.example.com/.well-known/openid-configuration".to_string()
            ]
        );
        let urls = authorization_server_metadata_urls("https://auth.example.com/tenant/").unwrap();
        assert_eq!(
            urls[0],
            "https://auth.example.com/.well-known/oauth-authorization-server/tenant"
        );
        assert_eq!(
            urls[1],
            "https://auth.example.com/tenant/.well-known/openid-configuration"
        );
        assert!(authorization_server_metadata_urls("https://auth.example.com?x=1").is_err());
    }

    #[test]
    fn resource_metadata_url_uses_well_known_location() {
        let url = url::Url::parse("https://mcp.linear.app/mcp").unwrap();
        assert_eq!(
            resource_metadata_url(&url),
            "https://mcp.linear.app/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn parses_resource_metadata_header() {
        let header = r#"Bearer error="invalid_token", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource""#;
        assert_eq!(
            header_resource_metadata(Some(header)).as_deref(),
            Some("https://mcp.example.com/.well-known/oauth-protected-resource")
        );
        assert_eq!(header_resource_metadata(Some("Bearer realm=\"x\"")), None);
        assert_eq!(header_resource_metadata(None), None);
    }

    #[test]
    fn authorization_server_metadata_requires_exact_issuer_for_prm() {
        let value = json!({
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token"
        });
        let metadata = authorization_server_metadata(&value, "https://auth.example.com", true).unwrap();
        assert_eq!(metadata.token_endpoint, "https://auth.example.com/token");

        assert!(authorization_server_metadata(&value, "https://other.example.com", true).is_err());
        assert!(authorization_server_metadata(&value, "https://other.example.com", false).is_err());

        let missing = json!({"issuer": "https://auth.example.com"});
        assert!(authorization_server_metadata(&missing, "https://auth.example.com", true).is_err());
    }

    #[test]
    fn protected_resource_metadata_validates_resource_and_servers() {
        let value = json!({
            "resource": "https://mcp.linear.app",
            "authorization_servers": ["https://auth.example.com"]
        });
        let metadata = resource_metadata(&value, "https://mcp.linear.app").unwrap();
        assert_eq!(metadata.authorization_servers.len(), 1);
        assert!(resource_metadata(&value, "https://other").is_err());
        assert!(resource_metadata(&json!({"resource": "https://mcp.linear.app", "authorization_servers": []}), "https://mcp.linear.app").is_err());
    }

    #[test]
    fn redirect_input_parsing_checks_state() {
        let parsed = parse_redirect_input(
            "http://localhost:53700/callback?code=abc&state=xyz",
            "xyz",
        )
        .unwrap();
        assert_eq!(parsed.0, "abc");
        assert!(parse_redirect_input("code=abc&state=bad", "xyz").is_err());
        assert!(parse_redirect_input("", "xyz").is_err());
        let bare = parse_redirect_input("plain-code", "xyz").unwrap();
        assert_eq!(bare.0, "plain-code");
        assert_eq!(bare.1, "xyz");
    }

    /// OAUTH-1: oauth.ts:464-486. The manual paste completes first and no browser redirect ever
    /// arrives. With the old `tokio::join!` the flow waited for the callback waiter forever, so
    /// the bounded timeout here FAILS the test on that code path.
    #[test]
    fn manual_paste_first_returns_promptly_without_a_redirect() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let outcome = runtime.block_on(async {
            let server = start_callback_server("Test", "expected-state").await.unwrap();
            let manual: crate::utils::oauth::types::OnManualCodeInput =
                Arc::new(|| Box::pin(async { Ok("plain-code".to_string()) }));
            let outcome = tokio::time::timeout(
                Duration::from_secs(5),
                race_callback_and_manual(
                    server.wait_for_code(),
                    manual_code_future(manual.clone(), "expected-state".to_string(), server.canceller()),
                ),
            )
            .await
            .expect("manual paste must not wait for a browser redirect (OAUTH-1 hang)");
            // oauth.ts:468 `cb.cancel()`: the successful paste settles the callback waiter, so a
            // later wait returns at once instead of blocking on a redirect that never arrives.
            let settled = tokio::time::timeout(Duration::from_millis(500), server.wait_for_code())
                .await
                .expect("the successful paste must cancel the callback waiter (oauth.ts:468)");
            assert!(settled.is_none());
            outcome
        });
        assert_eq!(
            outcome.result,
            Some(("plain-code".to_string(), "expected-state".to_string()))
        );
        assert!(outcome.error.is_none());
        assert!(!outcome.cancelled);
    }

    /// OAUTH-1: oauth.ts:484-485 `const fromCallback = await cb.waitForCode(); result =
    /// fromCallback ?? (await manual);` - the browser callback wins while the paste prompt stays
    /// open, so the login must return the callback result instead of waiting for the paste.
    #[test]
    fn browser_callback_first_wins_over_an_open_paste_prompt() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let outcome = runtime.block_on(async {
            let server = start_callback_server("Test", "expected-state").await.unwrap();
            let (manual_sender, manual_receiver) = tokio::sync::oneshot::channel::<Result<String, String>>();
            let shared_receiver = Arc::new(std::sync::Mutex::new(Some(manual_receiver)));
            let manual: crate::utils::oauth::types::OnManualCodeInput = Arc::new(move || {
                let shared_receiver = shared_receiver.clone();
                Box::pin(async move {
                    let receiver = shared_receiver
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take();
                    match receiver {
                        Some(receiver) => receiver
                            .await
                            .unwrap_or_else(|_| Err("Login cancelled".to_string())),
                        None => Err("Login cancelled".to_string()),
                    }
                })
            });
            let port = server.port;
            // Deliver the browser redirect without ever answering the paste prompt.
            let deliver = async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let Ok(mut stream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
                    return;
                };
                let request = "GET /callback?code=browser-code&state=expected-state HTTP/1.1\r\nHost: localhost\r\n\r\n";
                if AsyncWriteExt::write_all(&mut stream, request.as_bytes()).await.is_ok() {
                    let mut response = String::new();
                    let _ = stream.read_to_string(&mut response).await;
                }
            };
            let race = race_callback_and_manual(
                server.wait_for_code(),
                manual_code_future(manual.clone(), "expected-state".to_string(), server.canceller()),
            );
            let (outcome, ()) = tokio::time::timeout(Duration::from_secs(5), futures::future::join(race, deliver))
                .await
                .expect("callback win must not wait for the paste prompt (OAUTH-1 hang)");
            drop(manual_sender);
            outcome
        });
        assert_eq!(
            outcome.result,
            Some(("browser-code".to_string(), "expected-state".to_string()))
        );
    }


    #[test]
    fn authorization_url_replaces_existing_parameters() {
        let params = vec![
            ("client_id".to_string(), "cid".to_string()),
            ("response_type".to_string(), "code".to_string()),
            ("redirect_uri".to_string(), "http://localhost:1/callback".to_string()),
            ("state".to_string(), "fresh-state".to_string()),
        ];
        let url = authorization_url_with_params(
            "https://auth.example.com/authorize?response_type=token&state=stale&audience=api",
            &params,
        )
        .unwrap();
        let pairs: Vec<(String, String)> = url
            .query_pairs()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        // oauth.ts:446 `searchParams.set` replaces instead of appending.
        assert_eq!(pairs.iter().filter(|(key, _)| key == "response_type").count(), 1);
        assert_eq!(pairs.iter().filter(|(key, _)| key == "state").count(), 1);
        assert!(pairs.contains(&("response_type".to_string(), "code".to_string())));
        assert!(pairs.contains(&("state".to_string(), "fresh-state".to_string())));
        assert!(!pairs.contains(&("state".to_string(), "stale".to_string())));
        // A non-conflicting advertised parameter survives (TS `set` only touches sent names).
        assert!(pairs.contains(&("audience".to_string(), "api".to_string())));
        // `set` keeps the original position of a replaced key.
        assert_eq!(url.query_pairs().next().map(|(key, _)| key.to_string()).as_deref(), Some("response_type"));
    }

    #[test]
    fn random_state_is_url_safe_and_unique() {
        let first = random_state();
        let second = random_state();
        assert_ne!(first, second);
        assert!(!first.contains('+') && !first.contains('/') && !first.contains('='));
        assert_eq!(first.len(), 43);
    }
}
