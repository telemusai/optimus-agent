//! Process-wide pooled reqwest clients keyed by builder policy.
//!
//! A per-request `reqwest::Client` drops its connection pool with it, so every provider turn
//! paid TCP + TLS setup again. Clients here are cached per builder policy (the redirect,
//! HTTP version and proxy options that vary per provider family) and kept for the process
//! lifetime, so a warm turn reuses the pooled connection instead of handshaking.
//!
//! Invariants:
//! - Credentials are PER REQUEST (Authorization headers), never stored on a client, matching
//!   the `responses_transport::http_client` precedent and its reuse test.
//! - No `ClientBuilder::timeout` is ever set here: that option is a TOTAL deadline
//!   (reqwest client.rs `RequestBuilder::timeout`) and would impose one deadline on every
//!   request sharing the client. Header-only wrapping stays at the call sites
//!   (`openai_completions::post_chat_completions`), total timeouts stay on the request
//!   builders that already pass them.
//! - Pool sizing follows the in-repo pooled precedent: 90s idle retention, otherwise reqwest
//!   defaults.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Idle-connection retention, matching the pooled `responses_transport` client.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// The builder options that vary per provider family; equal policies share one client.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ClientPolicy {
	/// reqwest defaults: standard redirects, HTTP/2 allowed. The plain OpenAI-compatible,
	/// anthropic, google, google-vertex, mistral, azure and codex SSE paths.
	Default,
	/// `redirect(Policy::none())`: the responses transport's opted-in providers and the
	/// bedrock responses client (signed requests that must not follow redirects).
	RedirectNone,
	/// Bedrock converse-stream: `redirect(Policy::none())` plus the conditional
	/// `http1_only()` / `no_proxy()` options resolved from the environment.
	Bedrock { force_http1: bool, no_proxy: bool },
	/// Kiro: `redirect(Policy::none())` + `connect_timeout(15s)`.
	Kiro,
}

static CLIENTS: OnceLock<Mutex<HashMap<ClientPolicy, Result<reqwest::Client, String>>>> = OnceLock::new();

/// Pooled client for the policy. Build failures are cached, so every caller of a failed
/// policy sees the same error string the first build produced.
pub(crate) fn try_shared_client(policy: ClientPolicy) -> Result<reqwest::Client, String> {
	let clients = CLIENTS.get_or_init(|| Mutex::new(HashMap::new()));
	let mut guard = clients.lock().unwrap();
	if let Some(cached) = guard.get(&policy) {
		return cached.clone();
	}
	let built = build(policy).map_err(|error| error.to_string());
	guard.insert(policy, built.clone());
	built
}

/// Infallible form for the sites that previously used `reqwest::Client::new()` or an
/// `expect`-ing pooled builder: a client-construction failure stays fatal.
pub(crate) fn shared_client(policy: ClientPolicy) -> reqwest::Client {
	try_shared_client(policy).expect("Shared HTTP client")
}

fn build(policy: ClientPolicy) -> reqwest::Result<reqwest::Client> {
	// reqwest's default `pool_idle_timeout` is already 90s; set it explicitly so the pooling
	// policy of every shared client is stated, not inherited.
	let builder = reqwest::Client::builder().pool_idle_timeout(POOL_IDLE_TIMEOUT);
	let builder = match policy {
		ClientPolicy::Default => builder,
		ClientPolicy::RedirectNone => builder.redirect(reqwest::redirect::Policy::none()),
		ClientPolicy::Bedrock { force_http1, no_proxy } => {
			let builder = builder.redirect(reqwest::redirect::Policy::none());
			let builder = if force_http1 { builder.http1_only() } else { builder };
			if no_proxy { builder.no_proxy() } else { builder }
		}
		ClientPolicy::Kiro => builder
			.redirect(reqwest::redirect::Policy::none())
			.connect_timeout(Duration::from_secs(15)),
	};
	builder.build()
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Two `shared_client` calls with the same policy must reuse one pooled connection:
	/// one accepted socket serves both requests. Mirrors the transport pool reuse test.
	#[tokio::test]
	async fn shared_client_reuses_one_connection_across_calls() {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			// Exactly one accepted connection: a second accept would hang the test.
			let (mut socket, _) = listener.accept().await.unwrap();
			for _ in 0..2 {
				let mut request = Vec::new();
				let mut buffer = [0_u8; 2048];
				while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
					let count = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buffer))
						.await
						.expect("request did not reuse the pooled connection")
						.unwrap();
					assert!(count > 0, "pooled connection closed early");
					request.extend_from_slice(&buffer[..count]);
					assert!(request.len() < 8192);
				}
				socket
					.write_all(
						b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok",
					)
					.await
					.unwrap();
			}
		});
		for _ in 0..2 {
			let response = shared_client(ClientPolicy::Default)
				.get(format!("http://{address}/probe"))
				.timeout(Duration::from_secs(5))
				.send()
				.await
				.unwrap();
			assert_eq!(response.status(), reqwest::StatusCode::OK);
			// Fully consume the response so reuse is observable on the next call.
			assert_eq!(response.text().await.unwrap(), "ok");
		}
		server.await.unwrap();
	}

	#[test]
	fn try_shared_client_returns_ok_for_every_policy() {
		try_shared_client(ClientPolicy::Default).unwrap();
		try_shared_client(ClientPolicy::RedirectNone).unwrap();
		try_shared_client(ClientPolicy::Bedrock { force_http1: false, no_proxy: false }).unwrap();
		try_shared_client(ClientPolicy::Bedrock { force_http1: true, no_proxy: true }).unwrap();
		try_shared_client(ClientPolicy::Kiro).unwrap();
	}
}
