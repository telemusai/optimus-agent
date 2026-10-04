//! Port of packages/ai/src/providers/bedrock-responses-client.ts
//!
//! The TypeScript builds an OpenAI SDK client whose `fetch` signs each request with
//! `@smithy/signature-v4` (SigV4) and lets `@aws-sdk/credential-provider-node` resolve the
//! credentials. The Rust port builds the same request itself with `reqwest`, signs it with the
//! SigV4 implementation in `amazon_bedrock.rs` (HMAC-SHA256 over `sha2`, because the `hmac`
//! crate is not in the workspace dependency list) and reads credentials from the same
//! environment variables with the same precedence.
//!
//! CREDENTIAL-CHAIN GAP (F6, unresolved by design - do not fake it): the TypeScript passes
//! `defaultProvider` / `options.credentialProvider` to `SignatureV4`
//! (`bedrock-responses-client.ts:48-54`), which is the full AWS default chain. The sources the
//! port actually implements are exactly the ones `resolve_aws_credentials` covers
//! (`amazon_bedrock.rs:656-730`): the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
//! `AWS_SESSION_TOKEN` environment triple, and the static `aws_access_key_id` /
//! `aws_secret_access_key` / `aws_session_token` keys of a named profile in
//! `$AWS_SHARED_CREDENTIALS_FILE` / `~/.aws/credentials` / `$AWS_CONFIG_FILE` / `~/.aws/config`.
//!
//! NOT implemented, and they cannot be, by the owner module alone: EC2 IMDS, ECS/container
//! credentials (`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` / `..._FULL_URI`), web identity
//! (`AWS_WEB_IDENTITY_TOKEN_FILE` + STS `AssumeRoleWithWebIdentity`), SSO, and
//! `credential_process`. Each needs a network round trip or a child process plus a clock-driven
//! cache, none of which exists in this crate; a request would fail with
//! "Could not load credentials from any providers" (`amazon_bedrock.rs:727-729`) instead of
//! signing. The exact missing owner symbol is the AWS SDK default provider chain behind
//! `defaultProvider` - it has no Rust counterpart in this workspace.

use std::time::Duration;

use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::Value;

use crate::types::{Model, StreamOptions};
use crate::utils::now_ms;
use crate::utils::sse_frames::SseFrames;
use crate::utils::stream_failure::ThrownStreamError;

use super::amazon_bedrock::{resolve_aws_credentials, sign_request_for_service, AwsCredentials};

/// The Rust counterpart of a value thrown out of the TypeScript `try` block of
/// `streamBedrockResponses` (`amazon-bedrock-responses.ts:48-90`).
///
/// It lives here because this module owns the transport that can raise each variant: the SSE
/// reader detects the mid-stream `throw`s and this module holds the response object for the
/// `responses.create(...)` reject.
#[derive(Debug, Clone)]
pub enum ResponsesRunError {
	/// `throw streamFailureFromStopReason(...)` / a `StreamFailureError` from the shared stream loop.
	Failure(Box<crate::utils::stream_failure::StreamFailureError>),
	/// `throw new Error(...)`.
	Message(String),
	/// A thrown SDK `APIError`, whose `status` / `headers` / `error` fields
	/// `extractStreamFailureParts` reads (`utils/stream-failure.ts:142-167`). It covers the two
	/// `APIError` throws the OpenAI SDK raises for this provider:
	///
	/// * `APIError.generate(status, body, ...)` when `responses.create(...)`
	///   (`amazon-bedrock-responses.ts:59`) rejects for a non-2xx HTTP status, and
	/// * `new APIError(undefined, data.error, undefined, response.headers)` when a streamed
	///   frame carries a truthy `error` (`Stream.fromSSEResponse`).
	///
	/// Same shape as `RunError::Value` in `openai_responses.rs:87-103`.
	ApiError(Value),
}

impl ResponsesRunError {
	pub fn message(message: impl Into<String>) -> Self {
		ResponsesRunError::Message(message.into())
	}

	/// The thrown value as the shared `stream-failure` helpers expect it.
	pub fn as_thrown(&self) -> ThrownStreamError<'_> {
		match self {
			ResponsesRunError::Failure(failure) => ThrownStreamError::Failure(failure),
			ResponsesRunError::Message(message) => ThrownStreamError::Message(message),
			ResponsesRunError::ApiError(value) => ThrownStreamError::Value(value),
		}
	}
}

impl std::fmt::Display for ResponsesRunError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			ResponsesRunError::Failure(failure) => write!(formatter, "{}", failure.message),
			ResponsesRunError::Message(message) => write!(formatter, "{}", message),
			// `APIError extends Error`, and `APIError.makeMessage` is the `message`.
			ResponsesRunError::ApiError(value) => {
				let message = value
					.get("message")
					.and_then(Value::as_str)
					.filter(|message| !message.is_empty());
				write!(formatter, "{}", message.unwrap_or("Provider stream failed"))
			}
		}
	}
}

impl std::error::Error for ResponsesRunError {}

/// TS: `interface BedrockResponsesAuthOptions extends StreamOptions`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BedrockResponsesAuthOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// Explicit signing region; otherwise taken from the selected endpoint.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub region: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub profile: Option<String>,
	/// TS: `credentialProvider?: ReturnType<typeof defaultProvider>`.
	///
	/// NOTE: the TypeScript accepts an injected AWS credential provider. The port cannot call
	/// into an SDK provider object, so an injected provider is represented by
	/// `credentials`, and `has_credential_provider` keeps the TS branches
	/// (`!options?.profile && !options?.credentialProvider`) reachable.
	#[serde(skip)]
	pub credentials: Option<AwsCredentials>,
	#[serde(skip)]
	pub has_credential_provider: bool,
}

impl BedrockResponsesAuthOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			..Default::default()
		}
	}
}

/// TS: `createBedrockResponsesClient(model, options): OpenAI`.
///
/// The port returns the resolved connection instead of an SDK client: the base URL, the region,
/// the SigV4 service name and the auth mode the request must use.
#[derive(Debug, Clone, PartialEq)]
pub struct BedrockResponsesClient {
	pub base_url: String,
	/// `new OpenAI({ apiKey: bearerToken || "<aws-sigv4>" })`.
	pub api_key: String,
	pub bearer_token: Option<String>,
	pub signer: Option<BedrockResponsesSigner>,
	/// `Object.fromEntries(defaultHeaders)` - a `None` value means "delete this header".
	pub default_headers: IndexMap<String, Option<String>>,
	pub max_retries: i64,
	/// The origin the credentials may be sent to.
	pub origin: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BedrockResponsesSigner {
	pub region: String,
	pub service: String,
}

/// `export function createBedrockResponsesClient(model, options?)`.
pub fn create_bedrock_responses_client(
	model: &Model,
	options: Option<&BedrockResponsesAuthOptions>,
) -> Result<BedrockResponsesClient, String> {
	let base_url = std::env::var("AWS_BEDROCK_BASE_URL")
		.map(|value| value.trim().to_string())
		.ok()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| model.base_url.clone());
	let url = url::Url::parse(&base_url).map_err(|_| "Invalid URL".to_string())?;
	if url.query().is_some()
		|| url.fragment().is_some()
		|| !url.username().is_empty()
		|| url.password().is_some()
		|| !(url.scheme() == "http" || url.scheme() == "https")
	{
		return Err(
			"Bedrock base URL must be an HTTP(S) API root without credentials, query, or fragment."
				.to_string(),
		);
	}
	let endpoint = match_endpoint_host(url.host_str().unwrap_or_default());
	let model_endpoint = url::Url::parse(&model.base_url)
		.ok()
		.and_then(|url| url.host_str().map(str::to_string))
		.and_then(|host| match_endpoint_host(&host));
	let runtime_model = is_runtime_model(&model.id);
	let service = if runtime_model { "bedrock" } else { "bedrock-mantle" };
	let region = options
		.and_then(|options| options.region.clone())
		.or_else(|| endpoint.as_ref().map(|endpoint| endpoint.region.clone()))
		.or_else(|| model_endpoint.as_ref().map(|endpoint| endpoint.region.clone()))
		.or_else(|| std::env::var("AWS_REGION").ok().filter(|value| !value.is_empty()))
		.or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()));
	let Some(region) = region else {
		return Err("Set AWS_REGION or pass a signing region for the Bedrock proxy.".to_string());
	};
	if let (Some(endpoint), Some(options)) = (endpoint.as_ref(), options) {
		if let Some(explicit_region) = options.region.as_ref() {
			if explicit_region != &endpoint.region {
				return Err(format!(
					"Bedrock endpoint region {} does not match signing region {}.",
					endpoint.region, explicit_region
				));
			}
		}
	}
	if let Some(endpoint) = endpoint.as_ref() {
		if (endpoint.service == "bedrock-runtime") != runtime_model {
			return Err(
				"Use openai.gpt-6-astra with Mantle, or a global./us. inference profile with Bedrock Runtime."
					.to_string(),
			);
		}
	}
	if model.id == "openai.gpt-6-astra" {
		if let Some(endpoint) = endpoint.as_ref() {
			if endpoint.region != "us-west-2" {
				return Err("GPT-6 Astra on Bedrock Mantle requires us-west-2 (Oregon).".to_string());
			}
		}
	}
	let explicit_key = options
		.and_then(|options| options.stream.api_key.clone())
		.filter(|key| key != "<authenticated>");
	let has_profile = options.and_then(|options| options.profile.clone()).is_some();
	let has_credential_provider = options.map(|options| options.has_credential_provider).unwrap_or(false);
	if explicit_key.is_some() && (has_profile || has_credential_provider) {
		return Err("Choose either a Bedrock bearer token or explicit AWS credentials.".to_string());
	}
	let bearer_token = explicit_key.or_else(|| {
		if !has_profile && !has_credential_provider {
			std::env::var("AWS_BEARER_TOKEN_BEDROCK")
				.ok()
				.filter(|value| !value.is_empty())
		} else {
			None
		}
	});
	let signer = if bearer_token.is_some() {
		None
	} else {
		Some(BedrockResponsesSigner {
			region: region.clone(),
			service: service.to_string(),
		})
	};
	let mut default_headers: IndexMap<String, Option<String>> = IndexMap::new();
	if let Some(model_headers) = model.headers.as_ref() {
		for (key, value) in model_headers {
			default_headers.insert(key.clone(), Some(value.clone()));
		}
	}
	if let Some(options) = options {
		if let Some(headers) = options.stream.headers.as_ref() {
			for (key, value) in headers {
				default_headers.insert(key.clone(), Some(value.clone()));
			}
		}
	}
	if default_headers
		.keys()
		.any(|key| key.to_lowercase() == "authorization")
	{
		return Err("Use Bedrock apiKey or AWS credentials instead of an Authorization header.".to_string());
	}
	let origin = match url.port() {
		Some(port) => format!("{}://{}:{}", url.scheme(), url.host_str().unwrap_or_default(), port),
		None => format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default()),
	};

	Ok(BedrockResponsesClient {
		base_url,
		api_key: bearer_token
			.clone()
			.unwrap_or_else(|| "<aws-sigv4>".to_string()),
		bearer_token,
		signer,
		default_headers,
		max_retries: 0,
		origin,
	})
}

/// `/^(bedrock-mantle|bedrock-runtime)\.([a-z0-9-]+)\.(?:api\.aws|amazonaws\.com)$/`
#[derive(Debug, Clone, PartialEq)]
pub struct BedrockEndpointMatch {
	pub service: String,
	pub region: String,
}

fn match_endpoint_host(hostname: &str) -> Option<BedrockEndpointMatch> {
	let lower = hostname.to_lowercase();
	let rest = lower
		.strip_prefix("bedrock-mantle.")
		.map(|rest| ("bedrock-mantle", rest))
		.or_else(|| lower.strip_prefix("bedrock-runtime.").map(|rest| ("bedrock-runtime", rest)))?;
	let (service, rest) = rest;
	let region = rest
		.strip_suffix(".api.aws")
		.or_else(|| rest.strip_suffix(".amazonaws.com"))?;
	if region.is_empty() || !region.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
		return None;
	}
	Some(BedrockEndpointMatch {
		service: service.to_string(),
		region: region.to_string(),
	})
}

/// `/^(?:global|us)\.openai\./`
fn is_runtime_model(model_id: &str) -> bool {
	model_id.starts_with("global.openai.") || model_id.starts_with("us.openai.")
}

/// The headers the `fetch` wrapper sets before sending: SigV4 or bearer.
pub fn build_signed_request_headers(
	client: &BedrockResponsesClient,
	target_url: &str,
	method: &str,
	headers: &IndexMap<String, String>,
	body: &str,
	options: Option<&BedrockResponsesAuthOptions>,
) -> Result<IndexMap<String, String>, String> {
	let target = url::Url::parse(target_url).map_err(|_| "Invalid URL".to_string())?;
	let target_origin = match target.port() {
		Some(port) => format!(
			"{}://{}:{}",
			target.scheme(),
			target.host_str().unwrap_or_default(),
			port
		),
		None => format!("{}://{}", target.scheme(), target.host_str().unwrap_or_default()),
	};
	if target_origin != client.origin {
		return Err("Refusing to send AWS credentials outside the configured Bedrock endpoint.".to_string());
	}

	let mut outgoing: IndexMap<String, String> = headers
		.iter()
		.filter(|(key, _)| key.to_lowercase() != "authorization")
		.map(|(key, value)| (key.clone(), value.clone()))
		.collect();

	if let Some(bearer_token) = client.bearer_token.clone() {
		outgoing.insert("authorization".to_string(), format!("Bearer {}", bearer_token));
	} else if let Some(signer) = client.signer.clone() {
		outgoing.insert("host".to_string(), target.host_str().unwrap_or_default().to_string());
		outgoing.shift_remove("x-amz-date");
		outgoing.shift_remove("x-amz-security-token");
		outgoing.shift_remove("x-amz-content-sha256");

		let credentials = match options.and_then(|options| options.credentials.clone()) {
			Some(credentials) => credentials,
			None => resolve_aws_credentials(&crate::providers::amazon_bedrock::BedrockOptions {
				profile: options.and_then(|options| options.profile.clone()),
				..Default::default()
			})
			.map_err(|error| error.to_string())?,
		};

		let mut query: Vec<(String, String)> = Vec::new();
		for (key, value) in target.query_pairs() {
			query.push((key.to_string(), value.to_string()));
		}

		let signed = sign_request_for_service(
			method,
			target.path(),
			&query,
			&outgoing,
			body,
			&signer.region,
			&signer.service,
			&credentials,
			now_ms(),
		);
		for (key, value) in signed.headers {
			outgoing.insert(key, value);
		}
	}

	Ok(outgoing)
}

impl BedrockResponsesClient {
	/// `client.responses.create(params, requestOptions)` for the Bedrock proxy: POST
	/// `<baseURL>/responses` with the SigV4 signature (or the bearer token) and return the
	/// raw response so the caller can read the status, headers and SSE body.
	pub async fn send_responses(
		&self,
		params: &serde_json::Map<String, Value>,
		options: Option<&BedrockResponsesAuthOptions>,
	) -> Result<reqwest::Response, String> {
		send_signed_responses_request(self, &Value::Object(params.clone()), options).await
	}
}

/// `globalThis.fetch(new Request(request, { headers, redirect: "manual" }))` for the
/// `/responses` endpoint: POST with the signed headers and the JSON body.
pub async fn send_signed_responses_request(
	client: &BedrockResponsesClient,
	params: &Value,
	options: Option<&BedrockResponsesAuthOptions>,
) -> Result<reqwest::Response, String> {
	let url = format!("{}/responses", client.base_url.trim_end_matches('/'));
	let body = params.to_string();

	let mut headers: IndexMap<String, String> = IndexMap::new();
	headers.insert("content-type".to_string(), "application/json".to_string());
	for (key, value) in client.default_headers.iter() {
		let Some(value) = value else {
			continue;
		};
		headers.insert(key.clone(), value.clone());
	}
	headers.insert(
		"host".to_string(),
		url::Url::parse(&url)
			.ok()
			.and_then(|url| url.host_str().map(str::to_string))
			.unwrap_or_default(),
	);

	let signed = build_signed_request_headers(client, &url, "POST", &headers, &body, options)?;

	let mut request = crate::providers::shared_http::try_shared_client(crate::providers::shared_http::ClientPolicy::RedirectNone)?
		.post(&url)
		.body(body);
	for (key, value) in signed {
		request = request.header(key, value);
	}
	if let Some(timeout_ms) = options.and_then(|options| options.stream.timeout_ms) {
		request = request.timeout(Duration::from_millis(timeout_ms.max(0.0) as u64));
	}

	let send = request.send();
	let response = match options.and_then(|options| options.stream.signal.as_ref()) {
		Some(signal) => tokio::select! {
			_ = signal.cancelled() => return Err("Request was aborted".to_string()),
			result = send => result,
		},
		None => send.await,
	};
	response.map_err(|error| error.to_string())
}

/// Test-only serialization for `AWS_*` environment access.
///
/// The transport tests in this module and the run-body tests in `amazon_bedrock_responses.rs`
/// both observe the process-global `AWS_*` variables, so they must share one lock. Tests only:
/// no production code touches this.
///
/// The lock itself lives in `crate::test_env`, which every `AWS_*`-writing test in this crate
/// takes; `amazon_bedrock_responses.rs` holds it directly and the writing tests of this module
/// hold it through `CleanAwsEnv`.
#[cfg(test)]
pub(crate) fn aws_env_test_lock() -> &'static std::sync::Mutex<()> {
	crate::test_env::env_lock()
}

/// The boxed response byte stream used by `responses_event_stream`.
pub type BedrockByteStream =
	std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// One decoded SSE frame, or the failure that ends the body read.
///
/// The TypeScript reads the body through the OpenAI SDK's `Stream.fromSSEResponse`
/// (`amazon-bedrock-responses.ts:59` -> `openai/resources/responses/responses.ts`
/// `this._client.responses.create(...)`), which:
///
/// * `throw`s `new Error(\`Could not parse message into JSON: ...\`)` for a frame whose
///   `data:` payload is not JSON, and
/// * `throw`s `new APIError(undefined, data.error, undefined, response.headers)` for a frame
///   whose parsed payload carries a truthy `error`.
///
/// A Rust stream cannot `throw`, so the failure travels as the `Err` item of the stream and
/// the owner (`amazon_bedrock_responses.rs` run body) re-raises it after the shared loop ends -
/// the same catch boundary and the same message. This mirrors `parse_sse_response`
/// (`openai_codex_responses.rs:1200-1265`, `Ok`/`Err` items re-raised by `map_codex_events`).
pub enum BedrockSseFrame {
	/// A parsed `data:` JSON payload.
	Event(Value),
	/// `throw new Error(...)` - the payload was not JSON.
	Message(String),
	/// `throw new APIError(undefined, data.error, undefined, response.headers)` - the payload
	/// nested a truthy `error` object (`data.error`), the in-stream API-error shape.
	ApiError(Value),
}

/// The local SSE reader for the Responses transport (`data: ...` frames).
///
/// CR, LF and CRLF terminate lines; `data:` lines are joined with `\n` and `[DONE]`
/// is ignored. Anything that is not JSON or that
/// carries a truthy `error` is an error rather than a silently dropped frame.
#[derive(Default)]
pub struct BedrockResponsesSseBuffer {
    frames: SseFrames,
}

impl BedrockResponsesSseBuffer {
	/// Appends a chunk and returns the frames it completed, in order.
	pub fn push(&mut self, bytes: &[u8]) -> Vec<BedrockSseFrame> {
        self.frames.push(bytes).iter().map(|data| decode_bedrock_sse_frame(data)).collect()
	}

	/// The frames still inside an incomplete trailing chunk.
	///
	/// The SDK's decoder is fed by the whole body and has no "unfinished frame" case, but a
	/// Rust body ends on the last byte, so `push` must have the same behaviour for the tail
	/// it never got a delimiter for. Without this, a truncated or un-delimited final frame
	/// would be kept in `buffer` forever and silently reported as a complete response.
	pub fn finish(&mut self) -> Vec<BedrockSseFrame> {
        self.frames.finish().iter().map(|data| decode_bedrock_sse_frame(data)).collect()
	}
}

/// Decode one complete `data:` payload, keeping the SDK's two failure modes.
fn decode_bedrock_sse_frame(data: &str) -> BedrockSseFrame {
	let parsed = match serde_json::from_str::<Value>(data) {
		Ok(parsed) => parsed,
		Err(cause) => {
			// SDK `Stream.fromSSEResponse`: `throw new Error(\`Could not parse message into
			// JSON: ...\`)`. This tree does not vendor `openai`'s `streaming.ts`, so the exact
			// template cannot be quoted from source; the port keeps the same throw and names
			// the real symptom - the payload that failed - plus the JSON parse error.
			return BedrockSseFrame::Message(format!(
				"Could not parse message into JSON: {} (JSON parse error: {})",
				data, cause
			));
		}
	};
	// SDK: `if (data && data.error) throw new APIError(undefined, data.error, undefined,
	// response.headers)`. `js_truthy` is the Rust port of the JavaScript truthiness test the
	// SDK uses, so an explicit JSON `null`/`false`/`0`/`""` must NOT throw.
	if let Some(error) = parsed.get("error").filter(|error| js_truthy(error)) {
		return BedrockSseFrame::ApiError(error.clone());
	}
	BedrockSseFrame::Event(parsed)
}

/// The JavaScript truthiness test (`Boolean(value)`) for a parsed JSON value.
///
/// Same semantics as `openai_completions.rs:32-45`: only `false`, `0`, `""`, `null` and
/// `undefined` are falsy; every object and array is truthy.
fn js_truthy(value: &Value) -> bool {
	match value {
		Value::Null => false,
		Value::Bool(value) => *value,
		Value::Number(number) => number.as_f64().map(|number| number != 0.0).unwrap_or(true),
		Value::String(text) => !text.is_empty(),
		Value::Array(_) | Value::Object(_) => true,
	}
}

/// Drain a `reqwest` response body into a `processResponsesStream` input stream, storing the
/// first body failure in `error_slot` for the run body to re-raise.
///
/// Unlike `unfold` returning `None` for `Some(Err(_))` - which silently truncated the
/// response and let it be reported as a complete answer - a body read error, an unparseable
/// frame and a `data.error` frame all END the stream with the failure preserved.
pub fn responses_event_stream(
	response: reqwest::Response,
	error_slot: std::sync::Arc<std::sync::Mutex<Option<ResponsesRunError>>>,
) -> crate::providers::openai_responses_shared::ResponsesEventStream {
	let byte_stream: BedrockByteStream = Box::pin(response.bytes_stream());
	Box::pin(futures::stream::unfold(
		(
			byte_stream,
			BedrockResponsesSseBuffer::default(),
			std::collections::VecDeque::<BedrockSseFrame>::new(),
			false,
			error_slot,
		),
		|(mut byte_stream, mut buffer, mut pending, mut finished, error_slot)| async move {
			loop {
				match pending.pop_front() {
					Some(BedrockSseFrame::Event(event)) => {
						return Some((event, (byte_stream, buffer, pending, finished, error_slot)));
					}
					Some(failure) => {
						// `throw` out of the `for await` loop: record the thrown value and end
						// the stream; the run body re-raises it after `processResponsesStream`
						// returns (the provider `catch`, `amazon-bedrock-responses.ts:79-90`).
                        record_bedrock_sse_failure(&error_slot, failure);
                        pending.clear();
                        finished = true;
						continue;
					}
					None if finished => return None,
					None => {}
				}
				match byte_stream.next().await {
					Some(Ok(bytes)) => pending.extend(buffer.push(&bytes)),
					Some(Err(error)) => {
						// The TypeScript body read rejects (`body` stream error) and the SDK
						// surfaces it; it must not masquerade as a completed stream.
						pending.push_back(BedrockSseFrame::Message(error.to_string()));
					}
					None => {
						// End of body: decode whatever the last chunk left behind.
						pending.extend(buffer.finish());
						finished = true;
					}
				}
			}
		},
	))
}

/// Store a stream failure once, keeping the FIRST one (the thrown value the TypeScript `catch`
/// would have seen). Only the first failure reached the `throw`, so later frames cannot
/// replace it.
fn record_bedrock_sse_failure(
	slot: &std::sync::Arc<std::sync::Mutex<Option<ResponsesRunError>>>,
	failure: BedrockSseFrame,
) {
	let value = match failure {
		BedrockSseFrame::Message(message) => ResponsesRunError::Message(message),
		BedrockSseFrame::ApiError(error) => ResponsesRunError::ApiError(error),
		// `record_bedrock_sse_failure` is only called for the non-event variants.
		BedrockSseFrame::Event(_) => return,
	};
	let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
	if slot.is_none() {
		*slot = Some(value);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::Model;

	/// The client reads `AWS_*` environment variables; tests that touch them must not run
	/// concurrently with tests that depend on them.
	///
	/// The lock is the crate-wide one from [`crate::test_env`], so this also serialises the
	/// `AWS_*` tests in `env_api_keys.rs`, `amazon_bedrock.rs` and
	/// `amazon_bedrock_responses.rs`; a per-module lock would still let those interleave.
	struct CleanAwsEnv {
		env: crate::test_env::ScopedEnv,
	}

	const AWS_ENV_NAMES: [&str; 7] = [
		"AWS_BEDROCK_BASE_URL",
		"AWS_REGION",
		"AWS_DEFAULT_REGION",
		"AWS_BEARER_TOKEN_BEDROCK",
		"AWS_PROFILE",
		"AWS_ACCESS_KEY_ID",
		"AWS_SECRET_ACCESS_KEY",
	];

	impl CleanAwsEnv {
		fn new() -> Self {
			// ScopedEnv holds the process-wide environment lock for the whole test body and
			// journals each previous value (or its absence) for restore on drop.
			let mut env = crate::test_env::ScopedEnv::new();
			for name in AWS_ENV_NAMES {
				env.remove(name);
			}
			Self { env }
		}

		/// Set `name` while the lock is held, journaled for restore on drop.
		fn set(&mut self, name: &str, value: impl AsRef<std::ffi::OsStr>) {
			self.env.set(name, value);
		}

		/// Remove `name` while the lock is held, journaled for restore on drop.
		fn remove(&mut self, name: &str) {
			self.env.remove(name);
		}
	}

	fn model(id: &str, base_url: &str) -> Model {
		Model::new(id, id, "bedrock-responses", "amazon-bedrock", base_url)
	}

	#[test]
	fn endpoint_host_pattern_matches_mantle_and_runtime() {
		assert_eq!(
			match_endpoint_host("bedrock-mantle.us-west-2.api.aws"),
			Some(BedrockEndpointMatch {
				service: "bedrock-mantle".to_string(),
				region: "us-west-2".to_string()
			})
		);
		assert_eq!(
			match_endpoint_host("bedrock-runtime.us-east-1.amazonaws.com"),
			Some(BedrockEndpointMatch {
				service: "bedrock-runtime".to_string(),
				region: "us-east-1".to_string()
			})
		);
		assert_eq!(match_endpoint_host("bedrock-runtime.us-east-1.example.com"), None);
		assert_eq!(match_endpoint_host("api.openai.com"), None);
		assert_eq!(match_endpoint_host("bedrock-mantle.US-WEST-2.API.AWS"), match_endpoint_host("bedrock-mantle.us-west-2.api.aws"));
	}

	#[test]
	fn runtime_models_are_global_and_us_openai_profiles() {
		assert!(is_runtime_model("global.openai.gpt-6-astra"));
		assert!(is_runtime_model("us.openai.gpt-6-astra"));
		assert!(!is_runtime_model("openai.gpt-6-astra"));
		assert!(!is_runtime_model("eu.openai.gpt-6-astra"));
	}

	#[test]
	fn base_url_rejects_credentials_query_and_fragment() {
		let _env = CleanAwsEnv::new();
		let error = create_bedrock_responses_client(&model("global.openai.gpt-6-astra", "https://x.api.aws/openai/v1?a=1"), None)
			.unwrap_err();
		assert_eq!(
			error,
			"Bedrock base URL must be an HTTP(S) API root without credentials, query, or fragment."
		);
		let error = create_bedrock_responses_client(&model("global.openai.gpt-6-astra", "ftp://x.api.aws/openai/v1"), None)
			.unwrap_err();
		assert_eq!(
			error,
			"Bedrock base URL must be an HTTP(S) API root without credentials, query, or fragment."
		);
	}

	#[test]
	fn signing_region_comes_from_option_endpoint_or_env() {
		let mut env = CleanAwsEnv::new();
		env.remove("AWS_REGION");
		env.remove("AWS_DEFAULT_REGION");

		let mut options = BedrockResponsesAuthOptions::default();
		options.region = Some("eu-west-1".to_string());
		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://proxy.example.com/openai/v1"),
			Some(&options),
		)
		.unwrap();
		assert_eq!(client.signer.as_ref().unwrap().region, "eu-west-1");

		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.ap-southeast-2.amazonaws.com/openai/v1"),
			None,
		)
		.unwrap();
		assert_eq!(client.signer.as_ref().unwrap().region, "ap-southeast-2");
		assert_eq!(client.signer.as_ref().unwrap().service, "bedrock");
		assert_eq!(client.api_key, "<aws-sigv4>");

		let error = create_bedrock_responses_client(&model("m", "https://example.com/v1"), None).unwrap_err();
		assert_eq!(error, "Set AWS_REGION or pass a signing region for the Bedrock proxy.");
	}

	#[test]
	fn endpoint_region_must_match_the_signing_region() {
		let _env = CleanAwsEnv::new();
		let mut options = BedrockResponsesAuthOptions::default();
		options.region = Some("us-east-1".to_string());
		let error = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.ap-southeast-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap_err();
		assert_eq!(
			error,
			"Bedrock endpoint region ap-southeast-2 does not match signing region us-east-1."
		);
	}

	#[test]
	fn runtime_and_mantle_endpoints_must_match_the_model() {
		let _env = CleanAwsEnv::new();
		let error = create_bedrock_responses_client(
			&model("openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			None,
		)
		.unwrap_err();
		assert_eq!(
			error,
			"Use openai.gpt-6-astra with Mantle, or a global./us. inference profile with Bedrock Runtime."
		);

		let error = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-mantle.us-west-2.api.aws/openai/v1"),
			None,
		)
		.unwrap_err();
		assert_eq!(
			error,
			"Use openai.gpt-6-astra with Mantle, or a global./us. inference profile with Bedrock Runtime."
		);
	}

	#[test]
	fn astra_on_mantle_requires_oregon() {
		let _env = CleanAwsEnv::new();
		let error = create_bedrock_responses_client(
			&model("openai.gpt-6-astra", "https://bedrock-mantle.us-east-1.api.aws/openai/v1"),
			None,
		)
		.unwrap_err();
		assert_eq!(error, "GPT-6 Astra on Bedrock Mantle requires us-west-2 (Oregon).");

		let client = create_bedrock_responses_client(
			&model("openai.gpt-6-astra", "https://bedrock-mantle.us-west-2.api.aws/openai/v1"),
			None,
		)
		.unwrap();
		assert_eq!(client.signer.as_ref().unwrap().service, "bedrock-mantle");
	}

	#[test]
	fn bearer_token_and_explicit_credentials_are_mutually_exclusive() {
		let mut env = CleanAwsEnv::new();
		env.remove("AWS_BEARER_TOKEN_BEDROCK");
		let mut options = BedrockResponsesAuthOptions::default();
		options.stream.api_key = Some("token".to_string());
		options.profile = Some("profile".to_string());
		let error = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap_err();
		assert_eq!(error, "Choose either a Bedrock bearer token or explicit AWS credentials.");

		let mut options = BedrockResponsesAuthOptions::default();
		options.stream.api_key = Some("token".to_string());
		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap();
		assert_eq!(client.api_key, "token");
		assert_eq!(client.bearer_token.as_deref(), Some("token"));
		assert!(client.signer.is_none());
		assert_eq!(client.max_retries, 0);

		// "<authenticated>" from the auth store is not an explicit bearer token.
		let mut options = BedrockResponsesAuthOptions::default();
		options.stream.api_key = Some("<authenticated>".to_string());
		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap();
		assert_eq!(client.api_key, "<aws-sigv4>");
		assert!(client.bearer_token.is_none());
	}

	#[test]
	fn default_headers_merge_model_then_options_and_reject_authorization() {
		let _env = CleanAwsEnv::new();
		let mut with_model_headers =
			model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1");
		with_model_headers.headers = Some(IndexMap::from([("x-model".to_string(), "1".to_string())]));
		let mut options = BedrockResponsesAuthOptions::default();
		options.stream.headers = Some(IndexMap::from([
			("x-option".to_string(), "2".to_string()),
			("x-model".to_string(), "3".to_string()),
		]));
		let client = create_bedrock_responses_client(&with_model_headers, Some(&options)).unwrap();
		assert_eq!(client.default_headers.get("x-model"), Some(&Some("3".to_string())));
		assert_eq!(client.default_headers.get("x-option"), Some(&Some("2".to_string())));

		let mut with_authorization =
			model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1");
		with_authorization.headers = Some(IndexMap::from([("Authorization".to_string(), "Bearer x".to_string())]));
		let error = create_bedrock_responses_client(&with_authorization, None).unwrap_err();
		assert_eq!(error, "Use Bedrock apiKey or AWS credentials instead of an Authorization header.");
	}

	#[test]
	fn base_url_env_override_wins_over_the_model() {
		let mut env = CleanAwsEnv::new();
		env.set("AWS_BEDROCK_BASE_URL", "  https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1  ");
		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			None,
		)
		.unwrap();
		assert_eq!(
			client.base_url,
			"https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1"
		);
		assert_eq!(client.signer.as_ref().unwrap().region, "eu-central-1");
		env.remove("AWS_BEDROCK_BASE_URL");
	}

	#[test]
	fn sigv4_headers_are_added_for_the_target_origin_only() {
		let _env = CleanAwsEnv::new();
		let credentials = AwsCredentials {
			access_key_id: "AKID".to_string(),
			secret_access_key: "secret".to_string(),
			session_token: Some("token".to_string()),
		};
		let mut options = BedrockResponsesAuthOptions::default();
		options.credentials = Some(credentials);
		options.has_credential_provider = true;

		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap();
		assert!(client.bearer_token.is_none());

		let mut headers: IndexMap<String, String> = IndexMap::new();
		headers.insert("content-type".to_string(), "application/json".to_string());
		headers.insert("host".to_string(), "bedrock-runtime.us-west-2.amazonaws.com".to_string());
		headers.insert("authorization".to_string(), "Bearer stale".to_string());

		let signed = build_signed_request_headers(
			&client,
			"https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/responses",
			"POST",
			&headers,
			"{\"model\":\"m\"}",
			Some(&options),
		)
		.unwrap();
		assert_eq!(signed["x-amz-security-token"], "token");
		assert!(signed["authorization"].starts_with("AWS4-HMAC-SHA256 Credential=AKID/"));
		assert!(signed["authorization"].contains("/us-west-2/bedrock/aws4_request"));
		assert_eq!(signed["x-amz-content-sha256"], crate::providers::amazon_bedrock::sha256_hex(b"{\"model\":\"m\"}"));

		let error = build_signed_request_headers(
			&client,
			"https://evil.example.com/openai/v1/responses",
			"POST",
			&headers,
			"{}",
			Some(&options),
		)
		.unwrap_err();
		assert_eq!(
			error,
			"Refusing to send AWS credentials outside the configured Bedrock endpoint."
		);
	}

	#[test]
	fn bearer_tokens_are_sent_as_authorization_headers() {
		let _env = CleanAwsEnv::new();
		let mut options = BedrockResponsesAuthOptions::default();
		options.stream.api_key = Some("token".to_string());
		let client = create_bedrock_responses_client(
			&model("global.openai.gpt-6-astra", "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"),
			Some(&options),
		)
		.unwrap();

		let mut headers: IndexMap<String, String> = IndexMap::new();
		headers.insert("host".to_string(), "bedrock-runtime.us-west-2.amazonaws.com".to_string());
		let signed = build_signed_request_headers(
			&client,
			"https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/responses",
			"POST",
			&headers,
			"{}",
			Some(&options),
		)
		.unwrap();
		assert_eq!(signed["authorization"], "Bearer token");
		assert!(signed.get("x-amz-date").is_none());
	}

	/// Serves one raw HTTP response, then closes the connection after `body` bytes.
	async fn serve_raw_response(
		head: String,
		body: Vec<u8>,
	) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut buffer = [0u8; 4096];
			// Drain the request head (and body, if the client sent one) before answering.
			let mut seen = Vec::new();
			loop {
				let count = socket.read(&mut buffer).await.unwrap();
				if count == 0 {
					return;
				}
				seen.extend_from_slice(&buffer[..count]);
				if seen.windows(4).any(|window| window == b"\r\n\r\n") {
					break;
				}
			}
			socket.write_all(head.as_bytes()).await.unwrap();
			socket.write_all(&body).await.unwrap();
			socket.flush().await.unwrap();
			// Drop closes the connection: a body shorter than `Content-Length` is a read error.
		});
		(address, server)
	}

	#[tokio::test]
	async fn body_read_error_is_surfaced_instead_of_ending_the_stream() {
		// F3/F7: the previous `unfold` mapped `Some(Err(_))` and the tail of the body to
		// `None` (end of stream), so a truncated response was reported to the caller as a
		// completed answer. The TypeScript body read rejects and the SDK surfaces it, so the
		// failure must end the stream WITH the thrown value preserved.
		let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 400\r\nConnection: close\r\n\r\n".to_string();
		// One complete frame, then a partial one; the body ends 360 bytes short of what the
		// headers promised, which is exactly the dropped-connection case.
		let body = b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\ndata: {\"type\":\"response.comp".to_vec();
		let (address, server) = serve_raw_response(head, body).await;
		let response = reqwest::Client::new()
			.get(format!("http://{address}/probe"))
			.send()
			.await
			.unwrap();
		let error_slot = std::sync::Arc::new(std::sync::Mutex::new(None));
		let mut events = responses_event_stream(response, error_slot.clone());

		let mut delivered = Vec::new();
		while let Some(event) = events.next().await {
			delivered.push(event);
		}
		assert_eq!(delivered.len(), 1, "the complete frame is still delivered");
		assert_eq!(delivered[0]["response"]["id"], "r1");

		let recorded = error_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
		match recorded {
			Some(ResponsesRunError::Message(message)) => {
				assert!(
					!message.is_empty(),
					"the body read failure must carry the transport error text"
				);
			}
			Some(ResponsesRunError::ApiError(value)) => {
				panic!("a body read failure is not an SDK APIError: {value}");
			}
			Some(ResponsesRunError::Failure(failure)) => {
				panic!("a body read failure is not a stream failure: {}", failure.message);
			}
			None => panic!(
				"a truncated body must surface an error, not be reported as a completed stream"
			),
		}
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[tokio::test]
	async fn unparseable_and_error_frames_reach_the_error_slot() {
		// F7: the SDK `throw`s for an unparseable `data:` payload and for a truthy `data.error`,
		// so the run body must see a thrown value and fail the request instead of silently
		// dropping the frames and reporting a completed (empty) answer.
		for (payload, expected) in [
			(
                "data: {\"type\":\"response.created\"}\n\ndata: {not json}\n\ndata: {\"type\":\"must-not-deliver\"}\n\n",
				"Could not parse message into JSON",
			),
			(
                "data: {\"type\":\"response.created\"}\n\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\ndata: {\"type\":\"must-not-deliver\"}\n\n",
				"overloaded_error",
			),
		] {
			let head = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
				payload.len()
			);
			let (address, server) = serve_raw_response(head, payload.as_bytes().to_vec()).await;
			let response = reqwest::Client::new()
				.get(format!("http://{address}/probe"))
				.send()
				.await
				.unwrap();
			let error_slot = std::sync::Arc::new(std::sync::Mutex::new(None));
			let mut events = responses_event_stream(response, error_slot.clone());
			let mut delivered = 0;
			while events.next().await.is_some() {
				delivered += 1;
			}
			assert_eq!(delivered, 1, "the leading valid frame is still delivered: {payload}");

			let recorded = error_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
			let rendered = match recorded {
				Some(ResponsesRunError::Message(message)) => message,
				Some(ResponsesRunError::ApiError(value)) => value.to_string(),
				Some(ResponsesRunError::Failure(failure)) => failure.message,
				None => panic!("no error recorded for payload: {payload}"),
			};
			assert!(
				rendered.contains(expected),
				"error must name the real symptom ({expected}): {rendered}"
			);
			tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
		}
	}

	#[test]
	fn sse_buffer_decodes_data_frames() {
		let mut buffer = BedrockResponsesSseBuffer::default();
		assert!(buffer.push(b"data: {\"type\":\"response.crea").is_empty());
		let events = buffer.push(b"ted\",\"response\":{\"id\":\"r1\"}}\n\ndata: [DONE]\n\n");
		assert_eq!(events.len(), 1);
		match &events[0] {
			BedrockSseFrame::Event(event) => assert_eq!(event["response"]["id"], "r1"),
			BedrockSseFrame::Message(message) => panic!("unexpected decode failure: {message}"),
			BedrockSseFrame::ApiError(error) => panic!("unexpected api error: {error}"),
		}
	}

    #[test]
    fn sse_buffer_preserves_unicode_for_every_chunk_split_and_line_ending() {
        for delimiter in ["\n", "\r\n", "\r"] {
            let wire = format!("data: {{\"type\":\"response.output_text.delta\",\"delta\":\"Ready ✓ 日本\"}}{delimiter}{delimiter}");
            for split in 0..=wire.len() {
                let mut buffer = BedrockResponsesSseBuffer::default();
                let mut frames = buffer.push(&wire.as_bytes()[..split]);
                frames.extend(buffer.push(&wire.as_bytes()[split..]));
                assert_eq!(frames.len(), 1, "complete frame must not need finish()");
                match frames.pop().unwrap() {
                    BedrockSseFrame::Event(event) => assert_eq!(event["delta"], "Ready ✓ 日本"),
                    _ => panic!("valid UTF-8 frame must not fail"),
                }
                assert!(buffer.finish().is_empty());
            }
        }
    }

	#[test]
	fn sse_buffer_surfaces_unparseable_frames_instead_of_dropping_them() {
		// SDK `Stream.fromSSEResponse`: `throw new Error(\`Could not parse message into JSON: ...\`)`.
		// The old port dropped the frame (so the response looked complete and empty).
		let mut buffer = BedrockResponsesSseBuffer::default();
		let frames = buffer.push(b"data: {not json\n\n");
		assert_eq!(frames.len(), 1);
		match &frames[0] {
			BedrockSseFrame::Message(message) => {
				assert!(
					message.starts_with("Could not parse message into JSON: {not json"),
					"{message}"
				);
				assert!(message.contains("JSON parse error"), "{message}");
			}
			BedrockSseFrame::Event(event) => panic!("unparseable frame became an event: {event}"),
			BedrockSseFrame::ApiError(error) => panic!("unparseable frame became an api error: {error}"),
		}
	}

	#[test]
	fn sse_buffer_surfaces_nested_error_payloads_as_api_errors() {
		// SDK: `if (data && data.error) throw new APIError(undefined, data.error, undefined,
		// response.headers);`. The old port dropped it and reported a completed stream.
		let mut buffer = BedrockResponsesSseBuffer::default();
		let frames = buffer.push(
			b"data: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
		);
		assert_eq!(frames.len(), 1);
		match &frames[0] {
			BedrockSseFrame::ApiError(error) => assert_eq!(error["type"], "overloaded_error"),
			BedrockSseFrame::Event(event) => panic!("data.error frame became an event: {event}"),
			BedrockSseFrame::Message(message) => panic!("data.error frame became a parse error: {message}"),
		}

		// Falsy `error` values do NOT throw (`if (data && data.error)`), so `null`, `false`,
		// `0` and `""` stay ordinary events.
		for falsy in [b"null".as_slice(), b"false", b"0", b"\"\""] {
			let payload = format!("data: {{\"type\":\"response.created\",\"error\":{}}}\n\n", String::from_utf8_lossy(falsy));
			let mut buffer = BedrockResponsesSseBuffer::default();
			let frames = buffer.push(payload.as_bytes());
			assert_eq!(frames.len(), 1);
			assert!(matches!(frames[0], BedrockSseFrame::Event(_)), "falsy error must not throw");
		}
	}

	#[test]
	fn sse_buffer_finishes_the_trailing_frame_at_end_of_body() {
		// A body that ends without the `\n\n` delimiter must still be decoded, otherwise the
		// frame stays buffered forever and the truncation is reported as success.
		let mut buffer = BedrockResponsesSseBuffer::default();
		assert!(buffer.push(b"data: {\"type\":\"response.comp").is_empty());
		let frames = buffer.finish();
		assert_eq!(frames.len(), 1);
		assert!(matches!(frames[0], BedrockSseFrame::Message(_)), "a partial frame is not valid JSON");

		let mut buffer = BedrockResponsesSseBuffer::default();
		buffer.push(b"data: {\"type\":\"response.completed\"}");
		let frames = buffer.finish();
		assert_eq!(frames.len(), 1);
		match &frames[0] {
			BedrockSseFrame::Event(event) => assert_eq!(event["type"], "response.completed"),
			_ => panic!("a complete trailing frame must decode"),
		}

		// `[DONE]` and an empty tail produce nothing.
		let mut buffer = BedrockResponsesSseBuffer::default();
		buffer.push(b"data: [DONE]");
		assert!(buffer.finish().is_empty());
		#[allow(unused_mut)]
		let mut buffer = BedrockResponsesSseBuffer::default();
		assert!(buffer.finish().is_empty());
	}
}
