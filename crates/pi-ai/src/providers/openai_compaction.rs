//! Port of packages/ai/src/providers/openai-compaction.ts
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::compaction::{CompactionOptions, ProviderCompactionCheckpoint, ProviderCompactionResult};
use crate::models::calculate_cost;
use crate::types::{
	Api, Model, NativeCompactionValidation, Usage, NATIVE_COMPACTION_API_VERSION_V1,
};
use crate::utils::headers::header_map_to_record;
use crate::utils::stream_failure::parse_retry_after_ms;

const AZURE_MANAGED_PROVIDER: &str = "azure-openai-managed";
const AZURE_ASTRA_MODEL: &str = "gpt-6-astra";
const AZURE_GATEWAY_BASE_PATH: &str = "/azure-openai/v1";
const RESPONSES_COMPACT_PROTOCOL: &str = "openai-responses-compact-v1";

/// TS: `class CompactionRequestError extends Error`
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionRequestError {
	pub message: String,
	pub name: &'static str,
	pub status: i64,
	pub retry_after_ms: Option<f64>,
}

impl CompactionRequestError {
	pub fn new(message: impl Into<String>, status: i64, retry_after_ms: Option<f64>) -> Self {
		Self {
			message: message.into(),
			name: "CompactionRequestError",
			status,
			retry_after_ms,
		}
	}

	pub(crate) fn transport(message: impl Into<String>) -> Self {
		Self {
			message: message.into(),
			name: "TypeError",
			status: 0,
			retry_after_ms: None,
		}
	}
}

impl std::fmt::Display for CompactionRequestError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(formatter, "{}", self.message)
	}
}

impl std::error::Error for CompactionRequestError {}

/// TS: `normalizeEndpoint(value)`
fn normalize_endpoint(value: &str) -> Option<String> {
	let url = url::Url::parse(value).ok()?;
	if !matches!(url.scheme(), "http" | "https") {
		return None;
	}
	if !url.username().is_empty() || url.password().is_some() || url.query().is_some() || url.fragment().is_some() {
		return None;
	}
	let mut url = url;
	let path = url.path().trim_end_matches('/').to_string();
	url.set_path(&path);
	let serialized = url.to_string();
	Some(serialized.strip_suffix('/').unwrap_or(&serialized).to_string())
}

/// Return the exact endpoint only for the staged, live-validated Azure Astra route.
///
/// TS: `validatedNativeCompactionEndpoint(model)`
pub fn validated_native_compaction_endpoint(model: &Model) -> Option<String> {
	if model.provider != AZURE_MANAGED_PROVIDER || model.id != AZURE_ASTRA_MODEL || model.api != "openai-responses" {
		return None;
	}
	let capability = model.native_compaction.as_ref()?;
	if !capability.enabled
		|| capability.validation != NativeCompactionValidation::LiveVerified
		|| capability.protocol != RESPONSES_COMPACT_PROTOCOL
		|| capability.api_version != NATIVE_COMPACTION_API_VERSION_V1
		|| capability.provider != model.provider
		|| capability.model != model.id
	{
		return None;
	}
	let base_url = normalize_endpoint(&model.base_url)?;
	let endpoint = normalize_endpoint(&capability.endpoint)?;
	let parsed_base = url::Url::parse(&base_url).ok()?;
	if parsed_base.path() != AZURE_GATEWAY_BASE_PATH {
		return None;
	}
	let expected = normalize_endpoint(&format!("{base_url}/responses/compact"))?;
	if endpoint == expected {
		Some(endpoint)
	} else {
		None
	}
}

/// TS: `supportsOpenAICompaction(model)`
pub fn supports_openai_compaction(model: &Model) -> bool {
	(model.provider == "openai-codex" && model.api == "openai-codex-responses")
		|| (model.id == AZURE_ASTRA_MODEL && model.provider == "openai" && model.api == "openai-responses")
		|| validated_native_compaction_endpoint(model).is_some()
}

/// `headersToRecord` output reshaped into the `Value` `parse_retry_after_ms` reads,
/// mirroring the `Headers`-shaped input the TypeScript passes it.
fn header_record_to_value(headers: &IndexMap<String, String>) -> Value {
	Value::Object(
		headers
			.iter()
			.map(|(name, value)| (name.clone(), Value::String(value.clone())))
			.collect(),
	)
}

/// TS: `record(value)`
fn record(value: &Value) -> Option<&Map<String, Value>> {
	match value {
		Value::Object(map) => Some(map),
		_ => None,
	}
}

/// TS: `tokenCount(value)`
fn token_count(value: Option<&Value>) -> f64 {
	match value {
		Some(Value::Number(number)) => match number.as_f64() {
			Some(value) if value.is_finite() && value >= 0.0 => value,
			_ => 0.0,
		},
		_ => 0.0,
	}
}

/// TS: `estimatedWindowChars(items)`
///
/// The TypeScript uses `JSON.stringify` with a replacer that collapses base64 image URLs to
/// "(image)" and then adds 4800 characters per image. The port serialises with the same key
/// filter.
fn estimated_window_chars(items: &[Value]) -> f64 {
	let mut images = 0usize;
	let mut serialized = String::new();
	serialize_with_image_placeholder(&Value::Array(items.to_vec()), None, &mut images, &mut serialized);
	serialized.chars().count() as f64 + images as f64 * 4800.0
}

fn serialize_with_image_placeholder(value: &Value, key: Option<&str>, images: &mut usize, out: &mut String) {
	match value {
		Value::Object(map) => {
			out.push('{');
			let mut first = true;
			for (entry_key, entry_value) in map {
				if !first {
					out.push(',');
				}
				first = false;
				out.push_str(&serde_json::to_string(entry_key).unwrap_or_default());
				out.push(':');
				serialize_with_image_placeholder(entry_value, Some(entry_key), images, out);
			}
			out.push('}');
		}
		Value::Array(values) => {
			out.push('[');
			for (index, entry) in values.iter().enumerate() {
				if index > 0 {
					out.push(',');
				}
				serialize_with_image_placeholder(entry, key, images, out);
			}
			out.push(']');
		}
		Value::String(text) => {
			let collapsed = matches!(key, Some("image_url") | Some("url")) && text.starts_with("data:image/");
			if collapsed {
				*images += 1;
				out.push_str("\"(image)\"");
			} else {
				out.push_str(&serde_json::to_string(text).unwrap_or_default());
			}
		}
		Value::Null => out.push_str("null"),
		Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
		Value::Number(number) => out.push_str(&number.to_string()),
	}
}

/// TS: `decode(response) => response.json()` default.
pub type CompactionDecode =
	Arc<dyn Fn(reqwest::Response) -> BoxFuture<'static, Result<Value, CompactionRequestError>> + Send + Sync>;

/// TS: `requestOpenAICompaction(model, url, headers, body, options?, decode?)`
pub async fn request_openai_compaction(
	model: &Model,
	url: &str,
	headers: &IndexMap<String, String>,
	body: Map<String, Value>,
	options: Option<&CompactionOptions>,
	decode: Option<CompactionDecode>,
) -> Result<Option<ProviderCompactionResult>, CompactionRequestError> {
	let timeout_ms = options
		.and_then(|options| options.simple.stream.timeout_ms)
		.unwrap_or(1_200_000.0);
	let timeout = std::time::Duration::from_millis(timeout_ms.max(0.0) as u64);
	let signal = options.and_then(|options| options.simple.stream.signal.clone());

	// TS: `const replacement = await options?.onPayload?.(body, model);`
	let mut payload = body.clone();
	if let Some(on_payload) = options.and_then(|options| options.simple.stream.on_payload.clone()) {
		let replacement = on_payload(Value::Object(body.clone()), model).await;
		if let Some(replacement) = replacement {
			payload = match replacement {
				Value::Object(map) => map,
				_ => Map::new(),
			};
		}
	}

	let mut request_headers = reqwest::header::HeaderMap::new();
	for (key, value) in headers {
		if let (Ok(name), Ok(header_value)) = (
			reqwest::header::HeaderName::from_bytes(key.as_bytes()),
			reqwest::header::HeaderValue::from_str(value),
		) {
			request_headers.insert(name, header_value);
		}
	}

	let request = reqwest::Client::new()
		.post(url)
		.headers(request_headers)
		.timeout(timeout)
		.json(&Value::Object(payload));

	let send = request.send();
	let response = match signal.as_ref() {
		Some(signal) => tokio::select! {
			_ = signal.cancelled() => return Err(CompactionRequestError::new("Request was aborted", 0, None)),
			result = send => result,
		},
		None => send.await,
	};
	let response = response.map_err(|error| CompactionRequestError::transport(error.to_string()))?;

	if let Some(on_response) = options.and_then(|options| options.simple.stream.on_response.clone()) {
		on_response(
			crate::types::ProviderResponse {
				status: response.status().as_u16() as i64,
				headers: header_map_to_record(response.headers()),
			},
			model,
		)
		.await;
	}

	// TS: `parseRetryAfterMs(response.headers)` is read as a `Headers` getter, so the
	// Retry-After / Retry-After-Ms values are captured before the failure branch below
	// may consume the body (or `decode` may drop the response).
	let retry_after_ms = parse_retry_after_ms(Some(&header_record_to_value(&header_map_to_record(
		response.headers(),
	))));

	let status = response.status().as_u16() as i64;
	// Cancellation must cover body reads too, not only response headers. This owns
	// both custom SSE decoders and JSON/error bodies; dropping it closes the body.
	let read_payload = async {
		if matches!(status, 404 | 405 | 501) {
			return Ok(None);
		}
		if !(200..300).contains(&status) {
			// Do not include response bodies: providers may echo prompt or credential data.
			if status == 400 || status == 413 {
				let failure: Value = response.json().await.unwrap_or(Value::Null);
				// Preserve the direct OpenAI/Codex fallback contract. A validated gateway
				// route may fall back only for an explicitly unsupported endpoint above;
				// size and validation failures remain visible failures.
				if validated_native_compaction_endpoint(model).is_none()
					&& (status == 413
						|| record(&failure)
							.and_then(|map| map.get("error"))
							.and_then(record)
							.and_then(|error| error.get("code"))
							.and_then(Value::as_str)
							== Some("context_length_exceeded"))
				{
					return Ok(None);
				}
			}
			return Err(CompactionRequestError::new(
				format!("Server compaction failed (HTTP {status})"),
				status,
				retry_after_ms,
			));
		}

		let payload: Value = match decode {
			Some(decode) => match decode(response).await {
				Ok(payload) => payload,
				Err(mut error) => {
					if error.retry_after_ms.is_none() {
						error.retry_after_ms = retry_after_ms;
					}
					return Err(error);
				}
			},
			None => {
				let body = response
					.bytes()
					.await
					.map_err(|error| CompactionRequestError::transport(error.to_string()))?;
				serde_json::from_slice(&body)
					.map_err(|error| CompactionRequestError::new(error.to_string(), 0, None))?
			}
		};
		Ok(Some(payload))
	};
	let payload = match signal.as_ref() {
		Some(signal) => tokio::select! {
			biased;
			_ = signal.cancelled() => return Err(CompactionRequestError::new("Request was aborted", 0, None)),
			result = read_payload => result?,
		},
		None => read_payload.await?,
	};
	let Some(payload) = payload else {
		return Ok(None);
	};
	if payload.is_null() && validated_native_compaction_endpoint(model).is_none() {
		return Ok(None);
	}
	let payload_map = record(&payload);
	let output = payload_map
		.and_then(|map| map.get("output"))
		.and_then(Value::as_array)
		.cloned();
	let valid_output = match &output {
		Some(items) => {
			items.iter().all(|item| record(item).is_some())
				&& items.iter().any(|item| {
					record(item)
						.map(|map| {
							map.get("type").and_then(Value::as_str) == Some("compaction")
								&& map
									.get("encrypted_content")
									.and_then(Value::as_str)
									.map(|value| !value.is_empty())
									.unwrap_or(false)
						})
						.unwrap_or(false)
				})
		}
		None => false,
	};
	if !valid_output {
		return Err(CompactionRequestError::new(
			"Server compaction returned no valid encrypted checkpoint",
			0,
			None,
		));
	}
	let output = output.unwrap_or_default();

	let raw_usage = payload_map.and_then(|map| map.get("usage")).and_then(record);
	let mut usage: Option<Usage> = None;
	if let Some(raw_usage) = raw_usage {
		let input_details = raw_usage.get("input_tokens_details").and_then(record);
		let input = token_count(raw_usage.get("input_tokens"));
		let cached = input.min(token_count(input_details.and_then(|details| details.get("cached_tokens"))));
		let output_tokens = token_count(raw_usage.get("output_tokens"));
		let total_tokens = token_count(raw_usage.get("total_tokens"));
		let mut usage_value = Usage {
			input: input - cached,
			output: output_tokens,
			cache_read: cached,
			cache_write: 0.0,
			total_tokens: if total_tokens != 0.0 { total_tokens } else { input + output_tokens },
			cost: crate::types::UsageCost::zero(),
		};
		calculate_cost(model, &mut usage_value, None);
		usage = Some(usage_value);
	}

	let endpoint = validated_native_compaction_endpoint(model);
	let estimated_tokens = usage
		.as_ref()
		.map(|usage| usage.output)
		.unwrap_or(0.0)
		.max((estimated_window_chars(&output) / 4.0).ceil());

	Ok(Some(ProviderCompactionResult {
		checkpoint: ProviderCompactionCheckpoint {
			version: 1,
			provider: model.provider.clone(),
			api: model.api.clone(),
			model: model.id.clone(),
			base_url: model.base_url.clone(),
			endpoint: endpoint.map(|_| url.to_string()),
			items: output
				.iter()
				.filter_map(|item| record(item).cloned())
				.collect(),
			estimated_tokens,
		},
		usage,
	}))
}

/// Codex v2 returns one checkpoint; the client retains up to 64k tokens of recent user context.
///
/// TS: `buildCodexCompactedWindow(input, checkpoint)`
pub fn build_codex_compacted_window(input: &[Value], checkpoint: &Value) -> Vec<Value> {
	let mut remaining_chars = 64_000.0 * 4.0;
	let mut retained: Vec<Value> = Vec::new();
	for item in input.iter().rev() {
		if remaining_chars <= 0.0 {
			break;
		}
		let Some(item_map) = record(item) else {
			continue;
		};
		let item_type = item_map.get("type").and_then(Value::as_str);
		let role = item_map.get("role").and_then(Value::as_str).unwrap_or_default();
		if (item_type.is_some() && item_type != Some("message")) || !matches!(role, "user" | "developer" | "system") {
			continue;
		}
		let size = estimated_window_chars(std::slice::from_ref(item));
		if size <= remaining_chars {
			retained.push(item.clone());
			remaining_chars -= size;
			continue;
		}
		// Only the boundary message is shortened. Preserve complete image blocks when
		// they fit; never slice image data or serialized provider items.
		let mut content: Vec<Value> = Vec::new();
		let blocks: Vec<Value> = match item_map.get("content") {
			Some(Value::String(text)) => vec![serde_json::json!({"type": "input_text", "text": text})],
			Some(Value::Array(blocks)) => blocks.clone(),
			_ => Vec::new(),
		};
		for block in blocks.iter().rev() {
			if remaining_chars <= 128.0 {
				break;
			}
			let Some(block_map) = record(block) else {
				continue;
			};
			let block_size = estimated_window_chars(std::slice::from_ref(block));
			if block_size <= remaining_chars - 128.0 {
				content.insert(0, block.clone());
				remaining_chars -= block_size;
			} else if block_map.get("type").and_then(Value::as_str) == Some("input_text") {
				if let Some(text) = block_map.get("text").and_then(Value::as_str) {
					let keep = (remaining_chars - 128.0).max(0.0) as usize;
					let chars: Vec<char> = text.chars().collect();
					let start = chars.len().saturating_sub(keep);
					let sliced: String = chars[start..].iter().collect();
					let mut shortened = block_map.clone();
					shortened.insert("text".to_string(), Value::String(sliced));
					content.insert(0, Value::Object(shortened));
					remaining_chars = 0.0;
				}
			}
		}
		if !content.is_empty() {
			let mut updated = item_map.clone();
			updated.insert("content".to_string(), Value::Array(content));
			retained.push(Value::Object(updated));
		}
		remaining_chars = 0.0;
	}
	retained.reverse();
	retained.push(checkpoint.clone());
	retained
}

/// Kept so the module references the same `Api` alias the TypeScript module does.
pub fn _api_marker(_api: &Api) {}

/// `shortHash` is exported by utils/hash; the port re-exports it here because the TypeScript
/// module imports it through the same barrel the provider uses.
pub use crate::utils::hash::short_hash as openai_compaction_short_hash;

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{InputModality, ModelCost, NativeCompactionCapability};
	use crate::utils::hash::short_hash;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};

	fn model() -> Model {
		Model {
			id: "gpt-5".to_string(),
			provider: "openai".to_string(),
			api: "openai-responses".to_string(),
			base_url: "https://api.openai.com/v1".to_string(),
			input: vec![InputModality::Text],
			cost: ModelCost::zero(),
			..Default::default()
		}
	}

	fn astra_model() -> Model {
		let mut model = model();
		model.id = AZURE_ASTRA_MODEL.to_string();
		model.provider = AZURE_MANAGED_PROVIDER.to_string();
		model.base_url = "https://gateway.example.com/azure-openai/v1".to_string();
		model.native_compaction = Some(NativeCompactionCapability {
			protocol: RESPONSES_COMPACT_PROTOCOL.to_string(),
			provider: AZURE_MANAGED_PROVIDER.to_string(),
			model: AZURE_ASTRA_MODEL.to_string(),
			endpoint: "https://gateway.example.com/azure-openai/v1/responses/compact".to_string(),
			api_version: "v1".to_string(),
			enabled: true,
			validation: NativeCompactionValidation::LiveVerified,
		});
		model
	}

	#[test]
	fn normalize_endpoint_rejects_credentials_query_and_fragment() {
		assert_eq!(
			normalize_endpoint("https://example.com/v1/"),
			Some("https://example.com/v1".to_string())
		);
		assert_eq!(normalize_endpoint("https://example.com/v1"), Some("https://example.com/v1".to_string()));
		assert_eq!(normalize_endpoint("https://user@example.com/v1"), None);
		assert_eq!(normalize_endpoint("https://example.com/v1?x=1"), None);
		assert_eq!(normalize_endpoint("https://example.com/v1#frag"), None);
		assert_eq!(normalize_endpoint("ftp://example.com/v1"), None);
		assert_eq!(normalize_endpoint("not a url"), None);
	}

	#[test]
	fn validated_native_compaction_endpoint_requires_live_verified_route() {
		let astra = astra_model();
		assert_eq!(
			validated_native_compaction_endpoint(&astra).as_deref(),
			Some("https://gateway.example.com/azure-openai/v1/responses/compact")
		);

		let mut unverified = astra.clone();
		unverified.native_compaction.as_mut().unwrap().validation = NativeCompactionValidation::DocumentationVerified;
		assert_eq!(validated_native_compaction_endpoint(&unverified), None);

		let mut disabled = astra.clone();
		disabled.native_compaction.as_mut().unwrap().enabled = false;
		assert_eq!(validated_native_compaction_endpoint(&disabled), None);

		let mut wrong_path = astra.clone();
		wrong_path.base_url = "https://gateway.example.com/other/v1".to_string();
		assert_eq!(validated_native_compaction_endpoint(&wrong_path), None);

		let mut wrong_endpoint = astra.clone();
		wrong_endpoint.native_compaction.as_mut().unwrap().endpoint =
			"https://gateway.example.com/azure-openai/v1/responses/other".to_string();
		assert_eq!(validated_native_compaction_endpoint(&wrong_endpoint), None);

		let mut wrong_model = astra.clone();
		wrong_model.id = "gpt-5".to_string();
		assert_eq!(validated_native_compaction_endpoint(&wrong_model), None);
	}

	#[test]
	fn supports_openai_compaction_matches_typescript() {
		let mut codex = model();
		codex.provider = "openai-codex".to_string();
		codex.api = "openai-codex-responses".to_string();
		assert!(supports_openai_compaction(&codex));

		let mut astra_direct = model();
		astra_direct.id = AZURE_ASTRA_MODEL.to_string();
		assert!(supports_openai_compaction(&astra_direct));

		assert!(supports_openai_compaction(&astra_model()));
		assert!(!supports_openai_compaction(&model()));
	}

	#[test]
	fn token_count_matches_typescript() {
		assert_eq!(token_count(Some(&serde_json::json!(12))), 12.0);
		assert_eq!(token_count(Some(&serde_json::json!(-1))), 0.0);
		assert_eq!(token_count(Some(&serde_json::json!("12"))), 0.0);
		assert_eq!(token_count(Some(&Value::Null)), 0.0);
		assert_eq!(token_count(None), 0.0);
	}

	#[test]
	fn estimated_window_chars_collapses_images() {
		let items = vec![serde_json::json!({
			"type": "message",
			"content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
		})];
		let size = estimated_window_chars(&items);
		// Serialised JSON without the base64 body plus the 4800-char image heuristic.
		assert!(size > 4800.0);
		assert!(size < 4900.0);
	}

	#[test]
	fn estimated_window_chars_keeps_non_image_urls() {
		let items = vec![serde_json::json!({"url": "https://example.com/a"})];
		let size = estimated_window_chars(&items);
		assert!(size < 100.0);
	}

	#[test]
	fn build_codex_compacted_window_retains_recent_user_context() {
		let input = vec![
			serde_json::json!({"type": "message", "role": "assistant", "content": "skip me"}),
			serde_json::json!({"type": "message", "role": "user", "content": "keep me"}),
		];
		let checkpoint = serde_json::json!({"type": "compaction", "encrypted_content": "x"});
		let window = build_codex_compacted_window(&input, &checkpoint);
		assert_eq!(window.len(), 2);
		assert_eq!(window[0].get("role").and_then(Value::as_str), Some("user"));
		assert_eq!(window[1], checkpoint);
	}

	#[test]
	fn build_codex_compacted_window_shortens_boundary_message_only() {
		let long_text = "a".repeat(64_000 * 4 + 5_000);
		let input = vec![
			serde_json::json!({"type": "message", "role": "user", "content": "older"}),
			serde_json::json!({"type": "message", "role": "user", "content": long_text}),
		];
		let checkpoint = serde_json::json!({"type": "compaction"});
		let window = build_codex_compacted_window(&input, &checkpoint);
		assert_eq!(window.len(), 2);
		let kept = window[0].get("content").and_then(Value::as_str).unwrap_or_default();
		assert!(kept.chars().count() < 64_000 * 4 + 5_000);
	}

	#[test]
	fn build_codex_compacted_window_returns_checkpoint_only_for_empty_input() {
		let checkpoint = serde_json::json!({"type": "compaction"});
		let window = build_codex_compacted_window(&[], &checkpoint);
		assert_eq!(window, vec![checkpoint]);
	}

	#[test]
	fn short_hash_matches_typescript_for_known_inputs() {
		// Fixed vectors evaluated from packages/ai/src/utils/hash.ts, including
		// UTF-16 surrogate-pair input used by JavaScript's charCodeAt.
		assert_eq!(short_hash(""), "k4n83c7h0j2b");
		assert_eq!(short_hash("fc_abc"), "1qaoofs1vxahbg");
		assert_eq!(short_hash("hello world"), "n7rb4n1m39uz8");
		assert_eq!(short_hash("😀"), "13wj7r7usi372");
	}

	#[tokio::test]
	async fn request_openai_compaction_returns_none_for_unsupported_endpoint() {
		// 404/405/501 fall back. A local listener is not started: instead the transport
		// failure path is asserted through an unroutable URL, which maps to an error.
		let model = model();
		let result = request_openai_compaction(
			&model,
			"http://127.0.0.1:1/responses/compact",
			&IndexMap::new(),
			Map::new(),
			None,
			None,
		)
		.await;
		assert!(result.is_err());
	}

	async fn response_server(response: Vec<u8>) -> (String, tokio::task::JoinHandle<()>) {
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = [0u8; 4096];
			socket.read(&mut request).await.unwrap();
			socket.write_all(&response).await.unwrap();
		});
		(format!("http://{address}/responses/compact"), server)
	}

	#[tokio::test]
	async fn body_transport_and_malformed_json_have_distinct_error_classes() {
		for (body, expected_name) in [
			(
				b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{\"output\":[]".as_slice(),
				"TypeError",
			),
			(
				b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nConnection: close\r\n\r\nnot-json".as_slice(),
				"CompactionRequestError",
			),
		] {
			let (url, server) = response_server(body.to_vec()).await;
			let error = request_openai_compaction(&model(), &url, &IndexMap::new(), Map::new(), None, None)
				.await
				.unwrap_err();
			server.await.unwrap();
			assert_eq!(error.name, expected_name);
			assert_eq!(error.status, 0);
		}
	}

	#[tokio::test]
	async fn decoder_retry_after_uses_header_only_when_decoder_has_no_value() {
		for (decoder_retry, expected) in [(None, 1250.0), (Some(25.0), 25.0)] {
			let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nRetry-After-Ms: 1250\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".to_vec();
			let (url, server) = response_server(response).await;
			let decode: CompactionDecode = Arc::new(move |_response| {
				Box::pin(async move { Err(CompactionRequestError::new("decoder failed", 418, decoder_retry)) })
			});
			let error = request_openai_compaction(&model(), &url, &IndexMap::new(), Map::new(), None, Some(decode))
				.await
				.unwrap_err();
			server.await.unwrap();
			assert_eq!(error.status, 418);
			assert_eq!(error.retry_after_ms, Some(expected));
			assert_eq!(error.message, "decoder failed");
		}
	}

	#[test]
	fn header_record_to_value_feeds_parse_retry_after_ms() {
		// TS: `parseRetryAfterMs(response.headers)` reads the `Retry-After` getter.
		let mut headers = IndexMap::new();
		headers.insert("retry-after".to_string(), "3".to_string());
		assert_eq!(
			parse_retry_after_ms(Some(&header_record_to_value(&headers))),
			Some(3000.0)
		);

		let mut headers = IndexMap::new();
		headers.insert("Retry-After-Ms".to_string(), "1500".to_string());
		assert_eq!(
			parse_retry_after_ms(Some(&header_record_to_value(&headers))),
			Some(1500.0)
		);

		assert_eq!(parse_retry_after_ms(Some(&header_record_to_value(&IndexMap::new()))), None);
	}

	#[test]
	fn compaction_request_error_shape() {
		let error = CompactionRequestError::new("Server compaction failed (HTTP 500)", 500, Some(1500.0));
		assert_eq!(error.name, "CompactionRequestError");
		assert_eq!(error.status, 500);
		assert_eq!(error.retry_after_ms, Some(1500.0));
		assert_eq!(error.to_string(), "Server compaction failed (HTTP 500)");
	}
}
